"""
文件记忆（File-Based Memory）
============================
极简长期记忆：不使用向量库，只用两个 Markdown 文件。

- `user_profile.md`      —— 用户偏好（手动维护，跨任务长期有效）
- `project_history.md`   —— 历次评估的核心结论（每次任务结束后自动追加）

设计要点（这些细节决定它能不能长期用下去）：
1. **只读最近 N 条**：project_history.md 会随时间无限增长。若整份文件塞进
   Prompt，几十次评估后必然撑爆上下文、并且让模型注意力被历史噪音稀释。
   因此读取时按条目切分，只取最近 `max_entries` 条，再按字符数二次截断。
2. **超阈值自动压缩**：截断只是"少读"，文件本身仍在膨胀、磁盘与启动开销
   持续增长。超过阈值时把旧条目交给大模型提炼成 3 条核心经验并重写文件。
   压缩函数通过 **回调注入** 大模型调用，本模块不 import 任何 LLM 客户端——
   否则会与节点模块形成循环导入（节点 → 本模块，本模块不反向依赖），
   并且让离线单元测试全部失效。
3. **路径锚定项目根目录**：api_server 可能从任意 CWD 启动，用 `__file__`
   定位，避免记忆文件落到奇怪的地方。
4. **写入加锁**：FastAPI 在线程池里跑同步工作流，理论上存在并发追加，
   用 threading.Lock 保证条目不会交错撕裂。压缩的重写是"先写临时文件再
   os.replace"，保证中途失败不会留下半截文件。
5. **写入幂等**：节点可能因重试被再次执行，用 state 里的标记位挡住重复写。
6. **硬上限兜底**：压缩依赖摘要器与"旧条目够多"，两者都可能长期不满足
   （没接大模型 / 摘要失败 / 每轮只多一条）。写入后立即检查字符数上限，
   超了就丢掉最旧的条目 —— 不调模型、不做摘要，只保证文件有界。
   软压缩保结论，硬裁剪保体积，两者都要。
"""
from __future__ import annotations

import os
import re
import tempfile
import threading
from datetime import datetime
from typing import Callable, Optional

from utils.logging_setup import get_logger

log = get_logger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def _env_int(name: str, default: int) -> int:
    """读取整数型环境变量；非法值静默退回默认，不让配置失误炸掉主流程。"""
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        log.info(f"  [FileMemory] {name}={raw!r} 不是合法整数，使用默认值 {default}")
        return default


def _env_path(name: str, default_filename: str) -> str:
    """
    读取可选的路径配置。

    注意：.env 里写 `USER_PROFILE_PATH=`（留空）时 os.getenv 返回空字符串而不是
    默认值，直接用会得到相对 CWD 的错误路径。这里显式把空值当作"未配置"。
    """
    raw = (os.getenv(name) or "").strip()
    return raw if raw else os.path.join(PROJECT_ROOT, default_filename)


USER_PROFILE_PATH = _env_path("USER_PROFILE_PATH", "user_profile.md")
PROJECT_HISTORY_PATH = _env_path("PROJECT_HISTORY_PATH", "project_history.md")

# 压缩阈值（字符数）与压缩时保留的最近条目数
HISTORY_COMPRESS_THRESHOLD = _env_int("HISTORY_COMPRESS_THRESHOLD", 3000)
KEEP_RECENT_ENTRIES = _env_int("HISTORY_KEEP_RECENT", 2)
# 至少要有这么多"旧条目"才值得压缩；见 compress_history_if_needed 的自限说明
MIN_ENTRIES_TO_COMPRESS = 2

# ─── 硬上限（不依赖大模型）────────────────────────────────
#
# 压缩是"软"防线：它要一个大模型摘要器，还要"旧条目够多"。于是有三种情况会
# 让 project_history.md 无限增长，而**没有任何日志会报错**：
#   1. 摘要器没接（build_memory_context 不带 summarizer）；
#   2. 摘要调用失败（本地模型没起来 / 超时）；
#   3. 旧条目不足 MIN_ENTRIES_TO_COMPRESS（每轮只多 1 条时很容易卡在这）。
#
# 后果不是"上下文爆掉"（读取侧只取最近 N 条，那一道是有效的），而是文件本身
# 无限长：history_char_count() 每次全量读，报告里"当前 N 字符"持续变大，
# 备份/同步/CI 的 checkout 都被它拖着。
#
# 所以再加一道**确定性**防线：超过上限就丢掉最旧的条目，不调模型、不看分数、
# 不做摘要。它保证的是"文件有界"，不是"信息无损"—— 两者都要，顺序是先压缩
# （尽量保住结论），再硬裁（保证有界）。
HISTORY_HARD_LIMIT = _env_int("HISTORY_HARD_LIMIT", 12000)
# 硬裁时保留的最近条目数。比 KEEP_RECENT_ENTRIES 大：硬裁是"没摘要可用"时的
# 兜底，这时候更该多留原始记录，而不是按压缩的激进口径只留 2 条。
HISTORY_KEEP_ON_TRIM = _env_int("HISTORY_KEEP_ON_TRIM", 8)

_WRITE_LOCK = threading.Lock()

# 条目标题形如 "## 2026-10-17 20:12 | score 9"
_ENTRY_SPLIT = re.compile(r"(?m)^## ")
LESSONS_HEADING = "项目核心经验（自动压缩）"

# 压缩回调签名：接收旧条目的 Markdown 文本，返回"3 条核心经验"的 Markdown 正文
Summarizer = Callable[[str], str]

USER_PROFILE_TEMPLATE = """# 用户画像（User Profile）

> 本文件由人工维护。每次生成报告前会被读取，作为 System Prompt 的一部分。
> 只写「跨任务长期有效」的偏好；单次任务的信息请交给 project_history.md 自动记录。

## 关注领域
- （示例）CCUS / CO2 地质封存

## 报告风格偏好
- （示例）结论先行，分点陈述，不要空话套话

## 表达习惯
- （示例）中文专业术语优先，必要处保留英文原名
"""

PROJECT_HISTORY_TEMPLATE = """# 项目评估历史（Project History）

> 每次评估结束后，系统会自动追加一条记录（时间 / 参数 / 核心结论 / 评审分）。
> 读取时只取最近若干条，避免上下文无限膨胀。

"""


def _ensure_file(path: str, template: str) -> None:
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(template)


def ensure_files() -> None:
    """确保两个记忆文件存在（首次运行自动创建带模板的空文件）。"""
    _ensure_file(USER_PROFILE_PATH, USER_PROFILE_TEMPLATE)
    _ensure_file(PROJECT_HISTORY_PATH, PROJECT_HISTORY_TEMPLATE)


def _safe_read(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return ""
    except Exception as e:  # 编码损坏等极端情况不应中断主流程
        log.warning(f"  [FileMemory] 读取 {os.path.basename(path)} 失败: {e}")
        return ""


def read_user_profile(max_chars: int = 1500) -> str:
    """读取用户偏好。超长时保留前部（偏好通常写在前面，越靠前越重要）。"""
    ensure_files()
    text = _safe_read(USER_PROFILE_PATH).strip()
    if not text:
        return ""
    if len(text) > max_chars:
        text = text[:max_chars] + "\n…（用户画像过长，已截断）"
    return text


def _split_entries(text: str) -> tuple[str, list[str]]:
    """
    把历史文件切成 (文件头, 条目列表)。

    条目以 "## " 开头（含压缩生成的"项目核心经验"块 —— 它会作为一条条目
    参与下一次压缩，因此多次压缩时经验会自然合并而不是重复堆叠）。
    """
    parts = _ENTRY_SPLIT.split(text or "")
    header = parts[0].rstrip() if parts else ""
    entries = [p.strip() for p in parts[1:] if p.strip()]
    return header, entries


def read_project_history(max_entries: int = 6, max_chars: int = 3000) -> str:
    """
    读取最近的评估历史。

    只取最近 `max_entries` 条 —— 这是本模块最关键的一处防御：
    历史文件会无限增长，全量注入迟早撑爆上下文。
    """
    ensure_files()
    text = _safe_read(PROJECT_HISTORY_PATH)
    if not text.strip():
        return ""

    _, entries = _split_entries(text)
    if not entries:
        return ""

    recent = entries[-max_entries:]
    body = "\n\n".join("## " + e for e in recent)
    if len(body) > max_chars:
        body = "…（更早的历史条目已省略）\n\n" + body[-max_chars:]
    return body


def history_char_count() -> int:
    """
    历史文件的**字符数**（不是字节数）。

    阈值语义必须是字符：中文一个字符占 3 字节，用 os.path.getsize 比较 3000
    会在约 1000 个汉字（≈3000 字节）时就触发压缩，比预期早三倍。
    Prompt 预算关心的也是字符/token，不是磁盘字节。
    """
    return len(_safe_read(PROJECT_HISTORY_PATH))


def history_needs_compression(threshold: int | None = None) -> bool:
    """只读判断：当前历史文件是否已超过压缩阈值（按字符数）。"""
    ensure_files()
    limit = HISTORY_COMPRESS_THRESHOLD if threshold is None else threshold
    return history_char_count() > limit


def compress_history_if_needed(
    summarizer: Optional[Summarizer] = None,
    threshold: int | None = None,
    keep_recent: int | None = None,
) -> dict:
    """
    记忆压缩：旧记录 → 大模型提炼 3 条核心经验 → 原子重写文件。

    返回状态字典（便于日志与测试断言）：
        {"status": "skipped"|"compressed"|"failed", "reason": ...,
         "before_chars": int, "after_chars": int, "dropped_entries": int,
         "lessons": str}

    三个刻意的防御：

    1. **只压缩"旧"条目，最新 N 条原样保留。** 最近几次评估是纵向对比的
       主要依据，被 LLM 摘要成一句话就失去了参数级的可比性。
    2. **旧条目少于 MIN_ENTRIES_TO_COMPRESS 时直接跳过。** 这条是自限开关：
       压缩后旧条目归零，必须再积累若干条才会再次触发。否则若最新条目本身
       就超过阈值，每次读取都会调一次大模型 —— 延迟爆炸且结论反复漂移。
    3. **摘要失败绝不写文件。** 宁可继续用截断后的原文，也不能把历史写坏。
       重写走"临时文件 + os.replace"，中途崩溃不会留下半截文件。
    """
    ensure_files()
    limit = HISTORY_COMPRESS_THRESHOLD if threshold is None else threshold
    keep = KEEP_RECENT_ENTRIES if keep_recent is None else max(1, keep_recent)

    before_chars = history_char_count()

    if before_chars <= limit:
        return {"status": "skipped", "reason": "below_threshold",
                "before_chars": before_chars, "after_chars": before_chars,
                "dropped_entries": 0, "lessons": ""}

    if summarizer is None:
        # 没接大模型时不能静默跳过——否则"压缩了但没生效"会很难排查
        log.info("  [FileMemory] 历史已超阈值，但未提供摘要器，跳过压缩")
        return {"status": "skipped", "reason": "no_summarizer",
                "before_chars": before_chars, "after_chars": before_chars,
                "dropped_entries": 0, "lessons": ""}

    # 在锁内重读：避免与并发 append 交错，导致丢条目
    with _WRITE_LOCK:
        text = _safe_read(PROJECT_HISTORY_PATH)
        header, entries = _split_entries(text)

        if len(entries) <= keep:
            return {"status": "skipped", "reason": "not_enough_entries",
                    "before_chars": before_chars, "after_chars": before_chars,
                    "dropped_entries": 0, "lessons": ""}

        older, recent = entries[:-keep], entries[-keep:]
        if len(older) < MIN_ENTRIES_TO_COMPRESS:
            return {"status": "skipped", "reason": "too_few_old_entries",
                    "before_chars": before_chars, "after_chars": before_chars,
                    "dropped_entries": 0, "lessons": ""}

        old_text = "\n\n".join("## " + e for e in older)
        log.info(f"  [FileMemory] 历史 {before_chars} 字 > {limit}，正在压缩 "
              f"{len(older)} 条旧记录（保留最近 {keep} 条）...")

        try:
            lessons = (summarizer(old_text) or "").strip()
        except Exception as e:
            log.warning(f"  -> 压缩失败（{e}），保留原文件不做修改")
            return {"status": "failed", "reason": str(e),
                    "before_chars": before_chars, "after_chars": before_chars,
                    "dropped_entries": 0, "lessons": ""}

        if not lessons:
            log.info("  -> 摘要器返回空内容，保留原文件不做修改")
            return {"status": "failed", "reason": "empty_summary",
                    "before_chars": before_chars, "after_chars": before_chars,
                    "dropped_entries": 0, "lessons": ""}

        new_text = (
            f"{header}\n\n"
            f"## {LESSONS_HEADING}\n{lessons}\n\n"
            + "\n\n".join("## " + e for e in recent)
            + "\n\n"
        )
        if not _atomic_write(PROJECT_HISTORY_PATH, new_text):
            return {"status": "failed", "reason": "write_failed",
                    "before_chars": before_chars, "after_chars": before_chars,
                    "dropped_entries": 0, "lessons": ""}

        after_chars = len(new_text)
        log.info(f"  -> 压缩完成：{before_chars} → {after_chars} 字，"
              f"已归档 {len(older)} 条旧记录")
        return {"status": "compressed", "reason": "",
                "before_chars": before_chars, "after_chars": after_chars,
                "dropped_entries": len(older), "lessons": lessons}


def trim_history_if_needed(
    max_chars: int | None = None,
    keep_entries: int | None = None,
) -> dict:
    """
    硬裁：文件超过上限就丢掉最旧的条目。**不调用大模型，不做摘要。**

    与 ``compress_history_if_needed`` 的分工：

    ==============  ==================  ==================
    维度             压缩（软）          硬裁（硬）
    ==============  ==================  ==================
    依赖             需要摘要器          不需要
    触发             阈值 + 旧条目够多    超过硬上限
    保留             最近 N 条 + 经验块   最近 M 条 + 经验块
    失败可能         摘要失败即跳过      只在写盘失败时失败
    ==============  ==================  ==================

    返回 ``{"status": "skipped"|"trimmed"|"failed", "reason": ...,
    "before_chars": int, "after_chars": int, "dropped_entries": int}``。

    两条刻意的规则：

    1. **"项目核心经验"块永不裁掉。** 它是压缩产出的唯一沉淀，丢了就等于
       每次压缩白做，而且不会有任何报错。
    2. **至少保留 ``keep_entries`` 条最新记录。** 即使它们本身就超过上限也不裁 ——
       裁到 0 条等于把历史文件变成一个空壳，而"文件很小"看起来还挺健康。
       真出现这种情况时返回 ``reason="still_over_limit"``，让人知道上限没达成。
    """
    ensure_files()
    limit = HISTORY_HARD_LIMIT if max_chars is None else max_chars
    keep = HISTORY_KEEP_ON_TRIM if keep_entries is None else max(1, keep_entries)

    before_chars = history_char_count()
    if before_chars <= limit:
        return {"status": "skipped", "reason": "below_limit",
                "before_chars": before_chars, "after_chars": before_chars,
                "dropped_entries": 0}

    with _WRITE_LOCK:
        # 锁内重读：与 compress 一样，避免和并发 append 交错导致丢条目
        text = _safe_read(PROJECT_HISTORY_PATH)
        header, entries = _split_entries(text)

        lessons = [e for e in entries if e.startswith(LESSONS_HEADING)]
        plain = [e for e in entries if not e.startswith(LESSONS_HEADING)]

        if len(plain) <= keep:
            log.warning(
                f"  [FileMemory] 历史 {before_chars} 字超过硬上限 {limit}，"
                f"但只有 {len(plain)} 条记录（保留下限 {keep}），不裁 —— "
                f"请调大 HISTORY_HARD_LIMIT 或调小 HISTORY_KEEP_ON_TRIM"
            )
            return {"status": "skipped", "reason": "still_over_limit",
                    "before_chars": before_chars, "after_chars": before_chars,
                    "dropped_entries": 0}

        dropped = len(plain) - keep
        kept = plain[-keep:]
        parts = [header] if header else []
        parts += ["## " + e for e in lessons]
        parts += ["## " + e for e in kept]
        new_text = "\n\n".join(parts) + "\n\n"

        if not _atomic_write(PROJECT_HISTORY_PATH, new_text):
            return {"status": "failed", "reason": "write_failed",
                    "before_chars": before_chars, "after_chars": before_chars,
                    "dropped_entries": 0}

        after_chars = len(new_text)
        log.warning(
            f"  [FileMemory] 历史 {before_chars} 字超过硬上限 {limit}，"
            f"已丢弃最旧 {dropped} 条（保留最近 {keep} 条 + 经验块）："
            f"{before_chars} → {after_chars} 字。"
            f"这是兜底，不是压缩 —— 结论没有被提炼，只是被丢掉了"
        )
        return {"status": "trimmed", "reason": "over_hard_limit",
                "before_chars": before_chars, "after_chars": after_chars,
                "dropped_entries": dropped}


def _atomic_write(path: str, text: str) -> bool:
    """
    原子重写：先写同目录临时文件，再 os.replace。

    直接 open(path, "w") 在中途崩溃/掉电时会留下被截断的文件 —— 对记忆文件
    来说这是不可接受的，历史就永久丢了。
    """
    directory = os.path.dirname(path) or "."
    tmp_path = None
    try:
        fd, tmp_path = tempfile.mkstemp(dir=directory, prefix=".history-", suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
        return True
    except Exception as e:
        log.warning(f"  [FileMemory] 原子写入失败: {e}")
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        return False


def build_memory_context(
    profile_max_chars: int = 1500,
    history_max_entries: int = 6,
    history_max_chars: int = 3000,
    summarizer: Optional[Summarizer] = None,
    compress_threshold: int | None = None,
) -> str:
    """
    组装成可直接嵌入 System Prompt 的文本块。

    两个文件都为空时返回空字符串，调用方据此跳过注入。

    传入 `summarizer` 时，会在读取前先做一次"超阈值则压缩"检查——
    这样调用方只需一次调用就能同时拿到"压缩旧记录"和"注入最近记录"两个效果。
    """
    if summarizer is not None:
        compress_history_if_needed(summarizer=summarizer, threshold=compress_threshold)

    profile = read_user_profile(max_chars=profile_max_chars)
    history = read_project_history(
        max_entries=history_max_entries, max_chars=history_max_chars
    )

    blocks = []
    if profile:
        blocks.append(f"### 用户画像（长期偏好，请遵循）\n{profile}")
    if history:
        blocks.append(f"### 历史评估记录（最近 {history_max_entries} 次，供纵向对比）\n{history}")

    if not blocks:
        return ""
    return "## 文件记忆\n\n" + "\n\n".join(blocks)


def append_project_history(entry: str, timestamp: str | None = None) -> int:
    """
    追加一条历史记录，返回写入后的文件**字符数**。

    `entry` 应是已完成渲染的 Markdown 正文（不含 "## " 标题行），
    标题由本函数统一生成，保证格式一致、便于后续按条切分。
    """
    ensure_files()
    entry = (entry or "").strip()
    if not entry:
        return 0

    ts = timestamp or datetime.now().strftime("%Y-%m-%d %H:%M")
    block = f"## {ts}\n{entry}\n\n"

    with _WRITE_LOCK:
        with open(PROJECT_HISTORY_PATH, "a", encoding="utf-8") as f:
            f.write(block)

    # 追加之后立刻检查硬上限。放在这里而不是只靠读取前的压缩，是因为
    # 压缩可能一直不生效（没摘要器 / 摘要失败 / 旧条目不够），而写入是无条件的：
    # 只要还在写，文件就会一直长。写入侧必须自己兜住自己的增长。
    #
    # 注意 trim 内部会再拿一次 _WRITE_LOCK —— 这里必须**已经释放**，
    # threading.Lock 不可重入。
    trim = trim_history_if_needed()
    if trim["status"] == "trimmed":
        log.warning(
            f"  [FileMemory] 历史超过硬上限，已丢弃最旧 "
            f"{trim['dropped_entries']} 条（{trim['before_chars']} → "
            f"{trim['after_chars']} 字）"
        )
    return history_char_count()
