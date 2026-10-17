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
2. **路径锚定项目根目录**：api_server 可能从任意 CWD 启动，用 `__file__`
   定位，避免记忆文件落到奇怪的地方。
3. **写入加锁**：FastAPI 在线程池里跑同步工作流，理论上存在并发追加，
   用 threading.Lock 保证条目不会交错撕裂。
4. **写入幂等**：节点可能因重试被再次执行，用 state 里的标记位挡住重复写。
"""
from __future__ import annotations

import os
import re
import threading
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


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

_WRITE_LOCK = threading.Lock()

# 条目标题形如 "## 2026-10-17 20:12 | score 9"
_ENTRY_SPLIT = re.compile(r"(?m)^## ")

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
        print(f"  [FileMemory] 读取 {os.path.basename(path)} 失败: {e}")
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

    parts = _ENTRY_SPLIT.split(text)
    entries = [p.strip() for p in parts[1:] if p.strip()]
    if not entries:
        return ""

    recent = entries[-max_entries:]
    body = "\n\n".join("## " + e for e in recent)
    if len(body) > max_chars:
        body = "…（更早的历史条目已省略）\n\n" + body[-max_chars:]
    return body


def build_memory_context(
    profile_max_chars: int = 1500,
    history_max_entries: int = 6,
    history_max_chars: int = 3000,
) -> str:
    """
    组装成可直接嵌入 System Prompt 的文本块。

    两个文件都为空时返回空字符串，调用方据此跳过注入。
    """
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
    追加一条历史记录，返回写入后的文件字符数。

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
        return os.path.getsize(PROJECT_HISTORY_PATH)
