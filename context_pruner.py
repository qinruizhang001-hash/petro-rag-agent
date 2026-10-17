"""
上下文压缩（Context Pruning）
=============================
用便宜的本地模型把召回的长文本压成要点，再送给昂贵的云端模型。

为什么值得单独成模块：
- 这是一条**成本策略**，它的效果只能靠数字说话（压缩前后字符数、节省比例）。
  把判断逻辑做成纯函数，就能在 CI 里离线验证"阈值、边界、失败回退"，
  不必每次都跑一次真实的 7B 推理。
- 压缩失败必须是**可降级**的：模型输出为空、输出比原文还长、调用报错，
  这三种情况都必须原样返回原文。宁可多花钱，也不能把检索结果弄丢——
  报告质量依赖它，静默丢上下文比多花几分钱严重得多。

调用约定：
    prune_context(text, summarizer=fn) -> dict
其中 `summarizer` 是 `(prompt: str) -> str` 的回调。之所以用回调注入而不是
在本模块 import 大模型客户端：
1. 避免与 multi_agent_ccus 形成循环导入；
2. token 记账必须留在主流程里（压缩本身也是一次真实的 LLM 调用，
   不记账成本统计就会失真）。
"""
from __future__ import annotations

from typing import Callable, Optional

# 触发阈值（字符数）。低于它就不值得多付一次本地推理。
DEFAULT_PRUNE_THRESHOLD = 4000

# 压缩后至少要比原文短这么多才算"有效压缩"。
# 若模型只是把原文复述一遍（甚至加了开场白导致更长），视为失败并回退原文——
# 否则我们会用一次"没省钱的调用"换掉一份"信息完整的上下文"，纯亏。
MIN_GAIN_RATIO = 0.85

# 单次送进压缩模型的最大字符数。
#
# ⚠️ 这个数字不是随手取的。实测本地 Ollama（qwen2.5）的默认上下文是
# **4096 tokens**，且超限时返回 HTTP 400 而不是静默截断；而
# `extra_body={"options": {"num_ctx": ...}}` 在 OpenAI 兼容端点上**无效**
# （试过，仍按 4096 拒绝）。中文约 0.74 token/字，4096 tokens 扣除指令开销
# 后约合 5000 字。取 3000 字留足余量：单块必然放得下，不会因为超限而
# 让整次压缩失败、悄悄退回原文。
DEFAULT_CHUNK_CHARS = 3000

# 绝对上限：超过就先截断再分块，防止一次误传整本书导致几百次推理。
MAX_TOTAL_CHARS = 60000

# 合并各块要点后的条数上限。分块压缩会让要点条数随块数线性增长，
# 不设上限就等于把"压缩"变成"搬家"。
MAX_BULLETS = 12

PRUNE_PROMPT = """你是 CCUS 领域的技术助理。下面是从知识图谱中检索到的资料片段，它们将要被送给另一位工程师撰写风险评估报告。

请把它们压缩成**要点清单**，只保留会直接影响风险判断的信息：
- 具体的数值、阈值、判据（压力、温度、深度、密度、浓度等）
- 机理与因果链（例如"注入压力接近盖层突破压力 → 断层活化"）
- 明确的工程措施与监测手段

丢弃：重复表述、修辞、与 CCUS 地质封存无关的内容。

要求：
1. 每条一行，以 "- " 开头，最多 6 条。
2. 每条不超过 40 字，必须**保留原文中的数值**，不要换算、不要四舍五入。
3. 同一现象在不同工况下重复出现时，**合并成一条**并标注数值区间
   （例如"注入压力 250~290 bar 时隐伏断层可能活化"），不要每个工况各列一条。
4. 不要开场白、不要总结句、不要编号、不要分小标题。
5. 如果某条信息你不确定，就不要写——不允许编造原文没有的内容。

【检索资料】：
{context}

直接输出 "- " 开头的要点，不要输出其他任何内容。"""


def context_char_count(text: str) -> int:
    """
    按**字符**计数（不是字节）。

    这个区别在本项目里已经踩过一次坑：中文 3 字节/字符，用 os.path.getsize
    做阈值判断会让"3000"这个数字在 1000 字时就触发，提前三倍。
    """
    return len(text or "")


def needs_pruning(text: str, threshold: int = DEFAULT_PRUNE_THRESHOLD) -> bool:
    """是否需要压缩。阈值 <= 0 视为关闭该功能。"""
    if threshold is None or threshold <= 0:
        return False
    return context_char_count(text) > threshold


def build_prune_prompt(context: str, max_input_chars: int = DEFAULT_CHUNK_CHARS) -> str:
    """组装压缩指令。超长输入先截断，防止本地模型上下文溢出。"""
    body = context or ""
    if max_input_chars and len(body) > max_input_chars:
        body = body[:max_input_chars] + "\n…（原文过长，已截断）"
    return PRUNE_PROMPT.format(context=body)


def split_into_chunks(text: str, chunk_chars: int = DEFAULT_CHUNK_CHARS) -> list[str]:
    """
    按段落边界切块，每块不超过 chunk_chars 字符。

    为什么优先在空行处切：检索片段之间本来就以空行分隔，沿这个边界切
    不会把一条结论劈成两半。只有当单个段落本身超长时才硬切。

    为什么必须分块而不是直接截断：截断会让后半段上下文彻底消失，
    而压缩的本意是"保留信息、去掉冗余"。截断是丢信息，分块才是压缩。
    """
    text = text or ""
    if chunk_chars <= 0 or len(text) <= chunk_chars:
        return [text] if text else []

    paragraphs = [p for p in text.split("\n\n")]
    chunks: list[str] = []
    buf: list[str] = []
    buf_len = 0

    def flush():
        nonlocal buf, buf_len
        if buf:
            chunks.append("\n\n".join(buf))
            buf, buf_len = [], 0

    for para in paragraphs:
        plen = len(para)
        if plen > chunk_chars:
            # 单段就超长：先把已有缓冲吐出去，再硬切这一段
            flush()
            for i in range(0, plen, chunk_chars):
                chunks.append(para[i:i + chunk_chars])
            continue
        # 加上分隔符 "\n\n" 共 2 字符
        extra = plen + (2 if buf else 0)
        if buf_len + extra > chunk_chars:
            flush()
            extra = plen
        buf.append(para)
        buf_len += extra
    flush()
    return [c for c in chunks if c.strip()]


def _collect_bullets(text: str) -> list[str]:
    """抽取以 "- " 开头的行，去掉行首空白，保持顺序。"""
    out = []
    for line in (text or "").splitlines():
        s = line.strip()
        if s.startswith("- "):
            out.append(s)
    return out


def prune_context(
    text: str,
    summarizer: Optional[Callable[[str], str]] = None,
    threshold: int = DEFAULT_PRUNE_THRESHOLD,
    min_gain_ratio: float = MIN_GAIN_RATIO,
    chunk_chars: int = DEFAULT_CHUNK_CHARS,
    max_total_chars: int = MAX_TOTAL_CHARS,
    max_bullets: int = MAX_BULLETS,
) -> dict:
    """
    超过阈值时压缩文本，返回带统计的结果。

    返回字段：
        status        "pruned" | "skipped" | "failed"
        before_chars  压缩前字符数
        after_chars   压缩后字符数（失败/跳过时等于 before_chars）
        saved_chars   before - after
        saved_ratio   saved / before（0~1）
        compressed    实际要注入下游的文本（**永远可直接使用**）
        reason        跳过/失败的原因（便于日志与排障）
        chunks        分块数（1 表示单次调用）
        calls         实际调用模型的次数（= 成功分块数）
        prompts       实际发出的指令列表（便于测试与审计）

    三种失败都必须回退原文，这是本函数最重要的约定：
      - summarizer 为 None（未注入模型）
      - summarizer 抛异常
      - 输出为空 / 不含要点行 / 没有实质变短

    分块：超过 chunk_chars 的输入按段落切成多块分别压缩，要点合并去重。
    这是为了绕开本地模型 4096 tokens 的上下文上限（见 DEFAULT_CHUNK_CHARS）。
    任一块失败**不会**让整体失败——已成功的块照常贡献要点；
    全部块都失败才回退原文。
    """
    before = context_char_count(text)

    def _skip(reason: str) -> dict:
        return {
            "status": "skipped",
            "before_chars": before,
            "after_chars": before,
            "saved_chars": 0,
            "saved_ratio": 0.0,
            "compressed": text,
            "reason": reason,
            "chunks": 0,
            "calls": 0,
            "prompts": [],
        }

    def _fail(reason: str, prompts: list[str], calls: int = 0, chunks: int = 0) -> dict:
        return {
            "status": "failed",
            "before_chars": before,
            "after_chars": before,
            "saved_chars": 0,
            "saved_ratio": 0.0,
            "compressed": text,       # 失败一律保留原文
            "reason": reason,
            "chunks": chunks,
            "calls": calls,
            "prompts": prompts,
        }

    if not needs_pruning(text, threshold):
        return _skip(f"未达阈值（{before} ≤ {threshold} 字）")

    if summarizer is None:
        return _fail("未注入压缩模型", [])

    body = text
    if max_total_chars and len(body) > max_total_chars:
        body = body[:max_total_chars]

    pieces = split_into_chunks(body, chunk_chars)
    prompts: list[str] = []
    bullets: list[str] = []
    calls = 0
    errors: list[str] = []
    empty_chunks = 0
    nonbullet_chunks = 0
    exc_chunks = 0

    for piece in pieces:
        prompt = build_prune_prompt(piece, chunk_chars)
        prompts.append(prompt)
        try:
            out = summarizer(prompt)
        except Exception as e:  # 单块失败不拖垮整体
            exc_chunks += 1
            errors.append(f"第 {calls + 1} 块调用失败：{e}")
            continue
        calls += 1
        out = (out or "").strip()
        if not out:
            empty_chunks += 1
            errors.append(f"第 {calls} 块返回空内容")
            continue
        got = _collect_bullets(out)
        if not got:
            nonbullet_chunks += 1
            errors.append(f"第 {calls} 块无要点输出")
            continue
        bullets.extend(got)

    if not bullets:
        # 失败原因要具体：分块后统一说"无要点输出"会掩盖真正的病因
        # （模型返回空 vs 返回了散文），排障时看日志就分不清。
        if exc_chunks == len(pieces):
            reason = "压缩调用失败：" + "；".join(errors)
        elif empty_chunks == len(pieces):
            reason = "模型返回空内容"
        elif nonbullet_chunks == len(pieces):
            reason = "模型输出不是要点列表"
        else:
            reason = "；".join(errors) or "模型输出不是要点列表"
        return _fail(reason, prompts, calls, len(pieces))

    # 去重（保序）：分块后各块常给出同一句规律，重复要点只会白占上下文
    seen = set()
    unique: list[str] = []
    for b in bullets:
        key = b.strip()
        if key and key not in seen:
            seen.add(key)
            unique.append(key)

    if max_bullets and len(unique) > max_bullets:
        unique = unique[:max_bullets]

    joined = "\n".join(unique)
    after = context_char_count(joined)
    if after <= 0 or after >= before * min_gain_ratio:
        # 没有实质变短：说明这次调用白花钱，回退原文
        return _fail(
            f"压缩无收益（{before} → {after} 字，需低于 {int(before * min_gain_ratio)}）",
            prompts, calls, len(pieces),
        )

    return {
        "status": "pruned",
        "before_chars": before,
        "after_chars": after,
        "saved_chars": before - after,
        "saved_ratio": round((before - after) / before, 4),
        "compressed": joined,
        "reason": "；".join(errors),
        "chunks": len(pieces),
        "calls": calls,
        "prompts": prompts,
    }


# 对外别名：能力叫「上下文压缩（Context Pruning）」，函数名取 compress_context。
# 两者是同一个对象，不存在第二套实现 —— 只是让调用方用哪个名字读起来都顺。
compress_context = prune_context
