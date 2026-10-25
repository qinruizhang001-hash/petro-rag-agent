"""
Agent 4：评审员（规则化物理校验 + 模型打分）
============================================
这个节点曾经只做一件事：让大模型给报告打个 0~10 分。问题是**打分是黑盒**——
一份"注入压力已超过地层破裂压力、却仍判定风险可控"的报告，只要措辞漂亮、
结构完整，模型照样给 9 分。而这份报告会被当成工程依据。

现在的顺序反过来了：

  ① **规则引擎先跑**（``utils.physics_rules``）——纯 Python，确定性，
     用 P_inj 与 P_h / P_frac 的硬关系核对报告的说法。
     发现**矛盾**（说安全、实际超压；说顺畅、实际注不进）→ 直接 0 分打回，
     并给出可执行的修正提示（降到多少 bar / 加深到多少 m）。
  ② 规则通过后，才让模型做**质量**打分（结构、论据、可操作性）。
     物理事实作为 ground truth 写进 prompt，避免模型推翻确定性结论。

这个顺序不能换：模型打分便宜且主观，规则校验贵在确定性。先跑贵的、能一票
否决的那个，才不会被"看起来很好的错误答案"骗过去。

第三个要点是**别把没有结论的校验塞给模型**：参数不足时规则引擎返回
``unverifiable``，此时 ``render_prompt_block`` 产出空串，整个物理块从 prompt 里
消失（原因见该函数的 docstring —— 实测中模型会把"没算过"读成"没通过"）。
所以物理校验对评审 prompt 的影响是**有条件的**，而不是无脑追加一段文本。

闸门是硬的（v2.1 起）
--------------------
早期版本在**解析失败**时兜底 ``score = 10``（放行），理由是"默认 0 分会触发两次
无意义重写"。这个理由只算了成本，没算代价：一份解析失败的评审 + 一份超压的报告，
在日志里看起来就是"10 分，通过"。实测中 Ollama 偶发返回空串（超时前被截断），
于是"评审崩了"被记成"报告优秀"。

现在解析失败 → ``judge_unparseable``，评审自己重试一次；仍失败就**阻断**，
不写进通过分支。判定逻辑集中在 ``core/gates.py``（纯函数，可单测），
本节点只负责产出 ``quality_gate`` 这个结构化结论 —— 报告里到底过没过、
为什么没过，都从这一个字段读，不再靠日志考古。

物理校验与模型分数的关系是**单向**的：物理否决先发生，此时**不调用模型**，
模型分再高也覆盖不了它；反过来，模型只能压低分数，不能把物理否决改回来。
"""
from __future__ import annotations

import re

from core.config import get_llm, llm_chat
from core.gates import (
    QUALITY_THRESHOLD,
    STATUS_BLOCKED_PHYSICS,
    STATUS_JUDGE_UNPARSEABLE,
    STATUS_PASSED,
    STATUS_REWRITE,
    build_gate,
)
from core.state import AgentState
from utils.logging_setup import get_logger, llm_extra
from utils.probability_guard import (
    UNSUPPORTED_PROBABILITY_SCORE_CAP,
    strip_unsupported_probabilities,
    unsupported_probabilities,
)
from utils.physics_rules import evaluate_physics, render_prompt_block
from utils.tokens import record_llm_usage

log = get_logger(__name__)

# 评审 prompt 里同时装【检索证据】和【整篇报告】，两段原本都没有上限。
# 本地默认模型（qwen2.5，4096 窗口）实测被撑爆：某次 evaluate 请求
# 4097 tokens → HTTP 400 → 整个 case 直接判失败，RAGAS 只能记 0 分。
# 这里按字符给两段各设预算：正常长度（证据 ≤1500 字、报告 ≤3000 字）完全
# 不受影响，只有超长报告会被截断，且截断处写进 prompt，评审自己知道。
EVAL_EVIDENCE_CHARS = 1500
EVAL_REPORT_CHARS = 3000
_TRUNCATION_MARK = "\n…（以下内容超出评审预算，已截断）"


def _budget(text: str, limit: int) -> str:
    """把一段文本压到字符预算内；超长时保留开头并显式标注截断。"""
    body = (text or "").strip()
    if len(body) <= limit:
        return body
    return body[:limit].rstrip() + _TRUNCATION_MARK


_SCORE_RE = re.compile(
    r"(?:分数|评分|得分|score)\s*[：:＝=]?\s*(\d+(?:\.\d+)?)\s*(?:/\s*10)?", re.I
)
# 兜底：模型只回了一行 "8分" / "8/10"，没有"分数："前缀。
_BARE_SCORE_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(?:分|/\s*10)", re.M)


def _parse_score(text: str) -> tuple[float, str] | None:
    """从模型输出里解析 (分数, 评语)。解析不出来返回 None。

    容忍度是刻意放宽的：``分数：8`` / ``评分: 8/10`` / ``Score: 8`` / ``8分``
    都算解析成功。**放宽的是格式，不是结果** —— 解析不出分数时不再兜底放行，
    而是走 ``judge_unparseable``（见 ``core.gates``）。7B 评审经常在
    输出上限处被截断，只回一句"8 分"，把它判成失败会造成大量误伤。
    """
    if not text:
        return None
    m = _SCORE_RE.search(text) or _BARE_SCORE_RE.search(text)
    if not m:
        return None
    score = float(m.group(1))
    if not 0.0 <= score <= 10.0:
        return None                     # 11 分/88 分是解析到了别的数字，不是分数
    f = re.search(r"评语[：:]\s*(.+)", text, re.DOTALL)
    feedback = f.group(1).strip() if f else text.strip()
    return score, feedback


def _log_deep(deep: dict | None) -> None:
    """把深层物理结论打进日志。

    单独成函数是为了让"这次没跑深层判定"和"跑了但没算出来"在日志里可分辨：
    前者是 None，后者是 ``status="no_data"`` 且带 ``missing``。两者都不阻断，
    但排查"为什么断层没被判"时，这两条路的修法完全不同。
    """
    if not deep:
        log.info("[Physics·deep] 未运行", extra={"model": "rules:physics", "tokens": "-"})
        return
    if deep.get("status") != "ok":
        log.info(
            f"[Physics·deep] 未计算：缺 {', '.join(deep.get('missing') or [])}",
            extra={"model": "rules:physics", "tokens": "-"},
        )
        return
    geo = deep.get("geomechanics")
    cap = deep.get("caprock") or {}
    log.info(
        f"[Physics·deep] 断层：{geo['verdict'] if geo else '未判定（输入未提断层）'}"
        f" | 盖层：{cap.get('verdict')} | 阻断：{bool(deep.get('blocking'))}",
        extra={"model": "rules:physics", "tokens": "-"},
    )
    for reason in deep.get("blocking_reasons") or []:
        log.warning(f"[Physics·deep] blocking · {reason}",
                    extra={"model": "rules:physics", "tokens": "-"})


def evaluate_report(state: AgentState):
    log.info("[Agent 4] 审核报告：先做规则化物理校验，再由模型评质量")

    # ══════════════════════════════════════════════════════
    # ① 规则层：物理一致性（确定性，一票否决）
    # ══════════════════════════════════════════════════════
    physics = evaluate_physics(
        params_text=state.get("extracted_params", ""),
        report_text=state.get("final_report", ""),
        user_input=state.get("user_input", ""),
        # 深层物理（断层活化 / 盖层突破）在这里打开：它是**闸门**，必须
        # 独立于模型评分存在。``deep=False`` 时既有行为完全不变，但那样
        # "注入压力 1.2 倍静水就把断层推滑"这类报告会照样通过 —— 那正是
        # 本轮要补的漏洞。是否真的阻断由 ``physics_deep`` 的策略决定：
        # 只有输入明确提到断层、或明确给了柱高/注入速率时才允许拦。
        deep=True,
    )
    log.info(
        f"[Physics] {physics['status']} | {physics['summary']}",
        extra={"model": "rules:physics", "tokens": "-"},
    )
    for v in physics.get("violations") or []:
        log.warning(f"[Physics] {v['severity']} · {v['code']} · {v['message']}",
                    extra={"model": "rules:physics", "tokens": "-"})
    _log_deep(physics.get("deep"))

    if physics.get("blocking"):
        # 一票否决：不打模型分，直接打回。省一次调用，也避免模型"审美"压过物理。
        # 返回的 gate 状态是 blocked_physics —— 它在 core.gates.next_action 里
        # 只能变成 rewrite 或 fail，永远不可能变成 pass。
        log.warning("[Agent 4] 物理矛盾 → 判定 0 分，打回重写")
        return {
            "score": 0.0,
            "feedback": physics["feedback"],
            "physics_check": physics,
            "quality_gate": build_gate(
                status=STATUS_BLOCKED_PHYSICS,
                score=0.0,
                physics=physics,
                judge_status="skipped",
                reason=physics.get("summary") or "物理一致性校验未通过",
                detail="；".join(
                    v["message"] for v in physics.get("violations") or []
                    if v["severity"] == "blocking"
                ),
            ),
        }

    # ══════════════════════════════════════════════════════
    # ② 模型层：质量打分（物理事实作为 ground truth）
    # ══════════════════════════════════════════════════════
    physics_block = render_prompt_block(physics)
    # 只有规则引擎真的算出结论时，才追加"不得矛盾"的约束。
    #
    # 刻意**不**在这里补充额外的质量评分标准（如"建议是否含数值"）。理由：
    # 这次改造的目的是加物理硬校验，不是悄悄抬高所有报告的质量门槛。
    # 实测过：多加两条评分重点后，同一份报告因为"缺乏具体数值和监测手段"
    # 被扣到 6 分——这属于无关的行为变更，会让原本达标的报告开始反复重写。
    if physics_block:
        physics_part = f"\n{physics_block}\n"
        extra_rule = ("补充要求：不得与上面【物理一致性校验】的结论矛盾；"
                      "若结论为 `warning`，此项最多扣 1 分。\n")
    else:
        physics_part = ""
        extra_rule = ""

    prompt = f"""你是一个严格的CCUS学术评审员。请根据【检索到的文献】，评估【AI生成的回答】的质量。
{physics_part}
【用户问题】：{state['user_input']}
【检索到的文献】：{_budget(state.get('retrieved_knowledge', ''), EVAL_EVIDENCE_CHARS)}
【AI生成的回答】：{_budget(state.get('final_report', ''), EVAL_REPORT_CHARS)}

{extra_rule}请严格按以下格式输出，不要输出任何其他内容：
分数：[0-10的整数]
评语：[1-2句简短评语]"""

    _, model, tier, _ = get_llm("evaluate")
    response = llm_chat(
        "evaluate",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    result = response.choices[0].message.content or ""
    usage = record_llm_usage(state, "evaluate", response, model, tier)

    parsed = _parse_score(result)
    if parsed is None:
        # 一次重试，只为了让**评审自己**把格式说清楚，不是为了让分数变好看。
        # 重试仍失败 → judge_unparseable（阻断），不再有"默认 10 分放行"。
        log.warning(f"[Agent 4] 评审输出无法解析（首次）：{result[:120]!r} → 重试一次")
        retry_prompt = prompt + (
            "\n\n上次的输出无法解析。本次只允许输出两行，不要任何其它文字：\n"
            "分数：<0-10 的数字>\n评语：<一句话>"
        )
        response = llm_chat(
            "evaluate",
            messages=[{"role": "user", "content": retry_prompt}],
            temperature=0.0,
        )
        result = response.choices[0].message.content or ""
        usage = record_llm_usage({**state, **usage}, "evaluate_retry", response, model, tier)
        parsed = _parse_score(result)

    if parsed is None:
        log.error(f"[Agent 4] 评审两次都无法解析 → 阻断（judge_unparseable）：{result[:120]!r}")
        return {
            "score": 0.0,
            "feedback": (
                "评审模型两次输出都无法解析出分数，本次报告按未通过处理。"
                "请检查评审模型的输出上限（--judge-max-tokens / num_predict）是否够用。"
            ),
            "physics_check": physics,
            "unsupported_probabilities": [],
            "quality_gate": build_gate(
                status=STATUS_JUDGE_UNPARSEABLE,
                score=0.0,
                physics=physics,
                judge_status="unparseable",
                reason="评审输出无法解析出 0-10 的分数（已重试一次）",
                detail=(result or "")[:200],
            ),
            **usage,
        }

    score, feedback = parsed

    # 数值级告警不阻断流程，但必须让下一轮（或用户）看见
    warnings = [v["message"] for v in (physics.get("violations") or [])
                if v["severity"] == "warning"]
    if warnings:
        feedback = f"{feedback}\n（物理校验提示：{'；'.join(warnings)}）"

    # ══════════════════════════════════════════════════════
    # ③ 规则层：概率可溯源（确定性，扣分打回）
    # ══════════════════════════════════════════════════════
    # RAGAS 评估实测：一份报告写了 5 个"XX 风险概率 12%/8%/5%/15%/10%
    # （100 次蒙特卡洛模拟）"，而整条流程只算过密度和压力曲线 —— 这 5 个数字
    # 一个都无处可查，忠实度因此掉到 0.095。模型是被提示词里的数字示例带偏的，
    # 改提示词能减轻，但不能指望 7B 每次都照做，所以这里用代码兜底。
    evidence = " ".join(
        [
            str(state.get("retrieved_knowledge", "")),
            " ".join(str(fact) for fact in (state.get("tool_facts") or [])),
        ]
    )
    fabricated = unsupported_probabilities(state.get("final_report", ""), evidence)
    if fabricated:
        capped = min(score, UNSUPPORTED_PROBABILITY_SCORE_CAP)
        log.warning(
            f"[Agent 4] {len(fabricated)} 处概率无法溯源 → 分数上限 "
            f"{UNSUPPORTED_PROBABILITY_SCORE_CAP:g}（原 {score:g}）：{fabricated[:3]}"
        )
        feedback = (
            f"{feedback}\n（概率溯源校验：以下概率在检索证据与工具结果中都找不到，"
            f"必须整句删掉（不要改写成「未做量化」，那也是一句无依据的话）：{'；'.join(fabricated)}）"
        )
        score = capped

    log.info(
        f"[Agent 4] 当前得分：{score:g} 分 | 评语：{feedback[:120]}",
        extra=llm_extra(model, tier),
    )

    passed = score >= QUALITY_THRESHOLD
    gate = build_gate(
        status=STATUS_PASSED if passed else STATUS_REWRITE,
        score=score,
        physics=physics,
        reason=(
            f"评审得分 {score:g} ≥ 阈值 {QUALITY_THRESHOLD:g}"
            if passed else
            f"评审得分 {score:g} < 阈值 {QUALITY_THRESHOLD:g}，需重写"
        ),
        detail=feedback[:400],
    )
    log.info(f"[Gate] {gate['status']} · {gate['reason']}")
    return {
        "score": score,
        "feedback": feedback,
        "physics_check": physics,
        "unsupported_probabilities": fabricated,
        "quality_gate": gate,
        **usage,
    }
