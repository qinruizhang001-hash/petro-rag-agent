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

另外，解析失败时兜底 ``score = 10``（放行）——这个选择看起来危险，但反过来
（默认 0 分）会让一次解析抖动触发两次无意义的重写，成本更高且结果更差。
注意：**物理校验失败不走这个兜底**，它是一票否决。
"""
from __future__ import annotations

import re

from core.config import get_llm, llm_chat
from core.state import AgentState
from utils.logging_setup import get_logger, llm_extra
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


def _parse_score(text: str) -> tuple[float, str] | None:
    """从模型输出里解析 (分数, 评语)。解析不出来返回 None。"""
    if not text:
        return None
    m = re.search(r"分数[：:]\s*(\d+(?:\.\d+)?)", text)
    if not m:
        return None
    score = float(m.group(1))
    f = re.search(r"评语[：:]\s*(.+)", text, re.DOTALL)
    feedback = f.group(1).strip() if f else text.strip()
    return score, feedback


# 「概率 … 12%」——只盯概率断言，不去管"含水饱和度 12%"这类普通数值。
_PROBABILITY_CLAIM = re.compile(r"(?:概率|比例|占比)[^。；\n]{0,14}?(\d+(?:\.\d+)?)\s*%")
_ANY_PERCENT = re.compile(r"(\d+(?:\.\d+)?)\s*%")

# 编造概率的扣分上限。定得比"重写阈值"低，是为了让这类报告一定被打回，
# 而不是靠提示词求模型自觉 —— 实测 7B 会照着示例编 5 个百分数。
UNSUPPORTED_PROBABILITY_SCORE_CAP = 4.0
_SENTENCE_END = re.compile(r"(?<=[。！？!?])")


def unsupported_probabilities(report: str, evidence: str) -> list[str]:
    """报告里出现、但证据/工具结果里没有的概率/比例断言。

    RAGAS 的 Faithfulness 正是被这类断言拉低的：报告写「断层活化风险概率
    12%（100 次蒙特卡洛模拟）」，而全流程只算过密度和压力曲线，12% 无处可查。
    这里用确定性规则先拦一遍，比让模型自查可靠。

    「比例」也在拦截范围内：只匹配「概率」时，报告改写成「压力累积达到破裂
    压力的比例为 95%」就绕过了守卫，而 95% 同样是凭空来的（实测忠实度 0.545）。
    """
    allowed = set(_ANY_PERCENT.findall(evidence or ""))
    claims: list[str] = []
    for match in _PROBABILITY_CLAIM.finditer(report or ""):
        if match.group(1) not in allowed:
            claims.append(match.group(0).strip())
    return claims


def strip_unsupported_probabilities(report: str, evidence: str) -> tuple[str, list[str]]:
    """把无法溯源的百分比连同它所在的整句一起删掉。

    只扣分等于承认"这段文字不该发出去"，却照样把它发给用户：实测某次重写
    预算耗尽后，一份带 5 个编造概率的报告被强制放行，RAGAS 忠实度 0.000。
    扣分能推动重写，但推动失败时必须有人动手删，那个人只能是代码。

    删**整句**而不是把数字替换成「该风险未做概率量化」，是因为替换文本本身
    就是一句证据支持不了的新陈述：实测一份报告被替换出 11 处「（该风险未做
    概率量化）」，Faithfulness 里这些句子全部判为无依据 —— 修幻觉的代码
    反而在制造无法溯源的句子。整句删掉不留残句。
    """
    claims = unsupported_probabilities(report, evidence)
    if not claims:
        return report, []
    unsupported = set(claims)
    kept: list[str] = []
    for sentence in _SENTENCE_END.split(report):
        if not sentence.strip():
            continue
        offending = [
            match.group(0).strip()
            for match in _PROBABILITY_CLAIM.finditer(sentence)
            if match.group(0).strip() in unsupported
        ]
        if offending:
            continue
        kept.append(sentence)
    cleaned = "".join(kept).strip()
    return cleaned, claims


def evaluate_report(state: AgentState):
    log.info("[Agent 4] 审核报告：先做规则化物理校验，再由模型评质量")

    # ══════════════════════════════════════════════════════
    # ① 规则层：物理一致性（确定性，一票否决）
    # ══════════════════════════════════════════════════════
    physics = evaluate_physics(
        params_text=state.get("extracted_params", ""),
        report_text=state.get("final_report", ""),
        user_input=state.get("user_input", ""),
    )
    log.info(
        f"[Physics] {physics['status']} | {physics['summary']}",
        extra={"model": "rules:physics", "tokens": "-"},
    )
    for v in physics.get("violations") or []:
        log.warning(f"[Physics] {v['severity']} · {v['code']} · {v['message']}",
                    extra={"model": "rules:physics", "tokens": "-"})

    if physics.get("blocking"):
        # 一票否决：不打模型分，直接打回。省一次调用，也避免模型"审美"压过物理。
        log.warning("[Agent 4] 物理矛盾 → 判定 0 分，打回重写")
        return {
            "score": 0.0,
            "feedback": physics["feedback"],
            "physics_check": physics,
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

    parsed = _parse_score(result)
    if parsed is None:
        # 解析抖动 → 放行。理由见模块 docstring：默认 0 分会触发两次无意义重写。
        score, feedback = 10.0, result
        log.warning(f"[Agent 4] 评审输出无法解析，按 10 分放行：{result[:80]}")
    else:
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
    return {
        "score": score,
        "feedback": feedback,
        "physics_check": physics,
        "unsupported_probabilities": fabricated,
        **record_llm_usage(state, "evaluate", response, model, tier),
    }
