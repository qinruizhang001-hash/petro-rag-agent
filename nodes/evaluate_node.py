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
【检索到的文献】：{state['retrieved_knowledge']}
【AI生成的回答】：{state['final_report']}

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

    log.info(
        f"[Agent 4] 当前得分：{score:g} 分 | 评语：{feedback[:120]}",
        extra=llm_extra(model, tier),
    )
    return {
        "score": score,
        "feedback": feedback,
        "physics_check": physics,
        **record_llm_usage(state, "evaluate", response, model, tier),
    }
