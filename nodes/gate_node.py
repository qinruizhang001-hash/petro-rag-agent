"""
失败态落盘节点（``mark_failed``）
=================================
质量闸门的第三个出口。它存在的唯一理由是：**失败必须留下痕迹**。

为什么需要一个节点，而不是让条件边直接连到 ``END``
--------------------------------------------------
条件边只能返回分支名，改不了 state。如果低分/物理否决直接连 ``END``，
最终状态里就只有 ``score=3`` 和一份看起来完全正常的 ``final_report`` ——
API 返回 200，前端照常渲染，报告照常进历史记忆。下游没有任何字段能区分
"这份报告通过了"和"这份报告被拒绝了"。

所以这里做三件事，顺序不能换：

1. **写状态**：``quality_gate.status = failed``，并保留原因。
   这是唯一的对外契约，API / 前端 / 历史记忆都读它。
2. **物理否决时隔离正文**：一份"注入压力已超过破裂压力、却判定风险可控"的
   报告不能以任何形式交付。正文移到 ``draft_report``（可审计），
   ``final_report`` 只留阻断说明。这不是"删证据"，是"不让它冒充结论"。
3. **不写历史记忆**：失败报告不进 ``project_history.md``。
   那份文件是后续评估的先验，把被拒的结论写进去等于让系统自己教自己犯错。
   （路由上 ``fail`` 直接指向本节点，不经过 ``save_file_memory``。）
"""
from __future__ import annotations

from core.gates import STATUS_FAILED, is_failed
from core.state import AgentState
from utils.logging_setup import get_logger

log = get_logger(__name__)

_BANNER = "> ⛔ **本报告未通过质量闸门（status={status}），不得作为工程依据。**\n> 原因：{reason}\n"

_QUARANTINE = (
    "\n\n---\n\n"
    "报告正文未通过物理一致性校验，已隔离，不作为交付内容。\n"
    "（原始草稿保存在 `draft_report` 字段，仅供排查与审计。）\n"
)


def mark_failed(state: AgentState):
    gate = dict(state.get("quality_gate") or {})
    reason = gate.get("reason") or "未说明"
    physics_blocking = bool(gate.get("physics_blocking"))
    status = gate.get("status") or STATUS_FAILED

    banner = _BANNER.format(status=status, reason=reason)
    body = state.get("final_report") or ""
    draft = ""

    if physics_blocking:
        draft = body
        final_report = banner + _QUARANTINE
        # 隔离这件事必须留下**是哪条规则、差多少**：只写"物理否决"的话，
        # 事后只能从 draft_report 里反推，而草稿会被后续轮次覆盖。
        log.error(
            f"[Gate] 物理否决 → 正文已隔离（{len(body)} 字）"
            f"｜规则={gate.get('physics_status')}"
            f"｜rules_version={gate.get('physics_rules_version')}"
            f"｜原因={reason}"
        )
    else:
        final_report = banner + "\n" + body
        log.error(f"[Gate] 质量未达标 → 保留正文但标注失败：{reason}")

    gate.update({
        "status": STATUS_FAILED,
        "reason": reason,
        "physics_blocking": physics_blocking,
        "quarantined": physics_blocking,
        "delivered": False,
    })

    log.error(f"[Gate] 流程以失败结束：{reason}")

    update = {
        "quality_gate": gate,
        "report_status": STATUS_FAILED,
        "final_report": final_report,
        "history_saved": False,
    }
    if draft:
        update["draft_report"] = draft
    return update


def is_deliverable(state: AgentState) -> bool:
    """API / 前端复用的判定：只有非失败态的报告才算交付物。"""
    return not is_failed(state.get("quality_gate"))
