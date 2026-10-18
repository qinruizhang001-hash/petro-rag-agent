"""
人类在环（Human-in-the-Loop）
=============================
关键风险结论不能全自动通过 —— 报告里写的是"某口井该不该注"，一旦错了，
代价不是重跑一次，而是盖层压裂与 CO2 泄漏。所以图里插了一个**闸门节点**，
由人类决定放行还是打回。

两种编译版本共用同一个节点，靠 ``human_decision`` 区分：

  自动模式（``/api/analyze``、``/sync``、CLI）
      ``app`` / ``app_stream`` 没有 ``interrupt_before``，闸门直接放行，
      并把 ``{"approved": true, "auto": true}`` **显式写进 state**。
      为什么留痕而不是静默跳过：这样 API 响应里的
      ``human_review.auto == true`` 会告诉调用方"这次没人看过"，
      而不是伪装成"已人工确认"。

  人工模式（``/api/analyze/hitl``）
      ``app_hitl`` 带 ``interrupt_before=["hitl_gate"]``，跑到闸门前**暂停**，
      等 ``POST /api/approve`` 把决定写进 state 再续跑。
      通过 → 落盘结束；驳回 → 带审核意见回到 report 重写。

驳回次数在 API 层设上限（见 ``api_server._MAX_HUMAN_REJECTIONS``），
否则"人一直驳回、机器一直重写"会变成一个花钱的无底洞。
"""
from __future__ import annotations

from core.state import AgentState
from utils.logging_setup import get_logger

log = get_logger(__name__)

# 闸门在两种模式下的语义标签，供日志与前端复用
AUTO_APPROVAL = {
    "approved": True,
    "auto": True,
    "reviewer": "auto",
    "comment": "自动模式：未经人工确认",
}


def hitl_gate(state: AgentState):
    """
    审核闸门：读取 ``human_decision``，没有决定时按自动模式放行。

    注意它**不修改**已有的人工决定 —— 那是 ``/api/approve`` 写进去的，
    节点只负责把"没人审"这种情况补成一条显式的自动放行记录。
    """
    decision = state.get("human_decision") or {}

    if not decision:
        log.warning(
            "[HITL] 自动模式：未经人工审核直接放行（/api/analyze 路径）。"
            "需要人工确认请改用 POST /api/analyze/hitl + POST /api/approve"
        )
        return {"human_decision": dict(AUTO_APPROVAL)}

    approved = bool(decision.get("approved"))
    log.info(
        f"[HITL] 人工审核结果：{'✅ 通过' if approved else '⛔ 驳回'}"
        f"（审核人 {decision.get('reviewer') or '-'}"
        f"，意见：{decision.get('comment') or '无'}）"
    )
    return {}


def route_after_hitl(state: AgentState):
    """闸门的条件边：通过 → 落盘；驳回 → 回到 report 重写。"""
    decision = state.get("human_decision") or {}
    return "approve" if decision.get("approved") else "reject"
