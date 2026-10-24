"""
质量闸门（确定性判定，不是模型判断）
====================================
报告能不能出门，由这里的纯函数决定。放在独立模块里的理由和 ``physics_rules``
一样：**判定必须是可单测的、不依赖 LLM 的、结果可复现的**。

三个硬规则
----------
1. **物理否决优先。** ``blocked_physics`` 永远不可能走到 ``pass``：预算内重写，
   预算用尽即失败。模型评分在这一支上根本不会被调用，所以不存在"措辞漂亮
   把超压说过去"的路径。
2. **解析失败 = 失败。** 评审输出解析不出分数时，早期版本兜底 10 分放行，
   于是"评审崩了"在日志里显示为"10 分通过"。现在记为 ``judge_unparseable``，
   评审自身重试一次后仍失败就阻断。
3. **重写预算用尽 = 失败。** 早期版本在 ``retry_count >= 2`` 时"强制放行"，
   并在日志里写"结论质量未达阈值"。那是一个**伪装成通过**的出口：
   API 返回的 ``score`` 是 3 分，而流程状态是"结束"，调用方无从区分。
   现在低分只有两条路：继续重写，或明确失败。

状态机
------
::

    evaluate ──► passed ──────────────────────────► pass  → hitl → 落盘
             ├─► rewrite (低分) ─── 预算内 ──────► rewrite → aggregate
             │                   └─ 预算用尽 ────► fail
             ├─► blocked_physics ─ 预算内 ───────► rewrite → aggregate
             │                   └─ 预算用尽 ────► fail
             └─► judge_unparseable ──────────────► fail

``fail`` 不是"通过"的变体：``mark_failed`` 会写入 ``quality_gate.status=failed``，
物理否决还会把正文隔离（``draft_report``），不让一份"注入压力超过破裂压力却
判定风险可控"的报告以任何形式被当作交付物。
"""
from __future__ import annotations

# 通过阈值：与 CI 里 RAGAS 的门禁口径保持一致（见 eval_ragas.py --threshold）。
# 两个数字故意分开：一个是"报告自身质量分"，一个是"报告对证据的忠实度"，
# 衡量的是不同东西，但都不允许低于 8 折。
QUALITY_THRESHOLD = 8.0

# 重写预算。aggregate 每跑一次 retry_count +1，所以 2 表示"初稿 + 两次重写"。
# 与旧实现的上限相同，改的只是用尽之后的行为（失败，而不是放行）。
MAX_REPORT_REWRITES = 2

STATUS_PASSED = "passed"
STATUS_REWRITE = "rewrite"
STATUS_BLOCKED_PHYSICS = "blocked_physics"
STATUS_JUDGE_UNPARSEABLE = "judge_unparseable"
STATUS_FAILED = "failed"

# 失败态：这些状态下的报告不能被当作交付物。
FAILED_STATUSES = (STATUS_FAILED, STATUS_BLOCKED_PHYSICS, STATUS_JUDGE_UNPARSEABLE)


def build_gate(
    *,
    status: str,
    reason: str,
    score: float | None = None,
    threshold: float = QUALITY_THRESHOLD,
    physics: dict | None = None,
    judge_status: str = "ok",
    detail: str = "",
) -> dict:
    """组装写进 state 的 ``quality_gate``。

    字段命名对齐 API 响应，前端/评估脚本不必二次翻译：
    ``status`` / ``score`` / ``threshold`` / ``physics_status`` /
    ``physics_blocking`` / ``judge_status`` / ``reason``。
    """
    physics = physics or {}
    return {
        "status": status,
        "score": score,
        "threshold": threshold,
        "physics_status": physics.get("status"),
        "physics_blocking": bool(physics.get("blocking")),
        "physics_rules_version": physics.get("rules_version"),
        "judge_status": judge_status,
        "reason": reason,
        "detail": detail,
        "rewrite_budget": MAX_REPORT_REWRITES,
    }


def next_action(gate: dict | None, retry_count: int) -> str:
    """条件边的唯一判定入口：返回 ``pass`` / ``rewrite`` / ``fail``。

    **缺失或未知状态一律 fail（fail-closed）。** 状态字段因为任何原因丢了
    （老 checkpoint 续跑、节点异常返回空 dict），都不该被读成"通过" ——
    "没有结论"和"结论是通过"是两件事，把它们混在一起正是软闸门的来源。
    """
    status = (gate or {}).get("status")
    budget_left = int(retry_count or 0) < MAX_REPORT_REWRITES

    if status == STATUS_PASSED:
        return "pass"
    if status == STATUS_FAILED:
        return "fail"
    if status in (STATUS_BLOCKED_PHYSICS, STATUS_REWRITE):
        # 物理矛盾与低分都可以通过"改写措辞/补上超压事实"修复，所以先重写；
        # 但两者都绝不在预算用尽时转成 pass。
        return "rewrite" if budget_left else "fail"
    if status == STATUS_JUDGE_UNPARSEABLE:
        # 评审自己都读不出分数，重写报告没有意义（不知道改什么），直接阻断。
        return "fail"
    return "fail"


def is_failed(gate: dict | None) -> bool:
    """报告是否处于"不可交付"状态（供 API / 前端 / 历史记忆复用）。"""
    return (gate or {}).get("status") in FAILED_STATUSES
