"""
条件边函数（不是节点）
======================
LangGraph 的 ``add_conditional_edges`` 需要的是"读 state → 返回分支名"的纯函数。
它们不产出状态更新，所以不注册成节点 —— 混在 nodes 里只是为了让
"图的所有决策点"集中在一个目录下，便于一次读完全部跳转逻辑。

``check_quality`` 的三个出口
---------------------------
``pass``    → 落盘并结束
``rewrite`` → 回到 aggregate 重写
``fail``    → ``mark_failed``：写入失败状态、隔离物理否决的正文、结束

**没有第四个出口。** 旧实现在 ``retry_count >= 2`` 时"强制放行"，日志写一句
"结论质量未达阈值"就当成通过 —— API 照样返回 200、报告照样进历史记忆，
调用方看到的只有"流程结束"。那不是降级，是把失败伪装成成功。

判定本身在 ``core.gates.next_action``（纯函数、可单测），这里只负责读状态、
记日志。为什么不在条件边里做判断：条件边返回的是**分支名**，它改不了 state，
所以"过没过"必须由节点写进 ``quality_gate``，边只做路由。
"""
from __future__ import annotations

from core.gates import MAX_REPORT_REWRITES, next_action
from core.state import AgentState

from utils.logging_setup import get_logger

log = get_logger(__name__)


def check_quality(state: AgentState):
    gate = state.get("quality_gate") or {}
    retry_count = int(state.get("retry_count", 0) or 0)
    action = next_action(gate, retry_count)

    log.info(
        f"\n[System] 质量闸门：status={gate.get('status')} "
        f"score={gate.get('score')} 重写 {retry_count}/{MAX_REPORT_REWRITES}"
    )

    if action == "pass":
        log.info("[System] ✅ 报告达标，流程结束。")
        if state.get("personal_context"):
            log.info("[System] 📝 本次评估参考了个人长期记忆")
        if state.get("file_memory_context"):
            log.info("[System] 📝 本次评估参考了文件记忆（user_profile.md + project_history.md）")
        routing = state.get("model_routing") or {}
        if routing:
            log.info(f"[System] 🔀 模型路由：{routing}")
        return "pass"

    if action == "rewrite":
        log.warning(
            f"[System] ⚠️ 报告未达标（{gate.get('status')}），"
            f"重写 {retry_count}/{MAX_REPORT_REWRITES}：{gate.get('reason')}"
        )
        return "rewrite"

    # action == "fail"：这里必须明确说"失败"，不能出现"放行""降级通过"之类措辞。
    log.error(
        f"[System] ⛔ 质量闸门判定失败（status={gate.get('status')}，"
        f"重写 {retry_count}/{MAX_REPORT_REWRITES} 已用尽）：{gate.get('reason')}"
    )
    return "fail"


def route_after_retrieval_grade(state: AgentState):
    """Fan out to one bounded retry or to generation using graded evidence only."""
    grade = state.get("retrieval_grade") or {}
    if grade.get("status") == "retrying":
        log.info(
            "[Self-RAG] switching retrieval strategy to %s (retry %s/%s)",
            state.get("retrieval_strategy", "expanded"),
            state.get("retrieval_retry_count", 0),
            grade.get("retry_limit", 1),
        )
        return ["geology_retrieve", "wellbore_retrieve"]
    return ["geology_generate", "wellbore_generate"]


def route_memory(state: AgentState):
    """
    Memory Router 的条件边：根据 need_memory 决定走哪条路。
    """
    if state.get("need_memory"):
        return "search_memory"
    else:
        return "extract"
