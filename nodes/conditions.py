"""
条件边函数（不是节点）
======================
LangGraph 的 ``add_conditional_edges`` 需要的是"读 state → 返回分支名"的纯函数。
它们不产出状态更新，所以不注册成节点 —— 混在 nodes 里只是为了让
"图的所有决策点"集中在一个目录下，便于一次读完全部跳转逻辑。

``check_quality`` 的两个出口：``pass`` → 落盘并结束；``rewrite`` → 回到 report。
重写上限（retry_count >= 2）必须在这里硬拦，否则低分报告会无限循环。
"""
from __future__ import annotations

from core.state import AgentState

from utils.logging_setup import get_logger

log = get_logger(__name__)


def check_quality(state: AgentState):
    log.info(f"\n[System] 检查报告质量，当前得分：{state['score']}")
    retry_count = state.get("retry_count", 0)
    # 如果分数 >= 8 或者 重试次数 >= 2，强制结束循环（防止无限重写）
    if state["score"] >= 8 or retry_count >= 2:
        # 两条出口必须分开记日志：重试耗尽时的放行**不是**"达标"。
        # 把 3 分写成"报告达标"会让日志失去可信度，事后排查时无法区分
        # "这份报告真的过了"和"我们放弃了"。
        if state["score"] >= 8:
            log.info("[System] ✅ 报告达标，流程结束。")
        else:
            log.warning(
                f"[System] ⚠️ 重写次数已用尽（{retry_count}/2），"
                f"当前仅 {state['score']:g} 分，**强制放行**——结论质量未达阈值。"
            )
        if state.get("personal_context"):
            log.info("[System] 📝 本次评估参考了个人长期记忆")
        if state.get("file_memory_context"):
            log.info("[System] 📝 本次评估参考了文件记忆（user_profile.md + project_history.md）")
        routing = state.get("model_routing") or {}
        if routing:
            log.info(f"[System] 🔀 模型路由：{routing}")
        return "pass"
    else:
        log.info(f"[System] ⚠️ 报告未达标（重试 {retry_count}/2），打回重写！")
        return "rewrite"


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
