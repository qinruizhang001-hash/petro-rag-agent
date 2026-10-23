"""
LangGraph 工作流定义与编译
==========================
图的**全部结构**集中在这一个文件：节点注册、边、条件路由、编译。

为什么把图结构和节点实现分开：改路由（比如加一个节点、调一个分支条件）时，
应该只需要读这一个文件就能看懂整条链路，而不是在 1300 行里翻找
``add_edge``。节点文件只回答"这一步做什么"，这里只回答"步骤怎么串"。

拓扑（并行扇出 / 扇入）
-----------------------
::

    memory_router ─┬─ search_memory → inject_memory ─┐
                   └─────────────────────────────────┴─→ extract → rewrite_query
                                                                  │
                                            ┌─────────────────────┴─────────────────────┐
                                            ▼                                           ▼
                                    geology_retrieve                              wellbore_retrieve
                                            └──────────────────┬────────────────────────┘
                                                               ▼
                                                             grade ── low → retry both with expanded strategy
                                                               │         (max one retry; failed evidence withheld)
                                                               ▼
                                            ┌──────────────────┴────────────────────────┐
                                            ▼                                           ▼
                                    geology_generate                             wellbore_generate
                                            └──────────────────┬────────────────────────┘
                                                               ▼
                               aggregate → evaluate → hitl_gate → save_file_memory
                                           ▲           │
                                           └─ rewrite / reject

地质与井筒的检索阶段并行执行，``grade`` 是真屏障：它必须等两路 Top-K
候选都返回后才会评判。通过评判（或一次替代检索后仍不足）才并行生成分支
结论，随后 aggregate 等两路生成都完成才执行。低相关回路最多重检一次。

两条硬约束（写错不会报"结果不对"，而是直接抛异常或静默丢数据）：

1. 同一个 key 被两条分支并发写 → ``InvalidUpdateError``。所以
   ``token_usage`` / ``model_routing`` 在 ``core/state.py`` 里带 reducer，
   而分支各自的产物用**独立字段**（``geology_report`` / ``wellbore_report``）。
2. 分支内**不允许抛异常**：节点抛异常会让整个 superstep 失败，另一条分支
   已经算完的结果一起丢。所以 geology / wellbore 把异常收进自己的
   ``*_status``，由 aggregate 降级处理。

三个编译版本，差异只在编译参数：

  ``app``         带 SqliteSaver，供同步 ``invoke``（CLI、/api/analyze/sync）
  ``app_stream``  不带 checkpointer，供 ``astream_events``（SSE 端点）
  ``app_hitl``    带 SqliteSaver + ``interrupt_before=["hitl_gate"]``，
                  供 ``/api/analyze/hitl`` + ``/api/approve`` 的人工审核链路

必须分开的原因：``SqliteSaver`` 只实现同步接口，``astream_events`` 走异步路径
会直接抛 ``NotImplementedError: The SqliteSaver does not support async methods``。
无 checkpoint 的代价是流式请求不能跨连接续跑；附带好处是 checkpoint.db
不再随每次 API 调用膨胀。长期记忆由文件记忆与 Memory Router 承担，
与 checkpoint 是两套机制，不受影响。

``app_hitl`` 必须带 checkpointer：暂停意味着"进程可以退出、几小时后由另一次
HTTP 请求续跑"，状态只能落在 sqlite 里。
"""
from __future__ import annotations

import core  # noqa: F401  必须最先：触发 core/__init__ 的环境预置

import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph

from core.config import (
    CLOUD_BASE_URL,
    LOCAL_BASE_URL,
    MODEL_ROUTING,
    cloud_available,
    get_llm,
)
from core.state import AgentState
from file_memory import ensure_files as ensure_memory_files
from nodes.aggregate_node import aggregate_report
from nodes.conditions import check_quality, route_after_retrieval_grade, route_memory
from nodes.evaluate_node import evaluate_report
from nodes.extract_node import extract_params
from nodes.geology_node import generate_geology, retrieve_geology
from nodes.history_node import save_file_memory
from nodes.hitl_node import hitl_gate, route_after_hitl
from nodes.memory_node import inject_memory, memory_router, search_memory
from nodes.wellbore_node import generate_wellbore, retrieve_wellbore
from nodes.query_nodes import grade_node, rewrite_node
from utils.cost_model import compute_cost_estimation

from utils.logging_setup import get_logger

log = get_logger(__name__)

__all__ = [
    "app", "app_stream", "app_hitl",
    "NODE_LABELS", "compute_cost_estimation", "main",
]


workflow = StateGraph(AgentState)

# 注册节点
workflow.add_node("memory_router", memory_router)
workflow.add_node("search_memory", search_memory)
workflow.add_node("inject_memory", inject_memory)
workflow.add_node("extract", extract_params)
workflow.add_node("rewrite_query", rewrite_node)
# ─── 并行检索 → Self-RAG 评判 → 并行生成 ───
workflow.add_node("geology_retrieve", retrieve_geology)
workflow.add_node("wellbore_retrieve", retrieve_wellbore)
workflow.add_node("grade", grade_node)
workflow.add_node("geology_generate", generate_geology)
workflow.add_node("wellbore_generate", generate_wellbore)
# ─── 扇入 ───
workflow.add_node("aggregate", aggregate_report)
workflow.add_node("evaluate", evaluate_report)
# 人类在环闸门：自动模式下自己放行并留痕，人工模式下被 interrupt_before 拦住
workflow.add_node("hitl_gate", hitl_gate)
workflow.add_node("save_file_memory", save_file_memory)

# 入口 → memory_router
workflow.set_entry_point("memory_router")

# 条件边：memory_router → search_memory (需要记忆) 或 extract (不需要)
workflow.add_conditional_edges(
    "memory_router",
    route_memory,
    {
        "search_memory": "search_memory",
        "extract": "extract",
    }
)

# 记忆路径：search_memory → inject_memory → extract
workflow.add_edge("search_memory", "inject_memory")
workflow.add_edge("inject_memory", "extract")

# 重写只产出检索 query；保留原始 user_input / extracted_params 作为证据与回退。
workflow.add_edge("extract", "rewrite_query")

# ★ 并行检索 → Grader 真屏障：两路 Top-K 都回来后才评判。
workflow.add_edge("rewrite_query", "geology_retrieve")
workflow.add_edge("rewrite_query", "wellbore_retrieve")
workflow.add_edge(["geology_retrieve", "wellbore_retrieve"], "grade")

# 低相关时有界地回到两路检索（检索策略已切换为 expanded）；通过或耗尽后
# 才并行生成。generate 节点对被拒证据会直接 abstain，不调用模型补写。
workflow.add_conditional_edges("grade", route_after_retrieval_grade)
workflow.add_edge(["geology_generate", "wellbore_generate"], "aggregate")

# 汇总 → 评审
workflow.add_edge("aggregate", "evaluate")

# 评审不达标 → 回到 aggregate 重写。
# 注意重写**不重跑检索分支**：地质/井筒结论已在状态里，重写只重做融合与成稿。
# 这既是省时间（不必再付一次 rerank 与两次分支生成），也是语义正确
# （改的是措辞与结构，不是证据）。
workflow.add_conditional_edges(
    "evaluate",
    check_quality,
    {
        "pass": "hitl_gate",
        "rewrite": "aggregate"
    }
)

# 人工闸门：通过 → 落盘；驳回 → 带审核意见回到 aggregate 重写
workflow.add_conditional_edges(
    "hitl_gate",
    route_after_hitl,
    {
        "approve": "save_file_memory",
        "reject": "aggregate",
    }
)

# 评审通过（或重试耗尽）→ 写入文件记忆 → 结束
workflow.add_edge("save_file_memory", END)

# ============================================================
# 12. 初始化 Checkpoint 机制
# ============================================================
conn = sqlite3.connect("checkpoint.db", check_same_thread=False)
memory_saver = SqliteSaver(conn)
app = workflow.compile(checkpointer=memory_saver)

# ─── SSE 专用的无 checkpoint 编译版本 ─────────────────────
# 为什么必须分开编译：SqliteSaver 只实现了同步接口，而 astream_events 走异步
# 路径，会直接抛 `NotImplementedError: The SqliteSaver does not support async
# methods`。AsyncSqliteSaver 又需要 aiosqlite，且与同步 saver 共用同一个 sqlite
# 文件会有锁竞争。
#
# 无 checkpoint 的代价：流式请求不写断点，因此不能跨请求续跑。
# 但单次评估本就是一次跑完（反思循环在同一 run 内），而"长期记忆"由文件记忆
# 与 Memory Router 承担 —— 它们跟 checkpoint 是两套机制，不受影响。
# 附带好处：checkpoint.db 不再随每次 API 调用膨胀。
app_stream = workflow.compile()

# ─── HITL 专用编译版本：在人工闸门前暂停 ──────────────────
# 三个版本共用同一份图定义（节点、边、条件路由只有一处），差别只在编译参数：
#   app        带 checkpointer，同步 invoke，自动放行闸门
#   app_stream 不带 checkpointer，供 SSE 异步流
#   app_hitl   带 checkpointer + interrupt_before，跑到闸门前停住等人工决定
#
# 为什么必须带 checkpointer：暂停意味着"进程可以退出、几小时后由另一次 HTTP
# 请求续跑"，状态只能存在 sqlite 里。`POST /api/approve` 就是拿 thread_id
# 找到那个断点，把 human_decision 写进 state 再 invoke(None)。
app_hitl = workflow.compile(
    checkpointer=memory_saver,
    interrupt_before=["hitl_gate"],
)

# 节点名 → 面向用户的中文进度文案（SSE 的 node 事件用）
NODE_LABELS = {
    "memory_router": "判断是否需要调取个人记忆",
    "search_memory": "检索个人长期记忆",
    "inject_memory": "整理记忆上下文",
    "extract": "提取 CCUS 关键参数",
    "rewrite_query": "将问题重写为专业英文检索词",
    "geology_retrieve": "地质分支：检索断层 / 盖层证据",
    "wellbore_retrieve": "井筒分支：检索套管 / 压力异常证据",
    "grade": "Self-RAG：评估 Top-K 证据相关性",
    "geology_generate": "地质分支：基于通过评估的证据生成结论",
    "wellbore_generate": "井筒分支：基于通过评估的证据生成结论",
    "aggregate": "汇总两路结论并成稿",
    "evaluate": "评审报告质量",
    "hitl_gate": "人工审核闸门",
    "save_file_memory": "写入项目历史记忆",
}


def main() -> None:
    """CLI 演示：跑一次完整评估，把报告 / 路由 / 历史 / token 打到终端。

    与原 ``python multi_agent_ccus.py`` 的行为一致（那里现在只做入口转发）。
    """
    log.info("\n🚀 开始执行 CCUS 多智能体（多模型路由 + 文件记忆 + 反思循环）...")

    # 确保文件记忆的两个 .md 存在
    ensure_memory_files()

    # 启动时打印一次路由表，便于确认配置是否生效
    log.info("\n[Model Routing] 配置：")
    log.info(f"  local base : {LOCAL_BASE_URL}")
    log.info(f"  cloud base : {CLOUD_BASE_URL}")
    for _stage in MODEL_ROUTING:
        _cli, _mdl, _tier, _reason = get_llm(_stage)
        log.info(f"  {_stage:<10} → {_tier}:{_mdl}"
              + (f"  ⚠️ {_reason}，已降级" if _reason else ""))
    if not cloud_available():
        log.info("  （未配置 CLOUD_API_KEY，report 阶段将使用本地模型；"
              "在 .env 中设置后可启用云端）")
    log.info("  （网关连通性自检：python check_gateway.py）")

    # 你的测试输入（含压力/温度参数，用于触发 Function Calling 密度计算）
    test_input = "某CCUS项目注入压力250bar，温度80℃，存在隐伏碳盐酸岩储层断层，请评估风险。"

    # 配置 Thread ID（用于 Checkpoint）
    config = {"configurable": {"thread_id": "ccus_test_function_calling_001"}}

    # 执行工作流（已移除 interrupt_before，一次 invoke 跑完整个流程）
    final_state = app.invoke({"user_input": test_input}, config)

    log.info("\n" + "=" * 40)
    log.info("最终生成的 CCUS 风险评估报告：")
    log.info("=" * 40)
    log.info(final_state["final_report"])

    log.info("\n" + "=" * 40)
    log.info("模型路由（实际使用）：")
    log.info("=" * 40)
    log.info(final_state.get("model_routing", {}))

    log.info("\n" + "=" * 40)
    log.info("本次写入 project_history.md 的记录：")
    log.info("=" * 40)
    log.info(final_state.get("history_entry", "（无）"))

    log.info("\n" + "=" * 40)
    log.info("Token 消耗统计：")
    log.info("=" * 40)
    # 打印 Token 统计（根据你代码里的结构，这里可能需要微调）
    if "token_usage" in final_state:
        log.info(final_state["token_usage"])
    else:
        log.info("未找到 Token 统计，请检查代码中的 merge_token_usage 逻辑。")


if __name__ == "__main__":
    main()
