"""
LangGraph 工作流定义与编译
==========================
图的**全部结构**集中在这一个文件：节点注册、边、条件路由、编译。

为什么把图结构和节点实现分开：改路由（比如加一个节点、调一个分支条件）时，
应该只需要读这一个文件就能看懂整条链路，而不是在 1300 行里翻找
``add_edge``。节点文件只回答"这一步做什么"，这里只回答"步骤怎么串"。

两个编译版本，差异是 checkpoint：

  ``app``         带 SqliteSaver，供同步 ``invoke``（CLI、/api/analyze/sync）
  ``app_stream``  不带 checkpointer，供 ``astream_events``（SSE 端点）

必须分开的原因：``SqliteSaver`` 只实现同步接口，``astream_events`` 走异步路径
会直接抛 ``NotImplementedError: The SqliteSaver does not support async methods``。
无 checkpoint 的代价是流式请求不能跨连接续跑；附带好处是 checkpoint.db
不再随每次 API 调用膨胀。长期记忆由文件记忆与 Memory Router 承担，
与 checkpoint 是两套机制，不受影响。
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
from nodes.conditions import check_quality, route_memory
from nodes.evaluate_node import evaluate_report
from nodes.extract_node import extract_params
from nodes.history_node import save_file_memory
from nodes.memory_node import inject_memory, memory_router, search_memory
from nodes.report_node import generate_report
from nodes.retrieve_node import retrieve_knowledge
from nodes.web_search_node import should_search, web_search_save
from utils.cost_model import compute_cost_estimation

__all__ = ["app", "app_stream", "NODE_LABELS", "compute_cost_estimation", "main"]


workflow = StateGraph(AgentState)

# 注册节点
workflow.add_node("memory_router", memory_router)
workflow.add_node("search_memory", search_memory)
workflow.add_node("inject_memory", inject_memory)
workflow.add_node("extract", extract_params)
workflow.add_node("retrieve", retrieve_knowledge)
workflow.add_node("web_search", web_search_save)
workflow.add_node("report", generate_report)
workflow.add_node("evaluate", evaluate_report)
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

# 原有流程
workflow.add_edge("extract", "retrieve")

# ★ 关键：retrieve 之后，判断本地知识是否够用
workflow.add_conditional_edges(
    "retrieve",
    should_search,
    {
        "web_search": "web_search",
        "report": "report",
    }
)

# 联网搜索完成后，继续生成报告
workflow.add_edge("web_search", "report")

# 报告生成 → 评审
workflow.add_edge("report", "evaluate")

workflow.add_conditional_edges(
    "evaluate",
    check_quality,
    {
        "pass": "save_file_memory",
        "rewrite": "report"
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

# 节点名 → 面向用户的中文进度文案（SSE 的 node 事件用）
NODE_LABELS = {
    "memory_router": "判断是否需要调取个人记忆",
    "search_memory": "检索个人长期记忆",
    "inject_memory": "整理记忆上下文",
    "extract": "提取 CCUS 关键参数",
    "retrieve": "GraphRAG 图谱检索 + 精排",
    "web_search": "联网补充资料",
    "report": "生成风险评估报告",
    "evaluate": "评审报告质量",
    "save_file_memory": "写入项目历史记忆",
}


def main() -> None:
    """CLI 演示：跑一次完整评估，把报告 / 路由 / 历史 / token 打到终端。

    与原 ``python multi_agent_ccus.py`` 的行为一致（那里现在只做入口转发）。
    """
    print("\n🚀 开始执行 CCUS 多智能体（多模型路由 + 文件记忆 + 反思循环）...")

    # 确保文件记忆的两个 .md 存在
    ensure_memory_files()

    # 启动时打印一次路由表，便于确认配置是否生效
    print("\n[Model Routing] 配置：")
    print(f"  local base : {LOCAL_BASE_URL}")
    print(f"  cloud base : {CLOUD_BASE_URL}")
    for _stage in MODEL_ROUTING:
        _cli, _mdl, _tier, _reason = get_llm(_stage)
        print(f"  {_stage:<10} → {_tier}:{_mdl}"
              + (f"  ⚠️ {_reason}，已降级" if _reason else ""))
    if not cloud_available():
        print("  （未配置 CLOUD_API_KEY，report 阶段将使用本地模型；"
              "在 .env 中设置后可启用云端）")
    print("  （网关连通性自检：python check_gateway.py）")

    # 你的测试输入（含压力/温度参数，用于触发 Function Calling 密度计算）
    test_input = "某CCUS项目注入压力250bar，温度80℃，存在隐伏碳盐酸岩储层断层，请评估风险。"

    # 配置 Thread ID（用于 Checkpoint）
    config = {"configurable": {"thread_id": "ccus_test_function_calling_001"}}

    # 执行工作流（已移除 interrupt_before，一次 invoke 跑完整个流程）
    final_state = app.invoke({"user_input": test_input}, config)

    print("\n" + "=" * 40)
    print("最终生成的 CCUS 风险评估报告：")
    print("=" * 40)
    print(final_state["final_report"])

    print("\n" + "=" * 40)
    print("模型路由（实际使用）：")
    print("=" * 40)
    print(final_state.get("model_routing", {}))

    print("\n" + "=" * 40)
    print("本次写入 project_history.md 的记录：")
    print("=" * 40)
    print(final_state.get("history_entry", "（无）"))

    print("\n" + "=" * 40)
    print("Token 消耗统计：")
    print("=" * 40)
    # 打印 Token 统计（根据你代码里的结构，这里可能需要微调）
    if "token_usage" in final_state:
        print(final_state["token_usage"])
    else:
        print("未找到 Token 统计，请检查代码中的 merge_token_usage 逻辑。")


if __name__ == "__main__":
    main()
