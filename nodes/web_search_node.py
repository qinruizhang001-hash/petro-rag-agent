"""
联网补料：本地知识不够时自动搜索并沉淀
======================================
``should_search`` 是条件边（判断），``web_search_save`` 是节点（执行）。

触发规则刻意保守：结果太短、或包含"未找到/无相关记录"这类措辞才联网。
在知识图谱已经覆盖的领域里，联网只会引入不可控的外部文本，
反而稀释检索结果的可信度 —— 这里的目标是"兜底"，不是"补充"。

Tavily 未配置时 ``should_search`` 直接返回 ``report``，
整条支路被短路，不影响主流程。
"""
from __future__ import annotations

from core.state import AgentState
from tools.knowledge_tools import is_tavily_configured, search_and_save
from utils.tokens import count_tokens, merge_token_usage


def should_search(state: AgentState):
    """
    判断本地检索结果是否足够。
    结果太短或包含"未找到"类消息 → 触发联网搜索
    """
    local = state.get("retrieved_knowledge", "")

    # 如果 Tavily 没配置，直接跳过
    if not is_tavily_configured():
        print("[Web Router] Tavily 未配置，跳过联网搜索")
        return "report"

    # 简单规则：结果太短或包含"未找到"就触发搜索
    if len(local) < 150 or "未找到" in local or "无相关记录" in local or "没有相关" in local:
        print("[Web Router] 本地知识不足，触发联网搜索")
        return "web_search"
    else:
        print("[Web Router] 本地知识充足，直接生成报告")
        return "report"


def web_search_save(state: AgentState):
    """
    执行联网搜索，把有价值的结果写入本地 ChromaDB + Notion。
    返回搜索到的内容摘要，供报告生成使用。
    """
    print("\n[Web Search] 本地知识不足，正在联网搜索并自动沉淀...")

    # 用用户原始问题 + 提取的参数构造搜索 query
    query = state.get("extracted_params", state["user_input"])[:200]

    result = search_and_save(query, max_results=4)

    web_tokens = count_tokens(result)
    print(f"  -> 联网搜索完成，内容约 {web_tokens} tokens")

    return {
        "web_content": result,
        "token_usage": merge_token_usage(
            state, "web_search",
            {
                "query": query,
                "query_tokens": count_tokens(query),
                "result_tokens": web_tokens,
                "source": "tavily",
            }
        ),
    }
