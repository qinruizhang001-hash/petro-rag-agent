"""
记忆相关节点：判断 → 检索 → 注入
================================
三个节点构成一条可选支路。为什么不合并成一个：
``memory_router`` 只做关键词判断（零成本、零 IO），把它独立出来，
图里才能在"不需要记忆"时**直接跳过**检索与注入两个节点 ——
否则每次评估都会白跑一次 ChromaDB 查询。

``MEMORY_TRIGGERS`` 是纯关键词匹配，故意不用语义判断：
这一步跑在流程最前面，任何模型调用都会把首字延迟推后。
"""
from __future__ import annotations

from core.state import AgentState
from memory.memory_tools import search_personal_memory
from utils.tokens import count_tokens, merge_token_usage


MEMORY_TRIGGERS = ['我的', '我之前', '我上次', '我做过', '我写过', '我的笔记', '复盘', '总结一下我', '项目进展', '我学到', '失败经验', '论文笔记', '我读过', '上次评估', '之前的报告', '我记得', 'my ', 'i previously', 'last time', 'i did', 'i wrote', 'my notes', 'review', 'my project', 'i learned', 'failure', 'paper notes', 'i read', 'previous report', 'i remember']


def memory_router(state: AgentState):
    """
    入口节点：判断用户问题是否需要调用个人记忆。

    逻辑：关键词匹配 + 简单语义判断。
    如果命中任一触发词 → need_memory = True → 进入 search_memory 节点
    否则 → need_memory = False → 直接跳到 extract 节点
    """
    print("\n[Memory Router] 正在判断是否需要个人记忆...")
    user_text = state["user_input"].lower().strip()

    need_memory = any(trigger in user_text for trigger in MEMORY_TRIGGERS)

    if need_memory:
        print("  -> 检测到记忆触发词，将搜索个人长期记忆")
    else:
        print("  -> 无记忆触发词，直接进入 CCUS 分析流程")

    return {"need_memory": need_memory}


def search_memory(state: AgentState):
    """
    执行个人记忆检索。
    使用 search_personal_memory 工具，返回格式化的记忆内容。
    """
    print("\n[Search Memory] 正在检索个人长期记忆...")
    query = state["user_input"]

    try:
        memory_result = search_personal_memory(query)

        # 统计 token
        mem_tokens = count_tokens(memory_result)
        print(f"  -> 检索完成，记忆内容约 {mem_tokens} tokens")

        usage = merge_token_usage(
            state, "memory_search",
            {
                "query": query,
                "query_tokens": count_tokens(query),
                "result_tokens": mem_tokens,
                "source": "chromadb+notion",
            }
        )

        return {
            "personal_memory": memory_result,
            "token_usage": usage,
        }
    except Exception as e:
        print(f"  -> 记忆检索失败: {e}")
        return {
            "personal_memory": f"记忆检索失败: {e}",
            "token_usage": merge_token_usage(
                state, "memory_search",
                {"error": str(e), "result_tokens": 0}
            ),
        }


def inject_memory(state: AgentState):
    """
    把检索到的记忆整理成上下文，供后续节点使用。
    """
    print("\n[Inject Memory] 正在整理记忆上下文...")

    memory = state.get("personal_memory", "无相关个人记忆")

    # 如果记忆内容不是"未找到"消息，就构建上下文
    if "未在个人记忆中找到" in memory or "检索失败" in memory:
        personal_context = ""
        print("  -> 无有效记忆，跳过注入")
    else:
        personal_context = f"""
### 个人长期记忆（请优先参考）
{memory}

请在生成回答时，结合以上个人记忆中的内容。如果记忆与当前问题相关，必须引用。
"""
        print("  -> 记忆已注入上下文")

    return {"personal_context": personal_context}
