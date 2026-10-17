from __future__ import annotations

import os
import sys
import json
import re
import sqlite3
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from openai import OpenAI
import chromadb
from sentence_transformers import CrossEncoder

# ─── 环境预置 ──────────────────────────────────────────────
# HF_HUB_OFFLINE=1：跳过 HuggingFace 联网检查（bge-small-zh-v1.5 未缓存时
# 避免 5 次指数退避重试拖慢流程，直接走本地失败分支）。
# 如需下载该模型，临时 unset 此变量即可。
os.environ.setdefault("HF_HUB_OFFLINE", "1")

# ─── Personal Memory Module ─────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from memory.memory_manager import save_memory, retrieve_memory, search_semantic
from memory.memory_tools import search_personal_memory

# ─── Web Search + Auto-Save Module ──────────────────────────
from tools.knowledge_tools import search_and_save, is_tavily_configured

# ─── LightRAG GraphRAG 检索封装 ────────────────────────────
from lightrag_retriever import retrieve_documents, query_graph_answer

# 初始化 Ollama 客户端
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")


# ============================================================
# 0. 物理计算工具（供 Function Calling 调用）
# ============================================================
def calculate_co2_density(pressure_bar, temperature_celsius):
    """计算给定压力、温度下 CO2 密度的物理工具（简化状态方程）。"""
    print(f"  [Tool 调用] 正在计算压力 {pressure_bar} bar, 温度 {temperature_celsius}℃ 下的CO2密度...")
    density = 1.9 * pressure_bar / (temperature_celsius + 273.15)
    return f"{density:.2f} kg/m³"


try:
    import tiktoken
    _TOKEN_ENCODER = tiktoken.get_encoding("cl100k_base")
except Exception:
    _TOKEN_ENCODER = None

# 本地 Rerank 只加载一次，避免重复占内存
print("正在加载本地 Rerank 重排模型...")
reranker = CrossEncoder('./models/bge-reranker-v2-m3', max_length=512)
print("✅ Rerank 模型加载完成！")


def count_tokens(text: str) -> int:
    """统计文本 Token。优先 tiktoken(cl100k_base)，否则用 reranker tokenizer。"""
    if not text:
        return 0
    if _TOKEN_ENCODER is not None:
        return len(_TOKEN_ENCODER.encode(text))
    tokenizer = getattr(reranker, "tokenizer", None)
    if tokenizer is not None:
        return len(tokenizer.encode(text, add_special_tokens=False, truncation=False))
    return max(1, len(text) // 4)


def merge_token_usage(
    state: AgentState, stage: str, payload: dict, add_to_llm_total: bool = False
) -> dict:
    usage = dict(state.get("token_usage") or {})
    usage[stage] = payload
    if add_to_llm_total:
        usage["llm_total_tokens"] = int(usage.get("llm_total_tokens") or 0) + int(
            payload.get("total_tokens") or 0
        )
    else:
        usage.setdefault("llm_total_tokens", 0)
    return usage


def record_llm_usage(state: AgentState, stage: str, response) -> dict:
    usage_obj = getattr(response, "usage", None)
    prompt_tokens = int(getattr(usage_obj, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage_obj, "completion_tokens", 0) or 0)
    total_tokens = int(getattr(usage_obj, "total_tokens", 0) or 0) or (
        prompt_tokens + completion_tokens
    )
    print(
        f"  -> Token 消耗 [{stage}]: "
        f"prompt={prompt_tokens}, completion={completion_tokens}, total={total_tokens}"
    )
    return merge_token_usage(
        state,
        stage,
        {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        },
        add_to_llm_total=True,
    )


# ============================================================
# 1. 状态结构（新增 need_memory + personal_context）
# ============================================================
class AgentState(TypedDict):
    user_input: str
    extracted_params: str
    retrieved_knowledge: str
    final_report: str
    score: float
    feedback: str
    token_usage: dict
    # ─── 记忆系统字段 ───
    need_memory: bool              # Memory Router 判断是否需要搜记忆
    personal_memory: str           # 检索到的原始记忆内容
    personal_context: str          # 整理后注入给后续节点的上下文
    # ─── 联网搜索字段 ───
    web_content: str               # 联网搜索+自动沉淀的新内容
    # ─── 反思重试计数 ───
    retry_count: int               # 报告重写次数，用于防止无限循环


# ============================================================
# 2. Memory Router（新入口节点）
# ============================================================

# 记忆触发关键词
MEMORY_TRIGGERS = [
    # 中文
    "我的", "我之前", "我上次", "我做过", "我写过", "我的笔记",
    "复盘", "总结一下我", "项目进展", "我学到", "失败经验",
    "论文笔记", "我读过", "上次评估", "之前的报告", "我记得",
    # English
    "my ", "i previously", "last time", "i did", "i wrote", "my notes",
    "review", "my project", "i learned", "failure", "paper notes",
    "i read", "previous report", "i remember",
]


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


# ============================================================
# 3. Search Memory 节点
# ============================================================
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


# ============================================================
# 4. Inject Memory 节点
# ============================================================
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


# ============================================================
# 5. 参数提取（原有 Agent 1）
# ============================================================
def extract_params(state: AgentState):
    print("\n[Agent 1] 正在提取 CCUS 项目关键参数...")
    prompt = f"你是一个CCUS专家。请从以下描述中提取评估碳封存风险所需的关键参数：\n{state['user_input']}"
    response = client.chat.completions.create(
        model="qwen2.5", messages=[{"role": "user", "content": prompt}], temperature=0.1
    )
    return {
        "extracted_params": response.choices[0].message.content,
        "token_usage": record_llm_usage(state, "extract", response),
    }


# ============================================================
# 6. 知识检索（GraphRAG 图谱检索 + 本地 Rerank 精排）
# ============================================================
def retrieve_knowledge(state: AgentState):
    print("\n[Agent 2] 正在通过 LightRAG 知识图谱检索，并执行本地 Rerank 精排...")
    query = state["extracted_params"]

    # ─── 第一阶段：LightRAG 图谱检索（实体 + 关系 + chunk） ──
    raw_docs = retrieve_documents(query, mode="hybrid")

    if not raw_docs:
        empty_usage = merge_token_usage(
            state,
            "retrieve",
            {
                "query_tokens": count_tokens(query),
                "candidate_docs": 0,
                "candidate_tokens": 0,
                "rerank_input_tokens": 0,
                "injected_context_tokens": 0,
                "note": "知识图谱中无相关记录",
                "engine": "lightrag",
            },
        )
        print("  -> Token 消耗 [retrieve]: 图谱中无相关记录")
        return {"retrieved_knowledge": "知识库中无相关记录", "token_usage": empty_usage}

    print(f"  -> LightRAG 图谱检索返回 {len(raw_docs)} 个候选片段")

    # ─── 第二阶段：本地 Rerank 精排（语义重排打分） ────────
    print(f"  -> 正在对 {len(raw_docs)} 个候选片段进行本地 Rerank 精排...")
    pairs = [[query, doc] for doc in raw_docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, raw_docs), key=lambda x: x[0], reverse=True)

    # 只保留语义最相关的 Top-2，降低后续 LLM 上下文 Token
    top_docs = [doc for score, doc in ranked[:2]]
    retrieved_text = "\n\n".join(top_docs)

    query_tokens = count_tokens(query)
    candidate_tokens = sum(count_tokens(doc) for doc in raw_docs)
    rerank_input_tokens = query_tokens * len(raw_docs) + candidate_tokens
    injected_context_tokens = count_tokens(retrieved_text)

    print(f"  -> 精排完成！已从 {len(raw_docs)} 个片段中过滤出最相关的 {len(top_docs)} 个。")
    print(
        f"  -> Token 消耗 [retrieve]: query={query_tokens}, "
        f"candidates={candidate_tokens}, rerank_input≈{rerank_input_tokens}, "
        f"injected_context={injected_context_tokens}"
    )
    for i, (score, doc) in enumerate(ranked[:5], start=1):
        print(f"     #{i} score={float(score):.4f} tokens={count_tokens(doc)}")

    return {
        "retrieved_knowledge": retrieved_text,
        "token_usage": merge_token_usage(
            state,
            "retrieve",
            {
                "query_tokens": query_tokens,
                "candidate_docs": len(raw_docs),
                "candidate_tokens": candidate_tokens,
                "rerank_input_tokens": rerank_input_tokens,
                "injected_context_tokens": injected_context_tokens,
                "top_k": len(top_docs),
                "engine": "lightrag",
            },
        ),
    }


# ============================================================
# 7. 报告生成（原有 Agent 3，注入 personal_context + web_content + 自动保存）
# ============================================================
def generate_report(state: AgentState):
    print("\n[Agent 3] 正在生成 CCUS 项目综合评估报告...")

    # 构建记忆上下文（从 inject_memory 节点注入的 personal_context）
    personal_context = state.get("personal_context", "")

    # 构建联网搜索上下文
    web_context = ""
    if state.get("web_content"):
        web_context = f"\n【联网搜索补充内容（已自动入库）】：\n{state['web_content']}"

    prompt = f"""你是一个资深的CCUS首席工程师。请严格基于以下信息，撰写一份CCUS项目风险评估报告。
【提取的 CCUS 参数】：{state['extracted_params']}
【知识库检索到的标准】：{state['retrieved_knowledge']}
{web_context}
{personal_context}
【上一版报告的修改意见】：{state.get('feedback', '无')}

【写作硬性约束】：如果知识库中没有相关记录，必须直接输出"知识库中无相关记录"。
请分点列出：1. 主要地质封存风险 2. 主要井筒风险 3. 缓解与监测建议。

【重要工具使用规则】：
如果你在撰写报告时，需要计算特定压力和温度下的 CO2 密度，请直接输出一个 JSON 指令，格式如下（不要加其他废话）：
注意：pressure_bar 和 temperature_celsius 必须从【提取的 CCUS 参数】中读取用户实际提供的数值，禁止编造或照抄示例数值。
```json
{{"tool": "calculate_co2_density", "pressure_bar": 250, "temperature_celsius": 80}}
```
系统会自动执行该计算并将结果返回给你，你再把结果写进最终报告里。"""
    response = client.chat.completions.create(
        model="qwen2.5", messages=[{"role": "user", "content": prompt}], temperature=0.3
    )
    report_text = response.choices[0].message.content
    usage = record_llm_usage(state, "report", response)

    # ─── Function Calling：拦截 JSON 工具指令并二次执行 ────
    tool_match = re.search(r'```json\s*(\{.*?\})\s*```', report_text, re.DOTALL)
    if tool_match:
        try:
            tool_data = json.loads(tool_match.group(1))
            if tool_data.get("tool") == "calculate_co2_density":
                p = tool_data.get("pressure_bar")
                t = tool_data.get("temperature_celsius")
                density_result = calculate_co2_density(p, t)
                print(f"  [System] 工具计算完成：{density_result}，正在让大模型整合进报告...")
                follow_up_prompt = (
                    f"你刚才调用的工具计算结果为：{density_result}。"
                    f"请把这个具体数值自然地融合进你的最终报告中，不要输出JSON了。"
                )
                final_response = client.chat.completions.create(
                    model="qwen2.5",
                    messages=[
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": report_text},
                        {"role": "user", "content": follow_up_prompt}
                    ],
                    temperature=0.3
                )
                report_text = final_response.choices[0].message.content
                usage = record_llm_usage(state, "report_tool_call", final_response)
        except Exception as e:
            print(f"  [Warning] 工具解析失败: {e}")

    # ─── 自动保存报告到个人长期记忆 ───────────────────────
    # 只在第一次生成时保存（有 feedback 时说明是重写，不重复存）
    if not state.get("feedback"):
        try:
            print("  -> 正在保存报告到个人长期记忆...")
            save_memory(
                title=f"CCUS评估报告: {state['user_input'][:50]}",
                content=report_text,
                tags=["CCUS", "评估报告", "auto-save"]
            )
            print("  -> 报告已保存到个人长期记忆 (Notion + ChromaDB)")
        except Exception as e:
            print(f"  -> 记忆保存失败（不影响主流程）: {e}")

    current_retry = state.get("retry_count", 0)
    return {
        "final_report": report_text,
        "retry_count": current_retry + 1,
        "token_usage": usage,
    }


# ============================================================
# 8. 评审员（原有 Agent 4）
# ============================================================
def evaluate_report(state: AgentState):
    print("\n[Agent 4] 正在审核报告质量...")
    prompt = f"""你是一个严格的CCUS学术评审员。请根据【检索到的文献】，评估【AI生成的回答】的质量。
【用户问题】：{state['user_input']}
【检索到的文献】：{state['retrieved_knowledge']}
【AI生成的回答】：{state['final_report']}

请严格按以下格式输出，不要输出任何其他内容：
分数：[0-10的整数]
评语：[1-2句简短评语]"""

    response = client.chat.completions.create(
        model="qwen2.5", messages=[{"role": "user", "content": prompt}], temperature=0.0
    )
    result = response.choices[0].message.content

    try:
        score = int(result.split("分数：")[1].split("\n")[0].strip())
        feedback = result.split("评语：")[1].strip()
    except:
        score = 10
        feedback = result

    print(f"  -> 当前得分：{score} 分 | 评语：{feedback}")
    return {
        "score": score,
        "feedback": feedback,
        "token_usage": record_llm_usage(state, "evaluate", response),
    }


# ============================================================
# 9. 条件判断路由
# ============================================================
def check_quality(state: AgentState):
    print(f"\n[System] 检查报告质量，当前得分：{state['score']}")
    retry_count = state.get("retry_count", 0)
    # 如果分数 >= 8 或者 重试次数 >= 2，强制结束循环（防止无限重写）
    if state["score"] >= 8 or retry_count >= 2:
        print("[System] ✅ 报告达标，流程结束。")
        if state.get("personal_context"):
            print("[System] 📝 本次评估参考了个人长期记忆")
        return "pass"
    else:
        print(f"[System] ⚠️ 报告未达标（重试 {retry_count}/2），打回重写！")
        return "rewrite"


def route_memory(state: AgentState):
    """
    Memory Router 的条件边：根据 need_memory 决定走哪条路。
    """
    if state.get("need_memory"):
        return "search_memory"
    else:
        return "extract"


# ============================================================
# 10. 联网搜索：本地不够 → 自动搜索 → 写入本地库+Notion
# ============================================================
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


# ============================================================
# 11. 构建工作流图（Memory Router + Web Search 条件路由）
# ============================================================
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
        "pass": END,
        "rewrite": "report"
    }
)

# ============================================================
# 12. 初始化 Checkpoint 机制
# ============================================================
conn = sqlite3.connect("checkpoint.db", check_same_thread=False)
memory_saver = SqliteSaver(conn)
app = workflow.compile(checkpointer=memory_saver)

# ============================================================
# 12. 运行测试
# ============================================================
if __name__ == "__main__":
    print("\n🚀 开始执行 CCUS 多智能体（带反思循环 + 语义精排 + 外部裁判）...")

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
    print("Token 消耗统计：")
    print("=" * 40)
    # 打印 Token 统计（根据你代码里的结构，这里可能需要微调）
    if "token_usage" in final_state:
        print(final_state["token_usage"])
    else:
        print("未找到 Token 统计，请检查代码中的 merge_token_usage 逻辑。")