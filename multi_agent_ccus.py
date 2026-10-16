from __future__ import annotations

import os
import sqlite3
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from openai import OpenAI
import chromadb
from sentence_transformers import CrossEncoder

# 初始化 Ollama 客户端
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

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


# 1. 定义状态结构
class AgentState(TypedDict):
    user_input: str
    extracted_params: str
    retrieved_knowledge: str
    final_report: str
    score: float
    feedback: str
    token_usage: dict


# 2. 提取参数
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


# 3. 【升级版】知识检索：粗排 + 精排
def retrieve_knowledge(state: AgentState):
    print("\n[Agent 2] 正在本地CCUS知识库检索，并执行两阶段重排...")
    db_client = chromadb.PersistentClient(path=r"D:\python-learn\chroma_petro")
    # 注意：这里连接的是你之前清洗后存入的库名 ccus_knowledge_v2
    collection = db_client.get_or_create_collection(name="ccus_knowledge_v2")

    results = collection.query(query_texts=["channel formation mechanism in porous media"], n_results=5)
    raw_docs = results['documents'][0] if results['documents'] else []

    if not raw_docs:
        empty_usage = merge_token_usage(
            state,
            "retrieve",
            {
                "query_tokens": count_tokens(state["extracted_params"]),
                "candidate_docs": 0,
                "candidate_tokens": 0,
                "rerank_input_tokens": 0,
                "injected_context_tokens": 0,
                "note": "知识库中无相关记录",
            },
        )
        print("  -> Token 消耗 [retrieve]: 无命中文档")
        return {"retrieved_knowledge": "知识库中无相关记录", "token_usage": empty_usage}

    # 第二阶段：精排（语义重排打分）
    print(f"  -> 正在对 {len(raw_docs)} 篇候选文献进行语义精排...")
    query = state["extracted_params"]
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

    print(f"  -> 精排完成！已从 {len(raw_docs)} 篇中过滤出最相关的 {len(top_docs)} 篇。")
    print(
        f"  -> Token 消耗 [retrieve]: query={query_tokens}, "
        f"candidates={candidate_tokens}, rerank_input≈{rerank_input_tokens}, "
        f"injected_context={injected_context_tokens}"
    )
    for i, (score, doc) in enumerate(ranked, start=1):
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
            },
        ),
    }


# 4. 生成报告
def generate_report(state: AgentState):
    print("\n[Agent 3] 正在生成 CCUS 项目综合评估报告...")
    prompt = f"""你是一个资深的CCUS首席工程师。请严格基于以下信息，撰写一份CCUS项目风险评估报告。
【提取的 CCUS 参数】：{state['extracted_params']}
【知识库检索到的标准】：{state['retrieved_knowledge']}
【上一版报告的修改意见】：{state.get('feedback', '无')}

【写作硬性约束】：如果知识库中没有相关记录，必须直接输出“知识库中无相关记录”。
请分点列出：1. 主要地质封存风险 2. 主要井筒风险 3. 缓解与监测建议。"""
    response = client.chat.completions.create(
        model="qwen2.5", messages=[{"role": "user", "content": prompt}], temperature=0.3
    )
    return {
        "final_report": response.choices[0].message.content,
        "token_usage": record_llm_usage(state, "report", response),
    }


# 5. LLM 评审员（Agent 4）
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


# 6. 条件判断路由
def check_quality(state: AgentState):
    print(f"\n[System] 检查报告质量，当前得分：{state['score']}")
    if state["score"] >= 9:
        print("[System] ✅ 报告达标，流程结束。")
        return "pass"
    else:
        print("[System] ⚠️ 报告未达标，打回重写！")
        return "rewrite"


# 7. 构建工作流图
workflow = StateGraph(AgentState)
workflow.add_node("extract", extract_params)
workflow.add_node("retrieve", retrieve_knowledge)
workflow.add_node("report", generate_report)
workflow.add_node("evaluate", evaluate_report)

workflow.set_entry_point("extract")
workflow.add_edge("extract", "retrieve")
workflow.add_edge("retrieve", "report")
workflow.add_edge("report", "evaluate")

workflow.add_conditional_edges(
    "evaluate",
    check_quality,
    {
        "pass": END,
        "rewrite": "report"
    }
)

# 8. 初始化 Checkpoint 机制
conn = sqlite3.connect("checkpoint.db", check_same_thread=False)
memory = SqliteSaver(conn)
app = workflow.compile(checkpointer=memory, interrupt_before=["report"])

# 9. 运行测试
if __name__ == "__main__":
    config = {"configurable": {"thread_id": "ccus_reflection_002"}}
    test_input = "含烃碳酸盐岩储层中多孔介质的通道形成机制是什么？"

    print("🚀 开始执行 CCUS 多智能体（带反思循环 + 语义精排）...")
    events = app.invoke({"user_input": test_input, "token_usage": {}}, config)

    print("\n⏸️ 系统已暂停！参数提取和精排检索已完成。")
    print(f"提取的参数：\n{events['extracted_params']}")
    print(f"\n检索的文献片段：\n{events['retrieved_knowledge'][:200]}...")

    user_approval = input("\n是否批准生成正式报告？(y/n): ")

    if user_approval.lower() == 'y':
        print("\n[System] 人工审核通过，继续生成报告...")
        final_state = app.invoke(None, config)
        print("\n" + "=" * 40)
        print("最终生成的 CCUS 风险评估报告：")
        print("=" * 40)
        print(final_state["final_report"])
        print("\n" + "=" * 40)
        print("Token 消耗统计：")
        print("=" * 40)
        print(final_state.get("token_usage") or {})
    else:
        print("\n[System] ❌ 人工审核未通过，任务终止。")
        print(f"已产生的 Token 消耗：{events.get('token_usage') or {}}")