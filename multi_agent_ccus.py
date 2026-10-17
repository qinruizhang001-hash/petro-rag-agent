# ─── 环境预置（必须在任何重依赖 import 之前）────────────────
# HF_HUB_OFFLINE=1：跳过 HuggingFace 联网检查。
# ⚠️ 顺序至关重要：huggingface_hub 在 **import 时**就把 HF_HUB_OFFLINE 读进常量，
# 之后再 setdefault 完全无效（历史 bug：导致每次流程末尾卡 5 次指数退避重试，
# 每个模型文件各重试一遍，累计数分钟）。因此这段必须放在最顶部。
# 如需真正下载 bge-small-zh-v1.5，启动前 unset 此变量即可。
from __future__ import annotations  # 必须首行，且不触发重依赖

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

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

# ─── Personal Memory Module ─────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from memory.memory_manager import save_memory, retrieve_memory, search_semantic
from memory.memory_tools import search_personal_memory

# ─── Web Search + Auto-Save Module ──────────────────────────
from tools.knowledge_tools import search_and_save, is_tavily_configured

# ─── LightRAG GraphRAG 检索封装 ────────────────────────────
from lightrag_retriever import retrieve_documents, query_graph_answer

# ─── 文件记忆（File-Based Memory）───────────────────────────
from file_memory import (
    build_memory_context,
    append_project_history,
    ensure_files as ensure_memory_files,
)

from dotenv import load_dotenv

load_dotenv()


# ============================================================
# 0-A. 多模型路由（Model Routing）
# ============================================================
# 设计动机：不同角色对"智商"的要求差异极大。
#   - 参数提取 / 评审打分 → 结构化、短输出、判断简单 → 本地 7B 足够，边际成本为 0
#   - 报告生成           → 长文本、强逻辑、专业措辞 → 云端大模型质量差距最明显
# 把钱只花在真正影响产出质量的那一步。

LOCAL_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
LOCAL_API_KEY = os.getenv("LOCAL_API_KEY", "ollama")
LOCAL_MODEL = os.getenv("LOCAL_MODEL", "qwen2.5")

CLOUD_BASE_URL = os.getenv("CLOUD_BASE_URL", "https://api.deepseek.com/v1")
CLOUD_API_KEY = os.getenv("CLOUD_API_KEY", "").strip()
CLOUD_MODEL = os.getenv("CLOUD_MODEL", "deepseek-chat")

# 路由表：stage → "local" | "cloud"
# 想调策略时只改这张表，不用碰任何节点代码。
MODEL_ROUTING = {
    "extract": "local",     # Agent 1 参数提取
    "report": "cloud",      # Agent 3 报告生成（核心产出，用云端）
    "evaluate": "local",    # Agent 4 评审员
    "history": "local",     # 文件记忆的结论压缩
}

local_client = OpenAI(base_url=LOCAL_BASE_URL, api_key=LOCAL_API_KEY)
cloud_client = (
    OpenAI(base_url=CLOUD_BASE_URL, api_key=CLOUD_API_KEY) if CLOUD_API_KEY else None
)

# 兼容旧引用（app_ui.py 等外部脚本可能直接用 client）
client = local_client

_cloud_warned = False


def cloud_available() -> bool:
    return cloud_client is not None


def resolve_route(stage: str) -> tuple[str, str]:
    """把 stage 解析成实际执行的档位。返回 (tier, 降级原因)。"""
    tier = MODEL_ROUTING.get(stage, "local")
    if tier == "cloud" and not cloud_available():
        return "local", "CLOUD_API_KEY 未配置"
    return tier, ""


def get_llm(stage: str):
    """按路由表返回 (client, model_name, 实际档位, 降级原因)。"""
    tier, reason = resolve_route(stage)
    if tier == "cloud":
        return cloud_client, CLOUD_MODEL, "cloud", ""
    return local_client, LOCAL_MODEL, "local", reason


def llm_chat(stage: str, messages: list, temperature: float):
    """
    统一的大模型调用入口。

    所有节点都必须走这里 —— 这样"哪个角色用了哪个模型"才有唯一可信来源，
    也才能把 model / tier 如实记进 token_usage 供事后审计。
    """
    global _cloud_warned
    cli, model, tier, reason = get_llm(stage)

    if reason and not _cloud_warned:
        print(
            f"  [Router] ⚠️ {stage} 原本要走云端，但{reason}，已降级为本地 {LOCAL_MODEL}。\n"
            f"           配置方式：在 .env 中设置 CLOUD_API_KEY 与 CLOUD_BASE_URL。"
        )
        _cloud_warned = True

    label = f"{tier}:{model}" + (f"（降级：{reason}）" if reason else "")
    print(f"  [Router] {stage} → {label}")

    return cli.chat.completions.create(
        model=model, messages=messages, temperature=temperature
    )


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


def record_llm_usage(
    state: AgentState, stage: str, response, model: str = "", tier: str = ""
) -> dict:
    """
    记录一次 LLM 调用的 token 与路由信息。

    返回**可直接并入节点返回值的状态更新片段**：
        {"token_usage": {...}, "model_routing": {...}}
    节点里用 `return {"final_report": text, **record_llm_usage(...)}` 合并即可。
    """
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

    payload = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }
    # 记录实际使用的模型，供事后审计"路由是否真的生效"
    if model:
        payload["model"] = model
    if tier:
        payload["tier"] = tier

    new_usage = merge_token_usage(state, stage, payload, add_to_llm_total=True)

    # 同时维护一份扁平的路由快照：{stage: "tier:model"}
    routing = dict(state.get("model_routing") or {})
    if tier and model:
        routing[stage] = f"{tier}:{model}"

    return {"token_usage": new_usage, "model_routing": routing}


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
    # ─── 多模型路由 ───
    model_routing: dict            # {stage: "tier:model"}，记录每个角色实际用了哪个模型
    # ─── 文件记忆 ───
    file_memory_context: str       # 注入 Prompt 的文件记忆文本
    history_entry: str             # 本次评估压成的一条历史记录
    history_saved: bool            # 幂等标记，防止重跑重复追加


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
    _, model, tier, _ = get_llm("extract")
    response = llm_chat(
        "extract",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    return {
        "extracted_params": response.choices[0].message.content,
        **record_llm_usage(state, "extract", response, model, tier),
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
def _extract_pressure_temperature(params: str) -> tuple[float | None, float | None]:
    """
    从参数提取结果中抓取压力 / 温度数值。

    这是 Function Calling 的**确定性兜底**：模型有时会认为"信息已够"
    而跳过工具调用，直接凭参数写出报告。此时数值就变成了模型的推测。
    用正则兜住 P/T，保证密度值一定来自物理工具而非语言模型。
    """
    p_match = re.search(r"(?:压力|pressure)[^\d\-]{0,15}(\d+(?:\.\d+)?)", params or "", re.I)
    t_match = re.search(r"(?:温度|temperature)[^\d\-]{0,15}(\d+(?:\.\d+)?)", params or "", re.I)
    p_val = float(p_match.group(1)) if p_match else None
    t_val = float(t_match.group(1)) if t_match else None
    return p_val, t_val


def generate_report(state: AgentState):
    print("\n[Agent 3] 正在生成 CCUS 项目综合评估报告...")

    # ─── 文件记忆：读取 user_profile.md + project_history.md ───
    # 首次进入本节点时读取一次并写回 state，重写轮次直接复用（避免每轮重复 IO）
    file_memory_block = state.get("file_memory_context")
    if file_memory_block is None:
        file_memory_block = build_memory_context()
        if file_memory_block:
            print(f"  -> 文件记忆已加载（{len(file_memory_block)} 字）")
        else:
            print("  -> 文件记忆为空，跳过注入")

    # 构建记忆上下文（从 inject_memory 节点注入的 personal_context）
    personal_context = state.get("personal_context", "")

    # 构建联网搜索上下文
    web_context = ""
    if state.get("web_content"):
        web_context = f"\n【联网搜索补充内容（已自动入库）】：\n{state['web_content']}"

    # ─── System Prompt：角色 + 长期文件记忆 + 写作硬约束 ───
    system_prompt = f"""你是一个资深的CCUS首席工程师，负责撰写 CO2 地质封存项目的风险评估报告。

{file_memory_block if file_memory_block else "（暂无文件记忆）"}

【写作硬性约束】：
1. 如果知识库中没有相关记录，必须直接输出"知识库中无相关记录"。
2. 必须遵循上面「用户画像」中的报告风格偏好（结论先行、分点陈述、数值落地）。
3. **禁止使用"综上所述""总而言之""总的来说"等总结性套话**，直接分点陈述。
4. 如果本次场景与「历史评估记录」中的某次相近，应显式做纵向对比
   （例如"相比上次 250bar/80℃ 的评估，本次压力更高，断层活化风险上升"）。
5. 请分点列出：1. 主要地质封存风险 2. 主要井筒风险 3. 缓解与监测建议。"""

    # ─── User Prompt：本次任务的具体材料 ───
    user_prompt = f"""【提取的 CCUS 参数】：{state['extracted_params']}
【知识库检索到的标准】：{state['retrieved_knowledge']}
{web_context}
{personal_context}
【上一版报告的修改意见】：{state.get('feedback', '无')}

【工具调用规则（强制）】：
只要【提取的 CCUS 参数】中出现注入压力和温度，你**必须先输出**下面这个 JSON 指令，
然后等系统返回计算结果，再把该结果写进报告。不要自己口算密度，也不要跳过这一步。
pressure_bar 和 temperature_celsius 必须取自上方【提取的 CCUS 参数】的真实数值，
禁止编造、禁止照抄下面示例里的数字。
```json
{{"tool": "calculate_co2_density", "pressure_bar": 250, "temperature_celsius": 80}}
```
若参数中确实没有压力或温度，才跳过该步骤。"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    # 核心产出走云端模型（路由表决定）
    _, model, tier, _ = get_llm("report")
    response = llm_chat("report", messages=messages, temperature=0.3)
    report_text = response.choices[0].message.content
    state_update = record_llm_usage(state, "report", response, model, tier)
    # 用"已合并的状态"继续记录后续调用，否则第二段会覆盖第一段的 token 统计
    usage_state = {**state, **state_update}

    # ─── Function Calling：拦截 JSON 工具指令并二次执行 ────
    tool_match = re.search(r'```json\s*(\{.*?\})\s*```', report_text, re.DOTALL)
    density_result = None

    if tool_match:
        try:
            tool_data = json.loads(tool_match.group(1))
            if tool_data.get("tool") == "calculate_co2_density":
                p = tool_data.get("pressure_bar")
                t = tool_data.get("temperature_celsius")
                density_result = calculate_co2_density(p, t)
        except Exception as e:
            print(f"  [Warning] 工具指令解析失败: {e}")

    # ─── 确定性兜底：模型该调没调时，由代码补上 ─────────────
    # 本地 7B 模型在上下文变丰富后会"觉得自己能算"，从而跳过工具调用，
    # 于是密度变成模型口算值 —— 这会直接破坏"数值不可幻觉"的核心承诺。
    # 因此不让它靠自觉：参数里有 P/T 就必须有工具结果。
    if density_result is None:
        p_val, t_val = _extract_pressure_temperature(state.get("extracted_params", ""))
        if p_val is not None and t_val is not None:
            print("  [System] 模型未主动调用工具，但参数含压力/温度 → 启动确定性兜底计算")
            density_result = calculate_co2_density(p_val, t_val)

    if density_result is not None:
        print(f"  [System] 工具计算完成：{density_result}，正在让大模型整合进报告...")
        follow_up_prompt = (
            f"你刚才调用的工具计算结果为：{density_result}。"
            f"请把这个具体数值自然地融合进你的最终报告中，不要输出JSON了。"
        )
        # 二次整合必须复用同一个模型：若首轮用云端、整合用本地小模型，
        # 报告后半段的措辞与逻辑会明显掉档，这是最常见的隐性质量 bug。
        final_response = llm_chat(
            "report",
            messages=messages
            + [
                {"role": "assistant", "content": report_text},
                {"role": "user", "content": follow_up_prompt},
            ],
            temperature=0.3,
        )
        report_text = final_response.choices[0].message.content
        state_update = record_llm_usage(
            usage_state, "report_tool_call", final_response, model, tier
        )

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
        "file_memory_context": file_memory_block,
        **state_update,
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

    _, model, tier, _ = get_llm("evaluate")
    response = llm_chat(
        "evaluate",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
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
        **record_llm_usage(state, "evaluate", response, model, tier),
    }


# ============================================================
# 8-B. 文件记忆后处理节点（评审通过后落盘）
# ============================================================
def save_file_memory(state: AgentState):
    """
    把本次评估压缩成一条结构化历史记录，追加到 project_history.md。

    为什么单独做成一个节点，而不是塞进 evaluate_report：
    - evaluate_report 会被反思循环多次调用（rewrite 分支回到 report），
      在里面写文件会把中间稿和最终稿都写进去，历史立刻被污染。
    - 单独节点挂在 check_quality 的 pass 分支后面，天然保证"每次任务只写一次"。
    """
    print("\n[File Memory] 正在把本次评估写入 project_history.md...")

    if state.get("history_saved"):
        print("  -> 本次任务已记录过，跳过（幂等）")
        return {"history_saved": True}

    report = state.get("final_report", "")
    if not report:
        print("  -> 报告为空，跳过记录")
        return {"history_saved": True}

    # 用本地小模型做压缩 —— 这一步只是信息抽取，不需要云端智商
    prompt = f"""请从下面这份 CCUS 风险评估报告中，提取一条**极简**的历史记录，供后续评估做纵向对比。

【用户原始需求】：{state['user_input']}
【提取的参数】：{state.get('extracted_params', '')}
【评审分数】：{state.get('score', '-')}
【报告全文】：
{report}

请严格按以下 Markdown 格式输出，不要输出任何额外内容、不要加一级/二级标题：

**场景**: 一句话描述项目场景（储层类型、断层情况等）
**关键参数**: 列出压力/温度/注入速率等具体数值，没有就写"未提供"
**核心结论**: 2-3 句，只写最重要的风险判断和对应缓解措施
**评审分**: {state.get('score', '-')}/10"""

    try:
        _, model, tier, _ = get_llm("history")
        response = llm_chat(
            "history",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        entry = response.choices[0].message.content.strip()
        usage_update = record_llm_usage(state, "history", response, model, tier)
    except Exception as e:
        # 云端/本地都失败时降级为规则拼接，保证历史不断档
        print(f"  -> 结论抽取失败（{e}），改用规则拼接")
        entry = (
            f"**场景**: {state['user_input'][:80]}\n"
            f"**关键参数**: 见提取结果\n"
            f"**核心结论**: {report[:200].replace(chr(10), ' ')}…\n"
            f"**评审分**: {state.get('score', '-')}/10"
        )
        usage_update = {}

    size = append_project_history(entry)
    print(f"  -> 已追加到 project_history.md（当前 {size} 字节）")

    return {
        "history_entry": entry,
        "history_saved": True,
        **usage_update,
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
        if state.get("file_memory_context"):
            print("[System] 📝 本次评估参考了文件记忆（user_profile.md + project_history.md）")
        routing = state.get("model_routing") or {}
        if routing:
            print(f"[System] 🔀 模型路由：{routing}")
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

# ============================================================
# 12. 运行测试
# ============================================================
if __name__ == "__main__":
    print("\n🚀 开始执行 CCUS 多智能体（多模型路由 + 文件记忆 + 反思循环）...")

    # 确保文件记忆的两个 .md 存在
    ensure_memory_files()

    # 启动时打印一次路由表，便于确认配置是否生效
    print("\n[Model Routing] 配置：")
    for _stage, _tier in MODEL_ROUTING.items():
        _resolved, _reason = resolve_route(_stage)
        _mdl = CLOUD_MODEL if _resolved == "cloud" else LOCAL_MODEL
        print(f"  {_stage:<10} → {_resolved}:{_mdl}" + (f"  ⚠️ {_reason}，已降级" if _reason else ""))
    if not cloud_available():
        print("  （未配置 CLOUD_API_KEY，report 阶段将使用本地模型；"
              "在 .env 中设置后可启用云端）")

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