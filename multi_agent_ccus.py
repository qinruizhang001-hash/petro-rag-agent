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
    compress_history_if_needed,
    history_needs_compression,
    ensure_files as ensure_memory_files,
)

# ─── 图表生成（Function Calling 工具）──────────────────────
from chart_tools import generate_pressure_curve

# ─── 上下文压缩（Context Pruning）──────────────────────────
from context_pruner import prune_context, needs_pruning, context_char_count

# ─── 成本核算（Token Cost Model）───────────────────────────
from cost_model import estimate_cost, caption_text

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

# 路由表：stage → "local" | "cloud"，也可写成 "cloud:provider/model" 单独指定模型。
# 想调策略时只改这张表（或对应的 ROUTE_* 环境变量），不用碰任何节点代码。
#
# 支持 per-stage 指定模型，是因为一旦把 CLOUD_BASE_URL 指向本地聚合网关
# （magpie / one-api / LiteLLM 这类），不同厂商的模型会统一暴露成
# "provider/model" 命名，且各角色可以用不同厂商的模型 —— 这正是路由表的价值。
DEFAULT_ROUTING = {
    "extract": "local",      # Agent 1 参数提取（受动态复杂度路由管辖）
    "report_plan": "local",  # 工具规划：结构化短输出，本地足够（不参与动态路由）
    "report": "cloud",       # Agent 3 报告生成（受动态复杂度路由管辖）
    "evaluate": "local",     # Agent 4 评审员
    "history": "local",      # 文件记忆的结论压缩
    "summary": "local",      # 历史记忆的批量压缩（提炼核心经验）
    "prune": "local",        # 上下文压缩（用本地便宜模型换云端输入费）
}

# 路由解析是纯逻辑，已抽到 model_router.py 以便离线单测（见 tests/test_model_router.py）
from model_router import (
    build_routing,
    resolve_route as _resolve_route,
    classify_complexity,
    escalate_complexity,
    DYNAMIC_STAGES,
    COMPLEXITY_CHAR_THRESHOLD,
    RAG_MIN_SCORE,
)

# 允许用 ROUTE_<STAGE> 环境变量覆盖默认路由，例如 ROUTE_REPORT=cloud:kimi/kimi-k2
MODEL_ROUTING = build_routing(DEFAULT_ROUTING, os.environ)


def _ensure_loopback_bypass() -> None:
    """
    让回环地址绕过系统代理。

    这不是洁癖：若 http_proxy 指向一个本地代理，连 127.0.0.1 的请求会被转给它，
    代理在"目标端口没人监听"时返回 **502 Bad Gateway** 而不是 Connection refused。
    排查网关时会误以为是网关自己挂了（本项目就踩过这个坑）。

    必须在创建 OpenAI client **之前**执行 —— httpx 在构造时就读取这些环境变量。
    """
    loopback = ["localhost", "127.0.0.1", "::1", "0.0.0.0"]
    for var in ("NO_PROXY", "no_proxy"):
        existing = [p.strip() for p in (os.getenv(var) or "").split(",") if p.strip()]
        merged = existing + [h for h in loopback if h not in existing]
        os.environ[var] = ",".join(merged)


_ensure_loopback_bypass()

local_client = OpenAI(base_url=LOCAL_BASE_URL, api_key=LOCAL_API_KEY)
cloud_client = (
    OpenAI(base_url=CLOUD_BASE_URL, api_key=CLOUD_API_KEY) if CLOUD_API_KEY else None
)

# 兼容旧引用（app_ui.py 等外部脚本可能直接用 client）
client = local_client

_cloud_warned = False


def cloud_available() -> bool:
    return cloud_client is not None


def resolve_route_detail(stage: str, mode: str | None = None) -> tuple[str, str, str]:
    """
    把 stage 解析成 (tier, model_override, 降级原因)。

    model_override 为空表示用该档位的默认模型（LOCAL_MODEL / CLOUD_MODEL）。

    `mode` 是本次请求的复杂度档位（"local" / "cloud" / None）。
    对 DYNAMIC_STAGES（extract / report）它是**权威**的：静态路由表回答
    "这个角色通常该用什么"，复杂度路由回答"这一次值不值得"，后者更具体，
    所以优先级更高。其它角色（report_plan / evaluate / history / summary / prune）
    不受影响 —— 它们本来就该固定走便宜的本地模型。
    """
    tier, model_override, reason = _resolve_route(MODEL_ROUTING, stage, cloud_available())

    if not mode or stage not in DYNAMIC_STAGES:
        return tier, model_override, reason

    forced = "local" if mode == "local" else "cloud"
    if forced == tier:
        return tier, model_override, reason

    if forced == "cloud":
        if not cloud_available():
            return "local", model_override, "复杂度判定需云端，但 CLOUD_API_KEY 未配置"
        # 升到云端时沿用模型覆盖，但**仅当它本来就是云端 spec**：
        # `ROUTE_REPORT=cloud:kimi/kimi-k2` 的 kimi/kimi-k2 对云端端点有效，保留；
        # `ROUTE_REPORT=local:qwen2.5:14b` 的 qwen2.5:14b 是本地模型名，
        # 发给云端端点必然 404 —— 丢掉，回落到 CLOUD_MODEL。
        cloud_model = model_override if tier == "cloud" else ""
        return "cloud", cloud_model, f"复杂度路由 → 云端（原档位 {tier}）"

    # 降到本地：若原 spec 是云端专有模型名（如 ollama/qwen2.5:latest、
    # deepseek/deepseek-chat），这个模型名对本地端点毫无意义，必须丢弃，
    # 否则会把云端模型名发给 Ollama，直接 404。
    local_model = model_override if tier == "local" else ""
    return "local", local_model, f"复杂度路由 → 本地（原档位 {tier}）"


def resolve_route(stage: str, mode: str | None = None) -> tuple[str, str]:
    """把 stage 解析成实际执行的档位。返回 (tier, 降级原因)。"""
    tier, _model, reason = resolve_route_detail(stage, mode)
    return tier, reason


def get_llm(stage: str, mode: str | None = None):
    """按路由表返回 (client, model_name, 实际档位, 降级原因)。"""
    tier, model_override, reason = resolve_route_detail(stage, mode)
    if tier == "cloud":
        return cloud_client, (model_override or CLOUD_MODEL), "cloud", ""
    return local_client, (model_override or LOCAL_MODEL), "local", reason


def llm_chat(stage: str, messages: list, temperature: float, mode: str | None = None):
    """
    统一的大模型调用入口。

    所有节点都必须走这里 —— 这样"哪个角色用了哪个模型"才有唯一可信来源，
    也才能把 model / tier 如实记进 token_usage 供事后审计。
    """
    global _cloud_warned
    cli, model, tier, reason = get_llm(stage, mode)

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


class _StreamResult:
    """流式调用的返回值：攒好的全文 + token 统计 + 实际模型。"""

    __slots__ = ("text", "prompt_tokens", "completion_tokens", "model", "tier",
                 "usage_estimated")

    def __init__(self, text, prompt_tokens, completion_tokens, model, tier,
                 usage_estimated):
        self.text = text
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.model = model
        self.tier = tier
        self.usage_estimated = usage_estimated


def llm_chat_streamed(stage: str, messages: list, temperature: float, sink=None,
                      mode: str | None = None):
    """
    流式版本的大模型调用：每收到一个 delta 就交给 `sink`，同时攒全文。

    为什么不用 LangGraph 的 get_stream_writer 而是自己传 sink：
    实测 astream_events 会**丢弃** custom write，只保留 state 的 on_chain_stream
    （见 README「Streaming」一节）。所以 token 通道必须自己搭，不能指望框架。

    sink 是普通 callable（跨线程安全由调用方保证），传 dict：
        {"type": "token", "text": "..."}

    usage 处理：多数 OpenAI 兼容端点（含 Ollama）流式时不返回 usage，
    所以拿不到就按 tiktoken 估算，并在 token_usage 里标 usage_estimated=True ——
    宁可标"估算"，也不要让成本统计悄悄变成 0。
    """
    cli, model, tier, reason = get_llm(stage, mode)
    label = f"{tier}:{model}" + (f"（降级：{reason}）" if reason else "")
    print(f"  [Router] {stage} → {label}（流式）")

    def _open(with_usage: bool):
        kwargs = dict(model=model, messages=messages, temperature=temperature,
                      stream=True)
        if with_usage:
            kwargs["stream_options"] = {"include_usage": True}
        return cli.chat.completions.create(**kwargs)

    try:
        stream = _open(True)
    except Exception:
        # 部分网关/本地服务不认 stream_options，退回不带 usage 的流
        stream = _open(False)

    parts = []
    prompt_tokens = 0
    completion_tokens = 0
    for chunk in stream:
        usage = getattr(chunk, "usage", None)
        if usage is not None:
            prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
            completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        choices = getattr(chunk, "choices", None) or []
        if not choices:
            continue
        delta = getattr(choices[0].delta, "content", None)
        if delta:
            parts.append(delta)
            if sink is not None:
                try:
                    sink({"type": "token", "text": delta, "stage": stage})
                except Exception:
                    pass  # 推送失败绝不能打断主流程

    text = "".join(parts)
    estimated = False
    if completion_tokens <= 0:
        completion_tokens = count_tokens(text)
        estimated = True
    if prompt_tokens <= 0:
        prompt_tokens = sum(count_tokens(m.get("content", "")) for m in messages)
        estimated = True

    return _StreamResult(text, prompt_tokens, completion_tokens, model, tier,
                         estimated)


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
    return record_usage_payload(
        state,
        stage,
        prompt_tokens=int(getattr(usage_obj, "prompt_tokens", 0) or 0),
        completion_tokens=int(getattr(usage_obj, "completion_tokens", 0) or 0),
        model=model,
        tier=tier,
    )


def record_usage_payload(
    state: AgentState,
    stage: str,
    prompt_tokens: int,
    completion_tokens: int,
    model: str = "",
    tier: str = "",
    estimated: bool = False,
) -> dict:
    """record_llm_usage 的底层实现：直接给数字，供流式路径复用。"""
    prompt_tokens = int(prompt_tokens or 0)
    completion_tokens = int(completion_tokens or 0)
    total_tokens = prompt_tokens + completion_tokens
    print(
        f"  -> Token 消耗 [{stage}]: "
        f"prompt={prompt_tokens}, completion={completion_tokens}, "
        f"total={total_tokens}" + ("（估算）" if estimated else "")
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
    if estimated:
        payload["usage_estimated"] = True

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
    memory_compression: dict       # 本次是否触发历史压缩及压缩统计
    # ─── 图表 ───
    chart_info: dict               # generate_pressure_curve 的结果（路径/结论）
    # ─── Token 经济学：动态路由 ───
    complexity_mode: str           # 本次请求判定的档位："local" | "cloud"
    complexity_info: dict          # 判定依据（字符数/是否命中/RAG 相关度/日志标签）
    # ─── Token 经济学：上下文压缩 ───
    context_pruning: dict          # 压缩统计（前后字符数/状态/原因）
    cost_estimation: dict          # 本次任务的成本估算（人民币）


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

    # ─── 动态路由判定（入口）─────────────────────────────
    # 这是整个流程里第一次能看到 user_input 长度的位置，所以复杂度判定放这里。
    # 此时还没检索，rag_triggered=False 表示"尚未检索"而非"确定没命中"；
    # retrieve_knowledge 拿到 rerank 分数后会做单向升级。
    complexity = classify_complexity(state["user_input"])
    mode = complexity["mode"]
    print(f"  [Router] {complexity['label']} {complexity['reason']}")

    prompt = f"你是一个CCUS专家。请从以下描述中提取评估碳封存风险所需的关键参数：\n{state['user_input']}"
    _, model, tier, _ = get_llm("extract", mode)
    response = llm_chat(
        "extract",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        mode=mode,
    )
    return {
        "extracted_params": response.choices[0].message.content,
        "complexity_mode": mode,
        "complexity_info": complexity,
        **record_llm_usage(state, "extract", response, model, tier),
    }


# ============================================================
# 6. 知识检索（GraphRAG 图谱检索 + 本地 Rerank 精排）
# ============================================================
def _env_int(name: str, default: int) -> int:
    """读取整数型环境变量；非法值退回默认，不让配置失误炸掉主流程。"""
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        print(f"  [Config] {name}={raw!r} 不是合法整数，使用默认值 {default}")
        return default


# 注入给下游的片段数。调大能提高召回覆盖，但也直接抬高云端输入费 ——
# 这两个目标在 token 经济学里天然对立，所以做成显式配置而不是写死在代码里。
RAG_TOP_K = _env_int("RAG_TOP_K", 2)
# 上下文压缩阈值（字符）。超过就把长文本交给本地模型压成要点，再送给云端。
CONTEXT_PRUNE_THRESHOLD = _env_int("CONTEXT_PRUNE_THRESHOLD", 4000)


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
        # 没召回到东西 = RAG 未命中，复杂度判定维持 extract 阶段的结论
        complexity = classify_complexity(
            state["user_input"], rag_triggered=False, top_score=None
        )
        complexity = escalate_complexity(state.get("complexity_info"), complexity)
        print(f"  [Router] {complexity['label']} {complexity['reason']}")
        return {
            "retrieved_knowledge": "知识库中无相关记录",
            "token_usage": empty_usage,
            "complexity_mode": complexity["mode"],
            "complexity_info": complexity,
        }

    print(f"  -> LightRAG 图谱检索返回 {len(raw_docs)} 个候选片段")

    # ─── 第二阶段：本地 Rerank 精排（语义重排打分） ────────
    print(f"  -> 正在对 {len(raw_docs)} 个候选片段进行本地 Rerank 精排...")
    pairs = [[query, doc] for doc in raw_docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, raw_docs), key=lambda x: x[0], reverse=True)

    # 只保留语义最相关的 Top-K，降低后续 LLM 上下文 Token
    top_docs = [doc for score, doc in ranked[:RAG_TOP_K]]
    retrieved_text = "\n\n".join(top_docs)
    top_score = float(ranked[0][0]) if ranked else 0.0

    # ─── 复杂度升级：RAG 是否真的命中 ──────────────────────
    # 注意"召回了"≠"命中"：图谱对无关提问也会返回片段，但 rerank 分数接近 0。
    # 本项目实测无关提问的 Top-1 是 0.0001、真实 CCUS 提问在 0.27~0.74，
    # 所以用 RAG_MIN_SCORE 这条线把"有内容可用"和"凑数片段"分开，
    # 避免为了一个假命中就把整条链路推到云端。
    rag_triggered = top_score >= RAG_MIN_SCORE
    complexity = classify_complexity(
        state["user_input"], rag_triggered=rag_triggered, top_score=top_score
    )
    complexity = escalate_complexity(state.get("complexity_info"), complexity)
    mode = complexity["mode"]
    print(f"  [Router] {complexity['label']} {complexity['reason']}")

    query_tokens = count_tokens(query)
    candidate_tokens = sum(count_tokens(doc) for doc in raw_docs)
    rerank_input_tokens = query_tokens * len(raw_docs) + candidate_tokens

    print(f"  -> 精排完成！已从 {len(raw_docs)} 个片段中过滤出最相关的 {len(top_docs)} 个。")
    for i, (score, doc) in enumerate(ranked[:5], start=1):
        print(f"     #{i} score={float(score):.4f} tokens={count_tokens(doc)}")

    # ─── 第三阶段：上下文压缩（Context Pruning）────────────
    # 用本地便宜模型把长文本压成要点，再送给云端。
    # 为什么放在 retrieve 而不是 report 里：压缩的对象是"检索结果"，
    # 让它在这一节点就地变短，下游（报告 + 评审）拿到的就是精简版，
    # 两处都省 token，而不是只在报告那一步省。
    base_usage = merge_token_usage(
        state,
        "retrieve",
        {
            "query_tokens": query_tokens,
            "candidate_docs": len(raw_docs),
            "candidate_tokens": candidate_tokens,
            "rerank_input_tokens": rerank_input_tokens,
            "injected_context_tokens": count_tokens(retrieved_text),
            "top_k": len(top_docs),
            "top_score": round(top_score, 4),
            "rag_triggered": rag_triggered,
            "engine": "lightrag",
        },
    )

    prune_result = {
        "status": "skipped",
        "before_chars": context_char_count(retrieved_text),
        "after_chars": context_char_count(retrieved_text),
        "saved_chars": 0,
        "saved_ratio": 0.0,
        "reason": f"未达阈值（{CONTEXT_PRUNE_THRESHOLD} 字）",
    }
    prune_update: dict = {}

    if needs_pruning(retrieved_text, CONTEXT_PRUNE_THRESHOLD):
        print(f"  [Prune] 检索上下文 {context_char_count(retrieved_text)} 字 "
              f"> {CONTEXT_PRUNE_THRESHOLD} 字阈值 → 调用本地模型压缩")
        # 超长上下文会被切成多块分别压缩（本地模型只有 4096 tokens 上下文），
        # 所以这里必须**累加**每次调用的 token，不能只记最后一次 ——
        # 否则分 3 块压缩时成本统计只算 1/3。
        captured: dict = {
            "prompt_tokens": 0, "completion_tokens": 0, "calls": 0,
            "model": "", "tier": "", "estimated": False,
        }

        def _summarizer(prompt: str) -> str:
            _, p_model, p_tier, _ = get_llm("prune")
            resp = llm_chat(
                "prune",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            usage = getattr(resp, "usage", None)
            captured["prompt_tokens"] += int(getattr(usage, "prompt_tokens", 0) or 0)
            captured["completion_tokens"] += int(
                getattr(usage, "completion_tokens", 0) or 0
            )
            captured["calls"] += 1
            captured["model"] = p_model
            captured["tier"] = p_tier
            if usage is None:
                captured["estimated"] = True
            return resp.choices[0].message.content

        prune_result = prune_context(
            retrieved_text, summarizer=_summarizer, threshold=CONTEXT_PRUNE_THRESHOLD
        )

        if captured["calls"] > 0:
            # 压缩本身也是真实调用，必须记账，否则成本统计会漏掉自己
            prune_update = record_usage_payload(
                {**state, "token_usage": base_usage},
                "prune",
                captured["prompt_tokens"],
                captured["completion_tokens"],
                captured["model"],
                captured["tier"],
                estimated=captured["estimated"],
            )

        if prune_result["status"] == "pruned":
            retrieved_text = prune_result["compressed"]
            print(f"  [Prune] 🗜️ {prune_result['before_chars']} → "
                  f"{prune_result['after_chars']} 字"
                  f"（省 {prune_result['saved_chars']} 字 / "
                  f"{prune_result['saved_ratio'] * 100:.0f}%，"
                  f"{prune_result['chunks']} 块 / {prune_result['calls']} 次调用）")
        else:
            print(f"  [Prune] ⚠️ 压缩未生效，保留原文：{prune_result['reason']}")

    # 压缩后重新统计实际注入量（这是真正花在云端输入上的那份）
    injected_tokens = count_tokens(retrieved_text)
    if prune_result["status"] == "pruned":
        base_usage["retrieve"]["injected_context_tokens"] = injected_tokens
        base_usage["retrieve"]["prune_saved_chars"] = prune_result["saved_chars"]
    base_usage["retrieve"]["prune_status"] = prune_result["status"]

    print(
        f"  -> Token 消耗 [retrieve]: query={query_tokens}, "
        f"candidates={candidate_tokens}, rerank_input≈{rerank_input_tokens}, "
        f"injected_context={injected_tokens}"
    )

    update = {
        "retrieved_knowledge": retrieved_text,
        "token_usage": prune_update.get("token_usage", base_usage),
        "context_pruning": {
            k: prune_result.get(k)
            for k in ("status", "before_chars", "after_chars",
                      "saved_chars", "saved_ratio", "reason",
                      "chunks", "calls")
        },
        "complexity_mode": mode,
        "complexity_info": complexity,
    }
    # ⚠️ model_routing 只在压缩真的发生时才回写。
    # LangGraph 对 dict 字段是**替换**语义，不是合并：早先这里写成
    # `"model_routing": prune_update.get("model_routing", {})`，未压缩时
    # 返回空字典，把 extract 节点刚记好的路由整条抹掉了 —— 结果是
    # 审计用的 model_routing 里永远看不到 extract，而日志一切正常。
    # 这类"只在审计字段上丢信息"的 bug 不会报错，只能靠 E2E 断言兜住。
    if prune_update.get("model_routing"):
        update["model_routing"] = prune_update["model_routing"]
    return update


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


def _extract_depth(params: str) -> float | None:
    """
    从参数里抓储层深度（m）。抓不到返回 None，由 chart_tools 用默认值并标注"估算"。

    兼容 "深度 3000 m" / "埋深：2500米" / "depth 2000 m" 等写法。
    """
    m = re.search(
        r"(?:埋深|深度|depth)[^\d\-]{0,12}(\d+(?:\.\d+)?)\s*(?:m|米|meters?)?",
        params or "", re.I,
    )
    if not m:
        return None
    try:
        val = float(m.group(1))
    except ValueError:
        return None
    # 明显不合理（如把 3000 写成 3）就丢掉，交给默认值
    return val if 100 <= val <= 8000 else None


def _chart_payload(result: dict | None) -> dict:
    """把图表工具结果裁成可 JSON 序列化的精简结构（推前端 + 写状态）。"""
    if not result or result.get("status") != "ok":
        return {}
    keep = (
        "path", "rel_path", "url_path", "summary", "verdict", "verdict_text",
        "depth_m", "injection_pressure_bar", "fracture_bar_at_depth",
        "hydrostatic_bar_at_depth", "admissible_depth_min_m",
        "admissible_depth_max_m", "depth_assumed",
    )
    return {k: result.get(k) for k in keep if k in result}


_TOOL_FENCE = re.compile(r"```(?:json)?\s*\{.*?\}\s*```", re.DOTALL)


def strip_tool_json(text: str) -> str:
    """
    去掉正文里残留的工具 JSON 代码块。

    即便 prompt 明确禁止，模型仍可能把 ```json``` 混进终稿。这类内容一旦
    流到前端，用户看到的就是一串带花括号的机器指令 —— 属于"看起来像 bug"的
    输出，必须在入库/展示前清掉。
    """
    if not text:
        return ""
    cleaned = _TOOL_FENCE.sub("", text)
    # 清理因删除而产生的多余空行（保留段落间的单个空行）
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


# ============================================================
# 6-B. 成本核算（Token Cost Dashboard）
# ============================================================
def compute_cost_estimation(state: dict) -> dict:
    """
    从最终状态算出本次任务的预估成本（人民币）。

    为什么不把这个计算塞进某个节点：
    成本要在**所有** token 都记完之后才算得准，而最后一个节点是
    save_file_memory（它自己也会调一次模型）。放在节点里就得处理
    "早退分支没算成本"的问题；做成纯函数、由 API 层在收尾时调用，
    同步与流式两条路径共用同一份逻辑，两个数字不可能对不上。

    `caption` 一并算好，前端直接渲染 —— 见 cost_model.caption_text 的说明。
    """
    est = estimate_cost(
        state.get("token_usage") or {},
        state.get("model_routing") or {},
    )
    est["caption"] = caption_text(est)
    return est


SUMMARY_PROMPT = """你是一名 CCUS 项目档案管理员。下面是该项目的历次评估记录（旧记录）。

请把它们提炼成 **3 条最高级别的「项目核心经验」**，供后续评估直接复用。

要求：
1. 只要 3 条，每条一行，以 "- " 开头，不要编号、不要分小标题、不要开场白。
2. 每条必须**具体可复用**：包含触发条件 + 后果 + 应对措施，
   例如"注入压力 > 250 bar 且存在隐伏断层时，断层活化风险显著上升，需布设微地震监测"。
3. 合并重复场景，只保留规律性认识，不要罗列单次结论。
4. 若旧记录信息不足以支撑 3 条，有几条写几条，不要编造。

【旧评估记录】：
{old_text}

直接输出 3 条 "- " 开头的经验，不要输出其他任何内容。"""


def compress_history_with_llm(state: AgentState) -> dict:
    """
    超阈值时调用本地模型压缩历史记忆，返回可并入节点返回值的状态更新。

    为什么单独包一层而不直接把 llm_chat 塞进 file_memory：
    压缩是一次真实的 LLM 调用，token 消耗必须被记账（否则成本统计失真）；
    而 file_memory 模块不允许依赖任何 LLM 客户端（循环导入 + 离线单测）。
    所以这里做"回调注入"：file_memory 只管文件与阈值判断，模型调用留在本模块。
    """
    if not history_needs_compression():
        return {}

    captured: dict = {}

    def _summarizer(old_text: str) -> str:
        _, model, tier, _ = get_llm("summary")
        response = llm_chat(
            "summary",
            messages=[{
                "role": "user",
                "content": SUMMARY_PROMPT.format(old_text=old_text[:12000]),
            }],
            temperature=0.2,
        )
        captured["response"] = response
        captured["model"] = model
        captured["tier"] = tier
        return response.choices[0].message.content

    result = compress_history_if_needed(summarizer=_summarizer)

    if result["status"] == "compressed":
        print(f"  [Memory] 🗜️ 历史压缩：{result['before_chars']} → "
              f"{result['after_chars']} 字，{result['dropped_entries']} 条旧记录已提炼")
        print(f"  [Memory] 核心经验：\n{result['lessons']}")

    if "response" not in captured:
        return {}

    update = record_llm_usage(
        state, "history_compress", captured["response"],
        captured["model"], captured["tier"],
    )
    update["memory_compression"] = {
        "status": result["status"],
        "before_chars": result["before_chars"],
        "after_chars": result["after_chars"],
        "dropped_entries": result["dropped_entries"],
        "lessons": result["lessons"],
    }
    return update


def generate_report(state: AgentState, config=None):
    """
    Agent 3：生成评估报告。

    两阶段结构（这是有意为之，不是为了省事）：

      Phase A「工具规划」——非流式、短输出、低温。
        只让模型回答"需要调哪些工具、参数是什么"，把 JSON 指令拿回来。
      Phase B「终稿」——流式输出，且**只流这一遍**。

    为什么不沿用"一稿带工具指令 + 二次整合"：
    那种写法必须生成两遍完整报告，前端若两遍都流式，用户会看到报告重复出现，
    而且第一遍里的 ```json``` 会原样显示在屏幕上。成本也翻倍。
    拆成"短规划 + 单次终稿"后：只流一份、不出现原始 JSON、总 token 更少。
    """

    print("\n[Agent 3] 正在生成 CCUS 项目综合评估报告...")

    # ─── 文件记忆：读取 user_profile.md + project_history.md ───
    # 首次进入本节点时读取一次并写回 state，重写轮次直接复用（避免每轮重复 IO）
    file_memory_block = state.get("file_memory_context")
    compress_update = {}
    if file_memory_block is None:
        compress_update = compress_history_with_llm(state)
        file_memory_block = build_memory_context()
        if file_memory_block:
            print(f"  -> 文件记忆已加载（{len(file_memory_block)} 字）")
        else:
            print("  -> 文件记忆为空，跳过注入")

    personal_context = state.get("personal_context", "")
    web_context = ""
    if state.get("web_content"):
        web_context = f"\n【联网搜索补充内容（已自动入库）】：\n{state['web_content']}"

    # 所有流式调用共用的 token 推送通道（SSE 用；invoke 时为 None）
    sink = (config or {}).get("configurable", {}).get("token_sink")

    # ─── System Prompt：角色 + 长期文件记忆 + 写作硬约束 ───
    system_prompt = f"""你是一个资深的CCUS首席工程师，负责撰写 CO2 地质封存项目的风险评估报告。

{file_memory_block if file_memory_block else "（暂无文件记忆）"}

【写作硬性约束】：
1. 如果知识库中没有相关记录，必须直接输出"知识库中无相关记录"。
2. 必须遵循上面「用户画像」中的报告风格偏好（结论先行、分点陈述、数值落地）。
3. **禁止使用"综上所述""总而言之""总的来说"等总结性套话**，直接分点陈述。
4. 如果本次场景与「历史评估记录」中的某次相近，应显式做纵向对比
   （例如"相比上次 250bar/80℃ 的评估，本次压力更高，断层活化风险上升"）。
5. 请分点列出：1. 主要地质封存风险 2. 主要井筒风险 3. 缓解与监测建议。
6. **绝对不要输出 JSON、工具指令或图片 Markdown 路径**，只写正文。"""

    # ─── 本次任务的材料（两个阶段共用）───
    materials = f"""【提取的 CCUS 参数】：{state['extracted_params']}
【知识库检索到的标准】：{state['retrieved_knowledge']}
{web_context}
{personal_context}
【上一版报告的修改意见】：{state.get('feedback', '无')}"""

    # ══════════════════════════════════════════════════════
    # Phase A：工具规划（短输出、非流式）
    # ══════════════════════════════════════════════════════
    plan_prompt = materials + """

【任务：只做工具规划，不要写报告】
你有两个工具：
1. `calculate_co2_density` —— 参数中出现注入压力和温度时调用，不要自己口算密度。
2. `generate_pressure_curve` —— 必须调用，生成"地层压力随深度"地质图。
   depth 取参数中的储层埋深；参数里没有就省略该字段（系统会按默认值绘制并标注"估算"），
   **不要编造深度**。

只输出 json 代码块，每个工具一个，不要输出任何解释或正文。数值必须取自【提取的 CCUS 参数】：
```json
{"tool": "calculate_co2_density", "pressure_bar": 250, "temperature_celsius": 80}
```
```json
{"tool": "generate_pressure_curve", "depth": 2500, "injection_pressure_bar": 250}
```"""

    plan_msgs = [
        {"role": "system", "content": "你是工具调用规划器，只输出 JSON 代码块。"},
        {"role": "user", "content": plan_prompt},
    ]
    _, plan_model, plan_tier, _ = get_llm("report_plan")
    plan_resp = llm_chat("report_plan", messages=plan_msgs, temperature=0.0)
    state_update = record_llm_usage(
        {**state, **compress_update}, "report_plan", plan_resp, plan_model, plan_tier
    )
    usage_state = {**state, **compress_update, **state_update}

    # 扫描**所有** json 代码块（一份报告可能同时要密度和图）
    tool_requests = []
    for block in re.findall(r'```json\s*(\{.*?\})\s*```', plan_resp.choices[0].message.content or "", re.DOTALL):
        try:
            data = json.loads(block)
        except Exception as e:
            print(f"  [Warning] 工具指令解析失败: {e}")
            continue
        if isinstance(data, dict) and data.get("tool"):
            tool_requests.append(data)
    print(f"  -> 工具规划返回 {len(tool_requests)} 条指令")

    density_result = None
    chart_result = None
    for req in tool_requests:
        name = req.get("tool")
        if name == "calculate_co2_density":
            density_result = calculate_co2_density(
                req.get("pressure_bar"), req.get("temperature_celsius")
            )
        elif name == "generate_pressure_curve":
            try:
                chart_result = generate_pressure_curve(
                    req.get("depth"), req.get("injection_pressure_bar")
                )
            except Exception as e:
                print(f"  [Warning] 图表生成失败: {e}")

    # ─── 确定性兜底：模型该调没调时，由代码补上 ─────────────
    # 本地 7B 模型在上下文变丰富后会"觉得自己能算"，从而跳过工具调用，
    # 于是密度变成模型口算值 —— 这会直接破坏"数值不可幻觉"的核心承诺。
    # 因此不让它靠自觉：参数里有 P/T 就必须有工具结果。
    p_val, t_val = _extract_pressure_temperature(state.get("extracted_params", ""))
    if density_result is None and p_val is not None and t_val is not None:
        print("  [System] 模型未规划密度计算，但参数含压力/温度 → 确定性兜底")
        density_result = calculate_co2_density(p_val, t_val)

    # 图表同理：报告没有配图会显得业余，且输入几乎总能从参数推出。
    if chart_result is None or chart_result.get("status") != "ok":
        depth_val = _extract_depth(state.get("extracted_params", ""))
        print("  [System] 模型未规划图表 → 确定性兜底绘图")
        try:
            chart_result = generate_pressure_curve(depth_val, p_val)
        except Exception as e:
            print(f"  [Warning] 图表生成失败（不影响报告）: {e}")
            chart_result = None

    chart_ok = bool(chart_result and chart_result.get("status") == "ok")
    if chart_ok:
        print(f"  [System] 图表已生成：{chart_result['rel_path']}")
        # 立刻推给前端：用户不用等报告写完就能看到图
        if sink is not None:
            try:
                sink({"type": "chart", "chart": _chart_payload(chart_result)})
            except Exception:
                pass

    # ══════════════════════════════════════════════════════
    # Phase B：终稿（流式，唯一被流式展示的一遍）
    # ══════════════════════════════════════════════════════
    tool_facts = []
    if density_result is not None:
        tool_facts.append(f"CO2 密度 = {density_result}（由物理工具计算）")
    if chart_ok:
        tool_facts.append(f"地层压力曲线结论：{chart_result['summary']}")

    facts_block = ""
    if tool_facts:
        facts_block = (
            "【工具执行结果 —— 必须写进报告正文】\n- " + "\n- ".join(tool_facts)
            + "\n\n"
        )

    # 工具结果放在**最前面**：实测本地模型对结尾处的约束遵守度很差，
    # 把数值要求前置后，模型更可能把密度直接写进正文而不是靠代码兜底补。
    final_user = (
        facts_block
        + materials
        + "\n\n请撰写完整报告。上面工具结果里的数值必须原样写进正文"
          "（例如在地质风险部分点明该压力/温度下的 CO2 密度），"
          "不要改写、不要省略，也不要单独列成附注；"
          "不要输出 JSON 指令，不要在正文里写图片路径。"
    )
    final_msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": final_user},
    ]

    # ─── 动态路由：沿用本次请求的复杂度档位 ────────────────
    # 档位在 extract_params 里判定、在 retrieve_knowledge 里按 RAG 命中情况
    # 单向升级，这里只是消费它 —— 报告生成是整条链路里最贵的一步，
    # 简单请求不该为它付云端的钱。
    report_mode = state.get("complexity_mode") or "cloud"
    complexity = state.get("complexity_info") or {}
    if complexity.get("label"):
        print(f"  [Router] {complexity['label']} {complexity.get('reason', '')}")

    _, model, tier, _ = get_llm("report", report_mode)
    streamed = llm_chat_streamed("report", final_msgs, 0.3, sink=sink, mode=report_mode)
    report_text = strip_tool_json(streamed.text)
    state_update = record_usage_payload(
        usage_state, "report",
        streamed.prompt_tokens, streamed.completion_tokens, model, tier,
        estimated=streamed.usage_estimated,
    )

    # ─── 数值兜底：工具算出来的数不能"没写进报告"───────────
    # 模型偶尔会漏掉注入的数值。数值来自工具是核心承诺，漏了就等于没算，
    # 所以这里补一行显式数值，而不是让它自生自灭。
    if density_result is not None:
        numeric = density_result.split()[0] if density_result else ""
        if numeric and numeric not in report_text:
            print(f"  [System] 报告未包含工具算出的密度 {numeric} → 追加数值说明")
            report_text = report_text.rstrip() + (
                f"\n\n> CO2 密度 {density_result}，按注入压力与温度由状态方程工具计算。"
            )

    # ─── 报告末尾补齐图片引用（代码保证，不靠模型自觉）──────
    # 报告本体存 Markdown 相对路径，便于归档与其它消费者；
    # 前端用 chart 事件里的真实路径调 st.image，不会重复渲染。
    if chart_ok:
        ref = f"![地层压力随深度变化曲线]({chart_result['rel_path']})"
        if chart_result["rel_path"] not in report_text:
            report_text = report_text.rstrip() + f"\n\n{ref}\n"

    # ─── 自动保存报告到个人长期记忆 ───────────────────────
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
        "chart_info": _chart_payload(chart_result) if chart_ok else {},
        **compress_update,
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
    print(f"  -> 已追加到 project_history.md（当前 {size} 字符）")

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

# ============================================================
# 12. 运行测试
# ============================================================
if __name__ == "__main__":
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