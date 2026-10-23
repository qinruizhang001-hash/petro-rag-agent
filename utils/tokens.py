"""
Token 统计与用量记账
====================
三件事，都很容易被做错：

1. ``count_tokens`` —— 优先 tiktoken(cl100k_base)，退回 reranker 的 tokenizer，
   最后才用 len//4 粗估。中文按字符数近似会低估，所以前两条路必须优先。

   注意 reranker 是**兜底**，不是必需。它必须在用到时才导入：这个模块被
   ``core.config`` 间接引用，而 config 只是读环境变量、建 OpenAI 客户端 ——
   为了一个大概率用不上的 tokenizer 就在 import 期拉起 2.2 GB 的
   CrossEncoder，会让"只想读配置"的调用方也付出模型加载的代价，
   并且使 config 在无 torch 环境里直接不可导入。
2. ``merge_token_usage`` —— 把某个阶段的用量并进 state 的 ``token_usage`` 字典。
3. ``record_usage_payload`` —— **返回状态更新片段**而不是直接改 state。
   这一点是踩坑换来的：早先的实现返回一个裸 usage 字典，工具调用那一轮
   会把它覆盖掉，于是"第一遍的 token 数凭空消失"。返回 ``{"token_usage":…,
   "model_routing":…}`` 让调用方用 ``**`` 合并，累加才不会被静默吃掉。

记账的唯一可信来源是这里 —— 所有 LLM 调用都必须经过它，否则成本统计失真。
"""
from __future__ import annotations

try:
    import tiktoken

    _TOKEN_ENCODER = tiktoken.get_encoding("cl100k_base")
except Exception:
    _TOKEN_ENCODER = None

# 放在 tiktoken 之后 import，保持启动日志顺序与原实现一致
from core.state import AgentState  # noqa: E402

from utils.logging_setup import get_logger, llm_extra

log = get_logger(__name__)


def _fallback_tokenizer():
    """取 reranker 的 tokenizer，取不到就返回 None。

    延迟导入是刻意的：``core.reranker`` 在 import 期会加载 2.2 GB 的
    CrossEncoder，而这里的用途只是 tiktoken 缺席时的兜底计数。
    真实链路里 tiktoken 一定装着（requirements.txt），所以绝大多数情况下
    这个函数根本不会碰到 reranker。
    """
    try:
        from core.reranker import reranker
    except Exception as exc:  # torch / 模型文件缺失都算"没有兜底"
        log.debug(f"[tokens] reranker 兜底不可用：{exc}")
        return None
    return getattr(reranker, "tokenizer", None)


def count_tokens(text: str) -> int:
    """统计文本 Token。优先 tiktoken(cl100k_base)，否则用 reranker tokenizer。"""
    if not text:
        return 0
    if _TOKEN_ENCODER is not None:
        return len(_TOKEN_ENCODER.encode(text))
    tokenizer = _fallback_tokenizer()
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
    # 这条日志是"花了多少 token、走的哪个模型"的唯一权威记录：
    # 字段化（extra）而不是拼进正文，才能被 grep / 聚合。
    log.info(
        f"Token 消耗 [{stage}]: prompt={prompt_tokens}, "
        f"completion={completion_tokens}, total={total_tokens}"
        + ("（估算）" if estimated else ""),
        extra=llm_extra(model, tier, total_tokens),
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
