"""
配置与模型客户端
================
这里回答三个问题，全部集中在一处：

1. **用哪些端点/密钥** —— ``OLLAMA_BASE_URL`` / ``CLOUD_BASE_URL`` / ``CLOUD_API_KEY``，
   统一从 ``.env`` 读取（``load_dotenv()`` 必须在读取环境变量之前调用）。
2. **哪个角色用哪个模型** —— ``DEFAULT_ROUTING`` + ``ROUTE_<STAGE>`` 环境变量覆盖，
   解析逻辑是纯函数，已抽到 ``model_router.py`` 以便离线单测。
3. **怎么调** —— ``llm_chat`` / ``llm_chat_streamed`` 是**唯一**的调用入口。
   所有节点都必须走这里，"哪个角色实际用了哪个模型"才有唯一可信来源，
   也才能把 model / tier 如实记进 token_usage 供事后审计。

为什么把调用函数和配置放在一起：它们共享 ``local_client`` / ``cloud_client``
这两个单例，拆开只会让"某处绕过路由直接调 client"变得更容易发生。

**回环地址绕过代理**（``_ensure_loopback_bypass``）不是洁癖：若 http_proxy 指向
本地代理，连 127.0.0.1 的请求会被转给它，代理在目标端口无人监听时返回
**502** 而不是 Connection refused —— 排查网关时会误以为是网关挂了。
"""
from __future__ import annotations

import os

from dotenv import load_dotenv
from openai import OpenAI

from model_router import DYNAMIC_STAGES, build_routing, resolve_route as _resolve_route
from utils.tokens import count_tokens

from utils.logging_setup import get_logger, llm_extra

load_dotenv()

log = get_logger(__name__)


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
        log.warning(
            f"[Router] ⚠️ {stage} 原本要走云端，但{reason}，已降级为本地 {LOCAL_MODEL}。"
            f"配置方式：在 .env 中设置 CLOUD_API_KEY 与 CLOUD_BASE_URL。",
            extra=llm_extra(model, tier),
        )
        _cloud_warned = True

    label = f"{tier}:{model}" + (f"（降级：{reason}）" if reason else "")
    log.info(f"[Router] {stage} → {label}", extra=llm_extra(model, tier))

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
    log.info(f"[Router] {stage} → {label}（流式）", extra=llm_extra(model, tier))

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
