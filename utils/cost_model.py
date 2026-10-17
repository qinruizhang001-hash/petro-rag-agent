"""
成本核算（Token Cost Model）
============================
把 token 用量折算成人民币，并算出"相对全云端基线省了多少钱"。

为什么单独成模块：
- 这是一张**价格表 + 一次乘法**，但它决定了前端那句"已为你节省 ¥x"是否可信。
  算错了不会报错、只会安静地给出一个漂亮但虚假的数字——这类错误必须被单测钉住。
- 纯函数、无 IO → CI 里零依赖可测。

两个必须说清楚的定义（否则"节省"是个没有意义的数字）：

1. **按实际模型计价，不按档位标签。**
   本项目当前把 CLOUD_BASE_URL 指向本地 magpie 网关，上游只有 Ollama，
   所以 tier="cloud" 的调用实际由本地 qwen2.5 完成、真实成本≈电费。
   如果按 tier 计价，就会把免费的本地说成"花了 0.017 元"，再"省下"一个幻觉数字。
   因此定价依据是 **model 名**，tier 只在模型名无法识别时兜底。

2. **基线 = 把本次所有 token 都按付费云端模型（默认 deepseek-chat）列表价计费。**
   这是一个明确的、可复算的反事实，而不是"随便取个好看的分母"。
   它不是"上一版系统的真实花费"——上一版也可能跑在本地。这一点在返回值的
   `baseline_note` 里写清楚，避免被读成"你上个月账单省了多少"。
"""
from __future__ import annotations

# ─── 价格表：人民币 / 百万 tokens ────────────────────────────
# DeepSeek 官方定价（api-docs.deepseek.com/zh-cn/quick_start/pricing，2026-10 查）：
#   deepseek-flash    高峰 输入 2.0 / 输出 8.0   （空闲时段为一半）
#   deepseek-v4-pro   高峰 输入 9.0 / 输出 27.0
# 旧模型名 deepseek-chat 仍可调用，实际由 Flash 模型提供服务并按 Flash 价格计费。
# 这里统一取**高峰价**：宁可高估成本，也不要把账单说少了。
PRICING_CNY_PER_MTOK = {
    "deepseek-chat":    {"input": 2.0, "output": 8.0},
    "deepseek-flash":   {"input": 2.0, "output": 8.0},
    "deepseek-v4-pro":  {"input": 9.0, "output": 27.0},
    # 本地模型没有 API 账单，但有电费。折算口径：
    #   7B Q4 在消费级 GPU 上约 60 tok/s，整机功耗约 250W，电价约 ¥0.6/kWh
    #   → 输出 1M tokens ≈ 16667s ≈ 1.16 kWh ≈ ¥0.7
    #   prefill 比 decode 快一个数量级，故输入单价取输出的 1/8
    # 数量级正确即可：它的作用是让"本地并非零成本"这件事在报表里可见。
    "local":            {"input": 0.1, "output": 0.8},
}

# 无法识别模型名时的兜底价格（按云端最高档估，避免低估）
DEFAULT_CLOUD_PRICING = PRICING_CNY_PER_MTOK["deepseek-v4-pro"]
LOCAL_PRICING = PRICING_CNY_PER_MTOK["local"]

# 基线模型：反事实"全部走付费云端"用哪个模型的价格
DEFAULT_BASELINE_MODEL = "deepseek-chat"

# 命中这些子串即判定为本地自建（走电费口径）。
# 必须在云端关键字之前判断：网关上的 "ollama/deepseek-r1" 虽然名字含 deepseek，
# 但算力在本机，真实成本是电费。
LOCAL_MARKERS = ("ollama/", "ollama:", "qwen", "llama", "mistral", "gemma", "phi-",
                 "local", "localhost", "127.0.0.1")


def _normalize(model: str) -> str:
    return (model or "").strip().lower()


def _last_segment(model: str) -> str:
    """取 "provider/model" 里的 model 部分，便于精确匹配价格表。"""
    return _normalize(model).rsplit("/", 1)[-1]


def is_local_model(model: str) -> bool:
    """模型名是否指向本地自建推理。"""
    name = _normalize(model)
    return bool(name) and any(m in name for m in LOCAL_MARKERS)


def resolve_pricing(model: str, tier: str = "") -> tuple[dict, str, bool]:
    """
    解析某次调用的单价。

    返回 (pricing, 命中键, 是否走了兜底)。
    匹配顺序：本地关键字 → 精确名 → provider 前缀去掉后精确名 → 按 tier 兜底。
    """
    name = _normalize(model)

    if is_local_model(name):
        return LOCAL_PRICING, "local", False

    if name in PRICING_CNY_PER_MTOK:
        return PRICING_CNY_PER_MTOK[name], name, False

    seg = _last_segment(name)
    if seg in PRICING_CNY_PER_MTOK:
        return PRICING_CNY_PER_MTOK[seg], seg, False

    if tier == "local":
        return LOCAL_PRICING, "local", False

    return DEFAULT_CLOUD_PRICING, "fallback", True


def _stage_cost(pricing: dict, input_tokens: int, output_tokens: int) -> float:
    return (
        input_tokens / 1_000_000 * float(pricing["input"])
        + output_tokens / 1_000_000 * float(pricing["output"])
    )


def _iter_stage_usage(token_usage: dict):
    """
    遍历有真实 token 数的阶段。

    token_usage 里混着两类条目：
      - LLM 调用：{prompt_tokens, completion_tokens, total_tokens, model?, tier?}
      - 检索统计：{query_tokens, candidate_tokens, ...}（没有 prompt_tokens）
    只有前者该被计价，后者只是观测数据。用 "prompt_tokens" 是否存在来区分，
    比维护一张"哪些 stage 是 LLM"的名单更不容易漏。
    """
    for stage, payload in (token_usage or {}).items():
        if stage in ("llm_total_tokens", "retrieve"):
            continue
        if not isinstance(payload, dict):
            continue
        if "prompt_tokens" not in payload and "completion_tokens" not in payload:
            continue
        yield stage, payload


def estimate_cost(
    token_usage: dict,
    routing: dict | None = None,
    baseline_model: str = DEFAULT_BASELINE_MODEL,
    pricing: dict | None = None,
) -> dict:
    """
    按 token 用量估算人民币成本，并给出"相对全云端基线"的节省额。

    返回：
        total_cny        本次实际预估成本
        baseline_cny     反事实：全部 token 都按 baseline_model 列表价计费
        saved_cny        baseline - total（不为负，基线低于实际时置 0 并标 reason）
        saved_pct        saved / baseline
        breakdown        每个阶段的明细（stage/tier/model/输入输出/金额/单价来源）
        estimated        是否有阶段用了估算 token（流式响应常常拿不到 usage）
        unpriced         走了兜底价格的模型名（提醒补价格表）
    """
    routing = routing or {}
    table = pricing or PRICING_CNY_PER_MTOK

    breakdown = []
    total = 0.0
    baseline_total = 0.0
    estimated = False
    unpriced: list[str] = []

    for stage, payload in _iter_stage_usage(token_usage):
        pin = int(payload.get("prompt_tokens") or 0)
        pout = int(payload.get("completion_tokens") or 0)
        if pin <= 0 and pout <= 0:
            continue

        model = str(payload.get("model") or "")
        tier = str(payload.get("tier") or "")
        # model 缺失时退回路由快照（"tier:model"），保证仍有计价依据
        if not model:
            snap = str(routing.get(stage) or "")
            if ":" in snap:
                tier = tier or snap.split(":", 1)[0]
                model = snap.split(":", 1)[1]
            else:
                tier = tier or snap

        unit, matched, fell_back = resolve_pricing(model, tier)
        cost = _stage_cost(unit, pin, pout)
        base_unit, _, _ = resolve_pricing(baseline_model, "cloud")
        base_cost = _stage_cost(base_unit, pin, pout)

        if fell_back and model:
            unpriced.append(model)
        if payload.get("usage_estimated"):
            estimated = True

        total += cost
        baseline_total += base_cost
        breakdown.append({
            "stage": stage,
            "tier": tier or ("local" if matched == "local" else "cloud"),
            "model": model or "(未知)",
            "pricing_key": matched,
            "input_tokens": pin,
            "output_tokens": pout,
            "cny": round(cost, 8),
            "baseline_cny": round(base_cost, 8),
            "estimated": bool(payload.get("usage_estimated")),
        })

    saved = baseline_total - total
    note = ""
    if saved < 0:
        # 实际比基线还贵：通常意味着基线模型比实际用的模型便宜（例如实际用了 pro）
        note = f"实际成本高于 {baseline_model} 基线，节省额按 0 计"
        saved = 0.0

    return {
        "currency": "CNY",
        "total_cny": round(total, 8),
        "baseline_cny": round(baseline_total, 8),
        "saved_cny": round(saved, 8),
        "saved_pct": round(saved / baseline_total, 4) if baseline_total > 0 else 0.0,
        "baseline_model": baseline_model,
        "baseline_note": (
            f"基线 = 本次全部 token 按 {baseline_model} 列表价（高峰）计费；"
            "实际成本按各阶段真实使用的模型计价"
        ),
        "breakdown": breakdown,
        "stages_priced": len(breakdown),
        "estimated": estimated,
        "unpriced_models": sorted(set(unpriced)),
        "note": note,
        "pricing_cny_per_mtok": table,
    }


def format_cny(amount: float) -> str:
    """
    金额格式化：小数位随量级变化，保证有效数字不被截断。

    为什么不能统一用 2 位小数：本项目单次评估成本就在 0.002~0.02 元之间，
    统一两位会把 0.015 显示成 "0.01"（33% 的误差），把 0.0021 显示成 "0.00"
    （看起来像免费）。分档处理才是诚实的做法：1 元以下一律给四位小数。
    """
    amount = float(amount or 0)
    if amount <= 0:
        return "0.00"
    if amount < 0.0001:
        return f"{amount:.6f}"
    if amount < 1:
        # 本项目单次成本集中在 0.001~0.05 元，四位小数才能看出差异
        return f"{amount:.4f}"
    return f"{amount:.2f}"


def caption_text(est: dict) -> str:
    """
    生成前端那行说明文字。

    放在后端算，是为了让"成本"和"节省"永远来自同一次计算——
    前端各自格式化一次很容易出现"成本 0.0021、节省 0.015"这种量级错配。
    """
    if not est or est.get("stages_priced", 0) == 0:
        return "本次评估预估成本：— （未采集到 token 用量）"
    total = format_cny(est.get("total_cny"))
    saved = format_cny(est.get("saved_cny"))
    if est.get("saved_cny", 0) <= 0:
        return f"本次评估预估成本：¥{total}"
    return f"本次评估预估成本：¥{total}（已为你节省 ¥{saved}）"


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
