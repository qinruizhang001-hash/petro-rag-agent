"""
模型路由解析（纯函数，零重依赖）
================================
把"某角色该用哪个模型"的决策逻辑从图定义里剥离出来。

为什么单独成模块：
- 这段逻辑决定了"哪个角色用哪个厂商的模型"，出错的表现是**静默降级**
  （本来该走云端却退到本地 7B），从日志里很难察觉，必须被单元测试覆盖。
- 纯函数、无 IO、无第三方依赖 → CI 里可直接跑，不需要 Ollama 与模型文件。

路由配置语法（每个角色一条）：

    "local"                        → 本地默认模型
    "cloud"                        → 云端默认模型
    "cloud:deepseek/deepseek-chat" → 云端指定模型（网关常用 provider/model）
    "local:qwen2.5:14b"            → 本地指定模型
"""
from __future__ import annotations

VALID_TIERS = ("local", "cloud")


def parse_route(spec: str) -> tuple[str, str]:
    """
    解析单条路由配置，返回 (tier, model_override)。

    model_override 为空字符串表示沿用该档位的默认模型。
    未知档位不在这里报错，交给 resolve_route 统一降级 —— 因为配置文件写错
    不应该让整个进程启动失败，但必须在解析结果里留下可追踪的原因。
    """
    spec = (spec or "").strip()
    if not spec:
        return "local", ""
    if ":" in spec:
        tier, model = spec.split(":", 1)
        return tier.strip().lower(), model.strip()
    return spec.lower(), ""


def build_routing(defaults: dict, env: dict | None = None) -> dict:
    """
    合并默认路由与 ROUTE_<STAGE> 环境变量覆盖。

    `env` 显式传入是为了可测试（不依赖真实 os.environ）。
    空字符串视为"未覆盖"，避免 .env 里 `ROUTE_REPORT=` 这种留空写法
    把默认值悄悄清成空。
    """
    env = env or {}
    routing = {}
    for stage, default in defaults.items():
        override = (env.get(f"ROUTE_{stage.upper()}") or "").strip()
        routing[stage] = override or default
    return routing


def resolve_route(
    routing: dict, stage: str, cloud_available: bool
) -> tuple[str, str, str]:
    """
    把某角色解析成 (tier, model_override, 降级原因)。

    降级规则：
      - 云端不可用（未配 key）→ 退到本地，reason 说明原因
      - 档位写错               → 退到本地，reason 指出非法值
    两种降级都**必须带原因**，否则"以为在用云端、其实在用 7B"很难排查。
    """
    tier, model = parse_route(routing.get(stage, "local"))

    if tier not in VALID_TIERS:
        return "local", model, f"未知档位 {tier!r}（只支持 local / cloud）"

    if tier == "cloud" and not cloud_available:
        return "local", model, "CLOUD_API_KEY 未配置"

    return tier, model, ""


# ══════════════════════════════════════════════════════════
# 动态路由：按「本次请求的复杂度」决定档位
# ══════════════════════════════════════════════════════════
#
# 静态路由表回答的是"这个角色通常该用什么模型"；
# 动态路由回答的是"这一次请求值得用什么模型"。
#
# 判定依据只有两个，都是可解释、可复算的：
#   1. 用户输入长度 —— 短输入通常意味着参数齐全的简单查询；
#   2. 知识库是否真的命中了 —— 召回了但相关度接近 0，等于没召回，
#      这时为了"有 RAG"就升级到云端，纯粹是浪费钱。
#
# 关于第 2 点为什么需要一个相关度阈值：本项目实测 bge-reranker-v2-m3 对
# 无关提问（"今天天气怎么样"）的 Top-1 分数是 0.0001，而真实 CCUS 提问在
# 0.27~0.74。0.05 这条线把两类彻底分开，不存在灰色地带。

COMPLEXITY_CHAR_THRESHOLD = 50
RAG_MIN_SCORE = 0.05

LOCAL_MODE_LABEL = "[Cost-Saving: Local Mode]"
CLOUD_MODE_LABEL = "[Premium: Cloud Mode]"

# 受动态路由管辖的角色。
# report_plan 故意不在此列：它只输出几十个 token 的结构化 JSON，
# 无论请求多复杂都该留在本地（这是 Phase 18 就定下的成本决策）。
DYNAMIC_STAGES = ("extract", "report")


def classify_complexity(
    user_input: str,
    rag_triggered: bool = False,
    top_score: float | None = None,
    char_threshold: int = COMPLEXITY_CHAR_THRESHOLD,
    rag_min_score: float = RAG_MIN_SCORE,
) -> dict:
    """
    判定本次请求该走本地还是云端。

    返回：
        mode          "local" | "cloud"
        label         日志标签（[Cost-Saving: Local Mode] / [Premium: Cloud Mode]）
        chars         输入字符数
        rag_triggered 知识库是否真的命中
        top_score     rerank 最高分（未知时为 None）
        reasons       逐条判定依据（便于日志解释"为什么这么选"）
        reason        合并后的一句话说明

    注意：调用方在 extract 阶段还不知道 RAG 结果，此时 rag_triggered=False
    表示"尚未检索"，而不是"确定没命中"。真正的升级发生在 retrieve 之后，
    由 escalate_complexity 负责——它只允许 local → cloud 单向升级。
    """
    text = user_input or ""
    chars = len(text)
    reasons: list[str] = []

    if chars > char_threshold:
        reasons.append(f"输入 {chars} 字 > {char_threshold} 字阈值")
        long_enough = True
    else:
        reasons.append(f"输入 {chars} 字 ≤ {char_threshold} 字阈值")
        long_enough = False

    # 有 rerank 分数时以分数为准，不信任调用方传的布尔值 ——
    # 两者不一致时（例如传了 rag_triggered=False 但分数 0.74），
    # 若按布尔值走，日志会打出"知识库未命中（相关度 0.7400 < 0.05）"
    # 这种自相矛盾的句子，排查时极具误导性。
    if top_score is not None:
        hit = float(top_score) >= rag_min_score
    else:
        hit = bool(rag_triggered)

    if hit:
        if top_score is not None:
            reasons.append(f"知识库命中（相关度 {top_score:.4f} ≥ {rag_min_score}）")
        else:
            reasons.append("知识库命中")
    else:
        if top_score is not None:
            reasons.append(f"知识库未命中（相关度 {top_score:.4f} < {rag_min_score}）")
        else:
            reasons.append("知识库未命中")

    mode = "cloud" if (long_enough or hit) else "local"
    return {
        "mode": mode,
        "label": CLOUD_MODE_LABEL if mode == "cloud" else LOCAL_MODE_LABEL,
        "chars": chars,
        "char_threshold": char_threshold,
        "rag_triggered": hit,
        "top_score": top_score,
        "rag_min_score": rag_min_score,
        "reasons": reasons,
        "reason": "；".join(reasons) + f" → {'云端' if mode == 'cloud' else '本地'}",
    }


def escalate_complexity(prev: dict | None, new: dict) -> dict:
    """
    合并两次判定，只允许 local → cloud 的单向升级。

    为什么不允许降级：extract 阶段已经按"长输入 → 云端"付过费了，
    到了 retrieve 阶段发现"RAG 没命中"再降回本地，只会让同一份报告
    一半由云端写、一半由本地写，风格与质量都不一致。
    路由一旦升级就保持，可解释性优先于那点差价。
    """
    if not prev:
        return dict(new)
    if prev.get("mode") == "cloud" and new.get("mode") == "local":
        merged = dict(prev)
        merged["reason"] = (
            prev.get("reason", "") + f"；retrieve 阶段：{new.get('reason', '')}"
            "（已升级到云端，不再降级）"
        )
        return merged
    if prev.get("mode") == "local" and new.get("mode") == "cloud":
        merged = dict(new)
        merged["reason"] = (
            f"extract 阶段：{prev.get('reason', '')}；"
            f"retrieve 阶段升级：{new.get('reason', '')}"
        )
        return merged
    return dict(new)
