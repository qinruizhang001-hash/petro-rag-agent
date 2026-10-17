"""
模型路由解析（纯函数，零重依赖）
================================
把"某角色该用哪个模型"的决策逻辑从 multi_agent_ccus.py 剥离出来。

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
