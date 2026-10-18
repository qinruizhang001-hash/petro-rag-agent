"""
指数退避重试（Exponential Backoff）
===================================
并行分支调用云端模型时的限流防护：429 / 5xx / 连接类错误自动重试，
其余错误立刻抛出 —— **不该重试的错误重试三次，只会把一次失败变成三次等待**。

为什么放在 utils/ 而且不 import openai
--------------------------------------
``utils/`` 下的模块要能被 CI 的 ``unit-core`` 直接 import（那个 job 一个三方包
都不装）。所以这里不 ``from openai import RateLimitError``，而是按**异常类名 +
status_code 属性**判断是否瞬时错误：

    openai.APIConnectionError / APITimeoutError / RateLimitError /
    InternalServerError / APIStatusError(status_code=429|5xx)
    httpx.ConnectError / ReadTimeout / RemoteProtocolError

这样既不需要依赖，也不会因为 openai SDK 改版把重试逻辑悄悄关掉
（类名比 import 路径稳定得多）。

什么**不**重试
--------------
- 4xx（401 鉴权失败、404 模型名不对、400 参数非法）：重试只会重复同一个错误。
  这几种恰恰是最常见的配置问题，快速失败才能让人看到真实原因。
- 我们自己抛的逻辑错误（ValueError 等）。

抖动（jitter）
--------------
没有抖动的指数退避在并行场景下有个隐蔽问题：两条分支同时撞上 429，
同时按 1s/2s/4s 重试，会一直同步撞在一起。所以默认加 ±25% 的抖动，
把两条分支的重试时间错开。
"""
from __future__ import annotations

import random
import time
from typing import Any, Callable, TypeVar

from utils.logging_setup import get_logger

log = get_logger(__name__)

T = TypeVar("T")

# 按类名判断瞬时错误（不 import openai / httpx，见模块文档）
TRANSIENT_EXC_NAMES = frozenset({
    "APIConnectionError",
    "APITimeoutError",
    "RateLimitError",
    "InternalServerError",
    "ServiceUnavailableError",
    "APIStatusError",          # 需再看 status_code，见 _is_transient
    "ConnectError",
    "ConnectTimeout",
    "ReadTimeout",
    "WriteTimeout",
    "PoolTimeout",
    "RemoteProtocolError",
    "ReadError",
    "TimeoutException",
    "ConnectionError",
    "Timeout",
})

# 这些类名即使命中也要看 status_code：4xx 不该重试
_STATUS_GATED = frozenset({"APIStatusError"})


def status_of(exc: BaseException) -> int | None:
    """尽量从异常里挖出 HTTP 状态码；挖不到返回 None。"""
    for attr in ("status_code", "http_status", "code"):
        val = getattr(exc, attr, None)
        if isinstance(val, int):
            return val
    resp = getattr(exc, "response", None)
    val = getattr(resp, "status_code", None)
    return val if isinstance(val, int) else None


def is_transient(exc: BaseException) -> bool:
    """判断一个异常是否值得重试。

    规则（按顺序）：
      1. 有状态码且是 4xx → 不重试（配置问题，重试无意义）
      2. 状态码 429 / 5xx → 重试
      3. 类名在 TRANSIENT_EXC_NAMES 里 → 重试
      4. 其余 → 不重试
    """
    status = status_of(exc)
    if status is not None:
        if status == 429 or status >= 500:
            return True
        if 400 <= status < 500:
            return False
    return type(exc).__name__ in TRANSIENT_EXC_NAMES


def backoff_delay(
    attempt: int, base: float = 1.0, factor: float = 2.0,
    max_delay: float = 8.0, jitter: float = 0.25,
) -> float:
    """第 ``attempt`` 次重试（从 1 开始）应等待的秒数。

    ``attempt=1 → base``，之后每次乘 ``factor``，封顶 ``max_delay``，
    再叠加 ±``jitter`` 比例的随机抖动。
    """
    if attempt < 1:
        attempt = 1
    raw = base * (factor ** (attempt - 1))
    raw = min(raw, max_delay)
    if jitter <= 0:
        return raw
    return max(0.0, raw * (1.0 + random.uniform(-jitter, jitter)))


def call_with_retry(
    fn: Callable[..., T],
    *args: Any,
    attempts: int = 3,
    base_delay: float = 1.0,
    factor: float = 2.0,
    max_delay: float = 8.0,
    jitter: float = 0.25,
    label: str = "",
    sleep: Callable[[float], None] = time.sleep,
    **kwargs: Any,
) -> T:
    """调用 ``fn``，对瞬时错误做指数退避重试。

    :param attempts: 总尝试次数（含第一次）。``3`` 表示最多重试 2 次。
    :param sleep: 注入点，单测里替换成"只记录不真睡"。
    :raises: 最后一次失败的原异常（不包装，调用方看到的仍是真实错误）。
    """
    if attempts < 1:
        raise ValueError("attempts 必须 >= 1")

    tag = label or getattr(fn, "__name__", "call")
    last_exc: BaseException | None = None

    for attempt in range(1, attempts + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 — 是否重试由 is_transient 决定
            last_exc = exc
            if attempt >= attempts or not is_transient(exc):
                if attempt > 1:
                    log.warning(
                        f"  [Retry] {tag} 第 {attempt} 次尝试失败且不再重试："
                        f"{type(exc).__name__}: {exc}"
                    )
                raise
            delay = backoff_delay(
                attempt, base=base_delay, factor=factor,
                max_delay=max_delay, jitter=jitter,
            )
            log.warning(
                f"  [Retry] {tag} 第 {attempt}/{attempts} 次失败"
                f"（{type(exc).__name__}: {exc}），{delay:.1f}s 后重试"
            )
            sleep(delay)

    # 逻辑上不可达（循环内要么 return 要么 raise），仅为类型检查
    assert last_exc is not None
    raise last_exc
