"""
决策阈值（把 UQ 的概率接到"要不要告警"上）
==========================================
"压裂风险 12%" 本身不是结论。结论是"12% 是否越过了我们愿意接受的水平"。
前者是 UQ 的输出，后者是决策 —— 而决策需要一条**阈值规则**。

本模块存在的唯一理由：让这条规则**显式、可配置、可追溯**，而不是散落在
某个 ``if p > 0.05`` 里。散落的那一行有两个毛病：读者不知道 0.05 从哪来，
运维改不了它，而改它的人不会在报告里留下任何痕迹。

阈值是政策，不是物理常数
------------------------
这一点必须写在代码里，也必须写进报告。破裂压力是物理量（有公式、有出处）；
"多大风险才值得告警"是**风险接受准则**，属于项目/监管决策。ISO 27914:2017
（CO₂ 地质封存）与 DNV-RP-J203 都要求运营方在封存许可阶段定义量化的
风险接受准则，但**都不规定具体数值** —— 因此本模块给不出"标准规定的 5%"，
只能给出"本项目的默认值 + 它是怎么来的 + 怎么改"。

默认值与它们的来路
------------------
``WARN = 5%``
    技术理由，不是风险学理由：本项目默认模拟 100 次，5% 约合 5 个样本，
    这是二项比例还能给出有意义相对精度的下限（k=5、n=100 的 95% 区间是
    2.5%–10.8%；再往下区间就贴到 0，任何比较都变成噪声比较）。
    换句话说，把告警线设在 5% 以下，在 N=100 下**测不出来**。

``ALERT = 20%``
    纯政策选择，无标准依据。选它是因为"五次里有一次会压裂"在工程上
    已经不该按常规作业处理；但换一个项目完全可以设成 10% 或 30%。

两个数都可以用环境变量覆盖：``UQ_WARN_P_LEAK`` / ``UQ_ALERT_P_LEAK``。
覆盖之后 ``decide()`` 返回的 ``thresholds.source`` 会变成 ``"env"``，
报告里引用的就不再是默认值 —— 这条不能省，否则报告说的阈值和实际用的
阈值可能不是同一个数。

为什么概率必须带区间
--------------------
N=100 次模拟里 5 次 risky，P = 5%，其 95% Wilson 区间是 2.5%–10.8%。
如果告警线恰好是 5%，那么"触发了"这件事本身就在统计噪声里。所以
``decide()`` 除了 ``level`` 还给 ``resolved``：区间整体落在阈值同一侧时
才是可判定的；跨越阈值时明确写"不可区分，需增大模拟次数"，而不是
假装这个 5% 是精确的。
"""
from __future__ import annotations

import math
import os

from core.uq import wilson_interval

# 默认阈值（见模块文档：一个是分辨率下限，一个是政策选择）
DEFAULT_WARN_P_LEAK = 0.05
DEFAULT_ALERT_P_LEAK = 0.20

ENV_WARN = "UQ_WARN_P_LEAK"
ENV_ALERT = "UQ_ALERT_P_LEAK"

# 决策等级
LEVEL_OK = "ok"
LEVEL_WARN = "warn"
LEVEL_ALERT = "alert"

_LEVEL_CN = {LEVEL_OK: "正常", LEVEL_WARN: "提示", LEVEL_ALERT: "告警"}

# 阈值来路。写进返回值的 thresholds.provenance，报告引用阈值时必须带上。
PROVENANCE_DEFAULT = (
    "项目默认值（非标准规定）：warn=5% 是 N=100 次模拟下的分辨率下限，"
    "alert=20% 是政策选择；ISO 27914:2017 与 DNV-RP-J203 要求定义量化的"
    "风险接受准则但不规定数值"
)
PROVENANCE_ENV = "由环境变量覆盖（UQ_WARN_P_LEAK / UQ_ALERT_P_LEAK），非标准值"


def _read_float(env, name: str, default: float) -> tuple[float, bool]:
    """读阈值；非法值不静默忽略 —— 配置错了必须当场知道。"""
    raw = str(env.get(name) or "").strip()
    if not raw:
        return default, False
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"环境变量 {name}={raw!r} 不是数字") from None
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"环境变量 {name}={value} 必须在 [0, 1] 内（概率口径）")
    return value, True


def load_thresholds(environ=None) -> dict:
    """
    读阈值（默认值 + 环境变量覆盖）。返回 dict 而不是类实例：它要进 state、
    进 JSON、进前端，dict 是这条链路上唯一不需要序列化适配的形状。

    ``environ`` 只供测试注入，**不去改 os.environ**：临时改写进程环境变量
    在多线程的 uvicorn 里会让别的请求读到一半的状态，而阈值恰好是
    "读错一次就下一个错误结论"的东西。

    校验是**硬失败**：``warn >= alert`` 会抛 ValueError。如果只是让 warn 覆盖
    alert，告警等级会静默消失（永远进不了 alert），而这种错没人会在报告里
    发现 —— 报告只会少一段。
    """
    env = environ if environ is not None else os.environ
    warn, warn_env = _read_float(env, ENV_WARN, DEFAULT_WARN_P_LEAK)
    alert, alert_env = _read_float(env, ENV_ALERT, DEFAULT_ALERT_P_LEAK)

    if warn >= alert:
        raise ValueError(
            f"告警阈值必须大于提示阈值：warn={warn}、alert={alert}。"
            "顺序反了会让 alert 等级永远不可达，而报告只会少一段、不会报错"
        )

    return {
        "warn": warn,
        "alert": alert,
        "source": "env" if (warn_env or alert_env) else "default",
        "provenance": PROVENANCE_ENV if (warn_env or alert_env) else PROVENANCE_DEFAULT,
        "env_vars": {"warn": ENV_WARN, "alert": ENV_ALERT},
    }


def samples_needed(p: float, half_width: float = 0.01,
                   z: float = 1.959963984540054) -> int:
    """
    把比例 p 估到 ±half_width 需要多少次模拟（正态近似 ``n = z²p(1−p)/h²``）。

    用于回答"这个 12% 可不可信、要不要多跑几万次"。p 接近 0 或 1 时
    正态近似偏乐观，所以返回的是**下限**，文案里也这么写。
    """
    p = min(max(float(p), 0.0), 1.0)
    h = max(abs(float(half_width)), 1e-6)
    if p <= 0.0 or p >= 1.0:
        return 0
    return int(math.ceil(z * z * p * (1.0 - p) / (h * h)))


def decide(probabilities: dict, n: int, thresholds: dict | None = None,
           metric: str = "risky", label: str = "压裂风险") -> dict:
    """
    按阈值判定告警等级。

    参数
    ----
    probabilities : ``monte_carlo_outcomes`` 的 ``probabilities``（类别 → 概率）
    n             : 模拟次数（算二项区间要用）
    thresholds    : ``load_thresholds()`` 的结果；None 时现读
    metric        : 用作"泄漏/压裂"的类别名，默认 ``risky``
    label         : 该类别的人话名字，进文案

    返回
    ----
    ``{"level", "triggered", "resolved", "p", "p_ci", "rule", "thresholds",
       "notes", "text"}``

    ``resolved`` 的语义要说清楚：它是"**概率与阈值的相对位置在 95% 区间下
    是确定的**"，不是"结论确定"。``resolved=False`` 时 ``level`` 仍然给出
    （按点估计），但文案会明确写"不可区分"，调用方不该据此单独下结论。
    """
    th = thresholds if thresholds is not None else load_thresholds()
    probs = {str(k): float(v) for k, v in (probabilities or {}).items()}
    p = probs.get(metric, 0.0)
    n = max(0, int(n or 0))
    hits = int(round(p * n))

    lo, hi = wilson_interval(hits, n) if n > 0 else (0.0, 1.0)

    if p >= th["alert"]:
        level = LEVEL_ALERT
    elif p >= th["warn"]:
        level = LEVEL_WARN
    else:
        level = LEVEL_OK

    # 区间是否整体落在阈值的同一侧
    if level == LEVEL_OK:
        resolved = hi < th["warn"]
    elif level == LEVEL_ALERT:
        resolved = lo >= th["alert"]
    else:      # warn：要同时高于 warn 且低于 alert 才算判定清楚
        resolved = lo >= th["warn"] and hi < th["alert"]

    rule = (
        f"P({label}) ≥ {th['alert'] * 100:.0f}% → 告警；"
        f"≥ {th['warn'] * 100:.0f}% → 提示"
    )

    notes: list[str] = []
    if n <= 0:
        # n=0 时区间是 [0,1]，等级按 p 给出但完全不可判定。不写这句的话，
        # 文案会给出一个没有任何限定条件的"正常"，比不给结论更糟。
        notes.append("没有模拟次数，概率的置信区间无法计算，该判定不可用于决策")
    if n > 0 and not resolved:
        notes.append(
            f"P({label})={p * 100:.1f}% 的 95% 区间是 {lo * 100:.1f}%–{hi * 100:.1f}%，"
            f"跨越了判定边界，当前模拟次数下该等级与相邻等级不可区分"
        )
    if n > 0 and p > 0.0:
        need = samples_needed(p, 0.01)
        if need > n:
            notes.append(f"要把该概率估到 ±1 个百分点，至少需要约 {need} 次模拟（当前 {n} 次）")
    if th.get("source") == "env":
        notes.append("阈值由环境变量覆盖，报告引用时必须写明实际生效值而不是默认值")

    head = _LEVEL_CN.get(level, level)
    ci_part = (
        f"（95% 区间 {lo * 100:.1f}%–{hi * 100:.1f}%，{n} 次模拟）" if n > 0 else ""
    )
    if level == LEVEL_OK:
        tail = f"低于提示阈值 {th['warn'] * 100:.0f}%"
    elif level == LEVEL_WARN:
        tail = f"达到提示阈值 {th['warn'] * 100:.0f}%，未达告警阈值 {th['alert'] * 100:.0f}%"
    else:
        tail = f"达到告警阈值 {th['alert'] * 100:.0f}%"
    resolution = "" if resolved or n <= 0 else "；但该判定在当前模拟次数下不可区分"

    return {
        "level": level,
        "level_label": head,
        "triggered": level != LEVEL_OK,
        "resolved": bool(resolved),
        "metric": metric,
        "label": label,
        "p": p,
        "p_ci": [lo, hi],
        "hits": hits,
        "n": n,
        "rule": rule,
        "thresholds": th,
        "notes": notes,
        "text": f"{head}：P({label})={p * 100:.1f}%{ci_part} {tail}{resolution}。{rule}",
    }
