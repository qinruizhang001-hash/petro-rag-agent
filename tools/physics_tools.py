"""
物理计算工具（供 Function Calling 调用）
========================================
与 ``knowledge_tools``（联网检索）并列：这里是"算出来的事实"，
那里是"查回来的事实"，两者都不该由语言模型口算/编造。

``co2_density_value`` 用的是 Peng-Robinson 状态方程（实现见
``utils/co2_properties.py``），临界参数取 Span & Wagner 1996。它演示的是
**工具调用契约**（数值必须来自代码而非模型）：任何进入报告的数字都要能被
追溯到一个确定性函数，而不是模型记忆里的"大概 600 多"。

从 v1.6 起，对外暴露的不再是单点值，而是**分布**
-----------------------------------------------
地质参数有测量误差，单点值会让报告显得比事实更确定。所以这里对输入做
蒙特卡洛传播（100 次，输入 ±5% 扰动），返回：

    均值 + 90% 置信区间（p5 ~ p95）+ 模拟次数 + 扰动幅度

区间是**百分位区间**而不是 ``mean ± 1.645σ``：状态方程对 P 是线性的，
但注入风险判定（``verdict_of``）不是，边界附近的输出分布明显偏斜，
用对称区间会低估尾部概率。

为什么模拟次数是 100 而不是 10000
---------------------------------
100 次在毫秒级跑完，且 5% 分位对应第 5 个样本，仍然有统计意义；
10000 次能把分位数误差压到 1% 以内，但每次工具调用多花 100 倍时间，
而报告里只会写两位小数 —— 精度买过头了。需要更高精度时传 ``n=10000`` 即可。
"""
from __future__ import annotations

import pressure_model as pm
from core.uq import (
    DEFAULT_N,
    DEFAULT_REL_SIGMA,
    draw_relative,
    format_interval,
    make_rng,
    monte_carlo,
    summarize,
)
from utils.logging_setup import get_logger

log = get_logger(__name__)

DENSITY_UNIT = "kg/m³"
PRESSURE_UNIT = "bar"

_VERDICT_LABELS = {
    "ok": "注入可行",
    "risky": "压裂风险",
    "under": "压力不足（注不进）",
}


# ══════════════════════════════════════════════════════════
# CO2 密度
# ══════════════════════════════════════════════════════════
def co2_density_value(pressure_bar: float, temperature_celsius: float) -> float:
    """
    CO₂ 密度（kg/m³），Peng-Robinson 状态方程。

    单独拆出来是为了让蒙特卡洛只传播**一个**不确定源：
    模型函数本身必须是纯函数、单参数，否则 ``monte_carlo`` 无法包装。

    历史上这里是 ``ρ = 1.9·P/T``，在 250 bar / 80℃ 给 1.35 kg/m³。
    真实超临界 CO₂ 约 650 kg/m³ —— 差 460 倍，量纲就不对。
    这个数会进浮力（Δρ·g·H）、流度比、重力分异，错 460 倍不是"精度不够"，
    是把"封不住"判成"封得住"。实现见 ``utils/co2_properties.py``。
    """
    from utils.co2_properties import co2_density_kg_m3

    return co2_density_kg_m3(pressure_bar, temperature_celsius)


def co2_density_uq(
    pressure_bar: float,
    temperature_celsius: float,
    rel_sigma: float = DEFAULT_REL_SIGMA,
    n: int = DEFAULT_N,
    seed: int | None = None,
    keep_samples: bool = False,
) -> dict:
    """
    密度 + 不确定性。扰动的是**压力输入**（现场最容易波动、也最常被问到的量），
    温度按标称值处理。

    返回 ``summarize`` 结构（mean / p5 / p95 / cv / text …）。
    """
    result = monte_carlo(
        lambda p: co2_density_value(p, temperature_celsius),
        nominal=float(pressure_bar),
        rel_sigma=rel_sigma,
        n=n,
        seed=seed,
        lower_bound=1e-9,
        keep_samples=keep_samples,
    )
    summary = summarize(result, unit=DENSITY_UNIT, keep_samples=keep_samples)
    summary["pressure_bar"] = float(pressure_bar)
    summary["temperature_celsius"] = float(temperature_celsius)
    return summary


def calculate_co2_density(
    pressure_bar,
    temperature_celsius,
    uncertainty: float = DEFAULT_REL_SIGMA,
    n: int = DEFAULT_N,
    seed: int | None = None,
) -> str:
    """
    计算给定压力、温度下 CO2 密度，并给出 90% 置信区间。

    返回字符串（MCP 工具的既有契约，第一个 token 仍是均值数字，
    老调用方 ``float(text.split()[0])`` 不会坏）。

    示例输出::

        1.35 kg/m³（90% 置信区间 1.29–1.41 kg/m³，100 次蒙特卡洛模拟，输入 ±5% 扰动）
    """
    log.info(
        f"  [Tool 调用] CO2 密度：压力 {pressure_bar} bar（±{uncertainty:.0%} 扰动），"
        f"温度 {temperature_celsius}℃，{n} 次蒙特卡洛..."
    )
    summary = co2_density_uq(
        pressure_bar, temperature_celsius,
        rel_sigma=uncertainty, n=n, seed=seed,
    )
    return summary["text"]


# ══════════════════════════════════════════════════════════
# 地层压力
# ══════════════════════════════════════════════════════════
def hydrostatic_pressure_uq(
    depth_m: float,
    rel_sigma: float = DEFAULT_REL_SIGMA,
    n: int = DEFAULT_N,
    seed: int | None = None,
) -> dict:
    """
    静水压力 + 不确定性（扰动深度，因为深度是这里唯一的输入量）。

    为什么不是扰动水密度：1030 kg/m³ 是给定岩性下的确定常数，
    真正带误差的是"储层到底在 2500 m 还是 2600 m"。
    """
    result = monte_carlo(
        pm.hydrostatic_bar,
        nominal=float(depth_m),
        rel_sigma=rel_sigma,
        n=n,
        seed=seed,
        lower_bound=0.0,
    )
    summary = summarize(result, unit=PRESSURE_UNIT)
    summary["depth_m"] = float(depth_m)
    return summary


def fracture_pressure_uq(
    depth_m: float,
    rel_sigma: float = DEFAULT_REL_SIGMA,
    n: int = DEFAULT_N,
    seed: int | None = None,
) -> dict:
    """地层破裂压力 + 不确定性。"""
    result = monte_carlo(
        pm.fracture_bar,
        nominal=float(depth_m),
        rel_sigma=rel_sigma,
        n=n,
        seed=seed,
        lower_bound=0.0,
    )
    summary = summarize(result, unit=PRESSURE_UNIT)
    summary["depth_m"] = float(depth_m)
    return summary


def injection_risk_uq(
    depth_m: float,
    injection_pressure_bar: float,
    rel_sigma: float = DEFAULT_REL_SIGMA,
    n: int = DEFAULT_N,
    seed: int | None = None,
) -> dict:
    """
    注入风险的概率分布：同时扰动**深度**与**注入压力**，统计三种结局的频率。

    为什么两个都扰动：这两个量恰好一个来自地质解释（深度）、一个来自地面
    设备（压力），误差来源独立。只扰动压力会得到一个偏窄的风险分布，
    让"压裂概率"看起来比实际低。

    返回::

        {"n", "rel_sigma", "probabilities": {"ok": .76, "risky": .12, "under": .12},
         "counts", "text", "verdict", ...}

    其中 ``verdict`` 是标称值下的确定性判定（与 ``pressure_model.verdict_of``
    一致），概率则说明"这个判定有多稳"。
    """
    n = max(2, int(n))
    rng = make_rng(seed)

    counts: dict[str, int] = {}
    for _ in range(n):
        d = float(depth_m) * (1.0 + draw_relative(rng, rel_sigma))
        p = float(injection_pressure_bar) * (1.0 + draw_relative(rng, rel_sigma))
        label = pm.verdict_of(d, p)
        counts[label] = counts.get(label, 0) + 1

    probabilities = {k: v / n for k, v in counts.items()}
    nominal_verdict = pm.verdict_of(depth_m, injection_pressure_bar)

    detail = "、".join(
        f"{_VERDICT_LABELS.get(k, k)} {v * 100:.0f}%"
        for k, v in sorted(probabilities.items(), key=lambda kv: -kv[1])
    )
    text = (
        f"{detail}（标称判定：{_VERDICT_LABELS.get(nominal_verdict, nominal_verdict)}；"
        f"{n} 次蒙特卡洛，深度与压力各 ±{rel_sigma * 100:.0f}% 扰动）"
    )

    return {
        "n": n,
        "rel_sigma": float(rel_sigma),
        "dist": "normal",
        "counts": counts,
        "probabilities": probabilities,
        "verdict": nominal_verdict,
        "verdict_label": _VERDICT_LABELS.get(nominal_verdict, nominal_verdict),
        "depth_m": float(depth_m),
        "injection_pressure_bar": float(injection_pressure_bar),
        "unit": "",
        "text": text,
    }


# ══════════════════════════════════════════════════════════
# 断层活化 / 盖层封闭性（深层物理）
# ══════════════════════════════════════════════════════════
# 这两个工具与上面的密度/压力工具是**并列**关系：上面回答"注入压力对不对"，
# 这里回答"会不会滑、会不会漏"。后者的触发条件比前者更早 —— 断层在
# 1.5× 静水以下就可能滑动，而此时注入压力离破裂压力还有很大余量，
# 只看破裂压力会漏掉整个风险类别。
def fault_stability(
    depth_m: float,
    injection_pressure_bar: float,
    dip_deg: float = 60.0,
    azimuth_deg: float = 90.0,
) -> dict:
    """
    断层活化判定（Mohr-Coulomb + Terzaghi 有效应力）。

    ``dip_deg`` 默认 60°：最优产状在 45°+φ/2（μ=0.6 → 60.5°），不给倾角时
    取最不利产状比取"某个平均值"更保守，且这一点写进了返回的 ``assumptions``。

    返回结果含 ``text``（一句话）、``verdict``、``slip_ratio``、``pp_crit_bar``、
    ``margin_bar``、``assumptions``、``limitations``。**假设必须跟着数字一起
    返回**：这是用默认应力剖面算的，不是场地实测。
    """
    from utils.geomechanics import fault_reactivation

    result = fault_reactivation(
        depth_m=depth_m,
        injection_pressure_bar=injection_pressure_bar,
        dip_deg=dip_deg,
        azimuth_deg=azimuth_deg,
    )
    head = {
        "stable": "断层未活化",
        "near_critical": "断层接近活化",
        "reactivated": "断层已活化",
        "tensile_opening": "断层张性开启",
    }.get(result["verdict"], result["verdict"])
    result["text"] = (
        f"{head}：倾角 {result['plane']['dip_deg']:.0f}° 断层面滑移比 "
        f"{result['slip_ratio']}（≥1 即滑动），临界孔隙压力 "
        f"{result['pp_crit_bar']} bar，余量 {result['margin_bar']} bar"
        f"（按假设应力剖面计算，非实测）"
    )
    return result


def caprock_check(
    depth_m: float,
    injection_pressure_bar: float,
    temperature_celsius: float = 60.0,
    co2_column_height_m: float | None = None,
    injection_rate_m3_per_day: float | None = None,
) -> dict:
    """
    盖层封闭性判定：毛细管突破（Laplace/Washburn）与力学失效（σhmin + T0）
    两条独立路径，任一条失效即 ``blocking``。

    不给柱高也不给注入速率时返回 ``verdict="no_data"``，只给突破压力与
    可封柱高 —— "没算"和"算出来不合格"必须能区分。
    """
    from utils.caprock import caprock_integrity

    result = caprock_integrity(
        depth_m=depth_m,
        injection_pressure_bar=injection_pressure_bar,
        temperature_celsius=temperature_celsius,
        co2_column_height_m=co2_column_height_m,
        injection_rate_m3_per_day=injection_rate_m3_per_day,
    )
    result["text"] = result.get("summary") or (
        f"盖层突破压力 {result['entry_pressure_bar']:.0f} bar，"
        f"可封 CO₂ 柱高 {result['max_column_height_m']:.0f} m"
    )
    return result


# ══════════════════════════════════════════════════════════
# 兼容旧接口
# ══════════════════════════════════════════════════════════
def pressure_interval_text(depth_m: float, injection_pressure_bar: float,
                           rel_sigma: float = DEFAULT_REL_SIGMA,
                           n: int = DEFAULT_N, seed: int | None = None) -> str:
    """一行文本：静水压力区间 + 破裂压力区间（报告兜底文案用）。"""
    hydro = hydrostatic_pressure_uq(depth_m, rel_sigma=rel_sigma, n=n, seed=seed)
    frac = fracture_pressure_uq(depth_m, rel_sigma=rel_sigma, n=n, seed=seed)
    return (
        f"静水压力 {format_interval(hydro, PRESSURE_UNIT)}；"
        f"破裂压力 {format_interval(frac, PRESSURE_UNIT)}"
    )
