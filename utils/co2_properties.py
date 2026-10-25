"""
CO₂ 与地层水的物性（纯标准库）
==============================
这个模块存在的原因是一个具体错误：项目原先用 ``ρ = 1.9·P/T`` 当 CO₂ 状态方程，
在 250 bar / 80℃ 给出 **1.35 kg/m³**。真实超临界 CO₂ 在该条件下约 **650 kg/m³**
—— 差 460 倍。这不是"简化模型的精度不够"，是量纲就不对（1.9 这个系数无法由
任何单位制推出）。后果不是数字难看，而是**所有依赖密度差的结论都错了**：
盖层能封住多高的 CO₂ 柱（浮力 Δρ·g·H）、CO₂ 与水的流度比、重力分异速度，
全部与 Δρ 成正比。

所以换成 Peng-Robinson 状态方程（Peng & Robinson, 1976）：

    P = RT/(V − b) − a·α(T) / (V² + 2bV − b²)
    a = 0.45724·R²·Tc²/Pc,  b = 0.07780·R·Tc/Pc
    κ = 0.37464 + 1.54226ω − 0.26992ω²,  α = [1 + κ(1 − √Tr)]²

PR 是 CO₂ 地质封存建模的通用选择（商业储层模拟器的默认 CO₂ 状态方程之一）。
它把临界压缩因子固定在 0.307（真实 CO₂ 为 0.274），所以**临界点附近误差可达
10–15%**；远离临界点（本项目的典型工况 200–400 bar / 60–90℃）通常 < 5%。

## 边界（不藏着）

- 只适用于 **T ≥ 31℃**（CO₂ 临界温度 30.98℃）的单相致密/超临界区。
  低于临界温度时 PR 有三个根，本实现返回气相根并标记 ``phase="gas"``，
  此时数值只作量级参考。CCUS 储层温度不会低于这个值。
- 不做混合气体（杂质 O₂/H₂S/CH₄ 会改变密度，量级可达 5–10%）。
- 黏度**不是** T/P 的函数，是带出处的常数 + 可覆盖。理由是：可用的简单
  黏度关联式在超临界区误差与常数本身的不确定度同量级，写一个"看起来更
  精确"的公式只会制造虚假精度。黏度的影响在分流曲线里以敏感性形式给出。
"""
from __future__ import annotations

import math
import os

# ─── CO₂ 临界参数（Span & Wagner 1996，CO₂ 的参考状态方程）──
TC_K = 304.1282
PC_PA = 7.3773e6
OMEGA = 0.22394
MOLAR_MASS = 0.0440095      # kg/mol
R = 8.314462618             # J/(mol·K)

TC_CELSIUS = TC_K - 273.15  # 30.98℃

PROPERTIES_VERSION = "2026.10-p1"


# ══════════════════════════════════════════════════════════
# Peng-Robinson
# ══════════════════════════════════════════════════════════
def _pr_params(temperature_k: float) -> tuple[float, float]:
    """返回 (a, b)。a 含温度依赖的 α(T)。"""
    tr = temperature_k / TC_K
    kappa = 0.37464 + 1.54226 * OMEGA - 0.26992 * OMEGA ** 2
    alpha = (1.0 + kappa * (1.0 - math.sqrt(tr))) ** 2
    a = 0.45724 * R * R * TC_K * TC_K / PC_PA * alpha
    b = 0.07780 * R * TC_K / PC_PA
    return a, b


def saturation_pressure_bar(temperature_celsius: float) -> float:
    """
    CO₂ 饱和蒸气压（bar），Span & Wagner (1996) 的 Wagner 式：

        ln(P/Pc) = (Tc/T)·(a1·τ + a2·τ^1.5 + a3·τ^2.5 + a4·τ^5),  τ = 1 − T/Tc

    20℃ 给 57.4 bar（实测 57.3），误差 < 0.5%。T ≥ Tc 时返回 ``inf``
    （不存在液气平衡）。

    用途只有一个：判断低于临界温度时 PR 方程该取哪个根。没有它就无法区分
    "150 bar / 20℃ 的 CO₂ 是液体还是气体"，而那决定了密度差 3 倍以上。
    """
    t = float(temperature_celsius) + 273.15
    if t >= TC_K:
        return float("inf")
    tau = 1.0 - t / TC_K
    a1, a2, a3, a4 = -7.0602087, 1.9391218, -1.6463597, -3.2995634
    ln_pr = (TC_K / t) * (
        a1 * tau + a2 * tau ** 1.5 + a3 * tau ** 2.5 + a4 * tau ** 5
    )
    return math.exp(ln_pr) * PC_PA / 1e5


def _pressure_of_volume(v: float, t: float, a: float, b: float) -> float:
    """PR 方程右端（Pa）。"""
    return R * t / (v - b) - a / (v * v + 2.0 * b * v - b * b)


def _crossings(pressure_pa: float, temperature_k: float) -> list[float]:
    """
    找出 P(V) = 目标压力 的全部体积根（从小到大）。

    低于临界温度时 PR 有三个根（液、亚稳、气），高于时只有一个。
    用对数扫描 + 二分逐个找出，比 Cardano 稳定：临界点附近三个根靠得极近，
    解析解的数值稳定性反而不如扫描。
    """
    a, b = _pr_params(temperature_k)
    v_lo = b * 1.001
    v_hi = 1.0                      # 1 m³/mol，远超任何气态体积
    n = 4000
    ratio = (v_hi / v_lo) ** (1.0 / (n - 1))
    roots: list[float] = []
    prev_v, prev_p = v_hi, _pressure_of_volume(v_hi, temperature_k, a, b)
    v = v_hi
    for _ in range(n):
        v /= ratio
        if v <= v_lo:
            break
        p = _pressure_of_volume(v, temperature_k, a, b)
        if (prev_p - pressure_pa) * (p - pressure_pa) <= 0.0:
            lo, hi = v, prev_v
            p_hi = _pressure_of_volume(hi, temperature_k, a, b)
            for _ in range(200):
                mid = 0.5 * (lo + hi)
                pm = _pressure_of_volume(mid, temperature_k, a, b)
                if (pm - pressure_pa) * (p_hi - pressure_pa) <= 0:
                    lo = mid
                else:
                    hi = mid
                if hi - lo < 1e-16:
                    break
            roots.append(0.5 * (lo + hi))
        prev_v, prev_p = v, p
    roots.sort()
    return roots


def _molar_volume(pressure_pa: float, temperature_k: float) -> tuple[float, str]:
    """
    解摩尔体积。返回 (V, phase)。

    高于临界温度：只有一个根（致密/超临界相）。
    低于临界温度：用饱和蒸气压判断该取哪个根 —— 压力高于饱和压力取**最小**
    体积（液相），否则取最大体积（气相）。取错分支会让密度差三倍以上，
    而且两种取值都是"合理的数字"，不会报错。
    """
    roots = _crossings(pressure_pa, temperature_k)
    if not roots:
        return R * temperature_k / max(pressure_pa, 1e-9), "ideal-fallback"
    if temperature_k >= TC_K:
        return roots[-1], "supercritical"
    psat_pa = saturation_pressure_bar(temperature_k - 273.15) * 1e5
    if pressure_pa > psat_pa:
        return roots[0], "liquid"
    return roots[-1], "gas"


def co2_density_kg_m3(pressure_bar: float, temperature_celsius: float) -> float:
    """
    CO₂ 密度（kg/m³），Peng-Robinson 状态方程。

    与旧实现 ``1.9·P/T`` 的差别不是精度，是量级：250 bar / 80℃ 下
    本函数给约 650 kg/m³，旧式给 1.35 kg/m³。

    参数越界（T ≤ 0 K、P ≤ 0）抛 ``ValueError`` 而不是返回 0 ——
    返回 0 会让下游的浮力计算静默变成"不浮"，比报错危险得多。
    """
    p = float(pressure_bar) * 1e5
    t = float(temperature_celsius) + 273.15
    if t <= 0:
        raise ValueError("温度必须高于绝对零度")
    if p <= 0:
        raise ValueError("压力必须为正")
    v, _phase = _molar_volume(p, t)
    return MOLAR_MASS / v


def co2_phase(pressure_bar: float, temperature_celsius: float) -> str:
    """返回 CO₂ 相态标签：supercritical / gas / liquid / ideal-fallback。"""
    p = float(pressure_bar) * 1e5
    t = float(temperature_celsius) + 273.15
    if t >= TC_K and p >= PC_PA:
        return "supercritical"
    if t >= TC_K:
        return "gas"
    _v, phase = _molar_volume(p, t)
    return phase


# ══════════════════════════════════════════════════════════
# 黏度
# ══════════════════════════════════════════════════════════
# 出处：Span & Wagner (1996) 在 200–400 bar / 320–370 K 给出 0.03–0.08 mPa·s；
# 取 0.06 为中值。**不建模为 T/P 的函数**：可用的简单关联式在超临界区的
# 误差与这个区间本身的宽度同量级，写一个更"精确"的公式只是虚假精度。
CO2_VISCOSITY_DEFAULT = 0.06      # mPa·s


def co2_viscosity_mpa_s(pressure_bar: float = 250.0,
                        temperature_celsius: float = 60.0,
                        environ: dict | None = None) -> float:
    """CO₂ 黏度（mPa·s）。常数 + 环境变量覆盖（``CCUS_CO2_VISCOSITY``）。"""
    env = os.environ if environ is None else environ
    raw = (env.get("CCUS_CO2_VISCOSITY") or "").strip()
    if raw:
        try:
            v = float(raw)
            if 0.01 <= v <= 0.5:      # 超出这个范围一定是单位写错（cP/Pa·s）
                return v
        except ValueError:
            pass
    return CO2_VISCOSITY_DEFAULT


def water_viscosity_mpa_s(temperature_celsius: float) -> float:
    """
    纯水黏度（mPa·s），Vogel 型关联式：

        μ = 2.414e-5 · 10^(247.8/(T − 140))   [Pa·s，T 为 K]

    80℃ 给 0.351 mPa·s（标准值 0.355），20℃ 给 1.001（标准值 1.002）。
    注意常数是 2.414e-5 **Pa·s**，换算成 mPa·s 要再乘 1000 —— 这里写错过
    一次（写成 2.414e-4），结果小 100 倍，而数值本身"看起来像个小数"，
    不报错也不越界。所以测试里直接对 20℃/80℃ 两个标准值做断言。

    含盐地层水会略高（20% 盐度约 +10–20%），本项目不建模盐度对黏度的影响
    —— 分流曲线对 μ_w 不敏感（水相黏度在分母里被 CO₂ 的低黏度盖过）。
    """
    t = float(temperature_celsius) + 273.15
    if t <= 140.0:
        raise ValueError("Vogel 关联式在 T ≤ 140 K 无意义")
    return 2.414e-5 * 10.0 ** (247.8 / (t - 140.0)) * 1000.0


# ══════════════════════════════════════════════════════════
# 界面张力
# ══════════════════════════════════════════════════════════
# 出处：Bachu & Bennion (2009, J. Chem. Eng. Data 54) 与 Chalbaud et al.
# (2009, Adv. Water Resour. 32) 实测 CO₂–盐水界面张力在储层条件下
# 20–30 mN/m。取 25 为区间中值。
#
# **不做 T/P 的函数**：文献值随压力升高先降后趋平、随温度升高而升，
# 但不同实验体系差异可达 ±20%，超过任何简化关联式的解释力。盖层判定里
# 直接给 20/25/30 三档敏感性（见 utils/caprock.py），比伪造一个公式诚实。
IFT_DEFAULT = 25.0      # mN/m


def interfacial_tension_mnm(pressure_bar: float = 250.0,
                            temperature_celsius: float = 60.0,
                            environ: dict | None = None) -> float:
    """CO₂–盐水界面张力（mN/m）。常数 + ``CCUS_IFT_MNM`` 覆盖。"""
    env = os.environ if environ is None else environ
    raw = (env.get("CCUS_IFT_MNM") or "").strip()
    if raw:
        try:
            v = float(raw)
            if 5.0 <= v <= 70.0:
                return v
        except ValueError:
            pass
    return IFT_DEFAULT


def buoyancy_density_difference(rho_brine: float, pressure_bar: float,
                                temperature_celsius: float) -> float:
    """
    浮力密度差 Δρ = ρ_brine − ρ_CO2（kg/m³）。

    盖层能封多高的 CO₂ 柱、重力分异多快，全部与这个差值成正比。
    旧密度式在这里会给出 Δρ ≈ 1029（CO₂ 几乎"没有重量"），
    真实值约 380–450 —— 于是柱高上限被高估 2.3 倍以上。
    """
    return float(rho_brine) - co2_density_kg_m3(pressure_bar, temperature_celsius)
