"""
地层压力模型（纯 Python，零依赖）
=================================
静水压力 / 地层破裂压力 / 可注入深度窗口 —— 这三个式子是
"图上每条线"和"监控台上每个读数"的唯一来源。

为什么要单独一个模块
--------------------
原本这三个函数住在 ``chart_tools.py`` 里，但那个模块 ``import matplotlib``。
前端 ``app.py`` 要实时画曲线，就必须能算这三个式子；而前端有一条硬规矩
（CI 里 ``frontend-contract`` 会检查）：**禁止 import matplotlib**。
把公式复制一份到前端更糟 —— 两份实现迟早会漂移，图上的线和报告里的数字
就会对不上，而这种错不会报异常，只会悄悄给出一张"看起来对"的图。

所以公式下沉到这里：不带 numpy、不带 matplotlib、标量和序列都能吃。
``chart_tools.py`` 和 ``app.py`` 都从本模块取，谁也不许再写第二份。

单位约定
--------
深度 m、压力 bar（1 bar = 1e5 Pa）、密度 kg/m³、重力加速度 m/s²。
"""

from __future__ import annotations

# ─── 模型常数 ────────────────────────────────────────────
# 含盐地层水密度。用 1000 会让 2000 m 处少算 6 bar，跨深度比较时会有系统性偏差。
RHO_WATER = 1030.0          # kg/m³
G = 9.81                    # m/s²
# 破裂压力梯度常用 1.4–1.8 倍静水压力，这里取中值。
FRACTURE_RATIO = 1.5

# 前端在用户没填参数时的占位值（也是后端 generate_pressure_curve 的缺省值）
DEFAULT_DEPTH_M = 2000.0
DEFAULT_INJECTION_BAR = 250.0

# 与 chart_tools.generate_pressure_curve 的入参夹取保持一致：
# 防止模型传入离谱数值把图压扁成一条线。
MIN_DEPTH_M, MAX_DEPTH_M = 200.0, 8000.0
MIN_INJECTION_BAR, MAX_INJECTION_BAR = 20.0, 1200.0


def _map(x, f):
    """标量直接算，序列逐元素算。

    这里刻意不 ``import numpy``：前端不允许依赖 numpy，而 chart_tools 传进来的
    是 ``np.linspace`` 的结果。用"可迭代且不是字符串"来识别序列，
    numpy 数组天然满足，返回 list —— matplotlib 画 list 没有任何问题。
    """
    if isinstance(x, (str, bytes)):
        return f(x)
    if hasattr(x, "__iter__"):
        return [f(v) for v in x]
    return f(x)


def hydrostatic_bar(depth_m):
    """深度（m）→ 静水压力（bar）。P = ρgh，1 bar = 1e5 Pa。"""
    return _map(depth_m, lambda d: RHO_WATER * G * float(d) / 1e5)


def fracture_bar(depth_m):
    """深度（m）→ 地层破裂压力（bar）。"""
    return _map(depth_m, lambda d: RHO_WATER * G * float(d) / 1e5 * FRACTURE_RATIO)


def admissible_depth_range(injection_pressure_bar: float) -> tuple[float, float]:
    """
    给定注入压力，求"既注得进又不压裂"的深度区间 [d_min, d_max]。

        P_frac(d) >= P_inj  →  d >= P_inj / (ρg/1e5 · 1.5)   （下界：不压裂）
        P_h(d)    <= P_inj  →  d <= P_inj / (ρg/1e5)         （上界：注得进）
    """
    p = float(injection_pressure_bar)
    grad = RHO_WATER * G / 1e5          # bar/m
    return float(p / (grad * FRACTURE_RATIO)), float(p / grad)


def clamp_inputs(depth_m: float, injection_pressure_bar: float) -> tuple[float, float]:
    """把入参夹到可用范围，与后端 generate_pressure_curve 完全一致。"""
    d = max(MIN_DEPTH_M, min(float(depth_m), MAX_DEPTH_M))
    p = max(MIN_INJECTION_BAR, min(float(injection_pressure_bar), MAX_INJECTION_BAR))
    return d, p


def verdict_of(depth_m: float, injection_pressure_bar: float) -> str:
    """
    在储层深度这一行做判断，返回 ``ok`` / ``risky`` / ``under``。

    **只看储层深度**：浅部根本不注入，拿定值压力去比浅部破裂压力没有工程意义
    （早期版本按"整条曲线上是否有交点"判断，是错的）。
    """
    d, p = clamp_inputs(depth_m, injection_pressure_bar)
    p_h = hydrostatic_bar(d)
    p_f = fracture_bar(d)
    if p > p_f:
        return "risky"
    if p < p_h:
        return "under"
    return "ok"
