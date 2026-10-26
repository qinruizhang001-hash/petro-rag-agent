"""
孔隙弹性应力路径：注入抬高的不只是孔隙压力
============================================
``utils/geomechanics.py`` 的判据假定总应力不变（Δσ = 0），于是有效正应力
只随孔隙压力变化：

    Δσ' = −ΔP

这在**注入层内部**是偏保守的，因为真实储层侧向受限，孔隙压力上升会把
水平总应力一起顶上去。少了这一项，判据会把"有效应力下降"放大若干倍，
在临界状态附近给出偏严的结论 —— 偏严不危险，但它会让"断层稳定"这个
结论在最需要它的时候说不出口。

## 应力路径的推导（不是一条要背的公式）

单轴应变（侧向无应变，上覆荷载不变）条件下，孔隙弹性介质的水平应力为

    σ_h = ν/(1−ν)·(σ_v − αP) + αP

把 ν/(1−ν)·(−αP) 与 αP 合并：

    σ_h = ν/(1−ν)·σ_v + α·(1−2ν)/(1−ν)·P

对 P 求导就得到应力路径系数

    Δσ_h/ΔP = α·(1−2ν)/(1−ν)          （Biot-Willis；Zoback 2007 eq. 4.13）

有效水平应力则要再扣掉孔隙压力本身：

    Δσ'_h = Δσ_h − α·ΔP = −α·ν/(1−ν)·ΔP

α=0.8、ν=0.25 时：Δσ_h = 0.533ΔP，Δσ'_h = −0.267ΔP。
也就是说"总应力不变"把有效应力的下降**放大了 3.75 倍**。

## 方向性偏差（必须写进报告）

耦合口径让判定**更宽松**（余量更大、更不容易判活化）。这不是"更安全"，
而是"对注入层更真实"。它的适用范围只有注入层本身：

- 盖层与上下隔层不在注入层内。它们的应力变化由**应力转移**决定 ——
  三维孔隙弹性解在某些位置会**降低** σ_h，从而**促进**滑动。
  符号依赖几何与刚度对比，本项目没有地质力学求解器，不建模。
- 所以不能把"耦合口径余量更大"推广成"整个场地更安全"。

正因为如此，``geomechanics.fault_reactivation`` 的**判定闸门仍然走保守口径**
（总应力不变），耦合口径作为第二个数一并返回。把宽松的那个当结论，
是把一个已知的乐观偏差当成了修正。

## 一个可以证明的单调性（只在注入工况下成立）

对倾角 δ 的平面（法向水平分量 s = sin²δ），剪应力有闭式

    τ = √(s(1−s))·|σ_h − σ_v|

注入使 σ_h 上升（Δσ_h > 0），于是 |σ_h − σ_v| 减小、σ_n 增大 ——
两个方向都让滑移比 τ/(σ_n − P_p) 下降。所以**在 ΔP ≥ 0 时**，耦合口径的
滑移比恒不大于保守口径。这条不等式不是经验判断，是上面两个式子的直接
推论，``tests/test_poroelastic.py`` 用网格逐点验证它。

**前提不能丢**：ΔP < 0（输入压力低于原始孔隙压力，例如把枯竭/生产工况
套进来）时 σ_h 跟着下降，剪应力上升、正应力下降，耦合口径反而**更严重**。
这不是反例，是同一个式子的另一半 —— 测试对两个方向分别断言。

## 单位

与 ``geomechanics`` 一致：MPa。

## 索引

本模块的假设、适用范围与偏差方向，与其余物理模块并列在
``docs/physics-limits.md``（一页总表）。改公式或默认值时要同步改那张表 ——
只改代码不改表，读者会拿旧边界去读新数字。
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass

POROELASTIC_VERSION = "2026.10-p1"


@dataclass(frozen=True)
class PoroelasticParams:
    """孔隙弹性参数。

    ``alpha``（Biot 系数）与 ``poisson``（排水泊松比）都**不是**能从语料里
    读出来的量，只能用岩性范围 + 敏感性区间。覆盖方式::

        CCUS_BIOT_ALPHA=0.8
        CCUS_POISSON=0.25
    """

    alpha: float = 0.8        # Biot 系数 α（0–1）
    poisson: float = 0.25     # 排水泊松比 ν


PARAMETER_SOURCES = {
    "alpha": {
        "unit": "无量纲",
        "source": (
            "Biot 系数：砂岩 0.6–0.9，泥岩 0.4–0.7，裂缝性碳酸盐岩可接近 1"
            "（Zoback 2007, §4.3；Rutqvist 2012, Geotech. Geol. Eng. 30）。"
            "取 0.8 代表砂岩储层。α=1 退化为 Terzaghi 有效应力。"
        ),
    },
    "poisson": {
        "unit": "无量纲",
        "source": (
            "排水泊松比：砂岩 0.2–0.3（Mavko et al. 2009, Rock Physics Handbook）。"
            "取 0.25 为中值。ν 越大 → (1−2ν)/(1−ν) 越小 → 应力路径越弱。"
        ),
    },
}


def load_poroelastic_params(environ: dict | None = None) -> PoroelasticParams:
    """读环境变量；越界回退默认（与其余物理模块同策略，不截断）。"""
    env = os.environ if environ is None else environ

    def _num(key: str, default: float, lo: float, hi: float) -> float:
        raw = (env.get(key) or "").strip()
        if not raw:
            return default
        try:
            value = float(raw)
        except ValueError:
            return default
        return value if lo <= value <= hi else default

    return PoroelasticParams(
        alpha=_num("CCUS_BIOT_ALPHA", 0.8, 0.0, 1.0),
        poisson=_num("CCUS_POISSON", 0.25, 0.0, 0.49),
    )


def _assumed(environ: dict | None = None) -> list[str]:
    env = os.environ if environ is None else environ
    keys = {"alpha": "CCUS_BIOT_ALPHA", "poisson": "CCUS_POISSON"}
    return [f for f, var in keys.items() if not (env.get(var) or "").strip()]


def manifest(params: PoroelasticParams | None = None,
             environ: dict | None = None) -> dict:
    params = params or load_poroelastic_params(environ)
    values = asdict(params)
    return {
        "version": POROELASTIC_VERSION,
        "assumed": _assumed(environ),
        "parameters": {
            key: {
                "value": values[key],
                "unit": PARAMETER_SOURCES.get(key, {}).get("unit", ""),
                "source": PARAMETER_SOURCES.get(key, {}).get("source", ""),
            }
            for key in values
        },
    }


# ══════════════════════════════════════════════════════════
# 应力路径系数
# ══════════════════════════════════════════════════════════
def horizontal_stress_path(alpha: float, poisson: float) -> float:
    """Δσ_h/ΔP = α·(1−2ν)/(1−ν)。

    两个端点必须成立，测试锁定：
    - α=1, ν=0 → 1.0（Terzaghi 极限：总水平应力跟着孔隙压力等量上升）
    - ν=0.5 → 0.0（不可压缩：侧向受限不产生附加水平应力）

    ``poisson`` 必须 < 0.5，否则分母为 0 或为负 —— 排水泊松比在物理上
    不会达到 0.5（那对应体积不可压缩），所以这里直接拒绝而不是返回一个
    负数然后被下游当成"注入降低了水平应力"。
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"Biot 系数必须在 [0, 1]，实际 {alpha}")
    if not 0.0 <= poisson < 0.5:
        raise ValueError(f"泊松比必须在 [0, 0.5)，实际 {poisson}")
    return alpha * (1.0 - 2.0 * poisson) / (1.0 - poisson)


def effective_stress_path(alpha: float, poisson: float) -> float:
    """Δσ'_h/ΔP = −α·ν/(1−ν)，恒为负（注入总是降低有效水平应力）。

    与 ``horizontal_stress_path`` 的关系是定义式的，不是巧合：

        Δσ'_h/ΔP = Δσ_h/ΔP − α = α[(1−2ν)/(1−ν) − 1] = −α·ν/(1−ν)

    测试用这条恒等式交叉验证两个函数。
    """
    return horizontal_stress_path(alpha, poisson) - float(alpha)


def stress_after_injection(sv: float, shmin: float, shmax: float, pp: float,
                           delta_pp_mpa: float,
                           params: PoroelasticParams | None = None) -> dict:
    """注入 ``delta_pp_mpa`` 之后的应力状态（MPa）。

    单轴应变假设下：垂向总应力不变、两个水平主应力同步上升、孔隙压力上升。
    水平方向按**各向同性**处理（同一个 Δσ_h 加到 Shmin 与 SHmax 上）——
    这是单轴应变的本义，也是这个模型不能分辨走滑/逆冲区差异的原因。
    """
    p = params or load_poroelastic_params()
    dsig = horizontal_stress_path(p.alpha, p.poisson) * float(delta_pp_mpa)
    return {
        "sv": float(sv),
        "shmin": float(shmin) + dsig,
        "shmax": float(shmax) + dsig,
        "pp": float(pp) + float(delta_pp_mpa),
        "delta_sigma_h_mpa": dsig,
        "delta_sigma_h_eff_mpa": dsig - p.alpha * float(delta_pp_mpa),
        "path_ratio": horizontal_stress_path(p.alpha, p.poisson),
        "effective_path_ratio": effective_stress_path(p.alpha, p.poisson),
    }


def sensitivity(alpha_values=(0.6, 0.8, 1.0),
                poisson_values=(0.20, 0.25, 0.30)) -> list[dict]:
    """α × ν 的应力路径敏感性表。

    给区间而不是给一个数：这两个参数都没有实测来源，单点值会让读者
    以为它们是被测过的。α=1 那一列是 Terzaghi 上限（最宽松的耦合口径）。
    """
    out = []
    for a in alpha_values:
        for nu in poisson_values:
            out.append({
                "alpha": a,
                "poisson": nu,
                "path_ratio": round(horizontal_stress_path(a, nu), 4),
                "effective_path_ratio": round(effective_stress_path(a, nu), 4),
            })
    return out
