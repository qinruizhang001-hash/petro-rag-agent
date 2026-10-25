"""
相对渗透率与两相流近似（纯标准库）
==================================
``physics_rules`` 判定的是"压力对不对"，它默认 CO₂ 一注进去就在储层里均匀铺开。
真实情况不是这样：CO₂ 比水轻、比水易流动（黏度约 1/10），所以它会**指进**、
会**上浮**、会在前沿形成饱和度突变。这些东西决定了两个工程上真正关心的问题：

    CO₂ 注得进去吗？      → 流度比、注入指数
    注进去占了多少孔隙？  → 前沿后平均饱和度（Welge 1952）

模型是经典的一维两相不可混溶驱替（Buckley & Leverett 1942），相渗用
Brooks-Corey（1964）：

    Se   = (S_w − S_wr)/(1 − S_wr − S_gr)
    kr_w = Se^n_w,   kr_g = (1 − Se)^n_g
    f_w  = (kr_w/μ_w) / (kr_w/μ_w + kr_g/μ_g)

## 为什么用解析解而不是直接上数值求解器

一维 BL 方程有严格的解析解（Welge 切线法）：前沿饱和度 ``S_wf`` 由
"从 (S_wr, 0) 向 f_w 曲线作切线"确定，前沿后是平台。这给了数值求解器一个
**可对标的真值** —— ``utils/flow1d.py`` 的求解器就是拿它验证的。

## 边界

- 不可混溶、等温、一维。没有毛细管压力、没有溶解、没有相间传质。
  真实 CO₂ 注入中溶解度可达孔隙体积的百分之几，会让前锋变慢。
- Brooks-Corey 是经验式，n 与残余饱和度必须由岩心相渗曲线标定；
  默认值是砂岩代表值，不是任何具体区块的值。
"""
from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass

RELPERM_VERSION = "2026.10-rp1"


@dataclass(frozen=True)
class RelPermParams:
    """Brooks-Corey 相渗 + 两相黏度。

    默认值是**砂岩代表值**，不是某个区块的标定结果。有岩心相渗曲线时必须
    覆盖，否则分流曲线与前沿饱和度都只是"一类岩石"的答案。
    """

    swr: float = 0.25          # 束缚水饱和度
    sgr: float = 0.15          # 残余气（CO₂）饱和度
    nw: float = 2.0            # 水相指数
    ng: float = 2.0            # 气相指数
    mu_w: float = 0.5          # mPa·s，水相黏度
    mu_g: float = 0.06         # mPa·s，CO₂ 黏度


PARAMETER_SOURCES = {
    "swr": {
        "unit": "小数",
        "source": "砂岩束缚水饱和度典型 0.15–0.35（Corey 1954；Brooks & Corey 1964）；取 0.25。",
    },
    "sgr": {
        "unit": "小数",
        "source": (
            "CO₂ 残余气饱和度实测 0.05–0.25（Bachu & Bennion 2008 的岩心驱替）；"
            "取 0.15。这个值直接决定「注进去的 CO₂ 有多少留在孔隙里」，"
            "是盖层/容量评估里最敏感的参数之一。"
        ),
    },
    "nw": {"unit": "无量纲", "source": "Brooks-Corey 指数，砂岩水相常取 2（Brooks & Corey 1964）。"},
    "ng": {"unit": "无量纲", "source": "Brooks-Corey 指数，砂岩气相常取 2（Brooks & Corey 1964）。"},
    "mu_w": {
        "unit": "mPa·s",
        "source": "60℃ 地层水黏度约 0.47 mPa·s（Vogel 关联式给 0.47）；取 0.5。",
    },
    "mu_g": {
        "unit": "mPa·s",
        "source": "超临界 CO₂ 黏度 0.03–0.08 mPa·s（Span & Wagner 1996）；取 0.06。",
    },
}


def load_relperm_params(environ: dict | None = None) -> RelPermParams:
    env = os.environ if environ is None else environ

    def _num(key, default, lo, hi):
        raw = (env.get(key) or "").strip()
        if not raw:
            return default
        try:
            v = float(raw)
        except ValueError:
            return default
        return v if lo <= v <= hi else default

    p = RelPermParams(
        swr=_num("CCUS_SWR", 0.25, 0.0, 0.6),
        sgr=_num("CCUS_SGR", 0.15, 0.0, 0.5),
        nw=_num("CCUS_NW", 2.0, 0.5, 6.0),
        ng=_num("CCUS_NG", 2.0, 0.5, 6.0),
        mu_w=_num("CCUS_MU_W", 0.5, 0.05, 5.0),
        mu_g=_num("CCUS_MU_G", 0.06, 0.01, 0.5),
    )
    # 束缚水 + 残余气不能把可动孔隙占满，否则分母为 0
    if p.swr + p.sgr >= 0.95:
        return RelPermParams()
    return p


# ══════════════════════════════════════════════════════════
# 相渗与分流
# ══════════════════════════════════════════════════════════
def relperm(sw: float, params: RelPermParams | None = None) -> dict:
    """
    给定含水饱和度，返回两相相渗与分流量。

    饱和度超出 ``[swr, 1−sgr]`` 时**夹紧**而不是外推：外推会让 kr 变成负数或
    大于 1，而分流曲线在端点外没有物理意义。夹紧是显式的、可解释的，
    外推是静默的、会污染下游积分。
    """
    p = params or load_relperm_params()
    lo, hi = p.swr, 1.0 - p.sgr
    sw_c = min(max(sw, lo), hi)
    span = hi - lo
    se = (sw_c - lo) / span if span > 0 else 0.0

    kr_w = se ** p.nw
    kr_g = (1.0 - se) ** p.ng

    # 流度
    lam_w = kr_w / p.mu_w
    lam_g = kr_g / p.mu_g
    total = lam_w + lam_g
    f_w = lam_w / total if total > 0 else 0.0

    return {
        "sw": sw,
        "sw_clamped": sw_c,
        "se": se,
        "kr_w": kr_w,
        "kr_g": kr_g,
        "lambda_w": lam_w,
        "lambda_g": lam_g,
        "f_w": f_w,
        "f_g": 1.0 - f_w,
    }


def fractional_flow(sw: float, params: RelPermParams | None = None) -> float:
    """含水率 f_w（无因次）。"""
    return relperm(sw, params)["f_w"]


# ══════════════════════════════════════════════════════════
# Buckley-Leverett 前沿（Welge 切线法）
# ══════════════════════════════════════════════════════════
def bl_front(params: RelPermParams | None = None) -> dict:
    """
    求 BL 前沿的 **CO₂** 饱和度与前沿后平均饱和度。

    方向很容易搞反，这里写清楚：实际过程是 **CO₂ 驱替地层水**，被注入相是
    CO₂，所以切线要从 ``(S_g = 0, f_g = 0)`` 出发，去切 ``f_g(S_g)`` 曲线，
    而不是从水端 ``(S_w = S_wr, f_w = 0)`` 去切 ``f_w``。两者互为镜像，
    算出来完全不是一个数：本项目参数下水端切线给 CO₂ 饱和度 0.17，
    气端切线给 0.50 —— 差三倍，而且两种算法都不报错。

    **做法**：前沿饱和度 ``S_gf`` 使 ``f_g(S)/S`` 取最大（从原点作切线的切点）。
    先粗扫定位，再三分法细化 —— 直接解 ``df/dS·S − f = 0`` 在 S→0 附近
    分子分母同时趋零，不稳定；"最大化斜率"没有奇点。

    返回：
        sg_front          前沿处 CO₂ 饱和度
        fg_front          前沿处 CO₂ 分流量
        dfg_front         df_g/dS_g 在前沿处的值（= 前沿无因次速度）
        co2_saturation    前沿后平均 CO₂ 饱和度（Welge 公式）
        avg_sw_behind     对应的平均含水饱和度
        mobility_ratio    端点流度比 M = (kr_g^o/μ_g)/(kr_w^o/μ_w)
    """
    p = params or load_relperm_params()
    sg_max = 1.0 - p.swr
    if sg_max <= 1e-9:
        raise ValueError("swr 太大，可动孔隙为零")

    def fg(sg: float) -> float:
        return 1.0 - fractional_flow(1.0 - sg, p)

    def slope(sg: float) -> float:
        return fg(sg) / sg if sg > 1e-12 else 0.0

    # ① 粗扫
    n = 400
    best_s, best_v = 0.0, -1.0
    for i in range(1, n + 1):
        s = sg_max * i / (n + 1)
        v = slope(s)
        if v > best_v:
            best_s, best_v = s, v

    # ② 三分法细化
    step = sg_max / (n + 1)
    a, b = max(1e-9, best_s - step), min(sg_max, best_s + step)
    for _ in range(200):
        m1 = a + (b - a) / 3.0
        m2 = b - (b - a) / 3.0
        if slope(m1) < slope(m2):
            a = m1
        else:
            b = m2
        if b - a < 1e-12:
            break
    sg_f = 0.5 * (a + b)

    fg_f = fg(sg_f)
    # 对 S_g 的中心差分
    h = 1e-6
    dfg_f = (fg(sg_f + h) - fg(sg_f - h)) / (2 * h)
    # Welge: S̄_g = S_gf + (1 − f_gf)/(df_g/dS_g)
    avg_sg = sg_f + (1.0 - fg_f) / dfg_f if dfg_f > 0 else sg_f

    krw_end = 1.0
    krg_end = 1.0
    mobility = (krg_end / p.mu_g) / (krw_end / p.mu_w)

    return {
        "sg_front": round(sg_f, 6),
        "fg_front": round(fg_f, 6),
        "dfg_front": round(dfg_f, 6),
        "co2_saturation": round(avg_sg, 6),
        "avg_sw_behind": round(1.0 - avg_sg, 6),
        "mobility_ratio": round(mobility, 3),
        "version": RELPERM_VERSION,
        "params": {k: {"value": v, "unit": PARAMETER_SOURCES.get(k, {}).get("unit", ""),
                       "source": PARAMETER_SOURCES.get(k, {}).get("source", "")}
                   for k, v in asdict(p).items()},
    }


def injectivity_index(k_md: float, h_m: float, mu_mpa_s: float,
                      re_m: float = 500.0, rw_m: float = 0.1) -> dict:
    """
    径向稳态注入指数（Darcy）：

        J = q/ΔP = 2π·k·h / (μ · ln(re/rw))      [m³/(s·Pa)]

    这是"注得进去吗"的量级估计，不是产能预测：它假定单相、稳态、均质、
    无表皮。真实注入能力受相渗、压裂、非均质控制，量级可能差几倍。
    """
    if k_md <= 0 or h_m <= 0 or mu_mpa_s <= 0:
        raise ValueError("k / h / μ 必须为正")
    k = k_md * 9.869233e-16          # 1 mD = 9.869233e-16 m²
    mu = mu_mpa_s * 1e-3             # mPa·s → Pa·s
    j = 2.0 * math.pi * k * h_m / (mu * math.log(re_m / rw_m))
    return {
        "J_m3_per_s_per_pa": j,
        # 常用工程单位：m³/(d·MPa)
        "J_m3_per_day_per_mpa": j * 86400.0 * 1e6,
        "assumptions": "单相、稳态、均质、无表皮；实际注入能力受相渗与非均质控制",
    }


def render_block(front: dict) -> str:
    """渲染成报告可用文字。"""
    return "\n".join([
        f"【两相流特征（{front['version']}）】",
        f"- BL 前沿 CO₂ 饱和度 {front['sg_front']:.3f}，"
        f"前沿后平均 CO₂ 饱和度 {front['co2_saturation']:.3f}",
        f"- 前沿无因次速度 df_g/dS_g = {front['dfg_front']:.3f}",
        f"- 端点流度比 M = {front['mobility_ratio']:.1f}"
        f"（>1 表示 CO₂ 比水易流动，前沿会指进）",
    ])
