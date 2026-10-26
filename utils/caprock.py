"""
盖层封闭性：毛细管突破与力学失效（纯标准库）
============================================
``physics_rules`` 里那条 ``P_inj < 1.5×静水`` 回答的是"会不会把地层压裂"。
它**没有**回答盖层最常失效的那个机制：CO₂ 比水轻，会往上浮，浮力把 CO₂
往盖层孔隙里挤；盖层孔隙喉道越细，毛细管阻力越大，能封住的 CO₂ 柱就越高。
当柱高产生的浮力超过突破压力，CO₂ 就开始渗漏 —— 这个过程**不需要**超过
破裂压力，注入压力完全正常也可能发生。

两种失效模式，判据完全不同：

    毛细管突破    Δρ·g·H ≥ P_c,entry          P_c = 2γcosθ/r   （Laplace/Washburn）
    力学失效      P_inj ≥ σ_hmin + T_0        盖层被张性压开

## "动态"在哪里

静态判断只问"现在这个柱高安不安全"。真正有用的是"什么时候不安全"：
注入量随时间累积 → 剖面饱和度上升 → 等效 CO₂ 柱高增长 → 浮力增长 →
到某一刻越过突破压力。本模块把这个链条算出来，给出**突破时间**：

    柱高 H(t)  ←  一维两相驱替剖面（utils/flow1d.py）
    浮力 P_b(t) = Δρ·g·H(t)
    突破时刻 t*  满足 P_b(t*) = P_c

链条上每一环都是确定性计算，没有一处让模型"估计"。

## 边界（写在这里，不藏在文档里）

- 界面张力**不建模**为 T/P 的函数，用文献区间中值 + 三档敏感性
  （20/25/30 mN/m）。理由见 ``utils/co2_properties.py``：可用的简化关联式
  在超临界区的误差与文献值本身的分散度同量级。
- 孔喉半径是控制变量，通常只有 MICP 实测才有。没有实测时用 10 nm，
  并且**结论必须写明这是假设** —— 孔喉从 10 nm 变到 5 nm，突破压力翻倍，
  这个参数的不确定度远大于公式本身。
- 不考虑盖层本身的非均质性（裂缝、砂条带），那才是实际泄漏的主因。
  这里判的是"完整盖层"的理想情况。

## 索引

本模块的假设、适用范围与偏差方向，与其余物理模块并列在
``docs/physics-limits.md``（一页总表）。改公式或默认值时要同步改那张表 ——
只改代码不改表，读者会拿旧边界去读新数字。
"""
from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass

from utils import co2_properties as cp
from utils import geomechanics as gm
from utils import relperm as rp
from utils import flow1d

CAPROCK_VERSION = "2026.10-c1"
G = 9.81


@dataclass(frozen=True)
class CaprockParams:
    """盖层参数。**孔喉半径是最敏感的一个**，没有 MICP 实测时必须当作假设。"""

    ift_mnm: float = 25.0              # CO₂–盐水界面张力
    contact_angle_deg: float = 30.0    # 接触角（水湿，θ<90°）
    pore_throat_radius_nm: float = 10.0
    tensile_strength_mpa: float = 3.0  # 盖层抗张强度


PARAMETER_SOURCES = {
    "ift_mnm": {
        "unit": "mN/m",
        "source": (
            "CO₂–盐水界面张力在储层条件下 20–30 mN/m（Bachu & Bennion 2009, "
            "J. Chem. Eng. Data 54；Chalbaud et al. 2009, Adv. Water Resour. 32）。"
            "取 25 为中值，并给 20/25/30 三档敏感性。"
        ),
    },
    "contact_angle_deg": {
        "unit": "度",
        "source": (
            "CO₂–盐水–石英体系为水湿（θ < 90°），但 θ 随压力、温度、盐度上升，"
            "实测 20–50°（Espinoza & Santamarina 2017, Water Resour. Res.）。"
            "取 30°：偏大 → cosθ 偏小 → 突破压力偏低，属保守侧（宁可低估封闭性）。"
        ),
    },
    "pore_throat_radius_nm": {
        "unit": "nm",
        "source": (
            "泥质盖层孔喉半径 2–20 nm（MICP 实测，Schowalter 1979, AAPG Bull. 63 "
            "给出的进入压力 0.1–10 MPa 反推）。取 10 nm。"
            "**这是全模块最敏感的参数**：r 减半则 P_c 翻倍。无 MICP 数据时结论必须标注为假设。"
        ),
    },
    "tensile_strength_mpa": {
        "unit": "MPa",
        "source": (
            "页岩抗张强度 2–10 MPa（Rutqvist 2012, Geotech. Geol. Eng. 30）；"
            "取 3 MPa 偏下限，属保守侧。"
        ),
    },
}


def load_caprock_params(environ: dict | None = None) -> CaprockParams:
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

    return CaprockParams(
        ift_mnm=_num("CCUS_IFT_MNM", 25.0, 5.0, 70.0),
        contact_angle_deg=_num("CCUS_CONTACT_ANGLE", 30.0, 0.0, 85.0),
        pore_throat_radius_nm=_num("CCUS_PORE_THROAT_NM", 10.0, 1.0, 200.0),
        tensile_strength_mpa=_num("CCUS_TENSILE_STRENGTH", 3.0, 0.0, 20.0),
    )


def _assumed(environ: dict | None = None) -> list[str]:
    env = os.environ if environ is None else environ
    keys = {
        "ift_mnm": "CCUS_IFT_MNM",
        "contact_angle_deg": "CCUS_CONTACT_ANGLE",
        "pore_throat_radius_nm": "CCUS_PORE_THROAT_NM",
        "tensile_strength_mpa": "CCUS_TENSILE_STRENGTH",
    }
    return [f for f, var in keys.items() if not (env.get(var) or "").strip()]


def caprock_manifest(params: CaprockParams | None = None,
                     environ: dict | None = None) -> dict:
    params = params or load_caprock_params(environ)
    values = asdict(params)
    return {
        "version": CAPROCK_VERSION,
        "assumed": _assumed(environ),
        "parameters": {
            k: {"value": values[k],
                "unit": PARAMETER_SOURCES.get(k, {}).get("unit", ""),
                "source": PARAMETER_SOURCES.get(k, {}).get("source", "")}
            for k in values
        },
    }


# ══════════════════════════════════════════════════════════
# 毛细管突破压力
# ══════════════════════════════════════════════════════════
def entry_pressure_bar(ift_mnm: float, contact_angle_deg: float,
                       pore_throat_radius_nm: float) -> float:
    """
    Laplace/Washburn 突破压力（bar）：

        P_c = 2·γ·cosθ / r

    γ 用 mN/m（= 1e-3 N/m），r 用 nm（= 1e-9 m），结果 Pa 再转 bar。
    25 mN/m、30°、10 nm → 约 43 bar。泥质盖层实测进入压力 1–100 bar，
    量级一致。

    接触角 ≥ 90° 时 cosθ ≤ 0，表示非润湿相不需要额外压力就能进入 ——
    这在 CO₂–水体系里不会发生（水湿），但输入错时必须显式报错，
    而不是返回一个负数然后被下游当成"没有阻力"。
    """
    if pore_throat_radius_nm <= 0:
        raise ValueError("孔喉半径必须为正")
    theta = math.radians(float(contact_angle_deg))
    cos_t = math.cos(theta)
    if cos_t <= 0:
        raise ValueError(
            f"接触角 {contact_angle_deg}° ≥ 90°，cosθ ≤ 0：该体系不满足水湿假设"
        )
    gamma = float(ift_mnm) * 1e-3          # N/m
    r = float(pore_throat_radius_nm) * 1e-9  # m
    return 2.0 * gamma * cos_t / r / 1e5    # Pa → bar


def max_column_height_m(entry_bar: float, density_difference: float) -> float:
    """
    盖层能封住的 CO₂ 柱高上限（m）：

        H_max = P_c / (Δρ·g)

    Δρ 用真实 CO₂ 密度算（``utils/co2_properties.py``，Peng-Robinson）。
    用旧的 ``1.9·P/T`` 密度式会得到 Δρ ≈ 1029，柱高上限被高估 2.8 倍 ——
    这不是精度问题，是把"封不住"判成"封得住"。
    """
    if density_difference <= 0:
        raise ValueError("密度差必须为正（CO₂ 必须比地层水轻）")
    return float(entry_bar) * 1e5 / (density_difference * G)


# ══════════════════════════════════════════════════════════
# 主判定
# ══════════════════════════════════════════════════════════
def caprock_integrity(
    depth_m: float,
    injection_pressure_bar: float,
    temperature_celsius: float = 60.0,
    rho_brine: float = 1030.0,
    co2_column_height_m: float | None = None,
    injection_rate_m3_per_day: float | None = None,
    plume_radius_m: float = 500.0,
    reservoir_thickness_m: float = 50.0,
    porosity: float = 0.2,
    trap_area_m2: float | None = None,
    params: CaprockParams | None = None,
    stress_params: gm.StressParams | None = None,
    relperm_params: rp.RelPermParams | None = None,
    environ: dict | None = None,
) -> dict:
    """
    判断盖层会不会漏，以及（给定注入速率时）**什么时候**漏。

    ``co2_column_height_m`` 与 ``injection_rate_m3_per_day`` 二选一：
    - 给了柱高 → 静态判定（"这个柱高安不安全"）
    - 给了注入速率 → 动态判定（用一维驱替剖面算出柱高随时间增长，
      给出突破时间）

    两者都不给 → 只返回突破压力与柱高上限，``verdict="no_data"``，不阻断。

    返回：
        verdict           "sealed" | "capillary_breakthrough" | "mechanical_failure"
                          | "no_data"
        blocking          bool
        entry_pressure_bar, max_column_height_m
        column_height_m   本次评估用的柱高
        buoyancy_bar      该柱高产生的浮力
        margin_bar        P_c − P_b
        mechanical        {shmin_bar, limit_bar, margin_bar, ok}
        breakthrough_days 动态模式下的**前沿扫掠**时间（天）：CO₂ 前缘扫过
                          羽流半径所需时间。**这不是盖层突破时间** —— 盖层
                          突破看的是柱高长到 h_max，见 ``trap`` 的 fill_days。
                          两个事件量级差两个数量级，混用会把乐观的那个当结论。
        trap              {trap_area_m2, column_height_m, plume_column_height_m,
                          column_ratio, fill_days_early, fill_days_late,
                          governs, note}；未给 ``trap_area_m2`` 时为 None
        sensitivity       IFT 20/25/30 三档对应的 P_c 与 H_max
        assumptions / limitations / version / params
    """
    p = params or load_caprock_params(environ)
    sp = stress_params or gm.load_stress_params(environ)
    rpp = relperm_params or rp.load_relperm_params(environ)

    rho_co2 = cp.co2_density_kg_m3(injection_pressure_bar, temperature_celsius)
    drho = float(rho_brine) - rho_co2
    if drho <= 0:
        return {
            "verdict": "no_data",
            "blocking": False,
            "error": (
                f"CO₂ 密度 {rho_co2:.1f} kg/m³ ≥ 地层水密度 {rho_brine:.1f} kg/m³，"
                "浮力方向不成立；检查温度/压力/盐水密度输入"
            ),
            "version": CAPROCK_VERSION,
        }

    pc = entry_pressure_bar(p.ift_mnm, p.contact_angle_deg, p.pore_throat_radius_nm)
    h_max = max_column_height_m(pc, drho)

    # ── 力学失效：盖层被张性压开 ────────────────────────
    state = gm.stress_at_depth(depth_m, sp)
    mech_limit_bar = (state.shmin + p.tensile_strength_mpa) * 10.0   # MPa → bar
    mech_margin = mech_limit_bar - float(injection_pressure_bar)

    # ── 柱高：静态给定，或由注入速率动态推算 ────────────
    breakthrough_days = None
    column_height = co2_column_height_m
    dynamic_note = ""
    trap = None

    if column_height is None and injection_rate_m3_per_day:
        column_height, breakthrough_days, dynamic_note, eval_days = _dynamic_column(
            injection_rate_m3_per_day, plume_radius_m, reservoir_thickness_m,
            porosity, drho, pc, rpp,
        )
        if trap_area_m2:
            # 同一时刻、同一批物质，换一个面积口径：一维把 CO₂ 摊在 πr² 上，
            # 圈闭把它聚在 A_trap 上。两者是几何上下限，不是一个比另一个"准"。
            h_trap = trap_column_m(injection_rate_m3_per_day, trap_area_m2,
                                   porosity, rpp.swr, eval_days)
            # 柱高长到可封上限 h_max 需要多久，两个几何各算一次。
            # 一维口径的柱高其实不依赖孔隙度：H = q·t/A（体积除以面积），
            # 因为"等效柱高"已经把孔隙度折进去了 —— 这一点容易写错，
            # 写成 q·t/(A·φ) 会把时间高估 5 倍。
            plume_area = math.pi * float(plume_radius_m) ** 2
            days_trap_fill = trap_breakthrough_days(
                h_max, trap_area_m2, porosity, rpp.swr, injection_rate_m3_per_day)
            days_plume_fill = h_max * plume_area / float(injection_rate_m3_per_day)
            trap = {
                "trap_area_m2": round(float(trap_area_m2), 1),
                "column_height_m": round(h_trap, 2),
                "plume_column_height_m": round(float(column_height), 2),
                "column_ratio": round(h_trap / float(column_height), 2)
                if float(column_height) > 0 else None,
                "fill_days_early": round(days_trap_fill, 1),
                "fill_days_late": round(days_plume_fill, 1),
                "governs": "trap" if h_trap >= float(column_height) else "plume",
                "note": (
                    "圈闭口径是**保守**（更早突破）的一侧，一维口径是乐观的一侧；"
                    "若构造存在溢出点，柱高会被溢出高度截住，圈闭口径反而偏悲观。"
                    "两个 fill_days 是柱高长到可封上限所需时间，"
                    "与 breakthrough_days（CO₂ 前沿扫过羽流半径）不是同一个事件"
                ),
            }
            if h_trap > float(column_height):
                column_height = h_trap

    if column_height is None:
        return {
            "verdict": "no_data",
            "blocking": False,
            "entry_pressure_bar": round(pc, 2),
            "max_column_height_m": round(h_max, 1),
            "co2_density_kg_m3": round(rho_co2, 1),
            "density_difference": round(drho, 1),
            "mechanical": {
                "shmin_bar": round(state.shmin * 10.0, 1),
                "limit_bar": round(mech_limit_bar, 1),
                "margin_bar": round(mech_margin, 1),
                "ok": mech_margin >= 0,
            },
            "summary": (
                f"盖层突破压力 {pc:.0f} bar，可封 CO₂ 柱高 {h_max:.0f} m；"
                "未给出柱高或注入速率，未做封闭性判定"
            ),
            "sensitivity": _sensitivity(p, drho),
            "assumptions": _assumed(environ),
            "limitations": _limitations(),
            "version": CAPROCK_VERSION,
            "params": caprock_manifest(p, environ)["parameters"],
        }

    buoyancy = drho * G * float(column_height) / 1e5      # bar
    margin = pc - buoyancy

    if mech_margin < 0:
        verdict, blocking = "mechanical_failure", True
    elif margin <= 0:
        verdict, blocking = "capillary_breakthrough", True
    else:
        verdict, blocking = "sealed", False

    return {
        "verdict": verdict,
        "blocking": blocking,
        "entry_pressure_bar": round(pc, 2),
        "max_column_height_m": round(h_max, 1),
        "column_height_m": round(float(column_height), 2),
        "buoyancy_bar": round(buoyancy, 2),
        "margin_bar": round(margin, 2),
        "co2_density_kg_m3": round(rho_co2, 1),
        "density_difference": round(drho, 1),
        "mechanical": {
            "shmin_bar": round(state.shmin * 10.0, 1),
            "limit_bar": round(mech_limit_bar, 1),
            "margin_bar": round(mech_margin, 1),
            "ok": mech_margin >= 0,
        },
        "breakthrough_days": breakthrough_days,
        "trap": trap,
        "dynamic_note": dynamic_note,
        "summary": _summary(verdict, pc, h_max, column_height, margin, mech_margin),
        "sensitivity": _sensitivity(p, drho),
        "assumptions": _assumed(environ),
        "limitations": _limitations(),
        "version": CAPROCK_VERSION,
        "params": caprock_manifest(p, environ)["parameters"],
    }


def _dynamic_column(rate_m3_per_day: float, plume_radius_m: float,
                    thickness_m: float, porosity: float, drho: float,
                    pc_bar: float, rpp: rp.RelPermParams) -> tuple[float, float, str, float]:
    """
    由注入速率推算当前柱高与突破时间。

    几何假设：径向羽流，等效为一维线性驱替（面积 A = πr²，长度 L）。
    这是把径向问题压成一维的常用简化，代价是忽略了近井高速区与
    远场低速区的差异。写进 note，不当作事实。

    第四个返回值是**这个柱高对应的物理时刻**（天）。没有它，圈闭口径
    就只能拿另一个时刻的柱高来比，比出来的差值是几何差异还是时间差异
    就分不清了。
    """
    area = math.pi * float(plume_radius_m) ** 2
    pore_volume = area * float(thickness_m) * float(porosity)
    # 突破时的无因次时间（解析前沿速度的倒数）
    front = rp.bl_front(rpp)
    t_breakthrough_d = 1.0 / front["dfg_front"] if front["dfg_front"] > 0 else 1.0
    days_to_breakthrough = t_breakthrough_d * pore_volume / float(rate_m3_per_day)

    # 当前（突破前）柱高：用驱替剖面折算
    t_eval = min(0.9 * t_breakthrough_d, 0.49)
    profile = flow1d.solve(n_cells=200, t_end=t_eval, params=rpp)
    h = flow1d.co2_column_height(profile["sg"], 1.0 / 200, thickness_m, porosity)

    note = (
        f"按径向羽流半径 {plume_radius_m:.0f} m、厚 {thickness_m:.0f} m、"
        f"孔隙度 {porosity:.2f} 折算为一维线性驱替；"
        f"突破时间按解析前沿速度 {front['dfg_front']:.2f} 计"
    )
    return h, round(days_to_breakthrough, 1), note, t_eval * pore_volume / float(rate_m3_per_day)


def trap_column_m(rate_m3_per_day: float, trap_area_m2: float, porosity: float,
                  swr: float, days: float) -> float:
    """
    圈闭聚集口径下的 CO₂ 柱高（m）：

        h(t) = q·t / (A_trap·φ·(1−S_wr))

    与一维口径的区别不在公式，在**面积**：一维口径把 CO₂ 摊在 πr² 的整个
    羽流范围里，圈闭口径把它聚在一个封闭构造上。真实情况在两者之间，
    构造越紧闭越靠近后者。这是同一批物质、同一个时刻的两个几何极限，
    不是两个模型。

    ``φ(1−S_wr)`` 是 CO₂ 能占的孔隙体积分数：驱替把水推到束缚水，
    所以侵入区的含气饱和度上限就是 1−S_wr。
    """
    if trap_area_m2 <= 0:
        raise ValueError("圈闭面积必须为正")
    accessible = float(porosity) * (1.0 - float(swr))
    if accessible <= 0:
        raise ValueError("φ(1−S_wr) 必须为正：检查孔隙度与束缚水饱和度")
    return float(rate_m3_per_day) * float(days) / (float(trap_area_m2) * accessible)


def trap_breakthrough_days(h_max_m: float, trap_area_m2: float, porosity: float,
                           swr: float, rate_m3_per_day: float) -> float:
    """圈闭口径下柱高长到 ``h_max`` 需要多少天（即毛细管突破的**早**界）。"""
    if rate_m3_per_day <= 0:
        raise ValueError("注入速率必须为正")
    accessible = float(porosity) * (1.0 - float(swr))
    return float(h_max_m) * float(trap_area_m2) * accessible / float(rate_m3_per_day)


def _sensitivity(p: CaprockParams, drho: float) -> list[dict]:
    """
    IFT 三档敏感性。界面张力是全模块第二敏感的参数（仅次于孔喉半径），
    给三档比给一个"精确"值诚实。
    """
    out = []
    for ift in (20.0, 25.0, 30.0):
        pc = entry_pressure_bar(ift, p.contact_angle_deg, p.pore_throat_radius_nm)
        out.append({
            "ift_mnm": ift,
            "entry_pressure_bar": round(pc, 2),
            "max_column_height_m": round(max_column_height_m(pc, drho), 1),
        })
    return out


def _summary(verdict, pc, h_max, column, margin, mech_margin) -> str:
    if verdict == "mechanical_failure":
        return (
            f"注入压力已超过盖层力学失效界限（余量 {mech_margin:.0f} bar），"
            f"盖层可能被张性压开"
        )
    if verdict == "capillary_breakthrough":
        return (
            f"CO₂ 柱高 {column:.0f} m 超过盖层可封高度 {h_max:.0f} m，"
            f"浮力超出突破压力 {-margin:.0f} bar，存在毛细管渗漏"
        )
    return (
        f"CO₂ 柱高 {column:.0f} m 低于盖层可封高度 {h_max:.0f} m，"
        f"浮力余量 {margin:.0f} bar（突破压力 {pc:.0f} bar）"
    )


def _limitations() -> list[str]:
    return [
        "完整盖层假设：不考虑裂缝、砂条带等非均质泄漏通道（实际泄漏的主因）。",
        "界面张力不建模为 T/P 函数，用文献中值 + 三档敏感性。",
        "孔喉半径若来自默认值而非 MICP 实测，结论只能算假设下的量级判断。",
        # 方向性偏差必须写清楚：不是"可能有误差"，而是"误差朝乐观那一侧"
        "**未含重力分异时，一维口径把 CO₂ 摊在 πr² 的羽流范围里 → 柱高被低估 → "
        "突破时间偏长**：真实 CO₂ 会上浮并在盖层下聚集。给 ``trap_area_m2`` 时"
        "同时算圈闭口径，两个口径构成突破时间的区间（圈闭为早界、一维为晚界）；"
        "**不给圈闭面积时只剩乐观的那一侧**，此时的突破时间是上限，不能当安全裕度。",
    ]


def render_block(result: dict) -> str:
    """渲染成报告可用文字。"""
    if not result or result.get("verdict") in (None, "no_data"):
        if result.get("entry_pressure_bar"):
            return (
                f"【盖层封闭性（{result['version']}）】"
                f"突破压力 {result['entry_pressure_bar']:.0f} bar，"
                f"可封 CO₂ 柱高 {result['max_column_height_m']:.0f} m"
                "（未给柱高或注入速率，未做判定）"
            )
        return ""

    head = {
        "sealed": "盖层封闭（毛细管与力学均未失效）",
        "capillary_breakthrough": "盖层毛细管突破（浮力超过突破压力）",
        "mechanical_failure": "盖层力学失效（注入压力超过 σhmin + 抗张强度）",
    }[result["verdict"]]
    mech = result["mechanical"]
    lines = [
        f"【盖层封闭性（{result['version']}）】{head}",
        f"- 突破压力 {result['entry_pressure_bar']:.0f} bar，"
        f"可封 CO₂ 柱高 {result['max_column_height_m']:.0f} m",
        f"- 本次 CO₂ 柱高 {result['column_height_m']:.0f} m → "
        f"浮力 {result['buoyancy_bar']:.1f} bar，余量 {result['margin_bar']:.1f} bar",
        f"- 力学界限 {mech['limit_bar']:.0f} bar（σhmin {mech['shmin_bar']:.0f} + "
        f"抗张 {mech['limit_bar'] - mech['shmin_bar']:.0f}），余量 {mech['margin_bar']:.0f} bar",
        f"- 密度差 Δρ = {result['density_difference']:.0f} kg/m³"
        f"（CO₂ {result['co2_density_kg_m3']:.0f} kg/m³，Peng-Robinson）",
    ]
    if result.get("breakthrough_days") is not None:
        lines.append(
            f"- 按给定注入速率，CO₂ 前缘约 {result['breakthrough_days']:.0f} 天扫过羽流边界"
            "（前沿扫掠，不是盖层突破）"
        )
    trap = result.get("trap") or {}
    if trap:
        lines.append(
            f"- 圈闭口径（面积 {trap['trap_area_m2']:.0f} m²）：同一时刻柱高 "
            f"{trap['column_height_m']:.0f} m（一维口径 {trap['plume_column_height_m']:.0f} m，"
            f"{trap['column_ratio']}×）；柱高长到可封上限需 "
            f"{trap['fill_days_early']:.0f} 天，一维口径需 {trap['fill_days_late']:.0f} 天，"
            f"早界为保守侧"
        )
    if result["assumptions"]:
        lines.append(
            "- 前提：以下参数用了默认假设值，不是实测 —— "
            + "、".join(result["assumptions"])
        )
    sens = result.get("sensitivity") or []
    if sens:
        lines.append(
            "- 界面张力敏感性：" + "；".join(
                f"{s['ift_mnm']:.0f} mN/m → {s['entry_pressure_bar']:.0f} bar / "
                f"{s['max_column_height_m']:.0f} m" for s in sens
            )
        )
    return "\n".join(lines)
