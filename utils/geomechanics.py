"""
地应力状态与断层活化判定（确定性计算，不是模型判断）
=====================================================
``physics_rules`` 只回答一个问题：注入压力有没有超过 1.5× 静水。这足以拦住
"注入压力 400 bar、埋深 2500 m 还写风险可控"这种荒谬报告，但它不知道**断层**。
CCUS 项目里最常见的诱发地震与泄漏通道不是"把盖层压出裂缝"，而是"注入把孔隙
压力抬上去，作用在既有断层面上的有效正应力降下来，断层开始滑动"—— 后者在
1.5× 静水这条线**以下**就能发生。

所以这个模块补的是另一条判据，全部来自岩石力学教科书，没有可调的经验系数：

    有效正应力     σ'_n = σ_n − P_p                    （Terzaghi）
    Mohr-Coulomb   τ   = C + μ·σ'_n                    （滑动判据）
    临界孔隙压力   P_p,crit = σ_n − (τ − C)/μ
    摩擦平衡上限   (σ1 − P_p)/(σ3 − P_p) ≤ (√(μ²+1) + μ)²

最后一条（Zoback 2007, eq. 4.6；Jaeger & Cook & Zimmerman 2007）是"应力多边形"
的理论边界：如果一组输入的应力状态本身就违反它，说明那组数据里至少有一个分量
测错了 —— 地壳里不可能长期存在超过摩擦强度的差应力。这个检查在别处很少做，
但它是唯一能**从内部**发现"输入应力数据不自洽"的手段。

## 单位

模块内部一律 **MPa**（岩石力学的通用单位），深度 m，梯度 MPa/km。
返回字典里压力同时给 ``*_mpa`` 与 ``*_bar`` 两个键，**后缀必写** ——
本项目其余部分用 bar，混用是这类代码最常见的 bug 来源。

## 这个模块不是什么

- 不是三维应力场反演。输入是一维的（梯度或实测值），不做井壁崩落/诱导缝反演。
- 不做**应力转移**。``utils/poroelastic.py`` 给了注入层内部的单轴应变应力
  路径（总应力随孔隙压力上升），但它只在注入层成立；盖层与隔层的应力变化
  由三维孔隙弹性解决定，可以在某些位置**降低** σ_h 从而促进滑动，本项目
  没有地质力学求解器，不建模。因此判定闸门仍走"总应力不变"的保守口径，
  耦合口径作为第二个数返回（见 ``coupled`` 字段）。
"""
from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass

from utils import poroelastic as pe

# ─── 版本 ────────────────────────────────────────────────
# 与 physics_rules.RULES_VERSION 分开：两套判据的改动节奏不同，
# 合成一个版本号会让"这次报告为什么变了"无法回答。
GEOMECHANICS_VERSION = "2026.10-g1"

G = 9.81
MPA_PER_BAR = 0.1          # 1 bar = 0.1 MPa
BAR_PER_MPA = 10.0


# ══════════════════════════════════════════════════════════
# 参数
# ══════════════════════════════════════════════════════════
@dataclass(frozen=True)
class StressParams:
    """一维应力模型参数。

    **前三个梯度是假设值，不是测量值。** 有实测数据（DFIT、漏失试验、
    声波+密度反演）必须用环境变量覆盖，否则报告里的"断层稳定"结论
    只是对假设值的复述。返回结果里的 ``assumed`` 字段会列出哪些参数
    用了默认值 —— 报告必须据此写"基于假设的应力剖面"，不能写成实测结论。

    覆盖方式::

        CCUS_SV_GRADIENT=23.5      # MPa/km
        CCUS_PP_GRADIENT=10.1      # MPa/km
        CCUS_SHMIN_RATIO=0.72      # Shmin / Sv
        CCUS_SHMAX_RATIO=0.88      # SHmax / Sv
        CCUS_FRICTION=0.6          # 断层面摩擦系数
        CCUS_COHESION=0.0          # MPa，断层面黏聚力
    """

    sv_gradient: float = 23.0        # MPa/km，上覆压力梯度
    pp_gradient: float = 10.1        # MPa/km，孔隙压力梯度（静水）
    shmin_ratio: float = 0.75        # Shmin / Sv
    shmax_ratio: float = 0.90        # SHmax / Sv
    friction: float = 0.6            # μ
    cohesion: float = 0.0            # MPa


PARAMETER_SOURCES = {
    "sv_gradient": {
        "unit": "MPa/km",
        "source": (
            "上覆压力梯度。Zoback (2007, Reservoir Geomechanics, Table 4.1) 给出"
            "沉积盆地典型 22–25 MPa/km；取 23 为代表值。密度积分会更准，但需要"
            "逐层密度测井，本项目没有。"
        ),
    },
    "pp_gradient": {
        "unit": "MPa/km",
        "source": (
            "静水孔隙压力梯度 = ρ_w·g = 1030 × 9.81 / 1e6 × 1000 = 10.1 MPa/km，"
            "与 physics_rules.RHO_WATER 同源。异常压力区块必须覆盖。"
        ),
    },
    "shmin_ratio": {
        "unit": "× Sv",
        "source": (
            "**假设值**。正断层区 Shmin/Sv 多为 0.6–0.9（Zoback 2007, §4）。"
            "取 0.75 代表松驰盆地。实测手段：DFIT / 小型压裂闭合压力。"
        ),
    },
    "shmax_ratio": {
        "unit": "× Sv",
        "source": (
            "**假设值**。正断层区 SHmax 接近但小于 Sv，典型 0.85–0.95；取 0.90。"
            "实测手段：井壁崩落宽度反演、诱导裂缝方位。"
        ),
    },
    "friction": {
        "unit": "无量纲",
        "source": (
            "Byerlee (1978, Pure Appl. Geophys. 116) 的岩石摩擦定律："
            "正应力 < 200 MPa 时 μ ≈ 0.85，> 200 MPa 时 ≈ 0.6。"
            "本模块取 0.6 作为**保守下限** —— 摩擦越小越容易滑动，"
            "所以取下限意味着判定偏安全侧（宁可误报活化，不漏报）。"
        ),
    },
    "cohesion": {
        "unit": "MPa",
        "source": (
            "断层面黏聚力按 0 处理：既有断层面已发生过剪切，胶结基本破坏，"
            "这是诱发地震分析的标准假设（Rutqvist 2012, Geotech. Geol. Eng. 30）。"
            "完整岩体的 C 不应为 0，但这个模块只判断层活化，不判完整岩石破裂。"
        ),
    },
}


def load_stress_params(environ: dict | None = None) -> StressParams:
    """读环境变量；非法或越界回退默认（与 ``physics_rules.load_rule_params`` 同策略）。

    越界一律回退而不是截断：``CCUS_FRICTION=6`` 是漏了小数点，截断成 1.0
    会悄悄改变所有结论；回退到 0.6 至少是"用户没配"的语义，并且
    ``assumed`` 会把它列出来。
    """
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

    return StressParams(
        sv_gradient=_num("CCUS_SV_GRADIENT", 23.0, 15.0, 32.0),
        pp_gradient=_num("CCUS_PP_GRADIENT", 10.1, 4.0, 22.0),
        shmin_ratio=_num("CCUS_SHMIN_RATIO", 0.75, 0.3, 1.6),
        shmax_ratio=_num("CCUS_SHMAX_RATIO", 0.90, 0.3, 1.8),
        friction=_num("CCUS_FRICTION", 0.6, 0.2, 1.2),
        cohesion=_num("CCUS_COHESION", 0.0, 0.0, 30.0),
    )


def _assumed(params: StressParams, environ: dict | None = None) -> list[str]:
    """列出哪些参数**没有**被环境变量覆盖 —— 报告必须据此声明结论的前提。"""
    env = os.environ if environ is None else environ
    keys = {
        "sv_gradient": "CCUS_SV_GRADIENT",
        "pp_gradient": "CCUS_PP_GRADIENT",
        "shmin_ratio": "CCUS_SHMIN_RATIO",
        "shmax_ratio": "CCUS_SHMAX_RATIO",
        "friction": "CCUS_FRICTION",
        "cohesion": "CCUS_COHESION",
    }
    return [field for field, var in keys.items() if not (env.get(var) or "").strip()]


def stress_manifest(params: StressParams | None = None,
                    environ: dict | None = None) -> dict:
    """生效参数 + 出处 + 哪些是假设。写进报告，便于审计。"""
    params = params or load_stress_params(environ)
    values = asdict(params)
    return {
        "version": GEOMECHANICS_VERSION,
        "assumed": _assumed(params, environ),
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
# 应力状态
# ══════════════════════════════════════════════════════════
@dataclass(frozen=True)
class StressState:
    """某一深度的主应力状态（MPa）。σ1 ≥ σ2 ≥ σ3 由 magnitude 排序决定。"""

    depth_m: float
    sv: float       # 垂向主应力（≈上覆压力）
    shmin: float    # 最小水平主应力
    shmax: float    # 最大水平主应力
    pp: float       # 孔隙压力
    regime: str     # "NF" | "SS" | "RF"

    def as_dict(self) -> dict:
        return {
            "depth_m": round(self.depth_m, 1),
            "sv_mpa": round(self.sv, 2), "sv_bar": round(self.sv * BAR_PER_MPA, 1),
            "shmin_mpa": round(self.shmin, 2), "shmin_bar": round(self.shmin * BAR_PER_MPA, 1),
            "shmax_mpa": round(self.shmax, 2), "shmax_bar": round(self.shmax * BAR_PER_MPA, 1),
            "pp_mpa": round(self.pp, 2), "pp_bar": round(self.pp * BAR_PER_MPA, 1),
            "regime": self.regime,
        }


def classify_regime(sv: float, shmin: float, shmax: float) -> str:
    """
    Anderson (1951) 断层类型判别：看哪个主应力是**垂向**的。

    - ``NF`` 正断层：Sv 最大（σ1 = Sv）
    - ``SS`` 走滑：SHmax 最大，Sv 居中（σ2 = Sv）
    - ``RF`` 逆断层：Shmin 最大（σ3 = Sv）

    等号归到"上一个"分支，保证同一组输入永远得到同一个字符串 ——
    判定结果要写进报告，不能因为浮点相等与否而漂移。
    """
    if sv >= shmax >= shmin:
        return "NF"
    if shmax >= sv >= shmin:
        return "SS"
    return "RF"


def stress_at_depth(depth_m: float, params: StressParams | None = None) -> StressState:
    """按梯度算某一深度的应力状态。"""
    p = params or load_stress_params()
    d_km = float(depth_m) / 1000.0
    sv = p.sv_gradient * d_km
    pp = p.pp_gradient * d_km
    shmin = sv * p.shmin_ratio
    shmax = sv * p.shmax_ratio
    return StressState(
        depth_m=float(depth_m), sv=sv, shmin=shmin, shmax=shmax, pp=pp,
        regime=classify_regime(sv, shmin, shmax),
    )


def frictional_limit(friction: float) -> float:
    """摩擦平衡的应力比上限 K = (√(μ²+1) + μ)²。

    μ=0.6 → 3.12；μ=0.85 → 4.29。差应力一旦超过 K 倍，地壳里就不可能存在
    这样一组应力而断层不动。
    """
    return (math.sqrt(friction * friction + 1.0) + friction) ** 2


def stress_polygon(state: StressState, params: StressParams | None = None) -> dict:
    """
    检查这组应力是否落在摩擦平衡边界内（Zoback 2007, eq. 4.6）。

        (σ1 − P_p) / (σ3 − P_p) ≤ K

    违反它的含义**不是**"断层要滑了"，而是"这组输入数据不自洽" ——
    多半是 SHmax 被高估，或者孔隙压力被低估。这一点很重要：报告里如果
    直接引用这组应力做安全结论，那个结论的输入本身就有问题。

    返回 ``ratio`` / ``limit`` / ``ok`` / ``excess``（超出倍数）。
    """
    p = params or load_stress_params()
    s1 = max(state.sv, state.shmax, state.shmin)
    s3 = min(state.sv, state.shmax, state.shmin)
    k = frictional_limit(p.friction)
    denom = s3 - state.pp
    if denom <= 0:
        # 孔隙压力达到或超过最小主应力：有效应力归零，任何面都能张开
        return {
            "ratio": float("inf"), "limit": round(k, 3), "ok": False,
            "excess": float("inf"),
            "note": "孔隙压力已达到最小主应力，有效正应力归零",
        }
    ratio = (s1 - state.pp) / denom
    return {
        "ratio": round(ratio, 3),
        "limit": round(k, 3),
        "ok": ratio <= k,
        "excess": round(ratio / k, 3),
        "note": "" if ratio <= k else "输入应力状态超过摩擦平衡上限，数据不自洽",
    }


# ══════════════════════════════════════════════════════════
# 断层面上的应力
# ══════════════════════════════════════════════════════════
def plane_stress(dip_deg: float, azimuth_deg: float, state: StressState) -> dict:
    """
    给定产状的断层面上的正应力与剪应力（MPa）。

    坐标系：x 沿 SHmax，y 沿 Shmin，z 垂向向上。``azimuth_deg`` 是**倾向**
    相对 SHmax 方向的夹角（0° = 朝 SHmax 方向倾，90° = 朝 Shmin 方向倾）。

    法向量 n = (sinδ·cosφ, sinδ·sinφ, cosδ)，于是

        σ_n = σx·nx² + σy·ny² + σz·nz²
        τ   = |t|² − σ_n²  的平方根，其中 t = (σx·nx, σy·ny, σz·nz)

    为什么用三维而不是课本里的二维公式 ``σn = σv·cos²δ + σh·sin²δ``：
    二维式只对"走向垂直于水平主应力"的断层成立。报告里写"断层倾角 60°"
    时往往不给走向，如果强行套二维式，等于默认了最不利走向 —— 那会系统性
    高估风险。三维式在 φ=90° 时与二维式完全一致（有测试锁定），
    但在其它走向下给出正确（更小）的剪应力。
    """
    d = math.radians(float(dip_deg))
    a = math.radians(float(azimuth_deg))
    nx = math.sin(d) * math.cos(a)
    ny = math.sin(d) * math.sin(a)
    nz = math.cos(d)

    sigma_n = state.shmax * nx * nx + state.shmin * ny * ny + state.sv * nz * nz
    tx = state.shmax * nx
    ty = state.shmin * ny
    tz = state.sv * nz
    t_mag2 = tx * tx + ty * ty + tz * tz
    tau2 = max(t_mag2 - sigma_n * sigma_n, 0.0)   # 浮点误差可能给出 -1e-16
    return {
        "dip_deg": float(dip_deg),
        "azimuth_deg": float(azimuth_deg),
        "sigma_n_mpa": sigma_n,
        "tau_mpa": math.sqrt(tau2),
        "normal": (nx, ny, nz),
    }


def optimal_dip(state: StressState, params: StressParams | None = None,
                azimuth_deg: float = 90.0) -> float:
    """
    给定倾向，扫描出**随孔隙压力上升最先滑动**的倾角（度）。

    这里有个容易混的地方，值得写清楚 —— 两个"最不利产状"不是同一个角：

    1. **当前最接近滑动**的面：使滑移趋势 τ/(σn − Pp) 最大。它依赖当前的 Pp。
    2. **随压力上升最先滑的面**：使临界孔隙压力 σn − τ/μ 最小，等价于使
       Coulomb 破坏函数 τ − μ·σn 最大。它**不依赖** Pp，是应力状态的固有属性。

    工程上关心的是第 2 个（"压力再加多少会滑"），所以本函数取 argmin Pp_crit。
    经典结论是 45° + φ/2，φ = arctan μ；μ=0.6 时 ≈ 60.5°，与正断层实际
    倾角 60° 一致。若误用第 1 个目标，正断层区会算出 53° 左右 —— 差值不小，
    而且**不会报错**，只是把最不利产状算轻了。

    扫描而非解析求解：解析式对每个应力区间有不同形式，写错一个符号不会报错、
    只会给出错误答案。步长 0.5° 对工程判定足够。
    """
    p = params or load_stress_params()
    best_dip, best_crit = 0.0, float("inf")
    dip = 1.0
    while dip < 90.0:
        pl = plane_stress(dip, azimuth_deg, state)
        crit = critical_pore_pressure(state, pl["sigma_n_mpa"], pl["tau_mpa"], p)
        if crit < best_crit:
            best_dip, best_crit = dip, crit
        dip += 0.5
    return best_dip


def critical_pore_pressure(state: StressState, sigma_n: float, tau: float,
                           params: StressParams | None = None) -> float:
    """
    让指定面上的断层开始滑动所需的孔隙压力（MPa）。

        τ = C + μ(σ_n − P_p)   →   P_p,crit = σ_n − (τ − C)/μ

    ``τ < C`` 时无论孔隙压力多大都不会滑（黏聚力把它锁住了），返回 ``inf``。
    """
    p = params or load_stress_params()
    if tau <= p.cohesion:
        return float("inf")
    return sigma_n - (tau - p.cohesion) / p.friction


def critical_pore_pressure_optimal(state: StressState,
                                   params: StressParams | None = None) -> float:
    """
    最优产状断层的临界孔隙压力（MPa），用摩擦平衡式反解：

        (σ1 − P_p)/(σ3 − P_p) = K
        ⇒ P_p,crit = (K·σ3 − σ1) / (K − 1)

    与 ``critical_pore_pressure`` 在最优倾角处必须一致（测试锁定）。
    两条独立路径算同一个量，是这个模块唯一可靠的正确性证据 ——
    没有实测数据可对，就只能靠解析式互相验证。
    """
    p = params or load_stress_params()
    s1 = max(state.sv, state.shmax, state.shmin)
    s3 = min(state.sv, state.shmax, state.shmin)
    k = frictional_limit(p.friction)
    if k <= 1.0:
        return float("inf")
    return (k * s3 - s1) / (k - 1.0)


# ══════════════════════════════════════════════════════════
# 主判定：注入引起的断层活化
# ══════════════════════════════════════════════════════════
def _severity(verdict: str) -> int:
    """判定的严重度序，用于比较两个口径谁更宽松。"""
    return {"stable": 0, "near_critical": 1, "reactivated": 2,
            "tensile_opening": 3}.get(verdict, 0)


def _coupled_path(state: StressState, delta_pp_mpa: float, dip_deg: float,
                  azimuth_deg: float, params: StressParams,
                  poro: pe.PoroelasticParams | None,
                  environ: dict | None) -> dict:
    """
    单轴应变孔隙弹性口径下的同一套判据（见 ``utils/poroelastic.py``）。

    与保守口径的唯一区别是：注入后**总应力也变了**，所以 σn 与 τ 都要用
    新的应力状态重算，而不是只把 Pp 换掉。重算而不是套修正系数，是因为
    剪应力 τ = √(s(1−s))·|σh − σv| 会随 σh 上升而下降 —— 只改 Pp 会漏掉
    这一项，而它和正应力项同向，漏掉就等于把耦合的好处算了一半。
    """
    pp = poro or pe.load_poroelastic_params(environ)
    after = pe.stress_after_injection(
        state.sv, state.shmin, state.shmax, state.pp, delta_pp_mpa, pp
    )
    coupled_state = StressState(
        depth_m=state.depth_m, sv=after["sv"], shmin=after["shmin"],
        shmax=after["shmax"], pp=after["pp"],
        regime=classify_regime(after["sv"], after["shmax"], after["shmin"]),
    )
    plane = plane_stress(dip_deg, azimuth_deg, coupled_state)
    sigma_n = plane["sigma_n_mpa"]
    tau = plane["tau_mpa"]
    eff = sigma_n - coupled_state.pp

    if eff <= 0:
        slip_tendency = float("inf")
    elif tau <= params.cohesion:
        slip_tendency = 0.0
    else:
        slip_tendency = tau / eff
    slip_ratio = (slip_tendency / params.friction
                  if slip_tendency != float("inf") else float("inf"))
    pp_crit = critical_pore_pressure(coupled_state, sigma_n, tau, params)
    # 与保守口径同一套阈值语义：余量从**原始**孔隙压力算起，吃掉 80% 报 near_critical
    margin_from_original = pp_crit - state.pp

    if coupled_state.pp >= sigma_n:
        verdict = "tensile_opening"
    elif slip_ratio >= 1.0:
        verdict = "reactivated"
    elif margin_from_original != float("inf") and margin_from_original > 0 \
            and delta_pp_mpa >= 0.8 * margin_from_original:
        verdict = "near_critical"
    else:
        verdict = "stable"

    return {
        "verdict": verdict,
        "slip_ratio": _round(slip_ratio),
        "sigma_n_mpa": round(sigma_n, 2),
        "sigma_n_eff_mpa": round(eff, 2),
        "tau_mpa": round(tau, 2),
        "pp_crit_bar": _round(pp_crit * BAR_PER_MPA),
        "margin_bar": _round((pp_crit - coupled_state.pp) * BAR_PER_MPA),
        "delta_sigma_h_bar": round(after["delta_sigma_h_mpa"] * BAR_PER_MPA, 1),
        "path_ratio": round(after["path_ratio"], 4),
        "effective_path_ratio": round(after["effective_path_ratio"], 4),
        # 输入压力低于原始孔隙压力时，应力路径把 σh 往下带，耦合口径反而更严重。
        # 这不是异常输入（枯竭工况就是这样），但方向必须显式标出来，
        # 否则"耦合口径更宽松"这句话会被无条件套用。
        "depressurisation": delta_pp_mpa < 0,
        "state": coupled_state.as_dict(),
        "params": pe.manifest(pp, environ)["parameters"],
        "assumed": pe._assumed(environ),
        "note": (
            "单轴应变应力路径只对注入层本身成立；盖层与隔层的应力转移未建模，"
            "那部分可以在局部降低水平应力、促进滑动，所以这个口径的余量"
            "不能当作场地安全裕度"
            + ("；本次输入压力低于原始孔隙压力，应力路径反向（σh 下降），"
               "耦合口径比保守口径更严重" if delta_pp_mpa < 0 else "")
        ),
    }


def fault_reactivation(depth_m: float, injection_pressure_bar: float,
                       dip_deg: float = 60.0, azimuth_deg: float = 90.0,
                       params: StressParams | None = None,
                       environ: dict | None = None,
                       poro: pe.PoroelasticParams | None = None) -> dict:
    """
    注入压力抬升孔隙压力后，一条给定产状的断层会不会滑动。

    这是 CCUS 里比"超破裂压力"更早触发的风险：断裂带附近不需要把盖层压裂，
    只要孔隙压力抬到临界值，断层就会以微震/滑移的形式释放，既可能形成
    向上泄漏通道，也可能损伤井筒。

    判定用**两个等价口径**同时算，互为交叉验证：

    1. 滑移比 ``Ts/μ``（τ 与 μ·σ'_n 的比）≥ 1 → 滑动
    2. 注入附加孔隙压力 ``ΔP`` 与活化余量 ``P_p,crit − P_p`` 比较

    两者数学上等价（是同一个不等式的两种写法），如果一个 ≥1 另一个没超限，
    说明代码有 bug —— 所以结果里两个都给，并且 ``consistent`` 字段做断言。

    返回：
        {
          "verdict": "stable" | "near_critical" | "reactivated" | "no_data",
          "blocking": bool,
          "state": {...}, "plane": {...},
          "slip_ratio": float, "slip_tendency": float,
          "pp_crit_mpa": float, "pp_crit_bar": float,
          "margin_mpa": float, "margin_bar": float,
          "applied_overpressure_bar": float,
          "consistent": bool,
          "polygon": {...},
          "coupled": {...},          # 单轴应变孔隙弹性口径，见 utils/poroelastic.py
          "assumptions": [...], "limitations": [...],
          "version": "2026.10-g1",
          "params": {...},
        }
    """
    p = params or load_stress_params(environ)
    state = stress_at_depth(depth_m, p)
    plane = plane_stress(dip_deg, azimuth_deg, state)

    sigma_n = plane["sigma_n_mpa"]
    tau = plane["tau_mpa"]
    sigma_n_eff = sigma_n - state.pp
    pp_crit = critical_pore_pressure(state, sigma_n, tau, p)

    p_inj_mpa = float(injection_pressure_bar) * MPA_PER_BAR
    applied = p_inj_mpa - state.pp          # 注入带来的附加孔隙压力
    margin = pp_crit - state.pp             # 到活化还差多少

    # ── 口径 1：滑移比，必须在**注入后**的孔隙压力下评估 ──
    # 这是最容易写错的一处：用原地压力算出的滑移比只说明"现在稳不稳"，
    # 与"注入后会不会滑"是两个问题。判据必须用 P_inj。
    def _slip_tendency_at(pp: float) -> float:
        eff = sigma_n - pp
        if eff <= 0:
            return float("inf")
        if tau <= p.cohesion:
            return 0.0
        return tau / eff

    ts_current = _slip_tendency_at(state.pp)
    ts_injected = _slip_tendency_at(p_inj_mpa)
    slip_ratio = ts_injected / p.friction if ts_injected != float("inf") else float("inf")

    # ── 张性开启：与剪切滑动是两种机制，必须分开报 ──────────
    # 孔隙压力超过断层面上的正应力时，面是被"顶开"的 —— 此时黏聚力、
    # 摩擦系数都不再是控制因素。混进滑移判据会让高黏聚力断层在高压下
    # 仍被判成"稳定"，那是错的（面已经张开了，流体有通道）。
    tensile_opening = p_inj_mpa >= sigma_n

    # ── 口径 2：附加压力 vs 余量 ─────────────────────────
    if margin == float("inf"):
        pressure_triggered = False
    else:
        pressure_triggered = applied >= margin

    slip_triggered = slip_ratio >= 1.0
    # 两个口径数学上等价（同一不等式的两种写法）。浮点边界上可能差一个 ulp，
    # 用相对容差放过；真的不等说明代码有 bug，consistent=False 会暴露它。
    consistent = (slip_triggered == pressure_triggered) or (
        margin != float("inf") and abs(applied - margin) <= 1e-6 * max(1.0, abs(margin))
    )

    if tensile_opening:
        verdict, blocking = "tensile_opening", True
        # 张性开启时剪切判据不适用（摩擦/黏聚力都不是控制因素），
        # 两个口径的一致性是"不适用"而不是"不一致"。
        consistent = True
        mechanism = "tensile"
    elif slip_triggered or pressure_triggered:
        verdict, blocking = "reactivated", True
        mechanism = "shear"
    elif margin != float("inf") and margin > 0 and applied >= 0.8 * margin:
        # 余量吃掉 80% 以上：还没滑，但已经进入"稍有波动就滑"的区间。
        # 不阻断（没超就是没超），但报告必须写明余量。
        verdict, blocking = "near_critical", False
        mechanism = "shear"
    else:
        verdict, blocking = "stable", False
        mechanism = "shear"

    polygon = stress_polygon(state, p)
    coupled = _coupled_path(state, applied, dip_deg, azimuth_deg, p, poro, environ)

    return {
        "verdict": verdict,
        "blocking": blocking,
        "mechanism": mechanism,
        "tensile_opening": tensile_opening,
        "state": state.as_dict(),
        "plane": {
            "dip_deg": round(dip_deg, 1),
            "azimuth_deg": round(azimuth_deg, 1),
            "sigma_n_mpa": round(sigma_n, 2),
            "sigma_n_eff_mpa": round(sigma_n_eff, 2),
            "tau_mpa": round(tau, 2),
        },
        # slip_ratio 是判定口径（注入后），slip_tendency 是原地状态（现状描述）。
        # 两个都返回，因为报告里"目前稳定"和"注入后稳定"必须分开写。
        "slip_ratio": _round(slip_ratio),
        "slip_tendency": _round(ts_current),
        "slip_tendency_injected": _round(ts_injected),
        "pp_crit_mpa": _round(pp_crit),
        "pp_crit_bar": _round(pp_crit * BAR_PER_MPA),
        "margin_mpa": _round(margin),
        "margin_bar": _round(margin * BAR_PER_MPA),
        "applied_overpressure_bar": round(applied * BAR_PER_MPA, 1),
        "consistent": consistent,
        "polygon": polygon,
        "coupled": coupled,
        "optimal_dip_deg": optimal_dip(state, p, azimuth_deg),
        "assumptions": _assumed(p, environ),
        "limitations": [
            "一维应力模型：Shmin/SHmax 由 Sv 按比例给出，不是实测。",
            "孔隙弹性只建模注入层内部的单轴应变应力路径（coupled 字段）；"
            "盖层与隔层的应力转移未建模，那部分可以在局部降低 σh 并促进滑动。",
            "断层按平面、无黏聚力处理；实际断裂带宽度与结构面粗糙度未考虑。",
            "未考虑断层渗透性随剪切的演化（剪胀/剪缩对压力的反馈）。",
        ],
        "version": GEOMECHANICS_VERSION,
        "params": stress_manifest(p, environ)["parameters"],
    }


def _round(x: float, nd: int = 3):
    if x == float("inf"):
        return "inf"
    return round(x, nd)


def render_block(result: dict) -> str:
    """把结果渲染成报告可用的一段文字（不是给模型看，是给人看）。"""
    if not result or result.get("verdict") == "no_data":
        return ""
    st = result["state"]
    pl = result["plane"]
    head = {
        "stable": "断层未活化",
        "near_critical": "断层接近活化（余量不足 20%）",
        "reactivated": "断层已活化（注入附加压力超过活化余量）",
        "tensile_opening": "断层张性开启（孔隙压力已超过断层面正应力）",
    }[result["verdict"]]
    lines = [
        f"【断层稳定性（{result['version']}）】{head}",
        f"- 应力状态：{st['depth_m']:.0f} m，Sv {st['sv_bar']:.0f} bar / "
        f"SHmax {st['shmax_bar']:.0f} bar / Shmin {st['shmin_bar']:.0f} bar / "
        f"Pp {st['pp_bar']:.0f} bar，{st['regime']} 型",
        f"- 断层面（倾角 {pl['dip_deg']:.0f}°、倾向 {pl['azimuth_deg']:.0f}°）："
        f"σn {pl['sigma_n_mpa']:.1f} MPa，τ {pl['tau_mpa']:.1f} MPa，"
        f"有效正应力 {pl['sigma_n_eff_mpa']:.1f} MPa",
        f"- 滑移比 τ/(μσ'n) = {result['slip_ratio']}（≥1 即滑动）",
        f"- 临界孔隙压力 {result['pp_crit_bar']} bar，"
        f"当前余量 {result['margin_bar']} bar，"
        f"注入附加压力 {result['applied_overpressure_bar']} bar",
    ]
    if result["assumptions"]:
        lines.append(
            "- 前提：以下参数用了默认假设值，不是实测 —— "
            + "、".join(result["assumptions"])
            + "。结论仅在这些假设成立时有效。"
        )
    cp = result.get("coupled") or {}
    if cp:
        same = cp.get("verdict") == result["verdict"]
        lines.append(
            f"- 孔隙弹性口径（单轴应变，Δσh/ΔP = {cp.get('path_ratio')}）："
            f"滑移比 {cp.get('slip_ratio')}，余量 {cp.get('margin_bar')} bar，"
            f"判定 {cp.get('verdict')}"
            + ("（与保守口径一致）" if same else "（比保守口径宽松，仅对注入层成立）")
        )
    if not result["polygon"]["ok"]:
        lines.append(
            f"- ⚠ 输入应力违反摩擦平衡上限（比 {result['polygon']['ratio']} > "
            f"上限 {result['polygon']['limit']}），应力数据本身不自洽"
        )
    return "\n".join(lines)
