"""
注入策略优化（把"诊断"接成"处方"）
==================================
在 v1.8 之前，系统的输出停在"这个方案压裂概率 47%，告警"。诊断到此为止：
读者知道危险，但不知道**改成什么**。这个模块补的是后半段 —— 给定同一套
物理与同一套不确定性，反解出一个满足风险预算的注入方案，并把它和当前方案
逐项对比。

为什么优化器必须是确定性的，而不是"让模型提建议"
------------------------------------------------
约束搜索有两个性质，恰好是语言模型最不可靠的地方：

1. **可核对**。给出的每个数都要能回代：``q = n·J·(p − P_h)`` 必须成立，
   风险必须等于同一批样本在该压力下的命中率。模型写出的"建议 1.2 Mt/yr"
   没有回代路径，无法判断它是算出来的还是编的。
2. **失败要说出来**。无解时必须返回"无解"，而不是给一个看起来合理的数。
   模型在无解时倾向于给一个折中值，那正是最危险的行为。

所以优化在这里是一个**约束求解**，不是一次生成。模型只负责转述结果。

决策变量为什么是"井底压力上限"，而不是"速率"
--------------------------------------------
这三件事的因果关系是单向的：

    井底压力 p  →（风险只由 p 决定）→  压裂概率
    井底压力 p  →（注入指数 J）    →  单井速率 q_w = J·(p − P_h)
    井数 n      →（并联）          →  总速率 q = n·q_w

风险**不直接依赖速率**：速率是压力差的函数，同一压力下换一台泵不影响
破裂压力。所以把速率当决策变量会让优化器在一个错误的自由度上搜索 ——
它会把"降低速率"当成降低风险的手段，而真正的机制是"降低井底压力"。
反过来说，井数不改变风险，只改变同一压力下能注入多少。
**默认按独立井计算**（不给井距即无干扰）；给了 ``well_spacing_m`` 之后
按稳态径向叠加计入井间干扰，加井就不再线性增注，并且存在饱和井数。

这个方向性是有代价的：它意味着**速率与风险在本模型里不能同时改善**。
想要"又安全又注得多"，唯一的手段是加井。这一点在 ``notes`` 里明写。

关键量：临界压力 ``P_s`` 与它的经验分布
--------------------------------------
对第 i 组扰动样本，泄漏判据是

    p_nominal·(1+ε_p) > 1.5·P_h(d_i)
    ⟺ p_nominal > 1.5·P_h(d_i) / (1+ε_p)  ≡  P_s,i

于是 ``P(泄漏 | p) = #{P_s,i < p} / n`` —— 风险就是 ``P_s`` 的经验分布函数。
这个改写让优化**不需要二分搜索**：把 ``P_s`` 排序后，"命中数不超过 k 的
最大压力"就是第 k 个次序统计量，一步到位，没有容差、没有迭代次数、
结果与实现顺序无关。二分搜索在样本量有限时还会引入"搜索误差"，
让人分不清"没找到更好的解"和"搜索没收敛"。

``P_s`` 的推导与 ``pressure_model.verdict_of`` **逐字等价**（后者用
``p_i > P_frac(d_i)``），所以本模块的风险口径与
``tools.physics_tools.injection_risk_uq`` 完全一致 —— 这一点由
``tests/test_optimize.py`` 用同一 seed 的逐样本相等来钉住。两个模块给出
不同的概率，只可能是因为有人改了其中一个的判据。

为什么约束用 Wilson 上界而不是点估计
------------------------------------
"P = 4.9%，低于 5% 预算" 是一句不成立的话：n=100 时 4.9% 的 95% 区间是
[2.0%, 11.4%]，它和 11% 无法区分。要求**上界** ≤ 预算，等价于说
"我有 95% 的把握这个方案不超预算"，这才是"合规"该有的含义。

代价是：n=100 时上界要 ≤5% 只允许 0 次命中（0/100 的上界是 3.6%，
1/100 是 5.5%），约束会退化成"一个样本都不许越界"，既保守又量化粗糙。
所以优化器默认跑 **n=1000**（纯 Python，排序 1000 个浮点约 0.1 ms），
而报告里的 100 次是给读者看分布的。两个数字都随结果返回，不混用。

样本内与样本外
--------------
在某一批样本上解出压力上限、再用**同一批样本**报告该方案的风险，是
样本内评估 —— 一定偏乐观（优化器正是挑着这批样本里最有利的那一档）。
所以 ``optimize_plan`` 用独立 seed 再抽一批，把 ``verification`` 单列。
两者差异超出二项噪声时，说明方案贴着阈值，``notes`` 会明说。

常数来路
--------
本模块不引入新的物理常数，全部取自既有模块或作为显式入参：

``1.5``            破裂压力/静水压力比，``pressure_model.FRACTURE_RATIO``
                   （常用区间 1.4–1.8 的中值）
``J``              径向稳态 Darcy，``utils.relperm.injectivity_index``
``μ``              CO₂ 黏度常数 0.06 mPa·s，``utils.co2_properties``
``ρ``              Peng-Robinson 状态方程，``utils.co2_properties``
``k / h / T``      入参，默认值与 ``utils.reservoir_backend.SimulationCase`` 一致
``skin``           入参，默认 0（理想完井）
``p_target``       默认取 ``core.risk_policy`` 的 warn 阈值
``cap_ratio``      入参，默认 1.0（不额外加设计余量）
"""
from __future__ import annotations

import math
from bisect import bisect_left

import pressure_model as pm
from core.risk_policy import decide, load_thresholds
from core.uq import DEFAULT_DIST, DEFAULT_REL_SIGMA, JointSampler, Marginal, wilson_interval

# ─── 默认入参（全部来自既有模块，不是新常数）──────────────
DEFAULT_PERM_MD = 100.0          # 与 reservoir_backend.SimulationCase 默认一致
DEFAULT_THICKNESS_M = 50.0       # 同上
DEFAULT_TEMPERATURE_C = 60.0     # 同上
DEFAULT_SKIN = 0.0               # 理想完井；>0 会降低注入能力（方向见 notes）
DEFAULT_YEARS = 20.0             # 规划期，仅用于累计封存量换算
DEFAULT_WELL_COUNTS = (1, 2, 4, 8)
CAP_RATIO_DEFAULT = 1.0          # 不额外加设计余量：风险预算已经在约束压力

# 井间干扰用到的两个几何量。**故意与 utils.relperm.injectivity_index 的
# 默认值取同一个数**（re=500 m / rw=0.1 m）：那个函数算的是 J，干扰系数
# 算的是同一套径向几何下的对数比，两个默认值一旦分叉，"注入能力"和
# "干扰代价"就会各自按不同的井距在说话，且不报错。
# tests/test_optimize.py 里有一条断言把两边钉死。
DEFAULT_DRAINAGE_RADIUS_M = 500.0
DEFAULT_WELLBORE_RADIUS_M = 0.1

# 优化器的模拟次数。与报告里的 100 次不同，理由见模块文档"为什么约束用
# Wilson 上界"：n=100 时 5% 预算只允许 0 次命中，约束会退化成最小值。
OPTIMIZE_N = 1000

# 状态码
STATUS_OK = "ok"
STATUS_INFEASIBLE = "infeasible"     # 有可注入的压力窗口，但达不到要求的速率
STATUS_NO_INJECTION = "no_injection"  # 连"注得进"都做不到

# 约束名（binding）
BIND_RISK = "risk"              # 风险预算先到
BIND_CAP = "pressure_cap"       # 破裂压力上限先到
BIND_RATE_CAP = "rate_cap"      # 外部速率上限先到


# ══════════════════════════════════════════════════════════
# 注入能力
# ══════════════════════════════════════════════════════════
def injectivity_m3_per_day_per_bar(
    permeability_md: float = DEFAULT_PERM_MD,
    thickness_m: float = DEFAULT_THICKNESS_M,
    temperature_celsius: float = DEFAULT_TEMPERATURE_C,
    skin: float = DEFAULT_SKIN,
) -> float:
    """
    注入指数 J [m³/(d·bar)]：单位压差下的日注入量。

    直接调 ``utils.relperm.injectivity_index``（径向稳态 Darcy），只做单位
    换算 —— **不在这里写第二份公式**。两份实现迟早会漂移，而漂移之后
    "优化建议的速率"和"两相流模块报的注入能力"会不一致，且不报错。

    ``skin`` 默认 0：这是**乐观**假设。真实完井有污染/射孔不完善，S>0 时
    J 按 ``ln(re/rw)/(ln(re/rw)+S)`` 衰减（S=5 约降到 63%）。所以本模块给出
    的速率是"压力约束下的上限"，不是"工程可实现值"。这句话进 ``notes``。
    """
    from utils.relperm import injectivity_index

    from utils.co2_properties import co2_viscosity_mpa_s

    mu = co2_viscosity_mpa_s(
        pm.hydrostatic_bar(2000.0), temperature_celsius
    )   # 黏度在本项目里是常数（不随 P/T 变），传标称压力只为接口一致
    index = injectivity_index(permeability_md, thickness_m, mu, skin=skin)
    # J_m3_per_day_per_mpa ÷ 10 = m³/(d·bar)（1 MPa = 10 bar）
    return float(index["J_m3_per_day_per_mpa"]) / 10.0


def rate_m3_per_day(
    pressure_bar: float,
    depth_m: float,
    injectivity: float,
    n_wells: int = 1,
    interference: float = 0.0,
) -> float:
    """
    井底压力 p → 总注入速率。

    无干扰时 ``q = n·J·(p − P_h)``；给了井间干扰系数 ``f`` 之后是

        q = n·J·(p − P_h) / (1 + (n−1)·f)

    分母就是"多井互相抢压差"的代价：n 口井各自要克服另外 n−1 口井造成的
    压降，所以同样的井底压力下，总速率**不是**线性放大。``f=0`` 时两条式子
    完全一致 —— 默认值就是 0，所以不传井距的调用方行为不变。

    压差为负时返回 0 而不是负数：``p < P_h`` 意味着注不进去（甚至倒流），
    负速率在后续质量换算里会变成一个"负的封存量"，不会报错但整段结论反了。
    """
    drawdown = float(pressure_bar) - float(pm.hydrostatic_bar(depth_m))
    n = float(n_wells)
    f = float(interference)
    if not math.isfinite(f) or not 0.0 <= f < 1.0:
        raise ValueError(f"井间干扰系数必须在 [0, 1) 内，收到 {f!r}")
    if not math.isfinite(n) or n < 1 or not n.is_integer():
        raise ValueError(f"井数必须为正整数，收到 {n_wells!r}")
    if not math.isfinite(injectivity) or injectivity <= 0:
        raise ValueError("注入指数必须为有限正数")
    if not math.isfinite(drawdown):
        raise ValueError("井底压力和埋深必须为有限数")
    if drawdown <= 0.0:
        return 0.0
    return n * float(injectivity) * drawdown / (1.0 + (n - 1.0) * f)


def pressure_bar_for_rate(
    rate_m3_day: float,
    depth_m: float,
    injectivity: float,
    n_wells: int = 1,
    interference: float = 0.0,
) -> float:
    """总注入速率 q → 井底压力 p，是 ``rate_m3_per_day`` 的严格逆运算。"""
    n = float(n_wells)
    f = float(interference)
    if not math.isfinite(f) or not 0.0 <= f < 1.0:
        raise ValueError(f"井间干扰系数必须在 [0, 1) 内，收到 {f!r}")
    if not math.isfinite(n) or n < 1 or not n.is_integer():
        raise ValueError(f"井数必须为正整数，收到 {n_wells!r}")
    if not math.isfinite(injectivity) or injectivity <= 0:
        raise ValueError("注入指数必须为有限正数")
    if not math.isfinite(rate_m3_day):
        raise ValueError("注入速率必须为有限数")
    if rate_m3_day <= 0.0:
        return float(pm.hydrostatic_bar(depth_m))
    return float(pm.hydrostatic_bar(depth_m)) + float(rate_m3_day) * (
        1.0 + (n - 1.0) * f
    ) / (n * float(injectivity))


def interference_factor(
    spacing_m: float,
    drainage_radius_m: float = DEFAULT_DRAINAGE_RADIUS_M,
    rw_m: float = DEFAULT_WELLBORE_RADIUS_M,
    skin: float = DEFAULT_SKIN,
) -> dict:
    """
    稳态径向叠加下的井间干扰系数 ``f``（无因次）。

    推导（不是抄公式，推导过程决定了它什么时候失效）
    -------------------------------------------------
    单井稳态径向 Darcy：

        q_w = 2πkh/μ · Δp_own / (ln(r_e/r_w) + S)  ⇒  Δp_own = q_w/J

    把每口邻井看成同强度线汇，它在目标井处造成的压降（同一外边界 r_e 的
    稳态解）是

        Δp_ij = q_w · μ/(2πkh) · ln(r_e/d_ij)

    两者相除，``μ/(2πkh)`` 消掉，只剩对数项之比：

        Δp_ij / Δp_own = ln(r_e/d) / (ln(r_e/r_w) + S) ≡ f

    假设每一对井都具有同一干扰系数 f，得到均值场近似：

        p − P_h = Δp_own · (1 + (n−1)f)  ⇒  q_total = n·J·(p−P_h)/(1+(n−1)f)

    **关键适用域**：平面里至多 3 个点可以两两等距。n≥4 时单一井距
    不可能代表每一对井的实际距离，f 只是"所有井对相同耦合"的**均值场情景**，
    不是 4/8 井的二维井网预测；不能把它写成"已模拟多井干扰"。真正的井网
    需每口井的坐标、不同的两两距离和求解耦合方程，并由外部模拟器对照。

    两个代数极限：``d → r_e`` 时 ``f → 0``；**固定 f** 下 ``n → ∞``
    时 ``q_total → J·(p−P_h)/f``。后者仅是均值场模型的数学性质，
    并非有限储层可无限加井的工程预测。

    为什么 ``f`` 一定小于 1：``d ≥ r_w`` 时 ``ln(r_e/d) ≤ ln(r_e/r_w)``，
    而 ``S ≥ 0``，所以分子不超过分母。f=1 意味着"邻井离你和井壁一样近"，
    物理上不可能，代码里也不该出现。

    偏差方向
    --------
    这是**等产量、等效井距、稳态、均质**的均值场情景。井位不均匀、
    注入率会重新分配、储层连通性和非均质未建模；误差可能向任一方向，
    **不能声称它系统性高估或低估**。只能用来做假设敏感性，不能作阻断或
    场地级井数决策。
    """
    spacing = float(spacing_m)
    re = float(drainage_radius_m)
    rw = float(rw_m)
    if not all(math.isfinite(v) for v in (spacing, re, rw, float(skin))):
        raise ValueError("井距、泄油半径、井筒半径及表皮系数必须为有限数")
    if rw <= 0 or spacing <= rw:
        raise ValueError(
            f"井距 {spacing} m 必须大于井筒半径 {rw} m，无法计算干扰"
        )
    if re <= rw:
        raise ValueError(f"泄油半径 {re} m 必须大于井筒半径 {rw} m")
    if skin < 0:
        raise ValueError(f"表皮系数不能为负，收到 {skin}")

    ln_own = math.log(re / rw) + float(skin)
    if spacing >= re:
        # 井距不小于泄油半径：两井各自圈一块地，稳态叠加的前提不成立。
        # 返回 0 而不是负数 —— ln(r_e/d) 在 d>r_e 时是负的，直接代进公式
        # 会得到"有邻井反而注得更多"，那是个纯粹的建模事故。
        return {
            "factor": 0.0,
            "active": False,
            "ln_own": ln_own,
            "ln_interference": 0.0,
            "spacing_m": spacing,
            "drainage_radius_m": re,
            "reason": (
                f"井距 {spacing:g} m ≥ 泄油半径 {re:g} m：两井分属独立泄油区，"
                f"稳态叠加不适用，未计干扰"
            ),
        }
    ln_interf = math.log(re / spacing)
    return {
        "factor": ln_interf / ln_own,
        "active": True,
        "ln_own": ln_own,
        "ln_interference": ln_interf,
        "spacing_m": spacing,
        "drainage_radius_m": re,
        "reason": (
            f"井距 {spacing:g} m / 泄油半径 {re:g} m → f={ln_interf / ln_own:.4f}"
        ),
    }


def mass_mt_per_year(rate_m3_day: float, density_kg_m3: float) -> float:
    """m³/d → Mt/yr。1 Mt = 1e9 kg。"""
    return float(rate_m3_day) * 365.0 * float(density_kg_m3) / 1e9


def rate_for_mass(mt_per_year: float, density_kg_m3: float) -> float:
    """Mt/yr → m³/d。"""
    return float(mt_per_year) * 1e9 / (365.0 * float(density_kg_m3))


# ══════════════════════════════════════════════════════════
# 临界压力样本（风险的唯一来源）
# ══════════════════════════════════════════════════════════
def critical_pressure_samples(
    depth_m: float,
    rel_sigma: float = DEFAULT_REL_SIGMA,
    n: int = OPTIMIZE_N,
    seed: int | None = None,
    dist: str = DEFAULT_DIST,
    corr=None,
) -> tuple[list[float], JointSampler]:
    """
    抽 ``n`` 组（深度, 压力）扰动，返回排序后的临界压力 ``P_s`` 与采样器。

    ``P_s,i = P_frac(d_i) / (1+ε_p,i)``，判据是 ``p > P_s,i`` 即压裂。
    与 ``injection_risk_uq`` 的 ``verdict_of(d_i, p_i)`` 等价（推导见模块文档），
    所以同一 seed 下两者的命中数**逐样本相等**，不是"统计上接近"。

    返回采样器是为了让调用方拿到 ``independent`` / ``empirical_correlation``
    —— 相关性是一个假设，必须能随结果一起交代。
    """
    n = max(2, int(n))
    dims = [
        {"name": "深度", "kind": dist, "rel_sigma": rel_sigma},
        {"name": "井底压力", "kind": dist, "rel_sigma": rel_sigma},
    ]
    sampler = JointSampler(dims, corr=corr, seed=seed)
    out: list[float] = []
    for _ in range(n):
        eps_d, eps_p = sampler.draw()
        d_i = float(depth_m) * (1.0 + eps_d)
        # 1+ε_p 可能被 lognormal/beta 压到极小但不为负；下界夹一下更安全
        out.append(float(pm.fracture_bar(d_i)) / max(1.0 + eps_p, 1e-6))
    out.sort()
    return out, sampler


def risk_at_pressure(pressure_bar: float, critical: list[float]) -> dict:
    """
    给定井底压力，返回命中数 / 概率 / Wilson 区间 / 决策等级。

    ``bisect_left`` 数的是 ``P_s < p``（严格小于），与判据
    ``p > P_frac`` 的严格不等号一致 —— 边界上"恰好等于"不算压裂。
    """
    n = len(critical)
    if n <= 0:
        raise ValueError("critical 样本为空：无法计算风险")
    hits = bisect_left(critical, float(pressure_bar))
    p = hits / n
    lo, hi = wilson_interval(hits, n)
    out = {
        "hits": hits,
        "n": n,
        "p": p,
        "p_ci": [lo, hi],
        "p_upper": hi,
        "pressure_bar": float(pressure_bar),
    }
    try:
        out["decision"] = decide({"risky": p}, n)
    except Exception:       # 阈值配置错误不该让整个优化结果消失
        out["decision"] = {}
    return out


def max_pressure_for_target(
    critical: list[float],
    p_target: float,
    cap_bar: float,
    floor_bar: float,
) -> dict:
    """
    在"Wilson 上界 ≤ p_target"下求最大井底压力。

    做法是找最大的 k 使 ``wilson_hi(k, n) ≤ p_target``，再取第 k 个次序
    统计量。取 ``critical[k]`` 时严格小于它的样本正好 k 个，所以命中数
    恰好是 k，不会多一个 —— 用 ``critical[k] + ε`` 就会多算一个，而
    "多算一个"在 n=1000 时就是 0.1 个百分点，看起来微不足道，却足以让
    上界从 4.99% 跳到 5.02%，把可行解变成不可行解。

    返回 ``{"pressure_bar", "k_max", "hits", "binding", "feasible"}``。
    """
    n = len(critical)
    if n <= 0:
        raise ValueError("critical 样本为空：无法求解压力上限")

    # wilson_hi 关于 k 单调递增，线性扫到第一个越界即可（n=1000，开销可忽略）
    k_max = 0
    for k in range(n + 1):
        if wilson_interval(k, n)[1] <= float(p_target):
            k_max = k
        else:
            break

    binding = BIND_RISK
    if k_max >= n:
        # 预算宽到允许全部样本越界，此时压力由破裂压力上限决定
        p_star = float(cap_bar)
        binding = BIND_CAP
    else:
        p_star = min(float(critical[k_max]), float(cap_bar))
        if p_star >= float(cap_bar) and float(critical[k_max]) > float(cap_bar):
            binding = BIND_CAP

    hits = bisect_left(critical, p_star)
    return {
        "pressure_bar": p_star,
        "k_max": k_max,
        "hits": hits,
        "p_upper": wilson_interval(hits, n)[1],
        "binding": binding,
        "feasible": p_star > float(floor_bar),
        "cap_bar": float(cap_bar),
        "floor_bar": float(floor_bar),
    }


# ══════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════
def optimize_plan(
    depth_m: float,
    injection_pressure_bar: float | None = None,
    storage_rate_mt_per_year: float | None = None,
    *,
    well_counts: tuple[int, ...] = DEFAULT_WELL_COUNTS,
    p_target: float | None = None,
    years: float = DEFAULT_YEARS,
    rel_sigma: float = DEFAULT_REL_SIGMA,
    n: int = OPTIMIZE_N,
    seed: int | None = None,
    dist: str = DEFAULT_DIST,
    corr=None,
    permeability_md: float = DEFAULT_PERM_MD,
    thickness_m: float = DEFAULT_THICKNESS_M,
    temperature_celsius: float = DEFAULT_TEMPERATURE_C,
    skin: float = DEFAULT_SKIN,
    cap_ratio: float = CAP_RATIO_DEFAULT,
    rate_cap_mt_per_year: float | None = None,
    well_spacing_m: float | None = None,
    drainage_radius_m: float = DEFAULT_DRAINAGE_RADIUS_M,
    thresholds: dict | None = None,
) -> dict:
    """
    给定当前方案（深度 + 井底压力，或深度 + 目标速率），求满足风险预算的
    注入方案。

    参数
    ----
    depth_m                 : 储层埋深（井位/目标层的代表深度）
    injection_pressure_bar  : 当前井底压力。给定时反推当前速率
    storage_rate_mt_per_year: 目标速率。给定它时，"推荐方案"以保住这个速率为
                              前提（加井是允许的手段）；不给则以当前速率为准
    well_counts             : 允许的井数组合（默认 1/2/4/8）。**不是**被优化的
                              连续变量 —— 井数是离散的工程选项，给出可比的
                              几档比给一个"最优井数"更可用（井数还受井场
                              面积、许可、管线约束，本模型不知道）
    p_target                : 风险预算。默认取 warn 阈值（5%），语义是
                              "95% 上界低于提示线"
    cap_ratio               : 井底压力上限 = cap_ratio × 破裂压力。默认 1.0，
                              即不再额外加设计余量（风险预算已经压住了压力）
    rate_cap_mt_per_year    : 设施/许可的速率上限，默认 None（不设）
    well_spacing_m          : 井距。**默认 None = 不计井间干扰**，与旧行为逐位
                              一致。给了之后按 ``interference_factor`` 的稳态
                              叠加计算：同样的井底压力下，n 口井不再线性放大。
                              为什么不给默认井距：井距是**设计输入**，不是地质
                              未知量；替用户假定一个井距，等于替他做了一个
                              他没做过的决定，而且这个假定会直接改变推荐方案。
    drainage_radius_m       : 单井泄油半径，默认 500 m（与 relperm 的径向几何
                              同源）。只在给了 ``well_spacing_m`` 时起作用。

    返回
    ----
    ``{"status", "current", "limit", "well_options", "recommended",
       "verification", "frontier", "physics", "binding", "notes", "text",
       "provenance"}``
    """
    depth = float(depth_m)
    if not 50.0 <= depth <= 9000.0:
        raise ValueError(f"埋深 {depth} m 超出 [50, 9000]，无法做注入策略优化")
    if cap_ratio <= 0.0:
        raise ValueError(f"cap_ratio 必须为正，收到 {cap_ratio}")
    if years <= 0.0:
        raise ValueError(f"规划期 years 必须为正，收到 {years}")
    if rate_cap_mt_per_year is not None and rate_cap_mt_per_year <= 0.0:
        raise ValueError(f"速率上限必须为正，收到 {rate_cap_mt_per_year}")

    # 井间干扰：不给井距就是 0，行为与引入本功能之前完全一致。
    if well_spacing_m is None:
        interference = {
            "factor": 0.0,
            "active": False,
            "ln_own": None,
            "ln_interference": None,
            "spacing_m": None,
            "drainage_radius_m": float(drainage_radius_m),
            "reason": "未提供井距：按独立井计算（无井间干扰）",
        }
    else:
        interference = interference_factor(
            well_spacing_m, drainage_radius_m, DEFAULT_WELLBORE_RADIUS_M, skin
        )
    f_int = float(interference["factor"])

    th = thresholds if thresholds is not None else load_thresholds()
    target = float(th["warn"] if p_target is None else p_target)
    if not 0.0 <= target <= 1.0:
        raise ValueError(f"风险预算必须在 [0, 1] 内，收到 {target}")

    p_h = float(pm.hydrostatic_bar(depth))
    p_frac = float(pm.fracture_bar(depth))
    cap_bar = p_frac * float(cap_ratio)
    j = injectivity_m3_per_day_per_bar(
        permeability_md, thickness_m, temperature_celsius, skin
    )
    # 密度取**静水压力**下的值，不是注入压力下的值：羽流主体坐在未扰动压力
    # 附近，注入井周围那一圈高压只占体积的一小部分。这一取舍有方向：p_h 下
    # 的密度更低（2500 m 处 777 vs 380 bar 下 866 kg/m³），同样的体积速率
    # 换算出更小的质量速率 —— 也就是**不夸大**可注入量。数值进 physics
    # 供回代，读者不必相信这里的选择。
    density = _density(p_h, temperature_celsius)

    critical, sampler = critical_pressure_samples(
        depth, rel_sigma=rel_sigma, n=n, seed=seed, dist=dist, corr=corr
    )
    limit = max_pressure_for_target(critical, target, cap_bar, p_h)

    # 这里**不四舍五入**：``physics`` 是回代核对的基准（启用干扰时含分母），
    # 取整会让"用返回的数复算"与"返回的数"差一点点，而这种差异恰好会被
    # 当成实现 bug。展示层（前端/文案）自己决定保留几位。
    physics = {
        "depth_m": depth,
        "hydrostatic_bar": p_h,
        "fracture_bar": p_frac,
        "cap_bar": cap_bar,
        "injectivity_m3_per_day_per_bar": j,
        "density_kg_m3": density,
        "permeability_md": float(permeability_md),
        "thickness_m": float(thickness_m),
        "temperature_celsius": float(temperature_celsius),
        "skin": float(skin),
        "cap_ratio": float(cap_ratio),
        "fracture_ratio": float(pm.FRACTURE_RATIO),
        # 干扰系数必须进 physics：回代核对要用的式子已经不是
        # q = n·J·(p−P_h)，而是 q = n·J·(p−P_h)/(1+(n−1)f)。
        # f 不出现在返回值里，读者就无法复算，也就无法判断这个数是算的还是编的。
        "interference_factor": float(interference["factor"]),
        "interference_active": bool(interference["active"]),
        "well_spacing_m": interference["spacing_m"],
        "drainage_radius_m": interference["drainage_radius_m"],
    }

    # ─── 当前方案 ────────────────────────────────────────
    current: dict = {}
    if injection_pressure_bar is not None:
        p_cur = float(injection_pressure_bar)
        q_cur = rate_m3_per_day(p_cur, depth, j, 1, f_int)
        current = {
            "pressure_bar": p_cur,
            "n_wells": 1,
            "rate_m3_per_day": q_cur,
            "rate_mt_per_year": mass_mt_per_year(q_cur, density),
            "verdict": pm.verdict_of(depth, p_cur),
            "risk": risk_at_pressure(p_cur, critical),
        }
    elif storage_rate_mt_per_year is not None:
        q_cur = rate_for_mass(storage_rate_mt_per_year, density)
        p_cur = pressure_bar_for_rate(q_cur, depth, j, 1, f_int)
        current = {
            "pressure_bar": p_cur,
            "n_wells": 1,
            "rate_m3_per_day": q_cur,
            "rate_mt_per_year": float(storage_rate_mt_per_year),
            "verdict": pm.verdict_of(depth, p_cur),
            "risk": risk_at_pressure(p_cur, critical),
        }

    # 要保住的速率：显式目标优先，否则沿用当前速率
    q_req = (
        rate_for_mass(storage_rate_mt_per_year, density)
        if storage_rate_mt_per_year is not None
        else (current.get("rate_m3_per_day") or 0.0)
    )

    # ─── 各井数下的可行速率 ──────────────────────────────
    #
    # 每一档井数要回答两个不同的问题，两个都要给：
    #   p_needed : 恰好满足目标速率所需的井底压力（越低越安全）
    #   q_max    : 该井数在风险预算内能做到的最大速率（余量）
    # 只给 q_max 会让"推荐方案"变成"最大化"，于是"保住 1 Mt/yr"的建议会
    # 输出 3 Mt/yr —— 读者无法判断这是建议还是上限。
    p_star = limit["pressure_bar"]
    options: list[dict] = []
    for n_w in sorted({int(w) for w in well_counts if int(w) >= 1}):
        q_max = rate_m3_per_day(p_star, depth, j, n_w, f_int)
        binding = limit["binding"]
        if rate_cap_mt_per_year is not None:
            q_ceiling = rate_for_mass(rate_cap_mt_per_year, density)
            if q_max > q_ceiling:
                q_max = q_ceiling
                binding = BIND_RATE_CAP
        p_need = (
            pressure_bar_for_rate(q_req, depth, j, n_w, f_int)
            if q_req > 0
            else p_star
        )
        options.append({
            "n_wells": n_w,
            "interference_scope": (
                "mean_field_not_a_well_layout" if f_int > 0 and n_w > 3
                else "pairwise_geometry" if f_int > 0 else "independent_wells"
            ),
            "pressure_bar": p_star,
            "pressure_needed_bar": p_need,
            "rate_m3_per_day": q_max,
            "rate_mt_per_year": mass_mt_per_year(q_max, density),
            "per_well_mt_per_year": mass_mt_per_year(q_max / n_w, density),
            "binding": binding,
            # 两个条件缺一不可：压力够（p_need ≤ p_star）只说明注入能力够，
            # 若外部速率上限比目标还低，加井也救不回来 —— 这是设施约束，
            # 不是地层约束，混在一起会让"无解"看起来像"再打一口井就行"
            "meets_target": q_req > 0 and p_need <= p_star and q_req <= q_max,
            "risk": risk_at_pressure(p_star, critical),
            "risk_at_needed": risk_at_pressure(min(p_need, p_star), critical),
        })

    # ─── 推荐 ────────────────────────────────────────────
    #
    # 语义固定为"**恰好**满足目标速率、用最少的井"：
    # 多给的速率不是建议的一部分，而是余量（单独放在 headroom 里）。
    # 这样"推荐方案"永远可以回代 —— q = n·J·(p − P_h) 两边相等。
    feasible = [o for o in options if o["meets_target"]]
    headroom: dict = {}
    if limit["pressure_bar"] <= p_h:
        status = STATUS_NO_INJECTION
        recommended = None
    elif q_req <= 0.0:
        # 没有目标速率（当前压力都注不进）：给出风险预算下的能力上限，
        # 取**最少**井数 —— 井数是成本，没有要求时没有理由多加井
        status = STATUS_OK
        best = min(options, key=lambda o: o["n_wells"])
        recommended = {
            "n_wells": best["n_wells"],
            "pressure_bar": best["pressure_bar"],
            "rate_m3_per_day": best["rate_m3_per_day"],
            "rate_mt_per_year": best["rate_mt_per_year"],
            "per_well_mt_per_year": best["per_well_mt_per_year"],
            "binding": best["binding"],
            "risk": best["risk"],
        }
    elif feasible:
        status = STATUS_OK
        best = min(feasible, key=lambda o: o["n_wells"])
        recommended = {
            "n_wells": best["n_wells"],
            "pressure_bar": best["pressure_needed_bar"],
            "rate_m3_per_day": q_req,
            "rate_mt_per_year": mass_mt_per_year(q_req, density),
            "per_well_mt_per_year": mass_mt_per_year(q_req / best["n_wells"], density),
            "binding": best["binding"],
            "risk": best["risk_at_needed"],
            "risk_if_pressed": best["risk"],
        }
        headroom = {
            "rate_mt_per_year": best["rate_mt_per_year"] - mass_mt_per_year(q_req, density),
            "rate_max_mt_per_year": best["rate_mt_per_year"],
            "fraction": best["rate_mt_per_year"] / max(mass_mt_per_year(q_req, density), 1e-9) - 1.0,
            "pressure_bar": best["pressure_bar"] - best["pressure_needed_bar"],
            "pressure_max_bar": best["pressure_bar"],
        }
    else:
        status = STATUS_INFEASIBLE
        best = max(options, key=lambda o: o["rate_m3_per_day"]) if options else None
        if best:
            # 压力必须由速率反推，不能直接抄 p_star：被设施上限截断时，
            # 那个速率对应的压力远低于 p_star，写成 p_star 会让
            # q = n·J·(p−P_h) 两边不等 —— 而这是唯一能回代核对的式子
            p_impl = pressure_bar_for_rate(
                best["rate_m3_per_day"], depth, j, best["n_wells"], f_int
            )
            recommended = {
                "n_wells": best["n_wells"],
                "pressure_bar": p_impl,
                "rate_m3_per_day": best["rate_m3_per_day"],
                "rate_mt_per_year": best["rate_mt_per_year"],
                "per_well_mt_per_year": best["per_well_mt_per_year"],
                "binding": best["binding"],
                "risk": risk_at_pressure(p_impl, critical),
            }
        else:
            recommended = None
        headroom = {
            "shortfall_mt_per_year": mass_mt_per_year(q_req, density)
            - (best["rate_mt_per_year"] if best else 0.0),
            "fraction": (
                (best["rate_mt_per_year"] / max(mass_mt_per_year(q_req, density), 1e-9) - 1.0)
                if best else -1.0
            ),
        }

    # ─── 样本外复核 ──────────────────────────────────────
    # 用独立 seed 再抽一批：在解出压力上限的同一批样本上报告风险，一定偏乐观
    verification: dict = {}
    if recommended is not None:
        check, _ = critical_pressure_samples(
            depth, rel_sigma=rel_sigma, n=n,
            seed=_derived_seed(seed), dist=dist, corr=corr,
        )
        verification = risk_at_pressure(recommended["pressure_bar"], check)
        verification["seed"] = _derived_seed(seed)
        verification["agrees"] = abs(verification["p"] - recommended["risk"]["p"]) <= (
            3.0 * _binomial_se(recommended["risk"]["p"], n) + 1.0 / n
        )

    # 明示 n>3 的均值场模型不是二维井位。不能让使用方只读 recommended
    # 就以为 4/8 井的实际两两距离相等。
    interference["has_non_geometric_options"] = f_int > 0 and any(
        o["n_wells"] > 3 for o in options
    )
    if recommended is not None:
        recommended["interference_scope"] = (
            "mean_field_not_a_well_layout"
            if f_int > 0 and recommended["n_wells"] > 3
            else "pairwise_geometry" if f_int > 0 else "independent_wells"
        )

    # ─── 前沿（单井，压力 → 风险 → 速率）────────────────
    frontier = _frontier(p_h, cap_bar, p_star, critical, depth, j, density)

    notes = _notes(
        status=status, limit=limit, target=target, p_h=p_h, j=j,
        skin=skin, recommended=recommended, verification=verification,
        sampler=sampler, cap_ratio=cap_ratio, th=th, critical=critical,
        rel_sigma=rel_sigma, current=current,
        rate_cap_mt_per_year=rate_cap_mt_per_year, q_req=q_req, density=density,
        interference=interference,
    )

    text = _text(
        status=status, current=current, recommended=recommended,
        limit=limit, target=target, verification=verification,
        p_h=p_h, q_req=q_req, density=density, th=th, headroom=headroom,
        critical=critical,
    )

    return {
        "status": status,
        "depth_m": depth,
        "n": len(critical),
        "rel_sigma": float(rel_sigma),
        "dist": dist,
        "dist_label": Marginal(dist, rel_sigma).perturbation_label(),
        "corr": sampler.corr,
        "independent": sampler.independent,
        "p_target": target,
        "years": float(years),
        "current": current,
        "limit": {
            "pressure_bar": limit["pressure_bar"],
            "k_max": limit["k_max"],
            "hits": limit["hits"],
            "p_upper": limit["p_upper"],
            "binding": limit["binding"],
            "feasible": limit["feasible"],
        },
        "well_options": options,
        "recommended": recommended,
        "headroom": headroom,
        "verification": verification,
        "frontier": frontier,
        "physics": physics,
        "binding": (recommended or {}).get("binding") or limit["binding"],
        "notes": notes,
        "text": text,
        "provenance": _provenance(thresholds=th, skin=skin),
    }


# ══════════════════════════════════════════════════════════
# 内部工具
# ══════════════════════════════════════════════════════════
def _density(pressure_bar: float, temperature_celsius: float) -> float:
    """
    质量换算用的 CO₂ 密度。

    取静水压力（注前地层压力）而不是井底压力：注入时压力更高、密度更大，
    所以用注前密度会把"Mt/yr"算小 —— 方向是保守的（不会高估封存量）。
    """
    from utils.co2_properties import co2_density_kg_m3

    try:
        density = float(co2_density_kg_m3(pressure_bar, temperature_celsius))
    except (ValueError, ArithmeticError, OverflowError) as exc:
        raise ValueError(
            "CO₂ 物性计算失败，无法从体积速率换算质量速率；拒绝用猜测密度给出注入方案"
        ) from exc
    if not math.isfinite(density) or density <= 0:
        raise ValueError("CO₂ 物性返回非有限正密度，无法给出注入方案")
    return density


def _binomial_se(p: float, n: int) -> float:
    """二项比例的标准误，用于判断样本内/样本外差异是否只是噪声。"""
    if n <= 0:
        return 0.0
    return math.sqrt(max(p * (1.0 - p), 0.0) / n)


def _derived_seed(seed: int | None) -> int | None:
    """样本外复核的种子：与求解用的种子错开，但仍是确定的（可复现）。"""
    if seed is None:
        return None
    return int(seed) + 7919      # 第 1000 个素数，取它只为"明显不同"


def _frontier(
    p_h: float, cap_bar: float, p_star: float,
    critical: list[float], depth: float, j: float, density: float,
    points: int = 9,
) -> list[dict]:
    """
    单井的"压力 → 风险 → 速率"曲线。

    上限取 ``min(cap_bar, p_h + 1.5·(p_star − p_h))``：把推荐点放在曲线的
    2/3 处，两侧都能看到 —— 只画到推荐点会让读者以为右边没有风险，
    画到破裂压力又会让左边的关键区间被压成一条竖线。
    """
    if p_star <= p_h:
        return []
    top = min(cap_bar, p_h + 1.5 * (p_star - p_h))
    if top <= p_h:
        return []
    out = []
    for i in range(points):
        p = p_h + (top - p_h) * i / (points - 1)
        q = rate_m3_per_day(p, depth, j, 1)
        r = risk_at_pressure(p, critical)
        out.append({
            "pressure_bar": round(p, 2),
            "rate_m3_per_day": round(q, 1),
            "rate_mt_per_year": round(mass_mt_per_year(q, density), 3),
            "p_leak": r["p"],
            "p_lo": r["p_ci"][0],
            "p_hi": r["p_ci"][1],
            "beyond_budget": r["p_upper"] > 0 and r["p"] > 0,
        })
    return out


def _notes(
    *, status: str, limit: dict, target: float, p_h: float, j: float,
    skin: float, recommended: dict | None, verification: dict,
    sampler: JointSampler, cap_ratio: float, th: dict,
    critical: list[float], rel_sigma: float, current: dict,
    rate_cap_mt_per_year: float | None, q_req: float, density: float,
    interference: dict | None = None,
) -> list[str]:
    """把"这个建议在什么条件下成立"逐条写出来。不写就等于默认它无条件成立。"""
    notes: list[str] = []

    if current and current.get("rate_m3_per_day", 0.0) <= 0.0:
        notes.append(
            f"当前井底压力 {current['pressure_bar']:.0f} bar 低于静水压力 "
            f"{p_h:.0f} bar，注入速率为 0（注不进）：下面的推荐是风险预算允许的"
            "能力上限，不是对当前方案的修正"
        )
    if rate_cap_mt_per_year is not None and status == STATUS_INFEASIBLE:
        notes.append(
            f"目标速率可能已超过设施/许可上限 {rate_cap_mt_per_year:.2f} Mt/yr；"
            "这类无解是设施约束造成的，加井不能解决"
        )

    if status == STATUS_NO_INJECTION:
        notes.append(
            f"在 {target * 100:.1f}% 的风险预算下，井底压力上限 "
            f"{limit['pressure_bar']:.1f} bar 不高于静水压力 {p_h:.1f} bar，"
            "注不进去：需要提高风险预算、换更深的储层，或先做增注措施"
        )
    if status == STATUS_INFEASIBLE:
        # 区分两种无解：设施上限造成的（加井无用）与地层注入能力造成的
        # （加井有效但仍不够）。混成一句会让读者按错误的方向去整改。
        facility = (
            rate_cap_mt_per_year is not None
            and mass_mt_per_year(q_req, density) > rate_cap_mt_per_year
        )
        if not facility:
            notes.append(
                "给定的井数组合都达不到要求的速率：即便按当前干扰情景计算"
                "仍不够；可能由注入指数（渗透率 × 厚度）或井场几何约束决定，"
                "不能仅归因于单井能力"
            )
    if recommended is not None and not verification.get("agrees", True):
        notes.append(
            f"样本外复核的概率 {verification['p'] * 100:.1f}% 与样本内 "
            f"{recommended['risk']['p'] * 100:.1f}% 差异超出二项噪声："
            "该方案贴着阈值，建议留出裕度（把风险预算调低一点再解）"
        )

    if skin <= 0.0:
        notes.append(
            "注入指数按无表皮计算（skin=0），这是乐观假设：实际完井污染会"
            f"使注入能力按 ln(re/rw)/(ln(re/rw)+S) 下降（S=5 约剩 63%），"
            "所以这里的速率是压力约束下的**上限**，不是工程可实现值"
        )
    else:
        notes.append(f"注入指数已计入表皮系数 S={skin:g}")

    notes.append(
        "本模型不含井筒摩阻与非达西（高速湍流）：两项都会让实际速率低于此处数值"
    )
    if interference and interference.get("active"):
        notes.append(
            f"井间干扰已计入（均值场情景）：等效井距 {interference['spacing_m']:g} m、"
            f"泄油半径 {interference['drainage_radius_m']:g} m、"
            f"f={interference['factor']:.4f}。"
            "速率按 q = n·J·(p−P_h)/(1+(n−1)f) 计算；固定 f 下的饱和"
            "只是均值场模型的代数极限，不是有限井场的工程结论"
        )
        if interference.get("has_non_geometric_options"):
            notes.append(
                "警告：平面中超过 3 口井不可能两两等距；4/8 井数字仅作等效耦合"
                "敏感性情景，不代表实际井位，不能用于场地级选井数。误差方向未知"
            )
    else:
        notes.append(
            "未计井间干扰（"
            + (interference or {}).get("reason", "未提供井距")
            + "）：'加井线性增注'在本模型里成立，真实井网下会偏低"
        )
    notes.append(
        "风险只由井底压力决定，与速率无关；同一压力下加井只增加速率、"
        "不改变风险。所以'又安全又注得多'在本模型里只有加井一条路"
    )
    if not sampler.independent:
        notes.append(
            "深度与压力按相关抽样（相关系数为 copula 参数，非边缘 Pearson）"
        )
    # 风险曲线为什么是阶跃的：这一步不解释，读者会以为"压力再加 5 bar 也没事"
    if len(critical) >= 2:
        lo, hi = critical[0], critical[-1]
        notes.append(
            f"风险对压力是阶跃型的：{lo:.0f}–{hi:.0f} bar 这条 {hi - lo:.0f} bar 的"
            f"带内命中率从 0 走到 1，带外两侧分别是'确定安全'与'确定压裂'。"
            f"带宽由 ±{rel_sigma * 100:.0f}% 扰动决定，不是物理事实 —— "
            "真实破裂压力梯度的离散度更大，实际曲线会更缓、更早开始抬升"
        )
    if abs(cap_ratio - 1.0) > 1e-9:
        notes.append(
            f"井底压力另受设计余量约束：上限 = {cap_ratio:g} × 破裂压力"
            f"（{limit['cap_bar']:.1f} bar）"
        )
    if th.get("source") == "env":
        notes.append("风险预算取自环境变量覆盖后的 warn 阈值，报告须写明实际生效值")
    notes.append(
        "压力为井底压力（BHP）；地面注入压力 = BHP − 井筒液柱压力，本模型不换算"
    )
    return notes


def _risk_phrase(risk: dict) -> str:
    """
    把"命中 0 次"写成它实际的意思。

    ``0/1000`` 不是"零风险"，是"这批样本里没看到"。写成"压裂概率 0.0%"
    会被读成"不会压裂"，而 0/1000 的 95% 上界是 0.4% —— 这个上界才是
    可以引用的数。这一条在报告里尤其重要：报告正文只允许出现工具给出的
    数字，所以措辞必须在工具这一层就正确。
    """
    p = risk.get("p", 0.0)
    lo, hi = risk.get("p_ci") or [0.0, 1.0]
    if risk.get("hits", 0) == 0:
        return (
            f"未观测到压裂（0/{risk.get('n', 0)} 次，95% 上界 {hi * 100:.1f}%）"
        )
    return (
        f"压裂概率 {p * 100:.1f}%（95% 区间 {lo * 100:.1f}%–{hi * 100:.1f}%）"
    )


def _text(
    *, status: str, current: dict, recommended: dict | None, limit: dict,
    target: float, verification: dict, p_h: float, q_req: float,
    density: float, th: dict, headroom: dict, critical: list[float],
) -> str:
    """一句话结论。数字全部来自上面的计算，不做二次加工。"""
    if not current and recommended is None:
        return "注入策略优化：参数不足，未给出方案"

    parts: list[str] = []
    if current:
        if current.get("rate_m3_per_day", 0.0) <= 0.0:
            parts.append(
                f"当前井底压力 {current['pressure_bar']:.0f} bar 低于静水压力 "
                f"{p_h:.0f} bar，注不进（速率 0）"
            )
        else:
            over = current["risk"]["p"] > target
            parts.append(
                f"当前井底压力 {current['pressure_bar']:.0f} bar 对应 "
                f"{current['rate_mt_per_year']:.2f} Mt/yr，"
                + _risk_phrase(current["risk"])
                + ("，超出风险预算" if over else "")
            )

    if status == STATUS_NO_INJECTION:
        parts.append(
            f"在 {target * 100:.0f}% 风险预算下无可行方案：压力上限 "
            f"{limit['pressure_bar']:.0f} bar 低于静水压力 {p_h:.0f} bar"
        )
        return "；".join(parts) + "。"

    if recommended is None:
        return "；".join(parts) + "。"

    rec_r = recommended["risk"]
    out_p = verification.get("p", rec_r["p"])
    out_n = verification.get("n", rec_r.get("n", 0))
    out_hi = (verification.get("p_ci") or [0.0, 1.0])[1]
    cur_rate = (current or {}).get("rate_mt_per_year")
    rate_note = ""
    if cur_rate and status != STATUS_INFEASIBLE:
        change = recommended["rate_mt_per_year"] / max(cur_rate, 1e-9) - 1.0
        if abs(change) < 0.02:
            rate_note = "，与当前速率持平"
        else:
            rate_note = f"，较当前 {change * 100:+.0f}%"
    parts.append(
        f"推荐 {recommended['n_wells']} 口井、井底压力 "
        f"{recommended['pressure_bar']:.0f} bar（破裂压力 {limit['cap_bar']:.0f} bar 的 "
        f"{recommended['pressure_bar'] / limit['cap_bar'] * 100:.0f}%），"
        f"速率 {recommended['rate_mt_per_year']:.2f} Mt/yr{rate_note}"
    )
    if status == STATUS_INFEASIBLE:
        short = headroom.get("shortfall_mt_per_year", 0.0)
        parts.append(
            f"但仍低于目标 {short:.2f} Mt/yr（能力受限，非风险受限）"
        )
    # 样本外复核用独立的 1000 次抽样，单独报，不混进样本内的数
    if verification:
        if out_p <= 0.0:
            parts.append(
                f"样本外复核未观测到压裂（0/{out_n} 次，95% 上界 "
                f"{out_hi * 100:.1f}%，预算 {target * 100:.0f}%）"
            )
        else:
            parts.append(
                f"样本外复核压裂概率 {out_p * 100:.1f}%（预算 {target * 100:.0f}%）"
            )
    if current and headroom.get("fraction") is not None and status == STATUS_OK:
        frac = headroom["fraction"]
        if frac >= 0.02:
            parts.append(
                f"该井数下风险预算允许的最高压力是 "
                f"{headroom.get('pressure_max_bar', limit['pressure_bar']):.0f} bar，"
                f"对应速率上限 {headroom.get('rate_max_mt_per_year', 0.0):.2f} Mt/yr"
                f"（比目标多 {frac * 100:.0f}%）"
            )
    return "；".join(parts) + "。"


def _provenance(thresholds: dict, skin: float) -> dict:
    """每个数的来路。报告引用建议速率时必须能回答"凭什么"。"""
    return {
        "fracture_ratio": (
            f"{pm.FRACTURE_RATIO:g}× 静水压力（pressure_model.FRACTURE_RATIO，"
            "常用区间 1.4–1.8 的中值）"
        ),
        "injectivity": (
            "径向稳态 Darcy J = 2πkh/(μ(ln(re/rw)+S))，"
            "utils.relperm.injectivity_index；单相、均质"
        ),
        "viscosity": "CO₂ 黏度常数 0.06 mPa·s（utils.co2_properties，区间 0.03–0.1）",
        "density": "Peng-Robinson 状态方程（utils.co2_properties），按注前静水压力取值",
        "p_target": (
            f"风险预算取 warn 阈值 {thresholds['warn'] * 100:.0f}%"
            f"（{thresholds.get('provenance', '')}）"
        ),
        "constraint": (
            "约束用 Wilson 95% 上界而非点估计：点估计 4.9% 与 11% 在 n=100 下"
            "无法区分，不能称'低于 5%'"
        ),
        "skin": f"表皮系数 S={skin:g}（0 = 理想完井，属乐观假设）",
    }
