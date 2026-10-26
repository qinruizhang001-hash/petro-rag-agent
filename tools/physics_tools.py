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
from core.risk_policy import decide
from core.sensitivity import DEFAULT_BASE, sobol_event
from core.uq import (
    DEFAULT_DIST,
    DEFAULT_N,
    DEFAULT_REL_SIGMA,
    JointSampler,
    Marginal,
    format_interval,
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


def _as_corr_matrix(corr) -> list[list[float]] | None:
    """
    把"标量相关系数"或"2×2 矩阵"统一成矩阵；``None`` 表示独立。

    只接受 2 个参数的相关（深度、压力），所以标量形式是安全的：
    二维情形下一个数就足以描述相关结构，让调用方去写 2×2 矩阵只是徒增出错面。
    """
    if corr is None:
        return None
    if isinstance(corr, (int, float)):
        rho = float(corr)
        if not -1.0 <= rho <= 1.0:
            raise ValueError(f"相关系数必须在 [-1, 1] 内，收到 {rho}")
        return [[1.0, rho], [rho, 1.0]]
    return corr


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
    dist: str = DEFAULT_DIST,
) -> dict:
    """
    密度 + 不确定性。扰动的是**压力输入**（现场最容易波动、也最常被问到的量），
    温度按标称值处理。

    ``dist`` 默认均匀（压力表误差有量程边界）。若把压力理解为"多个独立小误差
    的叠加"则用 ``normal``；若关心的是**密度**本身跨井/跨层的乘性差异，
    用 ``lognormal`` 更贴切（密度恒正、右偏）。三者给出的 90% 区间宽度不同，
    所以 ``dist_label`` 会随结果一起返回并被写进文案。

    返回 ``summarize`` 结构（mean / p5 / p95 / cv / dist_label / text …）。
    """
    result = monte_carlo(
        lambda p: co2_density_value(p, temperature_celsius),
        nominal=float(pressure_bar),
        rel_sigma=rel_sigma,
        n=n,
        seed=seed,
        dist=dist,
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
    dist: str = DEFAULT_DIST,
    corr: float | list[list[float]] | None = None,
    with_decision: bool = True,
) -> dict:
    """
    注入风险的概率分布：同时扰动**深度**与**注入压力**，统计三种结局的频率。

    为什么两个都扰动：这两个量恰好一个来自地质解释（深度）、一个来自地面
    设备（压力），误差来源独立。只扰动压力会得到一个偏窄的风险分布，
    让"压裂概率"看起来比实际低。

    相关性（``corr``）
    ------------------
    默认 ``None`` = **独立**。这是一个假设，不是事实，所以它会被写进返回值
    （``independent`` / ``corr`` / ``corr_empirical``）。什么时候该给相关：
    当深度和压力来自**同一次解释**时（例如压力由深度乘压力梯度算出来的），
    两者误差同向，独立抽样会低估"深度偏大且压力偏高"这种共同越界的概率。
    实测（见 tests/test_uq_advanced.py）：ρ=0.9 时 P(risky) 与 ρ=0 的差异
    在阈值附近可达数个百分点，足以改变决策等级。

    ``corr`` 可给标量（两参数的相关系数）或 2×2 矩阵。

    返回
    ----
    ``{"n", "rel_sigma", "dist", "dist_label", "probabilities", "counts",
       "verdict", "text", "decision", "corr", "independent", "corr_empirical",
       "priors"}``

    其中 ``verdict`` 是标称值下的确定性判定（与 ``pressure_model.verdict_of``
    一致），概率说明"这个判定有多稳"，``decision`` 说明"这个概率是否越过
    告警线"——三者是三个不同的问题，混在一句"存在一定风险"里就全丢了。
    """
    n = max(2, int(n))
    matrix = _as_corr_matrix(corr)
    dims = [
        {"name": "深度", "kind": dist, "rel_sigma": rel_sigma},
        {"name": "注入压力", "kind": dist, "rel_sigma": rel_sigma},
    ]
    sampler = JointSampler(dims, corr=matrix, seed=seed)

    counts: dict[str, int] = {}
    for _ in range(n):
        d, p = sampler.draw_values([float(depth_m), float(injection_pressure_bar)])
        label = pm.verdict_of(d, p)
        counts[label] = counts.get(label, 0) + 1

    probabilities = {k: v / n for k, v in counts.items()}
    nominal_verdict = pm.verdict_of(depth_m, injection_pressure_bar)

    detail = "、".join(
        f"{_VERDICT_LABELS.get(k, k)} {v * 100:.0f}%"
        for k, v in sorted(probabilities.items(), key=lambda kv: -kv[1])
    )
    dist_label = Marginal(dist, rel_sigma).perturbation_label()
    corr_note = "" if sampler.independent else f"，深度与压力相关系数 {matrix[0][1]:.2f}"
    text = (
        f"{detail}（标称判定：{_VERDICT_LABELS.get(nominal_verdict, nominal_verdict)}；"
        f"{n} 次蒙特卡洛，深度与压力各{dist_label}{corr_note}）"
    )

    out = {
        "n": n,
        "rel_sigma": float(rel_sigma),
        "dist": dist,
        "dist_label": dist_label,
        "counts": counts,
        "probabilities": probabilities,
        "verdict": nominal_verdict,
        "verdict_label": _VERDICT_LABELS.get(nominal_verdict, nominal_verdict),
        "depth_m": float(depth_m),
        "injection_pressure_bar": float(injection_pressure_bar),
        "unit": "",
        "text": text,
        # 相关性的口径必须随结果返回：默认独立这件事本身就是一个假设
        "corr": matrix,
        "independent": sampler.independent,
        "corr_empirical": sampler.empirical_correlation(
            n=2000, seed=seed
        ) if not sampler.independent else [[1.0, 0.0], [0.0, 1.0]],
        "priors": sampler.describe(),
    }
    if with_decision:
        try:
            out["decision"] = decide(probabilities, n)
        except Exception as e:      # 阈值配置错误不该让整份报告消失
            log.warning(f"  [Warning] 决策判定失败（保留概率结果）: {e}")
    return out


def injection_risk_sensitivity(
    depth_m: float,
    injection_pressure_bar: float,
    rel_sigma: float = DEFAULT_REL_SIGMA,
    n_base: int = DEFAULT_BASE,
    seed: int | None = None,
    dist: str = DEFAULT_DIST,
    event: str = "risky",
) -> dict:
    """
    对"是否压裂"这个**风险指标**做 Sobol 敏感性分析：深度和压力各占多少。

    为什么需要它：蒙特卡洛能说"压裂概率 12%"，但说不出"这 12% 主要是深度
    的不确定性造成的，还是压力的"。这两个答案对应的行动完全不同 ——
    前者要去补测深度（地震/测井），后者要去校压力表或降注入压力。
    没有这一步，UQ 只是把点估计换成了区间，读者仍然不知道该动哪个参数。

    被分解的是**事件指示函数** ``1[判定 == risky]``，不是压力值本身：
    报告关心的是"结论会不会翻"，而不是"压力算出来差几 bar"。

    返回 ``sobol_indices`` 结构 + ``text``。**输入假定独立**：这里没有
    相关矩阵参数，因为 Sobol 指数在相关输入下没有定义（见 core.sensitivity）。
    """
    dims = [
        {"name": "深度", "kind": dist, "rel_sigma": rel_sigma},
        {"name": "注入压力", "kind": dist, "rel_sigma": rel_sigma},
    ]
    return sobol_event(
        lambda values: pm.verdict_of(values[0], values[1]),
        dims,
        [float(depth_m), float(injection_pressure_bar)],
        target=event,
        n_base=n_base,
        seed=seed,
    )


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
    coupled = result.get("coupled") or {}
    if coupled and coupled.get("verdict") != result["verdict"]:
        result["text"] += (
            f"；若计入注入层内的孔隙弹性应力路径（Δσh/ΔP="
            f"{coupled.get('path_ratio')}），判定变为 {coupled.get('verdict')}、"
            f"余量 {coupled.get('margin_bar')} bar —— 该口径只对注入层成立"
        )
    return result


def caprock_check(
    depth_m: float,
    injection_pressure_bar: float,
    temperature_celsius: float = 60.0,
    co2_column_height_m: float | None = None,
    injection_rate_m3_per_day: float | None = None,
    trap_area_m2: float | None = None,
) -> dict:
    """
    盖层封闭性判定：毛细管突破（Laplace/Washburn）与力学失效（σhmin + T0）
    两条独立路径，任一条失效即 ``blocking``。

    不给柱高也不给注入速率时返回 ``verdict="no_data"``，只给突破压力与
    可封柱高 —— "没算"和"算出来不合格"必须能区分。

    ``trap_area_m2`` 给出圈闭面积时，同时算圈闭聚集口径的柱高与突破时间，
    与一维口径构成区间。一维口径把 CO₂ 摊在整个羽流面积上，柱高偏小、
    突破时间偏长 —— 那是**乐观**的一侧；圈闭口径是保守的一侧。
    """
    from utils.caprock import caprock_integrity

    result = caprock_integrity(
        depth_m=depth_m,
        injection_pressure_bar=injection_pressure_bar,
        temperature_celsius=temperature_celsius,
        co2_column_height_m=co2_column_height_m,
        injection_rate_m3_per_day=injection_rate_m3_per_day,
        trap_area_m2=trap_area_m2,
    )
    result["text"] = result.get("summary") or (
        f"盖层突破压力 {result['entry_pressure_bar']:.0f} bar，"
        f"可封 CO₂ 柱高 {result['max_column_height_m']:.0f} m"
    )
    trap = result.get("trap") or {}
    if trap:
        result["text"] += (
            f"；圈闭口径同一时刻柱高 {trap['column_height_m']:.0f} m"
            f"（一维口径 {trap['plume_column_height_m']:.0f} m），"
            f"长到可封上限需 {trap['fill_days_early']:.0f} 天，"
            f"一维口径需 {trap['fill_days_late']:.0f} 天（早界为保守侧）"
        )
    return result


# ══════════════════════════════════════════════════════════
# 注入策略优化（从"诊断"到"处方"）
# ══════════════════════════════════════════════════════════
def optimize_injection_strategy(
    depth_m: float,
    injection_pressure_bar: float | None = None,
    storage_rate_mt_per_year: float | None = None,
    well_counts: tuple[int, ...] = (1, 2, 4, 8),
    p_target: float | None = None,
    years: float = 20.0,
    rel_sigma: float = DEFAULT_REL_SIGMA,
    n: int = 1000,
    seed: int | None = None,
    dist: str = DEFAULT_DIST,
    corr: float | list[list[float]] | None = None,
    permeability_md: float = 100.0,
    thickness_m: float = 50.0,
    temperature_celsius: float = 60.0,
    skin: float = 0.0,
    cap_ratio: float = 1.0,
    rate_cap_mt_per_year: float | None = None,
    well_spacing_m: float | None = None,
    drainage_radius_m: float = 500.0,
    interference_coefficient: float | None = None,
) -> dict:
    """
    给定当前方案，反解一个满足风险预算的注入方案（井数 / 井底压力 / 速率）。

    ``well_spacing_m`` 默认 ``None`` = 按独立井计算（无井间干扰），与引入该
    参数之前逐位一致；给了之后按稳态径向叠加计入干扰，加井不再线性增注。
    井距是**设计输入**，所以没有默认值 —— 替调用方假定一个井距，等于替
    他做了一个他没做过的决定，而这个决定会直接改变推荐方案。

    ``interference_coefficient`` 是另一条入口：井位还没定时，评审真正想问的
    往往是"假设 f=0.2 会怎样"，而不是"300 m 井距会怎样"。直接给 f 跳过几何
    反算，返回值的 ``interference_source`` 会写 ``user_coefficient``，
    ``notes`` 也会明说这是假设而非某个可实现井网的结果。两者不能同时给。

    与 ``injection_risk_uq`` 的关系：那个工具回答"当前方案有多危险"，
    这个工具回答"改成什么才不危险"。两者用**同一套判据与同一批抽样**
    （同 seed 下压裂命中数逐样本相等，见 ``tests/test_optimize.py``），
    所以报告里同时引用两个结论时不会互相矛盾。

    为什么 ``n`` 默认 1000 而不是 100：约束用的是 Wilson 95% 上界，
    n=100 时上界 ≤5% 只允许 0 次命中（0/100 的上界 3.6%，1/100 是 5.5%），
    约束会退化成"一个样本都不许越界"。1000 次纯 Python 约几十毫秒。

    返回 ``core.optimize.optimize_plan`` 的结构 + ``text``。``status`` 三态：
    ``ok`` / ``infeasible``（能力不足）/ ``no_injection``（连注都注不进）。
    """
    from core.optimize import optimize_plan

    # 干扰口径写进日志：排障时"这次到底有没有计干扰"是最先要分清的一件事，
    # 而三态（未计 / 按井距 / 按给定系数）从结果里看不出区别。
    if interference_coefficient is not None:
        interference_label = f"f={interference_coefficient:g}（用户给定）"
    elif well_spacing_m is not None:
        interference_label = f"井距 {well_spacing_m:g} m"
    else:
        interference_label = "未计"
    if injection_pressure_bar is not None:
        target_label = f"井底压力 {injection_pressure_bar} bar"
    else:
        target_label = f"目标速率 {storage_rate_mt_per_year} Mt/yr"

    log.info(
        f"  [Tool 调用] 注入策略优化：埋深 {depth_m} m、{target_label}，"
        f"风险预算 {p_target if p_target is not None else '默认(warn)'}，"
        f"井间干扰 {interference_label}，{n} 次抽样..."
    )
    result = optimize_plan(
        depth_m,
        injection_pressure_bar=injection_pressure_bar,
        storage_rate_mt_per_year=storage_rate_mt_per_year,
        well_counts=well_counts,
        p_target=p_target,
        years=years,
        rel_sigma=rel_sigma,
        n=n,
        seed=seed,
        dist=dist,
        corr=_as_corr_matrix(corr),
        permeability_md=permeability_md,
        thickness_m=thickness_m,
        temperature_celsius=temperature_celsius,
        skin=skin,
        cap_ratio=cap_ratio,
        rate_cap_mt_per_year=rate_cap_mt_per_year,
        well_spacing_m=well_spacing_m,
        drainage_radius_m=drainage_radius_m,
        interference_coefficient=interference_coefficient,
    )
    physics = result.get("physics") or {}
    log.info(
        f"  [Tool 调用] 注入策略 → {result['text']} "
        f"（status={result.get('status')}，f={physics.get('interference_factor')}，"
        f"来源={physics.get('interference_source')}，"
        f"回代一致={((result.get('verification') or {}).get('agrees'))}）"
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
