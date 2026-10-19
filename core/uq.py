"""
不确定性量化（Uncertainty Quantification, UQ）
==============================================
地质参数从来不是确定值：同一口井的压力、同一层的孔隙度，换个时间、换个
测量手段就会变。把 ``ρ = 1.9·P/T`` 算出来的那个数当成"真值"写进报告，
是在用一个点估计掩盖一整片可能性 —— 工业界不会接受这种结论。

本模块提供蒙特卡洛传播的最小实现（纯标准库，无 numpy）：

    x_i = x_nominal · (1 + ε_i),  ε_i ~ U(-δ, +δ)        i = 1..N
    y_i = f(x_i)
    报告 = mean(y) + [p5(y), p95(y)]      ← 90% 区间

三个刻意的设计决定
------------------
1. **δ 是"最大扰动幅度"，不是标准差。** 现场说"压力 ±5%"，指的是仪表/解释
   误差的**边界**，而不是 1σ；按边界做均匀抽样，输出的 90% 区间恰好落在
   ±4.5% 左右（250 bar / 80℃ → 1.35 kg/m³，区间约 1.29–1.41），与工程直觉一致。
   若把 ±5% 当成 1σ 再做正态抽样，90% 区间会撑到 ±8.2%，结论看起来比实际
   更不确定 —— 两种口径都能自圆其说，但**必须明确选一种并写清楚**，
   最糟的是报告里写"±5%"、代码里按 1σ 算，读者无从判断。
   需要正态口径时显式传 ``dist="normal"``（此时 δ 就是 1σ）。
2. **用百分位区间，不用 mean ± 1.645·std。** 状态方程在参数边界附近是非线性的，
   输出分布会偏斜；正态假设算出来的对称区间会漏掉真实的尾部。
3. **种子可注入。** 生产环境不固定种子（每次都是新的抽样），但测试必须能
   复现 —— 否则"区间是不是 1.28~1.42"这种断言永远只能靠眼睛看。

为什么不引入 numpy / scipy
--------------------------
前端（app.py）有一条硬规矩：不许 import matplotlib，也不许依赖 numpy。
本模块的格式化结果会被前端直接展示，保持零依赖让它两边都能用；
N=100 的循环用标准库完全够快（微秒级）。
"""
from __future__ import annotations

import math
import os
import random
from typing import Callable, Iterable, Sequence

# 默认模拟次数。100 次是"能在毫秒内跑完"与"分位数还有意义"的平衡点：
# 100 个样本的 5% 分位约等于第 5 个值，再少就退化成 min/max 了。
DEFAULT_N = 100

# 默认输入不确定度：最大相对扰动幅度 ±5%
DEFAULT_REL_SIGMA = 0.05

# 默认分布：均匀（有界）。见模块文档第 1 条。
DEFAULT_DIST = "uniform"

# 置信水平（双侧 90% → 取 p5 / p95）
CI_LEVEL = 0.90

# 输出展示的有效位数
DEFAULT_DIGITS = 2


def _seed_from_env() -> int | None:
    raw = (os.getenv("UQ_SEED") or "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def make_rng(seed: int | None = None) -> random.Random:
    """
    统一的随机源构造。``seed`` 为 None 时先读 ``UQ_SEED`` 环境变量。

    单独暴露出来，是为了让"需要同时扰动多个输入"的模型（注入风险要同时扰
    深度和压力）复用同一套种子逻辑 —— 否则那些地方会各自 ``random.Random()``，
    测试里就再也复现不了。
    """
    if seed is None:
        seed = _seed_from_env()
    return random.Random(seed)


def draw_relative(rng: random.Random, rel_sigma: float, dist: str = DEFAULT_DIST) -> float:
    """
    抽一个相对扰动 ε（``x' = x·(1+ε)``）。

    ``uniform``：ε ~ U(-δ, +δ)，δ 就是"最大扰动幅度"。
    ``normal`` ：ε ~ N(0, δ)，此时 δ 是 1σ（无界，但 3σ 之外概率可忽略）。
    """
    rel_sigma = abs(float(rel_sigma))
    if dist == "normal":
        return rng.normalvariate(0.0, rel_sigma)
    return rng.uniform(-rel_sigma, rel_sigma)


def percentile(values: Sequence[float], q: float) -> float | None:
    """
    线性插值百分位（与 numpy 默认口径一致）。

    ``q`` 取 0~100。空序列返回 None —— 调用方必须显式处理"没有样本"，
    而不是拿到一个 0 当真值用。
    """
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    if q <= 0:
        return float(ordered[0])
    if q >= 100:
        return float(ordered[-1])

    pos = (len(ordered) - 1) * q / 100.0
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(ordered[lo])
    frac = pos - lo
    return float(ordered[lo] + (ordered[hi] - ordered[lo]) * frac)


def monte_carlo(
    fn: Callable[[float], float],
    nominal: float,
    rel_sigma: float = DEFAULT_REL_SIGMA,
    n: int = DEFAULT_N,
    seed: int | None = None,
    dist: str = DEFAULT_DIST,
    lower_bound: float | None = None,
    upper_bound: float | None = None,
    keep_samples: bool = False,
) -> dict:
    """
    对标称输入做 N 次扰动，把结果分布统计出来。

    参数
    ----
    fn           : 单参数物理模型（``x -> y``），必须无副作用
    nominal      : 标称输入值
    rel_sigma    : 相对不确定度。uniform 下是最大扰动幅度，normal 下是 1σ
                   （0.05 表示 ±5%）
    n            : 模拟次数
    seed         : 随机种子；None 时读 ``UQ_SEED`` 环境变量，再没有就真随机
    dist         : ``normal``（默认，无界）或 ``uniform``（±√3σ，同方差）
    lower_bound  : 输出的物理下界（如密度/压力必须 > 0）
    upper_bound  : 输出的物理上界
    keep_samples : 是否把 100 个样本一起返回（前端画直方图时用）

    返回
    ----
    ``{"n", "rel_sigma", "dist", "mean", "std", "p5", "p50", "p95",
       "min", "max", "cv", "samples"}``
    """
    n = max(2, int(n))
    rel_sigma = abs(float(rel_sigma))
    nominal = float(nominal)

    rng = make_rng(seed)

    samples: list[float] = []
    for _ in range(n):
        z = draw_relative(rng, rel_sigma, dist=dist)

        x = nominal * (1.0 + z)
        if lower_bound is not None and x < lower_bound:
            x = lower_bound
        if upper_bound is not None and x > upper_bound:
            x = upper_bound

        y = float(fn(x))
        if lower_bound is not None and y < lower_bound:
            y = lower_bound
        if upper_bound is not None and y > upper_bound:
            y = upper_bound
        samples.append(y)

    mean = sum(samples) / len(samples)
    var = sum((v - mean) ** 2 for v in samples) / (len(samples) - 1)
    std = math.sqrt(var)

    out = {
        "n": n,
        "rel_sigma": rel_sigma,
        "dist": dist,
        "mean": mean,
        "std": std,
        "p5": percentile(samples, (1 - CI_LEVEL) / 2 * 100),
        "p50": percentile(samples, 50),
        "p95": percentile(samples, (1 + CI_LEVEL) / 2 * 100),
        "min": min(samples),
        "max": max(samples),
        # 变异系数：σ/μ。它才是"这个结论有多确定"的单一指标，
        # 绝对标准差会随量纲变化，没法跨参数比较。
        "cv": (std / mean) if mean else 0.0,
        "nominal": nominal,
    }
    if keep_samples:
        out["samples"] = samples
    return out


def monte_carlo_outcomes(
    fn: Callable[[float], str],
    nominal: float,
    rel_sigma: float = DEFAULT_REL_SIGMA,
    n: int = DEFAULT_N,
    seed: int | None = None,
    dist: str = DEFAULT_DIST,
) -> dict:
    """
    分类型蒙特卡洛：``fn`` 返回类别（如 ``ok`` / ``risky`` / ``under``）。

    返回 ``{"n": 100, "counts": {...}, "probabilities": {...}}``。
    概率是"模拟中出现该类别的比例"，不是模型给的主观置信度 —— 这是两者
    最重要的区别：前者可复现、可审计，后者不可证伪。
    """
    n = max(2, int(n))
    rel_sigma = abs(float(rel_sigma))
    rng = make_rng(seed)

    counts: dict[str, int] = {}
    for _ in range(n):
        z = draw_relative(rng, rel_sigma, dist=dist)
        label = str(fn(float(nominal) * (1.0 + z)))
        counts[label] = counts.get(label, 0) + 1

    return {
        "n": n,
        "rel_sigma": rel_sigma,
        "dist": dist,
        "counts": counts,
        "probabilities": {k: v / n for k, v in counts.items()},
    }


def format_value(value: float, digits: int = DEFAULT_DIGITS) -> str:
    """
    数值格式化：小量级保留更多位，避免 0.00 这种没信息量的输出。

    ``< 1`` 的数值多留两位有效小数，但去掉尾随的 0 —— "0.7850" 看起来
    像仪器读数精度，而它其实只是格式化留下的尾巴。
    """
    if value is None:
        return "—"
    if abs(value) >= 100:
        return f"{value:.1f}"
    if abs(value) >= 1:
        return f"{value:.{digits}f}"
    return f"{value:.{digits + 2}f}".rstrip("0").rstrip(".")


def format_interval(summary: dict, unit: str = "", digits: int = DEFAULT_DIGITS) -> str:
    """
    把统计结果写成一句可直接进报告的话::

        "1.35 kg/m³（90% 置信区间 1.29–1.41，100 次蒙特卡洛模拟，输入 ±5% 扰动）"

    这段文本是模型的输入，也是报告的兜底文案 —— 所以必须自带全部限定条件
    （模拟次数、扰动幅度、置信水平），否则读者无法判断这个区间可不可信。
    """
    if not summary:
        return ""
    mean = summary.get("mean")
    lo = summary.get("p5")
    hi = summary.get("p95")
    if mean is None or lo is None or hi is None:
        return ""

    unit_part = f" {unit}" if unit else ""
    sigma = summary.get("rel_sigma") or 0.0
    n = summary.get("n") or 0
    return (
        f"{format_value(mean, digits)}{unit_part}"
        f"（90% 置信区间 {format_value(lo, digits)}–{format_value(hi, digits)}{unit_part}，"
        f"{n} 次蒙特卡洛模拟，输入 ±{sigma * 100:.0f}% 扰动）"
    )


def summarize(
    summary: dict,
    unit: str = "",
    digits: int = DEFAULT_DIGITS,
    keep_samples: bool = False,
) -> dict:
    """
    把 ``monte_carlo`` 的原始结果整理成"可直接进 state / JSON / 前端"的结构。

    统一在这里补 ``text`` 字段，避免节点、API、前端三处各写一遍格式化 ——
    三份实现迟早会漂移，然后报告里的区间和看板上的区间对不上。
    """
    if not summary:
        return {}

    out = {
        "mean": round(float(summary["mean"]), 6),
        "std": round(float(summary["std"]), 6),
        "p5": round(float(summary["p5"]), 6),
        "p50": round(float(summary["p50"]), 6),
        "p95": round(float(summary["p95"]), 6),
        "min": round(float(summary["min"]), 6),
        "max": round(float(summary["max"]), 6),
        "cv": round(float(summary["cv"]), 6),
        "n": int(summary["n"]),
        "rel_sigma": float(summary["rel_sigma"]),
        "dist": summary.get("dist", "normal"),
        "nominal": round(float(summary.get("nominal", 0.0)), 6),
        "unit": unit,
        "ci_level": CI_LEVEL,
        "text": format_interval(summary, unit=unit, digits=digits),
    }
    if keep_samples and summary.get("samples"):
        out["samples"] = [round(float(v), 6) for v in summary["samples"]]
    return out


def uq_block(uq_summary: dict) -> str:
    """
    生成注入报告 Prompt 的"概率性结论"块。

    与 ``core.temporal.build_conflict_notice`` 同一个思路：把硬约束写成
    代码生成的文本，而不是指望 System Prompt 里的一句"请包含不确定性"。
    """
    if not uq_summary:
        return ""

    lines = ["【不确定性量化结果 —— 报告必须以此为准】"]
    for key, label in (("density", "CO2 密度"), ("injection", "注入风险")):
        item = uq_summary.get(key) or {}
        text = item.get("text")
        if text:
            lines.append(f"- {label}：{text}")
        elif item.get("probabilities"):
            probs = item["probabilities"]
            detail = "、".join(
                f"{k} {v * 100:.0f}%" for k, v in sorted(probs.items(), key=lambda kv: -kv[1])
            )
            lines.append(
                f"- {label}：{detail}"
                f"（{item.get('n', 0)} 次蒙特卡洛，输入 ±{(item.get('rel_sigma') or 0) * 100:.0f}% 扰动）"
            )

    if len(lines) == 1:
        return ""

    lines += [
        "",
        "写作要求（违反即视为报告不合格）：",
        "1. 上述数值必须按「均值 + 90% 置信区间」的形式写入正文，"
        "**禁止只写单一确定值**（例如只写 1.35 kg/m³ 而不给区间）。",
        "2. 概率性结论必须带上限定条件（模拟次数与扰动幅度），"
        "不要改写成\"约为\"、\"大约\"这类模糊表述。",
        "3. 若给出风险判断，必须同时给出概率（如\"压裂风险概率 12%\"），"
        "不要用\"存在一定风险\"这种无法证伪的措辞。",
        "",
    ]
    return "\n".join(lines)
