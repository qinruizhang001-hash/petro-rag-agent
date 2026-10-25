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

v1.7 补齐的四个缺口
-------------------
早期实现只做了"均匀分布 + 输入独立 + 不用于决策"，这四条都补上了：

1. **先验分布不只均匀**（``Marginal``）。±5% 到底是"边界"还是"1σ"，
   在均匀与正态下差一倍宽的区间；而对渗透率这类正值、乘性误差的量，
   正确的先验是**对数正态**（右偏、恒正），不是对称分布。现在支持
   ``uniform`` / ``normal`` / ``lognormal`` / ``triangular`` / ``beta``，
   每一种都在 ``PRIOR_NOTES`` 里写清适用场景，并且**分布名字必须进报告文案** ——
   只写"±5%"而不写分布，读者无法判断区间的口径。
2. **参数可以相关**（``JointSampler``）。独立抽样是一个假设，不是事实：
   同一份密度测井同时决定 σv 与 σhmin，同一份深度解释同时决定孔隙压力与
   温度。用一个 **Gaussian copula** 把相关矩阵注入抽样：先在潜变量空间
   取相关正态 ``z = L·g``，再逐维做 ``u_i = Φ(z_i)`` 与 ``ε_i = F_i⁻¹(u_i)``。
   好处是**边缘分布完全不变**，只改联合结构 —— 否则"加了相关性"会顺手把
   每个参数自己的分布也改掉，两个变化混在一起就没法归因。
   注意：输入的相关矩阵 ρ 是**潜变量（copula）相关系数**，不是边缘分布之间
   的 Pearson 相关；对均匀边缘，二者的关系是 ``ρ_s = (6/π)·arcsin(ρ/2)``。
3. **敏感性分析**（见 ``core.sensitivity``）。蒙特卡洛只回答"结论有多不确定"，
   不回答"是谁造成的不确定"。Sobol 分解给出每个输入的一阶贡献与总效应，
   后者减前者就是交互项 —— 这决定了"该去补测哪个参数"。
4. **概率进入决策**（见 ``core.risk_policy``）。P(压裂) = 12% 本身不是结论，
   "12% 是否越过告警线"才是。阈值是**风险接受准则**（政策选择），不是物理
   常数，所以它必须可配置、可追溯，并且要连同概率的**二项置信区间**一起给 ——
   100 次模拟下 5% 与 8% 的区别在统计噪声里，拿它做硬判定是自欺。

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


# ══════════════════════════════════════════════════════════
# 分布函数（标准库实现）
# ══════════════════════════════════════════════════════════
def normal_cdf(z: float) -> float:
    """标准正态 CDF：Φ(z) = 0.5·(1 + erf(z/√2))。"""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


# Acklam 有理逼近的系数（相对误差 < 1.15e-9），随后用一步 Halley 迭代
# 把精度推到 1e-15。为什么不用查表：查表要在边界外推，而 copula 抽样的
# u 会接近 0 和 1（正态尾部的极值点恰好在那里）。
_A = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
      1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
_B = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
      6.680131188771972e+01, -1.328068155288572e+01)
_C = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
      -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
_D = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
      3.754408661907416e+00)
_P_LOW = 0.02425


def inverse_normal_cdf(p: float) -> float:
    """
    标准正态分位数 Φ⁻¹(p)，``p`` 必须落在 (0, 1) 开区间。

    ``p`` 恰好等于 0 或 1 时返回 ``-inf`` / ``inf`` —— 这是数学事实，
    调用方（copula / Sobol 的 u 空间映射）必须自己保证 u 严格在开区间内，
    不能靠这里悄悄夹一下把无界分布变成有界分布。
    """
    p = float(p)
    if p <= 0.0:
        return -math.inf
    if p >= 1.0:
        return math.inf

    if p < _P_LOW:
        q = math.sqrt(-2.0 * math.log(p))
        x = (((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]) / \
            ((((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0)
    elif p <= 1.0 - _P_LOW:
        q = p - 0.5
        r = q * q
        x = (((((_A[0] * r + _A[1]) * r + _A[2]) * r + _A[3]) * r + _A[4]) * r + _A[5]) * q / \
            (((((_B[0] * r + _B[1]) * r + _B[2]) * r + _B[3]) * r + _B[4]) * r + 1.0)
    else:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        x = -(((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]) / \
            ((((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0)

    # Halley 精化：e = Φ(x) - p，一步就够（三阶收敛）
    e = normal_cdf(x) - p
    u = e * math.sqrt(2.0 * math.pi) * math.exp(x * x / 2.0)
    return x - u / (1.0 + x * u / 2.0)


def _betacf(a: float, b: float, x: float) -> float:
    """正则化不完全 Beta 的连分式部分（Lentz 算法，Numerical Recipes §6.4）。"""
    max_it, eps, tiny = 300, 3e-16, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, max_it + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _betainc(a: float, b: float, x: float) -> float:
    """正则化不完全 Beta 函数 I_x(a, b)。"""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln_beta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(ln_beta + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def _beta_ppf(u: float, a: float, b: float) -> float:
    """
    Beta 分布分位数：对 I_x(a,b) 做二分求逆。

    二分而不是 Newton：I_x 在 a,b 很小时两端极陡，Newton 会冲出 (0,1) 区间；
    二分 60 次就有 1e-18 的区间宽度，比 Newton 慢不了几个微秒，却不会失败。
    """
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _betainc(a, b, mid) < u:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-14:
            break
    return 0.5 * (lo + hi)


# ══════════════════════════════════════════════════════════
# 先验分布（作用在相对扰动 ε 上）
# ══════════════════════════════════════════════════════════
MARGINAL_KINDS = ("uniform", "normal", "lognormal", "triangular", "beta")

# 每种先验的适用场景。这段文字会进报告，所以不是"给开发者看的注释"，
# 而是"给读者判断区间口径的依据"。
PRIOR_NOTES = {
    "uniform": "有界、无偏好：只知道误差上限（仪表量程、解释误差）时用，δ 是边界",
    "normal": "无界、对称：误差是多个小效应叠加（中心极限定理），δ 是 1σ",
    "lognormal": "恒正、右偏、乘性误差：渗透率/注入率/流度这类量的标准先验",
    "triangular": "有界、有最可能值：专家给出 min/最可能/max 时用，比均匀更集中",
    "beta": "有界、形状可调：需要集中在中间但不退化成正态时用（a=b 对称）",
}


class Marginal:
    """
    单个参数的**相对扰动**先验分布：``x = nominal · (1 + ε)``。

    为什么作用在相对扰动上而不是绝对值上：现场给出的不确定度几乎总是相对的
    （"压力 ±5%"、"渗透率差一个数量级"），而绝对标准差会随量纲漂移，
    没法跨参数比较。相对化之后 ``rel_sigma`` 对所有参数都是同一个量纲。

    参数
    ----
    kind      : ``MARGINAL_KINDS`` 之一
    rel_sigma : 不确定度。uniform/triangular/beta 下是**半宽**（边界），
                normal 下是 1σ，lognormal 下是 ln 空间的 σ
    mode_rel  : 仅 triangular，最可能值相对标称值的偏移（默认 0，即标称值就是众数）
    alpha/beta: 仅 beta，两个形状参数

    ``from_unit(u)`` 与 ``draw(rng)`` 必须给出**同一个分布**：前者供 copula 与
    Sobol 用（必须能对给定的 u 反解），后者供普通蒙特卡洛用（更快，走原生
    ``random`` 方法）。两条路径的随机流不同，但分布相同 —— 这一点在测试里
    用"两条路径的均值/方差一致"钉住。
    """

    def __init__(self, kind: str = DEFAULT_DIST, rel_sigma: float = DEFAULT_REL_SIGMA,
                 *, mode_rel: float = 0.0, alpha: float = 2.0, beta: float = 2.0):
        kind = str(kind or DEFAULT_DIST).lower()
        if kind not in MARGINAL_KINDS:
            raise ValueError(
                f"未知先验分布 {kind!r}；可选：{', '.join(MARGINAL_KINDS)}"
            )
        self.kind = kind
        self.rel_sigma = abs(float(rel_sigma))
        self.mode_rel = float(mode_rel)
        self.alpha = float(alpha)
        self.beta = float(beta)
        if kind == "beta" and (self.alpha <= 0 or self.beta <= 0):
            raise ValueError("Beta 分布的 a、b 必须为正")

    # ─── 支撑区间 ───────────────────────────────────
    def support(self) -> tuple[float, float]:
        """ε 的取值范围。无界分布返回 ±inf —— 调用方自己决定要不要截断。"""
        d = self.rel_sigma
        if self.kind == "normal":
            return (-math.inf, math.inf)
        if self.kind == "lognormal":
            # ε = exp(σz) - 1，下界恒 > -1（保证 x > 0），上界 +inf
            return (-1.0, math.inf)
        return (-d, d)

    # ─── 由 u∈(0,1) 反解（copula / Sobol 用）──────────
    def from_unit(self, u: float) -> float:
        """给定 u ~ U(0,1)，返回该先验的分位数 ε。"""
        u = float(u)
        # Sobol 的 u 空间是 [0,1]，端点处 Φ⁻¹ 会发散；夹到开区间内部。
        # 这不是"改分布"，是把不可表示的点挪到最近的可表示点（1e-12 量级）。
        u = min(max(u, 1e-12), 1.0 - 1e-12)

        d = self.rel_sigma
        if self.kind == "uniform":
            return -d + 2.0 * d * u
        if self.kind == "normal":
            return d * inverse_normal_cdf(u)
        if self.kind == "lognormal":
            return math.exp(d * inverse_normal_cdf(u)) - 1.0
        if self.kind == "triangular":
            return self._triangular_ppf(u)
        return -d + 2.0 * d * _beta_ppf(u, self.alpha, self.beta)

    def _triangular_ppf(self, u: float) -> float:
        lo, hi, mode = -self.rel_sigma, self.rel_sigma, self.mode_rel
        mode = min(max(mode, lo), hi)          # 众数必须在支撑内
        if hi <= lo:
            return lo
        split = (mode - lo) / (hi - lo)        # 众数处的累积概率
        if u <= split:
            return lo + math.sqrt(u * (hi - lo) * (mode - lo))
        return hi - math.sqrt((1.0 - u) * (hi - lo) * (hi - mode))

    # ─── 直接抽样（普通蒙特卡洛用）──────────────────
    def draw(self, rng: random.Random) -> float:
        d = self.rel_sigma
        if self.kind == "uniform":
            # 与旧实现 rng.uniform(-δ, +δ) 完全一致（同为一次 random() 调用），
            # 这样历史种子的抽样序列不会因为重构而改变。
            return rng.uniform(-d, d)
        if self.kind == "normal":
            return rng.normalvariate(0.0, d)
        if self.kind == "lognormal":
            return math.exp(rng.normalvariate(0.0, d)) - 1.0
        if self.kind == "triangular":
            mode = min(max(self.mode_rel, -d), d)
            return rng.triangular(-d, d, mode)
        return -d + 2.0 * d * _beta_ppf(rng.random(), self.alpha, self.beta)

    # ─── 文案 ───────────────────────────────────────
    def perturbation_label(self) -> str:
        """
        进报告文案的"扰动口径"。**必须带分布名** —— 只写"±5%"时，
        均匀与正态的 90% 区间差一倍宽，读者无从判断。
        """
        d = self.rel_sigma * 100.0
        if self.kind == "uniform":
            return f"±{d:.0f}% 均匀扰动"
        if self.kind == "normal":
            return f"±{d:.0f}% 正态扰动（1σ）"
        if self.kind == "lognormal":
            return f"对数正态扰动（σ_ln={d:.0f}%，中位数不偏移）"
        if self.kind == "triangular":
            return f"±{d:.0f}% 三角扰动（众数偏 {self.mode_rel * 100:+.0f}%）"
        return f"±{d:.0f}% Beta({self.alpha:g},{self.beta:g}) 扰动"

    def describe(self) -> str:
        return f"{self.kind}：{PRIOR_NOTES[self.kind]}"

    def to_dict(self) -> dict:
        out = {"kind": self.kind, "rel_sigma": self.rel_sigma}
        if self.kind == "triangular":
            out["mode_rel"] = self.mode_rel
        if self.kind == "beta":
            out["alpha"] = self.alpha
            out["beta"] = self.beta
        return out

    def __repr__(self) -> str:      # pragma: no cover - 调试用
        return f"Marginal({self.to_dict()})"


def make_marginal(spec) -> Marginal:
    """
    接受 ``Marginal``、字符串（分布名）或 dict，统一转成 ``Marginal``。

    dict 形式让配置/JSON 能直接描述先验（``{"kind": "lognormal", "rel_sigma": 0.2}``），
    而不必在调用点 ``import`` 这个类。
    """
    if isinstance(spec, Marginal):
        return spec
    if isinstance(spec, str):
        return Marginal(spec)
    if isinstance(spec, dict):
        return Marginal(
            spec.get("kind", spec.get("dist", DEFAULT_DIST)),
            spec.get("rel_sigma", DEFAULT_REL_SIGMA),
            mode_rel=spec.get("mode_rel", 0.0),
            alpha=spec.get("alpha", 2.0),
            beta=spec.get("beta", 2.0),
        )
    if spec is None:
        return Marginal()
    raise TypeError(f"无法把 {type(spec).__name__} 转成 Marginal")


def draw_relative(rng: random.Random, rel_sigma: float, dist: str = DEFAULT_DIST,
                  **kwargs) -> float:
    """
    抽一个相对扰动 ε（``x' = x·(1+ε)``）。

    ``uniform``：ε ~ U(-δ, +δ)，δ 就是"最大扰动幅度"。
    ``normal`` ：ε ~ N(0, δ)，此时 δ 是 1σ（无界，但 3σ 之外概率可忽略）。
    另见 ``Marginal`` 的 lognormal / triangular / beta。

    保留这个函数而不是让调用方直接用 ``Marginal``：``tools/physics_tools.py``
    与若干测试按位置参数调它，签名不能变。
    """
    return Marginal(dist, rel_sigma, **kwargs).draw(rng)


# ══════════════════════════════════════════════════════════
# 参数相关性（Gaussian copula）
# ══════════════════════════════════════════════════════════
def validate_correlation(matrix, dim: int) -> list[list[float]]:
    """
    校验相关矩阵：方阵、对称、对角为 1、元素在 [-1,1]、正定。

    **校验放在这里而不是让 Cholesky 自己崩**：Cholesky 失败只会抛一句
    "math domain error"，而调用方拿到这句话无法知道是第几行第几列写错了。
    相关性写错是那种"不报错但结论全错"的输入，所以宁可多写 30 行。
    """
    m = [[float(v) for v in row] for row in matrix]
    if len(m) != dim or any(len(row) != dim for row in m):
        raise ValueError(f"相关矩阵必须是 {dim}×{dim}，实际 {len(m)} 行")
    for i in range(dim):
        if abs(m[i][i] - 1.0) > 1e-9:
            raise ValueError(f"相关矩阵对角必须为 1，第 {i + 1} 行是 {m[i][i]}")
        for j in range(dim):
            if abs(m[i][j] - m[j][i]) > 1e-9:
                raise ValueError(f"相关矩阵不对称：ρ[{i + 1}][{j + 1}]={m[i][j]} "
                                 f"而 ρ[{j + 1}][{i + 1}]={m[j][i]}")
            if abs(m[i][j]) > 1.0 + 1e-9:
                raise ValueError(f"相关系数越界：ρ[{i + 1}][{j + 1}]={m[i][j]}")
    # 正定性用 Cholesky 直接试（比算特征值简单，且失败点就是出错的位置）
    cholesky(m)
    return m


def cholesky(matrix) -> list[list[float]]:
    """
    下三角 Cholesky 分解 ``A = L·Lᵀ``。非正定时抛 ValueError 并指出是哪一行。

    正定是"相关矩阵合法"的充要条件。典型的非法输入是**两个参数完全相关、
    又各自与第三个参数相关 0.9**：``[[1, .9, 1], [.9, 1, .9], [1, .9, 1]]``
    行列式为 0，这种矩阵必须当场拒绝，不能悄悄加个对角扰动把它变成"能算"的
    矩阵 —— 那等于替用户改了输入。

    注意别把"高相关"和"不合法"搞混：三参数**两两**相关 0.9 是合法的
    （等相关矩阵的特征值是 1+2ρ 与 1−ρ，此处为 2.8 与 0.1，仍然正定）。
    所以这里只能真的去分解，不能靠"相关太高"这种启发式判断。
    """
    n = len(matrix)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                d = float(matrix[i][i]) - s
                if d <= 1e-12:
                    raise ValueError(
                        f"相关矩阵不是正定的（第 {i + 1} 个主元 = {d:.3g}）。"
                        "常见原因：三个及以上参数两两给了高相关，实际不可能同时成立"
                    )
                L[i][j] = math.sqrt(d)
            else:
                L[i][j] = (float(matrix[i][j]) - s) / L[j][j]
    return L


class JointSampler:
    """
    多参数联合抽样：**边缘分布各自独立指定，相关结构用 copula 注入**。

    算法（Gaussian copula）::

        g ~ N(0, I)                 独立标准正态
        z = L · g                   L 是相关矩阵的 Cholesky 因子 → z 服从 N(0, R)
        u_i = Φ(z_i)                逐维转成均匀（这一步保住了边缘分布）
        ε_i = F_i⁻¹(u_i)            再用各参数自己的分位数函数回到 ε

    为什么用 copula 而不是"直接对 ε 做线性变换"：后者会把边缘分布一起改掉
    （把两个均匀分布线性组合，结果不再是均匀分布），于是"加了相关性"和
    "改了分布"两个变化混在一起，报告里给出的区间到底是谁造成的就说不清了。

    相关系数的口径
    --------------
    ``corr`` 里的 ρ 是**潜变量相关系数**，不是边缘分布之间的 Pearson 相关。
    对均匀边缘有解析关系 ``ρ_s = (6/π)·arcsin(ρ/2)``（ρ=0.5 → 0.483），
    对其它边缘没有闭式解。所以 ``empirical_correlation()`` 直接**实测**一遍
    抽样结果的秩相关并写进返回值 —— 报告里要引用的必须是实测值，
    不是我们填进去的那个数。
    """

    def __init__(self, dims: Sequence, corr=None, seed: int | None = None):
        self.dims = [make_marginal(d) for d in dims]
        self.names = [str((d or {}).get("name", f"x{i + 1}")) if isinstance(d, dict)
                      else f"x{i + 1}" for i, d in enumerate(dims)]
        n = len(self.dims)
        if n == 0:
            raise ValueError("JointSampler 至少需要一个参数")
        if corr is None:
            self.corr = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
            self.independent = True
        else:
            self.corr = validate_correlation(corr, n)
            self.independent = all(
                abs(self.corr[i][j]) < 1e-12 for i in range(n) for j in range(n) if i != j
            )
        self._L = cholesky(self.corr)
        self.rng = make_rng(seed)

    def draw_unit(self) -> list[float]:
        """抽一组 u ~ U(0,1)^d（带相关结构）。Sobol 的输入空间就用这个。"""
        g = [self.rng.normalvariate(0.0, 1.0) for _ in self.dims]
        return [normal_cdf(sum(self._L[i][k] * g[k] for k in range(i + 1)))
                for i in range(len(self.dims))]

    def draw(self) -> list[float]:
        """抽一组相对扰动 ε。"""
        return [m.from_unit(u) for m, u in zip(self.dims, self.draw_unit())]

    def draw_values(self, nominals: Sequence[float]) -> list[float]:
        """抽一组**物理量** x = nominal·(1+ε)。"""
        return [float(x0) * (1.0 + e) for x0, e in zip(nominals, self.draw())]

    def empirical_correlation(self, n: int = 2000, seed: int | None = None) -> list[list[float]]:
        """
        实测抽样结果的 Pearson 相关（用独立种子，不扰动调用方的随机流）。

        存在的理由：``corr`` 填的是 copula 参数，边缘分布一换，实际相关系数
        就跟着变。报告里写"深度与压力相关 0.8"而实测是 0.78，读者没法判断
        是笔误还是口径差异 —— 所以把实测值一起返回。
        """
        probe = JointSampler(self.dims, self.corr, seed=seed)
        rows = [probe.draw() for _ in range(max(2, int(n)))]
        cols = list(zip(*rows))
        out = []
        for i, a in enumerate(cols):
            ma = sum(a) / len(a)
            va = sum((v - ma) ** 2 for v in a)
            row = []
            for j, b in enumerate(cols):
                if i == j:
                    row.append(1.0)
                    continue
                mb = sum(b) / len(b)
                vb = sum((v - mb) ** 2 for v in b)
                cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
                row.append(cov / math.sqrt(va * vb) if va > 0 and vb > 0 else 0.0)
            out.append(row)
        return out

    def describe(self) -> list[dict]:
        """每个参数的先验描述（进报告与前端）。"""
        return [
            {"name": name, "kind": m.kind, "rel_sigma": m.rel_sigma,
             "label": m.perturbation_label(), "note": PRIOR_NOTES[m.kind]}
            for name, m in zip(self.names, self.dims)
        ]


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


def wilson_interval(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """
    二项比例的 Wilson 得分区间（默认 95%）。

    为什么概率必须带区间：100 次模拟里 5 次 risky → P = 5%，而它的 95% 区间是
    [2.1%, 11.2%]。如果告警阈值恰好是 5%，"恰好触发"这件事本身就在噪声里 ——
    不把这个区间写出来，就是在用一个不可区分的数做硬判定。
    用 Wilson 而不是 Wald（p ± z√(p(1-p)/n)）：p 接近 0 或 1 时 Wald 会给出
    负下界或超过 1 的上界，而这两种情况恰恰是风险判定的常态。
    """
    n = int(n)
    if n <= 0:
        return (0.0, 1.0)
    p = min(max(int(successes) / n, 0.0), 1.0)
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


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
    rel_sigma    : 相对不确定度。uniform 下是最大扰动幅度，normal 下是 1σ，
                   lognormal 下是 ln 空间的 σ（0.05 表示 5%）
    n            : 模拟次数
    seed         : 随机种子；None 时读 ``UQ_SEED`` 环境变量，再没有就真随机
    dist         : ``MARGINAL_KINDS`` 之一。默认 ``uniform``（有界，δ 是边界）
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
        # 分布口径：报告里必须写明"±5%"是均匀还是正态，
        # 否则同一个 ±5% 对应的 90% 区间可以差一倍宽。
        "dist_label": Marginal(dist, rel_sigma).perturbation_label(),
    }
    if keep_samples:
        out["samples"] = samples
    return out


def monte_carlo_joint(
    fn: Callable[[list[float]], float],
    dims: Sequence,
    nominals: Sequence[float],
    n: int = DEFAULT_N,
    seed: int | None = None,
    corr=None,
    lower_bound: float | None = None,
    upper_bound: float | None = None,
    keep_samples: bool = False,
) -> dict:
    """
    多输入、可相关的蒙特卡洛。``fn`` 收一个**列表**（按 ``dims`` 顺序），
    返回标量。

    与单输入的 ``monte_carlo`` 并存而不是取代它：单输入路径被 MCP 工具、
    前端与一堆测试按位置参数调用，换签名会把它们全打断；而这里需要的是
    "每个输入有自己的分布 + 一个相关矩阵"。

    参数
    ----
    dims     : 每个输入的先验（``Marginal`` / 分布名 / dict），dict 可带 ``name``
    nominals : 每个输入的标称值
    corr     : 相关矩阵；None 表示独立（**这是一个假设，会被写进返回值**）

    返回在 ``monte_carlo`` 的键之外增加 ``inputs`` / ``corr`` / ``corr_empirical``
    / ``independent`` —— 相关矩阵与实际测到的相关都返回，报告引用时不必猜。
    """
    n = max(2, int(n))
    if len(dims) != len(nominals):
        raise ValueError(f"dims 有 {len(dims)} 个而 nominals 有 {len(nominals)} 个")
    sampler = JointSampler(dims, corr=corr, seed=seed)

    samples: list[float] = []
    for _ in range(n):
        x = sampler.draw_values(nominals)
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
        "mean": mean,
        "std": std,
        "p5": percentile(samples, (1 - CI_LEVEL) / 2 * 100),
        "p50": percentile(samples, 50),
        "p95": percentile(samples, (1 + CI_LEVEL) / 2 * 100),
        "min": min(samples),
        "max": max(samples),
        "cv": (std / mean) if mean else 0.0,
        "inputs": [float(v) for v in nominals],
        "names": list(sampler.names),
        "priors": sampler.describe(),
        "corr": sampler.corr,
        "independent": sampler.independent,
        # 实测相关：copula 参数换成边缘分布后会变，报告里要引用的必须是实测值。
        # 传同一个 seed，保证"定了种子就整份结果可复现"。
        "corr_empirical": sampler.empirical_correlation(
            n=min(2000, max(200, n * 4)), seed=seed
        ),
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

        "662.93 kg/m³（90% 置信区间 630.1–695.4 kg/m³，100 次蒙特卡洛模拟，输入±5% 均匀扰动）"

    这段文本是模型的输入，也是报告的兜底文案 —— 所以必须自带全部限定条件
    （模拟次数、扰动幅度、**分布类型**、置信水平），否则读者无法判断这个
    区间可不可信。分布类型是 v1.7 补的：同样写"±5%"，均匀口径的 90% 区间
    约 ±4.5%，正态口径约 ±8.2%，不写清是哪种等于没写。
    """
    if not summary:
        return ""
    mean = summary.get("mean")
    lo = summary.get("p5")
    hi = summary.get("p95")
    if mean is None or lo is None or hi is None:
        return ""

    unit_part = f" {unit}" if unit else ""
    n = summary.get("n") or 0
    label = summary.get("dist_label")
    if not label:
        # 兼容只带 rel_sigma/dist 的旧结构（MCP 路径、历史 JSON）
        label = Marginal(
            summary.get("dist") or DEFAULT_DIST,
            summary.get("rel_sigma") or 0.0,
        ).perturbation_label()
    return (
        f"{format_value(mean, digits)}{unit_part}"
        f"（90% 置信区间 {format_value(lo, digits)}–{format_value(hi, digits)}{unit_part}，"
        f"{n} 次蒙特卡洛模拟，输入 {label}）"
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
        # 多输入联合抽样（monte_carlo_joint）没有单一 rel_sigma：每个参数
        # 各有自己的先验。此时 rel_sigma 记 0 并靠 priors 表达口径，
        # 不能随便填一个"平均扰动" —— 那个数不存在。
        "rel_sigma": float(summary.get("rel_sigma") or 0.0),
        "dist": summary.get("dist", DEFAULT_DIST),
        "dist_label": summary.get("dist_label") or Marginal(
            summary.get("dist") or DEFAULT_DIST,
            summary.get("rel_sigma") or 0.0,
        ).perturbation_label(),
        "nominal": round(float(summary.get("nominal", 0.0)), 6),
        "unit": unit,
        "ci_level": CI_LEVEL,
        "text": format_interval(summary, unit=unit, digits=digits),
    }
    for extra in ("priors", "corr", "independent", "corr_empirical", "names"):
        if summary.get(extra) is not None:
            out[extra] = summary[extra]
    if keep_samples and summary.get("samples"):
        out["samples"] = [round(float(v), 6) for v in summary["samples"]]
    return out


def uq_block(uq_summary: dict) -> str:
    """
    生成注入报告 Prompt 的"概率性结论"块。

    与 ``core.temporal.build_conflict_notice`` 同一个思路：把硬约束写成
    代码生成的文本，而不是指望 System Prompt 里的一句"请包含不确定性"。

    v1.7 起这块还承载两件事：**敏感性排名**（谁造成的不确定）与
    **决策判定**（概率是否越过告警线）。三者都必须是代码算出来的文本 ——
    让 7B 自己从一堆数字里"总结一下主要风险来源"，它会编一个听起来合理的
    排名，而排名恰恰是这份报告最该可信的部分。
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
            dist_label = item.get("dist_label") or Marginal(
                item.get("dist") or DEFAULT_DIST,
                item.get("rel_sigma") or 0.0,
            ).perturbation_label()
            lines.append(
                f"- {label}：{detail}"
                f"（{item.get('n', 0)} 次蒙特卡洛，输入{dist_label}）"
            )

    # ─── 敏感性分析（Sobol）──────────────────────────
    sens = uq_summary.get("sensitivity") or {}
    if sens.get("text"):
        lines.append(f"- 敏感性分析（Sobol）：{sens['text']}")

    # ─── 决策判定 ───────────────────────────────────
    decision = uq_summary.get("decision") or {}
    if decision.get("text"):
        lines.append(f"- 决策判定：{decision['text']}")

    # ─── 注入策略优化（从"诊断"到"处方"）────────────
    plan = uq_summary.get("plan") or {}
    if plan.get("text"):
        lines.append(f"- 注入策略优化：{plan['text']}")

    if len(lines) == 1:
        return ""

    lines += [
        "",
        "写作要求（违反即视为报告不合格）：",
        "1. 上述数值必须按「均值 + 90% 置信区间」的形式写入正文，"
        "**禁止只写单一确定值**（例如只写 662.93 kg/m³ 而不给区间）。",
        "2. 概率性结论必须带上限定条件（模拟次数与扰动分布），"
        "不要改写成\"约为\"、\"大约\"这类模糊表述。",
        "3. 若给出风险判断，必须同时给出概率（如\"压裂风险概率 12%\"），"
        "不要用\"存在一定风险\"这种无法证伪的措辞。",
    ]
    # 编号用计数器而不是 ``5 if ... else 4``：多加一条要求时后者会静默重号，
    # 而重号在 Prompt 里表现为"两条要求都叫 4"，模型会只照做一条。
    index = 4
    if sens.get("text"):
        lines.append(
            f"{index}. 敏感性排名必须按上面给出的顺序写，**禁止自行重排或补充**"
            "未列出的参数；一阶指数与总效应指数要分开表述，"
            "二者的差值代表参数间交互作用，不要合并成一个\"影响因子\"。"
        )
        index += 1
    if decision.get("text"):
        lines.append(
            f"{index}. 决策判定（{decision.get('level', '')}）"
            "必须单独成段写明：触发了哪条阈值规则、概率是多少、"
            "以及概率的置信区间 —— 阈值是**风险接受准则**而不是物理常数，"
            "报告里必须让读者看出这一点。"
        )
        index += 1
    if plan.get("text"):
        lines.append(
            f"{index}. 注入策略建议必须照抄上面的井数与井底压力上限，"
            "**禁止自行给出未列出的注入速率、井数或压力值**；"
            "若推荐方案的风险仍高于预算，必须写明\"无可行方案\"，"
            "不得用折中值代替。速率是**压力约束下的上限**"
            "（未计井筒摩阻与非达西效应），引用时必须带上这句限定。"
        )
    lines.append("")
    return "\n".join(lines)
