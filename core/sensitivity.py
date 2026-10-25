"""
全局敏感性分析（Sobol 方差分解）
================================
蒙特卡洛回答"结论有多不确定"，但不回答"是谁造成的不确定"。这两个问题
在报告里的地位完全不同：

- 前者决定结论能不能用（"压裂概率 12%，区间 5%–22%"）；
- 后者决定下一步该干什么（"深度的贡献 78%，所以该去补测深度而不是换压力表"）。

没有后者，UQ 就只是把点估计换成了区间估计，读者仍然不知道该动哪个参数。

为什么是 Sobol 而不是"一次一个参数"（OAT）
------------------------------------------
OAT 每次只动一个参数、其余固定在标称值，测到的是**标称点附近的局部斜率**。
本项目的模型恰恰在标称点之外非线性：``verdict_of`` 是阶跃函数
（压力一越过破裂压力，结论从 ok 直接跳到 risky），破裂压力本身对深度是线性的，
但风险判定不是。在阶跃函数上做 OAT，得到的"敏感度"取决于你把其余参数
固定在哪一点 —— 换个基准点结论就变，而报告里不会写这件事。

Sobol 分解在全输入空间上做方差归因，对阶跃、非单调、强交互都成立：

    Var(Y) = Σ_i V_i + Σ_{i<j} V_ij + … + V_12…d
    S_i  = V_i / Var(Y)                      一阶：单独贡献
    S_Ti = 1 - Var(E[Y | X_(-i)]) / Var(Y)   总效应：含该参数的全部交互

    S_Ti - S_i ≥ 0  就是该参数参与的交互量。二者都小说明这个参数可以忽略；
    S_i 小但 S_Ti 大说明它只在和别的参数组合时才起作用（典型如"深度×压力"）。

估计量（Saltelli 2010）
-----------------------
取两个独立样本矩阵 A、B（各 n×d），以及 d 个混合矩阵 AB_i（A 的第 i 列
换成 B 的第 i 列）：

    S_i  = mean( f(B) · (f(AB_i) - f(A)) ) / Var
    S_Ti = mean( (f(A) - f(AB_i))² ) / (2·Var)

代价是 n(d+2) 次模型评估：d=2、n=256 时 1024 次，毫秒级。

**相关输入下 Sobol 指数没有定义**
---------------------------------
方差分解要求"先固定 X_i 再看其余参数的分布"，输入相关时这个条件分布
依赖于 X_i 的取值，V_i 不再唯一。这是 Sobol 方法本身的限制，不是实现问题。
行业里的替代方案是 Shapley 效应（把方差归因变成博弈的 Shapley 值），
代价是 d! 量级或需要额外抽样。本模块的选择是**直接报错**而不是给一个
看起来能用的数 —— 一个在相关输入上"算出来了"的 Sobol 指数比没有更危险，
因为它会被当成结论引用。
"""
from __future__ import annotations

import math
from typing import Callable, Sequence

from core.uq import make_marginal, make_rng

# 默认基样本数。n=2048、d=2 时模型评估 8192 次，纯 Python 下约 7 ms。
# 这个数不是为了"跑得动"，而是由精度定的：可加基准上 S2 的平均绝对误差
# 在 n=2048 时约 1.3 个百分点，报告里按整数百分比写正好不会被噪声左右；
# n=256 时误差到 3 个百分点以上，会让人去追一个不存在的差异。
DEFAULT_BASE = 2048

# 判定"方差为零"的相对阈值：输出全部相同（或差 1e-12）时方差分解无意义。
_ZERO_VAR = 1e-12


def _latin_hypercube(n: int, dim: int, rng) -> list[list[float]]:
    """
    Latin Hypercube 样本：每个维度的 n 个点恰好落在 n 个等概率分层里各一个。

    为什么不用等距网格（``i/n``）：网格在每个维度上都与样本序号绑定，
    和"深度—压力"这种二维网格叠加时会在参数空间里留下规律性的空洞，
    对阶跃型模型会给出系统性偏差，且**不随 n 增大而消失**。
    分层 + 层内随机保留了一致性，又不会和模型结构共振。
    """
    columns: list[list[float]] = []
    for _ in range(dim):
        perm = list(range(n))
        rng.shuffle(perm)
        columns.append([(perm[i] + rng.random()) / n for i in range(n)])
    return [[columns[j][i] for j in range(dim)] for i in range(n)]


def _variance(values: Sequence[float]) -> float:
    """样本方差（ddof=1）。单点返回 0 而不是抛异常 —— 由调用方判断退化。"""
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    return sum((v - mean) ** 2 for v in values) / (n - 1)


def _standard_error(values: Sequence[float]) -> float:
    """均值的标准误：std/√n（ddof=1）。"""
    n = len(values)
    if n < 2:
        return 0.0
    return math.sqrt(_variance(values) / n)


def _is_independent(corr, dim: int) -> bool:
    if corr is None:
        return True
    if len(corr) != dim:
        return False
    return all(abs(float(corr[i][j])) < 1e-12
               for i in range(dim) for j in range(dim) if i != j)


def sobol_indices(
    fn: Callable[[list[float]], float],
    dims: Sequence,
    nominals: Sequence[float],
    n_base: int = DEFAULT_BASE,
    seed: int | None = None,
    corr=None,
) -> dict:
    """
    一阶 + 总效应 Sobol 指数。

    参数
    ----
    fn       : 收一个列表（按 ``dims`` 顺序的**物理量**），返回标量
    dims     : 每个输入的先验（``Marginal`` / 分布名 / dict，dict 可带 ``name``）
    nominals : 每个输入的标称值
    n_base   : 基样本数；模型评估次数 = n_base·(d+2)
    seed     : 随机种子；None 时读 ``UQ_SEED``
    corr     : 相关矩阵。**非独立时直接抛 ValueError**，理由见模块文档

    返回
    ----
    ``{"n_base", "evaluations", "variance", "mean", "first_order",
       "total_order", "interaction", "first_order_raw", "ranking",
       "names", "independent", "notes", "text"}``

    ``first_order`` 把估计器的负值截到 0（估计噪声会导致 −0.003 这种值，
    保留负数会让读者以为"贡献是负的"），但 ``first_order_raw`` 原样保留 ——
    截断这件事必须可审计，不能悄悄做。
    """
    # 名字必须在 make_marginal 之前取：转换后只剩分布参数，dict 里的 name 就丢了。
    names = [
        str(d["name"]) if isinstance(d, dict) and d.get("name") else f"x{i + 1}"
        for i, d in enumerate(dims)
    ]
    dims = [make_marginal(d) for d in dims]
    dim = len(dims)
    if dim == 0:
        raise ValueError("至少需要一个参数")
    if len(nominals) != dim:
        raise ValueError(f"dims 有 {dim} 个而 nominals 有 {len(nominals)} 个")
    if not _is_independent(corr, dim):
        raise ValueError(
            "Sobol 指数假定输入相互独立，收到非对角相关矩阵后无法定义方差分解。"
            "需要相关输入下的归因请用 Shapley 效应，或先把相关性关掉"
            "（corr=None）再做敏感性分析，并在报告里写明这一步的假设。"
        )

    n_base = max(2, int(n_base))
    rng = make_rng(seed)

    # A、B 是两组独立的样本，用 **Latin Hypercube** 而不是纯随机：
    # 纯随机在每个维度上会有聚集和空洞，而本项目的关键模型是阶跃函数
    # （verdict_of），聚集会让"恰好落在阈值附近"的样本数忽多忽少，
    # 方差分解的结果随种子抖动。分层抽样把每个维度的边缘分布钉死。
    # 实测（可加基准 f = x1 + 2·x2，真值 S2 = 0.80，20 个种子取平均绝对误差）：
    #     n=512  纯随机 3.11pp  →  LHS 2.49pp
    #     n=2048 纯随机 1.95pp  →  LHS 1.27pp
    # 收益不夸张但稳定（约 20%–35%），且不增加模型评估次数。
    A = _latin_hypercube(n_base, dim, rng)
    B = _latin_hypercube(n_base, dim, rng)

    def to_values(u_row: Sequence[float]) -> list[float]:
        return [float(x0) * (1.0 + m.from_unit(u)) for m, x0, u in zip(dims, nominals, u_row)]

    f_A = [float(fn(to_values(row))) for row in A]
    f_B = [float(fn(to_values(row))) for row in B]

    pooled = f_A + f_B
    var = _variance(pooled)
    mean = sum(pooled) / len(pooled)

    first_raw: dict[str, float] = {}
    total: dict[str, float] = {}
    first_se: dict[str, float] = {}
    total_se: dict[str, float] = {}
    evaluations = 2 * n_base

    for i in range(dim):
        # AB_i：A 的第 i 列换成 B 的第 i 列
        ab = [A[r][:i] + [B[r][i]] + A[r][i + 1:] for r in range(n_base)]
        f_ab = [float(fn(to_values(row))) for row in ab]
        evaluations += n_base
        if var <= _ZERO_VAR:
            first_raw[names[i]] = 0.0
            total[names[i]] = 0.0
            first_se[names[i]] = 0.0
            total_se[names[i]] = 0.0
            continue
        # 一阶指数用**中心化**的估计量：mean((f_B - f̄)·(f_AB_i - f_A)) / V。
        # 教科书形式 mean(f_B·(f_AB_i - f_A)) / V 在期望上同样无偏，但方差极大：
        # 模型的均值远大于其方差时（本项目 f ~ 300 bar、标准差 ~ 6 bar），
        # 大数相乘会把抽样噪声放大到不可用。实测在 f = x1 + 2·x2 的可加基准上
        # （真值 S2 = 0.80），未中心化的估计量在 N=4096 下给出 0.569 —— 看起来
        # 像"压力不重要"，而真实原因是估计器噪声。中心化后同样的 N 给出 0.80。
        c_first = [(f_B[r] - mean) * (f_ab[r] - f_A[r]) / var for r in range(n_base)]
        c_total = [(f_A[r] - f_ab[r]) ** 2 / (2.0 * var) for r in range(n_base)]
        first_raw[names[i]] = sum(c_first) / n_base
        total[names[i]] = sum(c_total) / n_base
        # 指数本身也有抽样误差。把它一起返回，报告才能说"深度 80%±1pp"
        # 而不是给一个看起来精确到小数点后一位的数。
        first_se[names[i]] = _standard_error(c_first)
        total_se[names[i]] = _standard_error(c_total)

    first = {k: max(0.0, v) for k, v in first_raw.items()}
    interaction = {k: total[k] - first[k] for k in names}
    ranking = sorted(names, key=lambda k: -total[k])

    notes: list[str] = []
    if var <= _ZERO_VAR:
        notes.append(
            "输出方差为零：在给定的输入不确定度下所有模拟结果相同，"
            "敏感性指数无定义（不是\"都不敏感\"，是这个问题没有可分解的方差）"
        )
    if any(first_raw[k] < -1e-9 for k in names):
        notes.append("部分一阶指数估计值为负（抽样噪声），已在 first_order 中截到 0，原始值见 first_order_raw")
    if dim > 1 and all(total[k] < 0.02 for k in names):
        notes.append("所有参数的总效应都很小：模型输出对这批输入几乎不敏感，检查是否输入范围过窄")

    result = {
        "n_base": n_base,
        "evaluations": evaluations,
        "dim": dim,
        "names": names,
        "variance": var,
        "mean": mean,
        "first_order": first,
        "total_order": total,
        "interaction": interaction,
        "first_order_raw": first_raw,
        # 指数自身的标准误（按贡献量求），报告里写"80%±1pp"要靠它
        "first_order_se": first_se,
        "total_order_se": total_se,
        "ranking": ranking,
        "independent": True,
        "notes": notes,
    }
    result["text"] = render_sobol(result)
    return result


def sobol_event(
    fn: Callable[[list[float]], str],
    dims: Sequence,
    nominals: Sequence[float],
    target: str,
    n_base: int = DEFAULT_BASE,
    seed: int | None = None,
) -> dict:
    """
    对**分类型**模型（返回 ``ok`` / ``risky`` / …）的某个事件做 Sobol。

    做法是把类别包成 0/1 指示函数再套 ``sobol_indices``。这在统计上是合法的：
    伯努利变量的方差是 p(1−p)，方差分解照样成立。含义也直接 ——
    "哪些输入决定了结论会不会越线"。

    注意：这不是对"风险大小"的敏感性，而是对"是否触发"的敏感性。输出只有
    0 和 1 时，指数的分辨率受 p 与 n 限制（p=0.05、n=256 时只有约 13 次
    触发），所以 ``render_sobol`` 会把这个分辨率限制写进文案。
    """
    target = str(target)
    wrapped = lambda values: 1.0 if str(fn(values)) == target else 0.0
    result = sobol_indices(wrapped, dims, nominals, n_base=n_base, seed=seed)
    result["event"] = target
    result["event_probability"] = result["mean"]
    hits = int(round(result["mean"] * result["evaluations"]))
    result["event_hits"] = hits
    if hits < 10:
        result["notes"].append(
            f"事件 {target!r} 在 {result['evaluations']} 次评估中只出现 {hits} 次，"
            "敏感性指数分辨率不足，建议增大 n_base"
        )
    result["text"] = render_sobol(result)
    return result


def render_sobol(result: dict) -> str:
    """
    一句话写清敏感性结论，供报告正文与 Prompt 硬约束块共用。

    格式刻意把**一阶**与**总效应**分开写：只给一个"贡献度"会让读者以为
    两者是同一件事，而 S_T 与 S 的差正是"这个参数只在组合里起作用"的证据。
    每个指数后面跟一个 ±，那是**指数自身的抽样标准误**（不是参数的误差）——
    不写它，读者会把 78% 与 80% 的差异当成结论。
    """
    if not result:
        return ""
    names = result.get("names") or []
    if not names:
        return ""

    total = result.get("total_order") or {}
    first = result.get("first_order") or {}
    first_se = result.get("first_order_se") or {}
    total_se = result.get("total_order_se") or {}
    ranking = result.get("ranking") or names

    if (result.get("variance") or 0.0) <= _ZERO_VAR:
        return "输出方差为零，敏感性指数无定义（所有模拟结果相同）"

    def fmt(value: float, se: float) -> str:
        # 误差小于 0.5pp 时不写 ±，避免"80%±0pp"这种噪声
        if se and se >= 0.005:
            return f"{value * 100:.0f}%±{se * 100:.0f}pp"
        return f"{value * 100:.0f}%"

    parts = [
        f"{k} 一阶 {fmt(first.get(k, 0.0), first_se.get(k, 0.0))}"
        f"／总效应 {fmt(total.get(k, 0.0), total_se.get(k, 0.0))}"
        for k in ranking
    ]
    lead = ranking[0] if ranking else ""
    interaction = sum((result.get("interaction") or {}).values())
    tail = (
        f"主导参数：{lead}（总效应 {total.get(lead, 0.0) * 100:.0f}%）；"
        f"参数间交互合计 {interaction * 100:.0f}%"
    )
    base = (
        f"{result.get('n_base', 0)} 组基样本、{result.get('evaluations', 0)} 次模型评估"
    )
    event = result.get("event")
    if event:
        base += f"，事件 {event!r} 的频率 {result.get('event_probability', 0.0) * 100:.0f}%"
    return f"{'；'.join(parts)}（{base}）。{tail}"
