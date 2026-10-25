"""
一维两相驱替求解器（Buckley-Leverett 方程的有限体积解）
======================================================
``utils/relperm.py`` 给的是解析前沿（切线法），它只能回答"前沿饱和度是多少"。
真正需要的是**任意时刻的饱和度剖面** —— 因为盖层能不能封住，取决于 CO₂
在盖层下方**堆了多高**，而柱高是剖面积分出来的量。

方程（无因次形式）：

    ∂S_g/∂t_D + ∂f_g/∂x_D = 0
    t_D = q·t/(φ·A·L),   x_D = x/L

## 为什么值得写一个求解器

不是为了"更精确"——一维 BL 的解析解是精确的。是为了两件事：

1. **验证**。数值解与解析前沿必须一致（前沿位置、平台饱和度、物质守恒）。
   一个只进不出的求解器很容易在边界条件上悄悄漏质量，而那正是
   "柱高算小了、盖层判安全"的直接原因。
2. **任意边界条件**。解析解只对"从 t=0 连续注入"成立。实际会问"注了 30 天
   再停 90 天会怎样"，这时没有解析解。

## 边界

- 一维、不可混溶、不可压缩、无毛细管压力、无重力分异。
  重力分异（CO₂ 上浮）在垂直方向上会显著改变剖面，本求解器**不含**这一项，
  所以在 ``limitations`` 里明确写出。
- 一阶迎风格式，数值扩散会让前沿变宽。只用于工程量级判断，不用于
  前沿精细结构。
"""
from __future__ import annotations

from bisect import bisect_right

from utils.relperm import RelPermParams, bl_front, fractional_flow, load_relperm_params

FLOW_VERSION = "2026.10-f1"

# 分流曲线查表缓存。键是 frozen dataclass（可哈希），所以同一组参数只建一次表。
# 为什么需要：一次求解有几十万次通量计算，每次走 fractional_flow 都要重建一个
# dict，实测把 400 格 × 0.45 的求解拖到秒级。查表把这一步降到 O(1)。
_FG_CACHE: dict = {}
_TABLE_N = 4001


def _fg_table(params: RelPermParams) -> tuple[list[float], list[float]]:
    """预计算 f_g(S_g) 表。S_g 从 0 到 1−swr 均匀取点。"""
    cached = _FG_CACHE.get(params)
    if cached is not None:
        return cached
    sg_max = 1.0 - params.swr
    sgs = [sg_max * i / (_TABLE_N - 1) for i in range(_TABLE_N)]
    fgs = [1.0 - fractional_flow(1.0 - s, params) for s in sgs]
    # 端点强制对齐，避免浮点误差让 f_g(0) 不是 0
    fgs[0] = 0.0
    fgs[-1] = 1.0
    _FG_CACHE[params] = (sgs, fgs)
    return sgs, fgs


def _fg(sg: float, params: RelPermParams) -> float:
    """查表 + 线性插值。表步长 1/4000，插值误差 O(h²) ≈ 6e-8，可忽略。"""
    sgs, fgs = _fg_table(params)
    sg_max = sgs[-1]
    if sg <= 0.0:
        return 0.0
    if sg >= sg_max:
        return 1.0
    pos = sg / sg_max * (_TABLE_N - 1)
    i = int(pos)
    frac = pos - i
    return fgs[i] + (fgs[i + 1] - fgs[i]) * frac


def solve(n_cells: int = 400, t_end: float = 0.5, cfl: float = 0.4,
          params: RelPermParams | None = None) -> dict:
    """
    求解到无因次时间 ``t_end``，返回剖面与守恒检查。

    返回：
        x                 网格中心（无因次）
        sg                饱和度剖面
        t_end             实际推进到的无因次时间
        steps             时间步数
        mass_injected     注入的孔隙体积倍数（= t_end）
        mass_in_place     ∫S_g dx，必须等于 mass_injected
        mass_error        相对误差
        front_position    从剖面提取的前沿位置（S_g 降到前沿值一半处）
        plateau           前沿后平台饱和度（剖面最大值的中位段）
        analytic          relperm.bl_front() 的解析结果，供对比
    """
    p = params or load_relperm_params()
    sg_inj = 1.0 - p.swr          # 注入端 CO₂ 饱和度（把水推到束缚水）
    dx = 1.0 / n_cells
    x = [(i + 0.5) * dx for i in range(n_cells)]
    sg = [0.0] * n_cells

    # 把查表访问提到局部变量：内层要跑几十万次，属性查找在这里是实打实的开销
    sgs, fgs = _fg_table(p)
    n_last = _TABLE_N - 1
    sg_tab_max = sgs[-1]
    scale = n_last / sg_tab_max

    def fg_fast(s: float) -> float:
        if s <= 0.0:
            return 0.0
        if s >= sg_tab_max:
            return 1.0
        pos = s * scale
        i = int(pos)
        frac = pos - i
        return fgs[i] + (fgs[i + 1] - fgs[i]) * frac

    fg_inj = fg_fast(sg_inj)

    # CFL：最大特征速度
    v_max = 0.0
    h = 1e-6
    for i in range(1, 200):
        s = sg_inj * i / 200.0
        v = abs(fg_fast(s + h) - fg_fast(s - h)) / (2 * h)
        if v > v_max:
            v_max = v
    v_max = max(v_max, 1e-6)
    dt = cfl * dx / v_max

    steps = 0
    t = 0.0
    while t < t_end - 1e-12:
        dt_step = min(dt, t_end - t)
        coef = dt_step / dx
        # 显式迎风：界面通量取上游格心；入口按注入饱和度供气。
        # 写成两个列表推导而不是 for 循环 —— 这里做过实测，同样的格数与步数
        # 下纯 Python 循环慢 2–3 倍，而 CI 要在三个 Python 版本上各跑一遍。
        f = [fg_fast(s) for s in sg]
        upstream = [fg_inj] + f[:-1]
        sg = [
            min(max(s - coef * (fi - fp), 0.0), sg_inj)
            for s, fi, fp in zip(sg, f, upstream)
        ]
        t += dt_step
        steps += 1

    mass = sum(s * dx for s in sg)
    analytic = bl_front(p)

    # 前沿位置：从右往左找第一个超过"前沿值一半"的格心
    half = analytic["sg_front"] / 2.0
    front_pos = 1.0
    front_idx = n_cells
    for i in range(n_cells - 1, -1, -1):
        if sg[i] >= half:
            front_pos = x[i]
            front_idx = i
            break

    # 冲击前沿**紧后方**的饱和度，应当等于解析的 S_gf。
    # 注意不能取"中段中位数"当平台：BL 剖面在前沿之后不是平台，
    # 而是从 S_gf 向入口逐渐升高（扩展波）。取中位数会采到上升段，
    # 而且那个值随时间是增长的 —— 看起来像收敛，其实是没有物理含义。
    probe = max(0, front_idx - 3)
    sg_behind = sg[probe]

    # 前沿后平均饱和度（Welge 的物理含义）：扫过区域的物质 / 扫过长度
    swept_mass = sum(sg[: front_idx + 1]) * dx
    avg_behind = swept_mass / front_pos if front_pos > 0 else 0.0

    return {
        "x": x,
        "sg": sg,
        "t_end": round(t, 6),
        "steps": steps,
        "mass_injected": round(t_end, 6),
        "mass_in_place": round(mass, 6),
        "mass_error": round(abs(mass - t_end) / t_end, 6) if t_end else 0.0,
        "front_position": round(front_pos, 4),
        "sg_behind_front": round(sg_behind, 6),
        "avg_behind_front": round(avg_behind, 6),
        "analytic": analytic,
        "version": FLOW_VERSION,
        "limitations": [
            "一维、不可混溶、不可压缩；未含毛细管压力。",
            "未含重力分异：CO₂ 上浮会显著改变垂直剖面，本求解器不建模。",
            "一阶迎风，数值扩散使前沿变宽；只用于量级判断。",
        ],
    }


def co2_column_height(sg_profile: list, dx: float, total_height_m: float,
                      porosity: float) -> float:
    """
    由饱和度剖面折算 CO₂ 在孔隙中占的**等效柱高**（m）。

        H = Σ S_g,i · dx · L · φ

    也就是"如果把孔隙里的 CO₂ 全部聚成一段纯 CO₂ 柱，有多高"。
    这个量直接进盖层浮力计算（Δρ·g·H），是整条链上唯一连接
    "注了多少"与"盖层受多大压"的桥梁。
    """
    if not sg_profile:
        return 0.0
    return sum(sg_profile) * dx * float(total_height_m) * float(porosity)
