"""
一维 PINN 压力预测模型（Physics-Informed Neural Network）
========================================================
用 PyTorch 解一维压力扩散方程，为 CCUS 注入过程提供"算出来的"压力值，
与 ``tools/physics_tools.py`` 里的密度计算并列 —— 都是确定性数值工具，
不允许语言模型口算。

控制方程
--------
达西定律在均质一维介质里的压力扩散形式::

    ∂P/∂t = D · ∂²P/∂x²        D = 0.1 m²/s

``D = 0.1 m²/s`` 是典型储层的压力扩散系数量级（k/(μ·c_t·φ) 通常在
0.1~10 m²/s），不是随手取的数。

为什么必须在无量纲坐标里训练
----------------------------
物理量纲下 x ~ 100 m、t ~ 10⁶ s，残差 ``∂P/∂t - D·∂²P/∂x²`` 的两项量级
相差 6 个数量级（10⁻⁵ 对 10⁻³），优化器会只盯大的那一项，PDE 约束形同虚设。
所以用扩散时间 τ = L²/D 做时间尺度、用 L 做长度尺度::

    x̂ = x/L,  ŝ = t/τ   ⟹   ∂P/∂ŝ = ∂²P/∂x̂²

D 被吸收进坐标里，残差两项同量级，权重不需要再手工调。这不是取巧，
是 PINN 的标准做法；不这么做的模型看起来 loss 在降，实际上只满足了一个
退化方程。

为什么激活函数必须是 tanh
-------------------------
ReLU 的二阶导数几乎处处为 0，``∂²P/∂x²`` 恒等于 0 会让 physics loss
**凭空变成 0** —— 网络可以完全无视 PDE 却报告"物理残差为零"。tanh 光滑，
二阶导存在且非零，这是 PDE 残差能反传的前提。用 sin 也可以，ReLU 不行。

两种边界条件（这是本文件最需要解释的设计）
------------------------------------------
``baseline`` —— 需求里写的那组：初值 100，两端恒定 100。
    数学事实：由最大值原理，唯一解是 **P ≡ 100**。任何 (x, t) 都返回 100。
    模型会完美收敛，学到的是一个常数场 —— 它没有错，但也不含信息。

``injection`` —— 井口注入场景：P(0,t) = 120 bar（注入井筒压力），
    远端 P(L,t) = 100 bar，初值 100 bar。此时压力脉冲真实向外扩散，
    解是非平凡的，且**有解析解可对账**：t ≫ τ 后趋于线性稳态
    P(x) = 120 - 20·x/L，在 x = 50 m 处正好 110 bar。

保留 baseline 是因为它是需求原文，且它是"边界条件选错会让模型退化成常数"
的一个活证据；工具默认走 injection，否则"注入 10 天后"这个问题没有意义。

用法::

    python pinn_model.py                  # 训练两种场景并打印 loss 曲线
    python pinn_model.py --epochs 1000    # 指定轮数

    from pinn_model import predict_pressure
    predict_pressure(50, 10)              # 距井口 50 m、注入 10 天后的压力
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn as nn

# ─── 物理常数 ───────────────────────────────────────────────
D = 0.1                    # 压力扩散系数, m²/s
L = 100.0                  # 一维计算域长度, m（井口 x=0 → 远端 x=L）
T_MAX_DAYS = 30.0          # 训练覆盖的时间上限, 天
SECONDS_PER_DAY = 86400.0

TAU = L * L / D            # 扩散时间 L²/D = 100 000 s ≈ 1.16 天
S_MAX = T_MAX_DAYS * SECONDS_PER_DAY / TAU   # 归一化时间上限 ≈ 25.92

P0 = 100.0                 # 初始 / 远端压力, bar
P_WELL = 120.0             # 注入井筒压力, bar（仅 injection 场景）
P_SCALE = 50.0             # 网络输出缩放：P = P0 + P_SCALE · net(x̂, ŝ)

MODEL_DIR = Path(__file__).resolve().parent / "models"
SCENARIOS = ("baseline", "injection")

# 边界 / 初值项的权重。数据项权重大于 PDE 项是常规做法：
# 约束不住边界时，网络会用一个满足 PDE 但整体偏移的解去骗 loss。
W_PDE, W_BC, W_IC = 1.0, 20.0, 20.0


class PINN(nn.Module):
    """3 层 MLP：2 → 64 → 64 → 1。输入 (x̂, ŝ)，输出归一化压力。"""

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x_hat: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x_hat, s], dim=1))


def _p(net: PINN, x_hat: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """把网络输出还原成物理压力（bar）。"""
    return P0 + P_SCALE * net(x_hat, s)


def physics_residual(net: PINN, x_hat: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """计算 PDE 残差 ∂P/∂ŝ − ∂²P/∂x̂²（无量纲形式，见模块文档）。

    ``create_graph=True`` 是关键：我们要求的是**二阶**导，对一阶导再求导
    必须保留计算图，否则 ``grad`` 拿到的张量是叶子节点、再求导会报
    "does not require grad"。
    """
    x_hat = x_hat.clone().requires_grad_(True)
    s = s.clone().requires_grad_(True)
    p = _p(net, x_hat, s)

    dP_ds = torch.autograd.grad(p, s, grad_outputs=torch.ones_like(p),
                                create_graph=True)[0]
    dP_dx = torch.autograd.grad(p, x_hat, grad_outputs=torch.ones_like(p),
                                create_graph=True)[0]
    d2P_dx2 = torch.autograd.grad(dP_dx, x_hat, grad_outputs=torch.ones_like(dP_dx),
                                  create_graph=True)[0]
    return dP_ds - d2P_dx2


def physics_loss(net: PINN, n_collocation: int = 2000, seed: int | None = None) -> torch.Tensor:
    """PDE 残差的均方，即 physics loss。"""
    g = torch.Generator().manual_seed(seed) if seed is not None else None
    x_hat = torch.rand(n_collocation, 1, generator=g)
    s = torch.rand(n_collocation, 1, generator=g) * S_MAX
    return (physics_residual(net, x_hat, s) ** 2).mean()


def data_loss(net: PINN, scenario: str, n: int = 300, seed: int | None = None) -> torch.Tensor:
    """初值 + 两端边界条件。

    injection 场景的井口边界是**阶跃**（t>0 立刻 120 bar）：这让 t=0 处存在
    一个角点不连续，PINN 在该角附近误差偏大，但稳态解不受影响 —— 而我们
    关心的是天级尺度（≫ τ），已经远离角点。
    """
    g = torch.Generator().manual_seed(seed) if seed is not None else None

    # 初值：t = 0，全域 P = P0
    x_ic = torch.rand(n, 1, generator=g)
    s_ic = torch.zeros(n, 1)
    loss_ic = ((_p(net, x_ic, s_ic) - P0) ** 2).mean()

    # 左端 x = 0
    s_bc = torch.rand(n, 1, generator=g) * S_MAX
    x_left = torch.zeros(n, 1)
    p_left_target = P_WELL if scenario == "injection" else P0
    loss_left = ((_p(net, x_left, s_bc) - p_left_target) ** 2).mean()

    # 右端 x = L
    x_right = torch.ones(n, 1)
    loss_right = ((_p(net, x_right, s_bc) - P0) ** 2).mean()

    return loss_ic + loss_left + loss_right


def train(scenario: str = "baseline", epochs: int = 1000, lr: float = 1e-3,
          seed: int = 0, verbose: bool = True) -> dict:
    """训练一个场景，返回 loss 历史。

    1000 轮是需求给的预算。baseline 的常数解在几百轮内就收敛；injection 的
    阶跃边界需要更多轮才能把 x=0 附近的陡梯度压下来，所以默认给它更长的
    预算（见 ``DEFAULT_EPOCHS``）。
    """
    if scenario not in SCENARIOS:
        raise ValueError(f"未知场景 {scenario!r}，可选 {SCENARIOS}")

    torch.manual_seed(seed)
    net = PINN()
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    hist = {"scenario": scenario, "epochs": [], "total": [], "pde": [], "data": []}
    t0 = time.perf_counter()

    for ep in range(1, epochs + 1):
        opt.zero_grad()
        l_pde = physics_loss(net, seed=seed + ep)
        l_dat = data_loss(net, scenario, seed=seed + ep)
        loss = W_PDE * l_pde + l_dat
        loss.backward()
        opt.step()

        # .detach() 不是可有可无的：loss 带 requires_grad，直接 float() 会触发
        # "Converting a tensor with requires_grad=True to a scalar" 警告，
        # 而且历史里存的是计算图节点而不是数值，越攒越占内存。
        hist["epochs"].append(ep)
        hist["total"].append(float(loss.detach()))
        hist["pde"].append(float(l_pde.detach()))
        hist["data"].append(float(l_dat.detach()))

        if verbose and (ep == 1 or ep % max(1, epochs // 10) == 0 or ep == epochs):
            print(f"  epoch {ep:5d}/{epochs}  total={float(loss.detach()):.6e}  "
                  f"pde={float(l_pde.detach()):.3e}  data={float(l_dat.detach()):.3e}")

    hist["elapsed_s"] = round(time.perf_counter() - t0, 2)
    hist["final_total"] = hist["total"][-1]
    _save(net, scenario)
    return hist


def _weights_path(scenario: str) -> Path:
    return MODEL_DIR / f"pinn_{scenario}.pt"


def _save(net: PINN, scenario: str) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), _weights_path(scenario))


_CACHE: dict[str, PINN] = {}


def _load(scenario: str) -> PINN:
    """按场景加载权重（进程内缓存，避免每次工具调用都读盘）。"""
    if scenario in _CACHE:
        return _CACHE[scenario]
    path = _weights_path(scenario)
    if not path.exists():
        raise FileNotFoundError(
            f"缺少 {scenario} 场景权重 {path}。先运行 `python pinn_model.py` 训练一次。"
        )
    net = PINN()
    net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    net.eval()
    _CACHE[scenario] = net
    return net


def predict_pressure(x: float, t: float, scenario: str = "injection") -> float:
    """预测距井口 ``x`` 米、注入 ``t`` 天后的地层压力（bar）。

    Args:
        x: 距井口距离，米，取值 [0, L]。
        t: 注入时长，天，取值 [0, T_MAX_DAYS]。
        scenario: ``"injection"``（默认，井口 120 bar）或 ``"baseline"``
            （需求原文的边界条件，其解析解恒为 100 bar）。

    Returns:
        预测压力，单位 bar，保留 3 位小数。

    越界输入按计算域边界裁剪而不是抛异常：调用方是语言模型的工具调用，
    它偶尔会给出 x=150 这样的值，此时给出"最接近的可行答案"并让日志留痕，
    比让整条工作流失败更合适。
    """
    x = float(x)
    t = float(t)
    if not math.isfinite(x) or not math.isfinite(t):
        raise ValueError(f"x / t 必须是有限数值，收到 x={x}, t={t}")
    if scenario not in SCENARIOS:
        # 先校验场景名，再找权重。否则传错名字会报"缺少权重文件，请先训练"，
        # 把调用方引向"重训"这条错误的路（重训也不会造出 pinn_nonsense.pt）。
        raise ValueError(f"未知场景 {scenario!r}，可选 {SCENARIOS}")

    x_clamped = min(max(x, 0.0), L)
    t_clamped = min(max(t, 0.0), T_MAX_DAYS)
    if x_clamped != x or t_clamped != t:
        import logging
        logging.getLogger(__name__).warning(
            f"  [PINN] 输入越界已裁剪：x={x}→{x_clamped}, t={t}→{t_clamped}"
        )

    net = _load(scenario)
    x_hat = torch.tensor([[x_clamped / L]], dtype=torch.float32)
    s = torch.tensor([[t_clamped * SECONDS_PER_DAY / TAU]], dtype=torch.float32)
    with torch.no_grad():
        p = float(_p(net, x_hat, s))
    return round(p, 3)


def steady_state_pressure(x: float, p_well: float = P_WELL, p_far: float = P0) -> float:
    """injection 场景的解析稳态解（t ≫ τ 时的线性剖面），用于对账。

    这是给测试和报告用的"标准答案"：PINN 是数值解，没有可比的解析解就
    无法判断它学对了没有。t = 10 天 ≈ 8.6τ，已经充分接近稳态。
    """
    x = min(max(float(x), 0.0), L)
    return p_well - (p_well - p_far) * (x / L)


# ─── 输出 ───────────────────────────────────────────────────

_BLOCKS = "▁▂▃▄▅▆▇█"


def ascii_curve(values: list[float], width: int = 64, height: int = 12) -> str:
    """把 loss 序列画成终端里的 ASCII 曲线（对数刻度）。

    loss 跨越好几个数量级，线性刻度会让前 90% 的轮次挤在顶部一条线上，
    所以按 log10 取刻度 —— 这也是判断"还在降"还是"已经平台"的正确看法。
    """
    if not values:
        return ""
    logs = [math.log10(max(v, 1e-12)) for v in values]
    lo, hi = min(logs), max(logs)
    if hi - lo < 1e-9:
        hi = lo + 1.0

    # 降采样到 width 列（每列取区间最小值，保留"掉下去"的形状）
    step = max(1, len(logs) // width)
    cols = [min(logs[i:i + step]) for i in range(0, len(logs), step)][:width]

    grid = [[" "] * len(cols) for _ in range(height)]
    for c, v in enumerate(cols):
        frac = (v - lo) / (hi - lo)
        row = height - 1 - int(round(frac * (height - 1)))
        grid[row][c] = "●"

    lines = [f"  10^{hi:+.1f} ┐"]
    for i, row in enumerate(grid):
        label = "        │" if 0 < i < height - 1 else (
            f"  10^{hi:+.1f} ┤" if i == 0 else f"  10^{lo:+.1f} ┘")
        lines.append(label + "".join(row))
    lines.append("         └" + "─" * len(cols))
    lines.append(f"          1 {' ' * max(0, len(cols) - 12)}epoch {len(values)}")
    return "\n".join(lines)


def plot_curve(hist: dict, out: Path) -> Path:
    """把 loss 曲线存成 PNG（给 README / 汇报用）。matplotlib 是可选的。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 3.4), dpi=140)
    ax.plot(hist["epochs"], hist["total"], label="total", lw=1.8, color="#1f4e79")
    ax.plot(hist["epochs"], hist["pde"], label="physics (PDE)", lw=1.1, ls="--", color="#c0504d")
    ax.plot(hist["epochs"], hist["data"], label="IC + BC", lw=1.1, ls=":", color="#4f81bd")
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss (log scale)")
    ax.set_title(f"PINN training — {hist['scenario']} ({hist['epochs'][-1]} epochs, "
                 f"{hist['elapsed_s']}s)", fontsize=9)
    ax.grid(alpha=0.3, which="both", lw=0.5)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


DEFAULT_EPOCHS = {"baseline": 1000, "injection": 4000}


def main() -> int:
    ap = argparse.ArgumentParser(description="训练一维 PINN 压力预测模型")
    ap.add_argument("--epochs", type=int, default=None,
                    help="训练轮数（默认 baseline 1000 / injection 4000）")
    ap.add_argument("--scenario", choices=SCENARIOS, default=None,
                    help="只训练指定场景（默认两种都训）")
    ap.add_argument("--no-plot", action="store_true", help="不生成 PNG 曲线")
    args = ap.parse_args()

    scenarios = [args.scenario] if args.scenario else list(SCENARIOS)
    results = {}

    for sc in scenarios:
        epochs = args.epochs or DEFAULT_EPOCHS[sc]
        print(f"\n{'=' * 68}\n训练场景：{sc}   轮数：{epochs}\n{'=' * 68}")
        hist = train(sc, epochs=epochs, verbose=True)
        results[sc] = hist

        print(f"\n  Loss 下降曲线（{sc}，对数刻度）：")
        print(ascii_curve(hist["total"]))
        print(f"\n  首轮 total = {hist['total'][0]:.6e}  →  "
              f"末轮 total = {hist['total'][-1]:.6e}  "
              f"（下降 {hist['total'][0] / max(hist['total'][-1], 1e-12):.0f}×）")

        if not args.no_plot:
            png = plot_curve(hist, MODEL_DIR / f"loss_{sc}.png")
            print(f"  曲线图：{png}")

    # ─── 对账：把 PINN 的预测和解析解/常识比一下 ───
    print(f"\n{'=' * 68}\n预测结果\n{'=' * 68}")
    for sc in scenarios:
        print(f"\n[{sc}]")
        for x in (0.0, 25.0, 50.0, 75.0, 100.0):
            p = predict_pressure(x, 10.0, scenario=sc)
            print(f"  x={x:5.1f} m, t=10 d  →  P = {p:8.3f} bar")
        if sc == "injection":
            print(f"  解析稳态解 x=50 m      →  P = {steady_state_pressure(50):8.3f} bar")
            err = abs(predict_pressure(50, 10, scenario=sc) - steady_state_pressure(50))
            print(f"  |PINN − 解析解|        →  {err:.3f} bar")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
