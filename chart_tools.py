"""
专业图表生成（Function Calling 工具）
=====================================
把"地层压力随深度变化"画成一张可直接放进报告的地质图。

为什么要单独成模块：
- 绘图代码一旦塞进节点模块（nodes/aggregate_node.py），就会被那堆重依赖
  （LightRAG / reranker）拖着，连"能不能画出图"这种事都没法单独测。
- 这里只依赖 matplotlib + 文件系统，可以在 CI 里跑真实绘图断言。

物理模型（简化但自洽，图上每个数都能对上）
------------------------------------------
  静水压力     P_h(d)    = ρ_w·g·d / 1e5        ρ_w = 1030 kg/m³（含盐地层水）
  地层破裂压力 P_frac(d) = 1.5 · P_h(d)          破裂压力梯度常用 1.4–1.8 倍静水
  注入压力     P_inj     用户给定（井底压力，常数）

**正确读图方式**：注入压力是一条定值竖线，判断只在"储层深度"这一行做，
把 P_inj 与 P_frac(d_res)、P_h(d_res) 比较，得到三种工程结论：

  P_inj > P_frac(d_res)  → 超过地层破裂压力，盖层压裂 / 泄漏风险
  P_inj < P_h(d_res)     → 低于地层压力，注入困难（需要更高井口压力）
  介于两者之间            → 落在安全注入窗口内

另一个有用派生量是"可注入深度范围"：把 P_inj 这条竖线夹在两条曲线之间，
得到 [d_min, d_max]，即在这个深度区间内该注入压力才既注得进又不压裂地层。

⚠️ 早前版本把"浅部 P_inj > P_frac"当作风险区，那是错的 —— 浅部根本不注入，
   让定值压力去比浅部破裂压力没有工程意义。现在按上述窗口逻辑重写。

注意：这是报告用的工程近似，不是油藏数值模拟结果。
"""
from __future__ import annotations

import os
import uuid
from datetime import datetime

import matplotlib

matplotlib.use("Agg")  # 无显示环境（服务器 / CI）必须用非交互后端

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CHART_DIR = os.path.join(PROJECT_ROOT, "charts")

RHO_WATER = 1030.0          # kg/m³
G = 9.81                    # m/s²
FRACTURE_RATIO = 1.5        # 破裂压力 / 静水压力
DEFAULT_DEPTH_M = 2000.0
DEFAULT_INJECTION_BAR = 250.0

# ─── 中文字体：优先 CJK，找不到就退回英文标签 ──────────────
# 不能省：matplotlib 默认字体没有汉字，缺字体时中文全渲染成方块。
# Linux CI 上通常没有中文字体，所以准备两套标签。
_CJK_CANDIDATES = [
    "Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC",
    "Source Han Sans SC", "WenQuanYi Zen Hei", "PingFang SC",
]

_LABELS_ZH = {
    "title": "地层压力随深度变化曲线",
    "xlabel": "压力 (bar)",
    "ylabel": "深度 (m)",
    "hydro": "静水压力 P_h",
    "frac": "地层破裂压力 P_frac",
    "inj": "注入压力 P_inj",
    "window": "安全注入窗口（P_h ~ P_frac）",
    "depth": "储层深度",
    "range": "可注入深度范围",
    "risk": "超过破裂压力 → 压裂/泄漏风险",
    "under": "低于地层压力 → 注入困难",
    "ok": "落在安全注入窗口内",
}

_LABELS_EN = {
    "title": "Formation Pressure vs Depth",
    "xlabel": "Pressure (bar)",
    "ylabel": "Depth (m)",
    "hydro": "Hydrostatic P_h",
    "frac": "Fracture pressure P_frac",
    "inj": "Injection pressure P_inj",
    "window": "Safe injection window (P_h ~ P_frac)",
    "depth": "Reservoir depth",
    "range": "Admissible depth range",
    "risk": "Above fracture pressure -> risk",
    "under": "Below formation pressure -> hard to inject",
    "ok": "Within the safe injection window",
}


def _pick_cjk_font() -> str | None:
    try:
        from matplotlib import font_manager

        available = {f.name for f in font_manager.fontManager.ttflist}
        for name in _CJK_CANDIDATES:
            if name in available:
                return name
    except Exception:
        pass
    return None


def _apply_style() -> dict:
    """应用字体与样式，返回实际使用的标签表。"""
    cjk = _pick_cjk_font()
    if cjk:
        plt.rcParams["font.sans-serif"] = [cjk] + plt.rcParams["font.sans-serif"]
        labels = _LABELS_ZH
    else:
        labels = _LABELS_EN
    plt.rcParams["axes.unicode_minus"] = False
    return labels


def hydrostatic_bar(depth_m):
    """深度（m）→ 静水压力（bar）。1 bar = 1e5 Pa。"""
    return RHO_WATER * G * np.asarray(depth_m, dtype=float) / 1e5


def fracture_bar(depth_m):
    """深度（m）→ 地层破裂压力（bar）。"""
    return hydrostatic_bar(depth_m) * FRACTURE_RATIO


def admissible_depth_range(injection_pressure_bar: float) -> tuple[float, float]:
    """
    给定注入压力，求"既注得进又不压裂"的深度区间 [d_min, d_max]。

    P_frac(d) >= P_inj  →  d >= P_inj / (0.1013·1.5)   （下界：不压裂）
    P_h(d)    <= P_inj  →  d <= P_inj / 0.1013         （上界：注得进）
    返回 (d_min, d_max)，单位 m。
    """
    p = float(injection_pressure_bar)
    d_min = p / (RHO_WATER * G / 1e5 * FRACTURE_RATIO)
    d_max = p / (RHO_WATER * G / 1e5)
    return float(d_min), float(d_max)


def _relative_to_project(path: str) -> str:
    """
    把绝对路径转成"相对项目根"的 POSIX 风格路径，用于写进 Markdown。

    os.path.relpath 在跨盘符时（Windows 下 C: 与 D: 之间）会抛 ValueError，
    而不是返回一个奇怪的相对路径。图表目录可以被配置到别的盘，所以必须兜住：
    跨盘时退回绝对路径（换成正斜杠），至少不会崩。
    """
    try:
        rel = os.path.relpath(path, PROJECT_ROOT)
        if not rel.startswith(".."):
            return rel.replace(os.sep, "/")
    except ValueError:
        pass
    return os.path.abspath(path).replace(os.sep, "/")


def generate_pressure_curve(
    depth_m: float | None = None,
    injection_pressure_bar: float | None = None,
    output_dir: str | None = None,
    dpi: int = 140,
) -> dict:
    """
    生成"地层压力随深度变化"曲线图，返回图片路径与关键结论。

    返回字典（供 Agent 与前端消费）：
        {"status": "ok", "path", "rel_path", "url_path", "summary",
         "verdict": ok|risky|under, "depth_m", "injection_pressure_bar",
         "fracture_bar_at_depth", "hydrostatic_bar_at_depth",
         "admissible_depth_min_m", "admissible_depth_max_m",
         "depth_assumed": bool}
    """
    depth_assumed = depth_m in (None, "", 0)
    depth_m = float(depth_m) if not depth_assumed else DEFAULT_DEPTH_M
    injection_pressure_bar = (
        float(injection_pressure_bar) if injection_pressure_bar
        else DEFAULT_INJECTION_BAR
    )
    # 防止模型传入离谱数值把图压扁
    depth_m = max(200.0, min(depth_m, 8000.0))
    injection_pressure_bar = max(20.0, min(injection_pressure_bar, 1200.0))

    labels = _apply_style()

    p_h_at_depth = float(hydrostatic_bar(depth_m))
    p_frac_at_depth = float(fracture_bar(depth_m))
    d_min, d_max = admissible_depth_range(injection_pressure_bar)

    if injection_pressure_bar > p_frac_at_depth:
        verdict = "risky"
        verdict_text = labels["risk"]
    elif injection_pressure_bar < p_h_at_depth:
        verdict = "under"
        verdict_text = labels["under"]
    else:
        verdict = "ok"
        verdict_text = labels["ok"]

    # ─── 绘图 ─────────────────────────────────────────
    d_top = max(0.0, min(d_min, depth_m) * 0.45)
    d_bottom = max(depth_m, d_max) * 1.18
    depths = np.linspace(d_top, d_bottom, 240)
    p_hydro = hydrostatic_bar(depths)
    p_frac = fracture_bar(depths)

    fig, ax = plt.subplots(figsize=(7.4, 5.2))

    # 安全窗口：两条曲线之间的带状区域
    ax.fill_betweenx(depths, p_hydro, p_frac, color="#2ca02c", alpha=0.11,
                     label=labels["window"], zorder=1)

    ax.plot(p_hydro, depths, color="#1f77b4", lw=2.2, label=labels["hydro"], zorder=3)
    ax.plot(p_frac, depths, color="#ff7f0e", lw=2.2, ls="--",
            label=labels["frac"], zorder=3)
    ax.axvline(injection_pressure_bar, color="#d62728", lw=2.0, zorder=4,
               label=f"{labels['inj']} = {injection_pressure_bar:.0f} bar")

    # 可注入深度范围（两条曲线与该竖线的交点之间）
    ax.annotate("", xy=(injection_pressure_bar * 1.02, d_min),
                xytext=(injection_pressure_bar * 1.02, d_max),
                arrowprops=dict(arrowstyle="<->", color="#8e44ad", lw=1.6), zorder=5)
    ax.annotate(
        f"{labels['range']}\n{d_min:.0f}–{d_max:.0f} m",
        xy=(injection_pressure_bar, (d_min + d_max) / 2),
        xytext=(26, 0), textcoords="offset points",
        fontsize=8.5, color="#6c3483", va="center",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#d2b4de", alpha=0.92),
        zorder=6,
    )

    # 储层深度横线 + 该深度处三个压力点
    ax.axhline(depth_m, color="#7f7f7f", lw=1.0, ls=":", zorder=2)
    ax.scatter([p_h_at_depth], [depth_m], color="#1f77b4", s=46, zorder=7)
    ax.scatter([p_frac_at_depth], [depth_m], color="#ff7f0e", s=46, zorder=7)
    ax.scatter([injection_pressure_bar], [depth_m], color="#d62728", s=52,
               marker="D", zorder=8)

    color = {"risky": "#c0392b", "under": "#b9770e", "ok": "#1e8449"}[verdict]
    ax.annotate(
        f"{labels['depth']} {depth_m:.0f} m  ·  {verdict_text}\n"
        f"P_h={p_h_at_depth:.0f} | P_inj={injection_pressure_bar:.0f} | "
        f"P_frac={p_frac_at_depth:.0f} bar",
        xy=(injection_pressure_bar, depth_m),
        xytext=(14, 30), textcoords="offset points",
        fontsize=8.5, color=color, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.34", fc="white", ec=color, alpha=0.94),
        arrowprops=dict(arrowstyle="-", color=color, lw=0.9),
        zorder=9,
    )

    ax.set_xlabel(labels["xlabel"], fontsize=10)
    ax.set_ylabel(labels["ylabel"], fontsize=10)
    ax.set_title(labels["title"], fontsize=12.5, fontweight="bold", pad=12)
    ax.invert_yaxis()
    ax.grid(alpha=0.26, ls="--", lw=0.7)
    ax.legend(fontsize=8.2, loc="lower right", framealpha=0.93)
    ax.set_xlim(left=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    footnote = (
        "Hydrostatic: rho=1030 kg/m3 | Fracture gradient: 1.5 x hydrostatic | "
        "Engineering approximation, not a reservoir simulation"
    )
    if depth_assumed:
        footnote += f" | Depths assumed {depth_m:.0f} m (not provided)"
    fig.text(0.012, 0.014, footnote, fontsize=6.6, color="#8a8a8a")
    fig.tight_layout(rect=(0, 0.035, 1, 1))

    # ─── 落盘 ─────────────────────────────────────────
    out_dir = output_dir or CHART_DIR
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"pressure_curve_{stamp}_{uuid.uuid4().hex[:6]}.png"
    path = os.path.join(out_dir, filename)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)  # 必须关：否则反复调用会泄漏内存与文件句柄

    rel_path = _relative_to_project(path)
    assumed_note = "（未提供储层深度，按 2000 m 估算）" if depth_assumed else ""

    if verdict == "risky":
        summary = (
            f"注入压力 {injection_pressure_bar:.0f} bar 在 {depth_m:.0f} m 处"
            f"高于地层破裂压力 {p_frac_at_depth:.0f} bar，"
            f"存在盖层压裂与 CO2 泄漏风险，应降低井底压力。{assumed_note}"
        )
    elif verdict == "under":
        summary = (
            f"注入压力 {injection_pressure_bar:.0f} bar 在 {depth_m:.0f} m 处"
            f"低于地层压力 {p_h_at_depth:.0f} bar，难以有效注入；"
            f"该压力对应的可注入深度上限约 {d_max:.0f} m。{assumed_note}"
        )
    else:
        summary = (
            f"注入压力 {injection_pressure_bar:.0f} bar 在 {depth_m:.0f} m 处"
            f"位于安全注入窗口内（P_h={p_h_at_depth:.0f} bar、"
            f"P_frac={p_frac_at_depth:.0f} bar），可注入深度范围约 "
            f"{d_min:.0f}–{d_max:.0f} m。{assumed_note}"
        )

    return {
        "status": "ok",
        "path": path,
        "rel_path": rel_path,
        "url_path": f"/charts/{filename}",
        "summary": summary,
        "verdict": verdict,
        "verdict_text": verdict_text,
        "depth_m": depth_m,
        "injection_pressure_bar": injection_pressure_bar,
        "hydrostatic_bar_at_depth": round(p_h_at_depth, 1),
        "fracture_bar_at_depth": round(p_frac_at_depth, 1),
        "admissible_depth_min_m": round(d_min, 0),
        "admissible_depth_max_m": round(d_max, 0),
        "depth_assumed": bool(depth_assumed),
        "cjk_font": _pick_cjk_font() or "",
    }
