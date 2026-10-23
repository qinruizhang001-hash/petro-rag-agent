"""
控制台的可测部分（纯函数，不碰 Streamlit 运行时）
==================================================
``app.py`` 里有一半代码其实是"给定数字，算出一张图 / 一段 HTML"，
跟 Streamlit 没有关系。把它们留在一个必须在脚本上下文里才能 import 的文件里，
代价是 **CI 里一行都测不了** —— 只能靠人肉打开浏览器看。

所以这里收着所有不依赖 ``st`` 的东西：流水线定义、数字格式化、入参拼接、
三张 Plotly 图、流水线状态 HTML。``app.py`` 只负责布局与 HTTP。

依赖：plotly（无 matplotlib —— CI 的 ``frontend-contract`` 禁止前端引入绘图库）
"""
from __future__ import annotations

import plotly.graph_objects as go

from pressure_model import (
    admissible_depth_range,
    clamp_inputs,
    fracture_bar,
    hydrostatic_bar,
    verdict_of,
)

# ─── 流水线（顺序即拓扑：extract 之后两条分支并发，aggregate 扇入）───
PIPELINE = [
    ("memory_router", "记忆路由"),
    ("search_memory", "记忆检索"),
    ("inject_memory", "记忆注入"),
    ("extract", "参数提取"),
    ("rewrite_query", "查询重写"),
    ("geology_retrieve", "地质检索"),
    ("wellbore_retrieve", "井筒检索"),
    ("grade", "Self-RAG 评判"),
    ("geology_generate", "地质结论"),
    ("wellbore_generate", "井筒结论"),
    ("aggregate", "汇总结稿"),
    ("evaluate", "质量评审"),
    ("hitl_gate", "审核闸门"),
    ("save_file_memory", "历史落盘"),
]
NODE_SHORT = dict(PIPELINE)

VERDICT_TEXT = {
    "ok": ("安全", "#1a7f37"),
    "risky": ("超破裂压力", "#cf222e"),
    "under": ("低于静水压力", "#9a6700"),
}

# 蒙特卡洛三类结局的中文名（后端 tools.physics_tools 用同一套措辞，
# 前端这里只做展示，不参与判定）
UQ_VERDICT_LABELS = {
    "ok": "注入可行",
    "risky": "压裂风险",
    "under": "压力不足（注不进）",
}

# 图表配色：一条线一个固定色，前端与后端 PNG（chart_tools）保持一致
COL_HYDRO = "#1f77b4"
COL_FRAC = "#ff7f0e"
COL_INJ = "#cf222e"
GRID = "rgba(128,128,128,.18)"


# ─── 格式化 ──────────────────────────────────────────────
def fmt_s(seconds) -> str:
    try:
        s = float(seconds)
    except (TypeError, ValueError):
        return "—"
    return f"{s:.1f}s" if s < 60 else f"{s / 60:.1f}min"


def fmt_ms(ms) -> str:
    if ms is None:
        return "—"
    return f"{ms / 1000:.2f}s" if ms >= 1000 else f"{ms:.0f}ms"


def fmt_tok(n) -> str:
    try:
        n = int(n)
    except (TypeError, ValueError):
        return "—"
    return f"{n / 1000:.1f}k" if n >= 1000 else str(n)


def compose_input(desc: str, depth, p_inj, temp, scenario) -> str:
    """
    把左侧表单拼进描述里。

    为什么不在前端解析文本：参数提取是后端的职责（它有先验、有规则校验）。
    但用户既然在控制台上填了数，就不该让他再在正文里重复一遍 ——
    所以拼成一段带标题的结构化块，并写明"冲突时以此为准"，
    让 extract 节点有明确的取舍依据，而不是两个数打架。
    """
    desc = (desc or "").strip()
    block = (
        "【结构化参数（来自控制台表单，与正文冲突时以此为准）】\n"
        f"- 储层埋深：{float(depth):.0f} m\n"
        f"- 注入压力：{float(p_inj):.0f} bar\n"
        f"- 地层温度：{float(temp):.0f} ℃\n"
        f"- 工况：{scenario}"
    )
    return f"{desc}\n\n{block}" if desc else block


# ─── 图 1：地层压力—深度曲线 ─────────────────────────────
def pressure_figure(depth_m: float, p_inj: float, height: int = 380) -> go.Figure:
    """
    实时曲线：两条地层压力包络 + 一条注入压力竖线 + 阈值线 + 工作点。

    读数只在**储层深度**这一行做（浅部不注入，拿定值压力去比浅部破裂压力
    没有工程意义）。所以阈值线画在 P_frac(d_res) 与 P_h(d_res) 上，
    而不是画"曲线交点"。
    """
    d_res, p = clamp_inputs(depth_m, p_inj)
    p_h_res = float(hydrostatic_bar(d_res))
    p_f_res = float(fracture_bar(d_res))
    d_min, d_max = admissible_depth_range(p)

    # 视窗：既能看到可注入窗口，也留出储层深度上下的余量
    d_top = max(0.0, min(d_min, d_res) * 0.45)
    d_bottom = max(d_res, d_max) * 1.18
    n = 120
    depths = [d_top + (d_bottom - d_top) * i / n for i in range(n + 1)]
    p_h = hydrostatic_bar(depths)
    p_f = fracture_bar(depths)

    verdict = verdict_of(d_res, p)
    v_text, v_color = VERDICT_TEXT[verdict]

    fig = go.Figure()

    # 安全注入窗口（两条包络之间）
    fig.add_trace(go.Scatter(
        x=list(p_h) + list(p_f)[::-1],
        y=depths + depths[::-1],
        fill="toself", fillcolor="rgba(26,127,55,.10)",
        line=dict(width=0), hoverinfo="skip",
        name="安全注入窗口",
    ))
    fig.add_trace(go.Scatter(
        x=p_h, y=depths, mode="lines", name="静水压力 P_h",
        line=dict(color=COL_HYDRO, width=2),
        hovertemplate="P_h=%{x:.0f} bar<br>深度 %{y:.0f} m<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=p_f, y=depths, mode="lines", name="破裂压力 P_frac",
        line=dict(color=COL_FRAC, width=2, dash="dash"),
        hovertemplate="P_frac=%{x:.0f} bar<br>深度 %{y:.0f} m<extra></extra>",
    ))

    # 注入压力：定值竖线
    fig.add_trace(go.Scatter(
        x=[p, p], y=[d_top, d_bottom], mode="lines", name=f"注入压力 {p:.0f} bar",
        line=dict(color=COL_INJ, width=2.4),
        hovertemplate="P_inj=%{x:.0f} bar<extra></extra>",
    ))

    # 风险阈值线：储层深度处的破裂压力与静水压力
    # 标注一律放在绘图区顶部并向下偏移：图例已经挪到绘图区外的上边距，
    # 两条竖线标注又在不同 x 上，三者不会再压到一起（第一版就是压着的）。
    fig.add_vline(
        x=p_f_res, line=dict(color=COL_INJ, width=1.2, dash="dot"),
        annotation_text=f"破裂阈值 {p_f_res:.0f} bar",
        annotation_position="top", annotation_yshift=6, annotation_font_size=10,
    )
    fig.add_vline(
        x=p_h_res, line=dict(color=COL_HYDRO, width=1.2, dash="dot"),
        annotation_text=f"静水下限 {p_h_res:.0f} bar",
        annotation_position="top", annotation_yshift=6, annotation_font_size=10,
    )
    fig.add_hline(
        y=d_res, line=dict(color="rgba(128,128,128,.6)", width=1, dash="dot"),
        annotation_text=f"储层 {d_res:.0f} m",
        annotation_position="right", annotation_font_size=10,
    )

    # 工作点
    fig.add_trace(go.Scatter(
        x=[p], y=[d_res], mode="markers+text",
        marker=dict(size=13, color=v_color, line=dict(color="#fff", width=1.5)),
        text=[v_text], textposition="middle right",
        textfont=dict(size=11, color=v_color),
        name="工作点",
        hovertemplate=f"{v_text}<br>P_inj=%{{x:.0f}} bar · 深度 %{{y:.0f}} m<extra></extra>",
    ))

    fig.update_layout(
        height=height,
        # 左边距留给 y 轴标题（4 位数刻度 + 竖排标题，8px 会被挤在一起）；
        # 上边距留给图例，右边距留给"储层 xxx m"那条横线标注 —— 留窄了会被裁掉。
        margin=dict(l=56, r=64, t=58, b=36),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.10,
                    xanchor="left", x=0, font=dict(size=10)),
        xaxis=dict(title="压力 (bar)", gridcolor=GRID, zeroline=False),
        # 深度向下增大 —— 地质图的基本约定，画反了没人能读
        yaxis=dict(title="深度 (m)", autorange="reversed",
                   gridcolor=GRID, zeroline=False),
    )
    return fig


# ─── 图 2：全链路耗时 ────────────────────────────────────
def timing_figure(rows: list[dict]) -> go.Figure:
    """横向条形图：每个节点的墙钟耗时。并行分支会明显重叠，这是要看的东西。"""
    rows = [r for r in rows if r.get("ms") is not None]
    rows.sort(key=lambda r: r["ms"])
    fig = go.Figure(go.Bar(
        x=[r["ms"] / 1000 for r in rows],
        y=[r["label"] for r in rows],
        orientation="h",
        marker=dict(color="rgba(31,119,180,.75)"),
        text=[fmt_ms(r["ms"]) for r in rows],
        textposition="outside",
        hovertemplate="%{y}<br>%{x:.2f}s<extra></extra>",
    ))
    fig.update_layout(
        height=max(150, 34 * len(rows) + 60),
        margin=dict(l=8, r=40, t=8, b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="秒", gridcolor=GRID),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
        showlegend=False,
    )
    return fig


# ─── 图 4：不确定性分布 ──────────────────────────────────
def uq_figure(
    samples: list[float],
    p5: float | None = None,
    p95: float | None = None,
    mean: float | None = None,
    unit: str = "",
    height: int = 280,
) -> go.Figure:
    """
    蒙特卡洛结果的分布直方图，叠加均值与 90% 区间边界。

    为什么值得画：只报"1.34（1.29–1.40）"时，读者无法判断这个区间是
    因为模型敏感还是因为抽样噪声；看到分布形状，才知道它是对称、偏斜、
    还是双峰（双峰通常意味着判定在某个阈值两侧来回跳，是真正需要注意的信号）。
    """
    xs = [float(v) for v in (samples or [])]
    fig = go.Figure()
    if not xs:
        fig.update_layout(height=height, margin=dict(l=8, r=8, t=8, b=8))
        return fig

    fig.add_trace(go.Histogram(
        x=xs,
        nbinsx=20,
        marker=dict(color="rgba(44,111,187,.60)", line=dict(width=0)),
        hovertemplate=f"%{{x:.3f}} {unit}<br>%{{y}} 次<extra></extra>",
        name="模拟样本",
    ))

    for value, color, dash, label in (
        (p5, "#cf222e", "dash", "p5"),
        (p95, "#cf222e", "dash", "p95"),
        (mean, "#1a7f37", "solid", "均值"),
    ):
        if value is None:
            continue
        fig.add_vline(
            x=float(value),
            line=dict(color=color, width=1.4, dash=dash),
            annotation_text=f"{label} {float(value):.2f}",
            annotation_position="top",
            annotation_font=dict(size=10, color=color),
        )

    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=28, b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        bargap=0.04,
        showlegend=False,
        xaxis=dict(title=f"模拟结果（{unit}）" if unit else "模拟结果",
                   gridcolor=GRID),
        yaxis=dict(title="频次", gridcolor=GRID),
    )
    return fig


# ─── 图 3：Token 分布 ────────────────────────────────────
def token_figure(breakdown: list[dict]) -> go.Figure:
    """按阶段堆叠：输入 vs 输出。输入通常占大头，所以两个都要画。"""
    stages = [b.get("stage") or "?" for b in breakdown]
    tin = [int(b.get("input_tokens") or 0) for b in breakdown]
    tout = [int(b.get("output_tokens") or 0) for b in breakdown]
    fig = go.Figure()
    fig.add_bar(name="输入", x=stages, y=tin, marker_color="rgba(31,119,180,.75)")
    fig.add_bar(name="输出", x=stages, y=tout, marker_color="rgba(255,127,14,.85)")
    fig.update_layout(
        barmode="stack", height=260,
        margin=dict(l=8, r=8, t=8, b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=1.12, x=0, font=dict(size=10)),
        xaxis=dict(gridcolor="rgba(0,0,0,0)"),
        yaxis=dict(title="tokens", gridcolor=GRID),
    )
    return fig


# ─── 流水线状态 ──────────────────────────────────────────
def pipeline_html(states: dict, timings: dict, running: bool) -> str:
    """
    states:  {node: "start"|"end"}
    timings: {node: {"start": ms, "end": ms}}

    没跑完的节点在运行中显示灰色省略号，跑完后显示"跳过"——
    这比永远灰色诚实：那一步是真的没执行（比如 memory_router 判定不需要记忆）。
    """
    out = ['<div class="pipe">']
    for node, label in PIPELINE:
        st_ = states.get(node)
        t = timings.get(node) or {}
        if st_ == "end":
            led, cls = "led-ok", "done"
            dur = (fmt_ms(t["end"] - t["start"])
                   if t.get("start") is not None and t.get("end") is not None else "—")
        elif st_ == "start":
            led, cls, dur = "led-run", "", "运行中"
        else:
            led, cls = "led-idle", "wait"
            # 从没跑过 → "—"；跑过但这一步没执行 → "跳过"。两者的区别是信息，
            # 一律写"跳过"会让人以为流程真的跳过了。
            if running:
                dur = "…"
            elif states:
                dur = "跳过"
            else:
                dur = "—"
        out.append(
            f'<div class="row {cls}"><span class="led {led}"></span>'
            f'<span class="name">{label}</span>'
            f'<span class="dur">{dur}</span></div>'
        )
    out.append("</div>")
    return "".join(out)


# ─── 图 5：五维风险雷达图 ────────────────────────────────
def risk_radar_figure(
    scores: dict[str, float],
    height: int = 340,
) -> go.Figure:
    """
    五维风险雷达图。分数 0–100，越高表示该维度风险越大。

    五个维度是 CCUS 项目评估的通用框架：
    · 地质风险 —— 储层封闭性、断层活性、盖层完整性
    · 井筒风险 —— 注入压力是否超破裂压力、井筒腐蚀
    · 环境风险 —— CO₂ 泄漏概率、地下水污染
    · 经济风险 —— 监测成本、钻探成本、碳价波动
    · 合规风险 —— 审批难度、法规符合性、社区接受度
    """
    categories = ["地质风险", "井筒风险", "环境风险", "经济风险", "合规风险"]
    values = [max(0.0, min(100.0, float(scores.get(c, 0)))) for c in categories]
    # 闭合雷达图：首尾相接
    values_closed = values + [values[0]]
    categories_closed = categories + [categories[0]]

    fig = go.Figure()

    # 背景网格：每 20 分一个浅色环
    for ring in [20, 40, 60, 80]:
        fig.add_trace(go.Scatterpolar(
            r=[ring] * len(categories_closed),
            theta=categories_closed,
            mode="lines",
            line=dict(color="rgba(128,128,128,.15)", width=1),
            fill="toself",
            fillcolor="rgba(128,128,128,.04)",
            hoverinfo="skip",
            showlegend=False,
        ))

    # 实际得分面
    color_fill = "rgba(207,34,46,.18)" if max(values) >= 70 else \
                 "rgba(154,103,0,.18)" if max(values) >= 40 else \
                 "rgba(26,127,55,.18)"
    color_line = "#cf222e" if max(values) >= 70 else \
                 "#9a6700" if max(values) >= 40 else \
                 "#1a7f37"

    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=categories_closed,
        mode="lines+markers+text",
        line=dict(color=color_line, width=2.2),
        marker=dict(size=7, color=color_line, line=dict(color="#fff", width=1.2)),
        text=[f"{v:.0f}" for v in values] + [""],
        textposition="top center",
        textfont=dict(size=10, color=color_line),
        fill="toself",
        fillcolor=color_fill,
        name="风险评分",
        hovertemplate="%{theta}: %{r:.0f}/100<extra></extra>",
    ))

    fig.update_layout(
        height=height,
        margin=dict(l=48, r=48, t=48, b=32),
        paper_bgcolor="rgba(0,0,0,0)",
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                visible=True, range=[0, 100],
                tickvals=[20, 40, 60, 80, 100],
                ticktext=["20", "40", "60", "80", "100"],
                gridcolor="rgba(128,128,128,.20)",
                tickfont=dict(size=9, color="rgba(128,128,128,.85)"),
            ),
            angularaxis=dict(
                gridcolor="rgba(128,128,128,.20)",
                tickfont=dict(size=11, color="rgba(80,80,80,.95)"),
                rotation=90,  # 让"地质风险"在正上方
                direction="clockwise",
            ),
        ),
        showlegend=False,
    )
    return fig


def compute_risk_scores(
    done: dict,
    depth_m: float,
    p_inj: float,
    temp: float,
) -> dict[str, float]:
    """
    从评估结果和当前参数计算五维风险分数（0–100）。

    不需要后端新增字段：所有输入都来自现有 API 响应和左侧参数。
    """
    from pressure_model import clamp_inputs, fracture_bar, hydrostatic_bar, verdict_of

    d, p = clamp_inputs(depth_m, p_inj)
    p_h = float(hydrostatic_bar(d))
    p_f = float(fracture_bar(d))
    margin = p_f - p  # 破裂压力裕度（bar）
    verdict = verdict_of(d, p)

    # ── 地质风险 ──
    # verdict 是核心：risky 直接给高分；under 也危险（注不进 = 项目失败）
    geo_base = {"ok": 25, "risky": 85, "under": 65}[verdict]
    # 裕度越小，地质风险越高（盖层/断层问题更易触发）
    geo_margin_penalty = max(0, (30 - margin)) * 1.5  # 裕度 < 30 bar 时加分
    geo_score = min(100, geo_base + geo_margin_penalty)

    # ── 井筒风险 ──
    # 从注入风险概率中取 risky 概率；如果没有 UQ 结果，用 verdict 近似
    inj_probs = ((done.get("uq") or {}).get("injection") or {}).get("probabilities") or {}
    if inj_probs:
        well_score = (inj_probs.get("risky", 0) * 80 +
                      inj_probs.get("under", 0) * 40 +
                      inj_probs.get("ok", 0) * 10)
    else:
        well_score = {"ok": 15, "risky": 90, "under": 50}[verdict]
    # 温度越高，井筒腐蚀/水泥环降解风险越高
    well_temp_bonus = max(0, (temp - 60)) * 0.4
    well_score = min(100, well_score + well_temp_bonus)

    # ── 环境风险 ──
    # 深度越浅，CO₂ 泄漏到地表的概率越高；>3000m 几乎可忽略
    if d < 1000:
        env_depth = 75
    elif d < 2000:
        env_depth = 50 - (d - 1000) * 0.025
    elif d < 3000:
        env_depth = 25 - (d - 2000) * 0.015
    else:
        env_depth = 10
    # 超破裂压力 → 诱导微地震 → 环境扰动
    env_frack_bonus = 25 if verdict == "risky" else 0
    env_score = min(100, env_depth + env_frack_bonus)

    # ── 经济风险 ──
    # 深度越深深井成本越高；注入压力接近破裂压力时需要昂贵实时监测
    econ_depth = min(60, d * 0.015)  # 2000m → 30 分
    # 裕度 < 20 bar 时，需要密集微地震监测和高频压力计 → 成本飙升
    econ_margin = max(0, (40 - margin)) * 1.2
    # 温度高 → 需要耐腐蚀材料 → 成本增加
    econ_temp = max(0, (temp - 50)) * 0.3
    econ_score = min(100, econ_depth + econ_margin + econ_temp)

    # ── 合规风险 ──
    # 超压、断层、浅层注入都是审批红线
    comp_base = {"ok": 20, "risky": 80, "under": 55}[verdict]
    # 浅层（<1000m）在许多司法管辖区禁止或严格限制 CO₂ 注入
    comp_shallow = 20 if d < 1000 else 0
    comp_score = min(100, comp_base + comp_shallow)

    return {
        "地质风险": geo_score,
        "井筒风险": well_score,
        "环境风险": env_score,
        "经济风险": econ_score,
        "合规风险": comp_score,
    }


# ─── 图 6：地层剖面示意图 ────────────────────────────────
def stratigraphy_figure(
    depth_m: float,
    p_inj: float,
    temp: float,
    has_fault: bool = True,
    height: int = 420,
) -> go.Figure:
    """
    简化地质地层剖面示意图（二维横截面）。

    用 Plotly 的填充面积图"伪造"地层，不是真实地质建模，
    目的是让非地质背景的决策者一眼看懂：
    "注入层在哪、盖层有多厚、断层会不会切穿储层"。

    随参数动态更新：
    · depth_m 变化 → 注入层上下移动
    · p_inj 变化 → 工作点颜色变化（绿/黄/红）
    · has_fault 变化 → 断层线显示/隐藏
    """
    from pressure_model import clamp_inputs, verdict_of

    d, p = clamp_inputs(depth_m, p_inj)
    verdict = verdict_of(d, p)

    # x 范围：0–2000 m（横向距离）
    x = list(range(0, 2001, 100))

    # ── 地层边界定义（y 向下为正，所以数值越大越深）──
    # 地表
    y_surface = [0] * len(x)
    # 土壤层底界（200m）
    y_soil = [200] * len(x)
    # 含水层底界（800m）
    y_aquifer = [800] * len(x)
    # 盖层底界 = 储层顶界（depth - 300m）
    y_caprock_top = [max(200, d - 500)] * len(x)
    # 储层底界（depth + 200m）
    y_reservoir_bottom = [d + 200] * len(x)
    # 基底底界（画到 depth + 800m）
    y_basement = [d + 800] * len(x)

    # 让盖层顶界有些起伏，看起来更像地质图
    import math
    y_caprock_top = [y + 20 * math.sin(xi / 300) + 15 * math.cos(xi / 500)
                     for xi, y in zip(x, y_caprock_top)]

    fig = go.Figure()

    # 1. 土壤层（浅棕色）
    fig.add_trace(go.Scatter(
        x=x + x[::-1],
        y=y_surface + y_soil[::-1],
        fill="toself", fillcolor="rgba(194,178,128,.35)",
        line=dict(width=0), hoverinfo="skip", showlegend=False,
    ))

    # 2. 含水层（浅蓝色）
    fig.add_trace(go.Scatter(
        x=x + x[::-1],
        y=y_soil + y_aquifer[::-1],
        fill="toself", fillcolor="rgba(135,206,235,.30)",
        line=dict(width=0), hoverinfo="skip", showlegend=False,
    ))

    # 3. 盖层（深灰色，密封层）
    fig.add_trace(go.Scatter(
        x=x + x[::-1],
        y=y_aquifer + y_caprock_top[::-1],
        fill="toself", fillcolor="rgba(105,105,105,.40)",
        line=dict(width=0), hoverinfo="skip", showlegend=False,
    ))

    # 4. 储层/注入层（砂岩色，高亮）
    fig.add_trace(go.Scatter(
        x=x + x[::-1],
        y=y_caprock_top + y_reservoir_bottom[::-1],
        fill="toself", fillcolor="rgba(218,165,32,.45)",
        line=dict(width=0), hoverinfo="skip",
        name="CO₂ 注入层",
    ))

    # 5. 基底（深褐色）
    fig.add_trace(go.Scatter(
        x=x + x[::-1],
        y=y_reservoir_bottom + y_basement[::-1],
        fill="toself", fillcolor="rgba(139,90,43,.35)",
        line=dict(width=0), hoverinfo="skip", showlegend=False,
    ))

    # ── 地层分界线（描边）──
    for y_vals, label, dash in (
        (y_surface, "地表", "solid"),
        (y_soil, "土壤层底界", "solid"),
        (y_aquifer, "含水层底界", "solid"),
        (y_caprock_top, "盖层底界（储层顶界）", "dash"),
        (y_reservoir_bottom, "储层底界", "solid"),
    ):
        fig.add_trace(go.Scatter(
            x=x, y=y_vals, mode="lines",
            line=dict(color="rgba(80,80,80,.55)", width=1, dash=dash),
            hoverinfo="skip", showlegend=False,
        ))

    # ── 注入井（竖线）──
    well_x = 1000
    fig.add_trace(go.Scatter(
        x=[well_x, well_x], y=[0, d + 50],
        mode="lines",
        line=dict(color="#1f4e79", width=3),
        hoverinfo="skip",
        name="注入井",
    ))
    # 井口标记
    fig.add_trace(go.Scatter(
        x=[well_x], y=[0],
        mode="markers+text",
        marker=dict(size=10, color="#1f4e79", symbol="triangle-up"),
        text=["井口"], textposition="top center",
        textfont=dict(size=10, color="#1f4e79"),
        hoverinfo="skip", showlegend=False,
    ))

    # ── 工作点（注入层深度处）──
    v_color = {"ok": "#1a7f37", "risky": "#cf222e", "under": "#9a6700"}[verdict]
    v_label = {"ok": "安全", "risky": "超压", "under": "欠压"}[verdict]
    fig.add_trace(go.Scatter(
        x=[well_x], y=[d],
        mode="markers+text",
        marker=dict(size=14, color=v_color, line=dict(color="#fff", width=2),
                    symbol="diamond"),
        text=[f"{v_label}\n{d:.0f}m"], textposition="middle right",
        textfont=dict(size=10, color=v_color),
        hovertemplate=f"深度: {d:.0f} m<br>注入压力: {p:.0f} bar<br> verdict: {v_label}<extra></extra>",
        showlegend=False,
    ))

    # ── 断层（斜线）──
    if has_fault:
        fault_x = list(range(600, 1601, 50))
        # 断层从 (600, 100) 斜向 (1600, d+400)
        fault_y = [100 + (xi - 600) * (d + 300) / 1000 for xi in fault_x]
        fig.add_trace(go.Scatter(
            x=fault_x, y=fault_y, mode="lines",
            line=dict(color="#cf222e", width=2.5, dash="dot"),
            name="推测断层",
        ))
        # 断层标注
        fig.add_annotation(
            x=1300, y=fault_y[14],
            text="推测断层",
            showarrow=True, arrowhead=2, arrowsize=1,
            arrowcolor="#cf222e",
            font=dict(size=10, color="#cf222e"),
            ax=40, ay=-30,
        )

    # ── 图例 ──
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(size=12, color="rgba(218,165,32,.65)"),
        name="CO₂ 注入层",
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(size=12, color="rgba(105,105,105,.55)"),
        name="盖层（密封层）",
    ))

    fig.update_layout(
        height=height,
        margin=dict(l=48, r=48, t=40, b=36),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.08,
                    xanchor="left", x=0, font=dict(size=10)),
        xaxis=dict(title="横向距离 (m)", gridcolor=GRID,
                   zeroline=False, range=[-100, 2100]),
        yaxis=dict(title="深度 (m)", autorange="reversed",
                   gridcolor=GRID, zeroline=False,
                   range=[d + 900, -50]),
    )
    return fig
