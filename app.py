"""
CCUS 智能评估监控台 — 前端（Streamlit）
========================================
前后端完全解耦：本文件只负责布局与 HTTP 调用，
所有 LangGraph / GraphRAG / Ollama 逻辑都在 api_server.py 后端；
所有"算图 / 算字"的纯函数在 ui_components.py（那里能被 CI 直接测）。

依赖：streamlit、plotly、requests
启动：streamlit run app.py --server.port 8501

布局为什么是"左参数 / 右看板"
-----------------------------
上一版是单列 + 顶部输入框 + 一路往下堆输出，读起来像聊天记录。
工业监控台的读法是相反的：**控制量固定在左边，被控量固定在右边**，
眼睛不需要在"我刚才输入了什么"和"现在是什么状态"之间来回找。
所以这里用 ``st.columns`` 切成 1 : 2.35，左侧宽度恒定，右侧永远在同一块地方
刷新，而不是"越聊越长"。

曲线为什么由前端算
------------------
``chart_tools.generate_pressure_curve`` 会生成一张 matplotlib PNG，但那是
"报告配图"：要跑一次后端、等几秒、拿回来一张静态图。
监控台要的是**参数一改曲线立刻动**，所以左侧三个数字一变，本文件直接用
``pressure_model`` 里的同一套公式重画 —— 公式只有一份（后端图表也用它），
所以图和报告里的数字不会对不上。

这也是 CI 里 ``frontend-contract`` 禁止 app.py import matplotlib 的原因：
前端只许画矢量图，不许带绘图库。

Token / 耗时数据从哪来
----------------------
``done`` 事件里带 ``cost_estimation``（后端算好的成本，前端只许展示不许重算，
否则"成本"和"节省"会来自两次计算）与 ``parallel_timing``（并行分支墙钟/串行/
加速比）；每个节点的真实耗时来自 ``node`` 事件的 ``t`` 字段（距请求开始的毫秒数），
前端拿 start/end 两帧相减 —— 不自己打点，避免把 SSE 排队时间算进节点耗时。
"""
import json
import os
import time

import requests
import streamlit as st

from pressure_model import (
    DEFAULT_DEPTH_M,
    DEFAULT_INJECTION_BAR,
    clamp_inputs,
    fracture_bar,
    hydrostatic_bar,
)
from ui_components import (
    NODE_SHORT,
    PIPELINE,
    UQ_VERDICT_LABELS,
    compose_input,
    compute_risk_scores,
    fmt_s,
    fmt_tok,
    pipeline_html,
    pressure_figure,
    risk_radar_figure,
    stratigraphy_figure,
    timing_figure,
    token_figure,
    uq_figure,
)

# 后端地址必须可配置：本地是 127.0.0.1:8000，Docker Compose 里是 http://api:8000。
# 写死会在容器里指向 Streamlit 自己，表现为"后端连不上"，但后端其实活得好好的。
API_BASE = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
API_URL = f"{API_BASE}/api/analyze"
HEALTH_URL = f"{API_BASE}/health"

# 后端要求 Authorization: Bearer <API_TOKEN>。/api/* 全部要带，/health 不用。
#
# 这个 token 只活在前端进程里，**不会**下发到浏览器：浏览器只跟 Streamlit 的
# Python 进程打交道，由它代发请求。所以把前端藏在 nginx 后面时，token 不必
# 暴露给终端用户 —— 这是"反向代理 + 服务端渲染"顺带拿到的安全属性。
API_TOKEN = (os.getenv("API_TOKEN") or "").strip()
AUTH_HEADERS = {"Authorization": f"Bearer {API_TOKEN}"} if API_TOKEN else {}

# 深层物理判定的中文名。前端只做**展示**，不重算 —— 判定结果全部由后端
# 规则引擎给出，这里只把枚举翻成中文，避免"前端自己算一个结论"的老问题。
_GEO_VERDICT_CN = {
    "stable": "未活化",
    "near_critical": "接近活化（余量 <20%）",
    "reactivated": "已活化",
    "tensile_opening": "张性开启",
}
_CAP_VERDICT_CN = {
    "sealed": "封闭",
    "capillary_breakthrough": "毛细管突破",
    "mechanical_failure": "力学失效",
    "no_data": "未判定",
}


def _num(value, nd: int = 2) -> str:
    """把可能是 None / 字符串（"inf"）/ 数字的字段格式化成展示文本。

    深层物理结果里的滑移比可能是字符串 ``"inf"``（有效正应力被压到 0），
    余量也可能是。直接套 ``:.1f`` 会在前端抛 ValueError —— 后端算对了，
    页面却白屏。
    """
    if value is None:
        return "—"
    if isinstance(value, str):
        return value
    try:
        return f"{float(value):.{nd}f}"
    except (TypeError, ValueError):
        return str(value)

st.set_page_config(
    page_title="CCUS 评估监控台",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── 样式 ────────────────────────────────────────────────
# 全部用中性 rgba，不写死白底/黑字：Streamlit 切浅色或深色主题都不会花。
# 关键在"密度"和"等宽数字"——工业界面的质感来自这两点，不是来自配色。
_CSS = """
<style>
.block-container { padding-top: 1.1rem; padding-bottom: 1.6rem; max-width: 1560px; }

/* 顶栏 */
.console-bar {
  display: flex; align-items: center; gap: 18px; flex-wrap: wrap;
  padding: 9px 14px; margin-bottom: 14px;
  border: 1px solid rgba(128,128,128,.28); border-radius: 6px;
  background: rgba(128,128,128,.055);
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 12.5px;
}
.console-bar .brand { font-weight: 700; letter-spacing: .05em; font-size: 14px; }
.console-bar .sep { flex: 1 1 auto; }
.console-bar .kv { color: rgba(128,128,128,.95); }
.console-bar .kv b { font-weight: 600; color: inherit; }

.led { display: inline-block; width: 8px; height: 8px; border-radius: 50%;
       margin-right: 6px; vertical-align: middle; }
.led-ok   { background: #1a7f37; box-shadow: 0 0 0 3px rgba(26,127,55,.16); }
.led-bad  { background: #cf222e; box-shadow: 0 0 0 3px rgba(207,34,46,.16); }
.led-run  { background: #9a6700; box-shadow: 0 0 0 3px rgba(154,103,0,.16); }
.led-idle { background: rgba(128,128,128,.55); }

/* 分区标题：小号、字距拉开、全大写 —— 控制台面板的通用语言 */
.panel-title {
  font-size: 10.5px; letter-spacing: .14em; text-transform: uppercase;
  color: rgba(128,128,128,.95); margin: 2px 0 8px 0;
  border-left: 3px solid rgba(128,128,128,.45); padding-left: 7px;
}

/* 流水线 */
.pipe { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
        font-size: 12px; line-height: 1.85; }
.pipe .row { display: flex; align-items: center; gap: 8px; }
.pipe .name { flex: 1 1 auto; }
.pipe .dur  { color: rgba(128,128,128,.95); font-variant-numeric: tabular-nums; }
.pipe .wait .name { opacity: .45; }

/* 读数条 */
.readout {
  display: flex; gap: 22px; flex-wrap: wrap; margin-top: 4px;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 12px; font-variant-numeric: tabular-nums;
}
.readout .item span { color: rgba(128,128,128,.95); margin-right: 5px; }

/* tabs 收一点高度 */
.stTabs [data-baseweb="tab-list"] { gap: 2px; }
.stTabs [data-baseweb="tab"] { padding: 4px 14px; font-size: 13px; }

/* 主按钮：Streamlit 默认主色是红（#ff4b4b）。控制台上红色只有一个含义 ——
   告警。启动评估是常规操作，改成深蓝，把红色留给真正的风险提示。 */
.stButton button[data-testid="stBaseButton-primary"],
.stButton > button[kind="primary"] {
  background: #1f4e79; border: 1px solid #17405f; color: #fff;
  font-weight: 600; letter-spacing: .04em; border-radius: 4px;
}
.stButton button[data-testid="stBaseButton-primary"]:hover,
.stButton > button[kind="primary"]:hover {
  background: #17405f; border-color: #123350; color: #fff;
}
</style>
"""

st.markdown(_CSS, unsafe_allow_html=True)


# ─── 后端探活 ────────────────────────────────────────────
@st.cache_data(ttl=5, show_spinner=False)
def backend_alive(timeout: float = 0.6) -> bool:
    """缓存 5 秒：每次拖滑块都发一次请求会把左侧交互拖慢。"""
    try:
        # /health 目前不鉴权，但仍带上 header：将来若把健康检查也收紧，
        # 这里不会突然变成"后端离线"的假警报。
        return requests.get(HEALTH_URL, timeout=timeout, headers=AUTH_HEADERS).status_code == 200
    except Exception:
        return False


def iter_sse(user_input: str, connect_timeout=10, read_timeout=600):
    """
    连接后端 SSE，逐个 yield 解析后的事件字典。

    超时拆成 connect/read 两段：建连要快（后端没起时立刻报错），
    读可以给足时间（一次完整评估含反思循环可能要 2 分钟）。
    """
    with requests.post(
        API_URL,
        json={"user_input": user_input},
        stream=True,
        timeout=(connect_timeout, read_timeout),
        headers={"Accept": "text/event-stream", **AUTH_HEADERS},
    ) as resp:
        if resp.status_code in (401, 403):
            # 鉴权失败是最常见的部署事故：前后端 token 不一致，或者只改了一边
            # 没重建容器。默认报错"HTTP 401"会让人去翻后端日志，其实问题在前端
            # 环境变量里，所以这里直接把原因写出来。
            yield {
                "type": "error",
                "message": (
                    f"后端鉴权失败（HTTP {resp.status_code}）。"
                    "请检查前端与后端的 API_TOKEN 是否一致："
                    "前端读环境变量 API_TOKEN，后端读同一个名字，改完需要重启容器。"
                ),
            }
            return
        if resp.status_code != 200:
            yield {"type": "error", "message": f"后端返回 HTTP {resp.status_code}"}
            return
        for raw in resp.iter_lines(decode_unicode=True):
            if not raw:
                continue
            if raw.startswith(":"):       # SSE 心跳注释，跳过
                continue
            if not raw.startswith("data:"):
                continue
            payload = raw[5:].strip()
            if not payload:
                continue
            try:
                yield json.loads(payload)
            except json.JSONDecodeError:
                # 宁可跳过一条坏帧，也不要整条流崩掉
                continue


# ══════════════════════════════════════════════════════════
# 页面骨架
# ══════════════════════════════════════════════════════════
alive = backend_alive()
meta = st.session_state.setdefault("meta", {})
done = meta.get("done") or {}
last_run = st.session_state.get("last_run") or {}

st.markdown(
    f"""
<div class="console-bar">
  <span class="brand">🏭 CCUS 评估监控台</span>
  <span class="kv"><span class="led {'led-ok' if alive else 'led-bad'}"></span>
    {'后端在线' if alive else '后端离线'}</span>
  <span class="kv">ENDPOINT <b>{API_BASE}</b></span>
  <span class="sep"></span>
  <span class="kv">RUN <b>{last_run.get('id') or '—'}</b></span>
  <span class="kv">耗时 <b>{fmt_s(last_run.get('elapsed'))}</b></span>
  <span class="kv">评分 <b>{(done.get('score') if done.get('score') is not None else '—')}/10</b></span>
</div>
""",
    unsafe_allow_html=True,
)

left, right = st.columns([1, 2.35], gap="large")

# ─── 左：控制区 ──────────────────────────────────────────
with left:
    st.markdown('<div class="panel-title">任务参数</div>', unsafe_allow_html=True)
    user_input = st.text_area(
        "项目描述",
        value="",
        height=104,
        label_visibility="collapsed",
        placeholder="例如：某CCUS项目埋深2500m，注入压力250bar，温度80℃，"
                    "存在隐伏碳酸盐岩储层断层，请评估风险。",
    )

    c1, c2 = st.columns(2)
    depth = c1.number_input("储层埋深 (m)", 200.0, 8000.0,
                            value=DEFAULT_DEPTH_M, step=50.0,
                            format="%.0f", key="depth")
    p_inj = c2.number_input("注入压力 (bar)", 20.0, 1200.0,
                            value=DEFAULT_INJECTION_BAR, step=5.0,
                            format="%.0f", key="p_inj")
    c3, c4 = st.columns(2)
    temp = c3.number_input("地层温度 (℃)", 20.0, 200.0, value=80.0,
                           step=5.0, format="%.0f", key="temp")
    scenario = c4.selectbox("工况", ["注入", "封存监测", "停注"], key="scenario")

    submitted = st.button("▶  启动评估", type="primary", width="stretch")
    if submitted and not user_input.strip():
        st.warning("请输入项目描述后再启动评估。")

    st.markdown('<div class="panel-title">流水线状态</div>', unsafe_allow_html=True)
    pipe_slot = st.empty()
    live_slot = st.empty()

    st.markdown('<div class="panel-title">阈值参考</div>', unsafe_allow_html=True)
    _d, _p = clamp_inputs(depth, p_inj)
    _ph, _pf = float(hydrostatic_bar(_d)), float(fracture_bar(_d))
    st.markdown(
        f'<div class="readout">'
        f'<div class="item"><span>P_h</span>{_ph:.0f} bar</div>'
        f'<div class="item"><span>P_frac</span>{_pf:.0f} bar</div>'
        f'<div class="item"><span>裕度</span>{_pf - _p:+.0f} bar</div>'
        f"</div>",
        unsafe_allow_html=True,
    )
    st.caption("左侧参数会作为结构化条件附加到描述后，由后端提取节点优先采用。")

# ─── 右：展示区 ──────────────────────────────────────────
with right:
    # 用 container 不用 empty：empty() 只承载一个元素，连续写三次
    # （标题 / 图 / caption）会被后一次覆盖前一次 —— 表现为"有 caption 时图就没了"。
    chart_slot = st.container()
    tab_report, tab_board = st.tabs(["📋  评估报告", "📊  数据看板"])
    with tab_report:
        report_slot = st.container()
    with tab_board:
        board_slot = st.container()

# 曲线：先按左侧参数画（不依赖任何一次运行）
with chart_slot:
    st.markdown('<div class="panel-title">地层压力 — 深度 实时曲线</div>',
                unsafe_allow_html=True)
    st.plotly_chart(
        pressure_figure(depth, p_inj),
        width="stretch",
        config={"displayModeBar": False},
        key="live_curve",
    )
    ci = done.get("chart") or {}
    if ci.get("depth_m") and ci.get("injection_pressure_bar"):
        same = (abs(ci["depth_m"] - float(depth)) < 1
                and abs(ci["injection_pressure_bar"] - float(p_inj)) < 1)
        st.caption(
            f"最近一次评估使用：埋深 {ci['depth_m']:.0f} m · 注入压力 "
            f"{ci['injection_pressure_bar']:.0f} bar · "
            f"P_frac {ci.get('fracture_bar_at_depth', 0):.0f} bar"
            + ("" if same else "（与左侧当前输入不同）")
        )

    # ─── 地层剖面示意图（随参数动态更新）──────────────────
    st.markdown('<div class="panel-title">地层剖面示意图</div>',
                unsafe_allow_html=True)
    # has_fault 的判定：从报告文本或地质状态中推断是否有断层
    report_text = (done.get("report") or "").lower()
    has_fault = "断层" in report_text or "fault" in report_text
    st.plotly_chart(
        stratigraphy_figure(depth, p_inj, temp, has_fault=has_fault),
        width="stretch",
        config={"displayModeBar": False},
        key="stratigraphy",
    )
    st.caption(
        "简化地质剖面：盖层（深灰）密封 CO₂，注入层（砂岩色）储存 CO₂。"
        "红色虚线为推测断层。工作点按当前 verdict 着色。"
    )


# ─── 一次评估 ────────────────────────────────────────────
def run_evaluation(payload_input: str) -> None:
    """跑一次评估：边流边更新左侧状态灯与报告区，结束后把结果落进 session_state。"""
    states: dict = {}
    timings: dict = {}
    t0 = time.time()
    run_id = time.strftime("%H%M%S")

    pipe_slot.markdown(pipeline_html(states, timings, running=True),
                       unsafe_allow_html=True)
    live_slot.markdown(
        '<div class="readout"><div class="item">'
        f'<span class="led led-run"></span>运行中 · RUN {run_id}</div></div>',
        unsafe_allow_html=True,
    )

    local: dict = {}

    def token_gen():
        try:
            for ev in iter_sse(payload_input):
                etype = ev.get("type")

                if etype == "token":
                    yield ev.get("text", "")

                elif etype == "node":
                    node = ev.get("node")
                    status = ev.get("status")
                    states[node] = status
                    t = timings.setdefault(node, {})
                    if ev.get("t") is not None:
                        t[status] = ev["t"]
                    pipe_slot.markdown(
                        pipeline_html(states, timings, running=True),
                        unsafe_allow_html=True,
                    )
                    live_slot.markdown(
                        '<div class="readout"><div class="item">'
                        f'<span class="led led-run"></span>运行中 · '
                        f'{time.time() - t0:.1f}s · '
                        f'{NODE_SHORT.get(node, node)}</div></div>',
                        unsafe_allow_html=True,
                    )

                elif etype == "chart":
                    local["chart"] = ev.get("chart") or {}

                elif etype == "uq":
                    # UQ 在终稿之前就算完，先存下来；done 事件里还会再带一份
                    # 完整版（含 samples），两者合并而不是覆盖。
                    local["uq"] = {**local.get("uq", {}), **(ev.get("uq") or {})}

                elif etype == "done":
                    local["done"] = ev

                elif etype == "error":
                    local["error"] = ev.get("message", "未知错误")

        except requests.exceptions.ReadTimeout:
            local["error"] = "读取超时：完整评估可能超过后端处理时间。"
        except requests.exceptions.ConnectionError:
            local["error"] = ("请求后端失败，请检查 api_server.py 是否在 8000 端口正常运行。")
        except Exception as e:
            local["error"] = f"请求后端失败：{e}"

    with report_slot:
        streamed = st.write_stream(token_gen())

    # 落库：后续任何一次 rerun（拖滑块、切标签页）都要能重新显示这一次的结果。
    # 报告正文必须一起存 —— run 结束后会 st.rerun()，那次渲染里流式内容已经销毁，
    # 不存下来报告区就是空白（这个坑第一次跑就会踩到）。
    elapsed = time.time() - t0
    st.session_state["meta"] = {
        "done": local.get("done"),
        "error": local.get("error"),
        "chart": local.get("chart"),
        "uq": local.get("uq"),
        "report": streamed or (local.get("done") or {}).get("report", ""),
        "states": states,
        "timings": timings,
        "elapsed": elapsed,
    }
    st.session_state["last_run"] = {"id": run_id, "elapsed": elapsed}


def render_report(meta: dict, depth: float, p_inj: float, temp: float) -> None:
    """报告页：正文 + 必须先被看见的告警 + 成本文案。"""
    err = meta.get("error")
    done_ = meta.get("done") or {}

    if err:
        st.error(f"{err}\n\n请确认后端：`python api_server.py`（8000 端口）")
        return
    if not done_:
        st.markdown('<div class="panel-title">等待任务</div>'
                    "左侧填写参数后点击 **启动评估**，报告将在此处流式输出。",
                    unsafe_allow_html=True)
        return

    # 质量闸门：失败态必须顶在最前面，且措辞要能让人停止使用这份报告。
    # 以前"重试用尽"会走到这里并被渲染成 `st.success("评估完成")` ——
    # 3 分的报告和 9 分的报告长得一模一样。
    gate = done_.get("quality_gate") or {}
    gate_failed = gate.get("status") == "failed"
    if gate_failed:
        st.error(
            "⛔ 质量闸门判定失败：本次报告**不得作为工程依据**。\n\n"
            f"- 状态：`{gate.get('status')}`\n"
            f"- 原因：{gate.get('reason') or '未说明'}\n"
            + ("- 正文已隔离（物理一致性校验未通过）\n"
               if gate.get("quarantined") else "")
        )
    else:
        st.success(f"✅ 评估完成（{done_.get('elapsed_s', '-')} 秒）")

    # 物理一致性校验：一票否决的结论必须顶在最前面，不能被折叠进详情
    physics = done_.get("physics_check") or {}
    if physics.get("blocking"):
        st.error(
            "⚠️ 物理一致性校验未通过：报告与地层压力约束矛盾，"
            "已自动重写仍未消除。请勿直接采用该结论。\n\n"
            + str(physics.get("summary") or "")
        )
    for v in physics.get("violations") or []:
        if v.get("severity") == "warning":
            st.warning(f"物理校验提示：{v.get('message')}")

    # 规则版本与生效参数：报告里每个"安全/不安全"的结论都应该能追到这两个值。
    # 以前它们硬编码在 utils/physics_rules.py 里，读者只能看到结论、看不到依据。
    rules_meta = physics.get("rules") or {}
    if physics.get("rules_version") and rules_meta:
        st.caption(
            f"物理规则版本 **{physics['rules_version']}** · "
            + " · ".join(
                f"{k} = {v.get('value')}{v.get('unit') or ''}"
                for k, v in rules_meta.items()
            )
        )
        with st.expander("规则参数出处（点开可核对）"):
            st.dataframe(
                [
                    {
                        "参数": k,
                        "取值": v.get("value"),
                        "单位": v.get("unit") or "",
                        "出处": v.get("source") or "",
                    }
                    for k, v in rules_meta.items()
                ],
                hide_index=True,
            )

    # ─── 深层物理：断层活化 / 盖层封闭性 / 两相流 ───
    # 这三项与上面两条压力线的结论**不是一回事**：断层在 1.5× 静水以下就可能
    # 活化，盖层毛细管突破与注入压力无关。所以它们单独成块，不塞进"物理校验
    # 提示"里 —— 混在一起会让读者以为"没超破裂压力"就等于"没有这些风险"。
    deep = physics.get("deep") or {}
    if deep.get("status") == "ok":
        geo = deep.get("geomechanics") or {}
        cap = deep.get("caprock") or {}
        rel = deep.get("relperm") or {}
        rows = []
        if geo:
            rows.append({
                "判据": "断层活化",
                "结论": _GEO_VERDICT_CN.get(geo.get("verdict"), geo.get("verdict")),
                "关键数值": (
                    f"滑移比 {geo.get('slip_ratio')}（≥1 即滑动）· "
                    f"临界孔隙压力 {geo.get('pp_crit_bar')} bar · "
                    f"余量 {geo.get('margin_bar')} bar"
                ),
            })
        else:
            rows.append({
                "判据": "断层活化",
                "结论": "未判定",
                "关键数值": "输入未提及断层（不用假设应力剖面做阻断）",
            })
        if cap.get("entry_pressure_bar") is not None:
            rows.append({
                "判据": "盖层封闭性",
                "结论": _CAP_VERDICT_CN.get(cap.get("verdict"), cap.get("verdict") or "—"),
                "关键数值": (
                    f"突破压力 {_num(cap.get('entry_pressure_bar'), 0)} bar · "
                    f"可封柱高 {_num(cap.get('max_column_height_m'), 0)} m"
                    + (f" · 本次柱高 {_num(cap.get('column_height_m'), 0)} m"
                       if cap.get("column_height_m") is not None else "")
                ),
            })
        if rel:
            # 字段可能缺失（旧缓存里的 physics_check 没有这一块），所以用 .get + 默认值
            # 而不是直接下标：一个 KeyError 会让整页白屏，而这里只是少一行表。
            rows.append({
                "判据": "两相流",
                "结论": f"流度比 M = {_num(rel.get('mobility_ratio'), 1)}",
                "关键数值": (
                    f"前缘 S_CO₂ {_num(rel.get('sg_front'), 2)} · "
                    f"波及区平均 {_num(rel.get('co2_saturation'), 2)}"
                ),
            })
        st.caption(f"深层物理判定 · {deep.get('version') or ''}")
        st.dataframe(rows, hide_index=True)
        with st.expander("深层判定的前提与阻断策略（结论只在这些假设下成立）"):
            for item in deep.get("assumptions") or []:
                st.markdown(f"- 假设：{item}")
            for item in deep.get("policy") or []:
                st.markdown(f"- 策略：{item}")
            for item in cap.get("limitations") or []:
                st.markdown(f"- 局限：{item}")

    # ─── 时间维度冲突：与物理告警同级，必须顶在正文之前 ───
    # 这类问题最容易被读者忽略：报告里出现一个"最新"的压力值，看起来毫无异常，
    # 而实际上它是从两条互相矛盾的证据里挑出来的。不显式说明，等于隐瞒。
    temporal = done_.get("temporal") or {}
    conflicts = temporal.get("conflicts") or []
    tstats = temporal.get("stats") or {}
    if conflicts:
        st.warning(
            "⏱ 检测到跨时间冲突数据：同一对象在不同时间点的数值不一致。"
            "报告已按**最新时间戳**取值，旧值仅作为演化对比保留。"
        )
        st.dataframe(
            [
                {
                    "对象": c.get("entity"),
                    "参数": c.get("metric_label"),
                    "旧数据": f"{c['older']['display']}（{c['older']['timestamp']}）",
                    "最新数据": f"{c['newer']['display']}（{c['newer']['timestamp']}）",
                    "变化": f"{c.get('delta_pct', 0):+.1f}%",
                    "间隔": c.get("span"),
                }
                for c in conflicts
            ],
            width="stretch",
            hide_index=True,
        )
    elif tstats.get("claims"):
        st.caption(
            f"⏱ 时间轴：识别到 {tstats['claims']} 条带时间戳的断言"
            f"（{len(tstats.get('entities') or [])} 个对象），未发现冲突。"
        )
    else:
        # 让时间维度这条能力在界面上"看得见"——即使本次检索没有任何带时间戳的
        # 断言，也给出一行状态，而不是整个能力静默消失。当前语料 70 个片段里
        # 只有 1 处含年份字样（还是 NeurIPS 2023 的引用），所以生产查询大概率
        # 走这条分支；要看到冲突告警，请往知识库里补带日期的监测记录。
        st.caption("⏱ 时间维度：本次检索未识别到带时间戳的断言（暂无历史冲突可比对）。")

    # 报告正文：rerun 后靠这里恢复（流式那次的内容已经随上一次渲染销毁）。
    # 不要用 <div> 包裹再分两次 markdown 输出：Streamlit 每次 markdown 是独立的
    # DOM 节点，未闭合的 div 会被浏览器就地补上闭合标签，包不住后面的内容。
    body = meta.get("report") or done_.get("report") or ""
    if body:
        st.markdown(body)

    # PINN 数值：报告正文里那句话是模型转述，可能被漏掉；
    # 结构化字段存在的意义就是"不靠模型复述也拿得到数值"。
    pinn = done_.get("pinn") or {}
    if pinn.get("text"):
        st.caption(f"🧮 {pinn['text']}")

    # 概率性结论：报告里应当已经写了，这里再给一份"未经模型转述"的原文，
    # 保证读者看到的是工具算出来的区间，而不是被改写过的措辞。
    uq = done_.get("uq") or meta.get("uq") or {}
    density_uq = uq.get("density") or {}
    if density_uq.get("text"):
        st.caption(f"🎲 {density_uq['text']}")

    # ─── 五维风险雷达图 ───────────────────────────────────
    # 基于现有评估结果和参数，计算五个维度的风险分数。
    # 不需要后端新增字段，纯前端计算，rerun 时随参数即时更新。
    scores = compute_risk_scores(done_, depth, p_inj, temp)
    st.markdown('<div class="panel-title">五维风险评估</div>',
                unsafe_allow_html=True)
    r1, r2 = st.columns([2, 1])
    with r1:
        st.plotly_chart(
            risk_radar_figure(scores),
            width="stretch",
            config={"displayModeBar": False},
            key="risk_radar",
        )
    with r2:
        st.markdown("**维度说明**")
        for dim, score in scores.items():
            color = "#cf222e" if score >= 70 else "#9a6700" if score >= 40 else "#1a7f37"
            st.markdown(
                f"<span style='color:{color};font-weight:600'>{dim}</span> "
                f"<span style='font-family:monospace'>{score:.0f}</span>",
                unsafe_allow_html=True,
            )
        max_dim = max(scores, key=scores.get)
        st.caption(
            f"当前最高风险维度：**{max_dim}**（{scores[max_dim]:.0f}/100）。"
            f"分数基于物理校验结果、UQ 概率和注入参数综合计算。"
        )

    review = done_.get("human_review") or {}
    if review.get("auto"):
        st.caption(
            "🔓 本次为全自动模式，结论未经人工确认。"
            "需要人工把关请调用 `POST /api/analyze/hitl` + `POST /api/approve`。"
        )

    # 成本文案由后端算好（cost_estimation.caption）：成本与"节省"必须来自
    # 同一次计算，前端各算一遍很容易出现量级错配。
    cost = done_.get("cost_estimation") or {}
    if cost.get("caption"):
        st.caption(cost["caption"])

    with st.expander("🔍 底层详情"):
        st.write("**提取的参数：**")
        st.code(done_.get("extracted_params", ""), language="markdown")
        if done_.get("model_routing"):
            st.write("**模型路由（实际使用）：**")
            st.json(done_["model_routing"])
        if physics:
            st.write("**物理一致性校验（规则引擎）：**")
            st.code(
                f"{physics.get('status')} · {physics.get('verdict')}\n"
                f"{physics.get('summary')}\n"
                + "\n".join(
                    f"[{c.get('ok')}] {c.get('code')}: {c.get('detail')}"
                    for c in physics.get("checks") or []
                ),
                language="text",
            )
        if done_.get("history_entry"):
            st.write("**已写入 project_history.md：**")
            st.code(done_["history_entry"], language="markdown")


def render_board(meta: dict) -> None:
    """看板页：耗时 / Token / 成本 / 路由。数字全部来自后端，前端只做排版。"""
    done_ = meta.get("done") or {}
    if not done_:
        st.markdown('<div class="panel-title">等待数据</div>'
                    "完成一次评估后，这里显示全链路耗时与 Token 花费。",
                    unsafe_allow_html=True)
        return

    cost = done_.get("cost_estimation") or {}
    breakdown = cost.get("breakdown") or []
    tin = sum(int(b.get("input_tokens") or 0) for b in breakdown)
    tout = sum(int(b.get("output_tokens") or 0) for b in breakdown)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("全链路耗时", fmt_s(done_.get("elapsed_s")))
    k2.metric("Token 总量", fmt_tok(tin + tout),
              delta=f"入 {fmt_tok(tin)} / 出 {fmt_tok(tout)}", delta_color="off")
    k3.metric("本次成本", f"¥{cost.get('total_cny', 0):.6f}",
              delta=f"基线 ¥{cost.get('baseline_cny', 0):.6f}", delta_color="off")
    k4.metric("节省比例", f"{(cost.get('saved_pct') or 0) * 100:.0f}%",
              delta=f"{len(breakdown)} 个阶段", delta_color="off")

    # ─── 不确定性量化（Monte Carlo）────────────────────────
    # 放在看板最上面：这是本次评估"结论有多可信"的唯一量化来源，
    # 比 token 花了几厘钱重要得多。
    uq = done_.get("uq") or meta.get("uq") or {}
    density_uq = uq.get("density") or {}
    injection_uq = uq.get("injection") or {}
    if density_uq or injection_uq:
        st.markdown('<div class="panel-title">不确定性量化（蒙特卡洛模拟）</div>',
                    unsafe_allow_html=True)

        if density_uq:
            u1, u2, u3, u4 = st.columns(4)
            u1.metric("密度均值", f"{density_uq.get('mean', 0):.2f} kg/m³",
                      delta=f"标称 {density_uq.get('nominal', 0):.2f}",
                      delta_color="off")
            u2.metric("90% 区间下界", f"{density_uq.get('p5', 0):.2f} kg/m³")
            u3.metric("90% 区间上界", f"{density_uq.get('p95', 0):.2f} kg/m³")
            u4.metric("变异系数 CV",
                      f"{(density_uq.get('cv') or 0) * 100:.1f}%",
                      delta=f"{density_uq.get('n', 0)} 次模拟",
                      delta_color="off")

            if density_uq.get("samples"):
                st.plotly_chart(
                    uq_figure(
                        density_uq["samples"],
                        p5=density_uq.get("p5"),
                        p95=density_uq.get("p95"),
                        mean=density_uq.get("mean"),
                        unit="kg/m³",
                    ),
                    width="stretch",
                    config={"displayModeBar": False},
                    key="uq_hist",
                )
            if density_uq.get("text"):
                st.caption(f"🎲 {density_uq['text']}")

        if injection_uq.get("probabilities"):
            st.markdown('<div class="panel-title">注入结局概率分布</div>',
                        unsafe_allow_html=True)
            probs = injection_uq["probabilities"]
            for key, prob in sorted(probs.items(), key=lambda kv: -kv[1]):
                st.progress(
                    min(max(float(prob), 0.0), 1.0),
                    text=f"{UQ_VERDICT_LABELS.get(key, key)} — {prob * 100:.0f}%",
                )
            st.caption(
                f"标称判定：{injection_uq.get('verdict_label', '—')}；"
                f"深度 {injection_uq.get('depth_m', 0):.0f} m / 注入压力 "
                f"{injection_uq.get('injection_pressure_bar', 0):.0f} bar；"
                f"深度与压力各 ±{(injection_uq.get('rel_sigma') or 0) * 100:.0f}% 扰动。"
            )

    st.markdown('<div class="panel-title">全链路耗时（节点墙钟）</div>',
                unsafe_allow_html=True)
    rows = []
    for node, _label in PIPELINE:
        t = (meta.get("timings") or {}).get(node) or {}
        if t.get("start") is not None and t.get("end") is not None:
            rows.append({"label": NODE_SHORT.get(node, node),
                         "ms": t["end"] - t["start"]})
    if rows:
        st.plotly_chart(timing_figure(rows), width="stretch",
                        config={"displayModeBar": False}, key="timing_bar")
        st.caption(
            "耗时来自后端 `node` 事件的毫秒时间戳（start/end 相减），"
            "不含 SSE 排队与网络延迟。地质与井筒两条分支是并发的，"
            "所以两段条会重叠 —— 重叠部分就是并行省下的时间。"
        )
    else:
        st.caption("本次未收到节点级时间戳（旧版后端不返回 `t` 字段）。")

    pt = done_.get("parallel_timing") or {}
    if pt.get("wall_s") is not None:
        st.markdown('<div class="panel-title">并行分支</div>', unsafe_allow_html=True)
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("地质分支", fmt_s(pt.get("geology_s")))
        p2.metric("井筒分支", fmt_s(pt.get("wellbore_s")))
        p3.metric("并行墙钟", fmt_s(pt.get("wall_s")))
        p4.metric("加速比", f"{pt.get('speedup', 0):.2f}×",
                  delta=f"串行需 {fmt_s(pt.get('serial_s'))}", delta_color="off")

    if breakdown:
        st.markdown('<div class="panel-title">Token 花费（按阶段）</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(token_figure(breakdown), width="stretch",
                        config={"displayModeBar": False}, key="token_bar")
        st.dataframe(
            [
                {
                    "阶段": b.get("stage"),
                    "档位": b.get("tier"),
                    "模型": b.get("model"),
                    "输入": b.get("input_tokens"),
                    "输出": b.get("output_tokens"),
                    "金额 ¥": b.get("cny"),
                }
                for b in breakdown
            ],
            width="stretch",
            hide_index=True,
        )
        if cost.get("baseline_note"):
            st.caption(cost["baseline_note"])
        if cost.get("estimated"):
            st.caption("⚠️ 部分阶段未返回 usage，token 数由本地估算。")

    with st.expander("🔍 路由与压缩明细"):
        complexity = done_.get("complexity") or {}
        if complexity:
            st.write("**动态路由判定：**")
            st.code(f"{complexity.get('label', '')} {complexity.get('reason', '')}",
                    language="text")
        pruning = done_.get("context_pruning") or {}
        if pruning:
            st.write("**上下文压缩：**")
            st.code(
                f"{pruning.get('status')}  "
                f"{pruning.get('before_chars')} → {pruning.get('after_chars')} 字"
                f"（省 {pruning.get('saved_chars')} 字）"
                + (f"\n{pruning['reason']}" if pruning.get("reason") else ""),
                language="text",
            )
        retrieval = done_.get("retrieval_quality") or {}
        if retrieval:
            st.write("**英文检索改写：**")
            st.code(
                f"地质：{retrieval.get('geology_query') or '—'}\n"
                f"井筒：{retrieval.get('wellbore_query') or '—'}",
                language="text",
            )
            grade = retrieval.get("grade") or {}
            st.write("**Self-RAG 证据评估：**")
            st.json({
                "status": grade.get("status"),
                "strategy": retrieval.get("strategy"),
                "retry_count": retrieval.get("retry_count"),
                "minimum_score": grade.get("minimum_score"),
                "geology": grade.get("geology"),
                "wellbore": grade.get("wellbore"),
                "web_evidence": retrieval.get("web_evidence_grade"),
                "geology_status": retrieval.get("geology_status"),
                "wellbore_status": retrieval.get("wellbore_status"),
                "branch_merge": retrieval.get("branch_merge"),
            })
        if done_.get("model_routing"):
            st.write("**模型路由：**")
            st.json(done_["model_routing"])


# ─── 触发 ────────────────────────────────────────────────
# 描述为空时只提示、不启动；started_run 决定后面是"跑一次"还是"从状态恢复"。
started_run = submitted and bool(user_input.strip())

if started_run:
    payload_input = compose_input(user_input, depth, p_inj, temp, scenario)
    with report_slot:
        st.markdown('<div class="panel-title">风险评估报告</div>',
                    unsafe_allow_html=True)
    run_evaluation(payload_input)
    # 必须 rerun：看板与顶栏在本次脚本执行时读到的还是旧 meta，
    # 不重跑一遍它们会是空的；重跑后统一从 session_state 恢复。
    st.rerun()

# ─── 常驻渲染（首屏与 rerun 后都走这里）───
with report_slot:
    if not started_run:
        render_report(meta, depth, p_inj, temp)
with board_slot:
    render_board(meta)

if not started_run:
    pipe_slot.markdown(
        pipeline_html(meta.get("states") or {}, meta.get("timings") or {},
                      running=False),
        unsafe_allow_html=True,
    )
