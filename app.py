"""
CCUS 多智能体风险评估系统 — 前端（Streamlit）
=============================================
前后端完全解耦：本文件只负责 UI 与 HTTP 调用，
所有 LangGraph / GraphRAG / Ollama 逻辑都在 api_server.py 后端。

依赖：streamlit、requests
启动：streamlit run app.py --server.port 8505

流式实现说明：
后端是 SSE，本文件用 `requests.post(stream=True)` + `iter_lines` 自己解析，
再用 `st.write_stream` 逐 token 渲染 —— 效果同 ChatGPT。
之所以不用 `st.write_stream(requests.get(...).iter_lines())` 一把梭：
SSE 里除了文字还有 node / chart / done / error 四类事件，
必须自己分流，否则 JSON 会直接被当正文吐到屏幕上。
"""
import json
import os

import requests
import streamlit as st

# 后端地址必须可配置：本地是 127.0.0.1:8000，Docker Compose 里是 http://api:8000。
# 写死会在容器里指向 Streamlit 自己，表现为"后端连不上"，但后端其实活得好好的。
API_BASE = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
API_URL = f"{API_BASE}/api/analyze"
HEALTH_URL = f"{API_BASE}/health"

st.set_page_config(page_title="CCUS 多智能体风险评估系统", layout="wide")
st.title("🏭 CCUS 多智能体风险评估系统")
st.caption("LangGraph 多智能体 + GraphRAG 知识图谱 · 流式输出 · 前后端解耦架构")

# ─── 输入区 ────────────────────────────────────────────
user_input = st.text_area(
    "请输入 CCUS 项目描述",
    placeholder="例如：某CCUS项目埋深2500m，注入压力250bar，温度80℃，"
                "存在隐伏碳酸盐岩储层断层，请评估风险。",
    height=120,
)


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
        headers={"Accept": "text/event-stream"},
    ) as resp:
        if resp.status_code != 200:
            yield {"type": "error",
                   "message": f"后端返回 HTTP {resp.status_code}"}
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


# ─── 提交 ──────────────────────────────────────────────
if st.button("启动评估", type="primary"):
    if not user_input.strip():
        st.warning("请输入项目描述后再启动评估。")
    else:
        meta: dict = {}
        progress = st.empty()
        progress.info("正在连接后端…")

        def token_gen():
            """把 SSE 事件流翻译成纯文本流；控制类事件顺手存进 meta。"""
            try:
                for ev in iter_sse(user_input.strip()):
                    etype = ev.get("type")

                    if etype == "token":
                        yield ev.get("text", "")

                    elif etype == "node":
                        if ev.get("status") == "start":
                            progress.info(f"⏳ {ev.get('label', ev.get('node'))}…")
                        else:
                            meta.setdefault("finished_nodes", []).append(
                                ev.get("node")
                            )

                    elif etype == "chart":
                        meta["chart"] = ev.get("chart") or {}
                        progress.info("📈 已生成地层压力曲线，继续撰写报告…")

                    elif etype == "done":
                        meta["done"] = ev
                        progress.empty()

                    elif etype == "error":
                        meta["error"] = ev.get("message", "未知错误")
                        progress.empty()

            except requests.exceptions.ReadTimeout:
                meta["error"] = "读取超时：完整评估可能超过后端处理时间。"
            except requests.exceptions.ConnectionError:
                meta["error"] = (
                    "请求后端失败，请检查 api_server.py 是否在 8000 端口正常运行。"
                )
            except Exception as e:
                meta["error"] = f"请求后端失败：{e}"

        try:
            st.markdown("### 📋 风险评估报告")
            st.write_stream(token_gen())
        except Exception as e:
            st.error(f"渲染中断：{e}")

        # ─── 错误 ───
        if meta.get("error"):
            st.error(
                f"{meta['error']}\n\n请确认后端：`python api_server.py`（8000 端口）"
            )

        # ─── 图表 ───
        chart = (meta.get("done") or {}).get("chart") or meta.get("chart") or {}
        if chart.get("path"):
            st.markdown("### 📈 地层压力—深度曲线")
            try:
                st.image(chart["path"], use_container_width=True)
            except Exception as e:
                st.warning(f"图片加载失败（{e}），文件路径：{chart['path']}")
            if chart.get("summary"):
                st.caption(chart["summary"])

        # ─── 底层详情 ───
        done = meta.get("done")
        if done:
            st.success(f"✅ 评估完成（{done.get('elapsed_s', '-')} 秒）")

            # ─── 物理一致性校验（规则引擎，一票否决）───
            # 放在最显眼的位置：如果重写次数用尽仍有物理矛盾，这份报告
            # 不能被当作工程依据，必须让用户一眼看到。
            physics = done.get("physics_check") or {}
            if physics.get("blocking"):
                st.error(
                    "⚠️ 物理一致性校验未通过：报告与地层压力约束矛盾，"
                    "已自动重写仍未消除。请勿直接采用该结论。\n\n"
                    + str(physics.get("summary") or "")
                )
            for v in physics.get("violations") or []:
                if v.get("severity") == "warning":
                    st.warning(f"物理校验提示：{v.get('message')}")

            # ─── 人工审核状态 ───
            review = done.get("human_review") or {}
            if review.get("auto"):
                st.caption(
                    "🔓 本次为全自动模式，结论未经人工确认。"
                    "需要人工把关请调用 `POST /api/analyze/hitl` + `POST /api/approve`。"
                )

            # ─── 成本仪表盘 ───
            # caption 由后端算好（cost_estimation.caption）：成本与"节省"必须
            # 来自同一次计算，前端各算一遍很容易出现量级错配。
            cost = done.get("cost_estimation") or {}
            if cost.get("caption"):
                st.caption(cost["caption"])

            with st.expander("🔍 查看底层详情"):
                score = done.get("score")
                c1, c2, c3 = st.columns(3)
                c1.metric("评审分数", f"{score if score is not None else '-'} / 10")
                c2.metric("模型路由数", len(done.get("model_routing") or {}))
                c3.metric("报告字数", len(done.get("report") or ""))

                st.write("**提取的参数：**")
                st.code(done.get("extracted_params", ""), language="markdown")

                if done.get("model_routing"):
                    st.write("**模型路由（实际使用）：**")
                    st.json(done["model_routing"])

                # ─── Token 经济学明细 ───
                if cost:
                    st.write("**成本构成（人民币）：**")
                    k1, k2, k3 = st.columns(3)
                    k1.metric("本次成本", f"¥{cost.get('total_cny', 0):.6f}")
                    k2.metric("全云端基线", f"¥{cost.get('baseline_cny', 0):.6f}")
                    k3.metric("节省比例", f"{(cost.get('saved_pct') or 0) * 100:.0f}%")
                    if cost.get("breakdown"):
                        st.dataframe(
                            [
                                {
                                    "阶段": b.get("stage"),
                                    "档位": b.get("tier"),
                                    "模型": b.get("model"),
                                    "输入 tokens": b.get("input_tokens"),
                                    "输出 tokens": b.get("output_tokens"),
                                    "金额 ¥": b.get("cny"),
                                }
                                for b in cost["breakdown"]
                            ],
                            use_container_width=True,
                        )
                    if cost.get("baseline_note"):
                        st.caption(cost["baseline_note"])
                    if cost.get("estimated"):
                        st.caption("⚠️ 部分阶段未返回 usage，token 数由本地估算。")

                complexity = done.get("complexity") or {}
                if complexity:
                    st.write("**动态路由判定：**")
                    st.code(
                        f"{complexity.get('label', '')} {complexity.get('reason', '')}",
                        language="text",
                    )

                pruning = done.get("context_pruning") or {}
                if pruning:
                    st.write("**上下文压缩：**")
                    st.code(
                        f"{pruning.get('status')}  "
                        f"{pruning.get('before_chars')} → {pruning.get('after_chars')} 字"
                        f"（省 {pruning.get('saved_chars')} 字）"
                        + (f"\n{pruning['reason']}" if pruning.get("reason") else ""),
                        language="text",
                    )

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

                if done.get("history_entry"):
                    st.write("**已写入 project_history.md：**")
                    st.code(done["history_entry"], language="markdown")
