"""
CCUS 多智能体风险评估系统 — 前端（Streamlit）
=============================================
前后端完全解耦：本文件只负责 UI 与 HTTP 调用，
所有 LangGraph / GraphRAG / Ollama 逻辑都在 api_server.py 后端。

依赖：streamlit、requests
启动：streamlit run app.py --server.port 8505
"""
import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/api/analyze"

st.set_page_config(page_title="CCUS 多智能体风险评估系统", layout="wide")
st.title("🏭 CCUS 多智能体风险评估系统")
st.caption("LangGraph 多智能体 + GraphRAG 知识图谱 · 前后端解耦架构")

# ─── 输入区 ────────────────────────────────────────────
user_input = st.text_area(
    "请输入 CCUS 项目描述",
    placeholder="例如：某CCUS项目注入压力250bar，温度80℃，存在隐伏碳酸盐岩储层断层，请评估风险。",
    height=120,
)

# ─── 提交与结果展示 ────────────────────────────────────
if st.button("启动评估", type="primary"):
    if not user_input.strip():
        st.warning("请输入项目描述后再启动评估。")
    else:
        try:
            with st.spinner("Agent团队正在全速运转，预计需要几十秒..."):
                resp = requests.post(
                    API_URL,
                    json={"user_input": user_input.strip()},
                    timeout=120,
                )

            if resp.status_code == 200:
                data = resp.json()

                if data.get("status") == "success":
                    st.success("✅ 评估完成")
                    st.markdown("### 📋 风险评估报告")
                    st.markdown(data.get("report", "（报告为空）"))

                    with st.expander("🔍 查看底层详情"):
                        st.write("**提取的参数：**")
                        st.code(data.get("extracted_params", ""), language="markdown")
                        st.metric("评审分数", f"{data.get('score', '-')} / 10")
                else:
                    st.error(f"后端返回错误：{data.get('message', '未知错误')}")
            else:
                st.error("请求后端失败，请检查 api_server.py 是否在 8000 端口正常运行。")

        except requests.exceptions.Timeout:
            st.error("请求后端失败，请检查 api_server.py 是否在 8000 端口正常运行。（请求超时，完整评估可能超过 120 秒）")
        except requests.exceptions.ConnectionError:
            st.error("请求后端失败，请检查 api_server.py 是否在 8000 端口正常运行。")
        except Exception as e:
            st.error(f"请求后端失败，请检查 api_server.py 是否在 8000 端口正常运行。（{e}）")
