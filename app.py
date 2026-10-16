import streamlit as st
from typing import TypedDict
from langgraph.graph import StateGraph, END
from openai import OpenAI
import chromadb

# 配置页面
st.set_page_config(page_title="CCUS 多智能体风险评估平台", layout="wide")
st.title("🏭 CCUS 多智能体风险评估平台")
st.write("基于本地大模型与真实学术文献（RAG）的多智能体协同分析系统")

# 初始化 Ollama 客户端和 ChromaDB
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
# 注意：确保你的 chroma_petro 文件夹就在当前项目根目录下
db_client = chromadb.PersistentClient(path="./chroma_petro")
collection = db_client.get_or_create_collection(name="ccus_knowledge")


# 1. 定义状态结构
class AgentState(TypedDict):
    user_input: str
    extracted_params: str
    retrieved_knowledge: str
    final_report: str


# 2. 定义 Agent 1：参数提取
def extract_params(state: AgentState):
    prompt = f"你是一个CCUS专家。请从以下描述中提取评估碳封存风险所需的关键参数（如注入速率、储层深度、断层情况、盖层特性等）：\n{state['user_input']}"
    response = client.chat.completions.create(
        model="qwen2.5", messages=[{"role": "user", "content": prompt}], temperature=0.1
    )
    return {"extracted_params": response.choices[0].message.content}


# 3. 定义 Agent 2：知识检索
def retrieve_knowledge(state: AgentState):
    results = collection.query(query_texts=[state['extracted_params']], n_results=2)
    retrieved_text = "\n".join(results['documents'][0]) if results['documents'] else "知识库中无相关记录"
    return {"retrieved_knowledge": retrieved_text}


# 4. 定义 Agent 3：报告生成（已加入防幻觉约束）
def generate_report(state: AgentState):
    prompt = f"""你是一个资深的CCUS首席工程师。请严格基于以下信息，撰写一份CCUS项目风险评估报告。
【提取的 CCUS 参数】：{state['extracted_params']}
【知识库检索到的标准】：{state['retrieved_knowledge']}
【写作硬性约束】：
1. 必须严格根据【知识库检索到的标准】中的原文内容进行回答。
2. 如果知识库中没有相关记录，必须直接输出“知识库中无相关记录，无法给出结论”，绝不能自行编造数据或文献。
3. 报告必须分点列出：1. 主要地质封存风险 2. 主要井筒风险 3. 缓解与监测建议。"""
    response = client.chat.completions.create(
        model="qwen2.5", messages=[{"role": "user", "content": prompt}], temperature=0.3
    )
    return {"final_report": response.choices[0].message.content}


# 5. 构建工作流
workflow = StateGraph(AgentState)
workflow.add_node("extract", extract_params)
workflow.add_node("retrieve", retrieve_knowledge)
workflow.add_node("report", generate_report)
workflow.set_entry_point("extract")
workflow.add_edge("extract", "retrieve")
workflow.add_edge("retrieve", "report")
workflow.add_edge("report", END)
app = workflow.compile()

# 6. 前端界面交互
user_input = st.text_area("请输入CCUS项目描述（例如：某CCUS项目位于含烃碳酸盐岩储层，注入速率50万吨/年，存在隐伏断层...）",
                          height=100)

if st.button("开始多智能体风险评估"):
    if user_input:
        with st.spinner("Agent团队正在提取参数、检索文献并生成报告，请稍候..."):
            final_state = app.invoke({"user_input": user_input})

            st.success("✅ 评估报告生成完毕！")
            st.subheader("📋 专属评估报告")
            st.write(final_state["final_report"])

            with st.expander("🔍 查看Agent底层工作详情"):
                st.write("**提取的参数：**", final_state["extracted_params"])
                st.write("**检索到的本地文献片段：**", final_state["retrieved_knowledge"])
    else:
        st.warning("请输入项目描述后再点击生成。")