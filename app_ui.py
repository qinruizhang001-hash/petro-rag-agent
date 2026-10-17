import streamlit as st
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from openai import OpenAI
import chromadb
import sys
import os

# 导入个人记忆模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from memory.memory_manager import retrieve_memory, save_memory, search_semantic
    from memory.memory_tools import search_personal_memory
    from memory.chroma_memory import count as memory_count, list_all as memory_list_all
    _MEMORY_READY = True
except Exception as e:
    _MEMORY_READY = False
    st.warning(f"个人记忆模块未就绪（不影响核心功能）: {e}")

# 导入联网搜索工具
try:
    from tools.knowledge_tools import search_and_save, is_tavily_configured
    _WEB_READY = True
except Exception as e:
    _WEB_READY = False
    st.warning(f"联网搜索模块未就绪（不影响核心功能）: {e}")

# 配置页面
st.set_page_config(page_title="CCUS 多智能体风险评估", layout="wide")
st.title("CCUS 多智能体风险评估")
st.write("基于本地大模型与真实学术文献（RAG）的多智能体协同分析系统")

# ─── 侧边栏：个人记忆面板 ───────────────────────────────────
with st.sidebar:
    st.header("📝 个人长期记忆")
    if _MEMORY_READY:
        mem_count = memory_count()
        st.metric("已存记忆条数", mem_count)

        # 手动保存记忆
        with st.expander("手动保存记忆"):
            mem_title = st.text_input("标题", key="manual_title")
            mem_content = st.text_area("内容", key="manual_content", height=80)
            mem_tags = st.text_input("标签（逗号分隔）", key="manual_tags")
            if st.button("保存到记忆"):
                if mem_title and mem_content:
                    tags = [t.strip() for t in mem_tags.split(",") if t.strip()] if mem_tags else None
                    save_memory(mem_title, mem_content, tags)
                    st.success("已保存！")
                else:
                    st.warning("标题和内容不能为空")

        # 搜索记忆
        st.divider()
        search_query = st.text_input("搜索个人记忆", placeholder="输入关键词...", key="search_input")
        if search_query and st.button("搜索记忆"):
            results = search_semantic(search_query, n_results=5)
            if results:
                for i, mem in enumerate(results, 1):
                    title = mem["metadata"].get("title", "Untitled")
                    tags = mem["metadata"].get("tags", "")
                    dist = mem.get("distance")
                    relevance = f"{1 - dist:.0%}" if dist is not None else ""
                    with st.expander(f"#{i} {title} {f'(相关度: {relevance})' if relevance else ''}"):
                        st.write(mem["document"][:500])
                        if tags:
                            st.caption(f"Tags: {tags}")
            else:
                st.info("未找到相关记忆")

        # 浏览全部记忆
        if st.button("浏览全部记忆"):
            all_mems = memory_list_all(limit=50)
            if all_mems:
                for mem in all_mems:
                    title = mem["metadata"].get("title", "Untitled")
                    with st.expander(f"{title}"):
                        st.write(mem["document"][:300])
            else:
                st.info("记忆库为空")
    else:
        st.info("个人记忆模块未加载。请先安装依赖: pip install notion-client python-dotenv")

    st.divider()
    st.caption("💡 Agent 会自动判断是否需要搜索记忆 + 本地不足时联网搜索并沉淀")

# 初始化 Ollama 客户端和 ChromaDB
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
db_client = chromadb.PersistentClient(path="./chroma_petro")
collection = db_client.get_or_create_collection(name="ccus_knowledge")

# 记忆触发关键词
MEMORY_TRIGGERS = [
    "我的", "我之前", "我上次", "我做过", "我写过", "我的笔记",
    "复盘", "总结一下我", "项目进展", "我学到", "失败经验",
    "论文笔记", "我读过", "上次评估", "之前的报告", "我记得",
    "my ", "i previously", "last time", "i did", "i wrote", "my notes",
    "review", "my project", "i learned", "failure", "paper notes",
    "i read", "previous report", "i remember",
]


# ============================================================
# AgentState（同步 multi_agent_ccus.py）
# ============================================================
class AgentState(TypedDict):
    user_input: str
    extracted_params: str
    retrieved_knowledge: str
    final_report: str
    score: float
    feedback: str
    need_memory: bool
    personal_memory: str
    personal_context: str
    web_content: str


# ============================================================
# Memory Router 节点
# ============================================================
def memory_router(state: AgentState):
    """判断是否需要调用个人记忆"""
    print("\n[Memory Router] 正在判断是否需要个人记忆...")
    user_text = state["user_input"].lower().strip()
    need_memory = any(trigger in user_text for trigger in MEMORY_TRIGGERS)

    if need_memory:
        print("  -> 检测到记忆触发词，将搜索个人长期记忆")
    else:
        print("  -> 无记忆触发词，直接进入 CCUS 分析流程")

    return {"need_memory": need_memory}


def search_memory(state: AgentState):
    """执行个人记忆检索"""
    print("\n[Search Memory] 正在检索个人长期记忆...")
    if not _MEMORY_READY:
        return {"personal_memory": "记忆模块未加载"}
    try:
        result = search_personal_memory(state["user_input"])
        print(f"  -> 检索完成，约 {len(result)} 字符")
        return {"personal_memory": result}
    except Exception as e:
        print(f"  -> 记忆检索失败: {e}")
        return {"personal_memory": f"记忆检索失败: {e}"}


def inject_memory(state: AgentState):
    """整理记忆上下文"""
    print("\n[Inject Memory] 正在整理记忆上下文...")
    memory = state.get("personal_memory", "")

    if not memory or "未在个人记忆中找到" in memory or "检索失败" in memory or "未加载" in memory:
        print("  -> 无有效记忆，跳过注入")
        return {"personal_context": ""}

    personal_context = f"""
### 个人长期记忆（请优先参考）
{memory}

请在生成回答时，结合以上个人记忆中的内容。如果记忆与当前问题相关，必须引用。
"""
    print("  -> 记忆已注入上下文")
    return {"personal_context": personal_context}


def route_memory(state: AgentState):
    if state.get("need_memory"):
        return "search_memory"
    else:
        return "extract"


# ============================================================
# 原有节点
# ============================================================
def extract_params(state: AgentState):
    prompt = (
        "你是一个CCUS专家。请从以下描述中提取评估碳封存风险所需的关键参数"
        "（如注入速率、储层深度、断层情况、盖层特性等）：\n"
        f"{state['user_input']}"
    )
    response = client.chat.completions.create(
        model="qwen2.5", messages=[{"role": "user", "content": prompt}], temperature=0.1
    )
    return {"extracted_params": response.choices[0].message.content}


def retrieve_knowledge(state: AgentState):
    results = collection.query(
        query_texts=[state["extracted_params"]], n_results=2
    )
    retrieved_text = (
        "\n".join(results["documents"][0]) if results["documents"] else "知识库中无相关记录"
    )
    return {"retrieved_knowledge": retrieved_text}


def generate_report(state: AgentState):
    personal_context = state.get("personal_context", "")
    web_context = ""
    if state.get("web_content"):
        web_context = f"\n【联网搜索补充内容（已自动入库）】：\n{state['web_content']}"

    prompt = f"""你是一个资深的CCUS首席工程师。请严格基于以下信息，撰写一份CCUS项目风险评估报告。
【提取的 CCUS 参数】：{state['extracted_params']}
【知识库检索到的标准】：{state['retrieved_knowledge']}
{web_context}
{personal_context}
【上一版报告的修改意见】：{state.get('feedback', '无')}

【写作硬性约束】：
1. 必须严格根据【知识库检索到的标准】中的原文内容进行回答。
2. 如果知识库中没有相关记录，必须直接输出"知识库中无相关记录，无法给出结论"，绝不能自行编造数据或文献。
3. 报告必须分点列出：1. 主要地质封存风险 2. 主要井筒风险 3. 缓解与监测建议。"""
    response = client.chat.completions.create(
        model="qwen2.5", messages=[{"role": "user", "content": prompt}], temperature=0.3
    )
    report = response.choices[0].message.content

    # 自动保存报告到个人长期记忆（仅第一次生成）
    if not state.get("feedback") and _MEMORY_READY:
        try:
            save_memory(
                title=f"CCUS评估报告: {state['user_input'][:50]}",
                content=report,
                tags=["CCUS", "评估报告", "auto-save"]
            )
        except Exception as e:
            print(f"记忆保存失败: {e}")

    return {"final_report": report}


def evaluate_report(state: AgentState):
    prompt = f"""你是一个严格的CCUS学术评审员。请根据【检索到的文献】，评估【AI生成的回答】的质量。
【用户问题】：{state['user_input']}
【检索到的文献】：{state['retrieved_knowledge']}
【AI生成的回答】：{state['final_report']}

请严格按以下格式输出，不要输出任何其他内容：
分数：[0-10的整数]
评语：[1-2句简短评语]"""
    response = client.chat.completions.create(
        model="qwen2.5", messages=[{"role": "user", "content": prompt}], temperature=0.0
    )
    result = response.choices[0].message.content
    try:
        score = int(result.split("分数：")[1].split("\n")[0].strip())
        feedback = result.split("评语：")[1].strip()
    except Exception:
        score = 10
        feedback = result
    return {"score": score, "feedback": feedback}


def check_quality(state: AgentState):
    if state["score"] >= 9:
        return "pass"
    else:
        return "rewrite"


# ============================================================
# Web Search 节点（本地不够 → 联网搜索 → 自动沉淀）
# ============================================================
def should_search(state: AgentState):
    """本地知识不足时触发联网搜索"""
    local = state.get("retrieved_knowledge", "")

    if not _WEB_READY or not is_tavily_configured():
        return "report"

    if len(local) < 150 or "未找到" in local or "无相关记录" in local or "没有相关" in local:
        print("[Web Router] 本地知识不足，触发联网搜索")
        return "web_search"
    else:
        print("[Web Router] 本地知识充足，直接生成报告")
        return "report"


def web_search_save(state: AgentState):
    """联网搜索 + 自动写入本地库"""
    query = state.get("extracted_params", state["user_input"])[:200]
    result = search_and_save(query, max_results=4)
    return {"web_content": result}


# ============================================================
# 构建工作流图（Memory Router 前置）
# ============================================================
workflow = StateGraph(AgentState)

workflow.add_node("memory_router", memory_router)
workflow.add_node("search_memory", search_memory)
workflow.add_node("inject_memory", inject_memory)
workflow.add_node("extract", extract_params)
workflow.add_node("retrieve", retrieve_knowledge)
workflow.add_node("web_search", web_search_save)
workflow.add_node("report", generate_report)
workflow.add_node("evaluate", evaluate_report)

workflow.set_entry_point("memory_router")

workflow.add_conditional_edges(
    "memory_router",
    route_memory,
    {
        "search_memory": "search_memory",
        "extract": "extract",
    }
)

workflow.add_edge("search_memory", "inject_memory")
workflow.add_edge("inject_memory", "extract")
workflow.add_edge("extract", "retrieve")

# ★ retrieve 之后判断是否需要联网搜索
workflow.add_conditional_edges(
    "retrieve",
    should_search,
    {
        "web_search": "web_search",
        "report": "report",
    }
)
workflow.add_edge("web_search", "report")
workflow.add_edge("report", "evaluate")

workflow.add_conditional_edges(
    "evaluate",
    check_quality,
    {
        "pass": END,
        "rewrite": "report",
    },
)

app = workflow.compile()


# ============================================================
# 前端界面
# ============================================================
user_input = st.text_area(
    "请输入问题或CCUS项目描述",
    placeholder="例如：1) 某CCUS项目位于含烃碳酸盐岩储层...  2) 总结一下我最近的项目进展  3) 我之前读过哪些关于强化学习的论文？",
    height=100,
)

# 提示：Memory Router 会自动判断
st.caption("💡 Agent 会自动判断你的问题是否需要搜索个人记忆")

if st.button("开始多智能体分析"):
    if user_input:
        with st.spinner("Agent 团队正在工作，请稍候..."):
            final_state = app.invoke({"user_input": user_input})

            # Memory Router 决策结果
            if final_state.get("need_memory"):
                st.info("🔍 Memory Router 判定：需要搜索个人记忆")
            else:
                st.info("⚡ Memory Router 判定：直接进入 CCUS 分析（无需记忆）")

            # Web Search 决策结果
            if final_state.get("web_content"):
                st.success("🌐 联网搜索已执行，新内容已自动写入本地知识库 + Notion")

            st.success("分析完成！")
            st.subheader("专属评估报告")
            st.write(final_state["final_report"])

            with st.expander("查看 Agent 底层工作详情"):
                st.write("**Memory Router 决策：**", "需要记忆" if final_state.get("need_memory") else "跳过记忆")

                # 显示记忆检索结果
                if final_state.get("personal_memory"):
                    st.write("**检索到的个人记忆：**")
                    st.code(final_state["personal_memory"][:500], language="markdown")

                if final_state.get("personal_context"):
                    st.write("**注入的记忆上下文：**")
                    st.code(final_state["personal_context"][:300], language="markdown")

                # 显示联网搜索结果
                if final_state.get("web_content"):
                    st.write("**联网搜索补充内容：**")
                    st.code(final_state["web_content"][:500], language="markdown")

                st.write("**提取的参数：**", final_state["extracted_params"])
                st.write("**检索到的本地文献片段：**", final_state["retrieved_knowledge"])
                st.write(f"**评审分数：** {final_state['score']} / 10")
                st.write(f"**评审意见：** {final_state['feedback']}")
    else:
        st.warning("请输入问题后再点击生成。")
