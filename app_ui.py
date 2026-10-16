import streamlit as st
from openai import OpenAI
import chromadb

# 配置页面
st.set_page_config(page_title="石油工程 AI 助手", layout="centered")
st.title("🛢️ 石油工程知识问答助手")
st.write("基于本地知识库（RAG）")

# 初始化 ChromaDB 和 Ollama
client_db = chromadb.PersistentClient(path="./chroma_petro")
# 注意这里改成了 petro_pdf_knowledge，对应你刚处理完的PDF库
collection = client_db.get_or_create_collection(name="petro_pdf_knowledge")
client_llm = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

user_query = st.text_input("请输入你的问题（例如：论文中的预测准确率是多少？）：")

if st.button("发送提问"):
    if user_query:
        with st.spinner("正在检索本地知识库并生成回答..."):
            # 1. 检索
            results = collection.query(query_texts=[user_query], n_results=2)
            retrieved_text = "\n".join(results['documents'][0]) if results['documents'] else "无相关记录"

            # 2. 拼接 Prompt
            prompt = f"""你是一个专业的石油工程AI助手。
请严格基于下面提供的知识库内容回答问题。
知识库内容：
{retrieved_text}

用户问题：{user_query}
"""
            # 3. 调用大模型
            response = client_llm.chat.completions.create(
                model="qwen2.5",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )

            # 4. 显示结果
            st.success("回答生成完毕！")
            st.write("### 专属回答：")
            st.write(response.choices[0].message.content)
            with st.expander("查看检索到的本地原始资料"):
                st.write(retrieved_text)