import chromadb
from pypdf import PdfReader

# 1. 初始化 ChromaDB（新建一个专门放 PDF 的库）
client = chromadb.PersistentClient(path="./chroma_petro")
collection = client.get_or_create_collection(name="petro_pdf_knowledge")

# 2. 读取 PDF
pdf_path = "paper.pdf"
print("正在读取论文 PDF，请稍候...")
reader = PdfReader(pdf_path)

full_text = ""
for page in reader.pages:
    text = page.extract_text()
    if text:
        full_text += text + "\n"

print(f"PDF读取完毕，共提取了 {len(full_text)} 个字符。")

# 3. 将长文本切分成小块（每块500字，重叠50字）
chunk_size = 500
overlap = 50
chunks = []
for i in range(0, len(full_text), chunk_size - overlap):
    chunk = full_text[i:i + chunk_size]
    if len(chunk) > 50:
        chunks.append(chunk)

print(f"正在向向量数据库存入 {len(chunks)} 个知识片段...")

# 4. 存入 ChromaDB
for i, chunk in enumerate(chunks):
    collection.add(
        documents=[chunk],
        ids=[f"pdf_chunk_{i}"]
    )

print("✅ 论文 PDF 成功入库！你的本地 AI 已经读完了这篇文献。")