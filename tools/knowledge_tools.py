"""
Knowledge Tools — Web Search + Auto-Save
=========================================
When local knowledge base isn't enough, search the web and save
valuable results to both local ChromaDB (CCUS knowledge base)
and Notion for future reference.

Core function:
  - search_and_save(query, max_results)
      → Tavily search → dedup → write ChromaDB + Notion → return summary
"""
import os
import hashlib
from datetime import datetime

# Load env
from dotenv import load_dotenv
load_dotenv()

# ─── Tavily (web search) ──────────────────────────────────────
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

_tavily = None

def _get_tavily():
    """Lazy-init Tavily client."""
    global _tavily
    if _tavily is None and TAVILY_API_KEY:
        from tavily import TavilyClient
        _tavily = TavilyClient(api_key=TAVILY_API_KEY)
    return _tavily


# ─── Notion (auto-save to knowledge base) ─────────────────────
NOTION_TOKEN = os.getenv("NOTION_TOKEN", "")
NOTION_PAGE_ID = os.getenv("NOTION_PAGE_ID", "").replace("-", "")

_notion = None

def _get_notion():
    """Lazy-init Notion client."""
    global _notion
    if _notion is None and NOTION_TOKEN and NOTION_PAGE_ID:
        from notion_client import Client
        _notion = Client(auth=NOTION_TOKEN)
    return _notion


# ─── ChromaDB (CCUS knowledge base, same as existing) ─────────
import chromadb

from utils.embeddings import describe as describe_embedding
from utils.embeddings import get_embedding_function
from utils.logging_setup import get_logger

log = get_logger(__name__)

CCUS_CHROMA_PATH = os.getenv("CCUS_CHROMA_PATH", r"D:\python-learn\chroma_petro")
CCUS_COLLECTION = os.getenv("CCUS_COLLECTION", "ccus_knowledge_v3")

_chroma_client = None
_chroma_collection = None
_embedding_fn = None

def _ensure_chroma():
    """Lazy-init ChromaDB collection (avoids loading embedding model at import)."""
    global _chroma_client, _chroma_collection, _embedding_fn
    if _chroma_collection is not None:
        return
    # 嵌入函数统一由 utils.embeddings 决定，必须与 ingest_ccus_batch.py 写入时
    # 用的完全一致。这里曾硬编码 BAAI/bge-small-zh-v1.5（512 维）去查一个
    # 384 维的库，查询路径一直是坏的 —— 详见 utils/embeddings.py 的说明。
    log.info(f"[Knowledge Tools] 正在加载嵌入模型: {describe_embedding()}")
    _embedding_fn = get_embedding_function()
    _chroma_client = chromadb.PersistentClient(path=CCUS_CHROMA_PATH)
    _chroma_collection = _chroma_client.get_or_create_collection(
        name=CCUS_COLLECTION,
        embedding_function=_embedding_fn
    )
    log.info(f"[Knowledge Tools] ChromaDB 就绪 (collection={CCUS_COLLECTION})")


def get_collection():
    """返回就绪的 CCUS 集合（必要时惰性初始化）。

    对外暴露这个函数，是为了让 ``chroma_retriever`` 之类的新检索入口
    不必自己去 import 模块级全局变量 —— 集合名/路径/嵌入函数只在这里定义，
    少一条绕路就少一次"两边不一致"的机会。
    """
    _ensure_chroma()
    return _chroma_collection


def is_tavily_configured() -> bool:
    """Check if Tavily API key is set."""
    return bool(TAVILY_API_KEY)


def search_and_save(query: str, max_results: int = 4) -> str:
    """
    Search the web for CCUS-related content and save valuable results
    to local ChromaDB knowledge base + Notion.

    Args:
        query:       Search query (will be enhanced with CCUS context)
        max_results: Max number of results to fetch

    Returns:
        Formatted string of saved content, or status message.
    """
    tavily = _get_tavily()
    if tavily is None:
        return "联网搜索未启用（未配置 TAVILY_API_KEY）。请在 .env 中设置。"

    try:
        # 1. Search — add CCUS context to the query
        enhanced_query = f"{query} CCUS OR 'carbon capture' OR 'Bo Zhang' Alberta"
        log.info(f"[Web Search] 正在搜索: {enhanced_query}")
        resp = tavily.search(query=enhanced_query, max_results=max_results)
        results = resp.get("results", [])

        if not results:
            return "未找到有价值的联网结果。"

        _ensure_chroma()
        notion = _get_notion()

        saved = []
        skipped = 0

        for item in results:
            title = item.get("title", "无标题")[:80]
            content = item.get("content", "")[:1200]
            source = item.get("url", "")

            # Skip low-quality results
            if len(content) < 100:
                continue

            # 2. Dedup — check if already in ChromaDB by content hash
            doc_id = hashlib.md5(f"{title}{source}".encode()).hexdigest()
            existing = _chroma_collection.get(ids=[doc_id])
            if existing["ids"]:
                skipped += 1
                continue

            # 3. Write to ChromaDB (CCUS knowledge base)
            _chroma_collection.add(
                documents=[content],
                metadatas=[{
                    "title": title,
                    "source": source,
                    "time": datetime.now().isoformat(),
                    "type": "web_auto"
                }],
                ids=[doc_id]
            )

            # 4. Write to Notion (failure doesn't break main flow)
            if notion:
                try:
                    # Chunk content to respect Notion's 2000-char limit
                    children = []
                    remaining = content
                    while remaining:
                        chunk = remaining[:2000]
                        remaining = remaining[2000:]
                        children.append({
                            "object": "block",
                            "type": "paragraph",
                            "paragraph": {
                                "rich_text": [{"type": "text", "text": {"content": chunk}}]
                            }
                        })
                    # Add source link
                    children.append({
                        "object": "block",
                        "type": "bookmark",
                        "bookmark": {"url": source}
                    })

                    notion.pages.create(
                        parent={"page_id": NOTION_PAGE_ID},
                        properties={
                            "title": {"title": [{"text": {"content": f"[Auto] {title}"}}]}
                        },
                        children=children
                    )
                except Exception as e:
                    log.warning(f"[Web Search] Notion 写入失败（不影响主流程）: {e}")

            saved.append(f"【{title}】\n{content}\n来源: {source}")

        if saved:
            msg = f"已保存 {len(saved)} 条新内容到本地知识库"
            if skipped:
                msg += f"（跳过 {skipped} 条重复）"
            log.info(f"[Web Search] {msg}")
            return "\n\n".join(saved)
        else:
            return "搜索完成，但没有新内容值得保存（可能已存在）。"

    except Exception as e:
        return f"搜索或保存失败: {str(e)}"
