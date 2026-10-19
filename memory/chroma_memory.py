"""
ChromaDB Semantic Memory (Personal)
====================================
Persistent vector store for personal long-term memory.
Separate from the CCUS knowledge base (chroma_petro).

The embedding model is lazy-loaded on first use to avoid
crashing at import time if HuggingFace Hub is unreachable.
"""
import sys
import os
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chromadb
from chromadb.utils import embedding_functions

from utils.logging_setup import get_logger

log = get_logger(__name__)

try:
    from .config import CHROMA_PATH, CHROMA_COLLECTION, EMBEDDING_MODEL
except ImportError:
    from memory.config import CHROMA_PATH, CHROMA_COLLECTION, EMBEDDING_MODEL

# Lazy-loaded resources (only initialized on first use)
_embedding_fn = None
_client = None
_collection = None


def _ensure_initialized():
    """Lazy init: load embedding model + create ChromaDB collection on first call."""
    global _embedding_fn, _client, _collection
    if _collection is not None:
        return

    log.info(f"[Memory/ChromaDB] 正在加载嵌入模型 {EMBEDDING_MODEL}...")
    _embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    _client = chromadb.PersistentClient(path=CHROMA_PATH)
    _collection = _client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        embedding_function=_embedding_fn
    )
    log.info(f"[Memory/ChromaDB] 初始化完成 (collection={CHROMA_COLLECTION})")


def add(
    text: str,
    metadata: dict | None = None,
    custom_id: str | None = None,
    timestamp: str | None = None,
) -> str:
    """
    Store a memory. Returns the memory ID.

    ``timestamp`` 是**强制字段**：每条记忆都必须能回答"什么时候写下的"。
    不传时用当前时间补，而不是留空 —— 缺失时间戳的记忆在时间维度检索里
    等于不存在，而"忘了传"是最常见的漏法，所以默认值放在这里而不是调用方。
    已有 metadata 里的 timestamp 优先（导入历史数据时不该被覆盖）。
    """
    _ensure_initialized()
    mem_id = custom_id or f"mem_{abs(hash(text))}"

    meta = dict(metadata or {})
    meta.setdefault("timestamp", timestamp or datetime.now().isoformat(timespec="seconds"))
    if timestamp:
        # 显式传入时以传入值为准（回填历史记录的场景）
        meta["timestamp"] = timestamp

    _collection.add(
        documents=[text],
        metadatas=[meta],
        ids=[mem_id]
    )
    log.info(f"[Memory/ChromaDB] Saved (id={mem_id}, ts={meta['timestamp']})")
    return mem_id


def search(query: str, n_results: int = 5, where: dict | None = None) -> list[dict]:
    """
    Semantic similarity search.
    Returns list of {id, document, metadata, distance} sorted by relevance.
    """
    _ensure_initialized()
    kwargs = {"query_texts": [query], "n_results": n_results}
    if where:
        kwargs["where"] = where

    results = _collection.query(**kwargs)
    memories = []
    if results["documents"] and results["documents"][0]:
        for i, doc in enumerate(results["documents"][0]):
            memories.append({
                "id": results["ids"][0][i],
                "document": doc,
                "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                "distance": results["distances"][0][i] if results["distances"] else None
            })
    return memories


def count() -> int:
    """Return total number of stored memories."""
    _ensure_initialized()
    return _collection.count()


def list_all(limit: int = 100) -> list[dict]:
    """Browse all stored memories."""
    _ensure_initialized()
    results = _collection.get(limit=limit)
    memories = []
    if results["documents"]:
        for i, doc in enumerate(results["documents"]):
            memories.append({
                "id": results["ids"][i],
                "document": doc,
                "metadata": results["metadatas"][i] if results["metadatas"] else {}
            })
    return memories
