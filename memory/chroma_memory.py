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


def add(text: str, metadata: dict | None = None, custom_id: str | None = None) -> str:
    """Store a memory. Returns the memory ID."""
    _ensure_initialized()
    mem_id = custom_id or f"mem_{abs(hash(text))}"
    _collection.add(
        documents=[text],
        metadatas=[metadata or {}],
        ids=[mem_id]
    )
    log.info(f"[Memory/ChromaDB] Saved (id={mem_id})")
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
