"""
Unified Memory Manager
======================
Single interface: writes to Notion + ChromaDB, retrieves from both.

This is what the CCUS agent calls to save and search personal memories.
"""

import sys
import os

# Ensure parent dir is on path so `memory` package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Use try/except for both relative and absolute import styles
try:
    from .notion_memory import add_note, search as notion_search
    from .chroma_memory import add as chroma_add, search as chroma_search
except ImportError:
    from memory.notion_memory import add_note, search as notion_search
    from memory.chroma_memory import add as chroma_add, search as chroma_search


def save_memory(title: str, content: str, tags: list | None = None) -> dict:
    """
    Save to BOTH stores:
      - Notion  → structured page (if configured)
      - ChromaDB → vector embedding (always works)

    Returns:
        {"notion_page_id": str|None, "chroma_id": str}
    """
    notion_page = add_note(title=title, content=content, tags=tags)

    chroma_id = chroma_add(
        text=content,
        metadata={
            "title": title,
            "tags": ",".join(tags) if tags else "",
            "source": "petro-agent",
            "notion_page_id": notion_page.get("id", "") if notion_page else ""
        }
    )

    return {
        "notion_page_id": notion_page.get("id", "") if notion_page else None,
        "chroma_id": chroma_id
    }


def retrieve_memory(query: str, n_results: int = 5) -> dict:
    """
    Unified retrieval: semantic (ChromaDB) + structured (Notion).

    Returns:
        {"semantic": [...], "structured": [...]}
    """
    semantic_results = chroma_search(query, n_results=n_results)

    notion_results = notion_search(query)
    structured = [
        {"id": p["id"], "title": p["title"], "url": p["url"]}
        for p in notion_results
    ]

    return {
        "semantic": semantic_results,
        "structured": structured
    }


def search_semantic(query: str, n_results: int = 3) -> list[dict]:
    """ChromaDB-only semantic search — fast, fuzzy."""
    return chroma_search(query, n_results=n_results)


def search_structured(query: str) -> list[dict]:
    """Notion-only search — exact keyword matching."""
    return notion_search(query)
