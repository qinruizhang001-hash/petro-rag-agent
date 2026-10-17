"""
Memory Tools
=============
Tool functions for the LangGraph agent to interact with personal memory.

Tools:
  - search_personal_memory(query)  → semantic + structured search
  - save_personal_memory(title, content, tags)  → save to both stores
"""

import sys
import os

# Ensure parent dir is on path so `memory` package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory.memory_manager import retrieve_memory, save_memory, search_semantic


def search_personal_memory(query: str) -> str:
    """
    Search personal long-term memory (Notion + ChromaDB).

    Includes: CCUS paper notes, project reflections, failure experiences,
    habit records, learning insights, etc.

    Triggers: when the question involves "我的", "之前", "笔记", "复盘",
    "项目进展", "我做过", "论文笔记" etc.

    Args:
        query: Natural language query

    Returns:
        Formatted string with matched memories, or "no results" message.
    """
    try:
        results = retrieve_memory(query, n_results=5)

        # Semantic results (ChromaDB)
        semantic = results.get("semantic", [])
        if not semantic:
            return "未在个人记忆中找到相关内容。"

        formatted = []
        for i, mem in enumerate(semantic, 1):
            title = mem.get("metadata", {}).get("title", "无标题记忆")
            tags = mem.get("metadata", {}).get("tags", "")
            dist = mem.get("distance")
            relevance = f" (相关度: {1 - dist:.0%})" if dist is not None else ""
            doc = mem.get("document", "").strip()
            tag_str = f" [{tags}]" if tags else ""
            formatted.append(f"【{i}. {title}】{tag_str}{relevance}\n{doc}")

        # Structured results (Notion) — append as supplementary
        structured = results.get("structured", [])
        if structured:
            formatted.append("\n--- Notion 中的相关页面 ---")
            for s in structured:
                formatted.append(f"- {s['title']} ({s['url']})")

        return "\n\n----------------\n\n".join(formatted)

    except Exception as e:
        return f"记忆检索失败: {str(e)}"


def save_personal_memory(title: str, content: str, tags: list = None) -> str:
    """
    Save a memory to personal long-term storage (Notion + ChromaDB).

    Used when the agent decides something is worth remembering.

    Args:
        title:   Short title
        content: Full content
        tags:    Optional tag list

    Returns:
        Confirmation message with IDs.
    """
    try:
        result = save_memory(title=title, content=content, tags=tags)
        return f"已保存到长期记忆。Notion ID: {result.get('notion_page_id', 'N/A')}, ChromaDB ID: {result['chroma_id']}"
    except Exception as e:
        return f"记忆保存失败: {str(e)}"
