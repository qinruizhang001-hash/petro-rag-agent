"""
Notion Structured Memory
=========================
Write / search personal notes to a Notion page.
Falls back gracefully if Notion is not configured.
"""
from typing import Optional

import sys
import os

from utils.logging_setup import get_logger

log = get_logger(__name__)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from .config import NOTION_TOKEN, NOTION_PAGE_ID, is_notion_configured
except ImportError:
    from memory.config import NOTION_TOKEN, NOTION_PAGE_ID, is_notion_configured

_notion = None

def _get_client():
    """Lazy-init Notion client. Returns None if not configured."""
    global _notion
    if _notion is None and is_notion_configured():
        from notion_client import Client
        _notion = Client(auth=NOTION_TOKEN)
    return _notion


def add_note(title: str, content: str, tags: Optional[list] = None) -> Optional[dict]:
    """
    Create a new page under the configured Notion page.
    Returns None if Notion is not configured.
    """
    notion = _get_client()
    if notion is None:
        log.info("[Memory/Notion] Not configured — skipping Notion write")
        return None

    # Chunk content to respect Notion's 2000-char limit per rich_text block
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

    if tags:
        tag_text = " ".join(f"#{t}" for t in tags)
        children.append({
            "object": "block",
            "type": "callout",
            "callout": {
                "rich_text": [{"type": "text", "text": {"content": f"Tags: {tag_text}"}}],
                "icon": {"type": "emoji", "emoji": "🏷️"}
            }
        })

    try:
        page = notion.pages.create(
            parent={"page_id": NOTION_PAGE_ID},
            properties={
                "title": {"title": [{"text": {"content": title}}]}
            },
            children=children
        )
        log.info(f"[Memory/Notion] Saved: {title}")
        return page
    except Exception as e:
        log.warning(f"[Memory/Notion] Error: {e}")
        return None


def search(query: str, page_size: int = 5) -> list:
    """
    Full-text search across accessible Notion pages.
    Returns a clean list of {id, title, url} dicts.
    """
    notion = _get_client()
    if notion is None:
        return []

    try:
        results = notion.search(query=query, page_size=page_size)
        pages = []
        for page in results.get("results", []):
            title = "Untitled"
            props = page.get("properties", {})
            if "title" in props and props["title"].get("title"):
                title = "".join(t.get("plain_text", "") for t in props["title"]["title"])
            elif page.get("title"):
                title = page["title"]
            pages.append({
                "id": page.get("id", ""),
                "title": title,
                "url": page.get("url", "")
            })
        return pages
    except Exception as e:
        log.warning(f"[Memory/Notion] Search error: {e}")
        return []
