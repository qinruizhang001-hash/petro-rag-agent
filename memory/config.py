"""
Personal Memory Config
======================
Adapted for integration into petro-rag-agent.

Uses the SAME Ollama endpoint and ChromaDB base path as the existing project.
The personal memory collection is stored in a separate ChromaDB folder
(./personal_chroma_db) to keep it isolated from the CCUS knowledge base.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# --- Notion ---
NOTION_TOKEN = os.getenv("NOTION_TOKEN", "")
NOTION_PAGE_ID = os.getenv("NOTION_PAGE_ID", "").replace("-", "")

# --- Personal ChromaDB (separate from CCUS knowledge base) ---
CHROMA_PATH = os.getenv("PERSONAL_CHROMA_PATH", "./personal_chroma_db")
CHROMA_COLLECTION = os.getenv("PERSONAL_CHROMA_COLLECTION", "personal_memory")

# --- Embedding Model (same as petro-rag-agent) ---
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")

# --- Ollama (same as existing multi_agent_ccus.py) ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5")


def is_notion_configured() -> bool:
    """Check if Notion is ready to use."""
    return bool(NOTION_TOKEN and NOTION_PAGE_ID)
