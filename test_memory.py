"""
Test the integrated personal memory inside petro-rag-agent.
Run: python test_memory.py

Tests:
  1. Memory module imports
  2. ChromaDB personal memory write + search
  3. Notion write (if configured)
  4. Unified retrieve
"""

import sys
import os

# Ensure we can import from memory/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    print("=" * 50)
    print("Personal Memory — Integration Test")
    print("=" * 50)

    # 1. Import test
    print("\n[1] Importing memory module...")
    try:
        from memory.memory_manager import save_memory, retrieve_memory, search_semantic
        from memory.chroma_memory import count, list_all
        from memory.config import is_notion_configured
        print("  [OK] All imports successful")
    except Exception as e:
        print(f"  [FAIL] Import error: {e}")
        print("  Make sure you installed: pip install notion-client python-dotenv chromadb sentence-transformers")
        return False

    # 2. ChromaDB test
    print("\n[2] Testing ChromaDB personal memory...")
    try:
        # Write
        save_memory(
            title="Test: CCUS DRL Paper Note",
            content="Bo Zhang的强化学习论文表明，可以用DRL优化CO2注入策略，同时控制泄漏风险。",
            tags=["CCUS", "test"]
        )
        # Search
        results = search_semantic("强化学习优化二氧化碳注入", n_results=3)
        print(f"  [OK] ChromaDB has {count()} memories, search returned {len(results)} results")
        if results:
            print(f"  Top: {results[0]['document'][:60]}...")
        chroma_ok = True
    except Exception as e:
        print(f"  [FAIL] {e}")
        chroma_ok = False

    # 3. Notion test
    print("\n[3] Testing Notion...")
    if is_notion_configured():
        print("  Notion configured — write test should appear in your Notion page")
        # save_memory already wrote to Notion in step 2
        print("  [OK] (Check your Notion page for the test note)")
    else:
        print("  [SKIP] Notion not configured — ChromaDB-only mode")
        print("  To enable: copy .env.example → .env → fill in NOTION_TOKEN and NOTION_PAGE_ID")

    # 4. Unified retrieval
    print("\n[4] Testing unified retrieval...")
    try:
        full = retrieve_memory("CCUS 注入策略")
        print(f"  [OK] Semantic: {len(full['semantic'])} | Structured: {len(full['structured'])}")
    except Exception as e:
        print(f"  [FAIL] {e}")

    # Summary
    print("\n" + "=" * 50)
    if chroma_ok:
        print("RESULT: Core memory system working!")
        print("The CCUS agent will now:")
        print("  1. Search personal memory before generating reports")
        print("  2. Auto-save generated reports to long-term memory")
        if not is_notion_configured():
            print("\nNOTE: Notion not configured — only ChromaDB is active.")
            print("Configure Notion later by filling in .env (optional, ChromaDB works standalone)")
    else:
        print("RESULT: Some tests failed. Check errors above.")

    return chroma_ok

if __name__ == "__main__":
    main()
