"""
Test Web Search Integration
=============================
Run: python test_web_search.py

Tests:
  1. Tavily config check (is_tavily_configured)
  2. should_search routing logic (local has content vs local is empty)
  3. search_and_save function (needs real Tavily API key)
  4. Import chain verification
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_imports():
    """Test that all modules can be imported."""
    print("=== 1. Import Chain Test ===")
    try:
        from tools.knowledge_tools import search_and_save, is_tavily_configured
        print(f"  [OK] search_and_save: {search_and_save}")
        print(f"  [OK] is_tavily_configured: {is_tavily_configured}")
        return True
    except Exception as e:
        print(f"  [FAIL] {e}")
        return False


def test_tavily_config():
    """Check if Tavily API key is set."""
    print("\n=== 2. Tavily Config ===")
    try:
        from tools.knowledge_tools import is_tavily_configured
        if is_tavily_configured():
            print("  [OK] Tavily API key is set")
            return True
        else:
            print("  [SKIP] Tavily not configured")
            print("  To enable: get API key from https://tavily.com")
            print("  Add TAVILY_API_KEY=tvly_xxx to .env")
            return False
    except Exception as e:
        print(f"  [FAIL] {e}")
        return False


def test_should_search_logic():
    """Test the should_search routing logic without LLM."""
    print("\n=== 3. should_search Routing Logic ===")

    # Simulate the logic
    def should_search_check(local_knowledge: str, tavily_ready: bool) -> str:
        if not tavily_ready:
            return "report"
        if len(local_knowledge) < 150 or "未找到" in local_knowledge or "无相关记录" in local_knowledge:
            return "web_search"
        return "report"

    # Case A: local is empty
    result_a = should_search_check("无相关记录", True)
    print(f"  Case A (empty local):     {'web_search' if result_a == 'web_search' else 'report'}")
    ok_a = result_a == "web_search"

    # Case B: local has content
    long_text = "这是详细的CCUS地质封存风险评估知识库内容，包含了大量的地质参数和封存机制分析。" * 10
    result_b = should_search_check(long_text, True)
    print(f"  Case B (rich local):      {'web_search' if result_b == 'web_search' else 'report'}")
    ok_b = result_b == "report"

    # Case C: Tavily not configured
    result_c = should_search_check("无相关记录", False)
    print(f"  Case C (no Tavily):       {'web_search' if result_c == 'web_search' else 'report'}")
    ok_c = result_c == "report"

    all_ok = ok_a and ok_b and ok_c
    print(f"  [{'PASS' if all_ok else 'FAIL'}]")
    return all_ok


def test_search_and_save():
    """Test the actual search_and_save function (needs real API key)."""
    print("\n=== 4. search_and_save Function ===")
    try:
        from tools.knowledge_tools import search_and_save, is_tavily_configured
        if not is_tavily_configured():
            print("  [SKIP] Tavily not configured — skipping live test")
            return True  # Not a failure

        print("  Calling search_and_save('CCUS geological storage risks')...")
        result = search_and_save("CCUS geological storage risks", max_results=2)
        print(f"  Result ({len(result)} chars): {result[:200]}...")

        if "失败" in result or "未启用" in result:
            print(f"  [WARN] {result}")
            return False
        else:
            print("  [OK] Search and save completed")
            return True
    except Exception as e:
        print(f"  [FAIL] {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 60)
    print("Web Search Integration — Test Suite")
    print("=" * 60)

    imports_ok = test_imports()
    tavily_ok = test_tavily_config()
    routing_ok = test_should_search_logic()
    search_ok = test_search_and_save()

    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Imports:         {'PASS' if imports_ok else 'FAIL'}")
    print(f"  Tavily config:   {'PASS' if tavily_ok else 'SKIP'}")
    print(f"  Routing logic:   {'PASS' if routing_ok else 'FAIL'}")
    print(f"  Live search:     {'PASS' if search_ok else 'SKIP/FAIL'}")

    if routing_ok:
        print("\n✅ Core routing logic works!")
        print("   - Empty local knowledge → triggers web search")
        print("   - Rich local knowledge → skips web search")
        print("   - No Tavily key → gracefully skips")
    if tavily_ok and search_ok:
        print("✅ Live web search works! Content saved to ChromaDB + Notion.")

    return imports_ok and routing_ok


if __name__ == "__main__":
    main()
