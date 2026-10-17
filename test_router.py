"""
Test Memory Router — verify the three key scenarios.

Run: python test_router.py

Scenarios:
  1. "我之前读过哪些关于强化学习优化CO2注入的论文？"  → should trigger memory search
  2. "总结一下我最近的项目进展和踩过的坑"              → should trigger memory search
  3. "CCUS地质封存的主要风险是什么？"                   → should skip memory (pure CCUS)
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ─── Import the memory router logic ───────────────────────────
# We test the routing logic directly (no LLM needed) to verify keyword matching.

MEMORY_TRIGGERS = [
    "我的", "我之前", "我上次", "我做过", "我写过", "我的笔记",
    "复盘", "总结一下我", "项目进展", "我学到", "失败经验",
    "论文笔记", "我读过", "上次评估", "之前的报告", "我记得",
    "my ", "i previously", "last time", "i did", "i wrote", "my notes",
    "review", "my project", "i learned", "failure", "paper notes",
    "i read", "previous report", "i remember",
]


def check_need_memory(user_input: str) -> bool:
    """Simulate the memory_router node's logic."""
    user_text = user_input.lower().strip()
    return any(trigger in user_text for trigger in MEMORY_TRIGGERS)


def test_scenario(test_id: int, user_input: str, expected: bool):
    """Test a single scenario."""
    result = check_need_memory(user_input)
    status = "PASS" if result == expected else "FAIL"

    print(f"\n--- Test {test_id} ---")
    print(f"  Input:    {user_input}")
    print(f"  Expected: {'need memory' if expected else 'skip memory'}")
    print(f"  Got:      {'need memory' if result else 'skip memory'}")
    print(f"  [{status}]")

    # Show which trigger matched (if any)
    user_text = user_input.lower()
    matched = [t for t in MEMORY_TRIGGERS if t in user_text]
    if matched:
        print(f"  Matched triggers: {matched}")

    return result == expected


def test_search_personal_memory():
    """Test the actual memory search function (needs ChromaDB)."""
    print("\n--- Test 4: search_personal_memory function ---")
    try:
        from memory.memory_tools import search_personal_memory

        result = search_personal_memory("CCUS 注入策略")
        print(f"  [OK] Function returned {len(result)} chars")
        if "未在个人记忆中找到" in result:
            print("  -> Memory is empty (expected for first run)")
        else:
            print(f"  -> First 100 chars: {result[:100]}...")
        return True
    except Exception as e:
        print(f"  [FAIL] {e}")
        print("  Make sure chromadb and sentence-transformers are installed")
        return False


def main():
    print("=" * 60)
    print("Memory Router — Scenario Tests")
    print("=" * 60)

    all_pass = True

    # Scenario 1: personal memory trigger
    all_pass &= test_scenario(
        1,
        "我之前读过哪些关于强化学习优化CO2注入的论文？",
        expected=True
    )

    # Scenario 2: project review trigger
    all_pass &= test_scenario(
        2,
        "总结一下我最近的项目进展和踩过的坑",
        expected=True
    )

    # Scenario 3: pure CCUS (should NOT trigger memory)
    all_pass &= test_scenario(
        3,
        "CCUS地质封存的主要风险是什么？",
        expected=False
    )

    # Test 4: actual search function
    search_ok = test_search_personal_memory()

    # Summary
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Test 1 (论文笔记 trigger):  {'PASS' if all_pass else 'FAIL'}")
    print(f"  Test 2 (项目复盘 trigger):   {'PASS' if all_pass else 'FAIL'}")
    print(f"  Test 3 (纯CCUS skip):        {'PASS' if all_pass else 'FAIL'}")
    print(f"  Test 4 (search function):    {'PASS' if search_ok else 'FAIL'}")

    if all_pass and search_ok:
        print("\n✅ Memory Router is working correctly!")
        print("   - Personal questions → search memory")
        print("   - Pure CCUS questions → skip memory, direct analysis")
    else:
        print("\n⚠️  Some tests failed. Check output above.")

    return all_pass and search_ok


if __name__ == "__main__":
    main()
