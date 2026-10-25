"""核对 README 里那几处互相引用的测试数字是否自洽。

为什么需要它
------------
README 用五处声明同一个测试总数（badge / highlights 表 / §3 正文 /
"Decoupled by construction" / 文末 checklist）。2026 年之前它们分别写着
1080、1106、989 —— 三个不同的数，并存了几个月。没有一条断言在核对，
所以没有任何东西会失败；而 989 本身也是错的，它漏算了图表 job 的 46 项。

这就是"文档不会自己保持一致"的具体形态：每一处单独看都像是抄对了，
错在它们抄的不是同一次测量。人工复核解决不了，因为它没有触发条件；
断言才有。

检查什么（全部是**内部一致性**，不硬编码任何数字）
--------------------------------------------------
1. badge 与所有 "**N unit tests" 声明的总数相同。
2. 所有 CI 计数声明（"1047 of them on every push" / "1047 in CI" /
   "CI runs 1047 of those tests" …）相同。
3. 分项之和等于 CI 计数：906 + 78 + 17 + 46 = 1047。
4. "the other N need torch" 的 N 等于 总数 − CI 计数。

刻意不检查"这些数字是否等于真实测试数" —— 那需要在 CI 里跑一遍完整依赖
（torch/chromadb），与这个 job 存在的理由冲突。真实总数由每次全量测试的
输出负责，README 负责的是**不要自相矛盾**。

用法::

    python scripts/check_readme_numbers.py            # 默认读 README.md
    python scripts/check_readme_numbers.py path.md
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# (标签, 正则, 期望值来源)  —— 期望值来源在下面由 _agree() 动态决定
_TOTAL_PATTERNS = [
    ("badge", r"tests-(\d+)%20unit"),
    ("highlights/§3/checklist", r"\*\*(\d+) unit tests"),
]

_CI_PATTERNS = [
    ("highlights", r"(\d+) of them on every push"),
    ("§3", r"(\d+) run on every push"),
    ("Decoupled by construction", r"(\d+) in CI"),
    ("Production Ready", r"(\d+) on every push"),
    ("checklist", r"CI runs (\d+) of those tests"),
]

_BREAKDOWN_PATTERNS = [
    ("stdlib-only", r"(\d+) stdlib-only"),
    ("dependency-backed", r"(\d+) dependency-backed"),
    ("gate/graph", r"(\d+) gate/graph"),
    ("chart/UI", r"(\d+) chart/UI"),
]

_LOCAL_PATTERN = r"the other (\d+) need torch"


def _collect(text: str, patterns: list[tuple[str, str]]) -> dict[str, set[int]]:
    """按标签收集所有命中值；某个声明整体消失也算错（说明检查已经失效）。"""
    found: dict[str, set[int]] = {}
    for label, pattern in patterns:
        values = {int(m) for m in re.findall(pattern, text)}
        if not values:
            raise SystemExit(
                f"::error::README 里找不到「{label}」的测试数声明（正则 {pattern}）。"
                f"如果这处声明被删掉了，请同步删掉本脚本里的对应条目 —— "
                f"留着一条永不触发的检查，等于没有检查。"
            )
        found[label] = values
    return found


def _agree(label: str, found: dict[str, set[int]]) -> int:
    """所有位置必须说同一个数。"""
    distinct: set[int] = set()
    for values in found.values():
        distinct |= values
    if len(distinct) != 1:
        detail = "；".join(f"{k}={sorted(v)}" for k, v in found.items())
        raise SystemExit(f"::error::README 的{label}互相矛盾：{detail}")
    return distinct.pop()


def check(text: str) -> str:
    total = _agree("测试总数", _collect(text, _TOTAL_PATTERNS))
    in_ci = _agree("CI 测试数", _collect(text, _CI_PATTERNS))
    parts = _collect(text, _BREAKDOWN_PATTERNS)

    if sum(v for values in parts.values() for v in values) != in_ci:
        detail = " + ".join(
            f"{k}={sorted(v)}" for k, v in parts.items()
        )
        raise SystemExit(
            f"::error::README 的分项之和 ≠ CI 计数（{detail} ≠ {in_ci}）。"
            f"改了总数却忘了改分项，是上一次 989 的来源。"
        )

    local = _collect(text, [("local-only", _LOCAL_PATTERN)])
    local_n = _agree("本地测试数", local)
    if local_n != total - in_ci:
        raise SystemExit(
            f"::error::README 说 {local_n} 项只在本地跑，但 {total} − {in_ci} = "
            f"{total - in_ci}。"
        )

    return f"{total} total · {in_ci} in CI · {local_n} local-only · 分项自洽"


def main(argv: list[str]) -> int:
    path = Path(argv[1]) if len(argv) > 1 else ROOT / "README.md"
    summary = check(path.read_text(encoding="utf-8"))
    print(f"README numbers OK: {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
