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

import json
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


# ─── 图谱归一化投影数字 ────────────────────────────────────
#
# 这一组不是"内部一致性"，而是**与审计记录比对**：README 里那几处
# 「4,988 → 4,415 节点 / 1,226 → 1,085 分量」必须等于 docs/graph-normalization.json
# 里存的值。JSON 由 `python normalize_graph.py` 生成，是这次预演的原始记录。
#
# 为什么值得单独一条：这几处数字在 README 里出现过三个不同版本（4,419 / 4,415），
# 全都没人发现 —— 因为没有任何东西在核对它们。规则改一条、语料加一篇，
# 投影就变，而 README 里那句话读起来依然斩钉截铁。
_PROJECTION_JSON = ROOT / "docs" / "graph-normalization.json"


def _fmt(n: int) -> str:
    return f"{n:,}"


def check_projection(text: str, json_path: Path) -> str:
    """
    README 的投影数字 vs 归一化报告的 JSON。JSON 不存在时跳过（不算失败）。

    "找不到"要算失败而不是跳过：如果哪天这几句话被改写成了别的形式，
    这个检查会静默变成永不触发的空壳 —— 与 ``_collect`` 里的理由相同。
    """
    if not json_path.exists():
        return f"跳过（{json_path.name} 不存在，先跑 python normalize_graph.py）"

    data = json.loads(json_path.read_text(encoding="utf-8"))
    before = data["before"]
    projected = data["projected"]

    probes = [
        ("节点", f"{_fmt(before['nodes'])} → ", r"([\d,]+) nodes", projected["nodes"]),
        ("连通分量", f"{_fmt(before['components'])} → ",
         r"([\d,]+) components", projected["components"]),
        # 最大分量后面必须跟一个百分比。这条限定不是凑格式，是为了避开另一处
        # 恰好也以 "3,073 → " 开头的数字：README 里还有一句
        # "enforcing the type constraint would take the largest component
        #  3,073 → 2,759 (−314)"，那是个**反事实**（不执行），不是这次投影。
        # 只按前缀匹配会把它当成投影，报一个假的不一致。
        ("最大分量", f"{_fmt(before['largest'])} → ",
         r"([\d,]+) \(([\d.]+) %\)", projected["largest"]),
    ]

    problems: list[str] = []
    for label, prefix, tail, expect in probes:
        pattern = re.escape(prefix) + tail
        # 有的正则有捕获组（最大分量那条要顺带锁住百分比），findall 会返回
        # 元组；统一只取第一组，也就是"箭头右边的数"。
        found = {
            int((m[0] if isinstance(m, tuple) else m).replace(",", ""))
            for m in re.findall(pattern, text)
        }
        if not found:
            problems.append(f"{label}：README 里找不到「{prefix}…」形式的投影数字")
        elif found != {expect}:
            problems.append(
                f"{label}：README 说 {sorted(found)}，JSON 记录 {expect}"
            )

    if problems:
        raise SystemExit(
            "::error::README 的归一化投影数字与 docs/graph-normalization.json 不一致：\n"
            + "\n".join("  - " + p for p in problems)
            + "\n修法：重跑 python normalize_graph.py，然后按新的 JSON 改 README。"
            "\n不要手算 —— 手算出来的数和 JSON 对不上时，没人知道该信哪个。"
        )
    return "与 graph-normalization.json 一致"


def main(argv: list[str]) -> int:
    path = Path(argv[1]) if len(argv) > 1 else ROOT / "README.md"
    text = path.read_text(encoding="utf-8")
    summary = check(text)
    projection = check_projection(text, _PROJECTION_JSON)
    print(f"README numbers OK: {summary}")
    print(f"README projection OK: {projection}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
