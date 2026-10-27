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

检查什么
--------
前四条是**内部一致性**（不硬编码任何数字）：

1. badge 与所有 "**N unit tests" 声明的总数相同。
2. 所有 CI 计数声明（"1047 of them on every push" / "1047 in CI" /
   "CI runs 1047 of those tests" …）相同。
3. 分项之和等于 CI 计数（stdlib-only + ops-API + dependency-backed +
   gate/graph + chart/UI）。
4. "the other N need torch" 的 N 等于 总数 − CI 计数。

后两条是**与审计记录比对** —— 从"自相矛盾"升级到"与实测不符"：

5. README 里的归一化投影（4,988 → 4,415 节点 / 1,226 → 1,085 分量 /
   3,073 → 2,829 (64.1 %)）必须等于 `docs/graph-normalization.json`。
6. `eval/ablation.json` 的 alias_merge 臂必须与 `docs/graph-normalization.json`
   给出同一组数字 —— 这两个文件**不会同时过期**：改一条噪声规则后只重跑
   其中一个，另一个就停在旧值上，而它读起来同样斩钉截铁。

第 5 条上线时当场抓到一处陈旧值（README 写 4,419，实际 4,415）；
第 6 条上线时抓到 `eval/ablation.json` 还停在加 `funding-body` 规则之前的
4419 / 1087 / 64.0 %。

刻意不检查"这些数字是否等于真实测试数" —— 那需要在 CI 里跑一遍完整依赖
（torch/chromadb），与这个 job 存在的理由冲突。真实总数由每次全量测试的
输出负责；本脚本负责的是**不要自相矛盾，也不要与审计记录不符**。

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
    # 运维接口那一组必须单列：它需要 fastapi/httpx，不能进 stdlib-only job，
    # 也不该混进 dependency-backed（那边装的是 openai/ragas 那条评估链）。
    # 分组一多，分项之和最容易漏 —— 而这个脚本正是为了拦住"改了总数忘了分项"。
    ("ops-API", r"(\d+) ops-API"),
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


# ─── 两份审计记录必须说同一件事 ────────────────────────────
#
# `docs/graph-normalization.json` 与 `eval/ablation.json` 各自记录了同一次投影：
# 前者是归一化预演的结果，后者是消融里 alias_merge 那一组的 with_merge 臂。
# 两者都进仓库，都不需要图谱文件就能读，所以可以在 CI 里直接比对。
#
# 为什么值得：这两个文件**不会同时过期**。改一条噪声规则后只跑
# `normalize_graph.py`，README 和 normalization.json 都更新了，ablation.json
# 还停在旧数字上 —— 而它看起来同样斩钉截铁。2026-10 实际发生过：ablation
# 报 4419 / 1087 / 64.0 %，normalization 报 4415 / 1085 / 64.1 %，
# 两边都"有出处"，读者无法判断该信哪个。
_ABLATION_JSON = ROOT / "eval" / "ablation.json"


def check_audit_records(
    normalization_path: Path = _PROJECTION_JSON,
    ablation_path: Path = _ABLATION_JSON,
) -> str:
    """两份审计记录里的同一次投影必须逐字段相等。缺文件时跳过（不算失败）。"""
    if not normalization_path.exists() or not ablation_path.exists():
        return "跳过（缺少 graph-normalization.json 或 eval/ablation.json）"

    norm = json.loads(normalization_path.read_text(encoding="utf-8"))
    abl = json.loads(ablation_path.read_text(encoding="utf-8"))

    before = norm["before"]
    projected = norm["projected"]
    offline = abl.get("offline") or {}

    merge_arm = next(
        (a for a in offline.get("arms", []) if a.get("arm") == "alias_merge"), None
    )
    if merge_arm is None:
        raise SystemExit(
            "::error::eval/ablation.json 里没有 alias_merge 这一组，无法交叉核对。"
            "\n如果这一组被改名或删掉，请同步改本函数 —— 留着一条永不触发的检查，"
            "等于没有检查。"
        )

    with_merge = merge_arm.get("with_merge") or {}
    problems: list[str] = []

    for label, got, expect in (
        ("图规模", (offline.get("nodes"), offline.get("edges")),
         (before["nodes"], before["edges"])),
        ("合并后节点", with_merge.get("nodes"), projected["nodes"]),
        ("合并后分量", with_merge.get("components"), projected["components"]),
        ("合并后最大分量", with_merge.get("largest"), projected["largest"]),
    ):
        if got != expect:
            problems.append(f"{label}：ablation={got}，normalization={expect}")

    if problems:
        raise SystemExit(
            "::error::两份审计记录对同一次投影给出了不同的数字：\n"
            + "\n".join("  - " + p for p in problems)
            + "\n修法：两个脚本都重跑一遍（先 normalize_graph.py，再 "
              "scripts/ablation.py --offline --write），不要只跑其中一个。"
        )
    return "ablation.json 与 normalization.json 一致"


def main(argv: list[str]) -> int:
    path = Path(argv[1]) if len(argv) > 1 else ROOT / "README.md"
    text = path.read_text(encoding="utf-8")
    summary = check(text)
    projection = check_projection(text, _PROJECTION_JSON)
    records = check_audit_records()
    print(f"README numbers OK: {summary}")
    print(f"README projection OK: {projection}")
    print(f"Audit records OK: {records}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
