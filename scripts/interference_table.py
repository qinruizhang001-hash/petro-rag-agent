#!/usr/bin/env python3
"""
生成「井间干扰参考表」（`docs/gap-register.md` §3 W7 里那张表）

为什么需要它
------------
2026-10-26 那张表的每个数都写死在文档里，没有任何命令能重新算出来。
本轮核对时发现 8 井 / 300 m 那一格文档写 **84.12**，当前代码给 **82.76** ——
所有格子统一偏低 1.6 %，反推过去是密度取 789.96 kg/m³ 而不是现在的
777.20 kg/m³（`_density(p_h, 60 ℃)`）。也就是说：那张表是**某一次**算出来的，
算它的那个状态没有被记录下来，后来密度口径一变，它就成了历史文物。

这类数字的危害不在于偏 1.6 %，而在于**没人能发现它偏了**。表格旁边还写着
「重要更正」「偏差方向」这类关于诚实的段落，而表本身无法复算。

所以：表格由本脚本产出，文档里放生成的块，测试逐字比对。
改了模型导致数字变化时，`python scripts/interference_table.py --write`
重新生成，diff 会告诉审阅者「物理变了」而不是「有人在手改文档」。

用法::

    python scripts/interference_table.py            # 打印表格
    python scripts/interference_table.py --write    # 就地更新文档里的生成块

退出码：0 = 正常；1 = 文档里找不到生成块（标记被删了）。
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.optimize import optimize_plan  # noqa: E402

DOC = ROOT / "docs" / "gap-register.md"

# 生成块的标记。文档里必须成对出现，否则 --write 会拒绝改动。
BEGIN = "<!-- BEGIN generated: interference_table -->"
END = "<!-- END generated: interference_table -->"

# 参考算例：2500 m 埋深、380 bar 井底压力、5 % 风险预算。
DEPTH_M = 2500.0
PRESSURE_BAR = 380.0
BUDGET = 0.05
SAMPLES = 200
SEED = 7
WELL_COUNTS = (2, 4, 8)
SPACINGS = (None, 300.0, 150.0)


def _rates(spacing: float | None) -> tuple[dict[int, float], float]:
    plan = optimize_plan(
        DEPTH_M, PRESSURE_BAR, n=SAMPLES, seed=SEED, well_spacing_m=spacing
    )
    rates = {o["n_wells"]: o["rate_mt_per_year"] for o in plan["well_options"]}
    return rates, plan["recommended"]["pressure_bar"]


def render() -> str:
    columns = [_rates(s) for s in SPACINGS]
    headers = ["无干扰" if s is None else f"井距 {s:g} m" for s in SPACINGS]

    lines = [
        BEGIN,
        f"参考算例：`optimize_plan({DEPTH_M:g}, {PRESSURE_BAR:g}, "
        f"n={SAMPLES}, seed={SEED}, well_spacing_m=…)`。",
        "",
        "| 方案 | " + " | ".join(headers) + " |",
        "|---|" + "---:|" * len(headers),
    ]
    for n in WELL_COUNTS:
        cells = [f"{rates.get(n, 0.0):.2f}" for rates, _ in columns]
        lines.append(f"| {n} 井 | " + " | ".join(cells) + " |")
    pressures = [f"{p:.0f} bar" for _, p in columns]
    lines.append("| 推荐压力 | " + " | ".join(pressures) + " |")
    lines.append(END)
    return "\n".join(lines)


def replace_block(text: str, block: str) -> str:
    pattern = re.compile(
        re.escape(BEGIN) + r".*?" + re.escape(END), re.DOTALL
    )
    if not pattern.search(text):
        raise SystemExit(
            f"::error::{DOC.relative_to(ROOT)} 里找不到生成块标记 {BEGIN} … {END}。"
            "标记被删掉时本脚本不会盲目插入 —— 那会让表格出现在错误的位置。"
        )
    return pattern.sub(lambda _: block, text, count=1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生成井间干扰参考表")
    parser.add_argument("--write", action="store_true", help="就地更新文档")
    args = parser.parse_args(argv)

    block = render()
    if not args.write:
        print(block)
        return 0

    updated = replace_block(DOC.read_text(encoding="utf-8"), block)
    DOC.write_text(updated, encoding="utf-8")
    print(f"已更新 {DOC.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
