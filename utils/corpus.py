"""
语料范围的定义 —— 单一事实来源。

为什么单独一个模块：``ingest_ccus_batch.py`` 和 ``scripts/corpus_audit.py``
必须对"哪些 PDF 属于语料"给出一致的答案。曾经不一致：入库侧排除了
``ccus4.pdf``（内容是电信 NOMA 上行功率控制，与 CCUS 无关），审计侧却仍然
用 ``glob("*.pdf")`` 把它算进分母，于是覆盖率永远显示 96.9%，看起来像
"还有 3% 没喂进去"，实际上那 3% 是**故意不喂**的。

这种偏差不会报错，只会让所有覆盖率数字失去意义 —— 一个永远差 3% 的指标，
没人会去追，也就没人会发现真正的覆盖率掉到 6% 时它也在显示 96.9%。

放在 utils/ 下而不是 ingest 脚本里，是因为审计脚本要能在**不导入 LightRAG**
（会拖进 torch / chromadb）的前提下使用它。本模块只依赖标准库。
"""
from __future__ import annotations

# 文件名 -> 排除原因。必须是显式清单：不允许用"看起来不像 CCUS"这类模糊规则，
# 否则"这篇为什么没进图谱"就变成一个查不出来的问题。
EXCLUDED_PAPERS: dict[str, str] = {
    "ccus4.pdf": "实际内容是 NOMA 上行功率控制（电信），与 CCUS 无关",
}


def is_excluded(filename: str) -> bool:
    """该 PDF 是否被排除在语料之外。"""
    return filename in EXCLUDED_PAPERS


def exclusion_reason(filename: str) -> str:
    return EXCLUDED_PAPERS.get(filename, "")


def count_included(pdfs: list[str]) -> int:
    """统计语料内的 PDF 数量（``pdfs`` 可传路径或文件名）。"""
    import os

    return sum(1 for p in pdfs if not is_excluded(os.path.basename(p)))
