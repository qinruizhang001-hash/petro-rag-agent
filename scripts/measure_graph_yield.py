"""
建图边际产出实验
================
问题：``ingest_ccus_batch.py`` 默认每篇只喂 4000 字符（摘要 + 引言），
全库覆盖率只有 6%。把预算提到全文，需要按实测吞吐跑约 4.3 小时。
**这 4.3 小时买得到什么？** 本脚本就是回答这个的。

做法：拿同一篇论文，在两个互相隔离的临时工作目录里各建一次图 ——
一次只喂前 ``--budget`` 个字符，一次喂全文 —— 然后比较：

- 实体数 / 关系数（绝对量）
- 每千字符的实体产出（**边际密度**：如果喂更多文字只是摊薄，这个数会下降）
- 孤立节点占比、连通分量数（图谱是变得更"连"还是更"散"）

为什么必须隔离：直接往 ``lightrag_storage`` 里插会污染现有 394 实体的图，
而且两次实验的实体会被合并去重，测出来的是增量不是产出。

用法::

    python scripts/measure_graph_yield.py --pdf ccus_docs/ccus_auto_16.pdf --budget 4000
    python scripts/measure_graph_yield.py --pdf ccus_docs/ccus_auto_16.pdf --budget 0   # 全文

``--budget 0`` 表示不截断。结果以 JSON 打到 stdout，便于两两对比。
注意：本脚本会真的调用 Ollama 做实体抽取，一篇 1 万字符的论文约 2 分钟。
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def extract_text(pdf_path: str) -> str:
    """与 ingest_ccus_batch 同一个抽取器（PyMuPDF 优先）。"""
    try:
        import pymupdf
    except ImportError:  # pragma: no cover
        import fitz as pymupdf  # type: ignore
    with pymupdf.open(pdf_path) as doc:
        return "\n".join(page.get_text("text") for page in doc)


def main() -> int:
    ap = argparse.ArgumentParser(description="建图边际产出实验（隔离工作目录）")
    ap.add_argument("--pdf", required=True, help="待实验的 PDF 路径")
    ap.add_argument("--budget", type=int, default=4000, help="喂入字符上限，0 表示全文")
    ap.add_argument("--keep", action="store_true", help="保留临时目录（排查用）")
    args = ap.parse_args()

    from utils.text_cleaning import clean_pdf_text, select_graph_text, strip_front_matter

    raw = extract_text(args.pdf)
    text = clean_pdf_text(raw)
    stripped = strip_front_matter(text)
    # 注意：select_graph_text 把 budget <= 0 当作"禁用"并返回空串，
    # 所以"全文"这条分支必须绕过它，否则会拿空文本去建图，
    # 测出来 0 实体，看起来像"全文反而更差"。
    body = select_graph_text(stripped, args.budget) if args.budget > 0 else stripped
    payload = f"Title: {os.path.basename(args.pdf)}\n\n{body}"

    workdir = tempfile.mkdtemp(prefix="graph_yield_")
    # 必须在 bridge 首次初始化之前改写模块全局，否则仍会写进真实存储。
    import lightrag_retriever

    lightrag_retriever.WORKING_DIR = workdir

    result = {
        "pdf": os.path.basename(args.pdf),
        "budget": args.budget,
        "extractable_chars": len(text),
        "payload_chars": len(payload),
        "workdir": workdir,
    }
    try:
        t0 = time.time()
        ok = lightrag_retriever.insert_text(
            payload, file_paths=[os.path.relpath(args.pdf, PROJECT_ROOT)]
        )
        result["seconds"] = round(time.time() - t0, 1)
        result["ok"] = ok
        if not ok:
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 1

        from utils.graph_analysis import summarize

        graphml = os.path.join(workdir, "graph_chunk_entity_relation.graphml")
        if os.path.exists(graphml):
            s = summarize(graphml)
            result.update(
                {
                    "nodes": s["nodes"],
                    "edges": s["edges"],
                    "components": s["components"],
                    "isolated": s["isolated"],
                    "largest": s["largest"],
                }
            )
            per_k = result["payload_chars"] / 1000
            result["nodes_per_1000_chars"] = round(s["nodes"] / per_k, 2) if per_k else 0
            result["edges_per_1000_chars"] = round(s["edges"] / per_k, 2) if per_k else 0
            result["isolated_ratio"] = (
                round(s["isolated"] / s["nodes"], 4) if s["nodes"] else 0
            )
        else:
            result["error"] = "graphml 未生成"
    finally:
        if not args.keep:
            shutil.rmtree(workdir, ignore_errors=True)
            result["workdir"] = "(removed)"

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
