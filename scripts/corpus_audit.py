"""
语料覆盖率与图谱健康度审计
==========================
回答一个**具体**问题：知识库里到底有多少语料？

不是"有多少篇论文"。23 篇这个数字会让人以为覆盖了 23 篇全文，实际上
``ingest_ccus_batch.py`` 每篇只喂 ``--graph-budget``（默认 4000）个字符，
也就是摘要 + 引言。剩下的方法、结果、讨论、数据表全都没进图谱。

所以本脚本的核心指标是**字符覆盖率**：

    覆盖率 = LightRAG 实际收到的字符 / PDF 里可抽取的字符

判定口径：
- 可抽取字符：PyMuPDF 从每个 PDF 抽到的文本长度（与摄取脚本同一个抽取器）
- 已入库字符：``ccus_docs/graph_ingest_state.json`` 里记录的 ``chars``
  —— 这是**真实喂进去的长度**，不是估算，摄取脚本逐篇写下的
- 抽取吞吐：同一份状态文件里的 ``seconds``，即本机实测的 chars/秒

为什么不用 ``kv_store_full_docs.json`` 算覆盖：那份存储里
``file_path`` 全是 ``unknown_source``（见 lineage 一节），无法可靠地
把 doc_id 映射回 PDF；用摄取状态文件反而是精确的。

用法::

    python scripts/corpus_audit.py            # 人类可读表格
    python scripts/corpus_audit.py --json     # 只输出 JSON（给 CI / 报告用）
    python scripts/corpus_audit.py --write    # 同时写 docs/corpus-audit.json

退出码：默认恒为 0（诊断工具）。加 ``--check`` 才变成门禁：覆盖率、
血缘可追溯率、实体数三项任一不达标就退出 1，用于**重建之后**的自检。

不能放进 CI：``lightrag_storage/`` 在 .gitignore 里（几十 MB 的抽取产物
不该进 git），托管 runner 上没有可审的数据。所以这是本地/部署侧的门禁，
不是 push 门禁。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sqlite3
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DOCS_DIR = os.path.join(PROJECT_ROOT, "ccus_docs")
LIGHTRAG_DIR = os.path.join(PROJECT_ROOT, "lightrag_storage")
CHROMA_DB = os.path.join(PROJECT_ROOT, "chroma_petro", "chroma.sqlite3")
GRAPH_STATE = os.path.join(DOCS_DIR, "graph_ingest_state.json")
GRAPHML = os.path.join(LIGHTRAG_DIR, "graph_chunk_entity_relation.graphml")
AUDIT_JSON = os.path.join(PROJECT_ROOT, "docs", "corpus-audit.json")

# 摄取脚本里的默认预算。写死会漂移，所以运行时从 argparse 定义里读。
INGEST_SCRIPT = os.path.join(PROJECT_ROOT, "ingest_ccus_batch.py")


# ─── 一、语料：可抽取 vs 已入库 ─────────────────────────────


def _extractable_chars(pdf_path: str) -> tuple[int, int]:
    """返回 (页数, 可抽取字符数)。抽不出来时返回 (0, 0)。"""
    try:
        import pymupdf  # 新版包名
    except ImportError:  # pragma: no cover - 老版本回退
        try:
            import fitz as pymupdf  # type: ignore
        except ImportError:
            return 0, 0
    try:
        with pymupdf.open(pdf_path) as doc:
            return doc.page_count, sum(len(p.get_text()) for p in doc)
    except Exception:
        return 0, 0


def load_graph_state() -> dict:
    if not os.path.exists(GRAPH_STATE):
        return {}
    try:
        with open(GRAPH_STATE, encoding="utf-8") as fh:
            return json.load(fh).get("papers", {})
    except (json.JSONDecodeError, OSError):
        return {}


def corpus_coverage() -> dict:
    state = load_graph_state()
    rows = []
    for path in sorted(glob.glob(os.path.join(DOCS_DIR, "*.pdf"))):
        name = os.path.basename(path)
        pages, extractable = _extractable_chars(path)
        rec = state.get(name) or {}
        ingested = int(rec.get("chars") or 0)
        seconds = rec.get("seconds")
        rows.append(
            {
                "file": name,
                "pages": pages,
                "extractable": extractable,
                "ingested": ingested,
                "coverage": round(ingested / extractable, 4) if extractable else 0.0,
                "seconds": seconds,
                # 本机实测吞吐（chars/s）。None 表示这篇没在状态文件里。
                "chars_per_sec": round(ingested / seconds, 1) if seconds else None,
                "in_graph": bool(rec),
            }
        )
    tot_ex = sum(r["extractable"] for r in rows)
    tot_in = sum(r["ingested"] for r in rows)
    tot_s = sum(r["seconds"] for r in rows if r["seconds"])
    return {
        "files": rows,
        "files_total": len(rows),
        "files_in_graph": sum(1 for r in rows if r["in_graph"]),
        "pages_total": sum(r["pages"] for r in rows),
        "extractable_total": tot_ex,
        "ingested_total": tot_in,
        "coverage": round(tot_in / tot_ex, 4) if tot_ex else 0.0,
        "extract_seconds_total": round(tot_s, 1),
        # 真实的抽取吞吐。旧注释里的 "31 秒/千字符" 是 CPU 时代的数字，
        # 这里用实际跑出来的秒数反算，避免继续引用过期常数。
        "sec_per_1000_chars": round(tot_s / (tot_in / 1000), 2) if tot_in and tot_s else None,
    }


def projection(cov: dict) -> list[dict]:
    """不同 budget 下的覆盖率与所需时间（按实测吞吐线性外推）。"""
    per_k = cov.get("sec_per_1000_chars")
    total = cov["extractable_total"]
    n = max(1, cov["files_total"])
    out = []
    for budget in (4000, 8000, 12000, 20000, 0):  # 0 = 全文
        if budget == 0:
            chars = total
            label = "全文（无预算）"
        else:
            chars = min(budget * n, total)
            label = f"每篇 {budget} 字符"
        out.append(
            {
                "label": label,
                "chars": chars,
                "coverage": round(chars / total, 4) if total else 0.0,
                "hours": round(chars / 1000 * per_k / 3600, 2) if per_k else None,
            }
        )
    return out


# ─── 二、chunk 粒度 ──────────────────────────────────────────


def chunk_stats() -> dict:
    path = os.path.join(LIGHTRAG_DIR, "kv_store_text_chunks.json")
    if not os.path.exists(path):
        return {"count": 0}
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    lens, per_doc = [], {}
    for v in data.values():
        text = v.get("content", "") if isinstance(v, dict) else str(v)
        lens.append(len(text))
        doc = (v.get("full_doc_id") or "?") if isinstance(v, dict) else "?"
        per_doc[doc] = per_doc.get(doc, 0) + 1
    lens.sort()
    return {
        "count": len(lens),
        "chars_total": sum(lens),
        "chars_mean": round(sum(lens) / len(lens)) if lens else 0,
        "chars_min": lens[0] if lens else 0,
        "chars_max": lens[-1] if lens else 0,
        "per_doc_min": min(per_doc.values()) if per_doc else 0,
        "per_doc_max": max(per_doc.values()) if per_doc else 0,
        "docs": len(per_doc),
    }


# ─── 三、图谱健康度 ─────────────────────────────────────────


def graph_health() -> dict:
    if not os.path.exists(GRAPHML):
        return {"exists": False}
    from utils.graph_analysis import summarize

    s = summarize(GRAPHML)
    s["exists"] = True
    total = s["nodes"] or 1
    s["isolated_ratio"] = round(s["isolated"] / total, 4)
    # 平均度：2E/N。孤立节点多、平均度低 = 抽出来的实体彼此不说话。
    s["mean_degree"] = round(2 * s["edges"] / total, 2)
    return s


# ─── 四、数据血缘 ───────────────────────────────────────────


def lineage() -> dict:
    path = os.path.join(LIGHTRAG_DIR, "kv_store_full_docs.json")
    if not os.path.exists(path):
        return {"docs": 0}
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    paths = [
        (v.get("file_path") or "") if isinstance(v, dict) else ""
        for v in data.values()
    ]
    unknown = sum(1 for p in paths if not p or p == "unknown_source")
    return {
        "docs": len(paths),
        "unknown_source": unknown,
        "traceable": len(paths) - unknown,
        "traceable_ratio": round((len(paths) - unknown) / len(paths), 4) if paths else 0.0,
    }


# ─── 五、向量库冗余 ─────────────────────────────────────────


def vector_stores() -> dict:
    if not os.path.exists(CHROMA_DB):
        return {"exists": False, "collections": []}
    con = sqlite3.connect(CHROMA_DB)
    try:
        rows = list(
            con.execute(
                """
                select co.name, count(e.id)
                from collections co
                left join segments s on s.collection = co.id
                left join embeddings e on e.segment_id = s.id
                group by co.name order by co.name
                """
            )
        )
    except sqlite3.Error:
        rows = []
    finally:
        con.close()

    # 哪些 collection 名字在代码里真的出现过 —— 没出现的就是死数据。
    code = []
    for root, dirs, files in os.walk(PROJECT_ROOT):
        dirs[:] = [d for d in dirs if d not in {".git", "__pycache__", "models", "node_modules", ".venv"}]
        for fn in files:
            if fn.endswith(".py"):
                try:
                    with open(os.path.join(root, fn), encoding="utf-8", errors="ignore") as fh:
                        code.append(fh.read())
                except OSError:
                    pass
    blob = "\n".join(code)

    collections = []
    for name, count in rows:
        collections.append(
            {
                "name": name,
                "vectors": count,
                # 精确到带引号的字面量，避免 petro_knowledge 命中 petro_knowledge_v3
                "referenced": bool(re.search(rf"['\"]{re.escape(name)}['\"]", blob)),
            }
        )
    return {
        "exists": True,
        "collections": collections,
        "vectors_total": sum(c["vectors"] for c in collections),
        "dead_collections": [c["name"] for c in collections if not c["referenced"]],
        "dead_vectors": sum(c["vectors"] for c in collections if not c["referenced"]),
    }


# ─── 六、当前预算（从摄取脚本读，不复制常数） ────────────────


def current_budget() -> int | None:
    if not os.path.exists(INGEST_SCRIPT):
        return None
    with open(INGEST_SCRIPT, encoding="utf-8", errors="ignore") as fh:
        text = fh.read()
    m = re.search(r'"--graph-budget"[^)]*?default=(\d+)', text, re.S)
    return int(m.group(1)) if m else None


# ─── 输出 ───────────────────────────────────────────────────


def audit() -> dict:
    cov = corpus_coverage()
    return {
        "corpus": cov,
        "projection": projection(cov),
        "chunks": chunk_stats(),
        "graph": graph_health(),
        "lineage": lineage(),
        "vectors": vector_stores(),
        "graph_budget_default": current_budget(),
    }


def render(a: dict) -> str:
    cov = a["corpus"]
    lines = []
    lines.append("═" * 78)
    lines.append("语料覆盖率")
    lines.append("═" * 78)
    lines.append(f"{'文件':22} {'页':>4} {'可抽取':>9} {'已入库':>8} {'覆盖':>7} {'秒':>6}")
    for r in cov["files"]:
        lines.append(
            f"{r['file']:22} {r['pages']:>4} {r['extractable']:>9,} "
            f"{r['ingested']:>8,} {r['coverage']:>6.1%} "
            f"{(r['seconds'] or 0):>6.0f}"
        )
    lines.append("-" * 78)
    lines.append(
        f"{'合计':22} {cov['pages_total']:>4} {cov['extractable_total']:>9,} "
        f"{cov['ingested_total']:>8,} {cov['coverage']:>6.1%}"
    )
    lines.append(
        f"  在库论文 {cov['files_in_graph']}/{cov['files_total']} 篇 · "
        f"实测吞吐 {cov['sec_per_1000_chars']} 秒/千字符"
    )

    lines.append("")
    lines.append("═" * 78)
    lines.append("如果把预算调高，代价是多少（按实测吞吐线性外推）")
    lines.append("═" * 78)
    lines.append(f"{'方案':22} {'字符':>10} {'覆盖':>7} {'预计耗时':>10}")
    for p in a["projection"]:
        hrs = f"{p['hours']:.1f} 小时" if p["hours"] is not None else "—"
        lines.append(f"{p['label']:22} {p['chars']:>10,} {p['coverage']:>6.1%} {hrs:>10}")

    ch = a["chunks"]
    lines.append("")
    lines.append("═" * 78)
    lines.append("chunk 粒度")
    lines.append("═" * 78)
    if ch["count"]:
        lines.append(
            f"{ch['count']} 个 chunk / {ch['docs']} 篇 → 每篇 {ch['per_doc_min']}–{ch['per_doc_max']} 个；"
            f"长度 均 {ch['chars_mean']}，范围 {ch['chars_min']}–{ch['chars_max']}"
        )
    else:
        lines.append("无 chunk 数据")

    g = a["graph"]
    lines.append("")
    lines.append("═" * 78)
    lines.append("图谱健康度")
    lines.append("═" * 78)
    if g.get("exists"):
        lines.append(
            f"{g['nodes']} 实体 / {g['edges']} 关系 · {g['components']} 个连通分量 · "
            f"最大 {g['largest']} ({g['largest_ratio']:.1%})"
        )
        lines.append(
            f"孤立节点 {g['isolated']} ({g['isolated_ratio']:.1%}) · 平均度 {g['mean_degree']}"
        )
        top = list(g["types"].items())[:6]
        lines.append("类型分布: " + " · ".join(f"{k}={v}" for k, v in top))
    else:
        lines.append("graphml 不存在")

    li = a["lineage"]
    lines.append("")
    lines.append("═" * 78)
    lines.append("数据血缘")
    lines.append("═" * 78)
    if li.get("docs"):
        verdict = "✅ 可追溯" if li["unknown_source"] == 0 else "❌ 不可追溯"
        lines.append(
            f"{li['docs']} 篇文档中 {li['unknown_source']} 篇 file_path 缺失/unknown_source "
            f"→ {verdict}（可追溯率 {li['traceable_ratio']:.0%}）"
        )

    v = a["vectors"]
    lines.append("")
    lines.append("═" * 78)
    lines.append("向量库")
    lines.append("═" * 78)
    if v.get("exists"):
        for c in v["collections"]:
            mark = "" if c["referenced"] else "   ← 代码里未引用（死数据）"
            lines.append(f"{c['name']:26} {c['vectors']:>6} 向量{mark}")
        lines.append(
            f"合计 {v['vectors_total']} 向量；未被引用 {len(v['dead_collections'])} 个集合 / "
            f"{v['dead_vectors']} 向量"
        )
    lines.append("")
    lines.append(f"当前 --graph-budget 默认值：{a['graph_budget_default']}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="语料覆盖率与图谱健康度审计")
    ap.add_argument("--json", action="store_true", help="只输出 JSON")
    ap.add_argument("--write", action="store_true", help=f"同时写入 {AUDIT_JSON}")
    ap.add_argument("--check", action="store_true",
                    help="按阈值判定（覆盖率/血缘/实体数），不达标退出 1")
    ap.add_argument("--min-coverage", type=float, default=0.95)
    ap.add_argument("--min-lineage", type=float, default=1.0)
    ap.add_argument("--min-entities", type=int, default=1000)
    args = ap.parse_args()

    a = audit()
    if args.write:
        os.makedirs(os.path.dirname(AUDIT_JSON), exist_ok=True)
        with open(AUDIT_JSON, "w", encoding="utf-8") as fh:
            json.dump(a, fh, ensure_ascii=False, indent=2)
        if not args.json:
            print(f"[written] {AUDIT_JSON}\n")
    print(json.dumps(a, ensure_ascii=False, indent=2) if args.json else render(a))

    if not args.check:
        return 0

    # 三项各自对应一类"不会报错但会静默降质"的失败：
    # 覆盖率低 = 建图被限流（曾经 6.2%）；血缘低 = 报告无法回溯到 PDF；
    # 实体太少 = 抽取大面积失败，但流程照样返回成功。
    checks = [
        ("覆盖率", a["corpus"]["coverage"], args.min_coverage, "≥"),
        ("血缘可追溯率", a["lineage"].get("traceable_ratio", 0.0), args.min_lineage, "≥"),
        ("实体数", a["graph"].get("nodes", 0), args.min_entities, "≥"),
    ]
    failed = []
    print("\n" + "─" * 78)
    for name, got, want, op in checks:
        ok = got >= want
        shown = f"{got:.1%}" if isinstance(got, float) and got <= 1 else f"{got:,}"
        target = f"{want:.0%}" if isinstance(want, float) and want <= 1 else f"{want:,}"
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {shown} (要求 {op} {target})")
        if not ok:
            failed.append(name)
    print("─" * 78)
    if failed:
        print(f"❌ 未达标：{'、'.join(failed)}")
        return 1
    print("✅ 全部达标")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
