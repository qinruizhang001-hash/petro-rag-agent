"""
CCUS 文献批量入库（ChromaDB + LightRAG）
========================================
把 ``ccus_docs/`` 下的 PDF 清洗、切片，增量写入两个知识存储。

两层为什么要分开对待
--------------------
这两步的成本差三个数量级，混在一起写必然做错取舍：

  ① **ChromaDB（向量检索）—— 便宜。** 只做 embedding，``bge-small-zh-v1.5``
     在 CPU 上批量推理很快。所以**全量入库**，保留完整正文，切片 1200 字符。
  ② **LightRAG（知识图谱）—— 贵。** 每个 chunk 都要跑一次 LLM 实体抽取，
     实测 qwen2.5:7b 在 CPU 上约 **31 秒 / 1000 字符**。20 篇论文全文约
     100 万字符 = 8 小时以上，不现实。所以每篇只喂开头 ``--graph-budget``
     字符（标题 + 摘要 + 引言），这一段决定"这篇论文讲什么问题、涉及哪些
     地质概念"，正是图谱最需要的部分。

这个预算不是偷懒，是被实测数字逼出来的。要全量建图，正确做法是换更快的
抽取模型或上 GPU，而不是把预算调大然后等一晚上。

增量与幂等
----------
- ChromaDB：``upsert`` + 确定性 ID（``{文件名}::{块序号}``），重跑不会重复。
- LightRAG：LightRAG 自身按内容哈希去重；另外用
  ``ccus_docs/graph_ingest_state.json`` 记录每篇论文喂了多少字符、
  内容哈希是多少，内容没变就跳过 —— 否则重跑会把同一篇论文再抽一遍，
  白烧几十分钟 CPU。

新集合 ``ccus_knowledge_v3`` 的来历
----------------------------------
v2 里已有 671 条（联网搜索自动入库的内容）。直接建一个只有 PDF 切片的 v3，
会让"把默认集合切到 v3"变成一次**静默的知识缩水**。所以 v3 建立时先把 v2
整库克隆过来（连向量一起搬，不重新算），再叠加 PDF 切片。这样 v3 是 v2 的
严格超集。

用法
----
    python ingest_ccus_batch.py                    # 完整入库
    python ingest_ccus_batch.py --stats            # 只看当前规模
    python ingest_ccus_batch.py --skip-graph       # 只灌 ChromaDB
    python ingest_ccus_batch.py --graph-budget 6000
    python ingest_ccus_batch.py --limit 2          # 先小样验证
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(PROJECT_ROOT, "ccus_docs")
MANIFEST_PATH = os.path.join(DOCS_DIR, "manifest.json")
GRAPH_STATE_PATH = os.path.join(DOCS_DIR, "graph_ingest_state.json")
LIGHTRAG_DIR = os.path.join(PROJECT_ROOT, "lightrag_storage")
GRAPHML_PATH = os.path.join(LIGHTRAG_DIR, "graph_chunk_entity_relation.graphml")

TARGET_COLLECTION = os.getenv("CCUS_COLLECTION_V3", "ccus_knowledge_v3")
SOURCE_COLLECTION = os.getenv("CCUS_COLLECTION", "ccus_knowledge_v2")
# 嵌入模型不在这里决定 —— 必须与查询端一致，统一由 utils.embeddings 提供。

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150


def log(msg: str) -> None:
    print(msg, flush=True)


# ══════════════════════════════════════════════════════════
# 路径解析
# ══════════════════════════════════════════════════════════
def resolve_chroma_path() -> str:
    """
    定位 ChromaDB 目录。

    仓库里同时存在两个 ``chroma_petro``：项目内的（几乎是空的，早期
    ``ingest_pdf.py`` 用相对路径建出来的）和 ``D:\\python-learn\\chroma_petro``
    （``tools/knowledge_tools.py`` 真正查询的那个）。写错地方不会报错，
    只会表现为"入库成功但检索查不到"。所以这里显式按优先级解析，并把
    最终路径打出来。
    """
    env = os.getenv("CCUS_CHROMA_PATH")
    if env:
        return env
    legacy = r"D:\python-learn\chroma_petro"
    if os.path.isdir(legacy):
        return legacy
    return os.path.join(PROJECT_ROOT, "chroma_petro")


# ══════════════════════════════════════════════════════════
# PDF 读取
# ══════════════════════════════════════════════════════════
def extract_pdf(path: str) -> str:
    """
    抽取 PDF 全文。失败返回空串（单篇失败不该中断批量入库）。

    优先 **PyMuPDF**，pypdf 只作兜底。这不是偏好问题：实测同一篇 arXiv 论文，
    pypdf 会把整段抽成

        "actiontostabilizeatmosphericconcentrationsandlimitglobalwarming"

    —— 该 PDF 文字层里单词靠坐标定位而非空格字符分隔，pypdf 识别不出来。
    这种输入会同时毁掉分词、嵌入和实体抽取，而且**不会报错**，只会让检索
    悄悄变差。PyMuPDF 能正确还原空格。

    换抽取器是这类流水线里性价比最高的一步：下游全部环节都吃这个文本。
    """
    try:
        import pymupdf

        doc = pymupdf.open(path)
        try:
            return "\n".join(page.get_text("text") for page in doc)
        finally:
            doc.close()
    except Exception as exc:
        log(f"    [WARN] PyMuPDF 抽取失败（{type(exc).__name__}），回退 pypdf")

    try:
        from pypdf import PdfReader

        reader = PdfReader(path)
    except Exception as exc:
        log(f"    [WARN] 无法打开 PDF（{type(exc).__name__}: {exc}）")
        return ""

    parts = []
    for page in reader.pages:
        try:
            txt = page.extract_text() or ""
        except Exception:
            continue
        if txt:
            parts.append(txt)
    return "\n".join(parts)


# ══════════════════════════════════════════════════════════
# 论文清单
# ══════════════════════════════════════════════════════════
def load_manifest() -> dict:
    if not os.path.exists(MANIFEST_PATH):
        return {}
    try:
        with open(MANIFEST_PATH, encoding="utf-8") as fh:
            data = json.load(fh)
        return {p["file"]: p for p in data.get("papers", [])}
    except (json.JSONDecodeError, OSError):
        return {}


def collect_pdfs(include_originals: bool = True) -> list[dict]:
    """
    收集待入库的 PDF。

    返回 [{file, path, title, arxiv_id, year, is_auto}]，按文件名排序。
    """
    manifest = load_manifest()
    items: list[dict] = []

    auto = sorted(f for f in os.listdir(DOCS_DIR) if re.fullmatch(r"ccus_auto_\d+\.pdf", f))
    for f in auto:
        meta = manifest.get(f, {})
        published = meta.get("published", "")
        items.append({
            "file": f,
            "path": os.path.join(DOCS_DIR, f),
            "title": meta.get("title", f),
            "arxiv_id": meta.get("arxiv_id", ""),
            "year": int(published[:4]) if published[:4].isdigit() else 0,
            "is_auto": True,
        })

    if include_originals:
        originals = sorted(f for f in os.listdir(DOCS_DIR) if re.fullmatch(r"ccus\d+\.pdf", f))
        for f in originals:
            items.append({
                "file": f,
                "path": os.path.join(DOCS_DIR, f),
                "title": f,
                "arxiv_id": "",
                "year": 0,
                "is_auto": False,
            })

    return items


# ══════════════════════════════════════════════════════════
# ChromaDB
# ══════════════════════════════════════════════════════════
def chroma_client_and_ef():
    import chromadb

    from utils.embeddings import describe, get_embedding_function

    path = resolve_chroma_path()
    log(f"  ChromaDB 路径: {path}")
    log(f"  嵌入模型: {describe()}")
    ef = get_embedding_function()
    client = chromadb.PersistentClient(path=path)
    return client, ef


def clone_collection(client, ef, src_name: str, dst_name: str, batch: int = 400) -> int:
    """
    把 src 整库克隆进 dst（连向量一起搬）。dst 非空则跳过。

    搬向量而不是重新 embed：671 条重算要几分钟，而且理论上会得到略有差异的
    向量，让"新旧内容"在同一个空间里不完全可比。
    """
    dst = client.get_or_create_collection(name=dst_name, embedding_function=ef)
    if dst.count() > 0:
        log(f"  [{dst_name}] 已有 {dst.count()} 条，跳过克隆。")
        return 0

    try:
        src = client.get_collection(name=src_name, embedding_function=ef)
    except Exception:
        log(f"  [{src_name}] 不存在，跳过克隆。")
        return 0

    total = src.count()
    log(f"  从 {src_name}（{total} 条）克隆到 {dst_name} ...")
    moved = 0
    offset = 0
    while offset < total:
        res = src.get(
            limit=batch,
            offset=offset,
            include=["documents", "metadatas", "embeddings"],
        )
        ids = list(res.get("ids") or [])
        if not ids:
            break

        # 注意：chromadb 把 embeddings 返回成 numpy 二维数组，不能写
        # `res.get("embeddings") or []` —— numpy 数组的真值判断会抛
        # "truth value of an array with more than one element is ambiguous"。
        raw_embs = res.get("embeddings")
        embs = [
            e.tolist() if hasattr(e, "tolist") else list(e)
            for e in (raw_embs if raw_embs is not None else [])
        ]

        raw_metas = res.get("metadatas")
        # ChromaDB 拒绝空 metadata（"Expected metadata to be a non-empty dict"），
        # 而 v2 里确实有早期写入时没带 metadata 的条目。补一个来源标记而不是
        # 传 None —— 传 None 会丢掉"这批是从 v2 继承来的"这个信息。
        metas = []
        for m in (raw_metas if raw_metas is not None else []):
            metas.append(m if isinstance(m, dict) and m else {"source": "legacy_v2"})
        if not metas:
            metas = [{"source": "legacy_v2"}] * len(ids)

        dst.add(
            ids=ids,
            documents=res.get("documents") or [""] * len(ids),
            metadatas=metas,
            embeddings=embs,
        )
        moved += len(ids)
        offset += len(ids)
        log(f"    {moved}/{total}")
    return moved


def sync_chroma(items: list[dict], clone_from_v2: bool = True) -> dict:
    """把 PDF 切片写入目标集合。返回统计信息。"""
    from utils.text_cleaning import chunk_text, clean_pdf_text

    client, ef = chroma_client_and_ef()

    cloned = 0
    if clone_from_v2:
        cloned = clone_collection(client, ef, SOURCE_COLLECTION, TARGET_COLLECTION)

    collection = client.get_or_create_collection(name=TARGET_COLLECTION, embedding_function=ef)
    before = collection.count()

    total_chunks = 0
    for i, it in enumerate(items, 1):
        raw = extract_pdf(it["path"])
        text = clean_pdf_text(raw)
        if len(text) < 200:
            log(f"  ({i:2}/{len(items)}) [SKIP] {it['file']} 抽取文本过少（{len(text)} 字符）")
            continue

        chunks = chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
        stem = os.path.splitext(it["file"])[0]
        ids = [f"{stem}::{j:04d}" for j in range(len(chunks))]
        metas = [{
            "paper_file": it["file"],
            "paper_stem": stem,
            "title": it["title"][:200],
            "arxiv_id": it["arxiv_id"] or "",
            "year": it["year"],
            "chunk_index": j,
            "n_chars": len(c),
            "source_type": "pdf",
            "ingested_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        } for j, c in enumerate(chunks)]

        # 先按文档删除，再写入。
        #
        # 只用 upsert 是不够的：换了抽取器 / 改了切片参数之后，新块数可能
        # **少于**上次（实测 165 → 130），upsert 只会覆盖前 130 个 ID，
        # 尾部 35 条旧块作为孤儿留在库里 —— 检索时会被召回，内容却来自
        # 上一版抽取结果。这种残留不会报错，只会让结果慢慢变脏。
        collection.delete(where={"paper_stem": stem})
        collection.upsert(ids=ids, documents=chunks, metadatas=metas)
        total_chunks += len(chunks)
        log(f"  ({i:2}/{len(items)}) {it['file']:20} {len(text):>7} 字符 → {len(chunks):>3} 块")

    after = collection.count()
    return {
        "collection": TARGET_COLLECTION,
        "cloned": cloned,
        "chunks_written": total_chunks,
        "before": before,
        "after": after,
    }


# ══════════════════════════════════════════════════════════
# 知识图谱
# ══════════════════════════════════════════════════════════
def graph_stats() -> dict:
    """从 graphml 读当前节点/边数。"""
    if not os.path.exists(GRAPHML_PATH):
        return {"nodes": 0, "edges": 0, "exists": False}
    with open(GRAPHML_PATH, encoding="utf-8") as fh:
        raw = fh.read()
    return {
        "nodes": len(re.findall(r"<node\s", raw)),
        "edges": len(re.findall(r"<edge\s", raw)),
        "exists": True,
    }


def load_graph_state() -> dict:
    if os.path.exists(GRAPH_STATE_PATH):
        try:
            with open(GRAPH_STATE_PATH, encoding="utf-8") as fh:
                return json.load(fh)
        except (json.JSONDecodeError, OSError):
            pass
    return {"papers": {}}


def save_graph_state(state: dict) -> None:
    with open(GRAPH_STATE_PATH, "w", encoding="utf-8") as fh:
        json.dump(state, fh, ensure_ascii=False, indent=2)


DOC_STATUS_PATH = os.path.join(LIGHTRAG_DIR, "kv_store_doc_status.json")


def reset_stuck_docs() -> int:
    """
    清理卡在 ``status=processing`` 的文档。

    批量建图被中断（进程被杀 / 请求超时）时，LightRAG 会把文档留在
    processing 状态。**后续插入会被这个状态挡住**：重跑不报错，只是安静地
    什么都不做 —— 这是最难排查的一类"假成功"。所以恢复流程必须是
    "先删半成品，再重新插入"。
    """
    if not os.path.exists(DOC_STATUS_PATH):
        log("  没有 doc_status 记录，无需清理。")
        return 0
    with open(DOC_STATUS_PATH, encoding="utf-8") as fh:
        status = json.load(fh)

    stuck = [k for k, v in status.items()
             if isinstance(v, dict) and v.get("status") == "processing"]
    if not stuck:
        log("  没有卡在 processing 的文档。")
        return 0

    log(f"  发现 {len(stuck)} 篇卡在 processing，正在删除...")
    from lightrag_retriever import delete_document

    removed = 0
    for doc_id in stuck:
        if delete_document(doc_id):
            removed += 1
            log(f"    ✅ 已删除 {doc_id[:24]}")
        else:
            log(f"    ❌ 删除失败 {doc_id[:24]}")
    return removed


def sync_graph(items: list[dict], budget: int, force: bool = False) -> dict:
    """把每篇论文的前 ``budget`` 字符喂给 LightRAG 建图。"""
    from utils.text_cleaning import clean_pdf_text, select_graph_text, strip_front_matter

    log("  正在初始化 LightRAG（加载图谱存储 + 后台 event loop）...")
    from lightrag_retriever import insert_text

    state = load_graph_state()
    before = graph_stats()
    log(f"  建图前：{before['nodes']} 节点 / {before['edges']} 边")

    done, skipped, failed = 0, 0, 0
    for i, it in enumerate(items, 1):
        raw = extract_pdf(it["path"])
        text = clean_pdf_text(raw)
        if len(text) < 200:
            log(f"  ({i:2}/{len(items)}) [SKIP] {it['file']} 文本过少")
            skipped += 1
            continue

        # 先去首页的作者/单位/地址块，**再**取预算。
        # 顺序反了的话，被删掉的单位行会先占掉预算，正文反而少喂几十上百字。
        body = select_graph_text(strip_front_matter(text), budget)
        # 标题单独前置：实体抽取需要知道"这篇讲的是什么"，否则摘要里的
        # 代词式表述（this study / the reservoir）抽不出可复用的节点。
        payload = f"Title: {it['title']}\n\n{body}"
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

        prev = state["papers"].get(it["file"])
        if prev and prev.get("hash") == digest and not force:
            log(f"  ({i:2}/{len(items)}) [SKIP] {it['file']} 内容未变（已建图）")
            skipped += 1
            continue

        t0 = time.time()
        ok = insert_text(payload)
        dt = time.time() - t0
        if ok:
            done += 1
            state["papers"][it["file"]] = {
                "hash": digest,
                "chars": len(payload),
                "seconds": round(dt, 1),
                "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            save_graph_state(state)   # 逐篇落盘：中途崩了不用从头再来
            log(f"  ({i:2}/{len(items)}) {it['file']:20} {len(payload):>5} 字符 → {dt:6.1f}s ✅")
        else:
            failed += 1
            log(f"  ({i:2}/{len(items)}) {it['file']:20} 建图失败 ❌")

    after = graph_stats()
    return {
        "before": before,
        "after": after,
        "inserted": done,
        "skipped": skipped,
        "failed": failed,
    }


# ══════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════
def print_stats() -> None:
    g = graph_stats()
    log(f"图谱：{g['nodes']} 节点 / {g['edges']} 边")
    try:
        client, _ = chroma_client_and_ef()
        for name in (SOURCE_COLLECTION, TARGET_COLLECTION):
            try:
                log(f"集合 {name}: {client.get_collection(name).count()} 条")
            except Exception:
                log(f"集合 {name}: 不存在")
    except Exception as exc:
        log(f"ChromaDB 不可用: {exc}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="CCUS 文献批量入库")
    ap.add_argument("--graph-budget", type=int, default=4000,
                    help="每篇论文喂给知识图谱的字符数（默认 4000）")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 篇（小样验证）")
    ap.add_argument("--no-originals", action="store_true", help="不处理原有 ccus1-5.pdf")
    ap.add_argument("--skip-chroma", action="store_true")
    ap.add_argument("--skip-graph", action="store_true")
    ap.add_argument("--no-clone", action="store_true", help="不把 v2 克隆进 v3")
    ap.add_argument("--force-graph", action="store_true", help="忽略状态文件重新建图")
    ap.add_argument("--reset-stuck", action="store_true",
                    help="先清理卡在 processing 的文档（中断后必须执行）")
    ap.add_argument("--stats", action="store_true", help="只打印当前规模")
    args = ap.parse_args(argv)

    if args.stats:
        print_stats()
        return 0

    log("=" * 66)
    log("CCUS 文献批量入库")
    log("=" * 66)

    if args.reset_stuck:
        log("\n[0/2] 清理卡死文档...")
        reset_stuck_docs()

    items = collect_pdfs(include_originals=not args.no_originals)
    if args.limit:
        items = items[: args.limit]
    log(f"\n待处理 PDF：{len(items)} 篇")
    for it in items:
        log(f"   - {it['file']:20} {it['title'][:58]}")

    summary: dict = {"papers": len(items)}

    # ① ChromaDB
    if not args.skip_chroma:
        log(f"\n[1/2] 写入 ChromaDB（集合 {TARGET_COLLECTION}，全量正文切片）...")
        try:
            summary["chroma"] = sync_chroma(items, clone_from_v2=not args.no_clone)
            c = summary["chroma"]
            log(f"\n  ✅ {c['collection']}: {c['before']} → {c['after']} 条"
                f"（克隆 {c['cloned']}，新增切片 {c['chunks_written']}）")
        except Exception as exc:
            log(f"\n  ❌ ChromaDB 入库失败：{type(exc).__name__}: {exc}")
            summary["chroma_error"] = str(exc)
    else:
        log("\n[1/2] 跳过 ChromaDB")

    # ② LightRAG
    if not args.skip_graph:
        log(f"\n[2/2] 扩充知识图谱（每篇前 {args.graph_budget} 字符）...")
        try:
            summary["graph"] = sync_graph(items, args.graph_budget, force=args.force_graph)
            g = summary["graph"]
            log(f"\n  ✅ 图谱：{g['before']['nodes']} → {g['after']['nodes']} 节点，"
                f"{g['before']['edges']} → {g['after']['edges']} 边"
                f"（新增 {g['inserted']} 篇，跳过 {g['skipped']}，失败 {g['failed']}）")
        except Exception as exc:
            log(f"\n  ❌ 建图失败：{type(exc).__name__}: {exc}")
            summary["graph_error"] = str(exc)
    else:
        log("\n[2/2] 跳过知识图谱")

    log("\n" + "=" * 66)
    log(json.dumps(summary, ensure_ascii=False, indent=2))
    log("=" * 66)
    return 0


if __name__ == "__main__":
    sys.exit(main())
