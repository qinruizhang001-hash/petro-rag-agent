"""知识库接口（``/api/kb/*``）。

为什么单独一个模块，而不是写进 ``api_server.py``
------------------------------------------------
和 ``api_ops.py`` 同一个理由，但压力更大：``api_server.py`` 顶部
``from workflow import ...`` 会加载 LangGraph 全图与 2.2 GB 重排模型。
知识库面板是控制台**首屏**就要渲染的东西，让它等一个 2.2 GB 的模型
既不合理也不必要 —— 语料清单、建图状态、切片原文全在 JSON 文件里，
一次 ``open()`` 就能读，与模型无关。

拆出来之后，本模块的**静态端点**（overview / documents）零重依赖：

* 不 import ``lightrag``（那会拉起 LLM 客户端）
* 不 import ``chromadb``（用 stdlib ``sqlite3`` 直读它的库文件）
* 不 import ``sentence_transformers``

因此 ``tests/test_api_kb.py`` 可以挂一个裸 ``FastAPI()`` 直接跑，
在 CI 的零依赖 job 里也成立。

检索端点确实需要模型，处理方式是**函数内懒加载** ``lightrag_retriever``：
不查不加载，且失败要能被区分（见下）。

三条贯穿本模块的约束
--------------------
1. **"没有"与"不知道"必须分开**
   ``POST /api/kb/search`` 的 ``status`` 有 ``ok`` / ``empty`` / ``unavailable``
   三个值，不是两个。``empty`` 是一句关于语料的事实陈述（"库里没有这个内容"），
   ``unavailable`` 是系统状态（"检索没跑起来"）。把后者渲染成前者，操作员会
   据此去补文献 —— 而真实原因只是 Ollama 没启动。

2. **缺口要显示，不能显示成完整**
   盘上 24 个文档、图谱里 22 个，差的那 2 个必须在 ``gaps`` 里逐条列出。
   一个"知识库共 22 篇"的数字读起来是完整的，而它其实漏了 2 篇。

3. **没有血缘的库不能假装有**
   Chroma 那 1982 条向量的 metadata 只有 ``{"source": "legacy_v2"}``，
   没有文件级血缘。所以 ``vector.lineage`` 是 ``false``，且带一句说明 ——
   界面上这条约束要显示出来，否则用户会以为向量分支的结论也能点回原文。

响应形状直接来自数据源，不额外包一层
------------------------------------
``overview`` 的每个数字都能回代到某个文件的某个字段（见各字段注释）。
包一层 ``{"data": {...}}`` 只会让前端多维护一份字段名。
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from kb_evidence import build_evidence
from utils.logging_setup import get_logger

log = get_logger(__name__)

ROOT = Path(__file__).resolve().parent
DOCS_DIR = ROOT / "ccus_docs"
LIGHTRAG_DIR = ROOT / "lightrag_storage"

#: 语料的扩展名白名单。``manifest.json`` / ``graph_ingest_state.json`` 也在
#: ``ccus_docs/`` 里，但它们是**元数据**不是语料 —— 算进"共 N 篇文档"会让
#: 那个数字虚高 3 篇，而虚高的方式恰好是读者不会怀疑的那种。
CORPUS_EXTS = {".pdf", ".md", ".txt"}

#: 单篇文档详情默认返回的切片数。一篇 82 chunk 的论文全文约 65 KB，
#: 全量下发会让面板点一下就卡住；``focus`` 参数可以定位到指定切片附近。
DETAIL_LIMIT = 60

#: LightRAG 支持的查询模式。请求模型里用 ``Literal`` 而不是 ``str``：
#: 传错一个模式名 LightRAG 会抛异常并变成 500，而 422 能直接说清是谁的错。

#: 不带鉴权依赖，由 api_server 在 include 时统一施加（与 api_ops 同一分工）。
router = APIRouter(tags=["kb"])


# ══════════════════════════════════════════════════════════
# 数据源读取（全部是文件读，无模型、无网络）
# ══════════════════════════════════════════════════════════

def _read_json(path: Path) -> Dict[str, Any]:
    """读一个 JSON 文件；不存在或损坏时返回空 dict。

    这里吞异常是对的：知识库面板要在**任何**状态下都能渲染出"当前是什么状态"。
    文件缺失本身就是一种状态（还没建图），不该让端点 500 —— 那会让界面
    从"知识库是空的"变成"知识库坏了"，而这两件事的处置完全不同。
    """
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except Exception as exc:  # noqa: BLE001 — 损坏的 JSON 只应体现为"读不到"
        log.warning(f"[KB] 读取失败 {path.name}: {type(exc).__name__}: {exc}")
        return {}


def _disk_corpus() -> List[Dict[str, Any]]:
    """盘上的语料文件（按名字排序，保证输出稳定）。"""
    out: List[Dict[str, Any]] = []
    if not DOCS_DIR.is_dir():
        return out
    for p in sorted(DOCS_DIR.iterdir(), key=lambda x: x.name):
        if not p.is_file() or p.suffix.lower() not in CORPUS_EXTS:
            continue
        out.append(
            {
                "name": p.name,
                "bytes": p.stat().st_size,
                "kind": p.suffix.lower().lstrip("."),
            }
        )
    return out


def _manifest_index() -> Dict[str, Dict[str, Any]]:
    """``manifest.json`` → {文件名: 元数据}。

    只有 ``ccus_auto_*.pdf`` 有这条记录（那是 arXiv 抓取脚本写的）。
    ``ccus1-5.pdf`` 与 ``A1井_历史压力监测.md`` 是手工放进去的，没有条目 ——
    界面上它们会显示"无题录"，这是如实的，不该拿文件名硬凑一个标题。
    """
    raw = _read_json(DOCS_DIR / "manifest.json")
    idx: Dict[str, Dict[str, Any]] = {}
    for paper in raw.get("papers") or []:
        name = paper.get("file")
        if not name:
            continue
        idx[name] = {
            "title": paper.get("title") or "",
            "authors": paper.get("authors") or [],
            "published": paper.get("published") or "",
            "url": paper.get("abs_url") or paper.get("pdf_url") or "",
            "arxiv_id": paper.get("arxiv_id") or "",
        }
    return idx


def _doc_status() -> Dict[str, Dict[str, Any]]:
    """图谱侧的文档状态表（键是 ``file_path``，即入库时写入的血缘）。"""
    return _read_json(LIGHTRAG_DIR / "kv_store_doc_status.json")


def _ingest_state() -> Dict[str, Dict[str, Any]]:
    """``graph_ingest_state.json`` → {文件名: {chars, seconds, at}}（建图耗时）。"""
    raw = _read_json(DOCS_DIR / "graph_ingest_state.json")
    return raw.get("papers") or {}


def _graph_stats() -> Dict[str, Any]:
    """图谱规模。

    **不能**用 ``len(读到的 dict)``：``kv_store_full_entities.json`` 的键是
    ``doc-<hash>``（每篇文档一条），值是 ``{"entity_names": [...]}``。
    直接取长度得到的是"22 篇文档"，而它会显示成"22 个实体" ——
    实测差 20 倍，且这个数字在界面上没有任何交叉校验能发现它是错的。
    所以必须摊平名字列表再去重。

    为什么不去解析 graphml（5.7 MB，节点边都在里面）：那要建整张图，
    本端点只需要两个数字。
    """
    ents = _read_json(LIGHTRAG_DIR / "kv_store_full_entities.json")
    rels = _read_json(LIGHTRAG_DIR / "kv_store_full_relations.json")

    names: set[str] = set()
    for v in (ents.values() if isinstance(ents, dict) else []):
        for n in (v or {}).get("entity_names") or []:
            if str(n).strip():
                names.add(str(n).strip())

    pairs: set[tuple[str, str]] = set()
    for v in (rels.values() if isinstance(rels, dict) else []):
        for pair in (v or {}).get("relation_pairs") or []:
            if isinstance(pair, (list, tuple)) and len(pair) >= 2:
                pairs.add((str(pair[0]).strip(), str(pair[1]).strip()))

    graphml = LIGHTRAG_DIR / "graph_chunk_entity_relation.graphml"
    return {
        "entities": len(names),
        "relationships": len(pairs),
        "graph_bytes": graphml.stat().st_size if graphml.exists() else 0,
    }


def _vector_stats() -> Dict[str, Any]:
    """Chroma 向量库状态 —— 用 stdlib ``sqlite3`` 直读，不 import ``chromadb``。

    为什么值得这么绕：``import chromadb`` 会连带拉起 ``onnxruntime`` /
    ``pypika`` / ``numpy``，在本机上约 1.2 秒。而知识库面板首屏就要显示
    这个数字。为了一个 ``COUNT(*)`` 付 1.2 秒，还会让 CI 的零依赖 job
    跑不起来，不划算。

    代价是**依赖了 Chroma 的内部表结构**（``collections`` / ``embeddings``）。
    所以整段包在 try 里：读不到就返回 ``available: false``，而不是猜一个数字。
    猜出来的"1982"比"未知"危险得多 —— 后者会有人去看，前者不会。
    """
    # 路径必须与检索端一致，否则探的库和查的库不是同一个。
    # 这里**不**从 tools.knowledge_tools 导入（那会拉起 sentence-transformers），
    # 而是读同一个环境变量 + 同一个默认值。两边默认值写死成一样的字符串，
    # 并有 tests/test_api_kb.py 的静态断言盯着这一行不许漂。
    path = Path(os.getenv("CCUS_CHROMA_PATH") or r"D:\python-learn\chroma_petro")
    db = path / "chroma.sqlite3"
    base = {
        # 不返回绝对路径：`D:\python-learn\chroma_petro` 在界面上是噪音，
        # 而且它是**每台机器都不同**的值 —— 一旦进 fixture 就永远无法逐字节比对
        # （这个坑在 ops fixture 上踩过一次，见 capture_ops_fixtures.py 的
        # `_ORIGIN_REWRITES`）。集合名 + 向量数足以说明"看的是哪个库"。
        "available": False,
        "active_collection": "",
        "vectors": 0,
        "collections": [],
        # 血缘：Chroma 侧的 metadata 只有 {"source": "legacy_v2"}，
        # 没有文件级信息 → 它的召回**无法**回溯到某一篇文献。
        # 这条不是待办，是当前的事实，界面必须照实显示。
        "lineage": False,
    }
    if not db.exists():
        return base

    try:
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=2.0)
        try:
            names = [r[0] for r in con.execute("SELECT name FROM collections")]
            counts = {
                r[0]: r[1]
                for r in con.execute(
                    "SELECT c.name, COUNT(e.id) FROM collections c "
                    "LEFT JOIN embeddings e ON e.segment_id IN "
                    "(SELECT id FROM segments WHERE collection = c.id) "
                    "GROUP BY c.name"
                )
            }
        finally:
            con.close()
    except sqlite3.Error as exc:
        log.warning(f"[KB] 读 Chroma 库失败（不影响图谱侧）：{exc}")
        return base

    # 当前在用的集合名从环境变量取，默认与 tools.knowledge_tools 一致。
    active = os.getenv("CCUS_COLLECTION") or "ccus_knowledge_v3"
    base.update(
        {
            "available": True,
            "active_collection": active,
            "vectors": int(counts.get(active, 0)),
            "collections": [
                {"name": n, "vectors": int(counts.get(n, 0))} for n in sorted(names)
            ],
        }
    )
    return base


# ══════════════════════════════════════════════════════════
# GET /api/kb/overview —— 知识库当前是什么状态
# ══════════════════════════════════════════════════════════

@router.get("/api/kb/overview")
def kb_overview():
    """语料 / 图谱 / 向量库三块状态 + 缺口清单。

    这个端点的用途不是"给个总数"，而是回答**"这个知识库能信到什么程度"**。
    所以除了规模，还要给出三件容易让人误判的事：

    * 盘上有、图谱里没有的文件（``gaps.not_ingested``）
    * 血缘缺失的文档数（``gaps.unknown_source``）
    * 向量库没有文件级血缘（``vector.lineage = false``）
    """
    corpus = _disk_corpus()
    status = _doc_status()
    manifest = _manifest_index()
    ingest = _ingest_state()
    graph = _graph_stats()
    vector = _vector_stats()

    ingested = {
        str(v.get("file_path") or "").strip()
        for v in status.values()
        if str(v.get("file_path") or "").strip()
    }
    # 血缘缺失：入库时没传 file_paths，LightRAG 会记成 unknown_source。
    # 这些文档在界面上会显示成一个空文件名，看起来只是"没标题"。
    unknown_source = sorted(
        f for f in ingested if f in ("", "unknown_source")
    )
    on_disk = {f["name"] for f in corpus}
    not_ingested = sorted(on_disk - ingested)
    # 反向缺口：图谱里有、盘上已经没有的文件（删了 PDF 但没删图谱贡献）。
    # 这类记录会让检索命中一篇"打不开的文献"，所以必须能看见。
    orphaned = sorted(f for f in ingested if f and f != "unknown_source" and f not in on_disk)

    chunks = sum(int(v.get("chunks_count") or 0) for v in status.values())
    chars = sum(int(v.get("content_length") or 0) for v in status.values())
    seconds = sum(float(v.get("seconds") or 0) for v in (ingest or {}).values())

    return {
        "corpus": {
            "files": len(corpus),
            "bytes": sum(f["bytes"] for f in corpus),
            "by_kind": {
                k: sum(1 for f in corpus if f["kind"] == k)
                for k in sorted({f["kind"] for f in corpus})
            },
        },
        "graph": {
            "documents": len(ingested),
            "chunks": chunks,
            "chars": chars,
            # 建图总耗时。这一项是给运维看的：新增一篇 12 万字符的论文要跑
            # 约 30 分钟，这个数字决定了"入库要不要排到夜里"。
            "ingest_seconds": round(seconds, 1),
            **graph,
        },
        "vector": vector,
        "gaps": {
            "not_ingested": not_ingested,
            "unknown_source": unknown_source,
            "orphaned": orphaned,
            # 状态不是 processed 的文档（processing / failed / pending）。
            # 这类文档会**阻塞**后续插入，是批量入库最常见的卡死原因。
            "unfinished": sorted(
                str(v.get("file_path") or "")
                for v in status.values()
                if str(v.get("status") or "") != "processed"
            ),
        },
        "readiness": {
            # 能否检索只取决于图谱里有没有东西，与模型无关 —— 模型挂了
            # 是 search 端点的 unavailable，不是这里的 not_ready。
            "graph": "ready" if ingested and chunks else "empty",
            "vector": "ready" if vector["vectors"] else "empty",
        },
        # 题录覆盖率：manifest 只覆盖 arXiv 抓下来的那批。
        "titles": {
            "covered": sum(1 for f in on_disk if f in manifest),
            "missing": sorted(f for f in on_disk if f not in manifest),
        },
    }


# ══════════════════════════════════════════════════════════
# GET /api/kb/documents —— 语料清单
# ══════════════════════════════════════════════════════════

@router.get("/api/kb/documents")
def kb_documents():
    """盘上每个文档一行：是否已建图、多少切片、题录、建图耗时。

    遍历的是**盘上的文件**而不是图谱里的记录。顺序上看起来反了，但只有这样
    "没进图谱的那 2 篇"才会出现在列表里 —— 遍历图谱的话它们根本不存在，
    而列表读起来仍然完整。
    """
    corpus = _disk_corpus()
    status = _doc_status()
    manifest = _manifest_index()
    ingest = _ingest_state()

    by_file: Dict[str, Dict[str, Any]] = {}
    for v in status.values():
        name = str(v.get("file_path") or "").strip()
        if name:
            by_file[name] = v

    rows: List[Dict[str, Any]] = []
    for f in corpus:
        st = by_file.get(f["name"]) or {}
        meta = manifest.get(f["name"]) or {}
        ing = ingest.get(f["name"]) or {}
        rows.append(
            {
                "name": f["name"],
                "bytes": f["bytes"],
                "kind": f["kind"],
                # ingested=False 才是重点：界面上要能一眼看出哪几篇不在库里。
                "ingested": bool(st),
                "status": str(st.get("status") or ""),
                "chunks": int(st.get("chunks_count") or 0),
                "chars": int(st.get("content_length") or 0),
                "title": meta.get("title") or "",
                "authors": meta.get("authors") or [],
                "published": meta.get("published") or "",
                "url": meta.get("url") or "",
                "ingested_at": str(st.get("created_at") or ing.get("at") or ""),
                "ingest_seconds": float(ing.get("seconds") or 0),
            }
        )
    return {"total": len(rows), "documents": rows}


# ══════════════════════════════════════════════════════════
# GET /api/kb/documents/{name} —— 切片原文（引用跳转的落点）
# ══════════════════════════════════════════════════════════

_chunks_cache: Optional[Dict[str, Any]] = None


def _text_chunks() -> Dict[str, Any]:
    """``kv_store_text_chunks.json``（1.9 MB）读一次缓存住。

    缓存是安全的：这是建图产物，只在入库时变化。进程内缓存一份意味着
    面板连点十篇文档只读一次盘。
    """
    global _chunks_cache
    if _chunks_cache is None:
        _chunks_cache = _read_json(LIGHTRAG_DIR / "kv_store_text_chunks.json")
    return _chunks_cache


@router.get("/api/kb/documents/{name}")
def kb_document(
    name: str,
    focus: Optional[str] = Query(
        default=None,
        description="定位到某个 chunk_id，返回它附近的切片（引用跳转用）",
    ),
    limit: int = Query(default=DETAIL_LIMIT, ge=1, le=200),
):
    """单篇文档的切片原文。

    这是引用角标的落点，所以**不能依赖模型**：``file_path`` 与切片正文都在
    ``kv_store_text_chunks.json`` 里，直接读文件即可。Ollama 挂了、图谱没初始化，
    引用跳转照样能用 —— 否则"点开看看原文"这个动作会在最需要它的时候失效。
    """
    # 只允许仓库内的文件名。``name`` 来自 URL，直接拼路径会让
    # ``../../.env`` 读到不该读的东西。
    safe = Path(name).name
    if not safe or safe != name:
        raise HTTPException(status_code=400, detail="文档名不合法")

    chunks = [
        {"chunk_id": k, **(v or {})}
        for k, v in _text_chunks().items()
        if str((v or {}).get("file_path") or "") == safe
    ]
    if not chunks:
        # 区分"这篇没建图"与"这个名字根本不存在"，前端提示不同。
        on_disk = any(f["name"] == safe for f in _disk_corpus())
        raise HTTPException(
            status_code=404,
            detail=(
                "该文档尚未建图（盘上有这个文件，图谱里没有它的切片）"
                if on_disk
                else f"未找到文档 {safe}"
            ),
        )

    chunks.sort(key=lambda c: int(c.get("chunk_order_index") or 0))
    total = len(chunks)

    start = 0
    focused: Optional[str] = None
    if focus:
        for i, c in enumerate(chunks):
            if c["chunk_id"] == focus:
                start = max(0, i - limit // 2)
                focused = focus
                break
    window = chunks[start : start + limit]

    meta = _manifest_index().get(safe) or {}
    st = next(
        (v for v in _doc_status().values() if str(v.get("file_path") or "") == safe),
        {},
    )
    return {
        "name": safe,
        "title": meta.get("title") or "",
        "url": meta.get("url") or "",
        "status": str(st.get("status") or ""),
        "total_chunks": total,
        "offset": start,
        "focused": focused,
        "has_more": start + len(window) < total,
        "chunks": [
            {
                "chunk_id": c["chunk_id"],
                "index": int(c.get("chunk_order_index") or 0),
                "chars": len(str(c.get("content") or "")),
                "content": str(c.get("content") or ""),
            }
            for c in window
        ],
    }


# ══════════════════════════════════════════════════════════
# POST /api/kb/search —— 检索（唯一需要模型的端点）
# ══════════════════════════════════════════════════════════

class KBSearchRequest(BaseModel):
    model_config = {"protected_namespaces": ()}

    query: str = Field(min_length=1, max_length=2000)
    mode: Literal["hybrid", "local", "global", "naive", "mix"] = "hybrid"
    with_graph: bool = Field(
        default=True,
        description="是否返回实体/关系。关掉只留原文切片，用于窄面板。",
    )


@router.post("/api/kb/search")
def kb_search(request: KBSearchRequest):
    """检索知识库，返回带出处的证据卡。

    返回 ``status`` 有三个值，**不是两个**：

    * ``ok``          检索到内容
    * ``empty``       检索跑完了，确实没有相关内容 —— 这是关于语料的事实
    * ``unavailable`` 检索没跑起来（Ollama 未启动 / 图谱未初始化）

    HTTP 一律 200。理由与 ``/health`` 相同：``unavailable`` 是**需要被看见的
    系统状态**，不是请求错误。返回 503 的话前端只会弹一个通用错误，
    操作员学不到"是模型没起"还是"是库空了"，而这两种处置完全相反。
    """
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="query 不能为空")

    started = time.perf_counter()
    try:
        # 懒加载：不查就不 import，避免首屏为一个用不到的模型付初始化代价。
        import lightrag_retriever

        evidence = lightrag_retriever.query_evidence(
            query, mode=request.mode, with_graph=request.with_graph
        )
    except Exception as exc:  # noqa: BLE001 — 任何失败都是"检索没跑起来"
        log.warning(f"[KB] 检索不可用：{type(exc).__name__}: {exc}")
        return {
            "status": "unavailable",
            "query": query,
            "mode": request.mode,
            "reason": f"{type(exc).__name__}: {exc}"[:300],
            "evidence": build_evidence({}, with_graph=False),
            "elapsed_ms": int((time.perf_counter() - started) * 1000),
        }

    empty = not (evidence["chunks"] or evidence["entities"] or evidence["relationships"])
    return {
        "status": "empty" if empty else "ok",
        "query": query,
        "mode": request.mode,
        "reason": "图谱中没有与该问题相关的内容" if empty else "",
        "evidence": evidence,
        "elapsed_ms": int((time.perf_counter() - started) * 1000),
    }
