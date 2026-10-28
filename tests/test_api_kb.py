"""知识库接口（api_kb）的测试。

这一组要守住的东西，按重要性排：

1. **鉴权没漏**。``api_kb.router`` 自己不带鉴权（不能反向导入
   ``require_token``，否则成环），鉴权靠 ``api_server`` 的 ``include_router``
   施加。漏写 ``dependencies`` 不会报错，只会让知识库裸奔 —— 而它返回的是
   论文题录与切片原文，是业务数据。所以有一条静态断言直接读源码。
2. **缺口要出现，不能显示成完整**。盘上有、图谱里没有的文件必须逐条列在
   ``gaps.not_ingested`` 里。一个"知识库共 22 篇"的数字读起来是完整的，
   而它其实漏了 2 篇。
3. **``unavailable`` 不能伪装成 ``empty``**。前者是"检索没跑起来"，
   后者是"库里没有这个内容"。把前者渲染成后者，操作员会去补文献，
   而真实原因只是 Ollama 没启动。
4. **图谱规模要数去重后的名字**。``kv_store_full_entities.json`` 的键是
   ``doc-<hash>``，直接取长度得到的是"文档数"，实测差 20 倍，且界面上
   没有任何交叉校验能发现它是错的。

测试不导入 ``api_server``：那会拉起 LangGraph 全图与 2.2 GB 重排模型。
语料也不依赖仓库里的 ``ccus_docs/`` —— 那 125 MB 的 PDF 是**输入不是代码**，
CI 里根本没有，所以每个用例都自己造一份临时语料。
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from fastapi.testclient import TestClient

import api_kb


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


class KBApiTestBase(unittest.TestCase):
    """每个用例一套临时语料 + 临时图谱。

    语料形状刻意造成"有缺口"：盘上 3 篇（``a.pdf`` / ``b.pdf`` / ``missing.md``），
    图谱里只有 ``a.pdf``。这是真实状态（本机 24 篇里 2 篇没入库），
    也是这个接口最该被检查的那件事。
    """

    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="kb-test-")
        root = Path(self._tmp)
        self.docs = root / "ccus_docs"
        self.lightrag = root / "lightrag_storage"

        # ─── 盘上语料 ───
        self.docs.mkdir(parents=True)
        (self.docs / "a.pdf").write_bytes(b"%PDF-1.4" + b"x" * 100)
        (self.docs / "b.pdf").write_bytes(b"%PDF-1.4" + b"y" * 50)
        (self.docs / "missing.md").write_text("# 井史\n", encoding="utf-8")
        # 元数据不是语料：这两个文件在 ccus_docs/ 里，但算进"共 N 篇"会让数字虚高
        _write(
            self.docs / "manifest.json",
            {
                "papers": [
                    {
                        "file": "a.pdf",
                        "title": "Seal integrity of caprock",
                        "authors": ["A. Author"],
                        "published": "2025-05-19",
                        "abs_url": "https://arxiv.org/abs/0000.00000",
                    }
                ]
            },
        )
        _write(
            self.docs / "graph_ingest_state.json",
            {"papers": {"a.pdf": {"chars": 100, "seconds": 10.0, "at": "2026-10-24T16:08:43+00:00"}}},
        )

        # ─── 图谱侧 ───
        self.lightrag.mkdir(parents=True)
        _write(
            self.lightrag / "kv_store_doc_status.json",
            {
                "doc-aaa": {
                    "file_path": "a.pdf",
                    "status": "processed",
                    "chunks_count": 2,
                    "content_length": 120,
                    "created_at": "2026-10-24T16:08:43+00:00",
                }
            },
        )
        # 键是 doc-<hash>；名字列表里故意放重复项，验证去重
        _write(
            self.lightrag / "kv_store_full_entities.json",
            {"doc-aaa": {"entity_names": ["Caprock", "CO2 Leakage", "Caprock"]}},
        )
        _write(
            self.lightrag / "kv_store_full_relations.json",
            {"doc-aaa": {"relation_pairs": [["Caprock", "CO2 Leakage"], ["Caprock", "CO2 Leakage"]]}},
        )
        _write(
            self.lightrag / "kv_store_text_chunks.json",
            {
                "chunk-2": {
                    "file_path": "a.pdf",
                    "chunk_order_index": 1,
                    "content": "第二片",
                },
                "chunk-1": {
                    "file_path": "a.pdf",
                    "chunk_order_index": 0,
                    "content": "第一片",
                },
            },
        )

        self._orig = (api_kb.DOCS_DIR, api_kb.LIGHTRAG_DIR)
        api_kb.DOCS_DIR = self.docs
        api_kb.LIGHTRAG_DIR = self.lightrag
        api_kb._chunks_cache = None

        app = FastAPI()
        app.include_router(api_kb.router)
        self.client = TestClient(app)

    def tearDown(self):
        api_kb.DOCS_DIR, api_kb.LIGHTRAG_DIR = self._orig
        api_kb._chunks_cache = None
        shutil.rmtree(self._tmp, ignore_errors=True)


# ══════════════════════════════════════════════════════════
# 一、鉴权与依赖边界
# ══════════════════════════════════════════════════════════
class TestAuthWiring(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parent.parent
        self.server = (self.root / "api_server.py").read_text(encoding="utf-8")
        self.module = (self.root / "api_kb.py").read_text(encoding="utf-8")

    def test_kb_router_is_included_with_token_dependency(self):
        lines = [ln for ln in self.server.split("\n") if "include_router(kb_router" in ln]
        self.assertEqual(
            len(lines), 1,
            "api_server.py 应当只有一处挂载 kb_router；多处挂载会让鉴权口径不一致",
        )
        self.assertIn(
            "Depends(require_token)", lines[0],
            "kb_router 必须带鉴权依赖。不带也能跑，但知识库就变成匿名可读了",
        )

    def test_kb_module_does_not_import_api_server(self):
        self.assertIsNone(
            re.search(r"^\s*(from|import)\s+api_server", self.module, re.M),
            "api_kb.py 不能导入 api_server（会成环，且会把 2.2 GB 模型拖进接口测试）",
        )

    def test_no_heavy_import_at_module_level(self):
        """``lightrag`` / ``chromadb`` / 嵌入模型只能在函数内懒加载。

        首屏就要渲染语料清单，为一个 ``COUNT(*)`` 付 1.2 秒的 import 代价
        不划算；更要紧的是 CI 的零依赖 job 会直接导入失败。
        """
        for mod in ("lightrag", "chromadb", "sentence_transformers", "torch"):
            self.assertIsNone(
                re.search(rf"^(from|import)\s+{mod}\b", self.module, re.M),
                f"api_kb.py 不能在模块顶层 import {mod}，必须放进函数内",
            )

    def test_chroma_default_path_matches_retrieval_side(self):
        """探的库必须与查的库是同一个。

        这里不 import ``tools.knowledge_tools``（那会拉起 sentence-transformers），
        所以默认值只能各写一份 —— 那就必须有断言盯着它们不许漂。
        漂了的后果不是报错，而是面板上"1982 条向量"与检索实际用的库无关。
        """
        src = (self.root / "tools" / "knowledge_tools.py").read_text(encoding="utf-8")
        m = re.search(r"CCUS_CHROMA_PATH\s*=\s*os\.getenv\(\s*[\"']CCUS_CHROMA_PATH[\"']\s*,\s*r?[\"']([^\"']+)[\"']", src)
        self.assertIsNotNone(m, "没找到 tools/knowledge_tools.py 里 CCUS_CHROMA_PATH 的默认值")
        self.assertIn(
            m.group(1), self.module,
            f"api_kb.py 里的 Chroma 默认路径与检索端不一致（检索端是 {m.group(1)}）",
        )


# ══════════════════════════════════════════════════════════
# 二、概览：缺口与规模
# ══════════════════════════════════════════════════════════
class TestOverview(KBApiTestBase):
    def setUp(self):
        super().setUp()
        # 向量库指向不存在的目录，让用例只考察语料与图谱
        self._env = mock.patch.dict(os.environ, {"CCUS_CHROMA_PATH": str(Path(self._tmp) / "nope")})
        self._env.start()

    def tearDown(self):
        self._env.stop()
        super().tearDown()

    def _overview(self) -> dict:
        r = self.client.get("/api/kb/overview")
        self.assertEqual(r.status_code, 200)
        return r.json()

    def test_metadata_files_are_not_counted_as_corpus(self):
        """``manifest.json`` / ``graph_ingest_state.json`` 在语料目录里，但不是语料。"""
        corpus = self._overview()["corpus"]
        self.assertEqual(corpus["files"], 3)
        self.assertEqual(corpus["by_kind"], {"md": 1, "pdf": 2})

    def test_not_ingested_files_are_listed(self):
        gaps = self._overview()["gaps"]
        self.assertEqual(gaps["not_ingested"], ["b.pdf", "missing.md"])

    def test_graph_counts_deduplicated_names(self):
        """回归：曾经把 ``len(dict)`` 当成实体数，报出 22 而不是 4826。"""
        graph = self._overview()["graph"]
        self.assertEqual(graph["documents"], 1)
        self.assertEqual(graph["entities"], 2)
        self.assertEqual(graph["relationships"], 1)

    def test_chunks_and_chars_come_from_doc_status(self):
        graph = self._overview()["graph"]
        self.assertEqual(graph["chunks"], 2)
        self.assertEqual(graph["chars"], 120)

    def test_title_coverage_exposes_untitled_files(self):
        titles = self._overview()["titles"]
        self.assertEqual(titles["covered"], 1)
        self.assertEqual(titles["missing"], ["b.pdf", "missing.md"])

    def test_vector_store_absent_is_reported_not_guessed(self):
        vector = self._overview()["vector"]
        self.assertFalse(vector["available"])
        self.assertEqual(vector["vectors"], 0)
        self.assertFalse(vector["lineage"], "Chroma 没有文件级血缘，这条必须是 false")

    def test_orphaned_graph_records_are_visible(self):
        """图谱里有、盘上已删的文件必须列出来 —— 否则会命中一篇打不开的文献。"""
        _write(
            self.lightrag / "kv_store_doc_status.json",
            {
                "doc-aaa": {"file_path": "a.pdf", "status": "processed", "chunks_count": 2},
                "doc-bbb": {"file_path": "deleted.pdf", "status": "processed", "chunks_count": 1},
            },
        )
        self.assertEqual(self._overview()["gaps"]["orphaned"], ["deleted.pdf"])

    def test_unfinished_documents_are_visible(self):
        """非 processed 状态会**阻塞**后续插入，是批量入库最常见的卡死原因。"""
        _write(
            self.lightrag / "kv_store_doc_status.json",
            {"doc-aaa": {"file_path": "a.pdf", "status": "processing", "chunks_count": 0}},
        )
        self.assertEqual(self._overview()["gaps"]["unfinished"], ["a.pdf"])

    def test_empty_dirs_do_not_crash(self):
        shutil.rmtree(self.docs)
        shutil.rmtree(self.lightrag)
        body = self._overview()
        self.assertEqual(body["corpus"]["files"], 0)
        self.assertEqual(body["readiness"]["graph"], "empty")


class TestVectorStats(KBApiTestBase):
    """Chroma 计数走 stdlib ``sqlite3``，这段 SQL 是本模块最脆的地方。"""

    def _fake_chroma(self) -> Path:
        path = Path(self._tmp) / "chroma"
        path.mkdir()
        con = sqlite3.connect(str(path / "chroma.sqlite3"))
        con.executescript(
            "CREATE TABLE collections (id TEXT PRIMARY KEY, name TEXT);"
            "CREATE TABLE segments (id TEXT PRIMARY KEY, collection TEXT);"
            "CREATE TABLE embeddings (id INTEGER PRIMARY KEY, segment_id TEXT);"
        )
        con.execute("INSERT INTO collections VALUES ('c1','ccus_knowledge_v3')")
        con.execute("INSERT INTO collections VALUES ('c0','legacy')")
        con.execute("INSERT INTO segments VALUES ('s1','c1')")
        con.execute("INSERT INTO segments VALUES ('s0','c0')")
        for i in range(3):
            con.execute("INSERT INTO embeddings (segment_id) VALUES ('s1')")
        con.execute("INSERT INTO embeddings (segment_id) VALUES ('s0')")
        con.commit()
        con.close()
        return path

    def test_counts_are_read_from_sqlite(self):
        path = self._fake_chroma()
        with mock.patch.dict(os.environ, {"CCUS_CHROMA_PATH": str(path)}):
            vector = self.client.get("/api/kb/overview").json()["vector"]
        self.assertTrue(vector["available"])
        self.assertEqual(vector["active_collection"], "ccus_knowledge_v3")
        self.assertEqual(vector["vectors"], 3)
        self.assertEqual(
            vector["collections"],
            [{"name": "ccus_knowledge_v3", "vectors": 3}, {"name": "legacy", "vectors": 1}],
        )

    def test_broken_database_is_reported_as_unavailable(self):
        """读不到就说读不到，不猜一个数字。猜出来的数比"未知"危险得多。"""
        path = Path(self._tmp) / "broken"
        path.mkdir()
        (path / "chroma.sqlite3").write_bytes(b"not a database")
        with mock.patch.dict(os.environ, {"CCUS_CHROMA_PATH": str(path)}):
            vector = self.client.get("/api/kb/overview").json()["vector"]
        self.assertFalse(vector["available"])
        self.assertEqual(vector["vectors"], 0)


# ══════════════════════════════════════════════════════════
# 三、文档清单与原文
# ══════════════════════════════════════════════════════════
class TestDocuments(KBApiTestBase):
    def test_list_includes_files_that_were_never_ingested(self):
        """遍历盘上文件而不是图谱记录 —— 否则那 2 篇根本不会出现在列表里，
        而列表读起来仍然完整。"""
        body = self.client.get("/api/kb/documents").json()
        self.assertEqual(body["total"], 3)
        by_name = {d["name"]: d for d in body["documents"]}
        self.assertTrue(by_name["a.pdf"]["ingested"])
        self.assertEqual(by_name["a.pdf"]["chunks"], 2)
        self.assertEqual(by_name["a.pdf"]["title"], "Seal integrity of caprock")
        self.assertFalse(by_name["b.pdf"]["ingested"])
        self.assertEqual(by_name["b.pdf"]["title"], "", "没有题录就是没有，不拿文件名硬凑")

    def test_detail_returns_chunks_in_document_order(self):
        body = self.client.get("/api/kb/documents/a.pdf").json()
        self.assertEqual(body["total_chunks"], 2)
        self.assertEqual([c["index"] for c in body["chunks"]], [0, 1])
        self.assertEqual(body["chunks"][0]["content"], "第一片")

    def test_focus_centers_on_the_requested_chunk(self):
        body = self.client.get("/api/kb/documents/a.pdf", params={"focus": "chunk-2"}).json()
        self.assertEqual(body["focused"], "chunk-2")
        self.assertIn("chunk-2", [c["chunk_id"] for c in body["chunks"]])

    def test_not_ingested_document_says_why(self):
        r = self.client.get("/api/kb/documents/missing.md")
        self.assertEqual(r.status_code, 404)
        self.assertIn("尚未建图", r.json()["detail"])

    def test_unknown_document_says_so(self):
        r = self.client.get("/api/kb/documents/nope.pdf")
        self.assertEqual(r.status_code, 404)
        self.assertIn("未找到", r.json()["detail"])

    def test_path_traversal_is_rejected(self):
        r = self.client.get("/api/kb/documents/..%5C.env")
        self.assertIn(r.status_code, (400, 404), "不能读到仓库外的文件")

    def test_detail_does_not_need_the_model(self):
        """引用跳转必须能在模型挂掉时用 —— 否则它会在最需要时失效。

        这里没有任何 mock：能返回内容本身就证明它只读了文件。
        """
        self.assertEqual(self.client.get("/api/kb/documents/a.pdf").status_code, 200)


# ══════════════════════════════════════════════════════════
# 四、检索：三种状态必须能分开
# ══════════════════════════════════════════════════════════
class _FakeRetriever:
    """替身：只实现 ``query_evidence``，不加载任何模型。"""

    def __init__(self, *, raises: Exception | None = None, evidence: dict | None = None):
        self.raises = raises
        self.evidence = evidence

    def query_evidence(self, query, mode="hybrid", with_graph=True):
        if self.raises:
            raise self.raises
        return self.evidence


def _search(client: TestClient, fake: _FakeRetriever, **body) -> dict:
    payload = {"query": "盖层完整性", **body}
    with mock.patch.dict(sys.modules, {"lightrag_retriever": fake}):
        r = client.post("/api/kb/search", json=payload)
    assert r.status_code == 200, r.text
    return r.json()


class TestSearchStates(KBApiTestBase):
    def test_unavailable_is_not_empty(self):
        """Ollama 没启动时不能说"库里没有这个内容"。"""
        body = _search(self.client, _FakeRetriever(raises=ConnectionError("Ollama 未启动")))
        self.assertEqual(body["status"], "unavailable")
        self.assertIn("Ollama", body["reason"])
        self.assertEqual(body["evidence"]["stats"]["chunks"], 0)

    def test_empty_is_reported_as_empty(self):
        from kb_evidence import build_evidence

        body = _search(self.client, _FakeRetriever(evidence=build_evidence({})))
        self.assertEqual(body["status"], "empty")
        self.assertIn("没有", body["reason"])

    def test_ok_returns_cards_with_sources(self):
        from kb_evidence import build_evidence

        data = {
            "entities": [],
            "relationships": [],
            "chunks": [
                {
                    "reference_id": "1",
                    "chunk_id": "chunk-1",
                    "file_path": "a.pdf",
                    "content": "盖层完整性是长期封存的关键",
                }
            ],
            "references": [],
        }
        body = _search(self.client, _FakeRetriever(evidence=build_evidence(data)))
        self.assertEqual(body["status"], "ok")
        card = body["evidence"]["chunks"][0]
        self.assertEqual(card["source"], "a.pdf")
        self.assertEqual(card["reference_id"], "1")
        self.assertEqual(body["evidence"]["references"], [{"reference_id": "1", "source": "a.pdf"}])

    def test_blank_query_is_rejected(self):
        r = self.client.post("/api/kb/search", json={"query": "   "})
        self.assertEqual(r.status_code, 400)

    def test_unknown_mode_is_422_not_500(self):
        r = self.client.post("/api/kb/search", json={"query": "x", "mode": "magic"})
        self.assertEqual(r.status_code, 422)


def _load_capture_script():
    """按路径加载 ``scripts/capture_kb_fixtures.py``。

    ``scripts/`` 不是包（没有 ``__init__.py``），但它里面的判定逻辑值得被测 ——
    尤其是"语料不全时该怎么办"，那是一个会静默偏向某一侧的判断。
    """
    import importlib.util
    import pathlib

    path = (
        pathlib.Path(__file__).resolve().parent.parent
        / "scripts"
        / "capture_kb_fixtures.py"
    )
    spec = importlib.util.spec_from_file_location("capture_kb_fixtures", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FixtureCorpusGateTests(unittest.TestCase):
    """语料不全时，fixture 校验必须说"没查"，不能说"不一致"。

    CI 的第一版把 ``documents.json`` 当无条件逐字节比对，而它比的是
    ``ccus_docs/`` 的目录列表 —— 那些 PDF 按设计不入仓库，于是这个 job
    在 CI 里永远红。永远红的检查比没有检查更糟：它会训练所有人忽略整个
    job，连同里面那些真的能跑的检查。

    所以三种情形必须分开，而且分开的方式要能被测试锁住：

    * 本机**多出**文件 → 真的漂移，必须红（fixture 里没有它，前端就不显示它）
    * 本机**缺少**文件 → 环境不同，跳过并说明原因
    * 完全一致 → 逐字节比对
    """

    @classmethod
    def setUpClass(cls):
        import json
        import pathlib

        cls.mod = _load_capture_script()
        path = (
            pathlib.Path(__file__).resolve().parent.parent
            / "web"
            / "src"
            / "fixtures"
            / "documents.json"
        )
        cls.fixture = json.loads(path.read_text(encoding="utf-8"))
        cls.names = [str(r["name"]) for r in cls.fixture["documents"]]

    def _payload(self, names, *, drop=None, add=None):
        template = dict(self.fixture["documents"][0])
        rows = []
        for n in names:
            row = dict(template, name=n)
            if drop:
                row.pop(drop, None)
            if add:
                row[add] = "x"
            rows.append(row)
        return {"total": len(rows), "documents": rows}

    def test_missing_files_are_skipped_with_a_reason(self):
        ok, note = self.mod._corpus_gate(self._payload(self.names[:1]))
        self.assertFalse(ok, "本机语料比 fixture 少时必须跳过逐字节比对")
        self.assertIn("少", note, "跳过时必须说明少了多少 —— 否则读起来像「没查」")

    def test_extra_file_is_a_real_drift_and_fails(self):
        payload = self._payload(self.names + ["ccus_new.pdf"])
        with self.assertRaises(SystemExit) as ctx:
            self.mod._corpus_gate(payload)
        self.assertIn("ccus_new.pdf", str(ctx.exception))

    def test_same_corpus_allows_byte_comparison(self):
        ok, note = self.mod._corpus_gate(self._payload(self.names))
        self.assertTrue(ok, "语料一致时必须走逐字节比对，不能一律跳过")
        self.assertEqual(note, "")

    def test_renamed_field_is_caught_even_when_corpus_differs(self):
        drift = self.mod._shape_drift(self._payload(self.names[:1], drop="chunks"))
        self.assertTrue(
            any("chunks" in d for d in drift),
            "字段改名与语料无关，跳过逐字节不等于这个文件没人管",
        )

    def test_added_field_is_caught(self):
        drift = self.mod._shape_drift(self._payload(self.names[:1], add="chunk_count"))
        self.assertTrue(any("chunk_count" in d for d in drift))


class SearchFixtureLayerTests(unittest.TestCase):
    """``search.json`` 在 CI 里到底被查了**哪一层**。

    这条链路的失败模式特别安静，两个方向都安静：

    * 少说一句「未比对（逐字节）」→ 本机开发一切照旧，CI 绿灯，而那份
      fixture 从此没人比对过。绿灯替没做的检查背了书。
    * 少说一句「已比对（结构与自洽）」→ 真做过的检查被抹掉。下一次有人想
      删掉这段代码时不会有任何阻力，因为看起来它什么也没干。

    所以这两句话本身要被测。测试直接喂 payload 给 ``_check_payloads``，
    不跑 ``capture()`` —— 后者会真的去连 Ollama（本机 21 秒，CI 里没有）。
    """

    @classmethod
    def setUpClass(cls):
        import json
        import pathlib

        cls.mod = _load_capture_script()
        cls.fx = pathlib.Path(__file__).resolve().parent.parent / "web" / "src" / "fixtures"
        cls.search = json.loads((cls.fx / "search.json").read_text(encoding="utf-8"))
        cls.corpus = json.loads(
            (cls.fx / "documents.json").read_text(encoding="utf-8")
        )

    def _ci_lines(self):
        """CI 的分支：没有 Ollama，所以 payloads 里没有 search。"""
        return self.mod._check_payloads(
            {
                "overview.json": json.loads(
                    (self.fx / "overview.json").read_text(encoding="utf-8")
                ),
                "documents.json": self.corpus,
            }
        )

    def _dev_lines(self):
        """开发机的分支：search 是刚重建的，走逐字节。"""
        payloads = {
            "overview.json": json.loads(
                (self.fx / "overview.json").read_text(encoding="utf-8")
            ),
            "documents.json": self.corpus,
            "search.json": self.search,
        }
        return self.mod._check_payloads(payloads)

    # ── 两层都要说 ──

    def test_ci_branch_says_byte_comparison_did_not_happen(self):
        text = "\n".join(self._ci_lines())
        self.assertIn(
            "未比对（逐字节）search.json",
            text,
            "没有 Ollama 时逐字节比对确实没做 —— 不说出来，绿灯就在替它背书",
        )

    def test_ci_branch_says_what_was_still_checked(self):
        text = "\n".join(self._ci_lines())
        self.assertIn(
            "已比对（结构与自洽）search.json",
            text,
            "结构与自洽这一层不需要模型，CI 里真跑了 —— 不说出来，"
            "下次有人删掉这段代码时不会有任何阻力",
        )

    def test_ci_branch_states_the_boundary(self):
        """说清"查不到什么"，比说清"查了什么"更难被写对。"""
        text = "\n".join(self._ci_lines())
        self.assertIn("当前后端会返回的那一份", text)

    def test_dev_branch_does_not_claim_it_was_skipped(self):
        text = "\n".join(self._dev_lines())
        self.assertNotIn("未比对（逐字节）search.json", text)
        self.assertIn("search.json", text)

    # ── 这一层真的能查出东西 ──

    def test_repo_fixture_is_self_consistent(self):
        problems = self.mod._audit_search(self.search, self.mod._fixture_corpus())
        self.assertEqual(problems, [], "仓库里的 search.json 自己跟自己不一致")

    def test_hand_edited_stats_is_caught(self):
        import copy

        payload = copy.deepcopy(self.search)
        payload["evidence"]["stats"]["chunks"] += 1
        problems = self.mod._audit_search(payload, self.mod._fixture_corpus())
        self.assertTrue(any("stats.chunks" in p for p in problems), problems)

    def test_changed_query_is_caught(self):
        import copy

        payload = copy.deepcopy(self.search)
        payload["query"] = "另一个问题"
        problems = self.mod._audit_search(payload, self.mod._fixture_corpus())
        self.assertTrue(any("query" in p for p in problems), problems)

    def test_unzeroed_elapsed_is_caught(self):
        """``elapsed_ms`` 非零说明这份文件不是 capture 写的 —— 它是每次跑都变的。"""
        import copy

        payload = copy.deepcopy(self.search)
        payload["elapsed_ms"] = 21034
        problems = self.mod._audit_search(payload, self.mod._fixture_corpus())
        self.assertTrue(any("elapsed_ms" in p for p in problems), problems)

    def test_chunk_source_outside_corpus_is_caught(self):
        """出处不在语料清单里 → 界面上点开就是 404。两个文件都在仓库里，CI 可查。"""
        import copy

        payload = copy.deepcopy(self.search)
        payload["evidence"]["chunks"][0]["source"] = "not_in_corpus.pdf"
        problems = self.mod._audit_search(payload, self.mod._fixture_corpus())
        self.assertTrue(any("404" in p for p in problems), problems)

    def test_ok_without_content_is_caught(self):
        import copy

        payload = copy.deepcopy(self.search)
        payload["evidence"]["chunks"] = []
        payload["evidence"]["entities"] = []
        payload["evidence"]["relationships"] = []
        problems = self.mod._audit_search(payload, self.mod._fixture_corpus())
        self.assertTrue(any("ok" in p for p in problems), problems)

    def test_empty_without_reason_is_caught(self):
        import copy

        payload = copy.deepcopy(self.search)
        payload["status"] = "unavailable"
        payload["reason"] = ""
        problems = self.mod._audit_search(payload, self.mod._fixture_corpus())
        self.assertTrue(any("原因" in p for p in problems), problems)

    # ── 写入路径：自相矛盾的证据包不许落盘 ──

    def test_write_path_refuses_to_persist_an_inconsistent_payload(self):
        """审了不等于接上了 —— 这条断言的是 ``main`` 里真的调了审计。

        把 ``capture()`` 换成一个返回坏 payload 的桩，然后要求 ``main``
        抛错**并且**没动磁盘上的文件。如果哪天有人为了"先写进去再看"
        把那行审计删掉，这条会红。
        """
        import copy
        import json
        import unittest.mock as mock

        target = self.fx / "search.json"
        before = target.read_text(encoding="utf-8")

        bad = copy.deepcopy(self.search)
        bad["evidence"]["stats"]["chunks"] += 1
        payloads = {
            "overview.json": json.loads(
                (self.fx / "overview.json").read_text(encoding="utf-8")
            ),
            "documents.json": self.corpus,
            "search.json": bad,
        }
        with mock.patch.object(self.mod, "capture", return_value=(payloads, "")):
            with self.assertRaises(SystemExit) as ctx:
                self.mod.main(["capture_kb_fixtures.py"])

        self.assertIn("不一致", str(ctx.exception))
        self.assertEqual(
            target.read_text(encoding="utf-8"),
            before,
            "审计没过就不该落盘 —— 写进去之后它就是前端唯一的数据来源",
        )


if __name__ == "__main__":
    unittest.main()
