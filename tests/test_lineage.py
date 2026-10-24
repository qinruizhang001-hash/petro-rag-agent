"""
文档血缘（file_path）回归测试
==============================
背景：``insert_text`` 早先不传 ``file_paths``，LightRAG 就把每篇文档的
``file_path`` 记成 ``unknown_source``。后果不是报错，而是**知识库说不清
自己从哪来**：

- 报告里引用的任何一句话，都无法回溯到具体 PDF；
- ``scripts/corpus_audit.py`` 打出来的可追溯率是 0%（实测 23/23 篇全中）；
- 想删掉某篇论文的影响（版权、错误结论）时，只能整库重建。

这类缺陷在功能测试里完全看不见 —— 检索照样返回结果，只是每条结果都
失去了出处。所以用两条互补的检查钉住它：

1. 行为检查：``insert_text`` 必须把 ``file_paths`` 透传给 LightRAG；
2. 调用点检查：``ingest_ccus_batch`` 的建图主流程必须真的传这个参数
   （只改接口不传参，等于没修）。

lightrag 是重依赖（会连带拉起向量库与嵌入模型），这里用 ``sys.modules``
桩替身，保证本测试在任何环境都能跑。
"""
from __future__ import annotations

import os
import sys
import types
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def _install_lightrag_stub() -> None:
    """在 import lightrag_retriever 之前把 lightrag 换成桩。"""
    if "lightrag" in sys.modules:
        return

    pkg = types.ModuleType("lightrag")

    class _FakeRAG:  # pragma: no cover - 仅用于占位，实例不会被创建
        def __init__(self, *a, **kw):
            pass

    class _FakeQueryParam:
        def __init__(self, *a, **kw):
            pass

    pkg.LightRAG = _FakeRAG
    pkg.QueryParam = _FakeQueryParam

    llm = types.ModuleType("lightrag.llm")
    ollama = types.ModuleType("lightrag.llm.ollama")

    async def _noop_complete(*a, **kw):  # pragma: no cover
        return ""

    async def _noop_embed(*a, **kw):  # pragma: no cover
        return []

    ollama.ollama_model_complete = _noop_complete
    ollama.ollama_embed = _noop_embed
    llm.ollama = ollama

    utils = types.ModuleType("lightrag.utils")

    class _EmbeddingFunc:
        def __init__(self, *a, **kw):
            pass

    utils.EmbeddingFunc = _EmbeddingFunc

    pkg.llm = llm
    pkg.utils = utils
    sys.modules["lightrag"] = pkg
    sys.modules["lightrag.llm"] = llm
    sys.modules["lightrag.llm.ollama"] = ollama
    sys.modules["lightrag.utils"] = utils


_install_lightrag_stub()

import lightrag_retriever  # noqa: E402


class _CapturingBridge:
    """只记录调用参数的假 bridge，不碰任何存储。"""

    def __init__(self):
        self.calls = []

    def insert_text(self, text, timeout=1800, file_paths=None):
        self.calls.append({"text": text, "timeout": timeout, "file_paths": file_paths})


class InsertTextForwardsLineageTests(unittest.TestCase):
    def setUp(self):
        self.bridge = _CapturingBridge()
        self._orig = lightrag_retriever._get_bridge
        lightrag_retriever._get_bridge = lambda: self.bridge

    def tearDown(self):
        lightrag_retriever._get_bridge = self._orig

    def test_file_paths_reach_the_bridge(self):
        ok = lightrag_retriever.insert_text(
            "Title: x\n\nbody", file_paths=["ccus_docs/ccus1.pdf"]
        )
        self.assertTrue(ok)
        self.assertEqual(self.bridge.calls[0]["file_paths"], ["ccus_docs/ccus1.pdf"])

    def test_omitting_file_paths_still_works_but_is_visible(self):
        """向后兼容：不传不报错，但默认值必须是 None（而不是伪造一个路径）。

        默认值若被写成 "unknown_source"，血缘审计会把"没传"误判成"传了"，
        这正是要避免的静默失败。
        """
        ok = lightrag_retriever.insert_text("body only")
        self.assertTrue(ok)
        self.assertIsNone(self.bridge.calls[0]["file_paths"])

    def test_bridge_passes_file_paths_into_ainsert(self):
        """真正落到 LightRAG 的那一步也要带参数，否则前面透传是空转。"""
        seen = {}

        class _FakeRag:
            def ainsert(self, text, file_paths=None):
                seen["text"] = text
                seen["file_paths"] = file_paths

        bridge = lightrag_retriever._LightRAGBridge.__new__(lightrag_retriever._LightRAGBridge)
        bridge._rag = _FakeRag()
        bridge._submit = lambda fn, timeout=None: fn()

        bridge.insert_text("body", file_paths=["a.pdf"])
        self.assertEqual(seen["file_paths"], ["a.pdf"])

    def test_failure_is_reported_not_raised(self):
        """单篇失败不能中断批量入库 —— 返回值必须是 False，而不是抛异常。"""

        def _boom(*a, **kw):
            raise RuntimeError("ollama down")

        lightrag_retriever._get_bridge = lambda: types.SimpleNamespace(
            insert_text=_boom
        )
        self.assertFalse(lightrag_retriever.insert_text("body", file_paths=["a.pdf"]))


class IngestCallSitePassesLineageTests(unittest.TestCase):
    """接口改对了但调用点不传参，等于没修。这条检查守的是调用点。"""

    def test_sync_graph_passes_file_paths(self):
        path = os.path.join(PROJECT_ROOT, "ingest_ccus_batch.py")
        with open(path, encoding="utf-8") as fh:
            src = fh.read()
        self.assertIn("insert_text(payload, file_paths=", src)
        # 文件名必须来自论文条目，不能是写死的字符串
        self.assertIn("it['file']", src.split("insert_text(payload, file_paths=")[1][:120])


if __name__ == "__main__":
    unittest.main(verbosity=2)
