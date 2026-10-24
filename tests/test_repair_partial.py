# -*- coding: utf-8 -*-
"""
``scripts/repair_partial_ingest.py`` 的回归测试。

锁定的是一条**静默失败**：入库被中断后，``full_docs`` 里有正文但零 chunk，
重跑时 LightRAG 只看 ``full_docs`` 就直接 return —— 不报错、不入库，而入库
脚本还会打 ✅。测试要保证：

1. 只认「状态未完成 **且** 零 chunk」为残留；
2. 有 chunk 落盘的文档一律不动（实体可能已进图，清了反而丢内容）；
3. 清理是幂等的、原子的，且不碰 chunk 存储与图谱。
"""

import json
import os
import sys
import tempfile
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.repair_partial_ingest import find_partial, repair  # noqa: E402


def _write(path, data):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False)


class _Storage:
    """一个最小的图谱存储夹具：三个 JSON 文件，够 LightRAG 判定用。"""

    def __init__(self):
        self.dir = tempfile.mkdtemp(prefix="repair-test-")

    def build(self, docs):
        """docs: {doc_id: (status, file_path, n_chunks)}"""
        full, status, chunks = {}, {}, {}
        for doc_id, (st, fp, n) in docs.items():
            full[doc_id] = {"content": "x" * 100, "file_path": fp}
            status[doc_id] = {
                "status": st,
                "file_path": fp,
                "chunks_list": [f"chunk-{doc_id}-{i}" for i in range(n)],
            }
            for i in range(n):
                chunks[f"chunk-{doc_id}-{i}"] = {"full_doc_id": doc_id, "content": "y"}
        _write(os.path.join(self.dir, "kv_store_full_docs.json"), full)
        _write(os.path.join(self.dir, "kv_store_doc_status.json"), status)
        _write(os.path.join(self.dir, "kv_store_text_chunks.json"), chunks)
        return self

    def read(self, name):
        with open(os.path.join(self.dir, name), encoding="utf-8") as fh:
            return json.load(fh)


class PartialDetectionTests(unittest.TestCase):
    def test_half_ingested_document_is_found(self):
        """状态停在 processing、正文已落盘、零 chunk —— 这才是残留。"""
        st = _Storage().build(
            {
                "doc-good": ("processed", "good.pdf", 10),
                "doc-half": ("processing", "half.pdf", 0),
            }
        )
        partial = find_partial(st.dir)
        self.assertEqual([p["file_path"] for p in partial], ["half.pdf"])
        self.assertEqual(partial[0]["status"], "processing")
        self.assertEqual(partial[0]["stored_chunks"], 0)
        self.assertTrue(partial[0]["in_full_docs"])

    def test_document_with_chunks_is_never_touched(self):
        """
        有 chunk 落盘就不清，即使状态不是 processed。

        理由：chunk 在，实体抽取可能已经部分合并进图；重跑时实体按名字合并
        不会重复，清掉反而会丢已入库的内容。宁可留一份半成品，不要删掉。
        """
        st = _Storage().build({"doc-x": ("failed", "x.pdf", 3)})
        self.assertEqual(find_partial(st.dir), [])

    def test_processed_documents_are_left_alone(self):
        st = _Storage().build({"doc-ok": ("processed", "ok.pdf", 5)})
        self.assertEqual(find_partial(st.dir), [])

    def test_dry_run_changes_nothing(self):
        st = _Storage().build({"doc-half": ("processing", "half.pdf", 0)})
        before = st.read("kv_store_full_docs.json")
        res = repair(st.dir, apply=False)
        self.assertEqual(len(res["partial"]), 1)
        self.assertFalse(res["applied"])
        self.assertEqual(st.read("kv_store_full_docs.json"), before)


class RepairTests(unittest.TestCase):
    def test_apply_removes_records_so_reingest_can_see_the_document(self):
        st = _Storage().build(
            {
                "doc-good": ("processed", "good.pdf", 4),
                "doc-half": ("processing", "half.pdf", 0),
            }
        )
        res = repair(st.dir, apply=True)
        self.assertEqual(res["removed"], ["half.pdf"])
        self.assertNotIn("doc-half", st.read("kv_store_full_docs.json"))
        self.assertNotIn("doc-half", st.read("kv_store_doc_status.json"))
        # 正常文档不受影响
        self.assertIn("doc-good", st.read("kv_store_full_docs.json"))

    def test_apply_does_not_touch_the_chunk_store(self):
        """清理只删两条记录，chunk 存储必须原样保留（那是别人的内容）。"""
        st = _Storage().build(
            {
                "doc-good": ("processed", "good.pdf", 4),
                "doc-half": ("processing", "half.pdf", 0),
            }
        )
        chunks_before = st.read("kv_store_text_chunks.json")
        repair(st.dir, apply=True)
        self.assertEqual(st.read("kv_store_text_chunks.json"), chunks_before)

    def test_repair_is_idempotent(self):
        st = _Storage().build({"doc-half": ("processing", "half.pdf", 0)})
        repair(st.dir, apply=True)
        second = repair(st.dir, apply=True)
        self.assertEqual(second["partial"], [])
        self.assertEqual(second["removed"], [])

    def test_missing_storage_is_not_an_error(self):
        """存储目录还不存在（首次部署）时应安静返回，而不是抛异常。"""
        empty = tempfile.mkdtemp(prefix="repair-empty-")
        self.assertEqual(find_partial(empty), [])


if __name__ == "__main__":
    unittest.main()
