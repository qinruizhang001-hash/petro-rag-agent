"""
离线单元测试：GraphRAG 片段构建器
================================
不需要 Ollama / 模型 / 图谱索引，纯内存断言，可在 CI 里秒级跑完。

运行：
    python -m unittest discover -s tests -v
或：
    python tests/test_fragment_builder.py
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fragment_builder import build_entity_map, build_fragments, extract_data


class TestBuildEntityMap(unittest.TestCase):
    def test_builds_type_prefixed_description(self):
        entities = [
            {"entity_name": "Fault activation", "entity_type": "Geological process",
             "description": "Shear slip along a pre-existing fault plane."},
            {"entity_name": "Caprock", "entity_type": "Geological formation",
             "description": "Low-permeability seal above the reservoir."},
        ]
        m = build_entity_map(entities)
        self.assertIn("Fault activation", m)
        self.assertTrue(m["Fault activation"].startswith("[Geological process]"))
        self.assertIn("Shear slip", m["Fault activation"])

    def test_skips_entries_without_name(self):
        m = build_entity_map([{"entity_name": "", "description": "orphan"}])
        self.assertEqual(m, {})

    def test_handles_missing_description_and_type(self):
        m = build_entity_map([{"entity_name": "CO2"}])
        self.assertEqual(m["CO2"], "[UNKNOWN] CO2")


class TestBuildFragments(unittest.TestCase):
    def _payload(self):
        return {
            "entities": [
                {"entity_name": "Fault activation", "entity_type": "Process",
                 "description": "Shear slip along a pre-existing fault plane caused by pore pressure buildup."},
                {"entity_name": "Caprock integrity", "entity_type": "Property",
                 "description": "The ability of a sealing formation to prevent CO2 migration."},
                {"entity_name": "Microseismicity", "entity_type": "Phenomenon",
                 "description": "Small-magnitude seismic events induced by injection."},
            ],
            "relationships": [
                {"src_id": "Fault activation", "tgt_id": "Caprock integrity",
                 "keywords": "compromises", "description": "Fault slip can breach the caprock seal."},
            ],
            "chunks": [{"content": "Injection at 250 bar raises pore pressure."}],
        }

    def test_relation_fragment_contains_both_entity_descriptions(self):
        docs = build_fragments(self._payload())
        rel = [d for d in docs if d.startswith("[关系]")]
        self.assertEqual(len(rel), 1)
        frag = rel[0]
        # 两端实体的描述都必须进入片段 —— 这是 rerank 区分度的来源
        self.assertIn("Shear slip along a pre-existing fault plane", frag)
        self.assertIn("ability of a sealing formation", frag)
        # 关系语义与关键词也必须保留
        self.assertIn("--[compromises]-->", frag)
        self.assertIn("breach the caprock seal", frag)

    def test_relation_fragment_is_long_enough_for_reranker(self):
        docs = build_fragments(self._payload())
        rel = [d for d in docs if d.startswith("[关系]")][0]
        self.assertGreater(len(rel), 80, "关系片段过短，rerank 会失去区分度")

    def test_uncovered_entity_kept_as_isolated_fragment(self):
        docs = build_fragments(self._payload())
        isolated = [d for d in docs if d.startswith("[实体]")]
        self.assertEqual(len(isolated), 1)
        self.assertIn("Microseismicity", isolated[0])

    def test_covered_entities_not_duplicated_as_isolated(self):
        docs = build_fragments(self._payload())
        for d in docs:
            if d.startswith("[实体]"):
                self.assertNotIn("Fault activation", d)
                self.assertNotIn("Caprock integrity", d)

    def test_chunks_appended_verbatim(self):
        docs = build_fragments(self._payload())
        self.assertIn("Injection at 250 bar raises pore pressure.", docs)

    def test_order_is_relations_then_entities_then_chunks(self):
        docs = build_fragments(self._payload())
        kinds = []
        for d in docs:
            if d.startswith("[关系]"):
                kinds.append("rel")
            elif d.startswith("[实体]"):
                kinds.append("ent")
            else:
                kinds.append("chunk")
        self.assertEqual(kinds, ["rel", "ent", "chunk"])

    def test_deduplicates_preserving_order(self):
        payload = {
            "entities": [{"entity_name": "A", "entity_type": "T", "description": "d"}],
            "relationships": [],
            "chunks": [{"content": "same text"}, {"content": "same text"}],
        }
        docs = build_fragments(payload)
        self.assertEqual(docs.count("same text"), 1)

    def test_empty_payload_returns_empty_list(self):
        self.assertEqual(build_fragments({}), [])
        self.assertEqual(build_fragments(None), [])

    def test_relationship_without_keywords_uses_plain_arrow(self):
        payload = {
            "entities": [
                {"entity_name": "A", "entity_type": "T", "description": "desc a"},
                {"entity_name": "B", "entity_type": "T", "description": "desc b"},
            ],
            "relationships": [{"src_id": "A", "tgt_id": "B", "description": "linked"}],
            "chunks": [],
        }
        docs = build_fragments(payload)
        rel = [d for d in docs if d.startswith("[关系]")][0]
        self.assertIn("-->", rel)
        self.assertNotIn("--[", rel)

    def test_unknown_endpoint_falls_back_to_raw_id(self):
        payload = {
            "entities": [],
            "relationships": [{"src_id": "X", "tgt_id": "Y", "keywords": "k", "description": "d"}],
            "chunks": [],
        }
        docs = build_fragments(payload)
        self.assertIn("X", docs[0])
        self.assertIn("Y", docs[0])


class TestExtractData(unittest.TestCase):
    def test_returns_data_on_success(self):
        self.assertEqual(extract_data({"status": "success", "data": {"a": 1}}), {"a": 1})

    def test_returns_empty_on_failure(self):
        self.assertEqual(extract_data({"status": "failure"}), {})
        self.assertEqual(extract_data(None), {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
