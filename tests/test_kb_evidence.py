"""知识库证据构建器（kb_evidence）的测试。

这一组要守住的东西，按重要性排：

1. **多值血缘不能被当成单个文件名**。实体的 ``file_path`` 是
   ``<SEP>`` 拼接的多来源（实测 ``CO2 Leakage`` 来自 16 篇）。第一版直接
   把它塞进一个字符串字段，界面上那一栏显示成一个既不像文件名也不能点开的
   长条，而且它**看起来像**一个文件名 —— 读者会以为"只来自一篇，只是名字长"。
2. **``unknown_source`` 要被数出来**。没写血缘的切片在界面上与正常证据
   长得一模一样，只是文件名那栏是空的。少一个文件名不构成任何提示，
   所以必须有一个 ``stats.unsourced`` 让它变成可显示的数字。
3. **``reference_id`` 是响应内编号，不是文档 ID**。来源表按它去重，
   但同一个文件占多个编号时**不能合并** —— 合并了正文角标就对不上。
4. **截断必须留痕**。``truncated=True`` 是让界面能说"已截断"的唯一依据。
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kb_evidence import (
    DESC_LIMIT,
    GRAPH_SEP,
    build_entities,
    build_evidence,
    build_references,
    build_relationships,
    normalize_source,
    split_sources,
    text_quality,
)


def _data(**over) -> dict:
    """一份最小的 LightRAG ``data`` 段（形状取自 convert_to_user_format）。"""
    base = {
        "entities": [
            {
                "entity_name": "CO2 Leakage",
                "entity_type": "risk",
                "description": "泄漏风险描述",
                # 多来源：这是实测形状，不是构造的极端值
                "file_path": f"ccus_auto_01.pdf{GRAPH_SEP}ccus_auto_02.pdf{GRAPH_SEP}ccus3.pdf",
                "source_id": f"chunk-a{GRAPH_SEP}chunk-b",
                "created_at": "",
            }
        ],
        "relationships": [
            {
                "src_id": "Caprock",
                "tgt_id": "CO2 Leakage",
                "description": "盖层完整性下降会提高泄漏风险",
                "keywords": "seal integrity,leakage risk",
                "file_path": f"ccus_auto_03.pdf{GRAPH_SEP}ccus_auto_04.pdf",
                "source_id": "chunk-c",
            }
        ],
        "chunks": [
            {
                "reference_id": "1",
                "chunk_id": "chunk-a",
                "file_path": "ccus_auto_01.pdf",
                "content": "原文切片一",
            },
            {
                "reference_id": "2",
                "chunk_id": "chunk-b",
                "file_path": "ccus_auto_02.pdf",
                "content": "原文切片二",
            },
        ],
        "references": [
            {"reference_id": "1", "file_path": "ccus_auto_01.pdf"},
            {"reference_id": "2", "file_path": "ccus_auto_02.pdf"},
        ],
    }
    base.update(over)
    return base


class TestSources(unittest.TestCase):
    def test_unknown_source_normalizes_to_empty(self):
        self.assertEqual(normalize_source("unknown_source"), "")
        self.assertEqual(normalize_source(""), "")
        self.assertEqual(normalize_source(None), "")
        self.assertEqual(normalize_source("ccus1.pdf"), "ccus1.pdf")

    def test_split_sources_breaks_sep_joined_paths(self):
        raw = f"a.pdf{GRAPH_SEP}b.pdf{GRAPH_SEP}c.pdf"
        self.assertEqual(split_sources(raw), ["a.pdf", "b.pdf", "c.pdf"])

    def test_split_sources_dedupes_and_drops_unknown(self):
        raw = f"a.pdf{GRAPH_SEP}unknown_source{GRAPH_SEP}a.pdf"
        self.assertEqual(split_sources(raw), ["a.pdf"])

    def test_entity_carries_a_list_not_a_joined_string(self):
        """回归：实体曾经把 16 个文件名糊成一个字段。"""
        card = build_entities(_data())[0]
        self.assertIsInstance(card["sources"], list)
        self.assertEqual(len(card["sources"]), 3)
        self.assertNotIn(GRAPH_SEP, "".join(card["sources"]))

    def test_relationship_carries_a_list(self):
        card = build_relationships(_data())[0]
        self.assertEqual(card["sources"], ["ccus_auto_03.pdf", "ccus_auto_04.pdf"])

    def test_chunk_carries_a_single_source(self):
        """切片只属于一篇文档，所以是标量 —— 与实体/关系不同，这是有意的。"""
        card = build_evidence(_data())["chunks"][0]
        self.assertEqual(card["source"], "ccus_auto_01.pdf")


class TestReferences(unittest.TestCase):
    def test_reference_table_maps_id_to_source(self):
        refs = build_references(build_evidence(_data())["chunks"])
        self.assertEqual(
            refs,
            [
                {"reference_id": "1", "source": "ccus_auto_01.pdf"},
                {"reference_id": "2", "source": "ccus_auto_02.pdf"},
            ],
        )

    def test_same_file_with_two_ids_is_not_merged(self):
        """同一个文件占两个编号时必须保留两行。

        编号是正文角标的锚点。按文件名去重会让第二条角标指向不存在的行，
        而界面上只是少了一行来源 —— 不会报错。
        """
        chunks = [
            {"reference_id": "1", "source": "a.pdf"},
            {"reference_id": "2", "source": "a.pdf"},
        ]
        self.assertEqual(len(build_references(chunks)), 2)

    def test_duplicate_id_is_collapsed(self):
        chunks = [
            {"reference_id": "1", "source": "a.pdf"},
            {"reference_id": "1", "source": "a.pdf"},
        ]
        self.assertEqual(len(build_references(chunks)), 1)


class TestStats(unittest.TestCase):
    def test_unsourced_chunks_are_counted(self):
        data = _data()
        data["chunks"][1]["file_path"] = "unknown_source"
        stats = build_evidence(data)["stats"]
        self.assertEqual(stats["chunks"], 2)
        self.assertEqual(stats["unsourced"], 1)
        self.assertEqual(stats["traceable"], 1)

    def test_all_traceable_reports_zero_unsourced(self):
        stats = build_evidence(_data())["stats"]
        self.assertEqual(stats["unsourced"], 0)
        self.assertEqual(stats["traceable"], 2)

    def test_sources_counts_distinct_files(self):
        self.assertEqual(build_evidence(_data())["stats"]["sources"], 2)


class TestGraphToggle(unittest.TestCase):
    def test_with_graph_false_returns_only_chunks(self):
        ev = build_evidence(_data(), with_graph=False)
        self.assertTrue(ev["chunks"])
        self.assertEqual(ev["entities"], [])
        self.assertEqual(ev["relationships"], [])

    def test_empty_data_is_not_an_error(self):
        """空 data 是"没有结果"，不是崩溃 —— 端点要靠它判 empty。"""
        ev = build_evidence({})
        self.assertEqual(ev["chunks"], [])
        self.assertEqual(ev["stats"]["chunks"], 0)


class TestTruncation(unittest.TestCase):
    def test_long_description_is_clipped_and_flagged(self):
        data = _data()
        data["entities"][0]["description"] = "长" * (DESC_LIMIT + 100)
        card = build_entities(data)[0]
        self.assertTrue(card["truncated"])
        self.assertLessEqual(len(card["description"]), DESC_LIMIT + 1)

    def test_short_description_is_not_flagged(self):
        self.assertFalse(build_entities(_data())[0]["truncated"])

    def test_empty_chunk_content_is_dropped(self):
        data = _data()
        data["chunks"][0]["content"] = "   "
        self.assertEqual(len(build_evidence(data)["chunks"]), 1)


#: 实测样本（`web/src/fixtures/search.json` 里排名第一的那条）。
#: 它不是构造的极端值 —— 查询「盖层完整性对 CO2 泄漏风险的影响」时，
#: 检索器把这条公式块排到了第一位，因为它满篇都是 CO2 / storage / pressure。
_FORMULA_CHUNK = (
    "\ufffd\U0001d461\U0001d461+1 = \U0001d436\U0001d436(\U0001d465\U0001d465\U0001d461\U0001d461)"
    "\U0001d465\U0001d465\U0001d461\U0001d461+1 + \U0001d437\U0001d437(\U0001d465\U0001d465\U0001d461\U0001d461)"
    "\U0001d462\U0001d462\U0001d461\U0001d461 \n( 4 ) \nWhere \U0001d434\U0001d434(\U0001d465\U0001d465"
    "\U0001d461\U0001d461) =\n\U0001d715\U0001d715\n\U0001d715\U0001d715\U0001d465\U0001d465\U0001d461"
    "\U0001d461, \U0001d435\U0001d435(\U0001d465\U0001d465\U0001d461\U0001d461) =\n\U0001d715\U0001d715"
    "\n\U0001d715\U0001d715\U0001d462\U0001d462\U0001d461\U0001d461. \nIn CO2 storage operations, "
    "specifically for CO2-water system, state variables"
)

_PROSE_CHUNK = (
    "as pressure maintenance, elevation effects, velocity and density of the injected CO2 "
    "are all accounted for in the model described above, and the caprock integrity is "
    "assessed against the resulting stress path."
)


class TestTextQuality(unittest.TestCase):
    """切片能不能读成句子 —— 因为"语义邻近"不等于"能当证据用"。

    这一组存在的原因是一次实测：问「盖层完整性对 CO2 泄漏风险的影响」，
    检索器排第一的是一条 ``𝑡+1 = 𝐶(𝑥𝑡)𝑥𝑡+1 + 𝐷(𝑥𝑡)𝑢𝑡`` 的离散格式公式。
    向量相似度确实高（满篇 CO2 / storage / pressure），但它对"盖层完整性"
    提供不了任何机理证据。界面上如果照样把它当第一条引用，读者会以为
    系统找到了支持。

    刻意**不重排**（理由见 kb_evidence 模块文档）：引用编号必须等于检索名次，
    否则报告里的 ``[1]`` 回不到检索日志。所以只标注、不改变顺序 ——
    与 ``indeterminate`` 的处理方式一致：不改结论，把"能证明什么"说清楚。
    """

    def test_formula_block_is_flagged(self):
        q = text_quality(_FORMULA_CHUNK)
        self.assertTrue(q["low"], "公式块必须被标出，否则它会作为第一条证据出现在面板上")
        self.assertGreater(q["math_ratio"], 0.15)

    def test_prose_is_not_flagged(self):
        q = text_quality(_PROSE_CHUNK)
        self.assertFalse(q["low"], "纯散文被误标会让这个标记失去意义")

    def test_prose_with_inline_math_is_not_flagged(self):
        # 实测里有一条 84.9 % 字母 / 7.8 % 数学符号的"散文夹行内公式"。
        # 它是正常的证据，必须不被标出 —— 否则阈值就画错了位置。
        mixed = _PROSE_CHUNK + " The relation 𝑞 = 𝑛·𝐽·(𝑝 − 𝑃) is used throughout."
        q = text_quality(mixed)
        self.assertFalse(q["low"])

    def test_empty_text_is_not_low_quality(self):
        # 空切片是"没内容"，不是"以公式为主"。两者在界面上的处置不同：
        # 前者要重新检索，后者要换个读法。
        self.assertFalse(text_quality("   ")["low"])

    def test_quality_is_computed_on_full_content_not_the_clip(self):
        # 截断会把后半段散文切掉，让一条正常切片因为前 2000 字里公式多而
        # 显得"以公式为主"。所以质量按完整正文算。
        data = _data()
        data["chunks"][0]["content"] = _FORMULA_CHUNK * 3 + _PROSE_CHUNK * 60
        card = build_evidence(data)["chunks"][0]
        self.assertTrue(card["truncated"])
        self.assertFalse(card["text_quality"]["low"])

    def test_low_quality_count_is_reported(self):
        data = _data()
        data["chunks"][0]["content"] = _FORMULA_CHUNK
        stats = build_evidence(data)["stats"]
        self.assertEqual(stats["low_quality_chunks"], 1)

    def test_clean_corpus_reports_zero(self):
        data = _data()
        for c in data["chunks"]:
            c["content"] = _PROSE_CHUNK
        self.assertEqual(build_evidence(data)["stats"]["low_quality_chunks"], 0)

    def test_every_chunk_carries_a_quality_field(self):
        # 缺字段会让界面上的标记静默消失（`undefined` 是 falsy）
        for card in build_evidence(_data())["chunks"]:
            self.assertIn("text_quality", card)
            self.assertIn("low", card["text_quality"])



if __name__ == "__main__":
    unittest.main()
