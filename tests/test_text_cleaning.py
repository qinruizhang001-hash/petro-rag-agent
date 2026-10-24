"""
utils.text_cleaning 的离线单测（纯标准库，CI 直接跑）
=====================================================
这些测试针对的不是"格式好看"，而是**静默失效**：

  - 断词没修 → "permea ability" 进了嵌入模型，检索从此查不到 permeability
  - 页码没删 → 大量 "12" "13" 成为独立 chunk，占据召回名额
  - 参考文献没砍 → 几十条文献条目被当成正文抽实体，图谱里长出一堆人名
  - 切片丢字符 → 关键数值恰好落在切口上，报告里就"没有这个数据"

以上任何一种都不会抛异常，只会让结果变差。所以必须锁住。
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.text_cleaning import (
    chunk_text,
    clean_pdf_text,
    graph_text,
    select_graph_text,
    strip_front_matter,
    strip_references,
)


class TestCleanPdfText(unittest.TestCase):
    def test_arxiv_stamp_removed(self):
        raw = (
            "arXiv:2606.27515v1 [physics.geo-ph] 25 Jun 2026\n"
            "Carbon capture and storage in saline aquifers is constrained by caprock."
        )
        out = clean_pdf_text(raw)
        self.assertNotIn("arXiv:", out)
        self.assertNotIn("physics.geo-ph", out)
        self.assertIn("Carbon capture and storage", out)

    def test_page_numbers_removed(self):
        raw = "First paragraph about CO2 storage.\n\n12\n\nSecond paragraph about faults.\n\n13\n"
        out = clean_pdf_text(raw)
        self.assertNotIn("\n12\n", out)
        self.assertNotIn("\n13", out)
        self.assertIn("First paragraph", out)
        self.assertIn("Second paragraph", out)

    def test_hyphenated_line_break_is_joined(self):
        """断词修复：perme-\\nability 必须变成 permeability，而不是 perme ability。"""
        raw = "The relative perme-\nability controls multiphase flow in the reservoir."
        out = clean_pdf_text(raw)
        self.assertIn("permeability", out)
        self.assertNotIn("perme ability", out)

    def test_hyphen_fix_happens_before_whitespace_collapse(self):
        """
        顺序回归：如果先压平换行再修断词，就没有 "\\n" 可依据了。
        这条测试锁的是调用顺序，不是正则本身。
        """
        raw = "caprock-\nintegrity is a leakage risk"
        self.assertIn("caprockintegrity", clean_pdf_text(raw))

    def test_capitalised_hyphen_is_not_joined(self):
        """CO2-\\nBased 这种是真连字符，不该被粘成 CO2Based。"""
        out = clean_pdf_text("CO2-\nBased storage")
        self.assertNotIn("CO2Based", out)

    def test_empty_input(self):
        self.assertEqual(clean_pdf_text(""), "")
        self.assertEqual(clean_pdf_text(None), "")

    def test_references_dropped_by_default(self):
        body = "Injection pressure must stay below fracture pressure. " * 12
        raw = body + "\n\nReferences\n[1] Smith et al. 2020.\n[2] Jones et al. 2021.\n"
        out = clean_pdf_text(raw)
        self.assertIn("Injection pressure", out)
        self.assertNotIn("Smith et al", out)

    def test_references_kept_when_disabled(self):
        body = "Injection pressure must stay below fracture pressure. " * 12
        raw = body + "\n\nReferences\n[1] Smith et al. 2020.\n"
        out = clean_pdf_text(raw, drop_references=False)
        self.assertIn("Smith et al", out)


class TestStripReferences(unittest.TestCase):
    def test_heading_in_first_half_is_not_treated_as_reference_list(self):
        """
        正文里出现 "References" 标题（比如一篇讲文献计量学的论文）时，
        不能把前半段正文当参考文献砍掉。
        """
        text = "References\n" + "body text about CO2 storage. " * 60
        self.assertEqual(strip_references(text), text)

    def test_last_heading_wins(self):
        """多次出现时取最后一个 —— 真正的文献列表在末尾。"""
        text = (
            "body text about CO2 storage. " * 40
            + "\nReferences\n"
            + "more body. " * 40
            + "\nReferences\n[1] Real citation.\n"
        )
        out = strip_references(text)
        self.assertNotIn("Real citation", out)
        self.assertIn("more body", out)

    def test_no_heading_returns_unchanged(self):
        text = "no reference heading here at all"
        self.assertEqual(strip_references(text), text)


class TestChunkText(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(chunk_text(""), [])

    def test_short_text_single_chunk(self):
        self.assertEqual(chunk_text("short text", size=1200), ["short text"])

    def test_no_paragraph_is_lost(self):
        paras = [f"Paragraph {i} talks about caprock integrity and fault reactivation." for i in range(20)]
        chunks = chunk_text("\n\n".join(paras), size=300, overlap=50)
        joined = "\n".join(chunks)
        for i in range(20):
            self.assertIn(f"Paragraph {i}", joined)

    def test_chunks_respect_size(self):
        paras = ["x" * 200 for _ in range(20)]
        chunks = chunk_text("\n\n".join(paras), size=300, overlap=50)
        # 硬切 + 重叠可能略微超出，但不应成倍超出
        for c in chunks:
            self.assertLessEqual(len(c), 300 + 60)

    def test_long_paragraph_is_hard_split_without_losing_chars(self):
        """单段超长（整页被抽成一个段落）时硬切，且不丢字符。"""
        text = "".join(str(i % 10) for i in range(1000))
        chunks = chunk_text(text, size=300, overlap=0)
        self.assertGreater(len(chunks), 1)
        self.assertEqual("".join(chunks), text)

    def test_overlap_creates_shared_content(self):
        paras = ["A" * 250, "B" * 250, "C" * 250]
        chunks = chunk_text("\n\n".join(paras), size=300, overlap=50)
        self.assertGreater(len(chunks), 1)
        # 相邻块之间应有共享尾巴
        self.assertTrue(chunks[0][-30:] in chunks[1])

    def test_invalid_args_raise(self):
        with self.assertRaises(ValueError):
            chunk_text("abc", size=0)
        with self.assertRaises(ValueError):
            chunk_text("abc", size=100, overlap=100)
        with self.assertRaises(ValueError):
            chunk_text("abc", size=100, overlap=-1)


class TestSelectGraphText(unittest.TestCase):
    def test_returns_full_text_when_under_budget(self):
        self.assertEqual(select_graph_text("short text", 100), "short text")

    def test_cuts_at_sentence_boundary(self):
        """
        句末标点落在窗口中段时也要在句号处收尾，不能硬切到词中间。

        早期实现只在窗口最后 20% 找句号，结果这种情况会退化成硬截，
        把 "fracture pres|sure" 切成两半。
        """
        text = "First sentence about caprock. Second sentence about faults. " + "x" * 500
        out = select_graph_text(text, 120)
        self.assertLessEqual(len(out), 120)
        self.assertTrue(out.endswith("."), f"应以句末标点结束，实际结尾: {out[-20:]!r}")
        self.assertIn("faults", out)

    def test_zero_budget_returns_empty(self):
        self.assertEqual(select_graph_text("anything", 0), "")

    def test_no_sentence_boundary_still_cuts(self):
        out = select_graph_text("y" * 1000, 100)
        self.assertEqual(len(out), 100)


class TestGraphText(unittest.TestCase):
    """
    入库脚本喂多少字进图，是这次知识库只有 6.2% 覆盖率的直接原因。

    历史事故：``sync_graph`` 直接调 ``select_graph_text(text, budget)``，
    而 ``--graph-budget 0`` 在那里的语义是"空串"，于是"全文"被读成"什么都不喂"。
    不会报错，只会建出一个只有标题的空图 —— 正是最难发现的那类 bug。
    """

    def test_zero_budget_means_full_text_not_empty(self):
        text = "a" * 5000
        self.assertEqual(graph_text(text, 0), text)

    def test_negative_budget_also_means_full_text(self):
        # 负数不该被理解成"倒着切"，只该被理解成"不设上限"。
        text = "b" * 5000
        self.assertEqual(graph_text(text, -1), text)

    def test_explicit_budget_still_truncates(self):
        out = graph_text("First sentence. Second sentence. " + "x" * 900, 120)
        self.assertLessEqual(len(out), 120)
        self.assertTrue(out.endswith("."))

    def test_short_text_passes_through_unchanged(self):
        self.assertEqual(graph_text("short", 4000), "short")

    def test_two_zero_semantics_are_deliberately_opposite(self):
        """
        锁住这个反直觉的设计：select_graph_text(0) == ""，graph_text(0) == 全文。

        合并成一个函数就必须二选一，而两种读法都有人会写错。分开且各自有
        测试钉住，比"写个注释提醒一下"可靠。
        """
        text = "c" * 100
        self.assertEqual(select_graph_text(text, 0), "")
        self.assertEqual(graph_text(text, 0), text)


class TestStripFrontMatter(unittest.TestCase):
    """
    首页作者/单位块会被实体抽取当成实体（实测 765 个节点里有 43 个是这类，
    占 5.6%）。它们不破坏检索，但会稀释图谱 —— 所以要在进图之前删掉。
    """

    def test_affiliations_removed(self):
        text = (
            "Geological CO2 Storage Assessment\n"
            "Jane Smith, Wei Zhang\n"
            "Department of Earth Sciences, AGH University of Krakow, Poland\n"
            "Corresponding author: jane@example.edu\n"
            "Abstract: Injection pressure must remain below the fracture pressure."
        )
        out = strip_front_matter(text)
        self.assertNotIn("AGH University", out)
        self.assertNotIn("jane@example.edu", out)
        self.assertNotIn("Department of Earth", out)
        self.assertIn("Geological CO2 Storage Assessment", out)
        self.assertIn("Injection pressure must remain", out)

    def test_postal_address_removed(self):
        text = (
            "10100 Burnet Rd., Bldg. 130, Austin, TX 78758, USA\n"
            "The caprock limits injection pressure."
        )
        out = strip_front_matter(text)
        self.assertNotIn("Burnet Rd", out)
        self.assertIn("caprock limits injection", out)

    def test_body_mentions_are_kept(self):
        """
        正文里出现 "University of Alberta" 往往是有意义的（研究由谁完成），
        而 window 之外的内容一律不动。
        """
        body = "x" * 2000 + "\nThis work was done at the University of Alberta."
        out = strip_front_matter(body)
        self.assertIn("University of Alberta", out)

    def test_clean_text_unchanged(self):
        text = "Injection pressure 250 bar at 2000 m depth.\nFracture pressure is 303 bar."
        self.assertEqual(strip_front_matter(text), text)

    def test_empty_input(self):
        self.assertEqual(strip_front_matter(""), "")


if __name__ == "__main__":
    unittest.main(verbosity=2)
