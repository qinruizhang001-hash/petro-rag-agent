"""
语料范围（哪些 PDF 属于语料）的测试。

为什么这个要单独测：覆盖率是这套系统对外唯一一个"量化质量"的数字，
而它是个分数 —— 分母和分子都必须来自同一份语料定义。曾经分母用
``glob("*.pdf")``、分子来自入库状态文件，于是被故意排除的 ``ccus4.pdf``
（电信 NOMA，与 CCUS 无关）一直算在分母里，覆盖率永远显示 96.9%。
恒定差 3% 的指标没人会追，所以真掉到 6% 时它也照样安静。
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.corpus import (
    EXCLUDED_PAPERS,
    count_included,
    exclusion_reason,
    is_excluded,
)

DOCS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ccus_docs")


class CorpusScopeTests(unittest.TestCase):
    def test_known_telecom_paper_is_excluded(self):
        self.assertTrue(is_excluded("ccus4.pdf"))
        self.assertIn("NOMA", exclusion_reason("ccus4.pdf"))

    def test_ccus_papers_are_not_excluded(self):
        for name in ("ccus1.pdf", "ccus5.pdf", "ccus_auto_01.pdf", "ccus_auto_18.pdf"):
            self.assertFalse(is_excluded(name), f"{name} 不应被排除")

    def test_every_exclusion_has_a_reason(self):
        """
        排除必须带理由。没有理由的排除等于"这篇为什么不在图里"查不出来，
        而语料范围恰恰是最容易被悄悄改动的地方。
        """
        for name, reason in EXCLUDED_PAPERS.items():
            self.assertTrue(reason.strip(), f"{name} 缺少排除理由")

    def test_excluded_files_actually_exist(self):
        """
        排除清单里的文件名必须真实存在。

        否则一个拼写错误就会让分母悄悄变小 —— 覆盖率反而"变好"，
        而且没有任何报错。这是最隐蔽的一类指标腐化。
        """
        for name in EXCLUDED_PAPERS:
            self.assertTrue(
                os.path.exists(os.path.join(DOCS_DIR, name)),
                f"排除清单里的 {name} 在 ccus_docs/ 下不存在（拼写错误？）",
            )

    def test_count_included_accepts_paths_and_names(self):
        pdfs = ["/tmp/a/ccus1.pdf", "ccus4.pdf", "/tmp/a/ccus_auto_02.pdf"]
        self.assertEqual(count_included(pdfs), 2)

    def test_count_included_on_empty_list(self):
        self.assertEqual(count_included([]), 0)


if __name__ == "__main__":
    unittest.main()
