"""
``normalize_graph.py`` 的统计报告测试（纯标准库，不需要图谱文件）
=================================================================
被测的是"报告能不能说清楚噪声从哪来"，不是"归一化跑得对不对"——
后者由 ``tests/test_entity_aliases.py`` 与 ``tests/test_graph_analysis.py``
用真实规则和真实图算法负责。

为什么报告值得单独测
--------------------
这份报告存在的唯一理由是**把零命中的规则也列出来**。``build_merge_plan``
只返回命中项，所以"某条规则从没命中过"与"这条规则不存在"在结果里长得
一模一样；只有报告能把两者分开。一旦报告的渲染逻辑坏了（比如只遍历命中项、
或者把未登记的原因渲染成空白），这个区别就重新消失，而且**不会有任何
异常**——文件照样生成，照样能读，只是结论变回"看起来没问题"。

另一类必须锁的是"数字不能手写"。语料换了、规则改了，手写数字不会跟着变，
也没有断言会红。所以这里的做法是：给两份不同的 payload，断言输出的数字
跟着变。
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from normalize_graph import (
    CONSIDERED_PATTERNS,
    REASON_NOTES,
    _reason_note,
    measure_considered,
    render_noise_report,
)
from utils.entity_aliases import noise_rule_names


def _payload(**overrides) -> dict:
    """一份最小但完整的 payload：结构真、数字假（数字故意都好认）。"""
    base = {
        "before": {"nodes": 4000, "edges": 5000, "components": 1200,
                   "largest": 3000, "isolated": 800},
        "projected": {"nodes": 3500, "edges": 4600, "components": 1000,
                      "largest": 2800, "isolated": 700, "largest_ratio": 0.8,
                      "isolated_ratio": 0.2, "top": [2800]},
        "plan": {
            "merges": [{"target": "CO2", "sources": ["CO2", "Carbon dioxide (CO2)"]}],
            "noise": [
                {"name": "Borello et al.", "reason": "citation"},
                {"name": "Cheung et al. (2012)", "reason": "citation"},
                {"name": "2021", "reason": "year"},
            ],
            "kept": 100,
        },
        "considered": [
            {"label": "会议 / 论文集名", "pattern": "x", "hits": 0, "samples": []},
            {"label": "机构名", "pattern": "y", "hits": 3,
             "samples": ["Agency A", "Agency B"]},
        ],
    }
    base.update(overrides)
    return base


class TestReportNumbersComeFromPayload(unittest.TestCase):
    """数字必须跟着 payload 变，不能是写在模板里的常量。"""

    def test_node_counts_are_interpolated(self):
        text = render_noise_report(_payload())
        self.assertIn("4000", text)          # before.nodes
        self.assertIn("3500", text)          # projected.nodes
        self.assertIn("1000", text)          # projected.components

    def test_changing_payload_changes_report(self):
        a = render_noise_report(_payload())
        p = _payload()
        p["before"] = dict(p["before"], nodes=9999)
        p["projected"] = dict(p["projected"], components=7777)
        b = render_noise_report(p)
        self.assertNotIn("9999", a)
        self.assertIn("9999", b)
        self.assertIn("7777", b)

    def test_reason_counts_are_counted_not_declared(self):
        text = render_noise_report(_payload())
        # 两条 citation、一条 year
        self.assertIn("| `citation` | 2 |", text)
        self.assertIn("| `year` | 1 |", text)

    def test_actual_row_appears_only_after_apply(self):
        dry = render_noise_report(_payload())
        self.assertNotIn("归一化后（实测）", dry)
        wet = render_noise_report(_payload(
            actual={"nodes": 3400, "edges": 4500, "components": 950,
                    "largest": 2700, "isolated": 690},
            deleted=3, failed_delete=1, merged=1, failed_merge=0,
        ))
        self.assertIn("归一化后（实测）", wet)
        self.assertIn("3400", wet)
        self.assertIn("删除成功 3 / 失败 1", wet)


class TestZeroHitRulesAreVisible(unittest.TestCase):
    """
    这是整份报告的核心：零命中的规则必须出现，且与命中项分开列。

    如果渲染时只遍历命中项，这一段会是空的 —— 而空段落在报告里看起来
    完全正常。所以断言"至少有一条零命中规则被列出来"，而不是只断言
    "没有报错"。
    """

    def test_zero_hit_section_lists_rules_with_no_hits(self):
        text = render_noise_report(_payload())
        self.assertIn("零命中的规则", text)
        section = text.split("零命中的规则", 1)[1]
        self.assertIn("| `doi` | 0 |", section)
        self.assertIn("| `email` | 0 |", section)

    def test_hit_rules_do_not_leak_into_zero_section(self):
        text = render_noise_report(_payload())
        section = text.split("零命中的规则", 1)[1].split("## 4.", 1)[0]
        self.assertNotIn("`citation`", section)
        self.assertNotIn("`year`", section)

    def test_every_declared_rule_has_an_explanation(self):
        """
        规则名与说明必须同步。少一条说明时报告会渲染成
        "（未登记说明，请补 REASON_NOTES）"—— 那行字本身不报错，
        但读者会以为那条规则没有含义。
        """
        missing = [r for r in noise_rule_names() if _reason_note(r).startswith("（未登记")]
        self.assertEqual(missing, [], f"以下原因缺说明：{missing}")

    def test_unknown_reason_is_marked_not_silent(self):
        self.assertIn("未登记", _reason_note("no-such-reason"))

    def test_reason_notes_only_describes_real_rules(self):
        """反向：说明表里不该留已经删掉的规则（会让人以为还有这条防线）。"""
        declared = set(noise_rule_names())
        extra = sorted(set(REASON_NOTES) - declared)
        self.assertEqual(extra, [], f"REASON_NOTES 里有已不存在的规则：{extra}")


class TestConsideredPatterns(unittest.TestCase):
    """
    "考虑过但没写成规则"的模式，命中数必须当场算出来。

    这解决的是一个具体的历史问题：会议名当时实测 0 命中，于是没有加规则。
    但"当时 0 命中"一旦写进文档，就变成了一句无法证伪的断言 —— 语料换了，
    它还是 0。现在每次运行都重算。
    """

    def test_hits_are_measured_from_given_names(self):
        nodes = [
            "GHGT-15 Conference Proceedings",
            "International Conference on Greenhouse Gas",
            "Caprock",
            "Saline Aquifer",
        ]
        rows = {r["label"]: r for r in measure_considered(nodes)}
        conf = rows["会议 / 论文集名"]
        self.assertEqual(conf["hits"], 2, conf)
        self.assertEqual(len(conf["samples"]), 2)

    def test_zero_hits_is_a_legitimate_result(self):
        rows = {r["label"]: r for r in measure_considered(["Caprock", "CO2 Plume"])}
        for label, row in rows.items():
            self.assertEqual(row["hits"], 0, label)
            self.assertEqual(row["samples"], [], label)

    def test_samples_are_capped(self):
        nodes = [f"Agency {i}" for i in range(20)]
        rows = {r["label"]: r for r in measure_considered(nodes)}
        self.assertLessEqual(len(rows["机构名（无 affiliation 关键词）"]["samples"]), 5)

    def test_report_flags_patterns_that_do_hit(self):
        text = render_noise_report(_payload())
        self.assertIn("需要人工看样例", text)

    def test_considered_table_is_not_empty(self):
        # 空表看起来像"我们什么都没考虑过"，必须钉住非空
        self.assertGreaterEqual(len(CONSIDERED_PATTERNS), 3)


class TestReportIsSelfDescribing(unittest.TestCase):
    def test_warns_against_hand_editing(self):
        text = render_noise_report(_payload())
        self.assertIn("不要手改", text)
        self.assertIn("python normalize_graph.py", text)

    def test_says_projection_is_computed_not_measured(self):
        text = render_noise_report(_payload())
        self.assertIn("预演", text)
        self.assertIn("算出来的", text)

    def test_handles_empty_plan(self):
        """空图 / 全部跳过时不能崩，也不能算出 NaN 或 1/0。"""
        text = render_noise_report({
            "before": {"nodes": 0, "edges": 0, "components": 0},
            "projected": {"nodes": 0, "edges": 0, "components": 0,
                          "largest": 0, "isolated": 0},
            "plan": {"merges": [], "noise": [], "kept": 0},
            "considered": [],
        })
        self.assertIn("待删脏节点 0 个", text)
        self.assertNotIn("ZeroDivisionError", text)
        self.assertNotIn("nan", text.lower().split("归一化")[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
