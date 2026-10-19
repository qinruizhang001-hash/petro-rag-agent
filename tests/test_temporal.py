"""
时间维度记忆（Temporal Knowledge）的离线单元测试
================================================
零依赖、零 IO、零模型，可在 CI 里秒级跑完。

这里守的是三条底线：

1. **不认错时间。** 裸四位数（"埋深 2000 m"）绝不能被当成 2000 年 ——
   一旦认错，每份报告都会凭空多出"跨世纪冲突"，而冲突提示是会改变
   模型取值行为的，误报的代价比漏报大得多。
2. **不丢量纲。** "2.5 MPa" 与 "25 bar" 是同一个值，不能判成冲突；
   反过来 250 bar 与 300 bar 必须判成冲突。
3. **冲突提示必须是指令，不是描述。** 提示块里出现"必须""禁止"这类词，
   是因为 7B 模型对"建议参考"基本无视；这条用断言钉住，防止哪天被
   "润色"成一段温和的说明文字。

运行::

    python -m unittest -v tests.test_temporal
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.temporal import (
    analyze_temporal,
    build_conflict_notice,
    detect_conflicts,
    extract_claims,
    parse_timestamp,
    stamp_entity_timestamp,
    ts_sort_key,
)


class TestParseTimestamp(unittest.TestCase):
    def test_chinese_year_month(self):
        self.assertEqual(parse_timestamp("2025年6月，A-1井实测压力 200 bar"), "2025-06")

    def test_chinese_year_only(self):
        self.assertEqual(parse_timestamp("2025年的评估结论"), "2025")

    def test_iso_date(self):
        self.assertEqual(parse_timestamp("2026-03-15 复测"), "2026-03-15")

    def test_iso_datetime(self):
        self.assertEqual(parse_timestamp("2026-03-15T10:00:00Z"), "2026-03-15")

    def test_month_out_of_range_degrades_to_year(self):
        self.assertEqual(parse_timestamp("2025年13月"), "2025")

    def test_bare_four_digit_number_is_not_a_date(self):
        # 这是最关键的一条：埋深 2000 m 不是 2000 年
        self.assertIsNone(parse_timestamp("埋深 2000 m，注入压力 250 bar"))

    def test_year_without_separator_is_not_a_date(self):
        self.assertIsNone(parse_timestamp("共 2025 条记录"))

    def test_empty_and_none(self):
        self.assertIsNone(parse_timestamp(""))
        self.assertIsNone(parse_timestamp(None))


class TestTsSortKey(unittest.TestCase):
    def test_granularity_is_comparable(self):
        keys = [ts_sort_key(t) for t in ("2024", "2025-01", "2025-06-01", "2025-12")]
        self.assertEqual(keys, sorted(keys))

    def test_month_beats_year(self):
        self.assertLess(ts_sort_key("2025"), ts_sort_key("2025-01"))


class TestExtractClaims(unittest.TestCase):
    def test_extracts_entity_metric_value_time(self):
        claims = extract_claims("2025年6月，A-1井实测地层压力为200 bar。")
        self.assertEqual(len(claims), 1)
        c = claims[0]
        self.assertEqual(c["entity"], "A-1井")
        self.assertEqual(c["metric"], "pressure")
        self.assertEqual(c["value"], 200.0)
        self.assertEqual(c["unit"], "bar")
        self.assertEqual(c["timestamp"], "2025-06")

    def test_mpa_is_normalised_to_bar(self):
        claims = extract_claims("2026年3月，A-1井地层压力 30 MPa。")
        self.assertEqual(claims[0]["unit"], "bar")
        self.assertAlmostEqual(claims[0]["value"], 300.0, places=6)

    def test_timestamp_carries_within_paragraph(self):
        # 同一次测量的两个参数：后半句没有年份，但属于同一个时间点
        claims = extract_claims("2025年6月，A-1井地层压力200 bar，温度80℃。")
        self.assertEqual({c["timestamp"] for c in claims}, {"2025-06"})
        self.assertEqual({c["metric"] for c in claims}, {"pressure", "temperature"})

    def test_entity_carries_across_sentences_in_same_text(self):
        claims = extract_claims("2025年6月，A-1井地层压力200 bar。温度80℃。")
        self.assertEqual({c["entity"] for c in claims}, {"A-1井"})

    def test_temperature_is_not_attributed_to_pressure(self):
        """同一句里的"压力 200 bar，温度 80℃"不能变成两条压力记录。

        如果关键词搜索不设边界，80℃ 会撞上前面的"压力"，于是系统会报告
        "压力从 200 变成 80，变化 -60%" —— 一条完全虚构的冲突，
        而且它会被写进报告、改变模型的取值行为。
        """
        claims = extract_claims("2025年6月，A-1井地层压力200 bar，温度80℃。")
        self.assertEqual({c["metric"] for c in claims}, {"pressure", "temperature"})
        self.assertEqual(detect_conflicts(claims), [])

    def test_claim_without_timestamp_is_dropped(self):
        # 没有时间锚点的数值进不了时间轴 —— 否则会制造假冲突
        self.assertEqual(extract_claims("地层压力 200 bar，温度 80℃。"), [])

    def test_percent_needs_keyword_to_be_porosity(self):
        claims = extract_claims("2025年6月，A-1井孔隙度 18%。")
        self.assertEqual(claims[0]["metric"], "porosity")

    def test_ambiguous_percent_is_skipped(self):
        # 饱和度不是孔隙度，宁可不抽，也不要抽错
        claims = extract_claims("2025年6月，A-1井含水饱和度 45%。")
        self.assertEqual(claims, [])

    def test_unknown_entity_falls_back_to_placeholder(self):
        claims = extract_claims("2025年6月，地层压力 200 bar。")
        self.assertEqual(claims[0]["entity"], "未指定井位")

    def test_depth_metric(self):
        claims = extract_claims("2025年6月，A-1井埋深 2500 m。")
        self.assertEqual(claims[0]["metric"], "depth")
        self.assertEqual(claims[0]["unit"], "m")


class TestDetectConflicts(unittest.TestCase):
    def _claims(self, text):
        return extract_claims(text)

    def test_conflict_detected_across_years(self):
        claims = self._claims(
            "2025年6月，A-1井实测地层压力为200 bar。"
            "2026年3月，A-1井地层压力升至300 bar。"
        )
        conflicts = detect_conflicts(claims)
        self.assertEqual(len(conflicts), 1)
        c = conflicts[0]
        self.assertEqual(c["older"]["timestamp"], "2025-06")
        self.assertEqual(c["newer"]["timestamp"], "2026-03")
        self.assertEqual(c["newer"]["value"], 300.0)
        self.assertAlmostEqual(c["delta_pct"], 50.0, places=1)
        self.assertEqual(c["span"], "9 个月")

    def test_small_difference_is_not_a_conflict(self):
        # 3% 的变化在测量噪声内，报警只会让报告充满噪音
        claims = self._claims(
            "2025年6月，A-1井地层压力 200 bar。2026年3月，A-1井地层压力 206 bar。"
        )
        self.assertEqual(detect_conflicts(claims), [])

    def test_same_timestamp_different_value_is_not_temporal_conflict(self):
        claims = self._claims(
            "2025年6月，A-1井地层压力 200 bar。2025年6月，A-1井地层压力 300 bar。"
        )
        self.assertEqual(detect_conflicts(claims), [])

    def test_different_entities_do_not_conflict(self):
        claims = self._claims(
            "2025年6月，A-1井地层压力 200 bar。2026年3月，B-2井地层压力 300 bar。"
        )
        self.assertEqual(detect_conflicts(claims), [])

    def test_newest_value_is_always_the_reference(self):
        claims = self._claims(
            "2024年1月，A-1井地层压力 200 bar。"
            "2025年1月，A-1井地层压力 210 bar。"
            "2026年1月，A-1井地层压力 300 bar。"
        )
        conflicts = detect_conflicts(claims)
        self.assertEqual(len(conflicts), 1)
        # 与"最近的一个不同值"比，而不是与最早的值比
        self.assertEqual(conflicts[0]["older"]["value"], 210.0)
        self.assertEqual(conflicts[0]["newer"]["value"], 300.0)

    def test_conflicts_sorted_newest_first(self):
        claims = self._claims(
            "2020年1月，A-1井地层压力 100 bar。2021年1月，A-1井地层压力 200 bar。"
            "2025年1月，B-2井地层压力 100 bar。2026年1月，B-2井地层压力 300 bar。"
        )
        conflicts = detect_conflicts(claims)
        self.assertEqual(len(conflicts), 2)
        self.assertEqual(conflicts[0]["newer"]["timestamp"], "2026-01")

    def test_max_conflicts_is_respected(self):
        text = "".join(
            f"20{20 + i}年1月，A-{i}井地层压力 {100 + i * 100} bar。"
            f"20{21 + i}年1月，A-{i}井地层压力 {200 + i * 100} bar。"
            for i in range(8)
        )
        self.assertLessEqual(len(detect_conflicts(extract_claims(text), max_conflicts=3)), 3)


class TestConflictNotice(unittest.TestCase):
    def _notice(self):
        claims = extract_claims(
            "2025年6月，A-1井实测地层压力为200 bar。"
            "2026年3月，A-1井地层压力升至300 bar。"
        )
        return build_conflict_notice(detect_conflicts(claims))

    def test_empty_conflicts_produce_no_notice(self):
        self.assertEqual(build_conflict_notice([]), "")

    def test_notice_contains_imperative_rules(self):
        notice = self._notice()
        self.assertIn("时间维度提示", notice)
        self.assertIn("必须优先采用时间戳最新的数据", notice)
        self.assertIn("禁止", notice)

    def test_notice_contains_both_values_and_timestamps(self):
        notice = self._notice()
        self.assertIn("200 bar", notice)
        self.assertIn("300 bar", notice)
        self.assertIn("2025-06", notice)
        self.assertIn("2026-03", notice)

    def test_notice_is_a_markdown_table(self):
        notice = self._notice()
        self.assertIn("| 对象 | 参数 | 旧数据 | 最新数据 |", notice)
        self.assertIn("| A-1井 |", notice)


class TestAnalyzeTemporal(unittest.TestCase):
    def test_returns_full_structure_even_when_empty(self):
        out = analyze_temporal([])
        self.assertEqual(out["claims"], 0)
        self.assertEqual(out["conflicts"], [])
        self.assertEqual(out["notice"], "")

    def test_docs_are_joined(self):
        out = analyze_temporal([
            "2025年6月，A-1井地层压力 200 bar。",
            "2026年3月，A-1井地层压力 300 bar。",
        ])
        self.assertEqual(out["claims"], 2)
        self.assertEqual(len(out["conflicts"]), 1)
        self.assertIn("A-1井", out["entities"])

    def test_no_conflict_when_values_agree(self):
        out = analyze_temporal(["2025年6月，A-1井地层压力 200 bar。",
                                "2026年3月，A-1井地层压力 201 bar。"])
        self.assertEqual(out["conflicts"], [])
        self.assertEqual(out["notice"], "")


class TestEntityTimestamp(unittest.TestCase):
    def test_reads_explicit_field(self):
        self.assertEqual(
            stamp_entity_timestamp({"entity_name": "A", "timestamp": "2025-06-01"}),
            "2025-06-01",
        )

    def test_reads_description(self):
        self.assertEqual(
            stamp_entity_timestamp({"entity_name": "A", "description": "2025年测得压力 200 bar"}),
            "2025",
        )

    def test_returns_none_when_absent(self):
        self.assertIsNone(stamp_entity_timestamp({"entity_name": "A", "description": "无时间"}))


if __name__ == "__main__":
    unittest.main(verbosity=2)
