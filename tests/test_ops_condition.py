"""
工况问答 Agent 的测试
=====================
这个 Agent 的失效方式很特别：它**永远能给出一个答案**，而错答案读起来和
对答案一模一样。所以测试的重点是"它在什么情况下必须说不知道"：

- 问句里没有测点名 → 必须拒绝回答，而不是挑一个最像的
- 问句里没有时间范围 → 必须标注用了默认窗口
- 窗口内没有数据 → 必须说没有数据，而不是说"正常"
- 数据有缺口 → 必须报覆盖率，而不是给一个看不出插值的均值
- 模型输出里有凭空数字 → 必须整篇回退，而不是逐句删数字
"""
from __future__ import annotations

import math
import os
import sys
import unittest
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ops import condition
from ops.contracts import QUALITY_BAD, Reading
from ops.thresholds import build_registry

T0 = datetime(2026, 10, 26, 8, 0)


def _reg():
    return build_registry(
        {
            "wellhead_p": {"warn": 80, "alert": 100, "critical": 120, "source": "设计文件"},
            "injection_rate": {"warn": 200, "alert": 150, "critical": 100, "source": "设计文件"},
        },
        origin="测试",
    )


def _series(sensor, values, minutes=10, start=T0):
    return [
        Reading(sensor, v, start + timedelta(minutes=minutes * i))
        for i, v in enumerate(values)
    ]


class WindowParsingTests(unittest.TestCase):
    def test_arabic_numerals(self):
        for text, hours in [
            ("过去 6 小时井口压力怎么样", 6.0),
            ("最近 30 分钟", 0.5),
            ("近 2 天", 48.0),
            ("过去3周", 504.0),
        ]:
            with self.subTest(text=text):
                w = condition.parse_window(text)
                self.assertEqual(w.hours, hours)
                self.assertTrue(w.explicit)

    def test_chinese_numerals(self):
        for text, hours in [
            ("最近三小时", 3.0),
            ("过去两天", 48.0),
            ("最近半小时", 0.5),
        ]:
            with self.subTest(text=text):
                w = condition.parse_window(text)
                self.assertEqual(w.hours, hours)

    def test_relative_words(self):
        self.assertEqual(condition.parse_window("今天的注入情况").hours, 24.0)
        self.assertEqual(condition.parse_window("本班工况").hours, 8.0)

    def test_missing_window_is_flagged_not_silent(self):
        # 静默用默认窗口会让回答看起来回答了用户的问题，实际问的是另一段时间
        w = condition.parse_window("井口压力现在多少")
        self.assertFalse(w.explicit)
        self.assertEqual(w.hours, condition.DEFAULT_WINDOW_HOURS)
        self.assertIn("没有可识别的时间范围", w.note)

    def test_absurd_window_is_clamped_with_note(self):
        w = condition.parse_window("过去 999 天")
        self.assertLessEqual(w.hours, condition.MAX_WINDOW_HOURS)
        self.assertIn("截断", w.note)


class SensorResolutionTests(unittest.TestCase):
    def test_aliases(self):
        for text, expect in [
            ("井口压力", "wellhead_p"),
            ("井口压", "wellhead_p"),
            ("注入速率", "injection_rate"),
            ("注入量", "injection_rate"),
        ]:
            with self.subTest(text=text):
                r = condition.resolve_sensors(text)
                self.assertIn(expect, r.sensors)

    def test_mva_expands_to_a_group(self):
        # MVA 是一套监测体系，不是一个指标。当成单一测点会丢掉层位信息
        r = condition.resolve_sensors("MVA 情况怎么样")
        self.assertGreaterEqual(len(r.sensors), 4)
        self.assertTrue(any(s.startswith("gw_") for s in r.sensors))

    def test_unknown_sensor_is_not_guessed(self):
        # 猜一个"最像的"测点，会给出一个数值正确、对象错误的回答 ——
        # 那种错误比"不知道"危险得多
        r = condition.resolve_sensors("最近设备状态怎么样")
        self.assertTrue(r.unresolved)
        self.assertEqual(r.sensors, [])
        self.assertIn("不猜", r.note)

    def test_longer_alias_wins(self):
        # "井口压力" 与 "井口压" 都在表里，长词在前才不会被截断匹配漏掉
        r = condition.resolve_sensors("井口压力")
        self.assertEqual(r.sensors, ["wellhead_p"])


class StatsTests(unittest.TestCase):
    def test_nearest_rank_percentile_returns_observed_values(self):
        # nearest-rank 永远取实测点，不会给出一个从未出现过的值 ——
        # 对"这口井实际到过多高"这类问题这是必需的
        vals = [float(i) for i in range(1, 21)]     # 1..20
        self.assertEqual(condition._percentile(vals, 95), 19.0)
        self.assertEqual(condition._percentile(vals, 5), 1.0)
        self.assertEqual(condition._percentile(vals, 100), 20.0)

    def test_coverage_reflects_window(self):
        rs = _series("wellhead_p", [70, 71, 72], minutes=10)   # 只覆盖 20 分钟
        a = condition.answer("过去 6 小时井口压力", rs, _reg())
        self.assertLess(a.stats[0].coverage, 0.2)
        self.assertIn("覆盖", a.text)

    def test_gaps_are_reported(self):
        rs = _series("wellhead_p", [70, 71, 72, 73], minutes=5)
        rs.append(Reading("wellhead_p", 74, rs[-1].ts + timedelta(minutes=120)))
        a = condition.answer("最近 4 小时井口压力", rs, _reg())
        self.assertTrue(a.stats[0].gaps)
        self.assertIn("缺口", a.text)
        self.assertIn("未做插值", a.text)

    def test_bad_quality_readings_are_excluded(self):
        rs = _series("wellhead_p", [70, 71, 72])
        rs.append(Reading("wellhead_p", 999, rs[-1].ts + timedelta(minutes=5), quality=QUALITY_BAD))
        a = condition.answer("最近 1 小时井口压力", rs, _reg())
        self.assertEqual(a.stats[0].n, 3)
        self.assertNotIn(999, [s.max_v for s in a.stats])

    def test_focus_percentile_follows_direction(self):
        rs = _series("wellhead_p", [70, 90, 110])
        a = condition.answer("最近 1 小时井口压力", rs, _reg())
        # 越高越坏 → 关注高分位
        self.assertEqual(a.stats[0].focus_value(), a.stats[0].p95)

        rs2 = _series("injection_rate", [220, 180, 140])
        a2 = condition.answer("最近 1 小时注入速率", rs2, _reg())
        # 越低越坏 → 关注低分位
        self.assertEqual(a2.stats[0].focus_value(), a2.stats[0].p5)

    def test_derived_stats_do_not_depend_on_summation_order(self):
        """均值/斜率必须与解释器版本无关。

        CPython 3.12 起 `sum()` 对浮点改用补偿求和：同一组读数在 3.11 上得到
        ``12.565999999999999``，在 3.13 上得到 ``12.566``。相对差 1e-16，物理上
        没有意义，但它会让前端 fixture 的逐字节比对在换解释器时变红 —— 而那种红
        会被人当成噪声关掉，连带关掉真正的漂移检测。

        所以派生量归一到 12 位有效数字。这条测试直接锁住"两种求和结果归一到
        同一个值"，而不是锁某个具体数字。

        用的是 pipeline_dp 的真实序列：它恰好是两种求和会分家的那一组
        （3.11 → ``12.566``，3.13 → ``12.565999999999999``），也就是 CI 上
        真的红过的那一次。用随便一组数据写这条测试会得到两个相同的结果，
        于是它永远是绿的，什么也没锁住。

        两种求和**都自己算**，不调用内建 `sum()`：内建的算法本身随版本变
        （3.12 起改补偿求和），用它当"其中一侧"会让这条测试在 3.10/3.11 上
        直接失败 —— 那正是第一版的样子。``math.fsum`` 是正确舍入且与版本无关的，
        拿它当"精确侧"，循环累加当"朴素侧"，两侧在哪个版本上都不同。
        """
        values = [
            12.40, 12.41, 12.42, 12.44, 12.45, 12.46, 12.48, 12.50,
            12.51, 12.52, 12.54, 12.55, 12.56, 12.58, 12.60, 12.61,
            12.62, 12.64, 12.65, 12.66, 12.68, 12.70, 12.71, 12.72, 12.74,
        ]
        naive = 0.0
        for v in values:                      # 朴素累加：任何版本都一样
            naive += v
        exact = math.fsum(values)             # 正确舍入，与版本无关

        self.assertNotEqual(
            naive / len(values),
            exact / len(values),
            "这组数据必须能让两种求和分家，否则这条测试是空的",
        )
        self.assertEqual(
            condition._stable(naive / len(values)),
            condition._stable(exact / len(values)),
            "两种求和顺序必须归一到同一个值，否则 fixture 会在换 Python 版本时红",
        )

    def test_derived_stats_keep_twelve_significant_digits(self):
        rs = _series("wellhead_p", [70.1, 70.2, 70.4])
        s = condition.answer("最近 1 小时井口压力", rs, _reg()).stats[0]
        for name, value in (("mean", s.mean), ("slope", s.slope_per_hour),
                            ("coverage", s.coverage)):
            digits = len(f"{value:.12g}".replace("-", "").replace(".", "").lstrip("0"))
            self.assertLessEqual(
                digits, 12, f"{name} 保留了超过 12 位有效数字：{value!r}"
            )


class AnswerBehaviourTests(unittest.TestCase):
    def test_now_defaults_to_latest_data_point(self):
        # 离线复盘时系统时钟与数据时刻相差几天；按系统时钟取窗口会落在数据之外，
        # 回答于是变成"无数据"，而真正的原因是窗口取错了
        rs = _series("wellhead_p", [70, 71, 72])
        a = condition.answer("过去 3 小时井口压力", rs, _reg())
        self.assertEqual(a.now, rs[-1].ts)
        self.assertIn("数据最新时刻", a.text)

    def test_explicit_now_is_respected(self):
        rs = _series("wellhead_p", [70, 71, 72])
        a = condition.answer("过去 3 小时井口压力", rs, _reg(), now=rs[-1].ts)
        self.assertEqual(a.now, rs[-1].ts)

    def test_empty_window_says_no_data_not_normal(self):
        rs = _series("wellhead_p", [70], start=T0)
        a = condition.answer(
            "过去 6 小时井口压力", rs, _reg(), now=T0 + timedelta(days=2)
        )
        self.assertIn("没有可信读数", a.text)
        self.assertIn("不等于工况正常", a.text)

    def test_no_readings_at_all(self):
        a = condition.answer("过去 1 小时井口压力", [], _reg())
        self.assertIn("没有收到任何时序数据", a.text)

    def test_unresolved_sensor_refuses_to_answer(self):
        rs = _series("wellhead_p", [70, 71])
        a = condition.answer("最近设备怎么样", rs, _reg())
        self.assertEqual(a.stats, [])
        self.assertIn("无法确定", a.text)

    def test_unresolved_notes_are_surfaced(self):
        rs = _series("wellhead_p", [70, 71])
        a = condition.answer("井口压力多少", rs, _reg())
        self.assertTrue(a.unresolved)
        self.assertTrue(any("默认" in n for n in a.unresolved))

    def test_anomaly_intent_adds_alarm_summary(self):
        rs = _series("wellhead_p", [105, 106, 107])
        a = condition.answer("过去 1 小时井口压力有异常吗", rs, _reg())
        self.assertTrue(a.alarm_summary)
        self.assertTrue(any("告警" in f for f in a.facts))

    def test_text_contains_only_traceable_numbers(self):
        # 模板回答的每个数字都来自 stats，因此守卫应为空
        rs = _series("wellhead_p", [70, 90, 110])
        a = condition.answer("最近 1 小时井口压力", rs, _reg())
        self.assertEqual(condition.find_unsourced_numbers(a.text, a.facts), [])


class NumberGuardTests(unittest.TestCase):
    def test_detects_number_absent_from_facts(self):
        facts = ["井口压力：均值 93 bar"]
        self.assertEqual(
            condition.find_unsourced_numbers("均值 93 bar，上涨 15%", facts), ["15"]
        )

    def test_accepts_number_present_in_facts(self):
        facts = ["井口压力：均值 93 bar，覆盖 100%"]
        self.assertEqual(
            condition.find_unsourced_numbers("均值 93 bar，覆盖 100%", facts), []
        )

    def test_llm_output_with_phantom_number_falls_back_to_template(self):
        # 逐句删数字会留下语法破碎的句子；而"通顺但数字没出处"比
        # "朴素但每个数字都能回代"危险得多，所以整篇回退
        rs = _series("wellhead_p", [70, 90, 110])
        a = condition.answer(
            "最近 1 小时井口压力", rs, _reg(),
            llm=lambda _p: "井口压力均值 90 bar，较昨日上升 47%。",
        )
        self.assertEqual(a.unsourced_numbers, ["47"])
        self.assertIn("回退模板", " ".join(a.unresolved))
        self.assertNotIn("47", a.text)

    def test_llm_output_without_phantom_numbers_is_kept(self):
        rs = _series("wellhead_p", [70, 90, 110])
        a = condition.answer(
            "最近 1 小时井口压力", rs, _reg(),
            llm=lambda _p: "井口压力均值 90 bar。",
        )
        self.assertEqual(a.text, "井口压力均值 90 bar。")
        self.assertEqual(a.unsourced_numbers, [])

    def test_llm_failure_falls_back_to_template(self):
        # 模型失败不能让回答消失：模板是完整答案，不是降级占位
        def boom(_p):
            raise RuntimeError("网关超时")

        rs = _series("wellhead_p", [70, 90, 110])
        a = condition.answer("最近 1 小时井口压力", rs, _reg(), llm=boom)
        self.assertIn("均值", a.text)
        self.assertTrue(any("网关超时" in n for n in a.unresolved))


if __name__ == "__main__":
    unittest.main(verbosity=2)
