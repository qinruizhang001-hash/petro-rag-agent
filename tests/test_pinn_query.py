"""
PINN 调用参数的自然语言抽取（零依赖，CI 可跑）
=============================================
``utils/pinn_query.py`` 是"用户问句 → (x 米, t 天)"的唯一入口。它一旦抽错，
后果不是报错而是**算出一个看起来合理的错值**：把"注入压力 250 bar"里的
250 当成距离，工具就会去算 x=250 m（越界裁剪成 100 m），然后报告里出现一个
精确到小数点后两位的、彻底错误的压力。这类错误最难发现，所以规则要穷举测。

三类用例：
  1. 必须命中 —— 中文/英文/单位混用/语序颠倒/小时换天
  2. 必须不命中 —— 只有距离没有时间、只有时间没有距离、纯参数描述
  3. 必须拒绝 —— 数值明显越界（那不是距离，是抽错了）
"""
from __future__ import annotations

import unittest

from utils.pinn_query import extract_pinn_query


class TestMustMatch(unittest.TestCase):
    def test_canonical_chinese(self):
        """需求里那句原话。"""
        self.assertEqual(
            extract_pinn_query("预测距离井口 50 米处，注入 10 天后的压力"),
            (50.0, 10.0),
        )

    def test_short_forms(self):
        cases = {
            "距井 50m、注入 10 天后压力多少": (50.0, 10.0),
            "井口50米，注入10天": (50.0, 10.0),
            "井口距离 30 米，注入 7 天后": (30.0, 7.0),
            "距离井口 80 米处注入 20 天后的压力": (80.0, 20.0),
        }
        for text, expect in cases.items():
            with self.subTest(text=text):
                self.assertEqual(extract_pinn_query(text), expect)

    def test_order_does_not_matter(self):
        """时间在前、距离在后，也必须能抽到。"""
        self.assertEqual(
            extract_pinn_query("注入 10 天后，距离井口 50 米处的压力"),
            (50.0, 10.0),
        )

    def test_hours_are_converted_to_days(self):
        """240 小时 = 10 天。单位换算错了会得到一个差 24 倍的答案。"""
        self.assertEqual(
            extract_pinn_query("井口距离 30 米，注入 240 小时后的压力"),
            (30.0, 10.0),
        )

    def test_english_word_order_is_not_supported(self):
        """英文语序（数字在前、单位在后）不匹配 —— 这是有意的取舍。

        支持它需要额外的 ``50 meters from the well`` 模式，而英文里的
        "10 days" 这类片段在历史记录/文献摘录里出现频率很高，宽松匹配会让
        兜底逻辑误触发。系统面向中文问法；英文提问仍可走模型规划那条路。
        """
        self.assertIsNone(
            extract_pinn_query("pressure at 50 meters from the well after 10 days")
        )

    def test_decimal_values(self):
        self.assertEqual(extract_pinn_query("井口 12.5 米，注入 2.5 天后"), (12.5, 2.5))


class TestMustNotMatch(unittest.TestCase):
    def test_parameter_description_without_coordinates(self):
        """"注入压力 250 bar"里的 250 不是距离。"""
        self.assertIsNone(extract_pinn_query("注入压力 250 bar，温度 80℃，深度 2500 m"))

    def test_distance_only(self):
        self.assertIsNone(extract_pinn_query("距离井口 50 米"))

    def test_time_only(self):
        self.assertIsNone(extract_pinn_query("注入 10 天"))

    def test_no_numbers(self):
        self.assertIsNone(extract_pinn_query("预测井口压力"))

    def test_empty_and_garbage(self):
        for text in ("", "   ", None, 123, [], {}):
            with self.subTest(text=repr(text)):
                self.assertIsNone(extract_pinn_query(text))


class TestMustReject(unittest.TestCase):
    def test_distance_outside_domain(self):
        """x > L 说明抽错了（或问题超出模型适用范围），不能悄悄裁剪后当真。"""
        self.assertIsNone(extract_pinn_query("距离井口 500 米处，注入 10 天后"))

    def test_time_outside_domain(self):
        self.assertIsNone(extract_pinn_query("距离井口 50 米处，注入 400 天后"))

    def test_boundary_values_are_accepted(self):
        """边界值本身合法：x=0 / x=100、t=0 / t=30。"""
        self.assertEqual(extract_pinn_query("距离井口 0 米，注入 0 天后"), (0.0, 0.0))
        self.assertEqual(extract_pinn_query("距离井口 100 米，注入 30 天后"), (100.0, 30.0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
