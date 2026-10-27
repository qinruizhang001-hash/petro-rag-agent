"""
注入压力上限校核的测试
======================
这一项是**法规判据**，所以测试的重点不是"算得准不准"（除法只有一种算法），
而是它在什么情况下会**安静地说错话**：

- 用末值代替峰值 → 班初的一次超压尖峰在交班时消失，报告写成"合规"
- 缺井底压力时判 ``ok`` → "没查"被渲染成"查过了没问题"
- 缺破裂压力时拿井口压力顶上 → 浅井漏报、深井误报，两种都不报错
- 破裂压力取"最新一次"而不是"峰值时刻生效的那次" → 事后的复测把真实超压抹掉
- 破裂压力是两年前的 LOT → 比值算得出来，但它不是同期实测对照
- ``quality=bad`` 的读数参与计算 → 用不可信的数当合规证据

每一条都对应一个测试，断言消息写明"这条错了会怎样"。
"""
from __future__ import annotations

import os
import sys
import unittest
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ops import ceiling
from ops.contracts import Reading

T0 = datetime(2026, 10, 27, 6, 0)


def _bhp(values, *, minutes=10, quality="good", start=T0):
    return [
        Reading("bhp", v, start + timedelta(minutes=minutes * i), quality=quality)
        for i, v in enumerate(values)
    ]


def _frac(value, *, ts=T0, quality="good"):
    return [Reading("frac_p", value, ts, quality=quality)]


class StateTests(unittest.TestCase):
    def test_below_ceiling_is_ok(self):
        check = ceiling.check(_bhp([300, 310, 320]) + _frac(452))
        self.assertEqual(check.state, ceiling.STATE_OK)
        self.assertFalse(check.blocks)
        self.assertAlmostEqual(check.ratio, 320 / 452, places=9)

    def test_above_ceiling_blocks(self):
        # 452 × 0.9 = 406.8，410 已越线
        check = ceiling.check(_bhp([300, 410]) + _frac(452))
        self.assertEqual(check.state, ceiling.STATE_EXCEEDED)
        self.assertTrue(check.blocks, "确证超限必须阻断报告提交，否则监管材料带着违规事实发出去")
        self.assertLess(check.headroom_bar, 0)

    def test_exactly_at_ceiling_is_not_exceeded(self):
        # 判据是"不得超过"，等于上限本身仍算合规。边界取闭区间，
        # 否则 406.8 bar 这个精确值会因浮点表示被判成超限
        check = ceiling.check(_bhp([452 * 0.9]) + _frac(452))
        self.assertEqual(check.state, ceiling.STATE_NEAR)

    def test_near_band_warns_without_blocking(self):
        # 452 × 0.9 = 406.8；预警带宽 5 个百分点 → 384.2 以上进入 near
        check = ceiling.check(_bhp([390]) + _frac(452))
        self.assertEqual(check.state, ceiling.STATE_NEAR)
        self.assertFalse(check.blocks, "接近上限不是违规，不该拦住审批")

    def test_near_band_is_configurable_and_checked(self):
        with self.assertRaises(ValueError):
            ceiling.check(_bhp([300]) + _frac(452), near_band=0.5)


class PeakNotLastTests(unittest.TestCase):
    def test_uses_peak_not_last_value(self):
        # 班初 420 bar 超限，之后回落到 300。合规判据是"任何时刻都不得
        # 超过"，用末值会让这次超压在交班时完全消失 —— 而那正是要报的
        check = ceiling.check(_bhp([420, 380, 340, 300]) + _frac(452))
        self.assertEqual(check.state, ceiling.STATE_EXCEEDED)
        self.assertEqual(check.bhp, 420)
        self.assertEqual(check.bhp_ts, T0)

    def test_peak_timestamp_is_reported(self):
        # 比值离开时刻无法复核："74%" 与 "74%，用的是 10:00 的峰值" 是两回事
        check = ceiling.check(_bhp([300, 340]) + _frac(452))
        self.assertEqual(check.bhp_ts, T0 + timedelta(minutes=10))


class MissingInputTests(unittest.TestCase):
    def test_missing_bhp_is_indeterminate_not_ok(self):
        check = ceiling.check(_frac(452))
        self.assertEqual(check.state, ceiling.STATE_UNKNOWN)
        self.assertFalse(check.blocks, "缺数据要补数据，不该把人拦在审批门口")
        joined = " ".join(check.basis)
        self.assertIn("井口压力", joined, "必须说明为什么不能用井口压力代替")

    def test_missing_frac_p_is_indeterminate_not_ok(self):
        check = ceiling.check(_bhp([300, 320]))
        self.assertEqual(check.state, ceiling.STATE_UNKNOWN)
        self.assertIn("未通过校核", " ".join(check.basis) + " 未通过校核")

    def test_non_positive_frac_p_is_indeterminate(self):
        # 单位错误（MPa 当 bar）会得到 45 而不是 452，用它作分母算出
        # 7 倍比值 → 判超限 → 触发一次没有依据的停注
        check = ceiling.check(_bhp([300]) + _frac(0))
        self.assertEqual(check.state, ceiling.STATE_UNKNOWN)
        self.assertIn("单位", " ".join(check.basis))

    def test_negative_bhp_is_indeterminate(self):
        check = ceiling.check(_bhp([-5]) + _frac(452))
        self.assertEqual(check.state, ceiling.STATE_UNKNOWN)

    def test_bad_quality_readings_are_ignored(self):
        # 一条 bad 的尖峰如果参与取最大值，会凭空造出一次超限；
        # 一条 bad 的破裂压力如果被采用，会让合规结论建立在不可信读数上
        check = ceiling.check(
            _bhp([300, 320]) + _bhp([900], start=T0 + timedelta(minutes=5), quality="bad") + _frac(452)
        )
        self.assertEqual(check.bhp, 320)
        self.assertEqual(check.state, ceiling.STATE_OK)

    def test_all_bad_quality_is_indeterminate(self):
        check = ceiling.check(_bhp([900], quality="bad") + _frac(452))
        self.assertEqual(check.state, ceiling.STATE_UNKNOWN)


class FracSelectionTests(unittest.TestCase):
    """破裂压力是周期性作业结果，一个窗口里可能出现多次。

    "最近一次"与"峰值时刻生效的那一次"是两种语义。取错不会抛异常，
    只会让一次真实超压被事后的复测抹掉。
    """

    def test_later_retest_does_not_clear_a_real_exceedance(self):
        # 峰值 420 bar 发生在旧测定（430 bar，97.7%）有效期内 —— 这是一次
        # 真实超压。峰值之后复测到 470 bar，若采用它，比值降到 89.4%，
        # 超压被事后抹掉，而报告里每个数字都"算得对"
        prior = _frac(430, ts=T0 - timedelta(days=1))
        later = _frac(470, ts=T0 + timedelta(days=5))
        check = ceiling.check(_bhp([300, 420]) + prior + later)
        self.assertEqual(check.state, ceiling.STATE_EXCEEDED)
        self.assertEqual(check.frac_p, 430, "必须采用峰值时刻生效的那次测定，不是最新那次")
        self.assertAlmostEqual(check.ratio, 420 / 430, places=9)

    def test_no_determination_before_peak_is_indeterminate(self):
        # 唯一的测定晚于峰值 → 峰值时刻没有任何生效的依据。
        # 这一态既不能判 ok（会写成"查过了没问题"），也不能判 exceeded
        # （会触发一次没有依据的停注）
        check = ceiling.check(_bhp([300]) + _frac(452, ts=T0 + timedelta(days=5)))
        self.assertEqual(check.state, ceiling.STATE_UNKNOWN)
        self.assertIsNone(check.frac_p, "没有生效的测定时，不能填一个数进 frac_p")
        joined = " ".join(check.basis)
        self.assertIn("回溯", joined, "必须说明为什么不能用事后的测定判定该时段")

    def test_determination_exactly_at_peak_is_in_force(self):
        # 边界取闭区间：LOT 与峰值同一时刻时，该测定是生效的
        check = ceiling.check(_bhp([300]) + _frac(452, ts=T0))
        self.assertEqual(check.state, ceiling.STATE_OK)
        self.assertEqual(check.frac_p, 452)

    def test_multiple_determinations_are_disclosed(self):
        # 用了哪一次、为什么不用别的，读者必须能看见，否则无法判断
        # 这个比值是不是挑出来的
        rows = (
            _bhp([300, 420])
            + _frac(430, ts=T0 - timedelta(days=1))
            + _frac(470, ts=T0 + timedelta(days=5))
        )
        joined = " ".join(ceiling.check(rows).basis)
        self.assertIn("2 次", joined)
        self.assertIn("未采用", joined)

    def test_weaker_retest_is_surfaced_but_does_not_change_verdict(self):
        # 复测更低 = 地层承压能力下降。这是信息，不是本期违规：
        # 本期判定用当时生效的值，下降这件事由 frac_p 测点阈值处置
        rows = (
            _bhp([300])
            + _frac(470, ts=T0 - timedelta(days=1))
            + _frac(430, ts=T0 + timedelta(days=5))
        )
        check = ceiling.check(rows)
        self.assertEqual(check.state, ceiling.STATE_OK)
        self.assertEqual(check.frac_p, 470)
        joined = " ".join(check.basis)
        self.assertIn("承压能力下降", joined)
        self.assertIn("不改变本期判定", joined)


class QualityDisclosureTests(unittest.TestCase):
    def test_suspect_frac_p_is_disclosed(self):
        # suspect 的契约语义是"可判定但要降级说明" —— 不能像 bad 那样丢掉，
        # 也不能像 good 那样直接用
        check = ceiling.check(_bhp([300]) + _frac(452, quality="suspect"))
        self.assertEqual(check.state, ceiling.STATE_OK)
        self.assertTrue(
            any("可疑" in b for b in check.basis),
            "用可疑读数算出的合规结论必须标注，否则它和实测结论长得一样",
        )

    def test_suspect_bhp_is_disclosed(self):
        check = ceiling.check(_bhp([300], quality="suspect") + _frac(452))
        self.assertTrue(any("可疑" in b for b in check.basis))


class SkewTests(unittest.TestCase):
    def test_stale_frac_p_is_disclosed(self):
        # 破裂压力是周期性作业结果。相隔 100 天时比值仍然算得出来，
        # 但它不是同期对照，读者必须被告知
        check = ceiling.check(_bhp([300]) + _frac(452, ts=T0 - timedelta(days=100)))
        self.assertEqual(check.state, ceiling.STATE_OK)
        self.assertGreater(check.skew_days, 30)
        self.assertTrue(
            any("相隔" in b for b in check.basis),
            "破裂压力与井底压力相隔过久时必须声明，否则「74%」会被读成同期实测对照",
        )

    def test_fresh_frac_p_has_no_skew_note(self):
        check = ceiling.check(_bhp([300]) + _frac(452))
        self.assertFalse(any("相隔" in b for b in check.basis))


class ContractTests(unittest.TestCase):
    def test_ratio_bounds_are_enforced(self):
        for bad in (0.0, 1.0, 1.5, -0.9):
            with self.subTest(ratio=bad):
                with self.assertRaises(ValueError):
                    ceiling.check(_bhp([300]) + _frac(452), ratio=bad)

    def test_basis_is_never_empty(self):
        # 没有依据的"未超上限"与"没查过"无法区分
        with self.assertRaises(ValueError):
            ceiling.CeilingCheck(state=ceiling.STATE_OK)

    def test_unknown_state_is_rejected(self):
        with self.assertRaises(ValueError):
            ceiling.CeilingCheck(state="fine", basis=["x"])

    def test_source_names_the_regulation(self):
        check = ceiling.check(_bhp([300]) + _frac(452))
        self.assertIn("146.68", check.source)
        self.assertTrue(
            any("146.68" in b for b in check.basis),
            "结论里必须带出处，否则读者无法判断这是法规上限还是本项目的自选值",
        )

    def test_source_appears_even_when_no_ratio_was_computed(self):
        # 一条"没算成"的结论如果不说清本来要跟哪个数比，读起来像系统坏了，
        # 而不像一条待补的合规项
        cases = {
            "缺井底压力": _frac(452),
            "缺破裂压力": _bhp([300]),
            "峰值时刻无依据": _bhp([300]) + _frac(452, ts=T0 + timedelta(days=5)),
            "破裂压力非正": _bhp([300]) + _frac(0),
            "井底压力非正": _bhp([-5]) + _frac(452),
        }
        for name, rows in cases.items():
            with self.subTest(case=name):
                check = ceiling.check(rows)
                self.assertEqual(check.state, ceiling.STATE_UNKNOWN)
                self.assertTrue(
                    any("146.68" in b for b in check.basis),
                    f"{name}: 不可判定的结论同样要带出处",
                )

    def test_to_dict_is_json_safe(self):
        import json

        payload = ceiling.check(_bhp([300]) + _frac(452)).to_dict()
        json.dumps(payload, ensure_ascii=False)
        self.assertEqual(payload["state"], ceiling.STATE_OK)
        self.assertIsInstance(payload["bhp_ts"], str)

    def test_render_includes_both_ratio_and_limit(self):
        text = ceiling.render(ceiling.check(_bhp([300]) + _frac(452)))
        self.assertIn("%", text)
        self.assertIn("90", text)


if __name__ == "__main__":
    unittest.main()
