# -*- coding: utf-8 -*-
"""
``utils/physics_deep.py`` 的回归测试：阻断策略与报告事实行。

这个模块最容易出的错不是算错，而是**判定范围搞错**：

1. 用假设应力剖面去阻断一份根本没提断层的报告 —— 把没问题的项目拦下来；
2. 明明提了断层却不判 —— 漏掉整类风险；
3. 把"张性开启"写成"剪切滑动" —— 结论对、机制错，读者会按错的机理
   去做防治（调摩擦系数 vs 降压，是两件事）；
4. 事实行只写数值不写假设 —— 通用应力剖面的结论被当成场地实测。

测试按这四条组织，另外钉住"没算"与"算出来不合格"必须可区分。
"""
import os
import sys
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils import physics_deep as pd  # noqa: E402
from utils.physics_rules import evaluate_physics  # noqa: E402


class NoDataTests(unittest.TestCase):
    def test_missing_depth_is_no_data_not_failure(self):
        """缺埋深 → no_data，不能当成"判定不通过"。"""
        r = pd.assess(depth_m=None, injection_pressure_bar=250.0)
        self.assertEqual(r["status"], "no_data")
        self.assertFalse(r["blocking"])
        self.assertIn("depth_m", r["missing"])

    def test_missing_pressure_is_no_data(self):
        r = pd.assess(depth_m=2500.0, injection_pressure_bar=None)
        self.assertEqual(r["status"], "no_data")
        self.assertIn("injection_pressure_bar", r["missing"])

    def test_no_facts_when_nothing_was_computed(self):
        """没算过就不能往报告里塞数值。"""
        self.assertEqual(pd.render_facts(pd.assess(None, None)), [])

    def test_no_data_result_has_no_caprock_numbers(self):
        """no_data 时不能有半截结果 —— 否则下游会读到 KeyError 或 0。"""
        r = pd.assess(None, 250.0)
        self.assertNotIn("caprock", r)
        self.assertNotIn("geomechanics", r)


class FaultDetectionTests(unittest.TestCase):
    def test_fault_word_triggers_judgement(self):
        ctx = pd.extract_fault_context("场地内存在一条断层，倾角 60°")
        self.assertTrue(ctx["has_fault"])
        self.assertEqual(ctx["dip_deg"], 60.0)

    def test_dip_without_fault_word_is_ignored(self):
        """只说"地层倾角 15°"不是断层 —— 拿它去判断层会凭空造出风险。"""
        ctx = pd.extract_fault_context("地层倾角 15°，储层埋深 2500 m")
        self.assertFalse(ctx["has_fault"])
        r = pd.assess(2500.0, 250.0, text="地层倾角 15°，储层埋深 2500 m")
        self.assertIsNone(r["geomechanics"])

    def test_implausible_dip_is_rejected(self):
        """倾角 120° 一定是抽取错误，不能拿去算。"""
        self.assertIsNone(pd.extract_fault_context("断层倾角 120°")["dip_deg"])

    def test_missing_dip_falls_back_to_default_and_says_so(self):
        r = pd.assess(2500.0, 250.0, text="该区块存在断层")
        self.assertEqual(r["geomechanics"]["dip_source"], "default-60")
        self.assertAlmostEqual(r["geomechanics"]["plane"]["dip_deg"], 60.0)

    def test_given_dip_is_marked_as_input(self):
        r = pd.assess(2500.0, 250.0, text="断层倾角 45°")
        self.assertEqual(r["geomechanics"]["dip_source"], "input")
        self.assertAlmostEqual(r["geomechanics"]["plane"]["dip_deg"], 45.0)


class BlockingPolicyTests(unittest.TestCase):
    """阻断策略：只有"输入确实涉及"时才允许拦。"""

    def test_no_fault_no_blocking_even_with_high_pressure(self):
        """
        600 bar / 2500 m 按默认应力剖面已经超过破裂压力，但输入没提断层，
        这里就不该由深层判定来拦 —— 那是 physics_rules 的职责，
        而且拦截理由不能是"按假设剖面你会滑"。
        """
        r = pd.assess(2500.0, 600.0, text="注入压力 600 bar，埋深 2500 m")
        self.assertIsNone(r["geomechanics"])
        self.assertFalse(r["blocking"])
        self.assertTrue(any("未提及断层" in p for p in r["policy"]))

    def test_fault_present_allows_blocking(self):
        r = pd.assess(2500.0, 600.0, text="断层倾角 60°，注入压力 600 bar")
        self.assertTrue(r["blocking"])
        self.assertTrue(r["geomechanics"]["blocking"])

    def test_stable_fault_does_not_block(self):
        r = pd.assess(2500.0, 250.0, text="断层倾角 60°")
        self.assertFalse(r["blocking"])
        self.assertEqual(r["geomechanics"]["verdict"], "stable")

    def test_policy_is_always_explained(self):
        """报告里要能看到"为什么这次没拦"。"""
        for text in ("断层倾角 60°", "注入压力 250 bar"):
            r = pd.assess(2500.0, 250.0, text=text)
            self.assertTrue(r["policy"], f"{text!r} 没有给出 policy 说明")

    def test_no_column_no_caprock_verdict(self):
        """没给柱高/速率时只给突破压力，不做封闭性判定。"""
        r = pd.assess(2500.0, 250.0, text="断层倾角 60°")
        self.assertEqual(r["caprock"]["verdict"], "no_data")
        self.assertGreater(r["caprock"]["entry_pressure_bar"], 0)
        self.assertTrue(any("柱高" in p for p in r["policy"]))


class MechanismTests(unittest.TestCase):
    """剪切滑动与张性开启是两种机制，措辞不能混。"""

    def test_shear_reactivation_message_mentions_slip_ratio(self):
        # 2500 m / 60°：临界孔隙压力约 364 bar，正应力约 467 bar。
        # 400 bar 落在两者之间 → 剪切滑动，且尚未张性开启。
        r = pd.assess(2500.0, 400.0, text="断层倾角 60°")
        geo = r["geomechanics"]
        self.assertEqual(geo["verdict"], "reactivated")
        self.assertEqual(geo["mechanism"], "shear")
        reason = r["blocking_reasons"][0]
        self.assertIn("滑移比", reason)
        self.assertNotIn("张性开启", reason)

    def test_tensile_opening_is_reported_as_tensile(self):
        # 900 bar = 90 MPa > 断层面正应力 → 面被顶开
        r = pd.assess(2500.0, 900.0, text="断层倾角 30°")
        geo = r["geomechanics"]
        self.assertEqual(geo["verdict"], "tensile_opening")
        self.assertEqual(geo["mechanism"], "tensile")
        reason = r["blocking_reasons"][0]
        self.assertIn("张性开启", reason)
        self.assertNotIn("滑移比", reason)

    def test_two_criteria_agree_across_the_range(self):
        """滑移比 ≥1 与 附加压力 ≥ 余量 必须始终同号（同一不等式的两种写法）。"""
        for pressure in (100, 200, 300, 350, 400, 450, 500):
            r = pd.assess(2500.0, float(pressure), text="断层倾角 60°")
            self.assertTrue(r["geomechanics"]["consistent"], f"{pressure} bar 不自洽")


class CaprockContextTests(unittest.TestCase):
    def test_column_height_is_extracted(self):
        ctx = pd.extract_column_context("CO2 柱高 1500 m，储层埋深 2500 m")
        self.assertEqual(ctx["column_m"], 1500.0)

    def test_injection_rate_is_extracted(self):
        ctx = pd.extract_column_context("注入速率 1000 m3/d")
        self.assertEqual(ctx["rate_m3_per_day"], 1000.0)

    def test_wan_unit_is_scaled(self):
        """"1 万方"是 1e4 m³，不是 1 m³。"""
        ctx = pd.extract_column_context("注入速率 1 万方/天")
        self.assertEqual(ctx["rate_m3_per_day"], 10000.0)

    def test_column_height_can_block(self):
        """
        1800 m 柱高超过可封高度（250 bar / 60℃ 下 Δρ≈256 kg/m³ → 约 1723 m）
        → 毛细管突破，允许阻断。
        """
        r = pd.assess(2500.0, 250.0, text="CO2 柱高 1800 m")
        self.assertEqual(r["caprock"]["verdict"], "capillary_breakthrough")
        self.assertTrue(r["blocking"])
        self.assertTrue(any("盖层失效" in x for x in r["blocking_reasons"]))

    def test_rate_gives_breakthrough_time(self):
        r = pd.assess(2500.0, 250.0, text="注入速率 1000 m3/d")
        self.assertIsNotNone(r["caprock"]["breakthrough_days"])


class RenderFactsTests(unittest.TestCase):
    def test_fault_fact_carries_its_assumption_on_the_same_line(self):
        """
        前提必须和数字在同一行。分开写时 7B 只抄数值，
        报告于是把一个假设剖面的结论写成场地实测结论。
        """
        facts = pd.render_facts(pd.assess(2500.0, 250.0, text="断层倾角 60°"))
        fault = [f for f in facts if f.startswith("断层稳定性")]
        self.assertEqual(len(fault), 1)
        self.assertIn("假设", fault[0])
        self.assertIn("滑移比", fault[0])

    def test_caprock_fact_says_default_pore_throat(self):
        facts = pd.render_facts(pd.assess(2500.0, 250.0))
        cap = [f for f in facts if f.startswith("盖层封闭性")]
        self.assertEqual(len(cap), 1)
        self.assertIn("默认值", cap[0])
        self.assertIn("Peng-Robinson", cap[0])

    def test_relperm_fact_is_always_present(self):
        """两相流特征只依赖岩性参数，任何有埋深+压力的输入都该有。"""
        facts = pd.render_facts(pd.assess(2500.0, 250.0))
        self.assertTrue(any(f.startswith("两相流") for f in facts))

    def test_no_fault_fact_when_input_has_no_fault(self):
        facts = pd.render_facts(pd.assess(2500.0, 250.0, text="注入压力 250 bar"))
        self.assertFalse(any(f.startswith("断层稳定性") for f in facts))

    def test_infinite_slip_ratio_does_not_crash_rendering(self):
        """
        有效正应力被压到 0 时滑移比是 inf，geomechanics 会把它转成字符串
        "inf"。直接套 :.0f 会抛 ValueError —— 即"算对了但写不出来"。
        """
        facts = pd.render_facts(pd.assess(2500.0, 900.0, text="断层倾角 30°"))
        self.assertTrue(any("inf" in f for f in facts))

    def test_facts_are_deterministic(self):
        a = pd.render_facts(pd.assess(2500.0, 250.0, text="断层倾角 60°"))
        b = pd.render_facts(pd.assess(2500.0, 250.0, text="断层倾角 60°"))
        self.assertEqual(a, b)


class EvaluateIntegrationTests(unittest.TestCase):
    """``evaluate_physics(deep=True)`` 的接线：违规码、默认关闭、修正要求。"""

    def _run(self, user_input, deep=True, p_inj=600.0):
        return evaluate_physics(
            params_text=f"注入压力 {p_inj:.0f} bar，储层埋深 2500 m，温度 60℃",
            report_text="注入压力 600 bar，处于安全范围内。",
            user_input=user_input,
            deep=deep,
        )

    def test_deep_off_keeps_previous_behaviour(self):
        """默认关闭时 deep 必须是 None，且不产生 DEEP_PHYSICS 违规。"""
        r = self._run("断层倾角 60°", deep=False)
        self.assertIsNone(r["deep"])
        self.assertFalse(any(v["code"] == "DEEP_PHYSICS" for v in r["violations"]))

    def test_deep_on_adds_blocking_violation(self):
        r = self._run("断层倾角 60°", deep=True)
        codes = [v["code"] for v in r["violations"] if v["severity"] == "blocking"]
        self.assertIn("DEEP_PHYSICS", codes)
        self.assertTrue(r["blocking"])
        self.assertEqual(r["status"], "violation")

    def test_deep_on_without_fault_context_does_not_block(self):
        """提到断层才算；普通输入即使深层判定跑了也不该多出一条违规。"""
        r = evaluate_physics(
            params_text="注入压力 250 bar，储层埋深 2500 m，温度 60℃",
            report_text="注入压力 250 bar 落在安全窗口内。",
            user_input="2500 m 埋深注入 250 bar 是否安全？",
            deep=True,
        )
        self.assertFalse(any(v["code"] == "DEEP_PHYSICS" for v in r["violations"]))
        self.assertEqual(r["deep"]["status"], "ok")

    def test_feedback_tells_the_model_what_to_fix(self):
        """阻断必须给出可执行动作，否则下一版只会换个说法继续错。"""
        r = self._run("断层倾角 60°", deep=True)
        self.assertIn("修正要求", r["feedback"])
        self.assertIn("假设", r["feedback"])

    def test_deep_result_rides_along_in_the_result(self):
        """结果里带完整深层结论，前端与审计不必重算。"""
        r = self._run("断层倾角 60°", deep=True)
        self.assertEqual(r["deep"]["status"], "ok")
        self.assertIn("caprock", r["deep"])
        self.assertEqual(r["deep"]["version"], pd.DEEP_VERSION)


if __name__ == "__main__":
    unittest.main()
