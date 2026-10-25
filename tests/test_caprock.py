# -*- coding: utf-8 -*-
"""
``utils/caprock.py`` 的回归测试。

盖层判定有两条完全不同的失效路径，混淆它们会让一份"注入压力远低于破裂压力"
的报告被判成安全，而实际风险是毛细管渗漏：

    毛细管突破  Δρ·g·H ≥ 2γcosθ/r     与注入压力无关
    力学失效    P_inj ≥ σhmin + T_0     与柱高无关

测试因此按这两条分开组织，另外钉住三件事：
1. 密度差必须来自 Peng-Robinson（旧式会给 1029，柱高上限高估 2.8 倍）；
2. 界面张力三档敏感性必须单调；
3. 一维模型忽略重力分异 —— 这个偏差**朝乐观方向**，必须在 limitations 里写明。
"""
import os
import sys
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils import caprock, geomechanics as gm  # noqa: E402


class EntryPressureTests(unittest.TestCase):
    def test_matches_hand_calculation(self):
        """25 mN/m、30°、10 nm → 2·0.025·cos30° / 1e-8 = 43.3 bar。"""
        pc = caprock.entry_pressure_bar(25.0, 30.0, 10.0)
        self.assertAlmostEqual(pc, 43.3, delta=0.2)

    def test_scales_linearly_with_ift(self):
        a = caprock.entry_pressure_bar(20.0, 30.0, 10.0)
        b = caprock.entry_pressure_bar(40.0, 30.0, 10.0)
        self.assertAlmostEqual(b / a, 2.0, places=6)

    def test_halving_pore_throat_doubles_entry_pressure(self):
        """
        孔喉半径是最敏感的参数：r 减半 → Pc 翻倍。

        这也是为什么"没有 MICP 实测"这件事必须在结论里写明 ——
        10 nm 与 5 nm 的差别，比界面张力 20 与 30 的差别大得多。
        """
        a = caprock.entry_pressure_bar(25.0, 30.0, 10.0)
        b = caprock.entry_pressure_bar(25.0, 30.0, 5.0)
        self.assertAlmostEqual(b / a, 2.0, places=6)

    def test_contact_angle_reduces_entry_pressure(self):
        wet = caprock.entry_pressure_bar(25.0, 0.0, 10.0)
        sixty = caprock.entry_pressure_bar(25.0, 60.0, 10.0)
        self.assertAlmostEqual(sixty / wet, 0.5, places=3)

    def test_non_wetting_contact_angle_is_rejected(self):
        """
        θ ≥ 90° 时 cosθ ≤ 0，水湿假设不成立。必须报错而不是返回负数 ——
        负数会被下游当成"没有毛细管阻力"，直接判成可突破。
        """
        with self.assertRaises(ValueError):
            caprock.entry_pressure_bar(25.0, 95.0, 10.0)

    def test_zero_radius_is_rejected(self):
        with self.assertRaises(ValueError):
            caprock.entry_pressure_bar(25.0, 30.0, 0.0)


class ColumnHeightTests(unittest.TestCase):
    def test_inverse_in_density_difference(self):
        h1 = caprock.max_column_height_m(43.3, 300.0)
        h2 = caprock.max_column_height_m(43.3, 150.0)
        self.assertAlmostEqual(h2 / h1, 2.0, places=6)

    def test_rejects_zero_density_difference(self):
        with self.assertRaises(ValueError):
            caprock.max_column_height_m(43.3, 0.0)


class IntegrityTests(unittest.TestCase):
    def setUp(self):
        self.base = dict(depth_m=2500, injection_pressure_bar=300,
                         temperature_celsius=80, rho_brine=1030.0, environ={})

    def test_thin_column_is_sealed(self):
        r = caprock.caprock_integrity(co2_column_height_m=50, **self.base)
        self.assertEqual(r["verdict"], "sealed")
        self.assertFalse(r["blocking"])
        self.assertGreater(r["margin_bar"], 0)

    def test_tall_column_breaks_through_by_capillarity_alone(self):
        """
        注入压力 300 bar 完全正常（远低于 461 bar 的力学界限），
        但柱高足够时照样漏 —— 这条路径与注入压力无关，正是
        physics_rules 那条 1.5× 静水判据看不到的。
        """
        r = caprock.caprock_integrity(co2_column_height_m=2000, **self.base)
        self.assertEqual(r["verdict"], "capillary_breakthrough")
        self.assertTrue(r["blocking"])
        self.assertLess(r["margin_bar"], 0)
        self.assertTrue(r["mechanical"]["ok"])   # 力学上仍然没超

    def test_boundary_is_at_max_column_height(self):
        """在 H_max 处应恰好由 sealed 翻成 breakthrough，不能差太远。"""
        probe = caprock.caprock_integrity(co2_column_height_m=1, **self.base)
        h_max = probe["max_column_height_m"]
        just_under = caprock.caprock_integrity(co2_column_height_m=h_max * 0.98, **self.base)
        just_over = caprock.caprock_integrity(co2_column_height_m=h_max * 1.02, **self.base)
        self.assertEqual(just_under["verdict"], "sealed")
        self.assertEqual(just_over["verdict"], "capillary_breakthrough")

    def test_mechanical_failure_is_a_separate_path(self):
        """柱高很小但注入压力超过 σhmin + T_0 → 力学失效，不是毛细管。"""
        r = caprock.caprock_integrity(
            co2_column_height_m=10,
            **{**self.base, "injection_pressure_bar": 500},
        )
        self.assertEqual(r["verdict"], "mechanical_failure")
        self.assertTrue(r["blocking"])
        self.assertFalse(r["mechanical"]["ok"])

    def test_density_difference_comes_from_peng_robinson(self):
        """
        Δρ 必须来自真实状态方程。旧式 1.9·P/T 会给 Δρ ≈ 1029，
        于是 H_max 被高估近 3 倍 —— 把"封不住"判成"封得住"。
        """
        r = caprock.caprock_integrity(co2_column_height_m=50, **self.base)
        self.assertGreater(r["density_difference"], 200)
        self.assertLess(r["density_difference"], 500)
        self.assertGreater(r["co2_density_kg_m3"], 500)

    def test_no_data_when_neither_column_nor_rate_given(self):
        r = caprock.caprock_integrity(**self.base)
        self.assertEqual(r["verdict"], "no_data")
        self.assertFalse(r["blocking"])
        self.assertGreater(r["entry_pressure_bar"], 0)

    def test_dynamic_mode_reports_breakthrough_time(self):
        r = caprock.caprock_integrity(
            injection_rate_m3_per_day=1000, **self.base
        )
        self.assertIsNotNone(r["breakthrough_days"])
        self.assertGreater(r["breakthrough_days"], 0)
        self.assertGreater(r["column_height_m"], 0)
        self.assertIn("一维", r["dynamic_note"])

    def test_faster_injection_breaks_through_sooner(self):
        slow = caprock.caprock_integrity(injection_rate_m3_per_day=500, **self.base)
        fast = caprock.caprock_integrity(injection_rate_m3_per_day=5000, **self.base)
        self.assertGreater(slow["breakthrough_days"], fast["breakthrough_days"])

    def test_co2_heavier_than_brine_is_rejected(self):
        """密度差为负说明输入错了（温度/压力/盐度），不能静默算出负浮力。"""
        r = caprock.caprock_integrity(
            co2_column_height_m=50,
            **{**self.base, "rho_brine": 400.0},
        )
        self.assertEqual(r["verdict"], "no_data")
        self.assertIn("error", r)


class SensitivityTests(unittest.TestCase):
    def setUp(self):
        self.r = caprock.caprock_integrity(
            depth_m=2500, injection_pressure_bar=300, temperature_celsius=80,
            co2_column_height_m=50, environ={},
        )

    def test_three_ift_levels_are_monotonic(self):
        sens = self.r["sensitivity"]
        self.assertEqual(len(sens), 3)
        pcs = [s["entry_pressure_bar"] for s in sens]
        self.assertEqual(pcs, sorted(pcs))
        heights = [s["max_column_height_m"] for s in sens]
        self.assertEqual(heights, sorted(heights))

    def test_sensitivity_is_reported_not_hidden(self):
        """界面张力不建模为 T/P 函数，就必须把敏感性摆到报告里。"""
        self.assertIn("界面张力敏感性", caprock.render_block(self.r))


class DisclosureTests(unittest.TestCase):
    def setUp(self):
        self.r = caprock.caprock_integrity(
            depth_m=2500, injection_pressure_bar=300, temperature_celsius=80,
            co2_column_height_m=50, environ={},
        )

    def test_assumptions_are_listed_when_defaults_used(self):
        self.assertEqual(len(self.r["assumptions"]), 4)

    def test_assumptions_shrink_when_configured(self):
        env = {"CCUS_PORE_THROAT_NM": "5"}
        r = caprock.caprock_integrity(
            depth_m=2500, injection_pressure_bar=300, temperature_celsius=80,
            co2_column_height_m=50, environ=env,
        )
        self.assertNotIn("pore_throat_radius_nm", r["assumptions"])
        self.assertIn("ift_mnm", r["assumptions"])

    def test_gravity_segregation_bias_is_stated_with_its_direction(self):
        """
        一维模型忽略重力分异 → 柱高被低估 → 突破时间偏长。
        这个偏差朝**乐观**方向，必须写明，否则会被当成安全裕度。
        """
        joined = " ".join(self.r["limitations"])
        self.assertIn("重力分异", joined)
        self.assertIn("偏乐观", joined)

    def test_every_parameter_has_a_source(self):
        fields = set(caprock.CaprockParams().__dataclass_fields__)
        self.assertEqual(fields, set(caprock.PARAMETER_SOURCES))
        for key, rec in self.r["params"].items():
            self.assertTrue(rec["source"].strip(), key)

    def test_report_block_contains_the_numbers(self):
        block = caprock.render_block(self.r)
        self.assertIn("突破压力", block)
        self.assertIn("柱高", block)
        self.assertIn("Peng-Robinson", block)


class IntegrationWithGeomechanicsTests(unittest.TestCase):
    def test_custom_stress_params_change_the_mechanical_limit(self):
        """
        力学界限用的是 σhmin + T_0，而 σhmin 来自应力模型。
        换一组应力参数，界限必须跟着变 —— 否则说明参数没接上。
        """
        base = dict(depth_m=2500, injection_pressure_bar=300,
                    temperature_celsius=80, co2_column_height_m=10, environ={})
        weak = caprock.caprock_integrity(
            stress_params=gm.StressParams(shmin_ratio=0.5), **base)
        strong = caprock.caprock_integrity(
            stress_params=gm.StressParams(shmin_ratio=0.9), **base)
        self.assertLess(weak["mechanical"]["limit_bar"], strong["mechanical"]["limit_bar"])

    def test_weak_shallow_formation_can_fail_mechanically_at_modest_pressure(self):
        """浅层 + 低水平应力：300 bar 就足够压开盖层。"""
        r = caprock.caprock_integrity(
            depth_m=800, injection_pressure_bar=300, temperature_celsius=45,
            co2_column_height_m=10, stress_params=gm.StressParams(shmin_ratio=0.6),
            environ={},
        )
        self.assertEqual(r["verdict"], "mechanical_failure")


if __name__ == "__main__":
    unittest.main()
