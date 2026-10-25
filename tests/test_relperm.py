# -*- coding: utf-8 -*-
"""
``utils/relperm.py`` 的回归测试。

核心风险不是"公式写错"——Brooks-Corey 只有三行——而是**方向搞反**：
CO₂ 驱替地层水，切线要从气端（S_g=0, f_g=0）出发；如果照着水驱油的教材
从水端作切线，算出的 CO₂ 饱和度会从 0.49 变成 0.17，差三倍，而且代码
不报错、结果"看起来合理"。测试里专门有一条钉住这个方向。
"""
import os
import sys
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils import relperm as rp  # noqa: E402


class RelPermTests(unittest.TestCase):
    def setUp(self):
        self.p = rp.load_relperm_params({})

    def test_endpoints(self):
        """束缚水处只有 CO₂ 流动，残余气处只有水流动。"""
        at_swr = rp.relperm(self.p.swr, self.p)
        self.assertAlmostEqual(at_swr["kr_w"], 0.0)
        self.assertAlmostEqual(at_swr["kr_g"], 1.0)
        self.assertAlmostEqual(at_swr["f_w"], 0.0)

        at_sgr = rp.relperm(1.0 - self.p.sgr, self.p)
        self.assertAlmostEqual(at_sgr["kr_w"], 1.0)
        self.assertAlmostEqual(at_sgr["kr_g"], 0.0)
        self.assertAlmostEqual(at_sgr["f_w"], 1.0)

    def test_saturation_is_clamped_not_extrapolated(self):
        """
        越界夹紧而不是外推：外推会让 kr 变负，而分流曲线在端点外没有物理意义。
        夹紧是显式的；外推是静默的，会污染下游积分。
        """
        below = rp.relperm(-0.5, self.p)
        self.assertAlmostEqual(below["kr_w"], 0.0)
        self.assertAlmostEqual(below["kr_g"], 1.0)
        above = rp.relperm(1.5, self.p)
        self.assertAlmostEqual(above["kr_w"], 1.0)
        self.assertAlmostEqual(above["kr_g"], 0.0)

    def test_fractional_flow_is_monotonic_and_bounded(self):
        prev = -1.0
        for i in range(0, 101):
            sw = self.p.swr + (1.0 - self.p.sgr - self.p.swr) * i / 100.0
            f = rp.fractional_flow(sw, self.p)
            self.assertGreaterEqual(f, 0.0)
            self.assertLessEqual(f, 1.0)
            self.assertGreaterEqual(f, prev - 1e-12)
            prev = f

    def test_high_viscosity_ratio_makes_co2_much_more_mobile(self):
        """CO₂ 黏度约水的 1/8，流度比应远大于 1 —— 这是指进的根源。"""
        front = rp.bl_front(self.p)
        self.assertGreater(front["mobility_ratio"], 5.0)


class BLFrontTests(unittest.TestCase):
    def setUp(self):
        self.p = rp.load_relperm_params({})
        self.front = rp.bl_front(self.p)

    def test_tangent_condition_holds(self):
        """
        前沿处必须满足切线条件：f_g/S_g = df_g/dS_g。

        这是 Welge 切线法的定义式，也是唯一能自我验证的地方 ——
        偏离它说明最大化那一步没收敛。
        """
        f = self.front
        self.assertAlmostEqual(
            f["fg_front"] / f["sg_front"], f["dfg_front"], delta=1e-3
        )

    def test_front_saturation_is_the_co2_side_not_the_water_side(self):
        """
        方向锁：CO₂ 驱水下，前沿 CO₂ 饱和度约 0.40、平均约 0.49。

        若误按"水驱 CO₂"的方向求切线，会得到 0.17 / 0.17 量级 ——
        差三倍且不报错。这条测试就是防止有人"照教材改回来"。
        """
        self.assertGreater(self.front["sg_front"], 0.30)
        self.assertGreater(self.front["co2_saturation"], 0.40)
        self.assertLess(self.front["co2_saturation"], 0.60)

    def test_average_is_above_front_value(self):
        """前沿后平均饱和度必须高于前沿处饱和度（剖面往入口方向升高）。"""
        self.assertGreater(self.front["co2_saturation"], self.front["sg_front"])

    def test_saturations_are_complementary(self):
        self.assertAlmostEqual(
            self.front["co2_saturation"] + self.front["avg_sw_behind"], 1.0, places=6
        )

    def test_mobility_ratio_formula(self):
        """M = (kr_g^o/μ_g)/(kr_w^o/μ_w)，端点相渗为 1 → M = μ_w/μ_g。"""
        self.assertAlmostEqual(
            self.front["mobility_ratio"], self.p.mu_w / self.p.mu_g, delta=0.01
        )

    def test_front_moves_faster_when_viscosity_contrast_grows(self):
        low = rp.bl_front(rp.RelPermParams(mu_g=0.3))
        high = rp.bl_front(rp.RelPermParams(mu_g=0.02))
        self.assertLess(low["dfg_front"], high["dfg_front"])

    def test_result_carries_sources(self):
        for key, rec in self.front["params"].items():
            self.assertTrue(rec["source"].strip(), key)
            self.assertTrue(rec["unit"].strip(), key)


class ParameterTests(unittest.TestCase):
    def test_every_parameter_has_a_source(self):
        fields = set(rp.RelPermParams().__dataclass_fields__)
        self.assertEqual(fields, set(rp.PARAMETER_SOURCES))

    def test_env_override(self):
        p = rp.load_relperm_params({"CCUS_SWR": "0.3", "CCUS_MU_G": "0.05"})
        self.assertAlmostEqual(p.swr, 0.3)
        self.assertAlmostEqual(p.mu_g, 0.05)

    def test_out_of_range_falls_back(self):
        p = rp.load_relperm_params({"CCUS_SWR": "5"})
        self.assertAlmostEqual(p.swr, 0.25)

    def test_saturated_residuals_fall_back_to_defaults(self):
        """
        swr + sgr 占满可动孔隙时分母为 0，必须回退默认而不是除零崩溃。
        """
        p = rp.load_relperm_params({"CCUS_SWR": "0.6", "CCUS_SGR": "0.4"})
        self.assertAlmostEqual(p.swr, 0.25)
        self.assertAlmostEqual(p.sgr, 0.15)


class InjectivityTests(unittest.TestCase):
    def test_matches_radial_darcy_ordering(self):
        """渗透率越高、层越厚，注入指数越大 —— 方向不能反。"""
        base = rp.injectivity_index(100, 20, 0.06)["J_m3_per_day_per_mpa"]
        thick = rp.injectivity_index(100, 40, 0.06)["J_m3_per_day_per_mpa"]
        perm = rp.injectivity_index(200, 20, 0.06)["J_m3_per_day_per_mpa"]
        self.assertAlmostEqual(thick / base, 2.0, delta=1e-6)
        self.assertAlmostEqual(perm / base, 2.0, delta=1e-6)

    def test_rejects_impossible_inputs(self):
        with self.assertRaises(ValueError):
            rp.injectivity_index(0, 20, 0.06)


if __name__ == "__main__":
    unittest.main()
