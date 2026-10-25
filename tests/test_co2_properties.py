# -*- coding: utf-8 -*-
"""
``utils/co2_properties.py`` 的回归测试。

这个文件存在的直接原因是一个 460 倍的错误：项目原先用 ``ρ = 1.9·P/T``
算 CO₂ 密度，在 250 bar / 80℃ 给出 1.35 kg/m³。测试的第一条就钉住这个量级，
防止有人"简化"回去。

没有 NIST 数据库可以对照，所以正确性靠三类证据：
1. **临界点**：真实 CO₂ 临界密度 467.6 kg/m³，PR 因 Zc 固定为 0.307 会低 15%
   —— 这是模型的已知系统偏差，测试用 20% 容差把它固定成"已知偏差"而不是"错误"。
2. **单调性**：定温升压密度必升、定压升温密度必降。写错符号这里会红。
3. **量纲关系**：密度必须在 CO₂ 物理可能区间内（超临界 200–1200 kg/m³），
   旧公式的 1.35 一眼就能被这条拦住。
"""
import os
import sys
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils import co2_properties as cp  # noqa: E402


class DensityScaleTests(unittest.TestCase):
    def test_is_not_the_old_formula(self):
        """
        旧式 1.9·P/T 给 1.35 kg/m³，新式给 600+ —— 差两个数量级以上。

        这条测试的写法（比较倍数而不是比较绝对值）是有意的：它锁的是
        "不许退回那个量纲不对的拟合式"，而不是锁某一个具体数值。
        """
        old = 1.9 * 250 / (80 + 273.15)
        new = cp.co2_density_kg_m3(250, 80)
        self.assertGreater(new / old, 100, f"新式 {new:.1f} vs 旧式 {old:.2f}")

    def test_supercritical_density_is_physically_possible(self):
        """超临界 CO₂ 密度必须在 200–1200 kg/m³；1.35 会被这条直接拦下。"""
        for p, t in ((250, 80), (300, 60), (150, 35), (400, 90), (200, 70)):
            rho = cp.co2_density_kg_m3(p, t)
            self.assertGreater(rho, 200, f"{p} bar / {t}℃ → {rho:.1f}")
            self.assertLess(rho, 1200, f"{p} bar / {t}℃ → {rho:.1f}")

    def test_critical_point_within_known_pr_bias(self):
        """
        真实临界密度 467.6 kg/m³。Peng-Robinson 把 Zc 固定在 0.307
        （真实 0.274），所以**必然**偏低约 15%。容差设 20% 是为了把这个
        偏差固定成已知量 —— 若哪天偏差突然变大，说明方程实现坏了。
        """
        rho = cp.co2_density_kg_m3(73.773, cp.TC_CELSIUS)
        self.assertAlmostEqual(rho / 467.6, 1.0, delta=0.20)

    def test_monotonic_in_pressure(self):
        prev = 0.0
        for p in (100, 150, 200, 250, 300, 400):
            rho = cp.co2_density_kg_m3(p, 60)
            self.assertGreater(rho, prev, f"{p} bar")
            prev = rho

    def test_monotonic_in_temperature(self):
        """定压下升温密度必降 —— 写反了符号这里会红。"""
        prev = float("inf")
        for t in (40, 60, 80, 100, 120):
            rho = cp.co2_density_kg_m3(300, t)
            self.assertLess(rho, prev, f"{t}℃")
            prev = rho

    def test_rejects_impossible_inputs(self):
        """越界必须抛错，不能返回 0 —— 返回 0 会让浮力静默变成"不浮"。"""
        with self.assertRaises(ValueError):
            cp.co2_density_kg_m3(250, -300)
        with self.assertRaises(ValueError):
            cp.co2_density_kg_m3(0, 60)


class BuoyancyTests(unittest.TestCase):
    def test_density_difference_is_realistic(self):
        """
        Δρ 必须在 200–500 kg/m³。旧密度式给 Δρ ≈ 1029，
        盖层柱高上限会被高估 2.8 倍以上。
        """
        drho = cp.buoyancy_density_difference(1030, 250, 80)
        self.assertGreater(drho, 200)
        self.assertLess(drho, 500)

    def test_density_difference_grows_with_pressure(self):
        """压力越高 CO₂ 越重，浮力越小。"""
        self.assertGreater(
            cp.buoyancy_density_difference(1030, 100, 60),
            cp.buoyancy_density_difference(1030, 400, 60),
        )


class ViscosityTests(unittest.TestCase):
    def test_water_viscosity_matches_standard_values(self):
        """
        Vogel 关联式的常数是 2.414e-5 **Pa·s**。这里曾经写成 2.414e-4，
        结果小 100 倍 —— 数值是个"看起来正常的小数"，不报错也不越界。
        所以直接对两个标准值断言。
        """
        self.assertAlmostEqual(cp.water_viscosity_mpa_s(20), 1.002, delta=0.01)
        self.assertAlmostEqual(cp.water_viscosity_mpa_s(80), 0.355, delta=0.01)

    def test_water_viscosity_decreases_with_temperature(self):
        self.assertGreater(cp.water_viscosity_mpa_s(20), cp.water_viscosity_mpa_s(80))
        self.assertGreater(cp.water_viscosity_mpa_s(80), cp.water_viscosity_mpa_s(150))

    def test_co2_viscosity_is_constant_with_documented_range(self):
        v = cp.co2_viscosity_mpa_s(250, 60, environ={})
        self.assertAlmostEqual(v, cp.CO2_VISCOSITY_DEFAULT)
        self.assertLess(v, cp.water_viscosity_mpa_s(60))

    def test_viscosity_override_and_fallback(self):
        self.assertAlmostEqual(
            cp.co2_viscosity_mpa_s(250, 60, environ={"CCUS_CO2_VISCOSITY": "0.08"}), 0.08
        )
        # 单位写错（Pa·s 写成 0.00006 或 cP 写成 60）必须回退
        self.assertAlmostEqual(
            cp.co2_viscosity_mpa_s(250, 60, environ={"CCUS_CO2_VISCOSITY": "60"}),
            cp.CO2_VISCOSITY_DEFAULT,
        )


class InterfacialTensionTests(unittest.TestCase):
    def test_default_is_inside_the_literature_range(self):
        self.assertTrue(20.0 <= cp.IFT_DEFAULT <= 30.0)

    def test_override_and_fallback(self):
        self.assertAlmostEqual(
            cp.interfacial_tension_mnm(environ={"CCUS_IFT_MNM": "22"}), 22.0
        )
        self.assertAlmostEqual(
            cp.interfacial_tension_mnm(environ={"CCUS_IFT_MNM": "500"}),
            cp.IFT_DEFAULT,
        )


class PhaseTests(unittest.TestCase):
    def test_labels(self):
        self.assertEqual(cp.co2_phase(250, 80), "supercritical")
        self.assertEqual(cp.co2_phase(50, 20), "gas")
        self.assertEqual(cp.co2_phase(150, 20), "liquid")


if __name__ == "__main__":
    unittest.main()
