# -*- coding: utf-8 -*-
"""
``utils/poroelastic.py`` 与 ``geomechanics`` 耦合口径的回归测试。

这个模块没有实测数据可对（手边没有 DFIT、没有应力测量），所以正确性靠
三条内部证据：

1. **两个端点必须精确成立**：α=1、ν=0 时应力路径系数必须是 1（Terzaghi
   极限，总应力跟着孔隙压力等量涨），ν→0.5 时趋于 0。
2. **恒等式必须闭合**：Δσ'_h/ΔP = Δσ_h/ΔP − α 是定义式的直接推论，
   两个函数各算一次，差一个 ulp 都说明其中一个写错了。
3. **单调性必须可证明**：τ = √(s(1−s))·|σh − σv| 在 σh 上升时下降，
   σn 同时上升，所以耦合口径的滑移比**恒不大于**保守口径。这不是经验
   判断，是闭式解的推论 —— 用网格逐点验证，任何一格反了都是 bug。

外加一件事：耦合口径**只对注入层成立**。这条边界不是文档措辞，而是
``coupled["note"]`` 里必须出现的字符串 —— 报告会把它抄进去，测试钉住它。
"""
import math
import os
import sys
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils import geomechanics as gm  # noqa: E402
from utils import poroelastic as pe  # noqa: E402


class StressPathCoefficientTests(unittest.TestCase):
    def test_terzaghi_limit(self):
        """α=1、ν=0：Δσh/ΔP=1，有效应力不再随孔隙压力变化。"""
        self.assertAlmostEqual(pe.horizontal_stress_path(1.0, 0.0), 1.0, places=12)
        self.assertAlmostEqual(pe.effective_stress_path(1.0, 0.0), 0.0, places=12)

    def test_hand_value_for_sandstone(self):
        """α=0.8、ν=0.25：0.8×0.5/0.75 = 0.5333；有效路径 −0.2667。"""
        self.assertAlmostEqual(pe.horizontal_stress_path(0.8, 0.25), 0.533333, places=6)
        self.assertAlmostEqual(pe.effective_stress_path(0.8, 0.25), -0.266667, places=6)

    def test_identity_between_total_and_effective_path(self):
        """Δσ'_h = Δσ_h − αΔP 是定义式，两个函数必须闭合。"""
        for alpha in (0.0, 0.3, 0.6, 0.8, 1.0):
            for nu in (0.0, 0.15, 0.25, 0.35, 0.45):
                with self.subTest(alpha=alpha, nu=nu):
                    total = pe.horizontal_stress_path(alpha, nu)
                    eff = pe.effective_stress_path(alpha, nu)
                    self.assertAlmostEqual(eff, total - alpha, places=12)

    def test_incompressible_limit_vanishes(self):
        """ν→0.5：侧向受限不产生附加水平应力（系数→0）。"""
        self.assertAlmostEqual(pe.horizontal_stress_path(1.0, 0.499999), 0.0, places=4)

    def test_path_decreases_with_poisson(self):
        values = [pe.horizontal_stress_path(0.8, nu) for nu in (0.0, 0.1, 0.2, 0.3, 0.4)]
        self.assertEqual(values, sorted(values, reverse=True))

    def test_path_increases_with_biot(self):
        values = [pe.horizontal_stress_path(a, 0.25) for a in (0.0, 0.2, 0.5, 0.8, 1.0)]
        self.assertEqual(values, sorted(values))

    def test_effective_path_is_never_positive(self):
        """注入永远降低有效水平应力 —— 返回正数说明符号写反了。"""
        for alpha in (0.1, 0.5, 1.0):
            for nu in (0.01, 0.2, 0.45):
                self.assertLessEqual(pe.effective_stress_path(alpha, nu), 0.0)

    def test_rejects_poisson_at_or_above_half(self):
        """ν=0.5 会让分母为零并让系数变号，必须拒绝而不是返回负数。"""
        for nu in (0.5, 0.6, -0.1):
            with self.assertRaises(ValueError):
                pe.horizontal_stress_path(0.8, nu)

    def test_rejects_biot_out_of_range(self):
        for alpha in (-0.1, 1.1):
            with self.assertRaises(ValueError):
                pe.horizontal_stress_path(alpha, 0.25)


class StressAfterInjectionTests(unittest.TestCase):
    def setUp(self):
        self.params = pe.PoroelasticParams(alpha=0.8, poisson=0.25)

    def test_vertical_stress_is_unchanged(self):
        out = pe.stress_after_injection(57.5, 43.1, 51.8, 25.3, 10.0, self.params)
        self.assertAlmostEqual(out["sv"], 57.5, places=12)

    def test_pore_pressure_rises_by_the_increment(self):
        out = pe.stress_after_injection(57.5, 43.1, 51.8, 25.3, 10.0, self.params)
        self.assertAlmostEqual(out["pp"], 35.3, places=12)

    def test_both_horizontal_stresses_rise_equally(self):
        """单轴应变是各向同性的水平加荷：Shmin 与 SHmax 同一个增量。"""
        out = pe.stress_after_injection(57.5, 43.1, 51.8, 25.3, 10.0, self.params)
        self.assertAlmostEqual(out["shmin"] - 43.1, out["shmax"] - 51.8, places=12)
        self.assertAlmostEqual(out["shmin"] - 43.1, 5.3333, places=4)

    def test_matches_the_closed_form(self):
        """
        σh = ν/(1−ν)·σv + α(1−2ν)/(1−ν)·P 是应力路径的解析式。

        用注入后的绝对应力与解析式对一次，确认 ``stress_after_injection``
        真的在解这个式子，而不是把增量随手加上去。
        """
        alpha, nu = 0.8, 0.25
        out = pe.stress_after_injection(57.5, 43.1, 51.8, 25.3, 10.0,
                                       pe.PoroelasticParams(alpha, nu))
        analytic = nu / (1 - nu) * 57.5 + alpha * (1 - 2 * nu) / (1 - nu) * out["pp"]
        # 解析式给的是"总水平应力"，用 Shmin 的基准差校正到同一口径
        self.assertAlmostEqual(out["shmin"] - 43.1, analytic - (nu / (1 - nu) * 57.5
                               + alpha * (1 - 2 * nu) / (1 - nu) * 25.3), places=9)

    def test_zero_biot_means_no_stress_change(self):
        out = pe.stress_after_injection(57.5, 43.1, 51.8, 25.3, 10.0,
                                       pe.PoroelasticParams(alpha=0.0, poisson=0.25))
        self.assertAlmostEqual(out["shmin"], 43.1, places=12)
        self.assertAlmostEqual(out["shmax"], 51.8, places=12)
        self.assertAlmostEqual(out["pp"], 35.3, places=12)


class ConfigTests(unittest.TestCase):
    def test_env_override_and_assumed_list(self):
        env = {"CCUS_BIOT_ALPHA": "0.6"}
        params = pe.load_poroelastic_params(env)
        self.assertAlmostEqual(params.alpha, 0.6)
        self.assertNotIn("alpha", pe._assumed(env))
        self.assertIn("poisson", pe._assumed(env))

    def test_out_of_range_env_falls_back_instead_of_clamping(self):
        """越界回退默认：截断会把"配错了"悄悄变成一个合法值。"""
        params = pe.load_poroelastic_params({"CCUS_POISSON": "0.7"})
        self.assertAlmostEqual(params.poisson, 0.25)

    def test_manifest_covers_every_field_with_a_source(self):
        fields = set(pe.PoroelasticParams().__dataclass_fields__)
        self.assertEqual(fields, set(pe.PARAMETER_SOURCES))
        for key, rec in pe.manifest()["parameters"].items():
            self.assertTrue(rec["source"].strip(), key)

    def test_sensitivity_table_is_monotone_in_biot(self):
        rows = pe.sensitivity(alpha_values=(0.6, 0.8, 1.0), poisson_values=(0.25,))
        ratios = [r["path_ratio"] for r in rows]
        self.assertEqual(ratios, sorted(ratios))


class CoupledVerdictTests(unittest.TestCase):
    """耦合口径与保守口径的关系 —— 本文件最重要的一组。"""

    def _severity(self, verdict):
        return {"stable": 0, "near_critical": 1, "reactivated": 2,
                "tensile_opening": 3}[verdict]

    def test_coupled_verdict_is_never_more_severe_on_a_grid(self):
        """
        τ = √(s(1−s))·|σh − σv| 随 σh 上升而下降，σn 上升 ——
        两个方向都让滑移比变小，所以**注入工况下**耦合口径恒不更严重。

        ΔP < 0（输入压力低于原始孔隙压力）时方向反转，那一支单独测。
        """
        for depth in (1200.0, 2500.0, 3500.0):
            for pressure in (250.0, 320.0, 380.0, 450.0):
                for dip in (30.0, 45.0, 60.0, 75.0):
                    for azimuth in (0.0, 45.0, 90.0):
                        with self.subTest(depth=depth, p=pressure, dip=dip, az=azimuth):
                            r = gm.fault_reactivation(depth, pressure, dip, azimuth)
                            if r["applied_overpressure_bar"] < 0:
                                continue
                            self.assertLessEqual(
                                self._severity(r["coupled"]["verdict"]),
                                self._severity(r["verdict"]),
                            )

    def test_coupled_slip_ratio_never_exceeds_fixed(self):
        for depth in (1500.0, 2500.0, 3200.0):
            for pressure in (280.0, 360.0, 430.0):
                with self.subTest(depth=depth, p=pressure):
                    r = gm.fault_reactivation(depth, pressure)
                    if r["applied_overpressure_bar"] < 0:
                        continue
                    fixed = r["slip_ratio"]
                    coupled = r["coupled"]["slip_ratio"]
                    if fixed == "inf" or coupled == "inf":
                        continue
                    self.assertLessEqual(coupled, fixed)

    def test_depletion_direction_reverses_the_inequality(self):
        """
        3200 m 处原始孔隙压力约 323 bar；给 280 bar 的输入压力是**降压**，
        应力路径把 σh 往下带，剪应力升、正应力降 → 耦合口径更严重。

        这条测试存在的意义：防止把"耦合更宽松"当成无条件结论。
        """
        r = gm.fault_reactivation(3200, 280)
        self.assertLess(r["applied_overpressure_bar"], 0)
        self.assertTrue(r["coupled"]["depressurisation"])
        self.assertGreater(float(r["coupled"]["slip_ratio"]), float(r["slip_ratio"]))
        self.assertIn("应力路径反向", r["coupled"]["note"])

    def test_injection_direction_is_flagged_as_such(self):
        r = gm.fault_reactivation(2500, 380)
        self.assertFalse(r["coupled"]["depressurisation"])
        self.assertNotIn("应力路径反向", r["coupled"]["note"])

    def test_zero_biot_reproduces_the_conservative_verdict_exactly(self):
        """α=0 时没有任何耦合效应，两个口径必须逐位相同。"""
        r = gm.fault_reactivation(2500, 380, poro=pe.PoroelasticParams(0.0, 0.25))
        self.assertEqual(r["coupled"]["verdict"], r["verdict"])
        self.assertAlmostEqual(float(r["coupled"]["slip_ratio"]),
                               float(r["slip_ratio"]), places=9)

    def test_reference_case_loosens_from_reactivated_to_stable(self):
        """
        2500 m / 380 bar 是 README 里引用的工况。

        保守口径判 reactivated（滑移比 1.19），耦合口径判 stable（0.396）。
        把这两个数钉住：物理一旦改动，报告里的例子会立刻不一致。
        """
        r = gm.fault_reactivation(2500, 380)
        self.assertEqual(r["verdict"], "reactivated")
        self.assertEqual(r["coupled"]["verdict"], "stable")
        self.assertAlmostEqual(float(r["slip_ratio"]), 1.19, places=2)
        self.assertAlmostEqual(float(r["coupled"]["slip_ratio"]), 0.396, places=3)
        self.assertAlmostEqual(r["coupled"]["path_ratio"], 0.5333, places=4)

    def test_coupled_block_carries_its_scope_note(self):
        """边界必须跟着数字走：耦合口径只对注入层成立。"""
        r = gm.fault_reactivation(2500, 380)
        self.assertIn("应力转移", r["coupled"]["note"])
        self.assertIn("注入层", r["coupled"]["note"])

    def test_coupled_block_has_every_documented_field(self):
        r = gm.fault_reactivation(2500, 380)
        for key in ("verdict", "slip_ratio", "sigma_n_mpa", "sigma_n_eff_mpa",
                    "tau_mpa", "pp_crit_bar", "margin_bar", "delta_sigma_h_bar",
                    "path_ratio", "effective_path_ratio", "state", "params",
                    "assumed", "note"):
            self.assertIn(key, r["coupled"])

    def test_limitations_no_longer_claim_coupling_is_absent(self):
        """
        旧版 limitations 写着"孔隙弹性耦合未建模"。现在建模了 ——
        那句话必须消失，否则读者会以为这个字段不存在。
        """
        r = gm.fault_reactivation(2500, 380)
        joined = " ".join(r["limitations"])
        self.assertNotIn("孔隙弹性耦合未建模", joined)
        self.assertIn("应力转移", joined)

    def test_render_block_reports_both_paths_when_they_differ(self):
        block = gm.render_block(gm.fault_reactivation(2500, 380))
        self.assertIn("孔隙弹性口径", block)
        self.assertIn("比保守口径宽松", block)

    def test_render_block_stays_silent_when_paths_agree(self):
        """两个口径结论一致时不加那一行，避免把报告塞满同义句。"""
        r = gm.fault_reactivation(2500, 380, poro=pe.PoroelasticParams(0.0, 0.25))
        self.assertNotIn("比保守口径宽松", gm.render_block(r))

    def test_result_is_deterministic(self):
        a = gm.fault_reactivation(2500, 380)
        b = gm.fault_reactivation(2500, 380)
        self.assertEqual(a["coupled"], b["coupled"])


class NoMagicNumberTests(unittest.TestCase):
    def test_every_coupled_parameter_is_documented(self):
        """耦合口径返回的参数必须每一项都带出处。"""
        r = gm.fault_reactivation(2500, 380)
        for key, rec in r["coupled"]["params"].items():
            self.assertTrue(rec["source"].strip(), key)
            self.assertIn("unit", rec)


if __name__ == "__main__":
    unittest.main()
