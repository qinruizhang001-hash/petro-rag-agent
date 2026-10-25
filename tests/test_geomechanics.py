# -*- coding: utf-8 -*-
"""
``utils/geomechanics.py`` 的回归测试。

这个模块没有任何实测数据可以对照 —— 手边没有 DFIT、没有井壁崩落反演。
所以正确性只能靠**内部自证**，测试的重点也在这里：

1. 两条独立路径算同一个量必须重合：指定产状断层的临界孔隙压力
   （σn − τ/μ）与摩擦平衡式反解（(Kσ3 − σ1)/(K−1)）在最优倾角处相等。
   一个用三维张量投影，一个用一维主应力代数，写错任何一个都会露馅。
2. 最优倾角必须回到经典结论 45° + φ/2（μ=0.6 → 60.5°）。
3. 两个判定口径（滑移比 / 压力余量）在**全区间**必须给出同一结论。
   它们数学等价，不一致只能是代码 bug。

外加参数可配置性与"不许有魔法数"的静态检查。
"""
import math
import os
import sys
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils import geomechanics as gm  # noqa: E402


class RegimeTests(unittest.TestCase):
    def test_normal_faulting_when_vertical_stress_is_largest(self):
        self.assertEqual(gm.classify_regime(sv=57.5, shmin=43.1, shmax=51.8), "NF")

    def test_strike_slip_when_shmax_is_largest(self):
        self.assertEqual(gm.classify_regime(sv=50.0, shmin=30.0, shmax=70.0), "SS")

    def test_reverse_faulting_when_shmin_is_largest(self):
        self.assertEqual(gm.classify_regime(sv=40.0, shmin=60.0, shmax=80.0), "RF")

    def test_ties_do_not_flip_between_calls(self):
        """相等时归属固定，判定结果要写进报告，不能随浮点相等与否漂移。"""
        self.assertEqual(gm.classify_regime(sv=50.0, shmin=50.0, shmax=50.0), "NF")
        self.assertEqual(gm.classify_regime(sv=50.0, shmin=50.0, shmax=50.0), "NF")


class FrictionalLimitTests(unittest.TestCase):
    def test_matches_published_values(self):
        """μ=0.6 → 3.12；μ=0.85 → 4.68（Zoback 2007, eq. 4.6 的常用取值）。"""
        self.assertAlmostEqual(gm.frictional_limit(0.6), 3.119, places=2)
        self.assertAlmostEqual(gm.frictional_limit(0.85), 4.676, places=2)

    def test_limit_grows_with_friction(self):
        self.assertLess(gm.frictional_limit(0.4), gm.frictional_limit(0.6))
        self.assertLess(gm.frictional_limit(0.6), gm.frictional_limit(0.9))


class StressPolygonTests(unittest.TestCase):
    def test_default_profile_is_inside_the_polygon(self):
        p = gm.load_stress_params({})
        st = gm.stress_at_depth(2500, p)
        poly = gm.stress_polygon(st, p)
        self.assertTrue(poly["ok"], poly)
        self.assertLess(poly["ratio"], poly["limit"])

    def test_overestimated_shmax_is_caught(self):
        """
        SHmax 给到 1.2×Sv 时，差应力超过摩擦上限 —— 地壳里不可能长期存在。

        这个检查的价值不在于"断层要滑了"，而在于**输入数据不自洽**：
        多半是 SHmax 测高了。若不做这一步，后面所有安全结论都建立在错数据上。
        """
        p = gm.StressParams(shmin_ratio=0.4, shmax_ratio=1.4, friction=0.6)
        st = gm.stress_at_depth(2500, p)
        poly = gm.stress_polygon(st, p)
        self.assertFalse(poly["ok"])
        self.assertGreater(poly["ratio"], poly["limit"])

    def test_zero_effective_stress_is_not_a_number(self):
        """孔隙压力压到最小主应力时有效应力归零，比值是无穷，不能返回 NaN。"""
        p = gm.StressParams(pp_gradient=20.0)
        st = gm.stress_at_depth(2500, p)
        poly = gm.stress_polygon(st, p)
        self.assertFalse(poly["ok"])
        self.assertEqual(poly["ratio"], float("inf"))


class PlaneStressTests(unittest.TestCase):
    def test_3d_reduces_to_the_textbook_2d_formula(self):
        """
        倾向沿 Shmin（azimuth=90°）时，三维张量投影必须退化成课本的
        σn = σv·cos²δ + σh·sin²δ、τ = (σv−σh)·sinδ·cosδ。

        这条退化关系是三维实现唯一能验证的地方 —— 否则写错一个分量也不会报错。
        """
        p = gm.load_stress_params({})
        st = gm.stress_at_depth(2500, p)
        for dip in (20.0, 45.0, 60.0, 75.0):
            pl = gm.plane_stress(dip, 90.0, st)
            d = math.radians(dip)
            sigma_n = st.sv * math.cos(d) ** 2 + st.shmin * math.sin(d) ** 2
            tau = (st.sv - st.shmin) * math.sin(d) * math.cos(d)
            self.assertAlmostEqual(pl["sigma_n_mpa"], sigma_n, places=6)
            self.assertAlmostEqual(pl["tau_mpa"], tau, places=6)

    def test_other_strikes_resolve_less_shear(self):
        """
        倾向偏离 Shmin 时剪应力必须更小 —— 这正是不能用二维式硬套的理由：
        报告只给"倾角 60°"时，套二维式等于默认了最不利走向，会高估风险。
        """
        p = gm.load_stress_params({})
        st = gm.stress_at_depth(2500, p)
        worst = gm.plane_stress(60.0, 90.0, st)["tau_mpa"]
        for azimuth in (0.0, 30.0, 60.0, 120.0):
            self.assertLessEqual(gm.plane_stress(60.0, azimuth, st)["tau_mpa"], worst + 1e-9)


class OptimalDipTests(unittest.TestCase):
    def test_recovers_the_classical_45_plus_phi_over_2(self):
        """
        随压力上升最先滑的产状是 45° + φ/2，φ = arctan μ。

        实测过：如果错把目标函数写成"当前滑移趋势最大"，正断层区会算出
        53.5°，偏差 7° 且**不报错** —— 只是把最不利产状算轻了。所以这条要钉住。
        """
        p = gm.load_stress_params({})
        classic = 45.0 + math.degrees(math.atan(p.friction)) / 2.0
        for depth in (1200, 2500, 3800):
            st = gm.stress_at_depth(depth, p)
            self.assertAlmostEqual(gm.optimal_dip(st, p), classic, delta=0.6)

    def test_optimal_dip_tracks_friction(self):
        """摩擦越大，最不利倾角越陡（φ 增大）。"""
        st = gm.stress_at_depth(2500, gm.StressParams())
        low = gm.optimal_dip(st, gm.StressParams(friction=0.4))
        high = gm.optimal_dip(st, gm.StressParams(friction=0.9))
        self.assertLess(low, high)


class TwoPathAgreementTests(unittest.TestCase):
    """
    两条独立路径必须重合。这是本模块最重要的测试。

    路径 A：三维张量投影 → σn、τ → Pp_crit = σn − τ/μ
    路径 B：一维主应力代数 → Pp_crit = (Kσ3 − σ1)/(K−1)

    前者若把法向量写错，或后者若把 K 的括号写错，都只在数值上体现，
    不会抛异常。
    """

    def test_optimal_plane_matches_the_frictional_limit_formula(self):
        for friction in (0.4, 0.6, 0.85):
            p = gm.StressParams(friction=friction)
            for depth in (1000, 2500, 4000):
                st = gm.stress_at_depth(depth, p)
                dip = gm.optimal_dip(st, p)
                pl = gm.plane_stress(dip, 90.0, st)
                path_a = gm.critical_pore_pressure(st, pl["sigma_n_mpa"], pl["tau_mpa"], p)
                path_b = gm.critical_pore_pressure_optimal(st, p)
                self.assertAlmostEqual(
                    path_a, path_b, places=3,
                    msg=f"μ={friction} depth={depth} dip={dip}: {path_a} vs {path_b}",
                )

    def test_off_optimal_planes_need_more_pressure(self):
        """非最优产状需要的压力一定更高，否则最优产状的"最优"没有意义。"""
        p = gm.load_stress_params({})
        st = gm.stress_at_depth(2500, p)
        best = gm.critical_pore_pressure_optimal(st, p)
        for dip in (20.0, 35.0, 45.0, 80.0, 89.0):
            pl = gm.plane_stress(dip, 90.0, st)
            crit = gm.critical_pore_pressure(st, pl["sigma_n_mpa"], pl["tau_mpa"], p)
            self.assertGreaterEqual(crit, best - 1e-6, f"dip={dip}")


class ReactivationTests(unittest.TestCase):
    def setUp(self):
        self.p = gm.load_stress_params({})

    def test_both_criteria_agree_across_the_whole_range(self):
        """
        滑移比与压力余量是同一不等式的两种写法，全区间必须同结论。

        ``consistent`` 字段就是这个断言的运行时版本 —— 它在报告里也会出现，
        因为一旦不一致，说明判定代码坏了，而报告仍然会被发出去。
        """
        for pinj in (100, 150, 200, 250, 300, 350, 363, 364, 400, 500, 700):
            r = gm.fault_reactivation(2500, pinj, 60, 90, self.p, {})
            self.assertTrue(r["consistent"], f"{pinj} bar: {r['slip_ratio']} / {r['margin_bar']}")

    def test_reactivation_is_monotonic_in_pressure(self):
        """压力越高不可能反而更安全 —— 判据若是分段拼的，这里会露出来。"""
        # 张性开启与剪切滑动都是"失稳"，排在同一档：压力继续升高时
        # 可能先剪切后张性，二者之间不存在"更安全"的方向。
        order = {"stable": 0, "near_critical": 1, "reactivated": 2, "tensile_opening": 2}
        last = -1
        for pinj in range(100, 900, 25):
            r = gm.fault_reactivation(2500, pinj, 60, 90, self.p, {})
            rank = order[r["verdict"]]
            self.assertGreaterEqual(rank, last, f"{pinj} bar 反而更安全了")
            last = rank

    def test_reactivated_verdict_is_blocking(self):
        # 400 bar 是剪切滑动：断层面正应力约 467 bar，还没被顶开
        r = gm.fault_reactivation(2500, 400, 60, 90, self.p, {})
        self.assertEqual(r["verdict"], "reactivated")
        self.assertTrue(r["blocking"])

    def test_stable_verdict_is_not_blocking(self):
        r = gm.fault_reactivation(2500, 300, 60, 90, self.p, {})
        self.assertEqual(r["verdict"], "stable")
        self.assertFalse(r["blocking"])

    def test_margin_grows_with_depth(self):
        """
        同样的注入压力，越深越安全：孔隙压力随深度涨，但有效正应力涨得更快
        （Sv 的梯度 23 > Pp 的 10.1）。这条若反了，说明梯度用错了。
        """
        shallow = gm.fault_reactivation(1500, 300, 60, 90, self.p, {})
        deep = gm.fault_reactivation(3500, 300, 60, 90, self.p, {})
        self.assertLess(shallow["margin_bar"], deep["margin_bar"])

    def test_cohesion_can_lock_the_fault(self):
        """黏聚力大于剪应力时，剪切滑动不可能（τ ≤ C 分支）。"""
        p = gm.StressParams(cohesion=30.0)
        r = gm.fault_reactivation(2500, 300, 60, 90, p, {})
        self.assertEqual(r["verdict"], "stable")
        self.assertEqual(r["pp_crit_bar"], "inf")

    def test_tensile_opening_is_not_reported_as_shear_slip(self):
        """
        孔隙压力超过断层面正应力时，面是被顶开的，与黏聚力/摩擦系数无关。

        如果把它混进剪切判据，一条高黏聚力断层在高压下会被判成"稳定"——
        而实际上流体已经有了通道。两种机制必须分开报。
        """
        p = gm.StressParams(cohesion=30.0)
        r = gm.fault_reactivation(2500, 900, 60, 90, p, {})
        self.assertEqual(r["verdict"], "tensile_opening")
        self.assertEqual(r["mechanism"], "tensile")
        self.assertTrue(r["blocking"])
        self.assertTrue(r["tensile_opening"])
        self.assertIn("张性开启", gm.render_block(r))

    def test_shear_slip_is_labelled_shear(self):
        r = gm.fault_reactivation(2500, 400, 60, 90, self.p, {})
        self.assertEqual(r["mechanism"], "shear")
        self.assertFalse(r["tensile_opening"])

    def test_higher_friction_delays_reactivation(self):
        low = gm.fault_reactivation(2500, 500, 60, 90, gm.StressParams(friction=0.3), {})
        high = gm.fault_reactivation(2500, 500, 60, 90, gm.StressParams(friction=0.9), {})
        self.assertGreater(high["margin_bar"], low["margin_bar"])

    def test_result_carries_version_assumptions_and_limitations(self):
        r = gm.fault_reactivation(2500, 300, 60, 90, self.p, {})
        self.assertEqual(r["version"], gm.GEOMECHANICS_VERSION)
        # 没配环境变量 → 六个参数全是假设，必须全部列出
        self.assertEqual(len(r["assumptions"]), 6)
        self.assertTrue(r["limitations"])
        self.assertIn("params", r)

    def test_report_block_states_the_assumptions(self):
        """
        结论必须自带前提。报告里出现"断层稳定"而不同时说应力剖面是假设的，
        就是一份误导性报告 —— 哪怕每个数字都算对了。
        """
        r = gm.fault_reactivation(2500, 300, 60, 90, self.p, {})
        block = gm.render_block(r)
        self.assertIn("断层未活化", block)
        self.assertIn("假设", block)
        self.assertIn("滑移比", block)


class ParameterTests(unittest.TestCase):
    def test_every_parameter_has_a_source(self):
        """禁止无出处魔法数：每个可调参数都必须能在报告里回答"这个数哪来的"。"""
        for key in gm.PARAMETER_SOURCES:
            self.assertTrue(gm.PARAMETER_SOURCES[key]["source"].strip(), key)
            self.assertTrue(gm.PARAMETER_SOURCES[key]["unit"].strip(), key)
        # 参数集合必须与 dataclass 字段完全一致，加了字段忘了写出处会被发现
        fields = set(gm.StressParams().__dataclass_fields__)
        self.assertEqual(fields, set(gm.PARAMETER_SOURCES))

    def test_env_override_takes_effect(self):
        p = gm.load_stress_params({"CCUS_SHMIN_RATIO": "0.6", "CCUS_FRICTION": "0.8"})
        self.assertAlmostEqual(p.shmin_ratio, 0.6)
        self.assertAlmostEqual(p.friction, 0.8)

    def test_out_of_range_falls_back_instead_of_truncating(self):
        """漏了小数点（0.6 → 6）必须回退，不能截断成 1.2 —— 那会悄悄改结论。"""
        p = gm.load_stress_params({"CCUS_FRICTION": "6"})
        self.assertAlmostEqual(p.friction, 0.6)
        p = gm.load_stress_params({"CCUS_SHMIN_RATIO": "15"})
        self.assertAlmostEqual(p.shmin_ratio, 0.75)

    def test_garbage_falls_back(self):
        p = gm.load_stress_params({"CCUS_SV_GRADIENT": "abc"})
        self.assertAlmostEqual(p.sv_gradient, 23.0)

    def test_assumed_list_shrinks_when_configured(self):
        env = {"CCUS_SHMIN_RATIO": "0.7", "CCUS_FRICTION": "0.7"}
        p = gm.load_stress_params(env)
        r = gm.fault_reactivation(2500, 300, 60, 90, p, env)
        self.assertNotIn("shmin_ratio", r["assumptions"])
        self.assertNotIn("friction", r["assumptions"])
        self.assertIn("sv_gradient", r["assumptions"])

    def test_manifest_lists_sources_for_every_parameter(self):
        m = gm.stress_manifest()
        self.assertEqual(m["version"], gm.GEOMECHANICS_VERSION)
        for key, rec in m["parameters"].items():
            self.assertIn("source", rec)
            self.assertTrue(rec["source"].strip(), key)


if __name__ == "__main__":
    unittest.main()
