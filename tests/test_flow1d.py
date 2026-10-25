# -*- coding: utf-8 -*-
"""
``utils/flow1d.py`` 的回归测试。

数值求解器的测试只有一种可信写法：**与解析解对标**。
一维 Buckley-Leverett 有严格的解析解（切线法），所以这里对三件事逐一比对：

    前沿位置        x_f = (df_g/dS_g)|front · t_D
    前沿紧后方饱和度  S_g ≈ S_gf
    前沿后平均饱和度  S̄_g ≈ Welge 值

外加物质守恒。守恒这条最容易被忽视，也最危险：一个边界条件写错的求解器
会让质量悄悄流失，表现为"柱高算小了、盖层判安全"——而剖面看起来完全正常。
"""
import os
import sys
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils import flow1d, relperm as rp  # noqa: E402


class ConservationTests(unittest.TestCase):
    def test_mass_is_conserved_before_breakthrough(self):
        """
        突破前，注入的孔隙体积必须全部还在域内。

        注意必须取突破**前**的时刻：解析突破时间 t_D = 1/dfg ≈ 0.494，
        超过它质量会正常流出，守恒检查就不成立了（那是物理，不是 bug）。
        """
        for t_end in (0.1, 0.2, 0.3, 0.45):
            r = flow1d.solve(n_cells=200, t_end=t_end)
            self.assertLess(r["mass_error"], 1e-6, f"t_D={t_end}")

    def test_mass_leaves_after_breakthrough(self):
        """突破后域内质量必须少于注入量，否则说明出口边界被写成了墙。"""
        r = flow1d.solve(n_cells=200, t_end=0.8)
        self.assertLess(r["mass_in_place"], r["mass_injected"])

    def test_saturation_never_exceeds_injected_value(self):
        r = flow1d.solve(n_cells=200, t_end=0.4)
        sg_inj = 1.0 - r["analytic"]["params"]["swr"]["value"]
        self.assertLessEqual(max(r["sg"]), sg_inj + 1e-9)
        self.assertGreaterEqual(min(r["sg"]), -1e-12)


class AnalyticBenchmarkTests(unittest.TestCase):
    """
    数值解 vs 解析解。容差按数值扩散的量级设定，不追求过紧：
    一阶迎风的前沿本来就比解析解宽，要求 1e-4 一致只会逼着把格式改复杂，
    而工程判据不需要那个精度。
    """

    def setUp(self):
        self.cases = {}
        for t_end in (0.15, 0.3, 0.45):
            self.cases[t_end] = flow1d.solve(n_cells=400, t_end=t_end)

    def test_front_position_matches(self):
        for t_end, r in self.cases.items():
            expected = r["analytic"]["dfg_front"] * t_end
            self.assertAlmostEqual(
                r["front_position"], expected, delta=0.02,
                msg=f"t_D={t_end}: {r['front_position']} vs {expected}",
            )

    def test_average_saturation_behind_front_matches_welge(self):
        """
        前沿后平均饱和度必须对上 Welge 值（约 0.49）。

        这个量直接决定"注进去的 CO₂ 占多少孔隙"，也就决定盖层柱高。
        """
        for t_end, r in self.cases.items():
            expected = r["analytic"]["co2_saturation"]
            self.assertAlmostEqual(
                r["avg_behind_front"], expected, delta=0.02,
                msg=f"t_D={t_end}: {r['avg_behind_front']} vs {expected}",
            )

    def test_saturation_just_behind_front_matches_shock_value(self):
        """冲击前沿紧后方应等于 S_gf（数值扩散会让它略低）。"""
        for t_end, r in self.cases.items():
            expected = r["analytic"]["sg_front"]
            self.assertAlmostEqual(
                r["sg_behind_front"], expected, delta=0.03,
                msg=f"t_D={t_end}: {r['sg_behind_front']} vs {expected}",
            )

    def test_front_advances_linearly_in_time(self):
        """前沿位置对 t_D 是线性的 —— 非线性说明格式有问题。"""
        t1, t2 = 0.15, 0.3
        x1 = self.cases[t1]["front_position"]
        x2 = self.cases[t2]["front_position"]
        self.assertAlmostEqual(x2 / x1, t2 / t1, delta=0.05)


class ColumnHeightTests(unittest.TestCase):
    def test_column_scales_with_porosity_and_thickness(self):
        """
        等效柱高 = ΣS·dx·L·φ，与孔隙度、层厚成正比。

        这个量是"注了多少"与"盖层受多大压"之间唯一的桥梁，
        比例关系错一步，整个封闭性判定就错。
        """
        sg = [0.4] * 100
        h1 = flow1d.co2_column_height(sg, 0.01, 50, 0.2)
        self.assertAlmostEqual(h1, 0.4 * 1.0 * 50 * 0.2, places=6)
        h2 = flow1d.co2_column_height(sg, 0.01, 50, 0.4)
        self.assertAlmostEqual(h2 / h1, 2.0, places=6)

    def test_empty_profile_is_zero(self):
        self.assertEqual(flow1d.co2_column_height([], 0.01, 50, 0.2), 0.0)


class LimitationsTests(unittest.TestCase):
    def test_result_declares_what_is_not_modelled(self):
        """
        求解器必须自己说清楚没建模什么。特别是重力分异 ——
        少了它，垂直方向的柱高会被显著低估，而结果看起来完全正常。
        """
        r = flow1d.solve(n_cells=50, t_end=0.1)
        joined = " ".join(r["limitations"])
        self.assertIn("重力分异", joined)
        self.assertIn("毛细管", joined)

    def test_version_is_present(self):
        self.assertEqual(flow1d.solve(n_cells=50, t_end=0.1)["version"], flow1d.FLOW_VERSION)


class ParameterPlumbingTests(unittest.TestCase):
    def test_custom_params_change_the_answer(self):
        """换一组相渗参数，结果必须变 —— 否则参数根本没传进去。"""
        default = flow1d.solve(n_cells=200, t_end=0.3)
        tighter = flow1d.solve(n_cells=200, t_end=0.3, params=rp.RelPermParams(sgr=0.3))
        self.assertNotAlmostEqual(
            default["avg_behind_front"], tighter["avg_behind_front"], places=3
        )


if __name__ == "__main__":
    unittest.main()
