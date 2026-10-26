"""
注入策略优化器的契约测试
========================
这个模块的每个输出都要能**回代核对**，所以测试也围绕回代展开：

1. ``q = n·J·(p − P_h)`` 必须两边相等 —— 否则"推荐速率"只是一个数；
2. 推荐方案的风险上界必须 ≤ 预算 —— 否则"合规"只是措辞；
3. 优化器的风险必须与 ``injection_risk_uq`` **逐样本相等**（同 seed）——
   否则两个模块对"压裂概率"的定义已经分叉，而报告会把它们并列引用；
4. 无解时必须返回无解 —— 这是本模块存在的意义之一，不能靠"看起来合理"
   的数字掩盖。

第 3 条是最重要的一条：它把"优化器偷偷换了判据"这种错误从"要读代码才
发现"变成"跑一次就红"。
"""
from __future__ import annotations

import math
import unittest
from unittest.mock import patch

import pressure_model as pm
from core.optimize import (
    BIND_CAP,
    BIND_RATE_CAP,
    CAP_RATIO_DEFAULT,
    DEFAULT_DRAINAGE_RADIUS_M,
    DEFAULT_WELLBORE_RADIUS_M,
    OPTIMIZE_N,
    STATUS_INFEASIBLE,
    STATUS_NO_INJECTION,
    STATUS_OK,
    critical_pressure_samples,
    injectivity_m3_per_day_per_bar,
    interference_factor,
    mass_mt_per_year,
    max_pressure_for_target,
    optimize_plan,
    pressure_bar_for_rate,
    rate_for_mass,
    rate_m3_per_day,
    risk_at_pressure,
)
from core.uq import wilson_interval
from tests import _graph_stubs

# 先声明"本模块要的是真实实现"，再 import。
#
# 为什么不能直接 ``from tools.physics_tools import injection_risk_uq``：
# ``tests/test_quality_gate.py`` / ``tests/test_negative_gate.py`` 在**模块顶层**
# 调用 ``install_leaf_stubs()``（桩必须早于 ``import workflow``），而
# ``unittest discover`` 是"先导入全部测试模块，再跑"。于是替身的存活期覆盖了
# 所有测试模块的导入期：只要本模块排在它们之后被加载，拿到的是
# ``_fake_injection_uq`` —— 它返回的 dict 没有 ``counts``，第 3 条契约
# （优化器与 UQ 逐样本相等）就退化成 4 个 ``KeyError: 'counts'``。
#
# 实测过：``python -m unittest tests.test_negative_gate tests.test_optimize``
# 必红；全量 discover 不复现，因为本模块恰好先被加载。那是碰巧，不是设计。
#
# ``release()`` 在没有装过桩时是空操作（返回 False），所以在 CI 的
# unit-core 里（只跑本模块，桩从未生效）行为完全不变。
_graph_stubs.release("tools.physics_tools")

from tools.physics_tools import injection_risk_uq  # noqa: E402


class TestInjectivity(unittest.TestCase):
    """注入指数：复用 relperm 的实现，只做单位换算。"""

    def test_matches_relperm_index_divided_by_ten(self):
        from utils.relperm import injectivity_index

        j = injectivity_m3_per_day_per_bar(100.0, 50.0, 60.0, 0.0)
        ref = injectivity_index(100.0, 50.0, 0.06)["J_m3_per_day_per_mpa"]
        self.assertAlmostEqual(j, ref / 10.0, places=9)

    def test_doubling_k_doubles_j(self):
        a = injectivity_m3_per_day_per_bar(100.0, 50.0, 60.0)
        b = injectivity_m3_per_day_per_bar(200.0, 50.0, 60.0)
        self.assertAlmostEqual(b / a, 2.0, places=9)

    def test_doubling_thickness_doubles_j(self):
        a = injectivity_m3_per_day_per_bar(100.0, 50.0, 60.0)
        b = injectivity_m3_per_day_per_bar(100.0, 100.0, 60.0)
        self.assertAlmostEqual(b / a, 2.0, places=9)

    def test_skin_reduces_injectivity_by_the_log_ratio(self):
        ln = math.log(500.0 / 0.1)
        a = injectivity_m3_per_day_per_bar(100.0, 50.0, 60.0, skin=0.0)
        b = injectivity_m3_per_day_per_bar(100.0, 50.0, 60.0, skin=5.0)
        self.assertAlmostEqual(b / a, ln / (ln + 5.0), places=9)

    def test_negative_skin_is_rejected(self):
        # 负表皮意味着增产措施，本模型不表达；静默接受会凭空放大注入能力
        with self.assertRaises(ValueError):
            injectivity_m3_per_day_per_bar(100.0, 50.0, 60.0, skin=-1.0)


class TestRatePressure(unittest.TestCase):
    """速率与压力的换算必须互为逆运算，否则推荐值无法回代。"""

    def test_round_trip(self):
        d, j = 2500.0, injectivity_m3_per_day_per_bar()
        p = 320.0
        q = rate_m3_per_day(p, d, j, 2)
        self.assertAlmostEqual(pressure_bar_for_rate(q, d, j, 2), p, places=9)

    def test_rate_scales_with_wells(self):
        d, j = 2500.0, injectivity_m3_per_day_per_bar()
        one = rate_m3_per_day(320.0, d, j, 1)
        four = rate_m3_per_day(320.0, d, j, 4)
        self.assertAlmostEqual(four / one, 4.0, places=9)

    def test_pressure_below_hydrostatic_gives_zero_rate(self):
        d = 2500.0
        p_h = float(pm.hydrostatic_bar(d))
        j = injectivity_m3_per_day_per_bar()
        self.assertEqual(rate_m3_per_day(p_h - 10.0, d, j, 1), 0.0)
        self.assertEqual(rate_m3_per_day(p_h, d, j, 1), 0.0)

    def test_mass_round_trip(self):
        q = 25000.0
        rho = 720.0
        mt = mass_mt_per_year(q, rho)
        self.assertAlmostEqual(rate_for_mass(mt, rho), q, places=6)


class TestCriticalSamples(unittest.TestCase):
    """``P_s`` 的支撑区间可以由解析式算出，用它钉住抽样范围。"""

    def test_sorted_and_bounded(self):
        d = 2500.0
        p_h = float(pm.hydrostatic_bar(d))
        critical, sampler = critical_pressure_samples(d, rel_sigma=0.05, n=500, seed=1)
        self.assertEqual(len(critical), 500)
        self.assertEqual(critical, sorted(critical))
        self.assertTrue(sampler.independent)
        # P_s = 1.5·P_h·f/(1+ε)，f,ε ∈ [0.95, 1.05]
        lo = 1.5 * p_h * 0.95 / 1.05
        hi = 1.5 * p_h * 1.05 / 0.95
        self.assertGreaterEqual(critical[0], lo - 1e-9)
        self.assertLessEqual(critical[-1], hi + 1e-9)

    def test_same_seed_same_samples(self):
        a, _ = critical_pressure_samples(2000.0, n=64, seed=11)
        b, _ = critical_pressure_samples(2000.0, n=64, seed=11)
        self.assertEqual(a, b)

    def test_correlated_sampling_changes_the_distribution(self):
        # 相关抽样不是"多一个参数"，它会改变联合分布，因此必须改变结果
        ind, s1 = critical_pressure_samples(2500.0, n=200, seed=3)
        cor, s2 = critical_pressure_samples(
            2500.0, n=200, seed=3, corr=[[1.0, 0.9], [0.9, 1.0]]
        )
        self.assertTrue(s1.independent)
        self.assertFalse(s2.independent)
        self.assertNotEqual(ind, cor)


class TestRiskAtPressure(unittest.TestCase):
    def test_hits_are_monotone_and_exact(self):
        critical, _ = critical_pressure_samples(2500.0, n=200, seed=5)
        prev = -1
        for p in (200, 250, 300, 320, 340, 360, 380, 420):
            hits = risk_at_pressure(float(p), critical)["hits"]
            self.assertGreaterEqual(hits, prev)
            prev = hits
        self.assertEqual(risk_at_pressure(0.0, critical)["hits"], 0)
        self.assertEqual(risk_at_pressure(1e6, critical)["hits"], 200)

    def test_interval_matches_wilson(self):
        critical, _ = critical_pressure_samples(2500.0, n=400, seed=5)
        p = critical[40]      # 恰好 40 个严格小于
        risk = risk_at_pressure(p, critical)
        self.assertEqual(risk["hits"], 40)
        self.assertEqual(risk["p_ci"], list(wilson_interval(40, 400)))

    def test_empty_samples_are_rejected(self):
        with self.assertRaises(ValueError):
            risk_at_pressure(300.0, [])


class TestMaxPressure(unittest.TestCase):
    def test_constraint_is_tight(self):
        """k_max 必须"刚好用满预算"：再多一个命中就超。"""
        critical, _ = critical_pressure_samples(2500.0, n=1000, seed=9)
        limit = max_pressure_for_target(critical, 0.05, 1e9, 0.0)
        n = len(critical)
        self.assertLessEqual(wilson_interval(limit["k_max"], n)[1], 0.05)
        if limit["k_max"] < n:
            self.assertGreater(wilson_interval(limit["k_max"] + 1, n)[1], 0.05)

    def test_hits_at_solution_equal_kmax(self):
        critical, _ = critical_pressure_samples(2500.0, n=1000, seed=9)
        limit = max_pressure_for_target(critical, 0.05, 1e9, 0.0)
        self.assertEqual(limit["hits"], limit["k_max"])
        self.assertEqual(limit["pressure_bar"], critical[limit["k_max"]])

    def test_cap_binds_when_it_is_lower(self):
        critical, _ = critical_pressure_samples(2500.0, n=1000, seed=9)
        limit = max_pressure_for_target(critical, 0.05, 300.0, 0.0)
        self.assertEqual(limit["binding"], BIND_CAP)
        self.assertEqual(limit["pressure_bar"], 300.0)
        self.assertLessEqual(limit["hits"], limit["k_max"])

    def test_wide_budget_falls_back_to_cap(self):
        critical, _ = critical_pressure_samples(2500.0, n=200, seed=9)
        limit = max_pressure_for_target(critical, 1.0, 400.0, 0.0)
        self.assertEqual(limit["k_max"], 200)
        self.assertEqual(limit["pressure_bar"], 400.0)
        self.assertEqual(limit["binding"], BIND_CAP)


class TestSolverAgreesWithBruteForce(unittest.TestCase):
    """次序统计量的推导，必须能被"直接穷举"独立复现。

    ``TestMaxPressure`` 检查的是"解处的 Wilson 上界刚好用满预算"——但那是
    **按同一个 k_max 在说话**：如果索引取错一位（比如用 ``critical[k+1]``），
    自洽性断言会跟着一起错，因为两边用的是同一个错值。这类错误不会让任何
    现有测试变红，只会让推荐压力系统性偏高或偏低一个样本。

    这里换成最笨的办法：把所有会让命中数发生变化的压力逐个代进
    ``risk_at_pressure``，取满足预算的最大者，再要求它等于求解器的答案。

    候选集只有 ``len(critical) + 2`` 个点，因为经验 CDF 只在样本值处跳变 ——
    在别处采样既发现不了新解，也证明不了旧解错。这也顺带说明为什么求解器
    不需要二分：可行域的边界就是样本值本身。
    """

    def _brute_force(self, critical, budget, cap_bar, floor_bar):
        """直接枚举，不使用 k_max / 次序统计量 / Wilson 单调性。"""
        candidates = {float(cap_bar)}
        candidates.update(float(v) for v in critical if float(v) <= float(cap_bar))
        best = None
        for p in sorted(candidates):
            if p <= float(floor_bar):
                continue
            if risk_at_pressure(p, critical)["p_upper"] <= float(budget):
                best = p
        return best

    def test_agrees_across_budgets_and_caps(self):
        for seed in (3, 9, 17):
            critical, _ = critical_pressure_samples(2500.0, n=400, seed=seed)
            for budget in (0.02, 0.05, 0.10, 0.25):
                for cap in (300.0, 360.0, 1e9):
                    with self.subTest(seed=seed, budget=budget, cap=cap):
                        solver = max_pressure_for_target(critical, budget, cap, 0.0)
                        brute = self._brute_force(critical, budget, cap, 0.0)
                        self.assertIsNotNone(
                            brute, "穷举认为无可行解，而求解器给了答案"
                        )
                        self.assertAlmostEqual(
                            solver["pressure_bar"], brute, places=9,
                            msg="求解器与穷举不一致：索引推导或 cap 处理有问题",
                        )

    def test_no_grid_point_above_the_answer_is_feasible(self):
        """反向：答案之上再高一点点必须立刻越界，否则"最大"是假的。"""
        critical, _ = critical_pressure_samples(2500.0, n=400, seed=9)
        solver = max_pressure_for_target(critical, 0.05, 1e9, 0.0)
        p_star = solver["pressure_bar"]
        if solver["binding"] == BIND_CAP:
            self.skipTest("cap 绑定：压力上限不是由风险决定的")
        above = [v for v in critical if v > p_star]
        self.assertTrue(above, "没有比解更高的样本，无法反向验证")
        self.assertGreater(
            risk_at_pressure(above[0], critical)["p_upper"], 0.05,
            "比解高一个样本的压力仍然满足预算 —— 说明解不是最大的",
        )


class TestWellInterference(unittest.TestCase):
    """井间干扰：默认不启用，启用后必须让"加井"不再是线性增益。

    这一组防的是两种相反的错误：

    * **默认行为被悄悄改掉** —— 不传井距时必须逐位等于旧公式。这个项目里
      所有对外数字都建立在 ``q = n·J·(p−P_h)`` 上，改默认值等于一次无声的
      结论变更。
    * **干扰项自己算错** —— 尤其是 ``d ≥ r_e`` 时 ``ln(r_e/d)`` 会变负，
      直接代进公式会得出"有邻井反而注得更多"。那不是精度问题，是符号错。
    """

    def test_absent_spacing_is_bit_identical_to_the_old_formula(self):
        """不传井距 = 无干扰 = 与引入本功能之前完全一致。"""
        j = injectivity_m3_per_day_per_bar()
        for n_w in (1, 2, 4, 8):
            with self.subTest(n_wells=n_w):
                new = rate_m3_per_day(350.0, 2500.0, j, n_w)
                old = n_w * j * (350.0 - pm.hydrostatic_bar(2500.0))
                self.assertAlmostEqual(new, old, places=12)

    def test_spacing_at_or_beyond_drainage_radius_is_not_interference(self):
        """井距 ≥ 泄油半径：两井各自圈地，稳态叠加不成立，f 必须是 0。"""
        for spacing in (DEFAULT_DRAINAGE_RADIUS_M, 800.0):
            with self.subTest(spacing=spacing):
                out = interference_factor(spacing)
                self.assertEqual(out["factor"], 0.0)
                self.assertFalse(out["active"])
                self.assertIn("独立泄油区", out["reason"])

    def test_factor_grows_as_wells_get_closer(self):
        """井距越小干扰越强 —— 单调性，且不依赖具体常数。"""
        factors = [
            interference_factor(s)["factor"] for s in (450.0, 300.0, 150.0, 20.0)
        ]
        self.assertEqual(factors, sorted(factors))
        self.assertGreater(factors[-1], factors[0])

    def test_factor_stays_below_one(self):
        """f=1 意味着"邻井离你和井壁一样近"，物理上不可能。"""
        for spacing in (0.2, 1.0, 10.0, 100.0, 499.0):
            with self.subTest(spacing=spacing):
                self.assertLess(interference_factor(spacing)["factor"], 1.0)

    def test_geometry_matches_the_injectivity_module(self):
        """干扰用的对数项必须与 ``relperm.injectivity_index`` 同源。

        两处各写一份 ``ln(re/rw)+S``，默认值一旦分叉，"注入能力"和"干扰代价"
        就会按不同的井距在说话，而且不报错。这里把两边钉死。
        """
        from utils.relperm import injectivity_index

        for re_m, rw_m, skin in ((500.0, 0.1, 0.0), (2000.0, 0.1, 3.0)):
            with self.subTest(re=re_m, skin=skin):
                expected = injectivity_index(
                    100.0, 50.0, 0.06, re_m=re_m, rw_m=rw_m, skin=skin
                )["ln_term"]
                got = interference_factor(
                    re_m / 2.0, re_m, rw_m, skin
                )["ln_own"]
                self.assertAlmostEqual(got, expected, places=12)

    def test_one_well_ignores_the_factor(self):
        """单井时 f 无从谈起：公式里 (n−1)=0，结果必须一模一样。"""
        j = injectivity_m3_per_day_per_bar()
        self.assertAlmostEqual(
            rate_m3_per_day(350.0, 2500.0, j, 1, 0.0),
            rate_m3_per_day(350.0, 2500.0, j, 1, 0.4),
            places=12,
        )

    def test_rate_and_pressure_round_trip(self):
        """有干扰时也必须能回代，否则推荐压力不可核对。"""
        j = injectivity_m3_per_day_per_bar()
        for n_w in (1, 2, 4):
            for f in (0.0, 0.08, 0.3):
                with self.subTest(n_wells=n_w, f=f):
                    q = rate_m3_per_day(360.0, 2500.0, j, n_w, f)
                    back = pressure_bar_for_rate(q, 2500.0, j, n_w, f)
                    self.assertAlmostEqual(back, 360.0, places=9)

    def test_well_count_saturates(self):
        """加井存在饱和：n→∞ 时 q → J·Δp/f，而不是无限增长。

        这是本功能唯一能给出的"加井不再有用"的结论。没有它，干扰就只是
        一个把数字调小一点的系数，不值得单独建模。
        """
        j = injectivity_m3_per_day_per_bar()
        f = 0.12
        drawdown = 360.0 - pm.hydrostatic_bar(2500.0)
        limit = j * drawdown / f

        q100 = rate_m3_per_day(360.0, 2500.0, j, 100, f)
        q1000 = rate_m3_per_day(360.0, 2500.0, j, 1000, f)
        self.assertLess(q100, limit)
        self.assertLess(q1000, limit)
        # 收敛到上限：1000 口井已经吃掉 99% 的可用速率
        self.assertGreater(q1000 / limit, 0.99)
        # 而线性模型在这个井数下会给出 8.3 倍的"上限"，正是要否掉的那个结论
        self.assertGreater(
            rate_m3_per_day(360.0, 2500.0, j, 1000, 0.0), 8.0 * limit
        )

    def test_invalid_geometry_is_rejected(self):
        with self.assertRaises(ValueError):
            interference_factor(0.05)          # 井距不大于井筒半径
        with self.assertRaises(ValueError):
            interference_factor(200.0, drainage_radius_m=0.05)
        for invalid in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    interference_factor(invalid)
                with self.assertRaises(ValueError):
                    rate_m3_per_day(350.0, 2500.0, 10.0, 4, invalid)
                with self.assertRaises(ValueError):
                    pressure_bar_for_rate(100.0, 2500.0, 10.0, 4, invalid)

    def test_non_geometric_four_well_scenario_is_identified(self):
        result = optimize_plan(2500.0, 380.0, n=200, seed=7, well_spacing_m=150.0)
        choices = {o["n_wells"]: o for o in result["well_options"]}
        self.assertEqual(choices[2]["interference_scope"], "pairwise_geometry")
        self.assertEqual(choices[4]["interference_scope"], "mean_field_not_a_well_layout")
        self.assertIn("平面中超过 3 口井不可能两两等距", " ".join(result["notes"]))


class TestInterferenceCoefficientTests(unittest.TestCase):
    """直接给定 f 的入口：井位没定时，评审问的是"假设 f=0.2 会怎样"。

    这条路径与"按井距反算"必须**可区分**：同样是 f=0.12，一个来自几何、
    一个是假设，在报告里的可信度完全不同。所以 source / scope / notes
    三处都要跟着变，只改数值不改标注等于把假设伪装成几何结果。
    """

    def test_zero_coefficient_matches_no_interference(self):
        base = optimize_plan(2500.0, 380.0, n=200, seed=7)
        zero = optimize_plan(
            2500.0, 380.0, n=200, seed=7, interference_coefficient=0.0
        )
        self.assertEqual(
            [o["rate_m3_per_day"] for o in base["well_options"]],
            [o["rate_m3_per_day"] for o in zero["well_options"]],
        )
        # f=0 时"用户给了系数"这件事不影响结论，scope 仍是独立井
        self.assertEqual(
            zero["recommended"]["interference_scope"], "independent_wells"
        )

    def test_rate_follows_the_formula(self):
        out = optimize_plan(
            2500.0, 380.0, n=200, seed=7, interference_coefficient=0.2
        )
        physics = out["physics"]
        self.assertAlmostEqual(physics["interference_factor"], 0.2, places=12)
        self.assertEqual(physics["interference_source"], "user_coefficient")
        n_w = 8
        q = rate_m3_per_day(
            360.0, 2500.0, physics["injectivity_m3_per_day_per_bar"], n_w, 0.2
        )
        manual = (
            n_w * physics["injectivity_m3_per_day_per_bar"]
            * (360.0 - physics["hydrostatic_bar"]) / (1 + (n_w - 1) * 0.2)
        )
        self.assertAlmostEqual(q, manual, places=9)

    def test_scope_says_assumed_not_geometric(self):
        out = optimize_plan(
            2500.0, 380.0, n=200, seed=7, interference_coefficient=0.2
        )
        self.assertEqual(
            out["recommended"]["interference_scope"], "assumed_coefficient"
        )
        for option in out["well_options"]:
            with self.subTest(n_wells=option["n_wells"]):
                self.assertEqual(option["interference_scope"], "assumed_coefficient")

    def test_note_calls_it_an_assumption(self):
        out = optimize_plan(
            2500.0, 380.0, n=200, seed=7, interference_coefficient=0.2
        )
        text = " ".join(out["notes"])
        self.assertIn("用户给定假设系数", text)
        # 不能写成"已按井距计算" —— 这句话在没有井距时是假的
        self.assertNotIn("等效井距", text)

    def test_spacing_and_coefficient_cannot_coexist(self):
        with self.assertRaises(ValueError) as ctx:
            optimize_plan(
                2500.0, 380.0, well_spacing_m=300.0, interference_coefficient=0.2
            )
        self.assertIn("不能同时给", str(ctx.exception))

    def test_out_of_range_coefficient_is_rejected(self):
        for bad in (1.0, 1.5, -0.1, float("nan"), float("inf")):
            with self.subTest(value=bad):
                with self.assertRaises(ValueError):
                    optimize_plan(2500.0, 380.0, interference_coefficient=bad)


class TestInterferenceInThePlan(unittest.TestCase):
    """把干扰接进 ``optimize_plan``：同样的井数，能力必须下降。"""

    def test_spacing_reduces_capacity_at_the_same_pressure(self):
        base = optimize_plan(2500.0, 380.0, n=300, seed=7)
        with_gap = optimize_plan(
            2500.0, 380.0, n=300, seed=7, well_spacing_m=150.0
        )
        base_opt = {o["n_wells"]: o for o in base["well_options"]}
        gap_opt = {o["n_wells"]: o for o in with_gap["well_options"]}
        for n_w in (2, 4, 8):
            with self.subTest(n_wells=n_w):
                self.assertLess(
                    gap_opt[n_w]["rate_m3_per_day"],
                    base_opt[n_w]["rate_m3_per_day"],
                )
        # 单井不受影响：干扰是井与井之间的事
        self.assertAlmostEqual(
            gap_opt[1]["rate_m3_per_day"], base_opt[1]["rate_m3_per_day"], places=6
        )

    def test_physics_block_carries_the_factor(self):
        """f 必须出现在 ``physics`` 里，否则回代核对的式子就断了。"""
        out = optimize_plan(2500.0, 380.0, n=300, seed=7, well_spacing_m=150.0)
        physics = out["physics"]
        self.assertTrue(physics["interference_active"])
        self.assertGreater(physics["interference_factor"], 0.0)
        self.assertEqual(physics["well_spacing_m"], 150.0)
        # 用返回的数复算，必须与返回的速率一致
        q = rate_m3_per_day(
            360.0, 2500.0,
            physics["injectivity_m3_per_day_per_bar"],
            4, physics["interference_factor"],
        )
        n_wells = 4
        f = physics["interference_factor"]
        manual = (
            n_wells * physics["injectivity_m3_per_day_per_bar"]
            * (360.0 - physics["hydrostatic_bar"]) / (1 + (n_wells - 1) * f)
        )
        self.assertAlmostEqual(q, manual, places=9)

    def test_notes_say_whether_interference_was_counted(self):
        """不写清楚，读者无法判断这个速率是乐观还是已打折。"""
        plain = " ".join(optimize_plan(2500.0, 380.0, n=200, seed=7)["notes"])
        self.assertIn("未计井间干扰", plain)

        counted = " ".join(
            optimize_plan(2500.0, 380.0, n=200, seed=7, well_spacing_m=150.0)["notes"]
        )
        self.assertIn("井间干扰已计入", counted)
        self.assertIn("饱和", counted)


class TestDensityCannotBeInvented(unittest.TestCase):
    def test_property_failure_does_not_become_a_plausible_number(self):
        with patch(
            "utils.co2_properties.co2_density_kg_m3", side_effect=ValueError("EOS failed")
        ):
            with self.assertRaisesRegex(ValueError, "拒绝用猜测密度"):
                optimize_plan(2500.0, 380.0, n=100, seed=7)

    def test_nonfinite_density_is_rejected(self):
        with patch("utils.co2_properties.co2_density_kg_m3", return_value=float("nan")):
            with self.assertRaisesRegex(ValueError, "非有限正密度"):
                optimize_plan(2500.0, 380.0, n=100, seed=7)


class TestSameRiskSemantics(unittest.TestCase):
    """
    优化器与 ``injection_risk_uq`` 的判据必须逐样本一致。

    这条不是"锦上添花的交叉验证"：报告里会同时出现"注入风险概率分布"
    与"优化后风险"，两者若用了不同判据，读者会看到两个都"有出处"却互相
    矛盾的数，而没有任何东西会报错。
    """

    def test_hits_match_for_several_cases(self):
        cases = [
            (2000.0, 250.0, 100),
            (2500.0, 320.0, 100),
            (2500.0, 380.0, 100),
            (3000.0, 400.0, 200),
        ]
        for depth, pressure, n in cases:
            with self.subTest(depth=depth, pressure=pressure, n=n):
                uq = injection_risk_uq(
                    depth, pressure, n=n, seed=7, with_decision=False
                )
                plan = optimize_plan(depth, pressure, n=n, seed=7)
                self.assertEqual(
                    plan["current"]["risk"]["hits"],
                    uq["counts"].get("risky", 0),
                )


class TestOptimizePlan(unittest.TestCase):
    def test_safe_plan_keeps_one_well(self):
        r = optimize_plan(2000.0, 250.0, seed=42)
        self.assertEqual(r["status"], STATUS_OK)
        self.assertEqual(r["recommended"]["n_wells"], 1)
        # 当前方案已在预算内 → 推荐就是当前压力，速率持平
        self.assertAlmostEqual(r["recommended"]["pressure_bar"], 250.0, places=6)
        self.assertGreater(r["headroom"]["fraction"], 0.0)

    def test_unsafe_plan_adds_wells_at_the_same_rate(self):
        r = optimize_plan(2500.0, 380.0, seed=42)
        self.assertEqual(r["status"], STATUS_OK)
        self.assertGreater(r["current"]["risk"]["p"], r["p_target"])
        self.assertEqual(r["recommended"]["n_wells"], 2)
        # 保住速率：推荐速率与当前速率相等
        self.assertAlmostEqual(
            r["recommended"]["rate_mt_per_year"],
            r["current"]["rate_mt_per_year"],
            places=6,
        )
        self.assertLess(
            r["recommended"]["pressure_bar"], r["current"]["pressure_bar"]
        )

    def test_rate_is_recoverable_from_pressure(self):
        """q = n·J·(p − P_h) 必须两边相等 —— 这是唯一能回代核对的式子。"""
        r = optimize_plan(2500.0, 380.0, seed=42)
        rec = r["recommended"]
        j = r["physics"]["injectivity_m3_per_day_per_bar"]
        p_h = r["physics"]["hydrostatic_bar"]
        expect = rate_m3_per_day(rec["pressure_bar"], 2500.0, j, rec["n_wells"])
        self.assertAlmostEqual(rec["rate_m3_per_day"], expect, places=3)
        self.assertAlmostEqual(
            rec["rate_m3_per_day"], rec["n_wells"] * j * (rec["pressure_bar"] - p_h),
            places=3,
        )

    def test_recommended_upper_bound_within_budget(self):
        for depth, pressure in ((2000.0, 250.0), (2500.0, 380.0), (3000.0, 420.0)):
            with self.subTest(depth=depth, pressure=pressure):
                r = optimize_plan(depth, pressure, seed=42)
                self.assertLessEqual(
                    r["recommended"]["risk"]["p_upper"], r["p_target"] + 1e-12
                )

    def test_out_of_sample_check_uses_another_seed(self):
        r = optimize_plan(2500.0, 380.0, seed=42)
        self.assertIn("verification", r)
        self.assertNotEqual(r["verification"]["seed"], 42)
        self.assertEqual(r["verification"]["n"], OPTIMIZE_N)

    def test_storage_target_too_high_is_infeasible(self):
        r = optimize_plan(2500.0, 380.0, storage_rate_mt_per_year=500.0, seed=42)
        self.assertEqual(r["status"], STATUS_INFEASIBLE)
        self.assertGreater(r["headroom"]["shortfall_mt_per_year"], 0.0)
        self.assertIn("能力受限", r["text"])

    def test_storage_target_met_with_fewer_wells(self):
        r = optimize_plan(2500.0, 380.0, storage_rate_mt_per_year=5.0, seed=42)
        self.assertEqual(r["status"], STATUS_OK)
        self.assertEqual(r["recommended"]["n_wells"], 1)
        self.assertAlmostEqual(r["recommended"]["rate_mt_per_year"], 5.0, places=6)

    def test_pressure_below_hydrostatic_reports_no_injection(self):
        r = optimize_plan(2500.0, 250.0, seed=42)
        self.assertEqual(r["status"], STATUS_OK)
        self.assertEqual(r["current"]["rate_m3_per_day"], 0.0)
        self.assertIn("注不进", r["text"])

    def test_cap_ratio_can_make_it_impossible(self):
        # cap_ratio=0.5 → 上限 0.75·P_h < P_h，任何压力都注不进
        r = optimize_plan(2500.0, 320.0, cap_ratio=0.5, seed=42)
        self.assertEqual(r["status"], STATUS_NO_INJECTION)
        self.assertIsNone(r["recommended"])

    def test_cap_ratio_binds_before_risk(self):
        r = optimize_plan(2500.0, 320.0, cap_ratio=0.9, seed=42)
        self.assertEqual(r["binding"], BIND_CAP)
        self.assertAlmostEqual(
            r["limit"]["pressure_bar"],
            0.9 * r["physics"]["fracture_bar"],
            places=6,
        )

    def test_rate_cap_makes_target_unreachable_and_keeps_pressure_consistent(self):
        r = optimize_plan(2500.0, 320.0, rate_cap_mt_per_year=3.0, seed=42)
        self.assertEqual(r["status"], STATUS_INFEASIBLE)
        self.assertEqual(r["recommended"]["binding"], BIND_RATE_CAP)
        rec = r["recommended"]
        self.assertAlmostEqual(rec["rate_mt_per_year"], 3.0, places=6)
        # 压力必须由被截断的速率反推，不能抄 p_star
        j = r["physics"]["injectivity_m3_per_day_per_bar"]
        expect = rate_m3_per_day(rec["pressure_bar"], 2500.0, j, rec["n_wells"])
        self.assertAlmostEqual(rec["rate_m3_per_day"], expect, places=3)
        self.assertLess(rec["pressure_bar"], r["limit"]["pressure_bar"])

    def test_deterministic(self):
        a = optimize_plan(2500.0, 380.0, seed=13)
        b = optimize_plan(2500.0, 380.0, seed=13)
        self.assertEqual(a["text"], b["text"])
        self.assertEqual(a["recommended"], b["recommended"])

    def test_prior_changes_the_answer(self):
        # 换先验必须换结果：均匀的 ±5% 与正态 1σ=±5% 的 90% 区间差一倍宽
        u = optimize_plan(2500.0, 380.0, seed=42, dist="uniform")
        n = optimize_plan(2500.0, 380.0, seed=42, dist="normal")
        self.assertNotEqual(u["limit"]["pressure_bar"], n["limit"]["pressure_bar"])
        self.assertIn("正态", n["dist_label"])


class TestContractAndProvenance(unittest.TestCase):
    def test_provenance_names_every_constant(self):
        r = optimize_plan(2500.0, 380.0, seed=42)
        prov = r["provenance"]
        for key in (
            "fracture_ratio", "injectivity", "viscosity", "density",
            "p_target", "constraint", "skin",
        ):
            self.assertIn(key, prov)
            self.assertTrue(prov[key])
        self.assertIn("Wilson", prov["constraint"])

    def test_notes_state_the_optimistic_assumption(self):
        r = optimize_plan(2500.0, 380.0, seed=42)
        joined = " ".join(r["notes"])
        self.assertIn("skin=0", joined)
        self.assertIn("非达西", joined)
        self.assertIn("阶跃", joined)

    def test_frontier_covers_the_transition(self):
        r = optimize_plan(2500.0, 380.0, seed=42)
        probs = [p["p_leak"] for p in r["frontier"]]
        self.assertEqual(probs, sorted(probs))
        self.assertGreater(probs[-1], probs[0])
        for row in r["frontier"]:
            self.assertLessEqual(row["p_lo"], row["p_leak"] + 1e-12)
            self.assertGreaterEqual(row["p_hi"], row["p_leak"] - 1e-12)

    def test_text_contains_the_decision_numbers(self):
        r = optimize_plan(2500.0, 380.0, seed=42)
        self.assertIn("推荐 2 口井", r["text"])
        self.assertIn("超出风险预算", r["text"])
        self.assertIn("预算 5%", r["text"])

    def test_rejects_bad_inputs(self):
        with self.assertRaises(ValueError):
            optimize_plan(10.0, 100.0)
        with self.assertRaises(ValueError):
            optimize_plan(2500.0, 320.0, cap_ratio=0.0)
        with self.assertRaises(ValueError):
            optimize_plan(2500.0, 320.0, years=0.0)
        with self.assertRaises(ValueError):
            optimize_plan(2500.0, 320.0, rate_cap_mt_per_year=-1.0)
        with self.assertRaises(ValueError):
            optimize_plan(2500.0, 320.0, p_target=1.5)


class DocumentedNumbersTests(unittest.TestCase):
    """文档里那张干扰表必须由代码产出，而不是"曾经算过一次"。

    2026-10-26 的版本手写死在 `docs/gap-register.md` 里，所有格子统一偏 1.6 %：
    反推密度是 789.96 kg/m³，而 `_density(p_h, 60 ℃)` 给 777.20。也就是说
    那张表来自一个没有被记录下来的状态，密度口径一变就再也没人能复现它。
    偏 1.6 % 本身不致命，致命的是**没有任何东西会因此变红**。
    """

    def test_gap_register_table_is_generated_not_hand_written(self):
        import scripts.interference_table as table

        doc = (table.ROOT / "docs" / "gap-register.md").read_text(encoding="utf-8")
        self.assertIn(
            table.BEGIN, doc,
            "文档里的生成块标记被删掉了 —— 表格又变回手写数字",
        )
        start = doc.index(table.BEGIN)
        end = doc.index(table.END) + len(table.END)
        self.assertEqual(
            doc[start:end].strip(),
            table.render().strip(),
            "docs/gap-register.md 的干扰表与代码算出来的不一致。"
            "跑 python scripts/interference_table.py --write 重新生成；"
            "如果 diff 里是数字变化，那是物理改了，需要在同一提交里说明原因。",
        )


class TestAggregateContract(unittest.TestCase):
    """
    真实现的返回结构，必须与下游**真正读的键**一致。

    与 ``tests/test_graph_stubs.py`` 的分工要说清楚，因为两者容易混淆：

    ==================  ==========================  ==================
    文件                 锁什么                      能抓到什么
    ==================  ==========================  ==================
    test_graph_stubs    替身提供了哪些**名字**（静态）  整图 import 阶段崩
    test_optimize（本类） 真实现返回哪些**键**（运行期）   报告生成时 KeyError
    ==================  ==========================  ==================

    两者都不能省。替身可以完美对齐 aggregate 的 import 列表，而真实现少返回
    一个 ``physics.interference_factor`` 时：替身测试全绿、CI 全绿、单元测试
    全绿 —— 因为替身里有那个键。真实运行到"写干扰声明"那一行才崩，而那时
    报告已经生成了 90%。

    所以这里不测"结构大致对不对"，只测**下游会解引用到的键**，
    并且连格式串都按下游的原样跑一遍：``f"{x:g}"`` 遇到 None 会 TypeError，
    而"键存在但值是 None"正是最容易漏的一种。
    """

    # 三种干扰来源各测一遍：none / user_coefficient / spacing_geometry。
    # 只测一种等于只验证一条分支，而这三条的 physics 块内容不同。
    CASES = (
        ("none", {}),
        ("user_coefficient", {"interference_coefficient": 0.2}),
        ("spacing_geometry", {"well_spacing_m": 300.0}),
    )

    def _plan(self, **kwargs) -> dict:
        from tools.physics_tools import optimize_injection_strategy

        return optimize_injection_strategy(
            depth_m=2500.0, injection_pressure_bar=300.0, n=200, seed=7, **kwargs
        )

    def test_top_level_keys_aggregate_reads(self):
        plan = self._plan()
        for key in ("status", "text", "physics", "notes", "current",
                    "recommended", "well_options", "verification"):
            self.assertIn(key, plan, f"aggregate 会读 plan[{key!r}]")
        self.assertIsInstance(plan["text"], str)
        self.assertTrue(plan["text"].strip(), "text 为空时这条事实不会进 tool_facts")
        self.assertIsInstance(plan["physics"], dict)

    def test_interference_block_is_consistent_for_every_source(self):
        for source, kwargs in self.CASES:
            with self.subTest(source=source):
                physics = self._plan(**kwargs)["physics"]
                self.assertEqual(physics.get("interference_source"), source)
                self.assertIn("interference_active", physics)
                self.assertIn("interference_factor", physics)

                # 这一行照抄 nodes/aggregate_node.py 的格式串。那边改了这里要跟着改，
                # 否则测试会"通过"而生产环境 TypeError。
                if physics.get("interference_active"):
                    f_val = physics.get("interference_factor")
                    self.assertIsInstance(f_val, float,
                                          f"{source}: 有 active 却没有可格式化的 f")
                    f"{f_val:.4f}"
                    self.assertGreater(f_val, 0.0)
                    self.assertLess(f_val, 1.0)

                # 同样是照抄：spacing_geometry 分支会 f"{...:g}" 打印井距与泄油半径
                if physics.get("interference_source") == "spacing_geometry":
                    for key in ("well_spacing_m", "drainage_radius_m"):
                        val = physics.get(key)
                        self.assertIsInstance(val, float,
                                              f"spacing_geometry 分支会用 {key} 做 :g 格式化")
                        f"{val:g}"

    def test_user_coefficient_does_not_fake_a_well_spacing(self):
        """
        给了假设系数就不该出现井距 —— 否则报告里"按等效井距 300 m"是一句
        凭空捏造的几何描述，而它读起来比"假设系数 0.2"可信得多。
        """
        physics = self._plan(interference_coefficient=0.2)["physics"]
        self.assertIsNone(physics.get("well_spacing_m"))
        self.assertAlmostEqual(physics["interference_factor"], 0.2, places=9)


if __name__ == "__main__":
    unittest.main()
