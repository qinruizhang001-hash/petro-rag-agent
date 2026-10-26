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

import pressure_model as pm
from core.optimize import (
    BIND_CAP,
    BIND_RATE_CAP,
    CAP_RATIO_DEFAULT,
    OPTIMIZE_N,
    STATUS_INFEASIBLE,
    STATUS_NO_INJECTION,
    STATUS_OK,
    critical_pressure_samples,
    injectivity_m3_per_day_per_bar,
    mass_mt_per_year,
    max_pressure_for_target,
    optimize_plan,
    pressure_bar_for_rate,
    rate_for_mass,
    rate_m3_per_day,
    risk_at_pressure,
)
from core.uq import wilson_interval
from tools.physics_tools import injection_risk_uq


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


if __name__ == "__main__":
    unittest.main()
