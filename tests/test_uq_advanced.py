"""
不确定性量化（v1.7：先验分布 / 参数相关性 / 联合抽样）的离线单元测试
====================================================================
零依赖、零 IO、零模型，可在 CI 里秒级跑完。

这里守的几条底线：

1. **每个先验的方差必须对得上解析值。** 均匀 U(−δ,δ) 的标准差是 δ/√3，
   三角是 δ/√6，Beta(2,2) 在 [−δ,δ] 上是 δ/√5，对数正态是
   √(exp(δ²)−1)。这些是硬指标：如果哪天有人把 ``rel_sigma`` 的语义从
   "半宽"改成"标准差"，方差会立刻偏离，而报告里的"±5%"文案不会变 ——
   这种不一致不会报错，只会让区间口径失真。
2. **copula 只改联合结构，不改边缘。** 加了相关系数之后，每个参数自己的
   标准差必须还是 δ/√3。否则"加了相关性"和"改了分布"两个变化混在一起，
   报告里区间变宽到底是谁造成的就说不清了。
3. **潜变量相关 ≠ 边缘 Pearson 相关。** 对均匀边缘有解析关系
   ρ_s = (6/π)·arcsin(ρ/2)（ρ=0.8 → 0.786）。用这个关系钉住实现：
   如果哪天有人把 z 直接当 ε 用（省掉 Φ 与 F⁻¹ 两步），这个断言会立刻失败。
4. **分布名字必须进文案。** 同样写"±5%"，均匀口径的 90% 区间约 ±4.5%，
   正态口径约 ±8.2%。文案里不写分布，读者无从判断区间口径。

运行::

    python -m unittest -v tests.test_uq_advanced
"""
from __future__ import annotations

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.uq import (
    MARGINAL_KINDS,
    JointSampler,
    Marginal,
    cholesky,
    format_interval,
    inverse_normal_cdf,
    make_marginal,
    monte_carlo,
    monte_carlo_joint,
    normal_cdf,
    validate_correlation,
    wilson_interval,
)


def _stats(values):
    """(均值, 标准差)，总体口径 —— 用于和解析方差比较。"""
    n = len(values)
    mean = sum(values) / n
    return mean, math.sqrt(sum((v - mean) ** 2 for v in values) / n)


def _quantile_stats(marginal: Marginal, n: int = 20000):
    """用分位点近似分布（中点法则），避免依赖随机数。"""
    values = [marginal.from_unit((i + 0.5) / n) for i in range(n)]
    return _stats(values)


class TestInverseNormalCdf(unittest.TestCase):
    """逆正态 CDF 是 copula 的核心：它错了，相关结构就是错的，且不会报错。"""

    def test_known_quantiles(self):
        # 标准正态的经典分位点（Abramowitz & Stegun 26.2.23 一带的标准值）
        for p, expected in ((0.025, -1.959964), (0.5, 0.0),
                            (0.975, 1.959964), (1e-6, -4.753424)):
            self.assertAlmostEqual(inverse_normal_cdf(p), expected, places=5)

    def test_symmetry(self):
        for p in (0.01, 0.1, 0.3, 0.45):
            self.assertAlmostEqual(
                inverse_normal_cdf(p), -inverse_normal_cdf(1 - p), places=9
            )

    def test_out_of_range_returns_infinity(self):
        # 不夹取：无界分布的分位数在端点就该发散，调用方自己负责 u 在开区间内
        self.assertEqual(inverse_normal_cdf(0.0), -math.inf)
        self.assertEqual(inverse_normal_cdf(1.0), math.inf)

    def test_roundtrip_with_cdf(self):
        for z in (-3.0, -1.0, 0.0, 0.7, 2.5):
            self.assertAlmostEqual(inverse_normal_cdf(normal_cdf(z)), z, places=9)


class TestMarginalVariance(unittest.TestCase):
    """每个先验的方差必须对上解析值（见模块文档第 1 条）。"""

    def test_uniform_sd_is_delta_over_sqrt3(self):
        mean, sd = _quantile_stats(Marginal("uniform", 0.05))
        self.assertAlmostEqual(mean, 0.0, places=3)
        self.assertAlmostEqual(sd, 0.05 / math.sqrt(3.0), places=3)

    def test_normal_sd_is_delta(self):
        mean, sd = _quantile_stats(Marginal("normal", 0.05), n=60000)
        self.assertAlmostEqual(mean, 0.0, places=3)
        self.assertAlmostEqual(sd, 0.05, places=3)

    def test_lognormal_mean_and_sd(self):
        # ε = exp(δz) − 1：均值 exp(δ²/2) − 1，标准差 √(exp(δ²)−1)
        d = 0.10
        mean, sd = _quantile_stats(Marginal("lognormal", d), n=60000)
        self.assertAlmostEqual(mean, math.exp(d * d / 2) - 1, places=4)
        # 分位点中点法对右偏分布有 O(1/n) 偏差，容差放到 1e-3
        self.assertAlmostEqual(sd, math.sqrt(math.exp(d * d) - 1), delta=1e-3)

    def test_triangular_sd_is_delta_over_sqrt6(self):
        mean, sd = _quantile_stats(Marginal("triangular", 0.05))
        self.assertAlmostEqual(mean, 0.0, places=3)
        self.assertAlmostEqual(sd, 0.05 / math.sqrt(6.0), places=3)

    def test_beta_symmetric_sd(self):
        # Beta(2,2) 缩放到 [−δ, δ]：方差 = δ²/5
        mean, sd = _quantile_stats(Marginal("beta", 0.05, alpha=2, beta=2))
        self.assertAlmostEqual(mean, 0.0, places=3)
        self.assertAlmostEqual(sd, 0.05 / math.sqrt(5.0), places=3)

    def test_all_kinds_are_covered(self):
        # 新增分布忘了写测试时，这条会提醒（而不是静默漏掉）
        self.assertEqual(
            set(MARGINAL_KINDS),
            {"uniform", "normal", "lognormal", "triangular", "beta"},
        )


class TestMarginalProperties(unittest.TestCase):
    def test_lognormal_is_positive_and_median_unbiased(self):
        m = Marginal("lognormal", 0.20)
        self.assertEqual(m.from_unit(0.5), 0.0)          # 中位数不偏移
        self.assertGreater(m.from_unit(1e-9), -1.0)      # ε > −1 → x > 0
        self.assertGreater(m.from_unit(1 - 1e-9), 0.5)

    def test_bounded_support(self):
        for kind in ("uniform", "triangular", "beta"):
            lo, hi = Marginal(kind, 0.05).support()
            self.assertAlmostEqual(lo, -0.05)
            self.assertAlmostEqual(hi, 0.05)

    def test_triangular_mode_is_the_nominal_by_default(self):
        m = Marginal("triangular", 0.05)
        self.assertAlmostEqual(m.from_unit(0.5), 0.0, places=6)

    def test_unknown_kind_is_rejected(self):
        with self.assertRaises(ValueError):
            Marginal("cauchy", 0.05)

    def test_beta_needs_positive_shapes(self):
        with self.assertRaises(ValueError):
            Marginal("beta", 0.05, alpha=0.0)

    def test_draw_and_from_unit_agree(self):
        """两条抽样路径（原生 rng / 分位函数）必须是同一个分布。"""
        import random

        for kind in MARGINAL_KINDS:
            m = Marginal(kind, 0.08)
            rng = random.Random(11)
            drawn = [m.draw(rng) for _ in range(20000)]
            quant = [m.from_unit((i + 0.5) / 20000) for i in range(20000)]
            dm, ds = _stats(drawn)
            qm, qs = _stats(quant)
            self.assertAlmostEqual(dm, qm, delta=0.004, msg=kind)
            self.assertAlmostEqual(ds, qs, delta=0.004, msg=kind)

    def test_uniform_draw_keeps_legacy_stream(self):
        """uniform 分支必须仍是一次 rng.uniform 调用（历史种子不漂移）。"""
        import random

        a = random.Random(7)
        b = random.Random(7)
        self.assertEqual(
            [Marginal("uniform", 0.05).draw(a) for _ in range(5)],
            [b.uniform(-0.05, 0.05) for _ in range(5)],
        )

    def test_make_marginal_accepts_str_and_dict(self):
        self.assertEqual(make_marginal("normal").kind, "normal")
        m = make_marginal({"kind": "lognormal", "rel_sigma": 0.3})
        self.assertEqual((m.kind, m.rel_sigma), ("lognormal", 0.3))
        self.assertIsInstance(make_marginal(None), Marginal)
        with self.assertRaises(TypeError):
            make_marginal(3.14)


class TestPerturbationLabel(unittest.TestCase):
    """文案必须带分布名 —— 这是 v1.7 补上的口径缺口。"""

    def test_each_kind_names_its_distribution(self):
        self.assertIn("均匀", Marginal("uniform", 0.05).perturbation_label())
        self.assertIn("正态", Marginal("normal", 0.05).perturbation_label())
        self.assertIn("对数正态", Marginal("lognormal", 0.05).perturbation_label())
        self.assertIn("三角", Marginal("triangular", 0.05).perturbation_label())
        self.assertIn("Beta", Marginal("beta", 0.05).perturbation_label())

    def test_interval_text_states_the_distribution(self):
        uniform = format_interval(
            monte_carlo(lambda x: x, 100.0, rel_sigma=0.05, seed=1, dist="uniform"),
            unit="bar",
        )
        normal = format_interval(
            monte_carlo(lambda x: x, 100.0, rel_sigma=0.05, seed=1, dist="normal"),
            unit="bar",
        )
        self.assertIn("均匀", uniform)
        self.assertIn("正态", normal)
        # 两种口径的区间宽度差一倍左右，文案必须能让读者看出区别
        self.assertNotEqual(uniform, normal)

    def test_normal_interval_is_wider_than_uniform(self):
        u = monte_carlo(lambda x: x, 100.0, rel_sigma=0.05, n=4000, seed=3, dist="uniform")
        n = monte_carlo(lambda x: x, 100.0, rel_sigma=0.05, n=4000, seed=3, dist="normal")
        self.assertGreater(n["p95"] - n["p5"], (u["p95"] - u["p5"]) * 1.5)


class TestCorrelationValidation(unittest.TestCase):
    """相关矩阵写错是"不报错但结论全错"的输入，必须当场拒绝。"""

    def test_asymmetric_matrix_is_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            validate_correlation([[1.0, 0.5], [0.3, 1.0]], 2)
        self.assertIn("不对称", str(ctx.exception))

    def test_diagonal_must_be_one(self):
        with self.assertRaises(ValueError):
            validate_correlation([[0.9, 0.5], [0.5, 1.0]], 2)

    def test_out_of_range_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_correlation([[1.0, 1.5], [1.5, 1.0]], 2)

    def test_wrong_shape_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_correlation([[1.0, 0.5, 0.0], [0.5, 1.0, 0.0]], 2)

    def test_non_positive_definite_is_rejected(self):
        """
        a 与 c 完全相关、且两者各自与 b 相关 0.9 → 行列式为 0，不合法。

        注意这条**不能**用"三参数两两相关 0.9"来测：等相关矩阵的特征值是
        1+2ρ=2.8 与 1−ρ=0.1，是正定的。高相关不等于非法，只有真去分解才知道。
        """
        bad = [[1.0, 0.9, 1.0], [0.9, 1.0, 0.9], [1.0, 0.9, 1.0]]
        with self.assertRaises(ValueError) as ctx:
            validate_correlation(bad, 3)
        self.assertIn("正定", str(ctx.exception))

    def test_high_but_legal_correlation_is_accepted(self):
        # 反向断言：两两 0.9 的相关矩阵必须能过（否则上面那条测试就成了摆设）
        ok = [[1.0, 0.9, 0.9], [0.9, 1.0, 0.9], [0.9, 0.9, 1.0]]
        self.assertEqual(validate_correlation(ok, 3), ok)

    def test_cholesky_recovers_the_matrix(self):
        corr = [[1.0, 0.6], [0.6, 1.0]]
        L = cholesky(corr)
        for i in range(2):
            for j in range(2):
                self.assertAlmostEqual(
                    sum(L[i][k] * L[j][k] for k in range(2)), corr[i][j], places=12
                )


class TestGaussianCopula(unittest.TestCase):
    def _dims(self, kind="uniform", rel=0.05):
        return [{"name": "a", "kind": kind, "rel_sigma": rel},
                {"name": "b", "kind": kind, "rel_sigma": rel}]

    def test_latent_rho_maps_to_rank_rho_for_uniform_marginals(self):
        """
        均匀边缘下 ρ_edge = (6/π)·arcsin(ρ_latent/2)。

        这条断言同时钉住三件事：copula 确实在起作用（ρ 不是原样传过去的）、
        Φ 与 F⁻¹ 两步都在（少一步这个等式就不成立）、以及实现方向没搞反。
        """
        sampler = JointSampler(self._dims(), corr=[[1.0, 0.8], [0.8, 1.0]], seed=1)
        empirical = sampler.empirical_correlation(n=20000, seed=2)[0][1]
        theory = 6.0 / math.pi * math.asin(0.8 / 2.0)      # 0.7859
        self.assertAlmostEqual(empirical, theory, delta=0.02)

    def test_marginals_survive_the_copula(self):
        """加了相关性之后，每个参数自己的标准差必须还是 δ/√3。"""
        sampler = JointSampler(self._dims(), corr=[[1.0, 0.9], [0.9, 1.0]], seed=3)
        rows = [sampler.draw() for _ in range(20000)]
        for col in range(2):
            _, sd = _stats([r[col] for r in rows])
            self.assertAlmostEqual(sd, 0.05 / math.sqrt(3.0), delta=0.002)

    def test_zero_offdiagonal_is_independent(self):
        sampler = JointSampler(self._dims(), corr=None, seed=5)
        self.assertTrue(sampler.independent)
        empirical = sampler.empirical_correlation(n=20000, seed=6)[0][1]
        self.assertLess(abs(empirical), 0.03)

    def test_negative_correlation_is_honoured(self):
        sampler = JointSampler(self._dims(), corr=[[1.0, -0.7], [-0.7, 1.0]], seed=7)
        self.assertLess(sampler.empirical_correlation(n=20000, seed=8)[0][1], -0.5)

    def test_draw_values_applies_nominals(self):
        sampler = JointSampler(self._dims(), corr=None, seed=9)
        for _ in range(50):
            a, b = sampler.draw_values([100.0, 50.0])
            self.assertGreater(a, 100.0 * 0.94)
            self.assertLess(a, 100.0 * 1.06)
            self.assertGreater(b, 50.0 * 0.94)

    def test_lognormal_marginal_in_copula_stays_positive(self):
        sampler = JointSampler(self._dims("lognormal", 0.3),
                               corr=[[1.0, 0.5], [0.5, 1.0]], seed=10)
        for _ in range(500):
            a, b = sampler.draw_values([100.0, 100.0])
            self.assertGreater(a, 0.0)
            self.assertGreater(b, 0.0)


class TestMonteCarloJoint(unittest.TestCase):
    def _dims(self):
        return [{"name": "a", "kind": "uniform", "rel_sigma": 0.05},
                {"name": "b", "kind": "uniform", "rel_sigma": 0.05}]

    def test_linear_model_mean_matches_nominal(self):
        r = monte_carlo_joint(lambda v: 2 * v[0] + v[1], self._dims(),
                              [100.0, 50.0], n=2000, seed=1)
        self.assertAlmostEqual(r["mean"], 250.0, delta=1.0)
        self.assertLess(r["p5"], r["p95"])

    def test_result_carries_the_correlation_assumption(self):
        r = monte_carlo_joint(lambda v: v[0] + v[1], self._dims(), [100.0, 100.0],
                              n=500, seed=1, corr=[[1.0, 0.6], [0.6, 1.0]])
        self.assertFalse(r["independent"])
        self.assertAlmostEqual(r["corr"][0][1], 0.6)
        self.assertEqual(r["names"], ["a", "b"])
        self.assertEqual(len(r["priors"]), 2)

    def test_independent_flag_when_no_corr(self):
        r = monte_carlo_joint(lambda v: v[0], self._dims(), [1.0, 1.0], n=100, seed=1)
        self.assertTrue(r["independent"])
        self.assertEqual(r["corr"], [[1.0, 0.0], [0.0, 1.0]])

    def test_mismatched_lengths_are_rejected(self):
        with self.assertRaises(ValueError):
            monte_carlo_joint(lambda v: v[0], self._dims(), [1.0], n=10, seed=1)

    def test_correlation_reduces_variance_of_a_difference(self):
        """a−b 在正相关下方差更小 —— 这是"相关性能改变结论"的最小证明。"""
        ind = monte_carlo_joint(lambda v: v[0] - v[1], self._dims(), [100.0, 100.0],
                                n=4000, seed=2)
        cor = monte_carlo_joint(lambda v: v[0] - v[1], self._dims(), [100.0, 100.0],
                                n=4000, seed=2, corr=[[1.0, 0.9], [0.9, 1.0]])
        self.assertLess(cor["std"], ind["std"] * 0.6)


class TestWilsonInterval(unittest.TestCase):
    def test_known_value(self):
        # k=5, n=100 的 95% Wilson 区间：0.066648 ± 0.045100
        lo, hi = wilson_interval(5, 100)
        self.assertAlmostEqual(lo, 0.021548, places=5)
        self.assertAlmostEqual(hi, 0.111748, places=5)

    def test_zero_hits_has_positive_upper_bound(self):
        lo, hi = wilson_interval(0, 100)
        # 浮点残差约 1e-18，不是精确的 0
        self.assertAlmostEqual(lo, 0.0, places=12)
        self.assertGreater(hi, 0.0)
        self.assertLess(hi, 0.05)

    def test_all_hits_has_lower_bound_below_one(self):
        lo, hi = wilson_interval(100, 100)
        self.assertEqual(hi, 1.0)
        self.assertLess(lo, 1.0)

    def test_interval_narrows_with_more_samples(self):
        narrow = wilson_interval(500, 10000)
        wide = wilson_interval(5, 100)
        self.assertLess(narrow[1] - narrow[0], (wide[1] - wide[0]) / 5)

    def test_degenerate_n(self):
        self.assertEqual(wilson_interval(0, 0), (0.0, 1.0))


class TestSummarizeCarriesDistribution(unittest.TestCase):
    def test_summarize_adds_dist_label(self):
        from core.uq import summarize

        out = summarize(monte_carlo(lambda x: x, 100.0, seed=1, dist="lognormal"),
                        unit="bar")
        self.assertEqual(out["dist"], "lognormal")
        self.assertIn("对数正态", out["dist_label"])
        self.assertIn("对数正态", out["text"])

    def test_default_dist_fallback_is_uniform_not_normal(self):
        """
        历史遗留：summarize 里 ``summary.get("dist", "normal")`` 的兜底值写的是
        normal，而实际默认是 uniform。兜底路径一旦被走到，文案与实现就差了一层。
        """
        from core.uq import summarize

        out = summarize({"mean": 1.0, "std": 0.1, "p5": 0.9, "p50": 1.0,
                         "p95": 1.1, "min": 0.8, "max": 1.2, "cv": 0.1,
                         "n": 100, "rel_sigma": 0.05}, unit="bar")
        self.assertEqual(out["dist"], "uniform")


if __name__ == "__main__":
    unittest.main(verbosity=2)
