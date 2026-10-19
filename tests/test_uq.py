"""
不确定性量化（Monte Carlo）的离线单元测试
==========================================
零依赖、零 IO、零模型，可在 CI 里秒级跑完。

这里守的几条底线：

1. **可复现。** 同一种子必须给出同一批样本，否则"区间是 1.29–1.41"
   这种断言只能靠眼睛看。
2. **口径不能漂。** 默认 ±5% 是**最大扰动幅度**（均匀分布），
   输出的 90% 区间应当落在 ±4.5% 左右。若哪天有人把默认改成 1σ 正态，
   区间会突然变成 ±8.2%，而报告里的文案还写着"±5%" —— 这种不一致
   不会报错，只会让结论看起来比实际更不确定。所以用断言钉住。
3. **物理工具的输出必须是"均值 + 区间"**，而不是一个点值。
   MCP 的旧调用方靠 ``float(text.split()[0])`` 取数，这条也要保住。

运行::

    python -m unittest -v tests.test_uq
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.uq import (
    DEFAULT_N,
    format_interval,
    format_value,
    monte_carlo,
    monte_carlo_outcomes,
    percentile,
    summarize,
    uq_block,
)


class TestPercentile(unittest.TestCase):
    def test_empty_returns_none(self):
        self.assertIsNone(percentile([], 50))

    def test_single_value(self):
        self.assertEqual(percentile([3.0], 50), 3.0)

    def test_median_of_odd_length(self):
        self.assertEqual(percentile([1, 2, 3], 50), 2.0)

    def test_median_interpolates_for_even_length(self):
        self.assertAlmostEqual(percentile([1, 2, 3, 4], 50), 2.5)

    def test_extremes(self):
        self.assertEqual(percentile([1, 2, 3], 0), 1.0)
        self.assertEqual(percentile([1, 2, 3], 100), 3.0)

    def test_p5_is_below_p95(self):
        values = list(range(101))
        self.assertLess(percentile(values, 5), percentile(values, 95))


class TestMonteCarlo(unittest.TestCase):
    def test_seed_makes_it_reproducible(self):
        a = monte_carlo(lambda x: x * 2, 100.0, seed=123, n=50)
        b = monte_carlo(lambda x: x * 2, 100.0, seed=123, n=50)
        self.assertEqual(a["mean"], b["mean"])
        self.assertEqual(a["p5"], b["p5"])

    def test_different_seeds_give_different_samples(self):
        a = monte_carlo(lambda x: x, 100.0, seed=1, n=50)
        b = monte_carlo(lambda x: x, 100.0, seed=2, n=50)
        self.assertNotEqual(a["mean"], b["mean"])

    def test_mean_approximates_nominal_for_linear_model(self):
        result = monte_carlo(lambda x: x, 250.0, rel_sigma=0.05, n=1000, seed=7)
        self.assertAlmostEqual(result["mean"], 250.0, delta=1.0)

    def test_default_ci_is_about_nine_percent_wide(self):
        """±5% 均匀扰动 → 90% 区间约 ±4.5%（线性模型）。"""
        result = monte_carlo(lambda x: x, 100.0, rel_sigma=0.05, n=2000, seed=11)
        self.assertGreater(result["p5"], 94.0)
        self.assertLess(result["p5"], 96.0)
        self.assertGreater(result["p95"], 104.0)
        self.assertLess(result["p95"], 106.0)

    def test_default_n_is_100(self):
        self.assertEqual(monte_carlo(lambda x: x, 1.0, seed=1)["n"], DEFAULT_N)

    def test_lower_bound_clips_unphysical_values(self):
        result = monte_carlo(lambda x: x, 1.0, rel_sigma=0.9, n=200, seed=3,
                             lower_bound=0.0)
        self.assertGreaterEqual(result["min"], 0.0)

    def test_samples_only_when_requested(self):
        self.assertNotIn("samples", monte_carlo(lambda x: x, 1.0, seed=1))
        self.assertEqual(
            len(monte_carlo(lambda x: x, 1.0, seed=1, n=100, keep_samples=True)["samples"]),
            100,
        )

    def test_normal_distribution_is_wider_than_uniform(self):
        """同样标 ±5%，正态口径的 90% 区间明显更宽 —— 这正是口径必须写清楚的原因。"""
        u = monte_carlo(lambda x: x, 100.0, rel_sigma=0.05, n=3000, seed=5, dist="uniform")
        n = monte_carlo(lambda x: x, 100.0, rel_sigma=0.05, n=3000, seed=5, dist="normal")
        self.assertGreater(n["p95"] - n["p5"], (u["p95"] - u["p5"]) * 1.4)


class TestMonteCarloOutcomes(unittest.TestCase):
    def test_probabilities_sum_to_one(self):
        out = monte_carlo_outcomes(lambda x: "ok" if x > 100 else "low",
                                   nominal=100.0, rel_sigma=0.05, n=100, seed=2)
        self.assertAlmostEqual(sum(out["probabilities"].values()), 1.0, places=6)
        self.assertEqual(out["n"], 100)

    def test_deterministic_category_is_certain(self):
        out = monte_carlo_outcomes(lambda x: "always", nominal=10.0, rel_sigma=0.01,
                                   n=100, seed=2)
        self.assertEqual(out["probabilities"], {"always": 1.0})


class TestFormatting(unittest.TestCase):
    def test_format_value_keeps_small_numbers_informative(self):
        self.assertEqual(format_value(0.785), "0.785")
        self.assertEqual(format_value(1.34), "1.34")
        self.assertEqual(format_value(250.0), "250.0")

    def test_interval_mentions_all_qualifiers(self):
        result = monte_carlo(lambda x: x, 250.0, seed=9, n=100)
        text = format_interval(result, unit="bar")
        self.assertIn("90% 置信区间", text)
        self.assertIn("100 次蒙特卡洛模拟", text)
        self.assertIn("±5%", text)

    def test_interval_of_empty_summary_is_empty(self):
        self.assertEqual(format_interval({}), "")


class TestSummarize(unittest.TestCase):
    def test_rounds_and_adds_text(self):
        raw = monte_carlo(lambda x: x * 0.01, 100.0, seed=4, n=100)
        out = summarize(raw, unit="kg/m³")
        for key in ("mean", "p5", "p95", "std", "cv", "n", "text", "unit"):
            self.assertIn(key, out)
        self.assertEqual(out["unit"], "kg/m³")
        self.assertIn("kg/m³", out["text"])
        self.assertLess(out["p5"], out["p95"])

    def test_samples_passed_through_when_requested(self):
        raw = monte_carlo(lambda x: x, 1.0, seed=4, n=100, keep_samples=True)
        out = summarize(raw, unit="", keep_samples=True)
        self.assertEqual(len(out["samples"]), 100)


class TestUqBlock(unittest.TestCase):
    def test_empty_summary_produces_no_block(self):
        self.assertEqual(uq_block({}), "")

    def test_block_contains_imperative_constraints(self):
        raw = summarize(monte_carlo(lambda x: x * 0.01, 250.0, seed=1, n=100),
                        unit="kg/m³")
        block = uq_block({"density": raw})
        self.assertIn("禁止只写单一确定值", block)
        self.assertIn("90% 置信区间", block)

    def test_injection_probabilities_are_rendered(self):
        block = uq_block({"injection": {
            "probabilities": {"ok": 0.8, "risky": 0.2},
            "n": 100, "rel_sigma": 0.05,
        }})
        self.assertIn("ok 80%", block)
        self.assertIn("risky 20%", block)


class TestPhysicsToolsUQ(unittest.TestCase):
    """物理工具的对外契约：均值 + 区间，且旧调用方仍能取到第一个数值。"""

    def setUp(self):
        from tools.physics_tools import (  # 延迟导入：避免 CI 无依赖时整文件跳过
            calculate_co2_density,
            co2_density_uq,
            injection_risk_uq,
        )
        self.calculate = calculate_co2_density
        self.density_uq = co2_density_uq
        self.injection_uq = injection_risk_uq

    def test_density_text_contains_interval(self):
        text = self.calculate(250, 80, seed=42)
        self.assertIn("90% 置信区间", text)
        self.assertIn("100 次蒙特卡洛", text)

    def test_density_first_token_is_a_float(self):
        # MCP Server 的旧解析方式，不能因为加了区间就失效
        text = self.calculate(250, 80, seed=42)
        self.assertIsInstance(float(text.split()[0]), float)

    def test_density_mean_matches_state_equation(self):
        out = self.density_uq(250, 80, seed=42)
        expected = 1.9 * 250 / (80 + 273.15)
        # 100 次抽样的均值本身有抽样误差（SE ≈ 0.003），不能要求到小数点后三位
        self.assertAlmostEqual(out["mean"], expected, delta=0.01)

    def test_density_ci_brackets_the_mean(self):
        out = self.density_uq(250, 80, seed=42)
        self.assertLess(out["p5"], out["mean"])
        self.assertLess(out["mean"], out["p95"])

    def test_density_is_reproducible(self):
        a = self.density_uq(250, 80, seed=7)
        b = self.density_uq(250, 80, seed=7)
        self.assertEqual((a["mean"], a["p5"], a["p95"]),
                         (b["mean"], b["p5"], b["p95"]))

    def test_injection_probabilities_sum_to_one(self):
        out = self.injection_uq(2500, 300, seed=42)
        self.assertAlmostEqual(sum(out["probabilities"].values()), 1.0, places=6)

    def test_injection_well_above_fracture_is_almost_certainly_risky(self):
        # 2500 m 的破裂压力约 379 bar，600 bar 远超，任何抽样都该判 risky
        out = self.injection_uq(2500, 600, seed=42)
        self.assertEqual(out["probabilities"].get("risky"), 1.0)

    def test_injection_text_mentions_simulation_count(self):
        out = self.injection_uq(2500, 300, seed=42)
        self.assertIn("100 次蒙特卡洛", out["text"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
