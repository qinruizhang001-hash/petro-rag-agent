"""
Sobol 敏感性分析的离线单元测试
==============================
零依赖、零 IO、零模型。

敏感性指数最容易"看起来对"：它永远是 0~1 之间的数，画成条形图永远好看，
但错 20 个百分点也不会抛异常。所以这里的测试全部用**解析真值**对标，
而不是"跑一遍看看结果合不合理"。

两个基准
--------
1. **可加模型** ``f = x1 + 2·x2``（x 为均匀分布，同宽度）::

       Var(f) = Var1 + 4·Var2 = 5·Var1
       S1 = 1/5 = 0.20,  S2 = 4/5 = 0.80
       S_T1 = S1,        S_T2 = S2        ← 无交互，总效应等于一阶

   这条同时验证两件事：一阶指数算对了，且"无交互"时总效应不会凭空变大。

2. **乘积模型** ``f = x1·x2``（x ~ U(0,1)）::

       Var(f) = 7/144
       V1 = Var(x1/2) = 1/48  →  S1 = 3/7 ≈ 0.4286
       S_T1 = 1 − Var(x2/2)/Var(f) = 4/7 ≈ 0.5714

   S_T > S 的部分就是交互项（此处 1/7）。这条验证"交互能被检出" ——
   如果实现漏掉 AB_i 矩阵（把 f(AB_i) 误写成 f(A)），交互会恒为 0，
   而可加模型那条测试**照样通过**。两条一起才能锁住。

运行::

    python -m unittest -v tests.test_sensitivity
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.sensitivity import (
    DEFAULT_BASE,
    render_sobol,
    sobol_event,
    sobol_indices,
)


def _uniform_dims(names, rel_sigma=0.05):
    return [{"name": n, "kind": "uniform", "rel_sigma": rel_sigma} for n in names]


class TestAdditiveBenchmark(unittest.TestCase):
    """f = x1 + 2·x2：S = (0.20, 0.80)，且无交互。"""

    def setUp(self):
        self.result = sobol_indices(
            lambda v: v[0] + 2.0 * v[1],
            _uniform_dims(["a", "b"]),
            [100.0, 100.0],
            n_base=8192,
            seed=1,
        )

    def test_first_order_matches_analytic(self):
        self.assertAlmostEqual(self.result["first_order"]["a"], 0.20, delta=0.03)
        self.assertAlmostEqual(self.result["first_order"]["b"], 0.80, delta=0.03)

    def test_total_order_equals_first_order_when_no_interaction(self):
        for name in ("a", "b"):
            self.assertAlmostEqual(
                self.result["total_order"][name],
                self.result["first_order"][name],
                delta=0.03,
                msg=name,
            )

    def test_interaction_is_near_zero(self):
        for name in ("a", "b"):
            self.assertLess(abs(self.result["interaction"][name]), 0.03, name)

    def test_ranking_puts_the_dominant_parameter_first(self):
        self.assertEqual(self.result["ranking"][0], "b")

    def test_first_order_sums_to_one(self):
        self.assertAlmostEqual(sum(self.result["first_order"].values()), 1.0, delta=0.04)

    def test_evaluation_count_matches_the_formula(self):
        # 代价 = n_base·(d + 2)：d 个 AB_i 矩阵，加 A、B
        self.assertEqual(self.result["evaluations"], 8192 * (2 + 2))

    def test_standard_error_is_reported_and_small(self):
        for name in ("a", "b"):
            se = self.result["total_order_se"][name]
            self.assertGreater(se, 0.0)
            self.assertLess(se, 0.02, name)


class TestProductBenchmark(unittest.TestCase):
    """f = x1·x2（x~U(0,1)）：S = 3/7，S_T = 4/7，交互 = 1/7。"""

    def setUp(self):
        self.result = sobol_indices(
            lambda v: v[0] * v[1],
            _uniform_dims(["a", "b"], rel_sigma=1.0),
            [0.5, 0.5],
            n_base=8192,
            seed=2,
        )

    def test_first_order_matches_analytic(self):
        for name in ("a", "b"):
            self.assertAlmostEqual(self.result["first_order"][name],
                                   3.0 / 7.0, delta=0.05, msg=name)

    def test_total_order_matches_analytic(self):
        for name in ("a", "b"):
            self.assertAlmostEqual(self.result["total_order"][name],
                                   4.0 / 7.0, delta=0.05, msg=name)

    def test_interaction_is_detected(self):
        """交互必须被检出 —— 漏掉 AB_i 矩阵的实现会在这里给出 0。"""
        for name in ("a", "b"):
            self.assertGreater(self.result["interaction"][name], 0.05, name)

    def test_total_order_exceeds_first_order(self):
        for name in ("a", "b"):
            self.assertGreater(self.result["total_order"][name],
                               self.result["first_order"][name])


class TestCorrelatedInputsAreRejected(unittest.TestCase):
    """
    Sobol 指数在相关输入下没有定义，必须**报错**而不是给一个能用的数。

    一个在相关输入上"算出来了"的指数比没有更危险：它会被当成结论引用，
    而方差分解的前提（固定 X_i 时其余参数的分布不变）已经不成立了。
    """

    def test_non_identity_correlation_raises(self):
        with self.assertRaises(ValueError) as ctx:
            sobol_indices(lambda v: v[0] + v[1], _uniform_dims(["a", "b"]),
                          [100.0, 100.0], n_base=64,
                          corr=[[1.0, 0.5], [0.5, 1.0]])
        message = str(ctx.exception)
        self.assertIn("独立", message)
        self.assertIn("Shapley", message)      # 要指出替代方案，不能只说"不行"

    def test_identity_correlation_is_accepted(self):
        result = sobol_indices(lambda v: v[0] + v[1], _uniform_dims(["a", "b"]),
                               [100.0, 100.0], n_base=64,
                               corr=[[1.0, 0.0], [0.0, 1.0]])
        self.assertTrue(result["independent"])


class TestDegenerateCases(unittest.TestCase):
    def test_constant_model_has_no_defined_indices(self):
        result = sobol_indices(lambda v: 42.0, _uniform_dims(["a", "b"]),
                               [1.0, 1.0], n_base=64, seed=1)
        self.assertEqual(result["variance"], 0.0)
        self.assertEqual(result["first_order"], {"a": 0.0, "b": 0.0})
        self.assertTrue(any("方差为零" in n for n in result["notes"]))
        self.assertIn("无定义", render_sobol(result))

    def test_single_parameter(self):
        # d=1 时 n_base 就是有效样本量，256 太小（实测偏差 6 个百分点）
        result = sobol_indices(lambda v: v[0] * 2, _uniform_dims(["only"]),
                               [10.0], n_base=4096, seed=1)
        # 单参数时该参数解释全部方差
        self.assertAlmostEqual(result["first_order"]["only"], 1.0, delta=0.02)
        self.assertAlmostEqual(result["total_order"]["only"], 1.0, delta=0.02)

    def test_empty_dims_raise(self):
        with self.assertRaises(ValueError):
            sobol_indices(lambda v: 1.0, [], [], n_base=16)

    def test_mismatched_nominals_raise(self):
        with self.assertRaises(ValueError):
            sobol_indices(lambda v: 1.0, _uniform_dims(["a", "b"]), [1.0], n_base=16)

    def test_negative_first_order_is_clamped_but_raw_is_kept(self):
        """截断到 0 必须可审计：first_order_raw 保留原值。"""
        result = sobol_indices(lambda v: v[0] * 0.0 + 1.0, _uniform_dims(["a"]),
                               [1.0], n_base=32, seed=1)
        for name in result["names"]:
            self.assertGreaterEqual(result["first_order"][name], 0.0)
            self.assertIn(name, result["first_order_raw"])


class TestEventSobol(unittest.TestCase):
    """对分类型模型（ok / risky / under）的某个事件做敏感性。"""

    @staticmethod
    def _verdict(values):
        # 与 pressure_model.verdict_of 同构的阶跃函数：压力越过阈值即 risky
        return "risky" if values[1] > 1.5 * values[0] else "ok"

    def test_event_probability_matches_its_frequency(self):
        result = sobol_event(self._verdict, _uniform_dims(["depth", "pressure"]),
                             [100.0, 140.0], target="risky", n_base=1024, seed=3)
        self.assertAlmostEqual(result["event_probability"],
                               result["event_hits"] / result["evaluations"], places=6)

    def test_both_parameters_share_the_variance(self):
        """判定是"压力 > 1.5×深度"这种比值形式，两个参数都该有贡献。"""
        result = sobol_event(self._verdict, _uniform_dims(["depth", "pressure"]),
                             [100.0, 150.0], target="risky", n_base=4096, seed=4)
        for name in ("depth", "pressure"):
            self.assertGreater(result["total_order"][name], 0.2, name)

    def test_text_names_the_event_and_the_frequency(self):
        result = sobol_event(self._verdict, _uniform_dims(["depth", "pressure"]),
                             [100.0, 150.0], target="risky", n_base=512, seed=5)
        self.assertIn("risky", result["text"])
        self.assertIn("频率", result["text"])

    def test_rare_event_gets_a_resolution_warning(self):
        # 阈值远高于可能范围 → 事件几乎不发生，指数分辨率不足
        result = sobol_event(self._verdict, _uniform_dims(["depth", "pressure"]),
                             [100.0, 160.0], target="risky", n_base=256, seed=6)
        if result["event_hits"] < 10:
            self.assertTrue(any("分辨率" in n for n in result["notes"]))


class TestRendering(unittest.TestCase):
    def setUp(self):
        self.result = sobol_indices(lambda v: v[0] + 2.0 * v[1],
                                    _uniform_dims(["深度", "注入压力"]),
                                    [100.0, 100.0], n_base=1024, seed=1)

    def test_text_separates_first_order_from_total(self):
        text = self.result["text"]
        self.assertIn("一阶", text)
        self.assertIn("总效应", text)

    def test_text_mentions_the_evaluation_count(self):
        self.assertIn(str(self.result["evaluations"]), self.result["text"])

    def test_text_names_the_dominant_parameter(self):
        self.assertIn("主导参数", self.result["text"])
        self.assertIn(self.result["ranking"][0], self.result["text"])

    def test_empty_result_renders_empty(self):
        self.assertEqual(render_sobol({}), "")

    def test_render_is_pure_function_of_the_result(self):
        self.assertEqual(render_sobol(self.result), self.result["text"])


class TestReproducibility(unittest.TestCase):
    def test_same_seed_gives_same_indices(self):
        a = sobol_indices(lambda v: v[0] * v[1], _uniform_dims(["a", "b"]),
                          [1.0, 1.0], n_base=256, seed=99)
        b = sobol_indices(lambda v: v[0] * v[1], _uniform_dims(["a", "b"]),
                          [1.0, 1.0], n_base=256, seed=99)
        self.assertEqual(a["first_order"], b["first_order"])
        self.assertEqual(a["total_order"], b["total_order"])

    def test_default_base_is_used(self):
        result = sobol_indices(lambda v: v[0] + v[1], _uniform_dims(["a", "b"]),
                               [1.0, 1.0], seed=1)
        self.assertEqual(result["n_base"], DEFAULT_BASE)


class TestPriorsAreHonoured(unittest.TestCase):
    """换先验必须换结果：如果分布被忽略，敏感性分析就只是"扫参数"。"""

    def test_lognormal_prior_changes_the_indices(self):
        model = lambda v: v[0] * v[1]
        uniform = sobol_indices(model, _uniform_dims(["a", "b"], rel_sigma=0.3),
                                [1.0, 1.0], n_base=2048, seed=7)
        lognormal = sobol_indices(
            model,
            [{"name": "a", "kind": "lognormal", "rel_sigma": 0.3},
             {"name": "b", "kind": "uniform", "rel_sigma": 0.3}],
            [1.0, 1.0], n_base=2048, seed=7,
        )
        self.assertNotAlmostEqual(lognormal["total_order"]["a"],
                                  uniform["total_order"]["a"], places=3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
