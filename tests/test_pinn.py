"""
PINN 模型的行为测试（需要 torch + 已训练权重）
==============================================
验证的是"模型学对了没有"，而不是"代码跑起来了"。三个层次：

1. **解析解对账** —— injection 场景在 t ≫ τ 后趋于线性稳态
   P(x) = 120 − 20·x/L。PINN 是数值解，没有解析解就没有判据，
   所以这个函数本身就是测试的一部分。
2. **物理残差** —— 模型在随机配点上必须真的满足 PDE，而不是只在训练点上。
3. **结构约束** —— 这一条最容易被误认为是风格问题：激活函数必须是 tanh。
   ReLU 的二阶导数几乎处处为 0，``∂²P/∂x²`` 恒等于 0 会让 physics loss
   **凭空变成 0**。也就是说"残差测试"抓不到这个 bug —— 换成 ReLU 之后
   残差反而更小。所以这里直接断言模块里没有 ReLU。

依赖说明：需要 ``models/pinn_*.pt``（先跑 ``python pinn_model.py``）与 torch，
所以不进 CI 的 unit-core job（那个 job 一个三方包都不装）。
"""
from __future__ import annotations

import unittest
from pathlib import Path

try:
    import torch

    from pinn_model import (
        L,
        P0,
        P_WELL,
        PINN,
        SCENARIOS,
        S_MAX,
        _load,
        physics_residual,
        predict_pressure,
        steady_state_pressure,
    )

    HAVE_TORCH = True
    IMPORT_ERR = ""
except Exception as exc:  # noqa: BLE001
    HAVE_TORCH = False
    IMPORT_ERR = f"{type(exc).__name__}: {exc}"

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
HAVE_WEIGHTS = all((MODEL_DIR / f"pinn_{s}.pt").exists() for s in ("baseline", "injection"))

requires_model = unittest.skipUnless(
    HAVE_TORCH and HAVE_WEIGHTS,
    f"需要 torch 与已训练权重（{IMPORT_ERR or '先运行 python pinn_model.py'}）",
)


@requires_model
class TestArchitecture(unittest.TestCase):
    def test_activation_is_tanh_not_relu(self):
        """结构约束：tanh 是 PDE 残差能反传的前提，不是风格偏好。

        换 ReLU 后 ``∂²P/∂x²`` 恒为 0，physics loss 变成 0，残差测试会
        "通过" —— 所以必须在这里挡住。
        """
        net = PINN()
        kinds = {type(m).__name__ for m in net.modules()}
        self.assertIn("Tanh", kinds)
        self.assertNotIn("ReLU", kinds, "ReLU 会让二阶导恒为 0，physics loss 失效")

    def test_three_linear_layers(self):
        """需求要求 3 层 MLP：2 → 64 → 64 → 1。"""
        net = PINN()
        linears = [m for m in net.modules() if isinstance(m, torch.nn.Linear)]
        self.assertEqual(len(linears), 3)
        self.assertEqual((linears[0].in_features, linears[-1].out_features), (2, 1))


@requires_model
class TestBaselineScenario(unittest.TestCase):
    """需求原文的边界条件：初值 100，两端恒定 100。"""

    def test_solution_is_constant_100(self):
        """由最大值原理，唯一解是 P ≡ 100 —— 模型必须学到这个常数场。

        这条测试同时是一个提醒：baseline 场景**不可能**给出有信息量的答案。
        任何位置、任何时间都返回 100，这是对的，也是没用的。
        """
        for x in (0.0, 25.0, 50.0, 75.0, 100.0):
            for t in (0.0, 1.0, 10.0, 30.0):
                with self.subTest(x=x, t=t):
                    self.assertAlmostEqual(predict_pressure(x, t, "baseline"), 100.0, delta=0.5)


@requires_model
class TestInjectionScenario(unittest.TestCase):
    """井口 120 bar、远端 100 bar —— 有非平凡解，且可对账。"""

    def test_matches_analytic_steady_state(self):
        """t = 10 天 ≈ 8.6τ，已充分接近稳态，可与线性解析解逐点比较。"""
        for x in (0.0, 25.0, 50.0, 75.0, 100.0):
            with self.subTest(x=x):
                self.assertAlmostEqual(
                    predict_pressure(x, 10.0), steady_state_pressure(x), delta=1.5
                )

    def test_profile_decays_from_well_to_far_field(self):
        """压力必须随距离单调下降：井口最高、远端最低。"""
        ps = [predict_pressure(x, 10.0) for x in (0, 20, 40, 60, 80, 100)]
        self.assertEqual(ps, sorted(ps, reverse=True), f"剖面不单调：{ps}")
        self.assertGreater(ps[0], ps[-1] + 10.0, "井口与远端几乎无压差，模型没学到扩散")

    def test_bounded_by_boundary_values(self):
        """域内压力应落在两端边界值之间（扩散方程的最大值原理）。"""
        for x in (10.0, 30.0, 50.0, 70.0, 90.0):
            p = predict_pressure(x, 10.0)
            with self.subTest(x=x):
                self.assertGreaterEqual(p, P0 - 1.0)
                self.assertLessEqual(p, P_WELL + 1.0)

    def test_initial_condition(self):
        """t = 0 时应回到初值 100 bar（井口角点不连续，允许更大误差）。"""
        self.assertAlmostEqual(predict_pressure(50.0, 0.0), P0, delta=2.0)


@requires_model
class TestPhysicsResidual(unittest.TestCase):
    def test_residual_is_small_off_training_points(self):
        """在**随机**配点上满足 PDE，而不是只记住训练点。

        实测：baseline 均值 0.031、injection 0.138（信号是 20 bar 的压差）。
        阈值 0.5 留了约 3 倍余量，既能容忍重训的抖动，也能挡住"模型塌成
        常数却恰好满足初值"以外的明显失败。
        """
        for scenario, limit in (("baseline", 0.5), ("injection", 0.5)):
            with self.subTest(scenario=scenario):
                torch.manual_seed(0)
                net = _load(scenario)
                x = torch.rand(2000, 1)
                s = torch.rand(2000, 1) * S_MAX
                mean_abs = float(physics_residual(net, x, s).abs().mean().detach())
                self.assertLess(mean_abs, limit, f"{scenario} 平均残差 {mean_abs:.3f} 过大")


@requires_model
class TestApiContract(unittest.TestCase):
    def test_returns_float_and_is_deterministic(self):
        p1 = predict_pressure(50, 10)
        p2 = predict_pressure(50, 10)
        self.assertIsInstance(p1, float)
        self.assertEqual(p1, p2, "同一输入两次调用结果不同，说明没走 eval() 或有权重随机")

    def test_out_of_range_inputs_are_clamped(self):
        """越界裁剪：语言模型偶尔给出 x=150，此时给最接近的可行答案。"""
        self.assertEqual(predict_pressure(150, 10), predict_pressure(L, 10))
        self.assertEqual(predict_pressure(-5, 10), predict_pressure(0, 10))
        self.assertEqual(predict_pressure(50, 400), predict_pressure(50, 30))

    def test_unknown_scenario_raises(self):
        with self.assertRaises(ValueError):
            predict_pressure(50, 10, scenario="nonsense")

    def test_non_finite_input_raises(self):
        with self.assertRaises(ValueError):
            predict_pressure(float("nan"), 10)

    def test_all_scenarios_loadable(self):
        for sc in SCENARIOS:
            with self.subTest(scenario=sc):
                self.assertTrue((MODEL_DIR / f"pinn_{sc}.pt").exists())


@requires_model
class TestSteadyStateReference(unittest.TestCase):
    def test_endpoints(self):
        self.assertAlmostEqual(steady_state_pressure(0), P_WELL, places=6)
        self.assertAlmostEqual(steady_state_pressure(L), P0, places=6)

    def test_linearity(self):
        self.assertAlmostEqual(steady_state_pressure(50), (P_WELL + P0) / 2, places=6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
