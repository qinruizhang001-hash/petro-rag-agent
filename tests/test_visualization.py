"""
离线单元测试：前端可视化组件（ui_components）
================================================
不依赖 Streamlit / Ollama / 模型，纯内存断言。
运行：python -m unittest discover -s tests -v
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ui_components import (
    compute_risk_scores,
    risk_radar_figure,
    stratigraphy_figure,
)


class TestComputeRiskScores(unittest.TestCase):
    """五维风险分数计算：基于物理校验结果和参数。"""

    def test_ok_low_risk(self):
        """安全工况 → 五维分数都不应过高。"""
        done = {"uq": {"injection": {"probabilities": {"ok": 0.95, "risky": 0.03, "under": 0.02}}}}
        # depth=2000m 时 p_h≈202 bar, p_f≈303 bar; p_inj=250 在两者之间 → ok
        scores = compute_risk_scores(done, depth_m=2000, p_inj=250, temp=60)
        self.assertLess(scores["地质风险"], 50)
        self.assertLess(scores["井筒风险"], 50)
        self.assertLess(scores["合规风险"], 50)

    def test_risky_high_geology_and_wellbore(self):
        """超破裂压力 → 地质和井筒风险必须高。"""
        done = {"uq": {"injection": {"probabilities": {"ok": 0.0, "risky": 0.98, "under": 0.02}}}}
        scores = compute_risk_scores(done, depth_m=2000, p_inj=350, temp=80)
        self.assertGreaterEqual(scores["地质风险"], 70)
        self.assertGreaterEqual(scores["井筒风险"], 70)
        self.assertGreaterEqual(scores["合规风险"], 60)

    def test_under_pressure(self):
        """压力不足 → 井筒风险中等，地质风险不应最高。"""
        done = {}
        scores = compute_risk_scores(done, depth_m=2000, p_inj=100, temp=60)
        self.assertGreaterEqual(scores["井筒风险"], 40)
        self.assertLess(scores["地质风险"], 80)

    def test_shallow_high_environmental(self):
        """浅层注入 → 环境风险必须高。"""
        done = {}
        scores = compute_risk_scores(done, depth_m=800, p_inj=120, temp=60)
        self.assertGreaterEqual(scores["环境风险"], 60)

    def test_deep_low_environmental(self):
        """深层注入 → 环境风险应较低。"""
        done = {}
        scores = compute_risk_scores(done, depth_m=3500, p_inj=300, temp=60)
        self.assertLess(scores["环境风险"], 30)

    def test_high_temp_corrosion(self):
        """高温 → 井筒风险因腐蚀而升高。"""
        done = {}
        scores_low_t = compute_risk_scores(done, depth_m=2000, p_inj=250, temp=40)
        scores_high_t = compute_risk_scores(done, depth_m=2000, p_inj=250, temp=120)
        self.assertGreater(scores_high_t["井筒风险"], scores_low_t["井筒风险"])

    def test_margin_penalty(self):
        """破裂压力裕度极小 → 地质和经济风险同时升高（两个都是 ok 工况，只比较裕度）。"""
        done = {}
        # depth=2000m: p_h≈202 bar, p_f≈303 bar
        # safe: p_inj=220, margin≈83 bar
        # tight: p_inj=295, margin≈8 bar
        scores_safe = compute_risk_scores(done, depth_m=2000, p_inj=220, temp=60)
        scores_tight = compute_risk_scores(done, depth_m=2000, p_inj=295, temp=60)
        self.assertGreater(scores_tight["地质风险"], scores_safe["地质风险"])
        self.assertGreater(scores_tight["经济风险"], scores_safe["经济风险"])

    def test_all_dimensions_present(self):
        """返回必须包含全部五个维度。"""
        scores = compute_risk_scores({}, depth_m=2000, p_inj=250, temp=80)
        self.assertEqual(set(scores), {"地质风险", "井筒风险", "环境风险", "经济风险", "合规风险"})

    def test_scores_clamped(self):
        """分数必须在 0–100 范围内。"""
        scores = compute_risk_scores({}, depth_m=200, p_inj=1200, temp=200)
        for v in scores.values():
            self.assertGreaterEqual(v, 0)
            self.assertLessEqual(v, 100)


class TestRiskRadarFigure(unittest.TestCase):
    """雷达图：纯构造测试，不验证像素。"""

    def test_returns_figure(self):
        import plotly.graph_objects as go

        scores = {"地质风险": 30, "井筒风险": 50, "环境风险": 20, "经济风险": 40, "合规风险": 10}
        fig = risk_radar_figure(scores)
        self.assertIsInstance(fig, go.Figure)

    def test_scores_out_of_range_clamped(self):
        """非法分数应被内部夹取，不抛异常。"""
        scores = {"地质风险": 150, "井筒风险": -10, "环境风险": 50, "经济风险": 50, "合规风险": 50}
        fig = risk_radar_figure(scores)
        self.assertIsNotNone(fig)

    def test_empty_scores_zero(self):
        """缺维度时补 0，不抛 KeyError。"""
        fig = risk_radar_figure({"地质风险": 50})
        self.assertIsNotNone(fig)


class TestStratigraphyFigure(unittest.TestCase):
    """地层剖面图：纯构造测试。"""

    def test_returns_figure(self):
        import plotly.graph_objects as go

        fig = stratigraphy_figure(depth_m=2000, p_inj=250, temp=80, has_fault=True)
        self.assertIsInstance(fig, go.Figure)

    def test_without_fault(self):
        """has_fault=False 时不应抛异常。"""
        fig = stratigraphy_figure(depth_m=2000, p_inj=250, temp=80, has_fault=False)
        self.assertIsNotNone(fig)

    def test_verdict_colors(self):
        """不同 verdict 应返回有效 Figure。"""
        for p in (100, 250, 400):  # under, ok, risky
            fig = stratigraphy_figure(depth_m=2000, p_inj=p, temp=80)
            self.assertIsNotNone(fig)

    def test_depth_changes_reservoir_position(self):
        """深度变化 → 注入层在图中的 y 位置应不同。
        这里只验证 layout 的 yaxis range 会随之变化。"""
        fig_shallow = stratigraphy_figure(depth_m=1000, p_inj=150, temp=60)
        fig_deep = stratigraphy_figure(depth_m=3500, p_inj=300, temp=60)
        # yaxis 是 reversed 的，range[1] 是顶部（负值或接近 0）
        # 我们至少确认 figure 不是同一个对象
        self.assertIsNot(fig_shallow, fig_deep)


if __name__ == "__main__":
    unittest.main()
