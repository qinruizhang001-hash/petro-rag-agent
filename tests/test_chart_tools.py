"""
离线单元测试：地质图表工具
==========================
matplotlib 用 Agg 后端，不需要显示器；不联网、不需要 Ollama。
CI 里能直接跑真实绘图断言。

运行:
    python -m unittest discover -s tests -v
"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chart_tools import (
    DEFAULT_DEPTH_M,
    FRACTURE_RATIO,
    admissible_depth_range,
    fracture_bar,
    generate_pressure_curve,
    hydrostatic_bar,
)


class TestPhysics(unittest.TestCase):
    def test_hydrostatic_at_2000m(self):
        """ρ=1030 kg/m³、g=9.81 → 2000 m 约 202 bar。"""
        self.assertAlmostEqual(float(hydrostatic_bar(2000)), 202.1, places=1)

    def test_hydrostatic_is_linear(self):
        self.assertAlmostEqual(float(hydrostatic_bar(4000)),
                               2 * float(hydrostatic_bar(2000)), places=6)

    def test_fracture_is_ratio_of_hydrostatic(self):
        d = 1500
        self.assertAlmostEqual(float(fracture_bar(d)),
                               float(hydrostatic_bar(d)) * FRACTURE_RATIO, places=6)

    def test_admissible_range_for_250bar(self):
        """250 bar 的可注入深度区间应约 1649–2474 m。"""
        d_min, d_max = admissible_depth_range(250)
        self.assertAlmostEqual(d_min, 1649, delta=3)
        self.assertAlmostEqual(d_max, 2474, delta=3)
        self.assertLess(d_min, d_max)


class TestVerdict(unittest.TestCase):
    """三种工程结论必须判对 —— 这是图能不能用的关键。"""

    def _gen(self, depth, pressure):
        with tempfile.TemporaryDirectory() as d:
            return generate_pressure_curve(depth, pressure, output_dir=d)

    def test_safe_window(self):
        r = self._gen(2000, 250)
        self.assertEqual(r["verdict"], "ok")
        self.assertLess(r["hydrostatic_bar_at_depth"], r["injection_pressure_bar"])
        self.assertLess(r["injection_pressure_bar"], r["fracture_bar_at_depth"])

    def test_above_fracture_pressure_is_risky(self):
        r = self._gen(1500, 250)   # P_frac(1500) ≈ 227 bar < 250
        self.assertEqual(r["verdict"], "risky")
        self.assertIn("破裂压力", r["summary"])

    def test_below_formation_pressure_is_under(self):
        r = self._gen(3000, 250)   # P_h(3000) ≈ 303 bar > 250
        self.assertEqual(r["verdict"], "under")
        self.assertIn("低于地层压力", r["summary"])

    def test_boundaries_are_consistent(self):
        """结论与三个压力值必须自洽，不能互相矛盾。"""
        for depth, p in [(1000, 200), (2000, 250), (3000, 400), (2500, 320)]:
            r = self._gen(depth, p)
            p_h = r["hydrostatic_bar_at_depth"]
            p_f = r["fracture_bar_at_depth"]
            p_inj = r["injection_pressure_bar"]
            if r["verdict"] == "risky":
                self.assertGreater(p_inj, p_f, f"{depth}/{p} 判 risky 但未超破裂压力")
            elif r["verdict"] == "under":
                self.assertLess(p_inj, p_h, f"{depth}/{p} 判 under 但不低于地层压力")
            else:
                self.assertTrue(p_h <= p_inj <= p_f,
                                f"{depth}/{p} 判 ok 但不在窗口内")


class TestChartOutput(unittest.TestCase):
    def test_creates_real_png(self):
        with tempfile.TemporaryDirectory() as d:
            r = generate_pressure_curve(2000, 250, output_dir=d)
            self.assertEqual(r["status"], "ok")
            self.assertTrue(os.path.exists(r["path"]))
            size = os.path.getsize(r["path"])
            self.assertGreater(size, 15000, "PNG 太小，可能是空图")

            with open(r["path"], "rb") as f:
                self.assertEqual(f.read(8), b"\x89PNG\r\n\x1a\n", "不是合法 PNG")

    def test_uses_unique_filenames(self):
        with tempfile.TemporaryDirectory() as d:
            a = generate_pressure_curve(2000, 250, output_dir=d)
            b = generate_pressure_curve(2000, 250, output_dir=d)
            self.assertNotEqual(a["path"], b["path"], "并发调用会互相覆盖")

    def test_rel_path_is_posix_and_inside_project(self):
        with tempfile.TemporaryDirectory() as d:
            r = generate_pressure_curve(2000, 250, output_dir=d)
            self.assertIn("/", r["url_path"])
            self.assertTrue(r["url_path"].startswith("/charts/pressure_curve_"))

    def test_defaults_when_depth_missing(self):
        with tempfile.TemporaryDirectory() as d:
            r = generate_pressure_curve(None, None, output_dir=d)
            self.assertEqual(r["depth_m"], DEFAULT_DEPTH_M)
            self.assertTrue(r["depth_assumed"], "未给深度时必须标注为估算")
            self.assertIn("估算", r["summary"])

    def test_extreme_inputs_are_clamped(self):
        """模型可能传入离谱数值，不能让图崩掉或坐标轴炸开。"""
        with tempfile.TemporaryDirectory() as d:
            r1 = generate_pressure_curve(999999, 250, output_dir=d)
            self.assertLessEqual(r1["depth_m"], 8000)
            r2 = generate_pressure_curve(2000, 999999, output_dir=d)
            self.assertLessEqual(r2["injection_pressure_bar"], 1200)
            for r in (r1, r2):
                self.assertTrue(os.path.exists(r["path"]))

    def test_no_figure_leak(self):
        """反复调用不能累积 pyplot 的 figure（否则长跑必泄漏）。"""
        import matplotlib.pyplot as plt

        with tempfile.TemporaryDirectory() as d:
            for _ in range(6):
                generate_pressure_curve(2000, 250, output_dir=d)
            self.assertEqual(len(plt.get_fignums()), 0, "figure 未关闭，存在泄漏")


if __name__ == "__main__":
    unittest.main(verbosity=2)
