"""pressure_model 的单元测试：公式、边界、以及"只有一份实现"这件事。

最后一条是本文件存在的真正理由 —— 前端曲线和后端 PNG 必须用同一组公式，
否则图上的线和报告里的数字会悄悄对不上，而且不会抛任何异常。
"""
import unittest

import pressure_model as pm


class TestHydrostatic(unittest.TestCase):
    def test_scalar_known_value(self):
        # 1030 kg/m³ × 9.81 × 2000 m / 1e5 = 202.086 bar
        self.assertAlmostEqual(pm.hydrostatic_bar(2000), 202.1, places=1)

    def test_linear_in_depth(self):
        self.assertAlmostEqual(pm.hydrostatic_bar(4000),
                               2 * pm.hydrostatic_bar(2000), places=6)

    def test_zero_depth_is_zero(self):
        self.assertEqual(pm.hydrostatic_bar(0), 0.0)

    def test_sequence_returns_sequence(self):
        out = pm.hydrostatic_bar([1000, 2000])
        self.assertIsInstance(out, list)
        self.assertEqual(len(out), 2)
        self.assertAlmostEqual(out[1], pm.hydrostatic_bar(2000), places=9)

    def test_accepts_numpy_like_iterable(self):
        """chart_tools 传的是 np.linspace 的结果，不能因为"不是 list"就崩。"""
        class FakeArray:
            def __iter__(self):
                return iter([1000.0, 2000.0])

        out = pm.hydrostatic_bar(FakeArray())
        self.assertEqual(len(out), 2)


class TestFracture(unittest.TestCase):
    def test_ratio(self):
        for d in (500, 2000, 5000):
            self.assertAlmostEqual(pm.fracture_bar(d),
                                   pm.hydrostatic_bar(d) * pm.FRACTURE_RATIO,
                                   places=9)

    def test_sequence(self):
        out = pm.fracture_bar([1000, 2000])
        self.assertIsInstance(out, list)
        self.assertGreater(out[1], out[0])


class TestAdmissibleRange(unittest.TestCase):
    def test_round_trip(self):
        """区间的两个端点必须真的落在两条曲线上 —— 否则窗口是假的。"""
        p = 250.0
        d_min, d_max = pm.admissible_depth_range(p)
        self.assertAlmostEqual(float(pm.fracture_bar(d_min)), p, places=6)
        self.assertAlmostEqual(float(pm.hydrostatic_bar(d_max)), p, places=6)

    def test_ordering(self):
        d_min, d_max = pm.admissible_depth_range(250)
        self.assertLess(d_min, d_max)

    def test_higher_pressure_needs_deeper(self):
        a = pm.admissible_depth_range(200)
        b = pm.admissible_depth_range(400)
        self.assertGreater(b[1], a[1])


class TestClamp(unittest.TestCase):
    def test_bounds(self):
        self.assertEqual(pm.clamp_inputs(10, 5), (pm.MIN_DEPTH_M, pm.MIN_INJECTION_BAR))
        self.assertEqual(pm.clamp_inputs(1e6, 1e6), (pm.MAX_DEPTH_M, pm.MAX_INJECTION_BAR))

    def test_inside_untouched(self):
        self.assertEqual(pm.clamp_inputs(2500, 300), (2500.0, 300.0))


class TestVerdict(unittest.TestCase):
    def test_ok(self):
        # 2000 m：P_h≈202 bar，P_frac≈303 bar，250 bar 落在窗口内
        self.assertEqual(pm.verdict_of(2000, 250), "ok")

    def test_under_just_below_hydrostatic(self):
        # 2500 m 处 P_h≈253 bar —— 250 bar 比它还低，属于"注不进去"
        self.assertEqual(pm.verdict_of(2500, 250), "under")

    def test_risky(self):
        # 浅处破裂压力低：1500 m 处 P_frac≈227 bar，注入 400 bar 必然超
        self.assertEqual(pm.verdict_of(1500, 400), "risky")

    def test_under(self):
        # 深处静水压力高：4000 m 处 P_h≈404 bar，注入 250 bar 注不进去
        self.assertEqual(pm.verdict_of(4000, 250), "under")

    def test_boundary_is_ok(self):
        """正好等于阈值算安全（判断用严格大于/小于）。"""
        d = 2000.0
        self.assertEqual(pm.verdict_of(d, float(pm.fracture_bar(d))), "ok")
        self.assertEqual(pm.verdict_of(d, float(pm.hydrostatic_bar(d))), "ok")

    def test_clamped_before_judgement(self):
        """离谱入参先被夹住再判，不会因为 1e9 bar 得出奇怪结论。"""
        self.assertEqual(pm.verdict_of(10, 1e9), pm.verdict_of(pm.MIN_DEPTH_M,
                                                              pm.MAX_INJECTION_BAR))


if __name__ == "__main__":
    unittest.main()
