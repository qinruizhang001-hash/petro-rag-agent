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
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests import _graph_stubs

# 同一个坑的另一半：整图测试把 ``chart_tools`` 换成只含
# ``generate_pressure_curve`` 的替身。本模块在顶层 import 了 7 个名字，
# 一旦加载顺序排在桩之后，报的是 "cannot import name 'DEFAULT_DEPTH_M'
# from 'chart_tools' (unknown location)" —— 听起来像打包问题，实际是替身。
# 这里显式要回真实模块；没装过桩时是空操作。
_graph_stubs.release("chart_tools")

from chart_tools import (  # noqa: E402
    DEFAULT_DEPTH_M,
    FRACTURE_RATIO,
    admissible_depth_range,
    fracture_bar,
    generate_pressure_curve,
    hydrostatic_bar,
    prune_charts,
)


class TestPhysics(unittest.TestCase):
    def test_single_source_of_truth(self):
        """公式只能有一份：chart_tools 必须复用 pressure_model，不能自己再写一遍。

        前端 app.py 用 pressure_model 实时画曲线、后端用 chart_tools 出报告配图。
        两份实现一旦漂移，图上的线和报告里的数字就会对不上，而且不会报错 ——
        所以这里断言的是"同一个函数对象"，不是"结果相等"。
        """
        import pressure_model

        self.assertIs(hydrostatic_bar, pressure_model.hydrostatic_bar)
        self.assertIs(fracture_bar, pressure_model.fracture_bar)
        self.assertIs(admissible_depth_range, pressure_model.admissible_depth_range)
        self.assertIs(FRACTURE_RATIO, pressure_model.FRACTURE_RATIO)

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


class PruneTests(unittest.TestCase):
    """图表目录的保留策略：只删自己生成的、按数量与总量双限、可预演。"""

    def _make(self, d, count, *, prefix="pressure_curve_", size=1024):
        """造 count 个假图，mtime 递增（第 i 个比第 i-1 个新）。"""
        import time

        paths = []
        base = time.time() - 10_000
        for i in range(count):
            p = os.path.join(d, f"{prefix}20260101-0000{i:02d}_abc{i:03d}.png")
            with open(p, "wb") as fh:
                fh.write(b"\0" * size)
            os.utime(p, (base + i, base + i))
            paths.append(p)
        return paths

    def test_keeps_the_newest_and_reports_names(self):
        with tempfile.TemporaryDirectory() as d:
            paths = self._make(d, 6)
            report = prune_charts(keep=2, directory=d)
            self.assertEqual(report["kept"], 2)
            self.assertEqual(len(report["deleted"]), 4)
            # 留下的是最新的两个
            survivors = sorted(os.listdir(d))
            self.assertEqual(survivors, [os.path.basename(p) for p in paths[-2:]])
            # 删掉的是最旧的两个 —— 名字必须可复核，不能只报计数
            self.assertIn(os.path.basename(paths[0]), report["deleted"])

    def test_dry_run_touches_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            self._make(d, 5)
            report = prune_charts(keep=1, dry_run=True, directory=d)
            self.assertEqual(len(os.listdir(d)), 5)
            self.assertEqual(len(report["deleted"]), 4)
            self.assertTrue(report["dry_run"])

    def test_only_its_own_files_are_candidates(self):
        with tempfile.TemporaryDirectory() as d:
            self._make(d, 4)
            stranger = os.path.join(d, "uploaded_by_user.png")
            note = os.path.join(d, "notes.txt")
            for p in (stranger, note):
                with open(p, "w", encoding="utf-8") as fh:
                    fh.write("x")
            prune_charts(keep=1, directory=d)
            self.assertTrue(os.path.exists(stranger), "误删了不是自己生成的文件")
            self.assertTrue(os.path.exists(note))

    def test_size_cap_trims_further(self):
        with tempfile.TemporaryDirectory() as d:
            # 5 个 × 1 MB，上限 2 MB → 数量合规（keep=5）但总量超限
            self._make(d, 5, size=1024 * 1024)
            report = prune_charts(keep=5, max_total_mb=2.0, directory=d)
            self.assertEqual(report["kept"], 2)
            self.assertEqual(report["reason"], "超出总量上限")
            self.assertLessEqual(len(os.listdir(d)), 2)

    def test_negative_keep_is_rejected(self):
        with self.assertRaises(ValueError):
            prune_charts(keep=-1, directory=tempfile.gettempdir())

    def test_missing_directory_is_not_an_error(self):
        report = prune_charts(keep=3, directory=os.path.join(
            tempfile.gettempdir(), "definitely-not-here-xyz"
        ))
        self.assertEqual(report["deleted"], [])
        self.assertIn("不存在", report["reason"])

    def test_no_op_does_not_blame_the_count_cap(self):
        """
        什么都没删时，原因不能写"超出保留数量"。

        "删了 0 个（原因：超出保留数量）"读起来像规则生效了并放行了 ——
        而实际是规则根本没触发。清理报告里最容易误导人的就是这一行，
        因为它让人不去查真正的增长来源。
        """
        with tempfile.TemporaryDirectory() as d:
            self._make(d, 3)
            report = prune_charts(keep=10, directory=d)
            self.assertEqual(report["deleted"], [])
            self.assertEqual(report["kept"], 3)
            self.assertNotIn("超出", report["reason"])
            self.assertIn("无需清理", report["reason"])


class AutoPruneWiringTests(unittest.TestCase):
    """
    出图之后**真的**会清理 —— 测的是"调用点接没接上"，不是清理逻辑本身。

    为什么必须单独一条：`prune_charts` 可以写得完全正确，而
    `generate_pressure_curve` 里那行 `_auto_prune(out_dir)` 被删掉、被挪到
    `return` 之后、或者缩进错位时，`PruneTests` 一条都不会红，目录照样无限
    增长 —— 而"目录无限增长"正是这套代码存在的理由。

    本项目已经在同一个形状上踩过一次：消融补丁没到达调用点，于是"关掉闸门"
    那一组跑出来和基线一模一样，看起来像"闸门没用"而不是"补丁没生效"。
    """

    def _seed_old(self, d: str) -> str:
        """放一张足够旧的图，keep=1 时它必须被删。"""
        path = os.path.join(d, "pressure_curve_20200101-000000_old000.png")
        with open(path, "wb") as fh:
            fh.write(b"\0" * 1024)
        os.utime(path, (1, 1))
        return path

    def test_generate_pressure_curve_prunes_after_writing(self):
        with tempfile.TemporaryDirectory() as d:
            old = self._seed_old(d)
            with mock.patch.dict(os.environ, {"CHART_RETENTION": "1"}):
                result = generate_pressure_curve(
                    depth_m=2000, injection_pressure_bar=250, output_dir=d
                )
            self.assertEqual(result["status"], "ok", result)
            remaining = sorted(os.listdir(d))
            self.assertEqual(len(remaining), 1, f"清理没有生效：{remaining}")
            self.assertNotIn(os.path.basename(old), remaining)

    def test_retention_zero_disables_pruning(self):
        """
        `CHART_RETENTION=0` 是"我要归档全部图"，不是"只留 0 张"。
        这两者搞反的后果是**静默删光**，所以单独钉住。
        """
        with tempfile.TemporaryDirectory() as d:
            old = self._seed_old(d)
            with mock.patch.dict(os.environ, {"CHART_RETENTION": "0"}):
                generate_pressure_curve(
                    depth_m=2000, injection_pressure_bar=250, output_dir=d
                )
            remaining = os.listdir(d)
            self.assertEqual(len(remaining), 2, f"关闭后不该删任何图：{remaining}")
            self.assertIn(os.path.basename(old), remaining)

    def test_pruning_failure_does_not_break_the_chart(self):
        """
        清理是维护动作，出图是产品。清理抛异常时图必须照常生成 ——
        否则"磁盘满了"会变成"报告没有图"，两件不相干的事被绑在一起。
        """
        with tempfile.TemporaryDirectory() as d:
            with mock.patch("chart_tools.prune_charts", side_effect=OSError("boom")):
                result = generate_pressure_curve(
                    depth_m=2000, injection_pressure_bar=250, output_dir=d
                )
            self.assertEqual(result["status"], "ok", result)
            self.assertTrue(os.path.exists(result["path"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
