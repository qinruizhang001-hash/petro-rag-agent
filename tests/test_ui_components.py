"""ui_components 的单元测试：控制台里那些"给定数字算出一张图/一段 HTML"的逻辑。

这些函数原来长在 app.py 里，而 app.py 一 import 就会执行整个 Streamlit 页面，
于是 CI 一行都测不到 —— 只能靠人肉开浏览器看。抽出来之后就能在这里断言。
"""
import unittest

import plotly.graph_objects as go

import ui_components as ui


class TestComposeInput(unittest.TestCase):
    def test_params_are_included(self):
        text = ui.compose_input("某项目有断层", 2500, 250, 80, "注入")
        for expect in ("2500 m", "250 bar", "80 ℃", "注入"):
            self.assertIn(expect, text)

    def test_description_is_preserved(self):
        text = ui.compose_input("某项目有断层", 2500, 250, 80, "注入")
        self.assertTrue(text.startswith("某项目有断层"))

    def test_precedence_is_stated(self):
        """必须写明冲突时以表单为准，否则模型拿到两个数会瞎猜。"""
        self.assertIn("以此为准", ui.compose_input("x", 1, 1, 1, "注入"))

    def test_empty_description(self):
        text = ui.compose_input("   ", 2500, 250, 80, "注入")
        self.assertTrue(text.startswith("【结构化参数"))
        self.assertNotIn("\n\n\n", text)

    def test_float_formatting(self):
        """2000.0 要写成 2000 而不是 2000.0 —— 描述里带小数点很像机器噪声。"""
        self.assertIn("2000 m", ui.compose_input("", 2000.0, 250.0, 80.0, "注入"))


class TestFormatters(unittest.TestCase):
    def test_fmt_s(self):
        self.assertEqual(ui.fmt_s(None), "—")
        self.assertEqual(ui.fmt_s("abc"), "—")
        self.assertEqual(ui.fmt_s(9.44), "9.4s")
        self.assertEqual(ui.fmt_s(93.2), "1.6min")

    def test_fmt_ms(self):
        self.assertEqual(ui.fmt_ms(None), "—")
        self.assertEqual(ui.fmt_ms(0), "0ms")
        self.assertEqual(ui.fmt_ms(850), "850ms")
        self.assertEqual(ui.fmt_ms(1500), "1.50s")

    def test_fmt_tok(self):
        self.assertEqual(ui.fmt_tok(None), "—")
        self.assertEqual(ui.fmt_tok(999), "999")
        self.assertEqual(ui.fmt_tok(12345), "12.3k")


class TestPipelineHtml(unittest.TestCase):
    def test_all_nodes_rendered(self):
        html = ui.pipeline_html({}, {}, running=False)
        self.assertEqual(html.count('class="row'), len(ui.PIPELINE))

    def test_never_run_shows_dash(self):
        html = ui.pipeline_html({}, {}, running=False)
        self.assertNotIn("跳过", html)   # 从没跑过就不能说"跳过"
        self.assertIn("—", html)

    def test_running_shows_ellipsis(self):
        html = ui.pipeline_html({"extract": "start"}, {}, running=True)
        self.assertIn("运行中", html)

    def test_done_shows_duration(self):
        html = ui.pipeline_html(
            {"extract": "end"},
            {"extract": {"start": 1000, "end": 4200}},
            running=False,
        )
        self.assertIn("3.20s", html)
        self.assertIn("led-ok", html)

    def test_skipped_after_run(self):
        """跑过一次但某节点没出现 → 说"跳过"，这是真实信息。"""
        html = ui.pipeline_html({"extract": "end"}, {}, running=False)
        self.assertIn("跳过", html)

    def test_unknown_node_does_not_break(self):
        html = ui.pipeline_html({"不存在的节点": "end"}, {}, running=False)
        self.assertEqual(html.count('class="row'), len(ui.PIPELINE))


class TestPressureFigure(unittest.TestCase):
    def test_returns_figure(self):
        self.assertIsInstance(ui.pressure_figure(2500, 250), go.Figure)

    def test_depth_axis_reversed(self):
        """深度向下增大 —— 画反了地质图就读不通。"""
        fig = ui.pressure_figure(2500, 250)
        self.assertEqual(fig.layout.yaxis.autorange, "reversed")

    def test_has_envelope_and_injection(self):
        fig = ui.pressure_figure(2500, 250)
        names = [t.name for t in fig.data]
        self.assertIn("静水压力 P_h", names)
        self.assertIn("破裂压力 P_frac", names)
        self.assertIn("安全注入窗口", names)
        self.assertTrue(any(n and n.startswith("注入压力") for n in names))

    def test_threshold_lines_present(self):
        """风险阈值线必须真的画在储层深度处的 P_frac / P_h 上。"""
        from pressure_model import fracture_bar, hydrostatic_bar

        fig = ui.pressure_figure(2500, 250)
        xs = [s.x0 for s in fig.layout.shapes if getattr(s, "x0", None) is not None]
        self.assertIn(round(float(fracture_bar(2500)), 6), [round(float(x), 6) for x in xs])
        self.assertIn(round(float(hydrostatic_bar(2500)), 6), [round(float(x), 6) for x in xs])

    def test_verdict_colour_changes(self):
        """超压时工作点必须是告警色，不能被画成绿色。"""
        risky = ui.pressure_figure(1500, 400)
        ok = ui.pressure_figure(2000, 250)
        self.assertEqual(risky.data[-1].marker.color, ui.VERDICT_TEXT["risky"][1])
        self.assertEqual(ok.data[-1].marker.color, ui.VERDICT_TEXT["ok"][1])

    def test_out_of_range_input_is_clamped(self):
        fig = ui.pressure_figure(1e9, 1e9)          # 不能抛异常
        self.assertIsInstance(fig, go.Figure)

    def test_margins_leave_room_for_annotations(self):
        """右边距要留够，"储层 xxx m" 的横线标注否则会被裁掉。"""
        fig = ui.pressure_figure(2500, 250)
        self.assertGreaterEqual(fig.layout.margin.r, 48)
        self.assertGreaterEqual(fig.layout.margin.t, 48)


class TestTimingFigure(unittest.TestCase):
    def test_ignores_incomplete_rows(self):
        fig = ui.timing_figure([{"label": "a", "ms": 1200},
                                {"label": "b", "ms": None}])
        self.assertEqual(len(fig.data[0].x), 1)

    def test_sorted_ascending(self):
        fig = ui.timing_figure([{"label": "慢", "ms": 3000},
                                {"label": "快", "ms": 500}])
        self.assertEqual(list(fig.data[0].y), ["快", "慢"])

    def test_seconds_conversion(self):
        fig = ui.timing_figure([{"label": "a", "ms": 2500}])
        self.assertAlmostEqual(fig.data[0].x[0], 2.5)


class TestTokenFigure(unittest.TestCase):
    def test_two_series(self):
        fig = ui.token_figure([{"stage": "extract", "input_tokens": 100,
                                "output_tokens": 20}])
        self.assertEqual([t.name for t in fig.data], ["输入", "输出"])
        self.assertEqual(fig.layout.barmode, "stack")

    def test_missing_tokens_treated_as_zero(self):
        fig = ui.token_figure([{"stage": "x"}])
        self.assertEqual(fig.data[0].y[0], 0)


if __name__ == "__main__":
    unittest.main()
