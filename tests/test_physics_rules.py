"""
离线单元测试：物理一致性校验规则引擎
====================================
这套规则是 Agent 4 的**一票否决权**——它说报告与物理约束矛盾，报告就 0 分打回。
所以测试重点不是"happy path 能过"，而是：

  · 该拦的必须拦住（超破裂压力却宣称安全 / 注不进却说注入顺畅）
  · 不该拦的不能误拦（点明了风险、用了不同破裂梯度、只缺深度）
  · 文本判定的**方向性**（"破裂压力高于注入压力"是反向表述，不能算已点明）
  · 否定词（"未超过""不超过""没有超过"）

零第三方依赖：CI 的 unit-core 直接跑它。
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.physics_rules import (
    FRACTURE_RATIO,
    acknowledges_fracture_exceeded,
    claims_smooth_injection,
    evaluate_physics,
    extract_parameters,
    fracture_bar,
    hydrostatic_bar,
    render_prompt_block,
)


class TestPressureFormulas(unittest.TestCase):
    def test_hydrostatic_at_2000m(self):
        """ρ=1030 kg/m³、g=9.81 → 2000 m 约 202 bar（与 chart_tools 同口径）。"""
        self.assertAlmostEqual(hydrostatic_bar(2000), 202.1, places=1)

    def test_fracture_is_ratio_of_hydrostatic(self):
        d = 1500
        self.assertAlmostEqual(fracture_bar(d),
                               hydrostatic_bar(d) * FRACTURE_RATIO, places=6)

    def test_matches_chart_tools_when_available(self):
        """
        两处各写了一份公式（这里不依赖 matplotlib，chart_tools 依赖），
        必须保证不漂移。有 matplotlib 时交叉校验，没有就跳过。
        """
        try:
            from chart_tools import fracture_bar as ct_fracture
            from chart_tools import hydrostatic_bar as ct_hydro
        except Exception:
            self.skipTest("chart_tools 不可用（缺 matplotlib），跳过交叉校验")

        for depth in (500, 1500, 2500, 4000):
            self.assertAlmostEqual(float(ct_hydro(depth)), hydrostatic_bar(depth),
                                   places=6)
            self.assertAlmostEqual(float(ct_fracture(depth)), fracture_bar(depth),
                                   places=6)


class TestParameterExtraction(unittest.TestCase):
    def test_chinese_bar_and_meter(self):
        p = extract_parameters("某项目埋深2500m，注入压力250bar，温度80℃")
        self.assertEqual(p["injection_pressure_bar"], 250.0)
        self.assertEqual(p["depth_m"], 2500.0)
        self.assertEqual(p["temperature_c"], 80.0)

    def test_mpa_is_converted_to_bar(self):
        """16 MPa = 160 bar。不换算会让校验结果差 10 倍，且看起来完全合理。"""
        p = extract_parameters("注入压力 16 MPa，储层深度 1.5 km")
        self.assertAlmostEqual(p["injection_pressure_bar"], 160.0)
        self.assertAlmostEqual(p["depth_m"], 1500.0)

    def test_injection_pressure_preferred_over_bare_pressure(self):
        """
        同一段文本里既有"注入压力 250 bar"又有"地层破裂压力 378 bar"时，
        必须抓前者。抓错的话校验就变成"拿破裂压力和破裂压力比"，永远通过。
        """
        p = extract_parameters("注入压力250bar，地层破裂压力378bar，埋深2500m")
        self.assertEqual(p["injection_pressure_bar"], 250.0)
        self.assertEqual(p["fracture_bar_stated"], 378.0)

    def test_depth_out_of_range_dropped(self):
        """把 3000 写成 3 时宁可判"缺深度"，也不要拿 3 m 去算破裂压力。"""
        p = extract_parameters("埋深3m，注入压力250bar")
        self.assertIsNone(p["depth_m"])

    def test_falls_back_to_user_input(self):
        p = extract_parameters("", "埋深2000m，注入压力300bar")
        self.assertEqual(p["injection_pressure_bar"], 300.0)
        self.assertEqual(p["depth_m"], 2000.0)


class TestAcknowledgmentDetection(unittest.TestCase):
    def test_forward_statement_counts(self):
        self.assertTrue(acknowledges_fracture_exceeded(
            "注入压力 250 bar 已超过 1500 m 处地层破裂压力 227 bar。"))

    def test_reverse_statement_does_not_count(self):
        """
        反向表述（破裂压力 > 注入压力）意思是"安全"，不能算已点明风险。
        这是本规则最容易写错的地方：按关键词共现判断会 100% 误判。
        """
        self.assertFalse(acknowledges_fracture_exceeded(
            "地层破裂压力 227 bar 高于注入压力。"))

    def test_negated_comparison_does_not_count(self):
        for text in ("注入压力未超过地层破裂压力。",
                     "注入压力不超过地层破裂压力。",
                     "注入压力没有超过地层破裂压力。"):
            self.assertFalse(acknowledges_fracture_exceeded(text), text)

    def test_passive_voice_counts(self):
        """"破裂压力被突破"语序相反，但语义上确实是点明了风险。"""
        self.assertTrue(acknowledges_fracture_exceeded("盖层破裂压力已被突破。"))

    def test_smooth_injection_claim_found(self):
        self.assertIn("注入顺畅", claims_smooth_injection("注入顺畅，压力充足。"))
        self.assertEqual(claims_smooth_injection("注入压力偏低，需要增压。"), "")


class TestEvaluatePhysics(unittest.TestCase):
    def test_above_fracture_without_acknowledgment_blocks(self):
        r = evaluate_physics(
            params_text="埋深1500m，注入压力250bar，温度80℃",
            report_text="本项目管理规范，风险可控，注入作业安全。",
        )
        self.assertEqual(r["status"], "violation")
        self.assertTrue(r["blocking"])
        self.assertEqual(r["verdict"], "above_fracture")
        self.assertIn("P_INJ_ABOVE_FRACTURE_UNACKNOWLEDGED",
                      [v["code"] for v in r["violations"]])

    def test_above_fracture_with_acknowledgment_passes(self):
        r = evaluate_physics(
            params_text="埋深1500m，注入压力250bar，温度80℃",
            report_text="注入压力 250 bar 已超过 1500 m 处地层破裂压力 227 bar，"
                        "存在盖层压裂与 CO2 泄漏风险。",
        )
        self.assertFalse(r["blocking"])
        self.assertEqual(r["status"], "pass")

    def test_feedback_is_actionable(self):
        """
        打回时必须给出**具体数值与方向**。只说"报告有物理错误"，
        下一版模型只会换个说法继续错。
        """
        r = evaluate_physics(
            params_text="埋深1500m，注入压力250bar，温度80℃",
            report_text="风险可控。",
        )
        fb = r["feedback"]
        self.assertIn("≤ 227", fb)          # 降到多少 bar
        self.assertIn("≥ 1649", fb)         # 或加深到多少 m
        self.assertIn("压裂", fb)

    def test_below_hydrostatic_with_smooth_claim_blocks(self):
        r = evaluate_physics(
            params_text="埋深2500m，注入压力250bar，温度80℃",
            report_text="注入顺畅，注入压力充足，作业无阻碍。",
        )
        self.assertTrue(r["blocking"])
        self.assertEqual(r["verdict"], "below_hydrostatic")
        self.assertIn("P_INJ_BELOW_HYDROSTATIC_CLAIMED_SMOOTH",
                      [v["code"] for v in r["violations"]])

    def test_below_hydrostatic_without_claim_does_not_block(self):
        """没说"顺畅"只是没提，不是矛盾 —— 不能因此打回。"""
        r = evaluate_physics(
            params_text="埋深2500m，注入压力250bar，温度80℃",
            report_text="注入压力偏低，建议提高井口压力。",
        )
        self.assertFalse(r["blocking"])
        self.assertEqual(r["verdict"], "below_hydrostatic")

    def test_safe_window_passes(self):
        r = evaluate_physics(
            params_text="埋深2500m，注入压力300bar，温度80℃",
            report_text="注入压力位于安全注入窗口内。",
        )
        self.assertEqual(r["status"], "pass")
        self.assertEqual(r["verdict"], "safe_window")

    def test_missing_depth_is_unverifiable_not_blocking(self):
        r = evaluate_physics(params_text="注入压力250bar，温度80℃",
                             report_text="风险可控。")
        self.assertEqual(r["status"], "unverifiable")
        self.assertFalse(r["blocking"])

    def test_missing_pressure_is_unverifiable(self):
        r = evaluate_physics(params_text="埋深2500m", report_text="...")
        self.assertEqual(r["status"], "unverifiable")
        self.assertFalse(r["blocking"])

    def test_implausible_pressure_is_warning_only(self):
        """
        5 bar 注进 2500 m 的储层几乎一定是单位/数值错误。这类问题必须提示，
        但它属于"数据可疑"，不该单独构成"报告与物理矛盾"，所以只是 warning。
        （注意 3000 bar 那种情况会**同时**触发超破裂压力，阻断的是后者。）
        """
        r = evaluate_physics(params_text="埋深2500m，注入压力5bar")
        self.assertIn("IMPLAUSIBLE_PRESSURE",
                      [v["code"] for v in r["violations"]])
        self.assertFalse(r["blocking"])

    def test_fracture_value_mismatch_is_warning(self):
        """报告写 500 bar、实际 379 bar（偏差 32%）→ 提示但不阻断。"""
        r = evaluate_physics(
            params_text="埋深2500m，注入压力300bar，破裂压力500bar",
            report_text="破裂压力 500 bar。",
        )
        codes = [v["code"] for v in r["violations"]]
        self.assertIn("FRACTURE_VALUE_MISMATCH", codes)
        self.assertFalse(r["blocking"])

    def test_fracture_value_within_gradient_range_is_accepted(self):
        """
        破裂梯度工程上取 1.4–1.8 倍静水，1.8 倍相对本模块的 1.5 倍是 +20%。
        容差 25% 必须覆盖它，否则用 1.8 倍梯度的报告会被全量误判。
        """
        d = 2500
        alt = hydrostatic_bar(d) * 1.8      # 用 1.8 倍梯度的报告
        r = evaluate_physics(
            params_text=f"埋深{d}m，注入压力300bar，破裂压力{alt:.0f}bar",
            report_text=f"破裂压力约 {alt:.0f} bar。",
        )
        self.assertNotIn("FRACTURE_VALUE_MISMATCH",
                         [v["code"] for v in r["violations"]])

    def test_density_mismatch_is_warning(self):
        r = evaluate_physics(
            params_text="埋深2500m，注入压力300bar，温度80℃",
            report_text="CO2 密度约 5.00 kg/m³。",
        )
        self.assertIn("DENSITY_MISMATCH", [v["code"] for v in r["violations"]])
        self.assertFalse(r["blocking"])

    def test_params_can_be_injected_directly(self):
        """测试时直接给数值，跳过文本抽取（抽取本身另有测试）。"""
        r = evaluate_physics(
            report_text="安全。",
            params={"injection_pressure_bar": 250, "depth_m": 1500,
                    "temperature_c": 80},
        )
        self.assertTrue(r["blocking"])

    def test_checks_are_exposed_for_audit(self):
        """每个判定都要能解释"为什么"，否则又变成一个黑盒。"""
        r = evaluate_physics(
            params_text="埋深2500m，注入压力300bar，温度80℃",
            report_text="安全窗口内。",
        )
        codes = [c["code"] for c in r["checks"]]
        self.assertIn("pressure_vs_fracture", codes)
        self.assertIn("pressure_vs_hydrostatic", codes)
        self.assertTrue(all("detail" in c for c in r["checks"]))


class TestPromptRendering(unittest.TestCase):
    """
    ``render_prompt_block`` 的"空串契约"。

    这不是格式问题，是行为问题：它决定了评审模型会不会把"没算过"读成
    "校验没通过"。曾经因此让一份正常报告从 7 分掉到 3 分，所以必须锁死。
    """

    def test_unverifiable_renders_nothing(self):
        """参数不足 → 不产生任何文本，调用方整块省略。"""
        r = evaluate_physics(params_text="注入压力250bar",
                             user_input="注入压力250bar")
        self.assertEqual(r["status"], "unverifiable")
        self.assertEqual(render_prompt_block(r), "")

    def test_unverifiable_text_never_reaches_the_model(self):
        """
        兜底断言：空串之外，任何"没算"的痕迹都不该出现。

        即便有人改成返回一句说明，也不能出现会让模型误判为"冲突"的词。
        """
        r = evaluate_physics(params_text="注入压力250bar",
                             user_input="注入压力250bar")
        block = render_prompt_block(r)
        for word in ("冲突", "违规", "矛盾", "未通过"):
            self.assertNotIn(word, block)

    def test_pass_renders_verdict_and_numbers(self):
        r = evaluate_physics(params_text="埋深2000m，注入压力250bar")
        self.assertEqual(r["status"], "pass")
        block = render_prompt_block(r)
        self.assertIn("pass / safe_window", block)
        self.assertIn("未发现违规", block)
        # 数值必须给出来，否则模型只能凭感觉判断报告里的数字是否合理
        self.assertIn("303", block)      # 2000 m 处破裂压力

    def test_violation_lists_the_reason(self):
        r = evaluate_physics(
            params_text="埋深1500m，注入压力250bar",
            report_text="风险可控，注入顺畅。",
        )
        self.assertEqual(r["status"], "violation")
        block = render_prompt_block(r)
        self.assertIn("violation", block)
        self.assertIn("违规[blocking]", block)

    def test_unknown_status_renders_nothing(self):
        """没有 status 字段（例如旧缓存里的 physics_check）也不能崩。"""
        self.assertEqual(render_prompt_block({}), "")
        self.assertEqual(render_prompt_block({"status": "bogus"}), "")


if __name__ == "__main__":
    unittest.main(verbosity=2)
