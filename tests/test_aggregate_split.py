"""拆出来的纯函数：Prompt 组装 / 工具清单 / 报告收尾。

``nodes/aggregate_node.py`` 原本是一个 965 行的神节点：合并、MCP 工具调度、
不确定性量化、流式输出、记忆写入全在一起。拆成
``aggregate_prompt`` / ``aggregate_support`` / ``aggregate_finalize`` 之后，
这里的测试能在**零三方依赖**下跑（不需要 torch / chromadb / Ollama），
而节点本身只剩编排。

拆分的验收标准是"行为不变"，所以这些测试盯的是**具体输出**：
Prompt 里必须还有那几条硬约束，收尾必须还在补数值、删编造百分比。
"""
from __future__ import annotations

import unittest

from nodes.aggregate_finalize import finalize_report
from nodes.aggregate_prompt import build_degraded_hint, build_materials, build_system_prompt
from nodes.aggregate_support import (
    chart_payload,
    extract_depth,
    extract_pressure_temperature,
    plan_payload,
    strip_tool_json,
    uq_payload,
)

BRANCH_OK = {
    "geology": "盖层为泥岩，破裂压力梯度约 1.5 倍静水。",
    "wellbore": "注入压力应低于破裂压力。",
    "degraded": False,
    "missing": [],
}

BRANCH_DEGRADED = {
    "geology": "地质模块未能提供可用证据，请人工复核",
    "wellbore": "井筒结论。",
    "degraded": True,
    "missing": ["geology"],
}


class SystemPromptTests(unittest.TestCase):
    def test_hard_constraints_survive_the_move(self):
        prompt = build_system_prompt("用户画像：结论先行")
        for must in (
            "用户画像：结论先行",
            "禁止使用",
            "绝对不要输出 JSON",
            "概率只能照抄",
            "最新时间戳",
            "每一句风险判断和建议都必须能在下面【证据】里找到出处",
        ):
            self.assertIn(must, prompt)

    def test_empty_memory_gets_an_explicit_placeholder(self):
        self.assertIn("（暂无文件记忆）", build_system_prompt(None))


class MaterialsTests(unittest.TestCase):
    def test_both_branches_and_feedback_are_included(self):
        materials = build_materials(
            {"extracted_params": "注入压力 300 bar", "feedback": "上一版漏了盖层"},
            BRANCH_OK,
            "个人记忆：偏保守",
        )
        self.assertIn("注入压力 300 bar", materials)
        self.assertIn(BRANCH_OK["geology"], materials)
        self.assertIn(BRANCH_OK["wellbore"], materials)
        self.assertIn("上一版漏了盖层", materials)
        self.assertIn("个人记忆：偏保守", materials)

    def test_temporal_conflict_forces_the_newer_value(self):
        materials = build_materials(
            {
                "extracted_params": "p",
                "temporal_conflicts": [{
                    "message": "A 井注入压力存在冲突",
                    "newer": {"timestamp": "2026-05", "display": "320 bar"},
                }],
            },
            BRANCH_OK,
            "",
        )
        self.assertIn("2026-05", materials)
        self.assertIn("320 bar", materials)
        self.assertIn("禁止取平均", materials)

    def test_degraded_branch_is_named(self):
        self.assertIn("geology", build_degraded_hint(BRANCH_DEGRADED))
        self.assertEqual(build_degraded_hint(BRANCH_OK), "")


class FinalizeTests(unittest.TestCase):
    def test_density_interval_is_backfilled_when_missing(self):
        text, _ = finalize_report(
            "正文。",
            density_result={"mean": 1.35, "p95": 1.42, "text": "密度 1.35（1.28–1.42）"},
        )
        self.assertIn("1.42", text)
        self.assertIn("90%", text)

    def test_density_interval_is_not_duplicated(self):
        text, _ = finalize_report(
            "正文里已经写了 1.42。",
            density_result={"mean": 1.35, "p95": 1.42, "text": "x"},
        )
        self.assertEqual(text, "正文里已经写了 1.42。")

    def test_chart_reference_is_appended_once(self):
        chart = {"status": "ok", "rel_path": "charts/a.png"}
        once, _ = finalize_report("正文。", chart_result=chart, chart_ok=True)
        twice, _ = finalize_report(once, chart_result=chart, chart_ok=True)
        self.assertEqual(once.count("charts/a.png"), 1)
        self.assertEqual(twice.count("charts/a.png"), 1)

    def test_degradation_note_is_appended(self):
        text, _ = finalize_report("正文。", branch=BRANCH_DEGRADED)
        self.assertIn("人工复核", text)

    def test_unsupported_percentages_are_stripped(self):
        text, removed = finalize_report(
            "断层活化概率 12%。盖层完整。", probability_source="无",
        )
        self.assertNotIn("12%", text)
        self.assertEqual(removed, ["概率 12%"])
        self.assertIn("盖层完整", text)


class SupportTests(unittest.TestCase):
    def test_tool_json_fence_is_removed(self):
        text = '正文\n```json\n{"tool": "x"}\n```\n结尾'
        self.assertNotIn("```", strip_tool_json(text))
        self.assertIn("结尾", strip_tool_json(text))

    def test_pressure_and_temperature_extraction(self):
        p, t = extract_pressure_temperature("注入压力 250 bar；温度 80℃")
        self.assertEqual((p, t), (250.0, 80.0))
        self.assertEqual(extract_pressure_temperature("无参数"), (None, None))

    def test_depth_extraction_rejects_implausible_values(self):
        self.assertEqual(extract_depth("埋深 2500 m"), 2500.0)
        self.assertIsNone(extract_depth("埋深 3 m"))
        self.assertIsNone(extract_depth("没有深度"))

    def test_uq_payload_computes_cv_and_keeps_samples(self):
        payload = uq_payload({
            "density": {"mean": 2.0, "std": 0.2, "p5": 1.7, "p95": 2.3,
                        "samples": [2.0, 2.1]},
        })
        self.assertAlmostEqual(payload["density"]["cv"], 0.1)
        self.assertEqual(payload["density"]["samples"], [2.0, 2.1])

    def test_chart_payload_drops_failed_results(self):
        self.assertEqual(chart_payload({"status": "error"}), {})
        kept = chart_payload({"status": "ok", "rel_path": "a.png", "junk": 1})
        self.assertEqual(kept["rel_path"], "a.png")
        self.assertNotIn("junk", kept)

    def test_uq_payload_carries_the_plan_block(self):
        """策略建议必须原样透传：前端"推荐方案"卡片只读这一处。"""
        plan = {"status": "ok", "text": "推荐 2 口井", "recommended": {"n_wells": 2}}
        payload = uq_payload({"plan": plan})
        self.assertEqual(payload["plan"], plan)

    def test_plan_payload_slims_risk_but_keeps_evidence(self):
        """
        裁的是重复，不是依据：``notes`` / ``provenance`` / ``frontier`` 必须留。

        这三样是"读者能不能判断该不该信"的全部信息。把它们裁掉，面板就只剩
        一句"推荐 2 口井、316 bar"，与一句编出来的建议没有区别。
        """
        full = {
            "status": "ok",
            "text": "x",
            "notes": ["n1"],
            "provenance": {"fracture_ratio": "1.5×"},
            "frontier": [{"pressure_bar": 300.0, "p_leak": 0.1}],
            "physics": {"hydrostatic_bar": 252.6, "injectivity_m3_per_day_per_bar": 524.2},
            "limit": {"pressure_bar": 352.9},
            "current": {"pressure_bar": 380.0, "rate_mt_per_year": 18.9,
                        "risk": {"hits": 503, "n": 1000, "p": 0.503,
                                 "p_ci": [0.47, 0.53], "p_upper": 0.53,
                                 "decision": {"level": "alert", "notes": ["x"]}}},
            "well_options": [{"n_wells": 2, "risk": {"p": 0.036},
                              "risk_at_needed": {"p": 0.0}}],
            "recommended": {"n_wells": 2, "pressure_bar": 316.3,
                            "risk": {"p": 0.0, "decision": {"level": "ok"}}},
            "verification": {"p": 0.0, "hits": 0, "n": 1000, "seed": 7961,
                             "agrees": True},
            "headroom": {"fraction": 0.57},
        }
        slim = plan_payload(full)
        self.assertEqual(slim["notes"], ["n1"])
        self.assertEqual(slim["provenance"]["fracture_ratio"], "1.5×")
        self.assertEqual(len(slim["frontier"]), 1)
        # 嵌套的 decision 不进网络：每个方案带一份会把体积放大数倍
        self.assertNotIn("decision", slim["current"]["risk"])
        self.assertNotIn("decision", slim["recommended"]["risk"])
        self.assertEqual(slim["current"]["risk"]["p"], 0.503)
        self.assertEqual(slim["verification"]["agrees"], True)

    def test_plan_payload_of_empty_is_empty(self):
        self.assertEqual(plan_payload(None), {})
        self.assertEqual(plan_payload({}), {})


if __name__ == "__main__":
    unittest.main()
