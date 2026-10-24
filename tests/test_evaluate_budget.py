"""评审 prompt 的输入预算。

本地默认模型（qwen2.5，4096 窗口）实测被评审 prompt 撑爆过：某次
evaluate 请求 4097 tokens → HTTP 400，整个 case 直接判失败，RAGAS 只能记 0。
这些测试钉住两件事：预算函数本身，以及「拼出来的 prompt 真的被压住了」。
"""
from __future__ import annotations

import unittest
from unittest.mock import patch

from nodes.evaluate_node import (
    EVAL_EVIDENCE_CHARS,
    EVAL_REPORT_CHARS,
    UNSUPPORTED_PROBABILITY_SCORE_CAP,
    _TRUNCATION_MARK,
    _budget,
    evaluate_report,
    strip_unsupported_probabilities,
    unsupported_probabilities,
)


class BudgetHelperTests(unittest.TestCase):
    def test_text_within_budget_is_untouched(self):
        self.assertEqual(_budget("证据", EVAL_EVIDENCE_CHARS), "证据")

    def test_over_budget_text_keeps_the_head_and_says_so(self):
        text = "报" * (EVAL_REPORT_CHARS + 500)
        result = _budget(text, EVAL_REPORT_CHARS)
        self.assertTrue(result.endswith(_TRUNCATION_MARK))
        self.assertEqual(len(result), EVAL_REPORT_CHARS + len(_TRUNCATION_MARK))

    def test_missing_text_becomes_empty_string(self):
        self.assertEqual(_budget(None, 10), "")


class EvaluatePromptBudgetTests(unittest.TestCase):
    def _run(self, state):
        captured = {}

        class _Message:
            content = "分数：8\n评语：结构清晰，证据充分。"

        class _Choice:
            message = _Message()

        class _Response:
            choices = [_Choice()]

        def fake_chat(_stage, messages, **_kwargs):
            captured["prompt"] = messages[0]["content"]
            return _Response()

        with patch(
            "nodes.evaluate_node.get_llm",
            return_value=(None, "local:qwen2.5", "local", None),
        ), patch("nodes.evaluate_node.llm_chat", side_effect=fake_chat):
            result = evaluate_report(state)
        return captured["prompt"], result

    def test_oversized_report_and_evidence_cannot_overflow_the_local_window(self):
        # 8 000 字报告 + 6 000 字证据是真实出现过的量级（案例 2 因此 400）。
        prompt, result = self._run(
            {
                "user_input": "CO2 注入压力管理",
                "final_report": "报告正文" * 2000,
                "retrieved_knowledge": "证据片段" * 1500,
                "extracted_params": "",
            }
        )
        self.assertEqual(result["score"], 8.0)
        # 预算后 prompt 上限 = 固定模板 + 两段预算，远低于 4096 token。
        self.assertLess(
            len(prompt),
            600 + EVAL_EVIDENCE_CHARS + EVAL_REPORT_CHARS + 200,
        )
        self.assertEqual(prompt.count(_TRUNCATION_MARK), 2)

    def test_normal_report_is_not_truncated(self):
        report = "正常长度的报告。" * 100  # 800 字，低于 3 000 字预算
        prompt, _ = self._run(
            {
                "user_input": "CO2 注入压力管理",
                "final_report": report,
                "retrieved_knowledge": "证据片段",
                "extracted_params": "",
            }
        )
        self.assertNotIn(_TRUNCATION_MARK, prompt)
        self.assertIn(report, prompt)


class UnsupportedProbabilityTests(unittest.TestCase):
    """RAGAS 实测：报告里 5 个"XX 风险概率 N%"全部无处可查，忠实度掉到 0.095。"""

    def test_probability_absent_from_evidence_is_flagged(self):
        claims = unsupported_probabilities(
            "断层活化风险概率为 12%（100 次蒙特卡洛模拟）。",
            "CO2 密度 = 0.0108 kg/m³（100 次蒙特卡洛模拟）",
        )
        self.assertEqual(claims, ["概率为 12%"])

    def test_probability_present_in_tool_output_is_accepted(self):
        evidence = "注入风险概率分布：注入可行 70%，压裂风险 30%"
        self.assertEqual(
            unsupported_probabilities("压裂风险概率 30%。", evidence), []
        )

    def test_non_probability_percentages_are_ignored(self):
        self.assertEqual(
            unsupported_probabilities("含水饱和度 12%，孔隙度 8%。", ""), []
        )

    def test_fabricated_ratios_are_caught_too(self):
        # 只拦「概率」时，报告改写成「压力累积达到破裂压力的比例为 95%」就绕过守卫，
        # 而 95% 同样是凭空来的（实测该题忠实度 0.545）。
        claims = unsupported_probabilities(
            "压力累积达到破裂压力的比例为 95%。", "地层压力曲线结论：P_frac=303 bar"
        )
        self.assertEqual(claims, ["比例为 95%"])
        cleaned, _ = strip_unsupported_probabilities("比例达到 95% 时风险上升。", "")
        self.assertNotIn("95%", cleaned)
        # 整句删掉，不留下「未做量化」这类同样无依据的替换句。
        self.assertEqual(cleaned, "")


class StripUnsupportedProbabilityTests(unittest.TestCase):
    """扣分只能推动重写；重写预算用尽后，得有人把数字删掉。"""

    def test_fabricated_percentages_are_removed_with_their_sentences(self):
        report = (
            "断层活化风险概率约为 20%。盖层破裂风险概率约为 5%。"
            "注入风险概率 30%。"
        )
        cleaned, removed = strip_unsupported_probabilities(
            report, "注入风险概率分布：注入可行 70%，压裂风险 30%"
        )
        self.assertEqual(removed, ["概率约为 20%", "概率约为 5%"])
        self.assertNotIn("20%", cleaned)
        self.assertNotIn("5%", cleaned)
        # 工具真的算过的 30% 必须原样保留。
        self.assertIn("注入风险概率 30%", cleaned)
        # 不能留下替换句：它本身也是证据支持不了的陈述，会被忠实度判为无依据。
        self.assertNotIn("该风险未做概率量化", cleaned)

    def test_report_without_claims_is_returned_untouched(self):
        report = "含水饱和度 12%，CO2 密度 0.0108 kg/m³。"
        cleaned, removed = strip_unsupported_probabilities(report, "")
        self.assertEqual(cleaned, report)
        self.assertEqual(removed, [])


class ProbabilityScoreCapTests(unittest.TestCase):
    def _run(self, report, evidence="CO2 密度 = 0.0108 kg/m³"):
        class _Message:
            content = "分数：9\n评语：结构清晰。"

        class _Choice:
            message = _Message()

        class _Response:
            choices = [_Choice()]

        with patch(
            "nodes.evaluate_node.get_llm",
            return_value=(None, "local:qwen2.5", "local", None),
        ), patch("nodes.evaluate_node.llm_chat", return_value=_Response()):
            return evaluate_report(
                {
                    "user_input": "q",
                    "final_report": report,
                    "retrieved_knowledge": evidence,
                    "tool_facts": [evidence],
                    "extracted_params": "",
                }
            )

    def test_fabricated_probabilities_cap_the_score_and_ask_for_a_rewrite(self):
        result = self._run("断层活化风险概率为 12%（100 次蒙特卡洛模拟）。")
        self.assertEqual(result["score"], UNSUPPORTED_PROBABILITY_SCORE_CAP)
        self.assertEqual(result["unsupported_probabilities"], ["概率为 12%"])
        self.assertIn("概率溯源校验", result["feedback"])

    def test_traceable_probabilities_keep_the_judge_score(self):
        result = self._run(
            "CO2 密度 = 0.0108 kg/m³，据此注入风险概率 30%。",
            evidence="CO2 密度 = 0.0108 kg/m³；注入风险概率 30%",
        )
        self.assertEqual(result["score"], 9.0)
        self.assertEqual(result["unsupported_probabilities"], [])


if __name__ == "__main__":
    unittest.main()
