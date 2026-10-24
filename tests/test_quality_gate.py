"""质量闸门：解析失败、重试用尽、物理否决三条路径都必须是"失败"而不是"通过"。

这些测试针对的是一个真实的坑：旧实现在 ``retry_count >= 2`` 时"强制放行"，
并且把"评审输出解析不出来"兜底成 10 分。两种行为都会让 API 返回 200、
报告进历史记忆，而调用方看到的只有"流程结束" —— 失败被伪装成了成功。
"""
from __future__ import annotations

import unittest
from unittest.mock import patch

# 必须在 import nodes.* 之前：评审节点经 utils.tokens 连带加载 2.3 GB 的 reranker，
# 桩掉叶子模块后，这个测试在只装了 langgraph + openai 的机器上也能跑（见 CI 的
# unit-graph job）。判定逻辑本身没有任何桩。
from tests import _graph_stubs

_graph_stubs.install_leaf_stubs()

from core.gates import (  # noqa: E402
    MAX_REPORT_REWRITES,
    QUALITY_THRESHOLD,
    STATUS_BLOCKED_PHYSICS,
    STATUS_FAILED,
    STATUS_JUDGE_UNPARSEABLE,
    STATUS_PASSED,
    build_gate,
    is_failed,
    next_action,
)
from nodes.conditions import check_quality
from nodes.evaluate_node import _parse_score, evaluate_report
from nodes.gate_node import mark_failed


def _response(content: str):
    class _Message:
        def __init__(self):
            self.content = content

    class _Choice:
        def __init__(self):
            self.message = _Message()

    class _Response:
        def __init__(self):
            self.choices = [_Choice()]

    return _Response()


def _run_evaluate(state, contents):
    """按顺序返回 ``contents`` 里的内容，模拟评审的多次输出。"""
    queue = list(contents)

    def fake_chat(_stage, messages, **_kwargs):
        return _response(queue.pop(0) if queue else "")

    with patch(
        "nodes.evaluate_node.get_llm",
        return_value=(None, "local:qwen2.5", "local", None),
    ), patch("nodes.evaluate_node.llm_chat", side_effect=fake_chat):
        return evaluate_report(state)


BASE_STATE = {
    "user_input": "注入压力 250 bar，埋深 2500 m，评估风险",
    "extracted_params": "注入压力 250 bar；埋深 2500 m",
    "final_report": "注入压力低于破裂压力，风险可控。",
    "retrieved_knowledge": "证据",
    "extracted_params_note": "",
}


class ParseScoreToleranceTests(unittest.TestCase):
    def test_common_formats_are_parsed(self):
        for text, expected in [
            ("分数：8\n评语：ok", 8.0),
            ("评分: 7.5/10", 7.5),
            ("Score: 6", 6.0),
            ("9分", 9.0),
            ("8/10", 8.0),
        ]:
            with self.subTest(text=text):
                parsed = _parse_score(text)
                self.assertIsNotNone(parsed, msg=text)
                self.assertEqual(parsed[0], expected)

    def test_garbage_is_not_parsed(self):
        for text in ["", "   ", "无法评价", "分数：很高", "分数：88"]:
            with self.subTest(text=text):
                self.assertIsNone(_parse_score(text))


class JudgeUnparseableTests(unittest.TestCase):
    def test_two_unparseable_outputs_block_the_report(self):
        result = _run_evaluate(dict(BASE_STATE), ["嗯", "还是不行"])
        gate = result["quality_gate"]
        self.assertEqual(gate["status"], STATUS_JUDGE_UNPARSEABLE)
        self.assertEqual(result["score"], 0.0)
        self.assertNotEqual(gate["status"], STATUS_PASSED)
        self.assertEqual(next_action(gate, retry_count=0), "fail")
        self.assertTrue(is_failed(gate))

    def test_one_retry_is_enough_when_the_judge_recovers(self):
        result = _run_evaluate(dict(BASE_STATE), ["嗯", "分数：9\n评语：结构完整"])
        self.assertEqual(result["quality_gate"]["status"], STATUS_PASSED)
        self.assertEqual(result["score"], 9.0)
        # 两次调用都要记账，否则重试的 token 会凭空消失。
        self.assertIn("evaluate", result["token_usage"])
        self.assertIn("evaluate_retry", result["token_usage"])


class LowScoreExhaustionTests(unittest.TestCase):
    def test_low_score_rewrites_then_fails_never_passes(self):
        gate = build_gate(status="rewrite", score=3.0, reason="3 < 8")
        # 预算内：重写
        self.assertEqual(next_action(gate, retry_count=0), "rewrite")
        self.assertEqual(next_action(gate, retry_count=MAX_REPORT_REWRITES - 1), "rewrite")
        # 预算用尽：失败。旧实现在这里是 "pass"。
        self.assertEqual(next_action(gate, retry_count=MAX_REPORT_REWRITES), "fail")
        self.assertEqual(next_action(gate, retry_count=99), "fail")

    def test_check_quality_reports_fail_after_budget_is_gone(self):
        state = {
            "quality_gate": build_gate(status="rewrite", score=3.0, reason="3 < 8"),
            "retry_count": MAX_REPORT_REWRITES,
        }
        self.assertEqual(check_quality(state), "fail")

    def test_check_quality_still_rewrites_while_budget_remains(self):
        state = {
            "quality_gate": build_gate(status="rewrite", score=3.0, reason="3 < 8"),
            "retry_count": 1,
        }
        self.assertEqual(check_quality(state), "rewrite")

    def test_passing_score_still_passes(self):
        state = {
            "quality_gate": build_gate(status="passed", score=QUALITY_THRESHOLD, reason="ok"),
            "retry_count": 0,
        }
        self.assertEqual(check_quality(state), "pass")

    def test_missing_gate_is_fail_closed(self):
        # 老 checkpoint 续跑 / 节点异常返回空 dict 时，不能把"没有结论"读成"通过"。
        self.assertEqual(check_quality({"retry_count": 0}), "fail")
        self.assertEqual(check_quality({"quality_gate": {}, "retry_count": 0}), "fail")
        self.assertEqual(next_action(None, 0), "fail")


class PhysicsVetoTests(unittest.TestCase):
    """超破裂压力时，模型分再高也不能让它通过。"""

    def test_model_score_cannot_rescue_a_physics_violation(self):
        # 报告坚称"风险可控"，物理规则算出 250 bar > 2500 m 处破裂压力 379 bar?
        # 不 —— 这里换成真的会触发 blocking 的组合：2500 m 处 P_frac≈379 bar，
        # 所以用 400 bar 注入压力（>379）来制造超压。
        state = {
            "user_input": "注入压力 400 bar，埋深 2500 m，评估风险",
            "extracted_params": "注入压力 400 bar；埋深 2500 m",
            "final_report": "注入压力远低于破裂压力，注入顺畅，风险可控。",
            "retrieved_knowledge": "证据",
        }
        # 只提供一次评审输出：若实现退化成"先问模型"，队列会被取空。
        result = _run_evaluate(state, ["分数：10\n评语：写得很好"])
        gate = result["quality_gate"]

        self.assertEqual(gate["status"], STATUS_BLOCKED_PHYSICS)
        self.assertTrue(gate["physics_blocking"])
        self.assertEqual(result["score"], 0.0)
        self.assertNotIn("token_usage", result)   # 物理否决时不调用模型
        for retry_count in (0, 1, 2, 5, 100):
            self.assertNotEqual(
                next_action(gate, retry_count), "pass",
                msg=f"retry_count={retry_count} 时物理否决被放行了",
            )


class MarkFailedTests(unittest.TestCase):
    def test_physics_veto_quarantines_the_body(self):
        draft = "注入压力远低于破裂压力，注入顺畅，风险可控，可通过验收。"
        state = {
            "final_report": draft,
            "quality_gate": build_gate(
                status=STATUS_FAILED,
                score=0.0,
                physics={"status": "violation", "blocking": True},
                reason="注入压力超过破裂压力",
            ),
        }
        update = mark_failed(state)
        self.assertEqual(update["report_status"], STATUS_FAILED)
        self.assertTrue(update["quality_gate"]["quarantined"])
        # 被拒的结论不能以任何形式出现在交付字段里。
        for claim in ("风险可控", "注入顺畅", "可通过验收"):
            self.assertNotIn(claim, update["final_report"])
        self.assertIn("不得作为工程依据", update["final_report"])
        # 原稿必须留档，否则无法排查"模型到底写了什么"。
        self.assertEqual(update["draft_report"], draft)
        self.assertFalse(update["history_saved"])

    def test_quality_failure_keeps_the_body_but_marks_it_failed(self):
        state = {
            "final_report": "正文内容。",
            "quality_gate": build_gate(status="rewrite", score=5.0, reason="5 < 8"),
        }
        update = mark_failed(state)
        self.assertIn("正文内容", update["final_report"])
        self.assertIn("不得作为工程依据", update["final_report"])
        self.assertNotIn("draft_report", update)
        self.assertEqual(update["quality_gate"]["status"], STATUS_FAILED)


if __name__ == "__main__":
    unittest.main()
