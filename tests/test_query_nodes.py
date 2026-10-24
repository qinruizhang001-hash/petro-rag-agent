"""Offline coverage for query rewriting and Self-RAG retrieval gating."""
from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from core import config as llm_config
from nodes import geology_node, query_nodes, wellbore_node
from nodes.conditions import route_after_retrieval_grade


def _response(text: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        usage=SimpleNamespace(prompt_tokens=20, completion_tokens=15),
    )


def _state(**over):
    state = {
        "user_input": "这口井打太深了，封存层会不会漏？",
        "extracted_params": "Well A; injection pressure 250 bar; depth 2000 m",
        "geology_query": "CCUS caprock integrity leakage injection pressure",
        "wellbore_query": "CO2 well integrity casing corrosion annular pressure",
        "geology_evidence": "Caprock breakthrough pressure is discussed in source A.",
        "wellbore_evidence": "Casing corrosion is discussed in source B.",
        "geology_candidates": [{"text": "Caprock breakthrough pressure is discussed in source A."}],
        "wellbore_candidates": [{"text": "Casing corrosion is discussed in source B."}],
        "retrieval_retry_count": 0,
        "retrieval_strategy": "primary",
        "token_usage": {},
        "model_routing": {},
    }
    state.update(over)
    return state


def _grade_reply(score=0.9, relevant=True, reason="direct branch evidence"):
    """单分支评分返回：Grader 每次只评判一个分支。"""
    return json.dumps({"score": score, "relevant": relevant, "reason": reason})


def _branch_grader(geo=None, well=None):
    """按 prompt 里的 Branch 字段返回对应的单域评分。"""
    geo = geo or _grade_reply()
    well = well or _grade_reply()

    def _call(_fn, _stage, **kwargs):
        content = kwargs["messages"][0]["content"]
        return _response(geo if "Branch: geology" in content else well)

    return _call


class TestRewriteNode(unittest.TestCase):
    def test_rewrites_to_branch_specific_english_queries_and_records_usage(self):
        payload = json.dumps({
            "geology_query": "CCUS fault reactivation caprock integrity at injection pressure",
            "wellbore_query": "CO2 casing corrosion cement sheath annular pressure",
        })
        with patch.object(query_nodes, "get_llm", return_value=(None, "qwen2.5", "local", "")), \
             patch.object(query_nodes, "call_with_retry", return_value=_response(payload)) as call, \
             patch.object(query_nodes, "record_llm_usage", return_value={"token_usage": {"query_rewrite": {"total_tokens": 35}}}) as record:
            out = query_nodes.rewrite_node(_state())

        self.assertEqual(out["query_rewrite_status"]["status"], "ok")
        self.assertIn("caprock integrity", out["geology_query"])
        self.assertIn("casing corrosion", out["wellbore_query"])
        self.assertEqual(call.call_args.args[1], "query_rewrite")
        self.assertEqual(call.call_args.kwargs["mode"], "local")
        record.assert_called_once()

    def test_generic_model_output_gets_domain_vocabulary_anchors(self):
        payload = json.dumps({
            "geology_query": "Geological storage risk at 250 bar",
            "wellbore_query": "Risks for a deep injection well",
        })
        with patch.object(query_nodes, "get_llm", return_value=(None, "qwen2.5", "local", "")), \
             patch.object(query_nodes, "call_with_retry", return_value=_response(payload)), \
             patch.object(query_nodes, "record_llm_usage", return_value={}):
            out = query_nodes.rewrite_node(_state())
        self.assertIn("caprock integrity", out["geology_query"])
        self.assertIn("casing corrosion", out["wellbore_query"])

    def test_rewrite_and_grade_are_pinned_to_local_even_if_route_is_overridden(self):
        with patch.dict(llm_config.MODEL_ROUTING, {"grade": "cloud", "query_rewrite": "cloud"}), \
             patch.object(llm_config, "cloud_client", object()):
            client, _model, tier, _reason = llm_config.get_llm("grade", mode="cloud")
        self.assertIs(client, llm_config.local_client)
        self.assertEqual(tier, "local")

    def test_invalid_rewrite_uses_technical_english_fallback(self):
        with patch.object(query_nodes, "get_llm", return_value=(None, "qwen2.5", "local", "")), \
             patch.object(query_nodes, "call_with_retry", return_value=_response("not json")), \
             patch.object(query_nodes, "record_llm_usage", return_value={}):
            out = query_nodes.rewrite_node(_state())

        self.assertEqual(out["query_rewrite_status"]["status"], "fallback")
        self.assertIn("caprock integrity", out["geology_query"])
        self.assertIn("casing corrosion", out["wellbore_query"])


class TestExternalEvidenceGrade(unittest.TestCase):
    def test_external_evidence_uses_local_grader_and_records_stage(self):
        response = _response(_grade_reply(0.9))
        with patch.object(query_nodes, "get_llm", return_value=(None, "qwen2.5", "local", "")) as get_llm, \
             patch.object(query_nodes, "call_with_retry", return_value=response) as call, \
             patch.object(query_nodes, "record_llm_usage", return_value={"token_usage": {"grade_web": {"total_tokens": 35}}}) as record:
            out = query_nodes.grade_external_evidence(_state(), "Relevant caprock leakage evidence.")
        get_llm.assert_called_once_with("grade_web", mode="local")
        self.assertEqual(call.call_args.args[1], "grade_web")
        self.assertTrue(out["grade"]["accepted"])
        record.assert_called_once()

    def test_external_grader_failure_rejects_evidence(self):
        with patch.object(query_nodes, "get_llm", side_effect=RuntimeError("local unavailable")):
            out = query_nodes.grade_external_evidence(_state(), "some retrieved result")
        self.assertFalse(out["grade"]["accepted"])
        self.assertEqual(out["grade"]["status"], "unavailable")


class TestGradeNode(unittest.TestCase):
    def _run_grade(self, state, geo=None, well=None):
        with patch.object(query_nodes, "get_llm", return_value=(None, "qwen2.5", "local", "")), \
             patch.object(query_nodes, "call_with_retry", side_effect=_branch_grader(geo, well)) as call, \
             patch.object(query_nodes, "record_llm_usage", return_value={"token_usage": {"grade": {"total_tokens": 35}}}):
            out = query_nodes.grade_node(state)
        self.assertEqual(call.call_args.args[1], "grade")
        self.assertEqual(call.call_args.kwargs["mode"], "local")
        return out

    def test_high_relevance_passes_and_merges_evidence(self):
        out = self._run_grade(_state())
        self.assertEqual(out["retrieval_grade"]["status"], "passed")
        self.assertTrue(out["retrieval_grade"]["geology"]["accepted"])
        self.assertTrue(out["retrieval_grade"]["wellbore"]["accepted"])
        self.assertIn("Caprock breakthrough", out["retrieved_knowledge"])
        self.assertIn("Casing corrosion", out["retrieved_knowledge"])
        self.assertEqual(out["retrieval_retry_count"], 0)

    def test_low_relevance_requests_one_expanded_retry(self):
        out = self._run_grade(_state(), geo=_grade_reply(0.2))
        self.assertEqual(out["retrieval_grade"]["status"], "retrying")
        self.assertEqual(out["retrieval_retry_count"], 1)
        self.assertEqual(out["retrieval_strategy"], "expanded")
        self.assertEqual(route_after_retrieval_grade(out), ["geology_retrieve", "wellbore_retrieve"])

    def test_low_relevance_after_retry_withholds_only_rejected_evidence(self):
        state = _state(retrieval_retry_count=1, retrieval_strategy="expanded")
        out = self._run_grade(state, geo=_grade_reply(0.1))
        self.assertEqual(out["retrieval_grade"]["status"], "exhausted")
        self.assertEqual(out["geology_evidence"], "")
        self.assertIn("Casing corrosion", out["wellbore_evidence"])
        self.assertEqual(route_after_retrieval_grade(out), ["geology_generate", "wellbore_generate"])

    def test_empty_evidence_cannot_be_accepted_even_if_grader_claims_relevant(self):
        state = _state(geology_candidates=[], geology_evidence="")
        out = self._run_grade(state)
        self.assertFalse(out["retrieval_grade"]["geology"]["accepted"])
        self.assertEqual(out["retrieval_grade"]["geology"]["score"], 0.0)

    def test_grader_failure_keeps_evidence_but_marks_status_unavailable(self):
        with patch.object(query_nodes, "get_llm", return_value=(None, "qwen2.5", "local", "")), \
             patch.object(query_nodes, "call_with_retry", side_effect=RuntimeError("Ollama offline")):
            out = query_nodes.grade_node(_state())
        self.assertEqual(out["retrieval_grade"]["status"], "unavailable")
        self.assertFalse(out["retrieval_grade"]["geology"]["accepted"])
        self.assertFalse(out["retrieval_grade"]["wellbore"]["accepted"])
        # 一条证据都不放行时报告只能凭记忆写，那比"相关性未知"更危险，所以兜底保留一路。
        self.assertEqual(out["retrieval_grade"]["fallback_domain"], "geology")
        self.assertIn("Caprock breakthrough", out["geology_evidence"])
        self.assertEqual(out["wellbore_evidence"], "")

    def test_both_branches_rejected_keeps_the_better_scored_evidence(self):
        # 兜底挑分数高的那一路：井筒 0.35 > 地质 0.10。
        state = _state(retrieval_retry_count=1, retrieval_strategy="expanded")
        out = self._run_grade(
            state, geo=_grade_reply(0.1), well=_grade_reply(0.35, relevant=False)
        )
        self.assertEqual(out["retrieval_grade"]["status"], "exhausted")
        self.assertEqual(out["retrieval_grade"]["fallback_domain"], "wellbore")
        self.assertEqual(out["geology_evidence"], "")
        self.assertIn("Casing corrosion", out["wellbore_evidence"])
        self.assertIn("Casing corrosion", out["retrieved_knowledge"])

    def test_grader_prompt_stays_within_local_context_budget(self):
        # 本地模型只有 4096 token 窗口；超长证据曾导致整条证据链被判定不可用。
        oversized = [{"text": "x" * 5000} for _ in range(12)]
        state = _state(geology_candidates=oversized, wellbore_candidates=oversized)
        prompts = []

        def fake_call(_fn, _stage, **kwargs):
            prompts.append(kwargs["messages"][0]["content"])
            return _response(_grade_reply())

        with patch.object(query_nodes, "get_llm", return_value=(None, "qwen2.5", "local", "")),              patch.object(query_nodes, "call_with_retry", side_effect=fake_call):
            query_nodes.grade_node(state)

        # 每个分支各判一次，两次都必须在预算内。
        self.assertEqual(len(prompts), 2)
        for prompt in prompts:
            self.assertLess(len(prompt), 3600)
        trimmed = query_nodes._candidate_texts(oversized)
        self.assertLessEqual(len(trimmed), query_nodes.GRADE_MAX_SNIPPETS)
        self.assertTrue(trimmed)
        self.assertLessEqual(sum(len(t) for t in trimmed), query_nodes.GRADE_DOMAIN_CHARS)
        self.assertLessEqual(max(len(t) for t in trimmed), query_nodes.GRADE_SNIPPET_CHARS)

    def test_grader_retries_once_with_a_shorter_prompt_before_giving_up(self):
        prompts = []
        error = RuntimeError(
            "Error code: 400 - request exceeds the available context size (4096 tokens)"
        )

        def fake_call(_fn, _stage, **kwargs):
            prompts.append(kwargs["messages"][0]["content"])
            if len(prompts) == 1:
                raise error
            return _response(_grade_reply())

        state = _state(
            geology_candidates=[{"text": "y" * 4000}],
            wellbore_candidates=[{"text": "y" * 4000}],
        )
        with patch.object(query_nodes, "get_llm", return_value=(None, "qwen2.5", "local", "")),              patch.object(query_nodes, "call_with_retry", side_effect=fake_call):
            out = query_nodes.grade_node(state)

        self.assertEqual(len(prompts), 3)          # geology 重试一次 + wellbore 一次
        self.assertLess(len(prompts[1]), len(prompts[0]))
        self.assertEqual(out["retrieval_grade"]["status"], "passed")
        self.assertTrue(out["geology_evidence"])
        self.assertNotIn("errors", out["retrieval_grade"])

    def test_one_branch_grader_failure_does_not_discard_the_other_branch(self):
        # 井筒 Grader 挂了，地质证据仍然可用：单分支 fail-closed，而不是全盘拒绝。
        def fake_call(_fn, _stage, **kwargs):
            if "Branch: wellbore" in kwargs["messages"][0]["content"]:
                raise RuntimeError("Ollama timeout")
            return _response(_grade_reply(0.9))

        state = _state(retrieval_retry_count=1, retrieval_strategy="expanded")
        with patch.object(query_nodes, "get_llm", return_value=(None, "qwen2.5", "local", "")),              patch.object(query_nodes, "call_with_retry", side_effect=fake_call):
            out = query_nodes.grade_node(state)

        self.assertIn("Caprock breakthrough", out["geology_evidence"])
        self.assertEqual(out["wellbore_evidence"], "")
        self.assertTrue(out["retrieval_grade"]["wellbore"]["unavailable"])
        self.assertIn("wellbore", out["retrieval_grade"]["errors"])


class TestRetrievalGating(unittest.TestCase):
    def test_geology_retry_switches_to_mix_mode_and_uses_expanded_query(self):
        state = _state(retrieval_strategy="expanded")
        mocked = {
            "retrieved_knowledge": "expanded evidence",
            "top_candidates": [{"text": "expanded evidence", "rerank_score": 0.7}],
            "token_usage": {},
        }
        with patch.object(geology_node, "retrieve_knowledge", return_value=mocked) as retrieve:
            out = geology_node.retrieve_geology(state)
        self.assertEqual(retrieve.call_args.kwargs["mode"], "mix")
        self.assertIn(state["user_input"], retrieve.call_args.kwargs["query_override"])
        self.assertEqual(out["geology_candidates"], mocked["top_candidates"])
        self.assertEqual(out["geology_status"]["retrieval_strategy"], "expanded")

    def test_wellbore_retry_expands_candidate_pool_and_query(self):
        state = _state(retrieval_strategy="expanded")
        with patch.object(wellbore_node, "retrieve_documents", return_value=["doc A", "doc B"]) as retrieve:
            out = wellbore_node.retrieve_wellbore(state)
        self.assertEqual(retrieve.call_args.kwargs["k"], 8)
        self.assertIn(state["user_input"], retrieve.call_args.args[0])
        self.assertEqual(len(out["wellbore_candidates"]), 2)
        self.assertEqual(out["wellbore_status"]["retrieval_strategy"], "expanded")

    def test_rejected_geology_evidence_never_reaches_generator_llm(self):
        state = _state(retrieval_grade={"geology": {"accepted": False, "reason": "off-topic"}})
        with patch.object(geology_node, "llm_chat") as chat:
            out = geology_node.generate_geology(state)
        chat.assert_not_called()
        self.assertEqual(out["geology_report"], "")
        self.assertIn("已阻止无依据生成", out["geology_status"]["grading_error"])

    def test_grading_rejection_preserves_original_retrieval_error(self):
        state = _state(
            geology_status={"ok": False, "error": "TimeoutError: graph query timed out"},
            retrieval_grade={"geology": {"accepted": False, "reason": "no evidence"}},
        )
        with patch.object(geology_node, "llm_chat") as chat:
            out = geology_node.generate_geology(state)
        chat.assert_not_called()
        self.assertEqual(out["geology_status"]["error"], "TimeoutError: graph query timed out")
        self.assertIn("已阻止无依据生成", out["geology_status"]["grading_error"])

    def test_rejected_wellbore_evidence_never_reaches_generator_llm(self):
        state = _state(retrieval_grade={"wellbore": {"accepted": False, "reason": "off-topic"}})
        with patch.object(wellbore_node, "llm_chat") as chat:
            out = wellbore_node.generate_wellbore(state)
        chat.assert_not_called()
        self.assertEqual(out["wellbore_report"], "")
        self.assertIn("已阻止无依据生成", out["wellbore_status"]["error"])

    def _generate_geology_with_web_result(self, accepted):
        state = _state(
            geology_evidence="Short local evidence.",
            geology_status={"ok": True, "elapsed_s": 0.1},
            retrieval_grade={"geology": {"accepted": True, "score": 0.8}},
        )
        captured = {}

        def fake_generate(_func, *_args, **kwargs):
            captured["prompt"] = kwargs["messages"][0]["content"]
            return _response("Grounded geology summary.")

        grade = {"accepted": accepted, "score": 0.8 if accepted else 0.1,
                 "status": "accepted" if accepted else "rejected",
                 "reason": "test"}
        with patch.object(geology_node, "should_search", return_value="web_search"), \
             patch.object(geology_node, "web_search_save", return_value={"web_content": "ungraded-web-noise"}), \
             patch.object(geology_node, "grade_external_evidence", return_value={"grade": grade}), \
             patch.object(geology_node, "get_llm", return_value=(None, "qwen2.5", "local", "")), \
             patch.object(geology_node, "call_with_retry", side_effect=fake_generate), \
             patch.object(geology_node, "record_llm_usage", return_value={}):
            out = geology_node.generate_geology(state)
        return out, captured["prompt"]

    def test_rejected_web_evidence_is_not_injected_into_generator(self):
        out, prompt = self._generate_geology_with_web_result(accepted=False)
        self.assertNotIn("ungraded-web-noise", prompt)
        self.assertNotIn("ungraded-web-noise", out["geology_evidence"])
        self.assertFalse(out["geology_status"]["web_evidence_grade"]["accepted"])

    def test_accepted_web_evidence_is_graded_before_generator_injection(self):
        out, prompt = self._generate_geology_with_web_result(accepted=True)
        self.assertIn("ungraded-web-noise", prompt)
        self.assertIn("ungraded-web-noise", out["geology_evidence"])
        self.assertTrue(out["geology_status"]["web_evidence_grade"]["accepted"])


if __name__ == "__main__":
    unittest.main()
