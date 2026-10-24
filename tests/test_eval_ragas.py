"""Offline contract tests for the RAGAS evaluation harness."""
from __future__ import annotations

import math
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import eval_ragas
from eval_ragas import (
    DEFAULT_CASE_TIMEOUT,
    DEFAULT_JUDGE_MAX_TOKENS,
    FAITHFULNESS_STATEMENT_INSTRUCTION,
    RELEVANCY_JUDGE_SYSTEM_PROMPT,
    RELEVANCY_QUESTION_INSTRUCTION,
    _build_judge_llms,
    _build_answer_relevancy,
    _calibrate_faithfulness_prompt,
    _calibrate_relevancy_prompt,
    _invoke_with_watchdog,
    append_checkpoint,
    parse_args,
    build_evaluation_record,
    collect_retrieved_contexts,
    finite_score,
    load_golden_set,
    read_checkpoint,
    summarize_scores,
)


class GoldenSetTests(unittest.TestCase):
    def test_contains_ten_unique_source_grounded_questions(self):
        cases = load_golden_set()
        self.assertEqual(len(cases), 10)
        self.assertEqual(len({case["id"] for case in cases}), 10)
        self.assertEqual(len({case["question"] for case in cases}), 10)
        for case in cases:
            self.assertTrue(case["reference"].strip())
            self.assertTrue(case["reference_contexts"])
            self.assertTrue(any("ccus_auto_" in text for text in case["reference_contexts"]))


class EvidenceAndGateTests(unittest.TestCase):
    def test_contexts_come_only_from_agent_evidence_and_are_deduplicated(self):
        state = {
            "geology_evidence": "Fault pressure evidence",
            "wellbore_evidence": "Fault pressure evidence",
            "supplemental_evidence": "- graph fallback fragment",
            "retrieved_knowledge": "Not the final prompt evidence",
        }
        self.assertEqual(
            collect_retrieved_contexts(state),
            ["Fault pressure evidence", "- graph fallback fragment"],
        )

    def test_supplemental_graph_evidence_counts_because_it_reaches_the_prompt(self):
        # 地质分支降级时 aggregate 会补取图谱片段并写进报告 Prompt；
        # 漏掉它会让 RAGAS 把"有依据的报告"判成无依据。
        case = load_golden_set()[0]
        record = build_evaluation_record(
            case,
            {
                "final_report": "A grounded report.",
                "supplemental_evidence": "- Pressure management reduces fault reactivation.",
                "tool_facts": ["CO2 密度 = 650 kg/m3（由物理工具 + 蒙特卡洛计算）"],
            },
        )
        self.assertEqual(record["error"], "")
        self.assertEqual(len(record["retrieved_contexts"]), 2)
        self.assertIn("650 kg/m3", record["retrieved_contexts"][1])

    def test_missing_response_or_context_fails_closed(self):
        case = load_golden_set()[0]
        record = build_evaluation_record(
            case,
            {"final_report": "A report without retrieved evidence."},
        )
        self.assertEqual(record["faithfulness"], 0.0)
        self.assertEqual(record["answer_relevancy"], 0.0)
        self.assertIn("no retrieved evidence", record["error"])

    def test_mean_faithfulness_at_threshold_passes_and_below_fails(self):
        at_threshold = [
            {"faithfulness": 0.8, "answer_relevancy": 0.6},
            {"faithfulness": 0.8, "answer_relevancy": 0.8},
        ]
        below_threshold = [
            {"faithfulness": 0.79, "answer_relevancy": 1.0},
            {"faithfulness": 0.8, "answer_relevancy": 1.0},
        ]
        self.assertTrue(summarize_scores(at_threshold, 0.8)["passed"])
        summary = summarize_scores(below_threshold, 0.8)
        self.assertFalse(summary["passed"])
        self.assertAlmostEqual(summary["faithfulness_mean"], 0.795)

    def test_nan_and_empty_results_are_fail_closed(self):
        self.assertEqual(finite_score(float("nan")), 0.0)
        self.assertEqual(finite_score(math.inf), 0.0)
        self.assertFalse(summarize_scores([], 0.8)["passed"])
        self.assertFalse(
            summarize_scores(
                [{"faithfulness": float("nan"), "answer_relevancy": 0.5}], 0.8
            )["passed"]
        )


class JudgeModelTests(unittest.TestCase):
    def test_judge_model_can_be_pointed_at_a_wider_context_variant(self):
        # 忠实度 prompt 装不下默认 4096 窗口，评判模型必须可独立指定。
        with patch.dict(os.environ, {"RAGAS_JUDGE_MODEL": "qwen2.5-8k"}):
            self.assertEqual(parse_args([]).judge_model, "qwen2.5-8k")
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("RAGAS_JUDGE_MODEL", None)
            self.assertEqual(parse_args(["--judge-model", "custom"]).judge_model, "custom")
            self.assertEqual(parse_args([]).judge_model, "")

    def test_judge_output_cap_defaults_above_the_truncation_point(self):
        # RAGAS 自带 1024，实测 2048 也会让陈述分解输出被截断，
        # instructor 抛 IncompleteOutputException → 该样本静默记 0.0。
        self.assertEqual(DEFAULT_JUDGE_MAX_TOKENS, 4096)
        self.assertEqual(parse_args([]).judge_max_tokens, DEFAULT_JUDGE_MAX_TOKENS)

    def test_only_the_relevancy_judge_gets_the_calibration_prompt(self):
        # 非承诺判定与中文提问的校准只属于「问题重写」这一步；
        # 塞进陈述分解（Faithfulness）只会给 7B 评判增加噪声。
        calls = []

        def fake_factory(**kwargs):
            calls.append(kwargs)
            return f"llm-{len(calls)}"

        faithfulness_llm, relevancy_llm = _build_judge_llms(
            fake_factory, client="client", model="qwen2.5-8k", output_cap=4096
        )

        self.assertEqual((faithfulness_llm, relevancy_llm), ("llm-1", "llm-2"))
        self.assertNotIn("system_prompt", calls[0])
        self.assertEqual(calls[1]["system_prompt"], RELEVANCY_JUDGE_SYSTEM_PROMPT)
        self.assertTrue(all(call["max_tokens"] == 4096 for call in calls))
        self.assertTrue(all(call["model"] == "qwen2.5-8k" for call in calls))

    def test_calibration_prompt_pins_noncommittal_and_chinese_question(self):
        # 没有这两条，本地 7B 评判会把带保留意见的技术报告判成 noncommittal=1
        # （RAGAS 直接乘 0），并且用英文重写中文问题（跨语言余弦只有 0.44）。
        prompt = RELEVANCY_JUDGE_SYSTEM_PROMPT
        self.assertIn("noncommittal", prompt)
        self.assertIn("中文", prompt)

    def test_case_timeout_is_configurable_and_positive_by_default(self):
        # 900s 实测不够：一次 360s 的本地卡顿 + 一次打回重写就能吃满。
        self.assertEqual(parse_args([]).case_timeout, DEFAULT_CASE_TIMEOUT)
        self.assertGreaterEqual(DEFAULT_CASE_TIMEOUT, 1200)
        self.assertEqual(parse_args(["--case-timeout", "60"]).case_timeout, 60)

    def test_honest_abstention_is_not_labelled_noncommittal(self):
        # RAGAS 把整个相关性得分乘 int(not noncommittal)。报告里「该风险未做概率
        # 量化，需现场评估」正是本 harness 强制的防幻觉措辞，默认规则却读成逃避，
        # 10 题里有 4 题因此得 0.000，而生成的问题完全对题。
        self.assertIn("not an evasion", RELEVANCY_QUESTION_INSTRUCTION)
        self.assertIn("not quantified", RELEVANCY_QUESTION_INSTRUCTION)

    def test_calibration_replaces_the_question_generation_prompt(self):
        class FakeMetric:
            question_generation = type("P", (), {"instruction": "original"})()

        metric = FakeMetric()
        _calibrate_relevancy_prompt(metric)

        self.assertIsNot(metric.question_generation.__class__, FakeMetric.question_generation.__class__)
        self.assertEqual(
            metric.question_generation.instruction, RELEVANCY_QUESTION_INSTRUCTION
        )

    def test_statement_decomposition_forbids_repetition(self):
        # 7B 会把同一句密度陈述重复到 6610 字符，撞上输出上限后 JSON 截断，
        # instructor 抛 IncompleteOutputException → 该样本被静默记 0.000。
        self.assertIn("never repeat", FAITHFULNESS_STATEMENT_INSTRUCTION)
        self.assertIn("At most 25 statements", FAITHFULNESS_STATEMENT_INSTRUCTION)

    def test_calibration_replaces_the_statement_generation_prompt(self):
        class FakeMetric:
            statement_generator_prompt = type("P", (), {"instruction": "original"})()

        metric = FakeMetric()
        _calibrate_faithfulness_prompt(metric)

        self.assertEqual(
            metric.statement_generator_prompt.instruction,
            FAITHFULNESS_STATEMENT_INSTRUCTION,
        )

    def test_relevancy_reports_similarity_instead_of_the_noncommittal_flag(self):        # 同一份报告，评判生成的问题完全对题，却仍返回 noncommittal=1
        # （因为报告如实写了「图谱证据未覆盖」）。RAGAS 会把整分乘 0，
        # 于是 4/10 题的相关性是 0.000 —— 量到的不是相关性。
        from types import SimpleNamespace

        class FakeAnswerRelevancy:
            def __init__(self, llm=None, embeddings=None):
                self.llm = llm
                self.embeddings = embeddings

            def calculate_similarity(self, question, generated):
                return [0.5, 0.7]

        metric = _build_answer_relevancy(FakeAnswerRelevancy, "llm", "emb")
        answers = [SimpleNamespace(question="压力累积比例是多少？", noncommittal=1)]

        score = metric._calculate_score(answers, {"user_input": "压力累积达到破裂压力多少？"})
        self.assertAlmostEqual(score, 0.6)

    def test_empty_regenerated_question_still_scores_zero(self):
        from types import SimpleNamespace

        class FakeAnswerRelevancy:
            def __init__(self, llm=None, embeddings=None):
                pass

            def calculate_similarity(self, question, generated):
                raise AssertionError("空问题不应进入相似度计算")

        metric = _build_answer_relevancy(FakeAnswerRelevancy, "llm", "emb")
        answers = [SimpleNamespace(question="", noncommittal=0)]

        self.assertTrue(
            math.isnan(metric._calculate_score(answers, {"user_input": "q"}))
        )


class CheckpointTests(unittest.TestCase):
    """看门狗会 os._exit 掉整个进程，所以每题跑完必须立刻落盘。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "ragas_report.partial.jsonl"

    def tearDown(self):
        self.tmp.cleanup()

    def test_finished_cases_round_trip_and_unknown_ids_are_ignored(self):
        cases = load_golden_set()
        append_checkpoint(self.path, {"case_id": cases[0]["id"], "response": "r1"})
        append_checkpoint(self.path, {"case_id": "not-in-golden-set", "response": "x"})
        append_checkpoint(self.path, {"case_id": cases[1]["id"], "response": "r2"})

        finished = read_checkpoint(self.path, cases)

        self.assertEqual(set(finished), {cases[0]["id"], cases[1]["id"]})
        self.assertEqual(finished[cases[0]["id"]]["response"], "r1")

    def test_a_truncated_last_line_does_not_discard_earlier_cases(self):
        # 进程被 os._exit 打断时，最后一行很可能只写了一半。
        cases = load_golden_set()
        append_checkpoint(self.path, {"case_id": cases[0]["id"], "response": "r1"})
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write('{"case_id": "' + cases[1]["id"] + '", "resp')

        finished = read_checkpoint(self.path, cases)

        self.assertEqual(list(finished), [cases[0]["id"]])

    def test_missing_checkpoint_is_not_an_error(self):
        self.assertEqual(read_checkpoint(self.path, load_golden_set()), {})

    def test_abort_writes_the_cases_that_already_finished(self):
        cases = load_golden_set()
        append_checkpoint(self.path, {"case_id": cases[0]["id"], "response": "r1"})
        report_path = Path(self.tmp.name) / "report.json"

        with patch.object(eval_ragas, "_ABORT_CONTEXT", {
            "checkpoint": self.path,
            "report_path": report_path,
            "threshold": 0.8,
            "cases": cases,
        }), patch.object(eval_ragas.os, "_exit", side_effect=SystemExit(3)):
            with self.assertRaises(SystemExit):
                eval_ragas._abort_hung_case("ccus-09-hang", 900)

        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertFalse(report["passed"])
        self.assertEqual(report["aborted_case"], "ccus-09-hang")
        self.assertEqual([row["case_id"] for row in report["results"]], [cases[0]["id"]])


class WatchdogTests(unittest.TestCase):
    def test_stalled_case_is_aborted_with_a_distinct_exit_code(self):
        # 本地流式生成曾整段挂死：不设看门狗会把 CI runner 永久占住。
        aborted = []

        class _SlowApp:
            def invoke(self, _state, _config):
                time.sleep(0.5)
                return {"final_report": "late"}

        with patch.object(
            __import__("eval_ragas"), "_abort_hung_case",
            side_effect=lambda case_id, timeout: aborted.append((case_id, timeout)),
        ):
            _invoke_with_watchdog(_SlowApp(), {"id": "ccus-01", "question": "q"}, 0.05)

        self.assertEqual(aborted, [("ccus-01", 0.05)])

    def test_fast_case_does_not_trip_the_watchdog(self):
        aborted = []

        class _FastApp:
            def invoke(self, _state, _config):
                return {"final_report": "ok"}

        with patch.object(
            __import__("eval_ragas"), "_abort_hung_case",
            side_effect=lambda case_id, timeout: aborted.append(case_id),
        ):
            state = _invoke_with_watchdog(_FastApp(), {"id": "ccus-01", "question": "q"}, 5)

        self.assertEqual(state["final_report"], "ok")
        self.assertEqual(aborted, [])


class IsolationTests(unittest.TestCase):
    def test_already_imported_config_is_forced_back_to_local(self):
        # 探针脚本会先导入 nodes.*，那时 core.config 已经按 .env 建好了云端客户端。
        # 之后清空 CLOUD_API_KEY 对它无效，复杂度升级仍会打云端并失败。
        import sys
        import types

        fake = types.ModuleType("core.config")
        fake.CLOUD_API_KEY = "sk-from-dotenv"
        fake.cloud_client = object()
        fake.DEFAULT_ROUTING = {"report": "cloud", "extract": "local"}
        fake.MODEL_ROUTING = {"report": "cloud", "extract": "local"}
        fake.build_routing = lambda defaults, env: {
            stage: env.get(f"ROUTE_{stage.upper()}", spec).split(":")[0]
            for stage, spec in defaults.items()
        }
        with patch.dict(sys.modules, {"core.config": fake}), \
             patch.dict(os.environ, {"ROUTE_REPORT": "local:qwen2.5"}):
            eval_ragas._pin_loaded_routing_to_local("qwen2.5")

        self.assertEqual(fake.CLOUD_API_KEY, "")
        self.assertIsNone(fake.cloud_client)
        self.assertEqual(fake.MODEL_ROUTING["report"], "local")

    def test_missing_config_module_is_not_an_error(self):
        import sys

        with patch.dict(sys.modules, {"core.config": None}):
            eval_ragas._pin_loaded_routing_to_local("qwen2.5")


if __name__ == "__main__":
    unittest.main()
