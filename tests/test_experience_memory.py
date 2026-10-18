"""经验库纯逻辑的离线单测（stdlib only，不需要 Ollama / 图谱 / torch）。

覆盖三组东西，每组都对应一个真实会出错的点：

1. **记录与计数** —— 半行、坏行、重复追加。经验文件是被"每次评估追加一行"
   写出来的，进程被杀就会留半行；读的时候一条坏行不能拖垮整次反思。
2. **触发时机** —— 只看 ``count % 20`` 会让重跑脚本对同一批数据反复反思，
   而"反思重复了"从外面看不出来，只会表现为语义记忆里出现近似重复的规律。
3. **解析** —— 7B 的输出几乎从不是干净的 JSON：带围栏、带散文、confidence
   写成字符串、一次给一个对象。解析器吃掉这些，否则一条规律都提炼不出来，
   而"没产出"和"没触发"在日志里长得一模一样。
"""
import json
import tempfile
import unittest
from pathlib import Path

from utils.experience_memory import (
    BATCH_SIZE,
    MIN_CONFIDENCE,
    append_experience,
    batch_window,
    build_experience,
    count_experiences,
    dedupe_patterns,
    due_batch,
    format_priors,
    parse_reflection,
    parse_verdicts,
    rank_priors,
    read_experiences,
    relevance,
    similarity,
)


def _state(**over):
    base = {
        "user_input": "某CCUS项目注入压力250bar，温度80℃，埋深2500m，请评估风险。",
        "extracted_params": "注入压力 250 bar，温度 80℃，储层埋深 2500 m",
        "final_report": "x" * 900,
        "score": 8.0,
        "geology_status": {"ok": True, "engine": "lightrag", "candidates": 129, "top_score": 0.9985},
        "wellbore_status": {"ok": True},
        "branch_merge": {"degraded": False, "reasons": {}},
        "physics_check": {"status": "pass", "verdict": "safe_window", "blocking": False},
        "retry_count": 0,
        "pinn_prediction": {},
    }
    base.update(over)
    return base


class TestBuildExperience(unittest.TestCase):
    def test_keeps_comparable_fields_and_drops_the_report(self):
        rec = build_experience(_state(), run_id="r1", ts="2026-10-18 10:00:00")
        self.assertEqual(rec["run_id"], "r1")
        self.assertEqual(rec["score"], 8.0)
        self.assertEqual(rec["retrieval"]["candidates"], 129.0)
        self.assertEqual(rec["retrieval"]["top_score"], 0.9985)
        self.assertEqual(rec["report_chars"], 900)
        self.assertNotIn("final_report", rec, "报告全文不该进经验库")

    def test_missing_and_garbage_fields_do_not_raise(self):
        # 这个函数会在异常路径上被调用：它抛异常就把"记录失败"升级成"评估失败"
        rec = build_experience({})
        self.assertIsNone(rec["score"])
        self.assertEqual(rec["report_chars"], 0)
        self.assertEqual(rec["retrieval"]["candidates"], None)
        self.assertFalse(rec["degraded"])

    def test_string_score_is_coerced(self):
        rec = build_experience(_state(score="9.5"))
        self.assertEqual(rec["score"], 9.5)

    def test_long_input_is_truncated(self):
        rec = build_experience(_state(user_input="长" * 5000, extracted_params="参" * 5000))
        self.assertLessEqual(len(rec["user_input"]), 400)
        self.assertLessEqual(len(rec["params"]), 600)


class TestStoreIO(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="exp_io_")
        self.path = Path(self.tmp) / "experiences.jsonl"

    def test_append_returns_running_count(self):
        for i in range(3):
            self.assertEqual(append_experience(self.path, {"i": i}), i + 1)

    def test_bad_lines_are_skipped_not_fatal(self):
        self.path.write_text(
            '{"score": 8}\n'
            "not json at all\n"
            '{"score": 9}\n'
            "\n"
            '{"score": 7}\n',
            encoding="utf-8",
        )
        self.assertEqual(count_experiences(self.path), 4)
        rows = read_experiences(self.path)
        self.assertEqual([r["score"] for r in rows], [8, 9, 7])

    def test_limit_returns_the_most_recent(self):
        for i in range(30):
            append_experience(self.path, {"i": i})
        rows = read_experiences(self.path, limit=20)
        self.assertEqual(len(rows), 20)
        self.assertEqual(rows[0]["i"], 10, "limit 必须取最近的 N 条，不是最早的")

    def test_missing_file_is_empty_not_an_error(self):
        self.assertEqual(count_experiences(self.path), 0)
        self.assertEqual(read_experiences(self.path), [])


class TestDueBatch(unittest.TestCase):
    def test_below_one_batch_never_triggers(self):
        for n in (0, 1, 19):
            self.assertIsNone(due_batch(n, []))

    def test_exactly_one_batch_triggers(self):
        self.assertEqual(due_batch(BATCH_SIZE, []), 1)

    def test_already_reflected_batch_does_not_trigger_again(self):
        # 这就是"重跑脚本不要重复反思"的那条约束
        self.assertIsNone(due_batch(BATCH_SIZE, [1]))

    def test_second_batch_triggers_at_forty(self):
        self.assertEqual(due_batch(BATCH_SIZE * 2, [1]), 2)
        self.assertIsNone(due_batch(BATCH_SIZE * 2, [1, 2]))

    def test_jumped_batches_only_returns_the_latest(self):
        # 上次反思失败、又攒到 45 条：只补最新那批，不回头补做旧批次
        self.assertEqual(due_batch(45, []), 2)

    def test_not_a_multiple_does_not_trigger(self):
        self.assertIsNone(due_batch(21, [1]))


class TestBatchWindow(unittest.TestCase):
    def test_window_is_the_completed_batch_not_the_last_twenty(self):
        # 45 条时"最后 20 条"是 [25,45)，那会跨进第 3 批，被两批各读一次
        self.assertEqual(batch_window(45, 2), (20, 40))
        self.assertEqual(batch_window(20, 1), (0, 20))
        self.assertEqual(batch_window(40, 2), (20, 40))

    def test_window_never_exceeds_available_rows(self):
        start, end = batch_window(20, 1)
        self.assertLessEqual(end, 20)
        self.assertGreaterEqual(start, 0)


class TestParseReflection(unittest.TestCase):
    def test_plain_array(self):
        out = parse_reflection('[{"pattern": "缺埋深时评分偏低", "confidence": 0.9}]')
        self.assertEqual(out, [{"pattern": "缺埋深时评分偏低", "confidence": 0.9}])

    def test_single_object_is_wrapped(self):
        out = parse_reflection('{"pattern": "检索相关度低时结论变短", "confidence": 0.88}')
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["confidence"], 0.88)

    def test_code_fence_and_prose_are_tolerated(self):
        raw = '好的，我分析了这 20 条记录：\n```json\n[{"pattern": "缺埋深时评分偏低", "confidence": 0.9}]\n```\n以上。'
        self.assertEqual(len(parse_reflection(raw)), 1)

    def test_prose_around_bare_json_is_tolerated(self):
        raw = '分析如下 [{"pattern": "缺埋深时评分偏低", "confidence": 0.9}] 完毕'
        self.assertEqual(len(parse_reflection(raw)), 1)

    def test_string_confidence_is_coerced(self):
        out = parse_reflection('[{"pattern": "缺埋深时评分偏低", "confidence": "0.95"}]')
        self.assertEqual(out[0]["confidence"], 0.95)

    def test_out_of_range_confidence_is_dropped(self):
        # 越界说明模型在编，夹紧不如丢弃
        self.assertEqual(parse_reflection('[{"pattern": "缺埋深时评分偏低", "confidence": 1.4}]'), [])
        self.assertEqual(parse_reflection('[{"pattern": "缺埋深时评分偏低", "confidence": -0.2}]'), [])

    def test_missing_confidence_is_dropped(self):
        self.assertEqual(parse_reflection('[{"pattern": "缺埋深时评分偏低"}]'), [])

    def test_too_short_pattern_is_dropped(self):
        self.assertEqual(parse_reflection('[{"pattern": "偏低", "confidence": 0.9}]'), [])

    def test_long_pattern_is_truncated_not_dropped(self):
        out = parse_reflection('[{"pattern": "%s", "confidence": 0.9}]' % ("长" * 400))
        self.assertEqual(len(out), 1)
        self.assertLessEqual(len(out[0]["pattern"]), 161)

    def test_min_confidence_filter(self):
        raw = '[{"pattern": "缺埋深时评分偏低", "confidence": 0.9},' \
              ' {"pattern": "注入压力高时评分偏低", "confidence": 0.5}]'
        self.assertEqual(len(parse_reflection(raw)), 2)
        self.assertEqual(len(parse_reflection(raw, min_confidence=0.8)), 1)

    def test_nested_patterns_key(self):
        raw = '{"patterns": [{"pattern": "缺埋深时评分偏低", "confidence": 0.9}]}'
        self.assertEqual(len(parse_reflection(raw)), 1)

    def test_garbage_returns_empty(self):
        for raw in ("", "模型今天不想输出 JSON", "[]", None):
            self.assertEqual(parse_reflection(raw), [])


class TestParseVerdicts(unittest.TestCase):
    def test_aligned_by_position(self):
        raw = '[{"supported": true, "reason": "全部支持"}, {"supported": false, "reason": "有反例"}]'
        out = parse_verdicts(raw, 2)
        self.assertTrue(out[0]["supported"])
        self.assertFalse(out[1]["supported"])
        self.assertEqual(out[1]["reason"], "有反例")

    def test_extra_verdicts_are_truncated(self):
        raw = '[{"supported": true}, {"supported": false}, {"supported": true}]'
        self.assertEqual(len(parse_verdicts(raw, 1)), 1)

    def test_missing_verdicts_leave_gaps(self):
        # 缺的位置由调用方按"未通过"处理；解析器不做补全猜测
        raw = '[{"supported": true}]'
        out = parse_verdicts(raw, 3)
        self.assertEqual(list(out), [0])

    def test_string_and_alias_flags(self):
        self.assertTrue(parse_verdicts('[{"supported": "true"}]', 1)[0]["supported"])
        self.assertTrue(parse_verdicts('[{"ok": true}]', 1)[0]["supported"])
        self.assertFalse(parse_verdicts('[{"supported": "false"}]', 1)[0]["supported"])

    def test_unparseable_is_empty(self):
        self.assertEqual(parse_verdicts("无法判断", 2), {})


class TestDedupe(unittest.TestCase):
    def test_identical_patterns_collapse_keeping_higher_confidence(self):
        out = dedupe_patterns([
            {"pattern": "缺埋深时评分偏低", "confidence": 0.7},
            {"pattern": "缺埋深时评分偏低", "confidence": 0.9},
        ])
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["confidence"], 0.9)

    def test_distinct_patterns_survive(self):
        out = dedupe_patterns([
            {"pattern": "缺埋深时评分偏低", "confidence": 0.9},
            {"pattern": "检索相关度低时结论明显变短", "confidence": 0.9},
        ])
        self.assertEqual(len(out), 2)


class TestSimilarityAndRanking(unittest.TestCase):
    def test_identical_text_is_one(self):
        self.assertAlmostEqual(similarity("缺埋深时评分偏低", "缺埋深时评分偏低"), 1.0)

    def test_unrelated_text_is_zero(self):
        self.assertEqual(similarity("缺埋深时评分偏低", "井筒腐蚀"), 0.0)

    def test_empty_text_is_zero(self):
        self.assertEqual(similarity("", "缺埋深时评分偏低"), 0.0)

    def test_absence_claim_fires_when_the_attribute_is_missing(self):
        # 这条是本模块最容易做错的地方：规律说"缺少埋深"，而问句里**没有**
        # "埋深"二字 —— 字面重叠为 0，纯相似度会把它丢掉，而它恰恰是
        # 最该被注入的那条。属性感知加成就是为这个场景加的。
        pattern = "缺少储层埋深时物理校验无法执行，评分普遍偏低"
        query = "某CCUS项目注入压力250bar，温度80℃，请评估风险。"
        self.assertEqual(similarity(pattern, query), 0.0)
        self.assertGreater(relevance(pattern, query), 0.0)
        self.assertEqual(len(rank_priors([{"pattern": pattern, "confidence": 0.9}], query)), 1)

    def test_absence_claim_scores_higher_when_the_attribute_is_missing(self):
        pattern = "缺少储层埋深时物理校验无法执行，评分普遍偏低"
        missing = relevance(pattern, "注入压力250bar，温度80℃")
        present = relevance(pattern, "注入压力250bar，温度80℃，储层埋深2500m")
        # 数据缺失时这条规律反而更该被提醒 —— 这是"缺失提醒"加成的意义
        self.assertGreater(missing, present)

    def test_irrelevant_prior_is_not_injected(self):
        # 相关度不足的先验塞进 prompt 只会让模型当真
        records = [{"pattern": "井筒腐蚀会导致套管失效", "confidence": 0.95}]
        self.assertEqual(rank_priors(records, "注入压力250bar温度80℃"), [])

    def test_single_bigram_overlap_is_not_enough(self):
        # 只共享一个非属性二元组的长文本，Jaccard 趋近 0；阈值挡掉这类偶然重叠
        records = [{"pattern": "监测手段与数据采集频率" * 20, "confidence": 0.95}]
        self.assertEqual(rank_priors(records, "监测"), [])

    def test_low_confidence_is_filtered_out(self):
        records = [{"pattern": "注入压力高时评分偏低", "confidence": 0.5}]
        self.assertEqual(rank_priors(records, "注入压力高吗", min_confidence=MIN_CONFIDENCE), [])

    def test_ranking_prefers_relevance_over_confidence(self):
        records = [
            {"pattern": "井筒腐蚀会导致套管失效并降低评分", "confidence": 0.99},
            {"pattern": "缺埋深时物理校验无法执行且评分偏低", "confidence": 0.81},
        ]
        out = rank_priors(records, "这个项目缺少埋深，评分怎么算", top_k=1)
        self.assertEqual(len(out), 1)
        self.assertIn("埋深", out[0]["pattern"])

    def test_top_k_is_respected(self):
        records = [{"pattern": f"缺埋深时评分偏低（第{i}版）", "confidence": 0.9} for i in range(6)]
        self.assertEqual(len(rank_priors(records, "缺埋深 评分", top_k=3)), 3)

    def test_format_priors_is_empty_string_when_nothing_selected(self):
        # 调用方用 if text: 判断是否拼进 prompt，占位文本会让每份 prompt 都变长
        self.assertEqual(format_priors([]), "")

    def test_format_priors_carries_confidence(self):
        text = format_priors([{"pattern": "缺埋深时评分偏低", "confidence": 0.9}])
        self.assertIn("缺埋深时评分偏低", text)
        self.assertIn("0.90", text)


if __name__ == "__main__":
    unittest.main()
