"""
并行分支合并 / 降级 / 计时的离线单测（零依赖）
=============================================
``utils/branch_merge.py`` 是"出问题时才被执行"的那段代码，所以这里穷举
四种组合：两条都成功、只挂地质、只挂井筒、两条都挂。

重点不在"能跑通"，而在两条**必须成立**的性质：

1. 缺失分支的文本**永远不是空串** —— 空串拼进 Prompt 会被模型当成
   "这条要求不存在"，而降级文案会被当成"这条要求失败了"。这两者对读者
   的意义完全相反。
2. ``parallel_timing_of`` 的 speedup 来自时间戳，不是按节点名推算 ——
   所以如果扇出被改回串行，这里的断言会先失败。
"""
from __future__ import annotations

import unittest

from utils.branch_merge import (
    GEOLOGY_MISSING_TEXT,
    WELLBORE_MISSING_TEXT,
    compose_branch_inputs,
    degradation_block,
    parallel_timing_of,
)


class TestComposeBranchInputs(unittest.TestCase):
    def test_both_ok_keeps_texts_and_is_not_degraded(self):
        info = compose_branch_inputs({
            "geology_report": "断层活化风险中等，建议微地震监测。",
            "wellbore_report": "套管腐蚀速率 0.05 mm/a，建议井下压力计。",
        })
        self.assertFalse(info["degraded"])
        self.assertEqual(info["missing"], [])
        self.assertTrue(info["geology_ok"] and info["wellbore_ok"])
        self.assertIn("断层活化", info["geology"])
        self.assertIn("套管腐蚀", info["wellbore"])

    def test_missing_geology_falls_back_to_required_sentence(self):
        info = compose_branch_inputs({
            "wellbore_report": "套管腐蚀速率 0.05 mm/a。",
            "geology_status": {"ok": False, "error": "TimeoutError: read timed out"},
        })
        self.assertTrue(info["degraded"])
        self.assertEqual(info["missing"], ["geology"])
        # 需求逐字要求：报告里必须出现这句话
        self.assertEqual(info["geology"], GEOLOGY_MISSING_TEXT)
        self.assertIn("TimeoutError", info["reasons"]["geology"])
        # 井筒数据仍然可用，不能被降级逻辑顺手抹掉
        self.assertIn("0.05", info["wellbore"])

    def test_missing_wellbore_keeps_geology(self):
        info = compose_branch_inputs({
            "geology_report": "盖层突破压力 32 MPa。",
            "wellbore_status": {"ok": False, "error": "ConnectionError: refused"},
        })
        self.assertEqual(info["missing"], ["wellbore"])
        self.assertEqual(info["wellbore"], WELLBORE_MISSING_TEXT)
        self.assertIn("32 MPa", info["geology"])

    def test_both_missing_still_returns_two_texts(self):
        info = compose_branch_inputs({})
        self.assertEqual(sorted(info["missing"]), ["geology", "wellbore"])
        self.assertEqual(info["geology"], GEOLOGY_MISSING_TEXT)
        self.assertEqual(info["wellbore"], WELLBORE_MISSING_TEXT)
        self.assertFalse(info["geology_ok"] or info["wellbore_ok"])

    def test_whitespace_only_report_counts_as_missing(self):
        # 模型返回 "   " 是常见失败形态：它不是空字符串，但对下游毫无价值
        info = compose_branch_inputs({"geology_report": "   \n\t ", "wellbore_report": "ok"})
        self.assertEqual(info["missing"], ["geology"])

    def test_reason_defaults_when_status_absent(self):
        info = compose_branch_inputs({"geology_report": ""})
        self.assertEqual(info["reasons"]["geology"], "分支未产出结论")

    def test_non_string_and_non_dict_inputs_do_not_raise(self):
        # 状态字段可能来自旧 checkpoint（None）或被写坏成别的类型
        info = compose_branch_inputs({"geology_report": None, "geology_status": "boom"})
        # 重点是"没抛异常"；两条都没给内容，所以两条都算缺失
        self.assertIn("geology", info["missing"])
        self.assertTrue(info["degraded"])


class TestDegradationBlock(unittest.TestCase):
    def test_empty_when_not_degraded(self):
        self.assertEqual(degradation_block({"degraded": False, "missing": []}), "")

    def test_contains_required_sentence_and_reason(self):
        info = compose_branch_inputs({"geology_status": {"error": "TimeoutError: x"}})
        block = degradation_block(info)
        self.assertIn(GEOLOGY_MISSING_TEXT, block)
        self.assertIn("TimeoutError", block)
        # 必须明确"不代表该维度无风险"，否则读者会把缺失当成没问题
        self.assertIn("不代表该维度无风险", block)

    def test_lists_both_modules(self):
        block = degradation_block(compose_branch_inputs({}))
        self.assertIn(GEOLOGY_MISSING_TEXT, block)
        self.assertIn(WELLBORE_MISSING_TEXT, block)

    def test_only_failed_module_is_listed(self):
        info = compose_branch_inputs({
            "geology_report": "ok", "wellbore_status": {"error": "boom"},
        })
        block = degradation_block(info)
        self.assertNotIn(GEOLOGY_MISSING_TEXT, block)
        self.assertIn(WELLBORE_MISSING_TEXT, block)


class TestParallelTiming(unittest.TestCase):
    def test_overlapping_branches_report_real_speedup(self):
        # 两条分支同时开始，地质跑 50s、井筒跑 10s：墙钟 50s，串行需 60s
        timing = parallel_timing_of({
            "geology_status": {"started_at": 100.0, "ended_at": 150.0, "elapsed_s": 50.0},
            "wellbore_status": {"started_at": 100.0, "ended_at": 110.0, "elapsed_s": 10.0},
        })
        self.assertAlmostEqual(timing["wall_s"], 50.0, places=1)
        self.assertAlmostEqual(timing["serial_s"], 60.0, places=1)
        self.assertAlmostEqual(timing["speedup"], 1.2, places=2)
        self.assertAlmostEqual(timing["overlap_s"], 10.0, places=1)

    def test_sequential_execution_shows_speedup_near_one(self):
        # 若有人把扇出改回串行，第二分支的 start 会等于第一分支的 end
        timing = parallel_timing_of({
            "geology_status": {"started_at": 100.0, "ended_at": 150.0, "elapsed_s": 50.0},
            "wellbore_status": {"started_at": 150.0, "ended_at": 160.0, "elapsed_s": 10.0},
        })
        self.assertAlmostEqual(timing["wall_s"], 60.0, places=1)
        self.assertAlmostEqual(timing["speedup"], 1.0, places=2)
        self.assertAlmostEqual(timing["overlap_s"], 0.0, places=1)

    def test_falls_back_to_max_elapsed_without_timestamps(self):
        timing = parallel_timing_of({
            "geology_status": {"elapsed_s": 40.0},
            "wellbore_status": {"elapsed_s": 5.0},
        })
        self.assertAlmostEqual(timing["wall_s"], 40.0, places=1)
        self.assertAlmostEqual(timing["speedup"], 1.12, places=2)   # 45/40

    def test_negative_wall_clock_is_clamped(self):
        # 跨进程续跑 / 时钟回拨时时间戳可能倒挂，不能算出负墙钟
        timing = parallel_timing_of({
            "geology_status": {"started_at": 200.0, "ended_at": 100.0, "elapsed_s": 50.0},
            "wellbore_status": {"started_at": 210.0, "ended_at": 150.0, "elapsed_s": 10.0},
        })
        self.assertGreaterEqual(timing["wall_s"], 0.0)
        self.assertGreaterEqual(timing["speedup"], 0.0)

    def test_empty_state_is_safe(self):
        timing = parallel_timing_of({})
        self.assertEqual(timing["serial_s"], 0.0)
        self.assertEqual(timing["speedup"], 1.0)


if __name__ == "__main__":
    unittest.main()
