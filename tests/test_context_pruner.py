"""
离线单元测试：上下文压缩（Context Pruning）
==========================================
不需要 Ollama / 网络，用假 summarizer 覆盖全部边界。

这里重点钉住的是**失败回退**：压缩出问题时必须原样返回原文。
如果哪天有人"优化"成 `return out or ""`，报告会因为上下文丢失而变差，
但不会有任何报错 —— 这类静默劣化只能靠测试挡。

运行:
    python -m unittest tests.test_context_pruner -v
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from context_pruner import (
    DEFAULT_CHUNK_CHARS,
    DEFAULT_PRUNE_THRESHOLD,
    MIN_GAIN_RATIO,
    build_prune_prompt,
    compress_context,
    context_char_count,
    needs_pruning,
    prune_context,
    split_into_chunks,
)


LONG = "断层活化风险。" * 800          # 5600 字，明显超过 4000
SHORT = "注入压力 250 bar，埋深 2500 m。"   # 远低于阈值


class TestThreshold(unittest.TestCase):
    def test_char_count_is_chars_not_bytes(self):
        """中文按字符计数：'中文' 是 2 字符 / 6 字节。"""
        self.assertEqual(context_char_count("中文"), 2)

    def test_below_threshold(self):
        self.assertFalse(needs_pruning(SHORT))

    def test_above_threshold(self):
        self.assertTrue(needs_pruning(LONG))

    def test_exactly_at_threshold_does_not_trigger(self):
        """边界用 >，不是 >=：恰好 4000 字不该多付一次推理。"""
        text = "a" * DEFAULT_PRUNE_THRESHOLD
        self.assertFalse(needs_pruning(text))

    def test_one_over_threshold_triggers(self):
        text = "a" * (DEFAULT_PRUNE_THRESHOLD + 1)
        self.assertTrue(needs_pruning(text))

    def test_threshold_zero_disables(self):
        self.assertFalse(needs_pruning(LONG, threshold=0))

    def test_negative_threshold_disables(self):
        self.assertFalse(needs_pruning(LONG, threshold=-1))


class TestPrompt(unittest.TestCase):
    def test_prompt_contains_context(self):
        p = build_prune_prompt("某段检索内容")
        self.assertIn("某段检索内容", p)

    def test_long_input_is_truncated(self):
        p = build_prune_prompt("字" * 20000, max_input_chars=1000)
        self.assertIn("已截断", p)
        # 截断后总长应远小于原文
        self.assertLess(len(p), 2000)


class TestPruneSuccess(unittest.TestCase):
    def test_pruned(self):
        result = prune_context(LONG, summarizer=lambda _p: "- 注入压力 250 bar 时断层活化风险上升")
        self.assertEqual(result["status"], "pruned")
        self.assertLess(result["after_chars"], result["before_chars"])
        self.assertEqual(result["saved_chars"], result["before_chars"] - result["after_chars"])
        self.assertGreater(result["saved_ratio"], 0)
        self.assertIn("断层活化", result["compressed"])
        self.assertEqual(result["reason"], "")

    def test_prompt_passed_to_summarizer(self):
        seen = {}

        def fake(prompt):
            seen["prompt"] = prompt
            return "- 要点一\n- 要点二"

        prune_context(LONG, summarizer=fake)
        self.assertIn("检索资料", seen["prompt"])
        self.assertIn("断层活化风险", seen["prompt"])

    def test_compressed_is_usable_text(self):
        result = prune_context(LONG, summarizer=lambda _p: "- 甲\n- 乙")
        self.assertIsInstance(result["compressed"], str)
        self.assertTrue(result["compressed"].strip())


class TestPruneSkips(unittest.TestCase):
    def test_below_threshold_skipped_without_calling_model(self):
        called = []

        def fake(_p):
            called.append(1)
            return "- 不该被调用"

        result = prune_context(SHORT, summarizer=fake)
        self.assertEqual(result["status"], "skipped")
        self.assertEqual(called, [])
        self.assertEqual(result["compressed"], SHORT)
        self.assertIn("未达阈值", result["reason"])

    def test_skipped_keeps_text_identical(self):
        result = prune_context(SHORT, summarizer=lambda _p: "- x")
        self.assertEqual(result["before_chars"], result["after_chars"])
        self.assertEqual(result["saved_chars"], 0)


class TestPruneFailuresFallBack(unittest.TestCase):
    """三种失败都必须回退原文 —— 这是本模块最重要的约定。"""

    def test_no_summarizer(self):
        result = prune_context(LONG, summarizer=None)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["compressed"], LONG)
        self.assertIn("未注入", result["reason"])

    def test_summarizer_raises(self):
        def boom(_p):
            raise RuntimeError("模型超时")

        result = prune_context(LONG, summarizer=boom)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["compressed"], LONG)
        self.assertIn("模型超时", result["reason"])

    def test_empty_output(self):
        result = prune_context(LONG, summarizer=lambda _p: "   ")
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["compressed"], LONG)
        self.assertIn("空", result["reason"])

    def test_non_list_output(self):
        """模型回了散文而不是要点 —— 结构不可信，回退。"""
        result = prune_context(LONG, summarizer=lambda _p: "这是一段总结性的散文，没有要点。")
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["compressed"], LONG)
        self.assertIn("要点列表", result["reason"])

    def test_no_gain(self):
        """输出几乎和原文一样长 —— 白花一次推理，必须回退。"""
        # 前面加 "- " 让它通过"要点列表"形态检查，才能走到"无收益"这一条
        echo = "- " + LONG
        result = prune_context(LONG, summarizer=lambda _p: echo)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["compressed"], LONG)
        self.assertIn("无收益", result["reason"])

    def test_small_gain_below_min_ratio(self):
        """只缩短 10%（未达 MIN_GAIN_RATIO=0.85 的要求）也算失败。"""
        kept = "- " + "字" * int(len(LONG) * 0.9)
        result = prune_context(LONG, summarizer=lambda _p: kept)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["compressed"], LONG)

    def test_gain_just_above_ratio_passes(self):
        """缩短到 80%（< 85% 门槛）应成功。"""
        kept = "- " + "字" * int(len(LONG) * 0.8)
        result = prune_context(LONG, summarizer=lambda _p: kept)
        self.assertEqual(result["status"], "pruned")
        self.assertLess(result["after_chars"], len(LONG) * MIN_GAIN_RATIO)

    def test_failed_result_has_all_keys(self):
        """失败结果的结构必须和成功一致，调用方不该写 if status 才能取值。"""
        result = prune_context(LONG, summarizer=None)
        for key in ("status", "before_chars", "after_chars", "saved_chars",
                    "saved_ratio", "compressed", "reason", "chunks", "calls",
                    "prompts"):
            self.assertIn(key, result)


class TestChunking(unittest.TestCase):
    """
    分块是必须的：本地 Ollama 默认只有 4096 tokens 上下文，超限直接 400。
    不分块的话，4000 字以上的输入会整次失败、悄悄退回原文——
    功能等于不存在。
    """

    def test_short_text_single_chunk(self):
        self.assertEqual(len(split_into_chunks("短文本", 3000)), 1)

    def test_empty_text_no_chunks(self):
        self.assertEqual(split_into_chunks("", 3000), [])

    def test_splits_on_paragraph_boundary(self):
        paras = ["段落" * 100 for _ in range(6)]   # 每段 200 字
        text = "\n\n".join(paras)
        chunks = split_into_chunks(text, 500)
        self.assertGreater(len(chunks), 1)
        for c in chunks:
            self.assertLessEqual(len(c), 500 + 2, f"块超长: {len(c)}")

    def test_oversized_paragraph_hard_split(self):
        """单个段落本身超长时只能硬切，但不能丢字符。"""
        text = "字" * 1200
        chunks = split_into_chunks(text, 500)
        self.assertEqual(len(chunks), 3)
        self.assertEqual("".join(chunks), text)

    def test_no_character_loss(self):
        """分块不能丢内容：拼回去应与原文等价（忽略空白差异）。"""
        paras = [f"第{i}段" + "内容" * 80 for i in range(8)]
        text = "\n\n".join(paras)
        chunks = split_into_chunks(text, 400)
        joined = "\n\n".join(chunks)
        self.assertEqual(joined.replace("\n", ""), text.replace("\n", ""))

    def test_chunk_count_recorded(self):
        text = "\n\n".join(["甲" * 200 for _ in range(6)])
        calls = []

        def fake(prompt):
            calls.append(prompt)
            return "- 要点"

        result = prune_context(text, summarizer=fake, threshold=100, chunk_chars=300)
        self.assertEqual(result["chunks"], len(calls))
        self.assertEqual(result["calls"], len(calls))
        self.assertEqual(len(result["prompts"]), len(calls))

    def test_all_chunks_contribute_bullets(self):
        text = "\n\n".join(["甲" * 200 for _ in range(6)])
        counter = {"n": 0}

        def fake(_p):
            counter["n"] += 1
            return f"- 第{counter['n']}块要点"

        result = prune_context(text, summarizer=fake, threshold=100, chunk_chars=300)
        self.assertEqual(result["status"], "pruned")
        for i in range(1, counter["n"] + 1):
            self.assertIn(f"第{i}块要点", result["compressed"])

    def test_duplicate_bullets_deduped(self):
        """
        各块常给出同一句规律，重复要点只会白占上下文。
        去重后只剩一条 —— 这是"有效的压缩"，不是失败。
        """
        text = "\n\n".join(["甲" * 200 for _ in range(6)])
        result = prune_context(
            text, summarizer=lambda _p: "- 同一句规律", threshold=100, chunk_chars=300
        )
        self.assertEqual(result["status"], "pruned")
        lines = result["compressed"].splitlines()
        self.assertEqual(len(lines), 1, lines)
        self.assertGreater(result["chunks"], 1, "应确实被分成了多块")

    def test_max_bullets_capped(self):
        text = "\n\n".join(["甲" * 200 for _ in range(6)])
        counter = {"n": 0}

        def fake(_p):
            counter["n"] += 1
            return "\n".join(f"- 要点{counter['n']}-{j}" for j in range(5))

        result = prune_context(
            text, summarizer=fake, threshold=100, chunk_chars=300, max_bullets=4
        )
        self.assertEqual(len(result["compressed"].splitlines()), 4)

    def test_one_chunk_failing_does_not_kill_all(self):
        """单块失败时，已成功的块仍应贡献要点。"""
        text = "\n\n".join(["甲" * 200 for _ in range(6)])
        counter = {"n": 0}

        def flaky(_p):
            counter["n"] += 1
            if counter["n"] == 2:
                raise RuntimeError("第二块超时")
            return f"- 第{counter['n']}块要点"

        result = prune_context(text, summarizer=flaky, threshold=100, chunk_chars=300)
        self.assertEqual(result["status"], "pruned")
        self.assertIn("第1块要点", result["compressed"])
        self.assertIn("第3块要点", result["compressed"])
        self.assertIn("第 2 块", result["reason"])   # 失败原因要留痕

    def test_all_chunks_failing_falls_back(self):
        text = "\n\n".join(["甲" * 200 for _ in range(6)])

        def boom(_p):
            raise RuntimeError("全都挂了")

        result = prune_context(text, summarizer=boom, threshold=100, chunk_chars=300)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["compressed"], text)

    def test_total_cap(self):
        """绝对上限：防止误传整本书导致几百次推理。"""
        text = "字" * 10000
        calls = []
        prune_context(
            text, summarizer=lambda p: (calls.append(1), "- 要点")[1],
            threshold=100, chunk_chars=300, max_total_chars=600,
        )
        self.assertLessEqual(len(calls), 2)


class TestPublicName(unittest.TestCase):
    """compress_context 与 prune_context 必须是同一个对象，不能是两套实现。"""

    def test_alias_is_same_function(self):
        self.assertIs(compress_context, prune_context)

    def test_alias_behaves_identically(self):
        r = compress_context(LONG, summarizer=lambda _p: "- 注入压力 250 bar 时断层活化")
        self.assertEqual(r["status"], "pruned")
        self.assertIn("250 bar", r["compressed"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
