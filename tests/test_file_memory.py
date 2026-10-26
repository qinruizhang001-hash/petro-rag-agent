"""
离线单元测试：文件记忆（File-Based Memory）
==========================================
不需要 Ollama / 模型 / 图谱，纯临时文件断言，CI 秒级跑完。

运行:
    python -m unittest discover -s tests -v
"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import file_memory


class FileMemoryTestBase(unittest.TestCase):
    """把两个记忆文件重定向到临时目录，避免污染真实项目文件。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._orig_profile = file_memory.USER_PROFILE_PATH
        self._orig_history = file_memory.PROJECT_HISTORY_PATH
        file_memory.USER_PROFILE_PATH = os.path.join(self._tmp.name, "user_profile.md")
        file_memory.PROJECT_HISTORY_PATH = os.path.join(self._tmp.name, "project_history.md")

    def tearDown(self):
        file_memory.USER_PROFILE_PATH = self._orig_profile
        file_memory.PROJECT_HISTORY_PATH = self._orig_history
        self._tmp.cleanup()

    def write_profile(self, text):
        with open(file_memory.USER_PROFILE_PATH, "w", encoding="utf-8") as f:
            f.write(text)

    def write_history(self, text):
        with open(file_memory.PROJECT_HISTORY_PATH, "w", encoding="utf-8") as f:
            f.write(text)

    def read_history(self) -> str:
        """读回历史文件。用 with 关文件 —— 裸 open().read() 会留 ResourceWarning。"""
        with open(file_memory.PROJECT_HISTORY_PATH, encoding="utf-8") as f:
            return f.read()


class TestEnsureFiles(FileMemoryTestBase):
    def test_creates_missing_files_with_template(self):
        file_memory.ensure_files()
        self.assertTrue(os.path.exists(file_memory.USER_PROFILE_PATH))
        self.assertTrue(os.path.exists(file_memory.PROJECT_HISTORY_PATH))
        with open(file_memory.USER_PROFILE_PATH, encoding="utf-8") as f:
            self.assertIn("用户画像", f.read())

    def test_does_not_overwrite_existing_content(self):
        self.write_profile("# 我的画像\n- 只关注砂岩储层\n")
        file_memory.ensure_files()
        self.assertIn("砂岩储层", file_memory.read_user_profile())


class TestReadUserProfile(FileMemoryTestBase):
    def test_returns_empty_when_only_template(self):
        # 模板本身包含示例文字，ensure_files 后应能读到内容而非崩溃
        profile = file_memory.read_user_profile()
        self.assertIsInstance(profile, str)

    def test_truncates_long_profile(self):
        self.write_profile("A" * 5000)
        out = file_memory.read_user_profile(max_chars=100)
        self.assertLess(len(out), 300)
        self.assertIn("截断", out)


class TestReadProjectHistory(FileMemoryTestBase):
    def _seed(self, n):
        blocks = ["# 历史\n\n"]
        for i in range(1, n + 1):
            blocks.append(f"## 2026-01-{i:02d} 10:00\n**核心结论**: 第{i}次\n\n")
        self.write_history("".join(blocks))

    def test_returns_only_recent_entries(self):
        """核心防御：历史无限增长时不能全量注入 Prompt。"""
        self._seed(20)
        out = file_memory.read_project_history(max_entries=3)
        self.assertIn("第20次", out)
        self.assertIn("第18次", out)
        self.assertNotIn("第1次", out)
        self.assertNotIn("第17次", out)

    def test_keeps_header_out_of_entries(self):
        self._seed(3)
        out = file_memory.read_project_history(max_entries=3)
        # 文件的一级标题不应被当作条目塞进上下文
        self.assertNotIn("# 历史", out)

    def test_char_cap_prefers_most_recent_text(self):
        self._seed(10)
        out = file_memory.read_project_history(max_entries=10, max_chars=200)
        self.assertIn("第10次", out)
        self.assertIn("省略", out)

    def test_empty_file_returns_empty_string(self):
        self.write_history("")
        self.assertEqual(file_memory.read_project_history(), "")


class TestAppendProjectHistory(FileMemoryTestBase):
    def test_appends_with_generated_heading(self):
        file_memory.append_project_history("**核心结论**: 断层活化风险高", timestamp="2026-02-02 10:00")
        with open(file_memory.PROJECT_HISTORY_PATH, encoding="utf-8") as f:
            content = f.read()
        self.assertIn("## 2026-02-02 10:00", content)
        self.assertIn("断层活化风险高", content)

    def test_multiple_appends_preserve_order(self):
        file_memory.append_project_history("第一条", timestamp="2026-02-02 10:00")
        file_memory.append_project_history("第二条", timestamp="2026-02-02 11:00")
        out = file_memory.read_project_history(max_entries=5)
        self.assertLess(out.index("第一条"), out.index("第二条"))

    def test_blank_entry_is_ignored(self):
        file_memory.ensure_files()
        before = os.path.getsize(file_memory.PROJECT_HISTORY_PATH)
        file_memory.append_project_history("   ")
        self.assertEqual(os.path.getsize(file_memory.PROJECT_HISTORY_PATH), before)

    def test_returns_char_count(self):
        size = file_memory.append_project_history("内容")
        self.assertEqual(size, file_memory.history_char_count())


class TestBuildMemoryContext(FileMemoryTestBase):
    def test_empty_when_both_files_empty(self):
        self.write_profile("")
        self.write_history("")
        self.assertEqual(file_memory.build_memory_context(), "")

    def test_includes_profile_only(self):
        self.write_profile("# 画像\n- 结论先行\n")
        self.write_history("")
        ctx = file_memory.build_memory_context()
        self.assertIn("用户画像", ctx)
        self.assertNotIn("历史评估记录", ctx)
        self.assertIn("结论先行", ctx)

    def test_includes_history_only(self):
        self.write_profile("")
        self.write_history("## 2026-01-01\n**核心结论**: x\n")
        ctx = file_memory.build_memory_context()
        self.assertIn("历史评估记录", ctx)
        self.assertNotIn("用户画像（长期偏好", ctx)

    def test_includes_both_sections(self):
        self.write_profile("# 画像\n- 数值落地\n")
        self.write_history("## 2026-01-01\n**核心结论**: y\n")
        ctx = file_memory.build_memory_context()
        self.assertIn("用户画像（长期偏好", ctx)
        self.assertIn("历史评估记录", ctx)


class FakeSummarizer:
    """记录调用次数与输入，返回固定的 3 条经验。"""

    def __init__(self, output="- 经验一\n- 经验二\n- 经验三", raises=False):
        self.output = output
        self.raises = raises
        self.calls = []

    def __call__(self, old_text):
        self.calls.append(old_text)
        if self.raises:
            raise RuntimeError("模拟大模型不可用")
        return self.output


class TestHistoryCompression(FileMemoryTestBase):
    """记忆压缩：超阈值把旧记录提炼成核心经验并原子重写文件。"""

    def seed(self, n=8, entry_chars=600):
        body = ["# 项目评估历史（Project History）\n\n> 说明行\n\n"]
        for i in range(1, n + 1):
            body.append(f"## 2026-03-{i:02d} 10:00\n**场景**: 第{i}次\n"
                        f"**核心结论**: {'X' * entry_chars}\n\n")
        self.write_history("".join(body))

    def read(self):
        with open(file_memory.PROJECT_HISTORY_PATH, encoding="utf-8") as f:
            return f.read()

    def test_skips_below_threshold(self):
        self.seed(n=2, entry_chars=50)
        s = FakeSummarizer()
        r = file_memory.compress_history_if_needed(summarizer=s, threshold=100000)
        self.assertEqual(r["status"], "skipped")
        self.assertEqual(r["reason"], "below_threshold")
        self.assertEqual(s.calls, [], "未超阈值时不应调用大模型")

    def test_compresses_above_threshold(self):
        self.seed()
        s = FakeSummarizer()
        r = file_memory.compress_history_if_needed(summarizer=s, threshold=1000)
        self.assertEqual(r["status"], "compressed")
        self.assertEqual(len(s.calls), 1, "一次压缩只应调用一次大模型")
        self.assertGreater(r["before_chars"], r["after_chars"])
        self.assertGreater(r["dropped_entries"], 0)

    def test_keeps_lessons_and_latest_entries(self):
        self.seed(n=8)
        file_memory.compress_history_if_needed(
            summarizer=FakeSummarizer("- 经验A\n- 经验B\n- 经验C"),
            threshold=1000, keep_recent=2,
        )
        text = self.read()
        self.assertIn(file_memory.LESSONS_HEADING, text)
        self.assertIn("经验A", text)
        self.assertIn("第8次", text, "最新记录必须原样保留")
        self.assertIn("第7次", text)
        self.assertNotIn("第1次", text, "旧记录应被压缩移除")

    def test_preserves_header(self):
        self.seed()
        file_memory.compress_history_if_needed(summarizer=FakeSummarizer(), threshold=1000)
        self.assertIn("# 项目评估历史（Project History）", self.read())

    def test_failure_leaves_file_untouched(self):
        self.seed()
        before = self.read()
        r = file_memory.compress_history_if_needed(
            summarizer=FakeSummarizer(raises=True), threshold=1000
        )
        self.assertEqual(r["status"], "failed")
        self.assertEqual(self.read(), before, "摘要失败时绝不能改动历史文件")

    def test_empty_summary_leaves_file_untouched(self):
        self.seed()
        before = self.read()
        r = file_memory.compress_history_if_needed(
            summarizer=FakeSummarizer(output="   "), threshold=1000
        )
        self.assertEqual(r["status"], "failed")
        self.assertEqual(r["reason"], "empty_summary")
        self.assertEqual(self.read(), before)

    def test_skips_without_summarizer(self):
        self.seed()
        before = self.read()
        r = file_memory.compress_history_if_needed(summarizer=None, threshold=1000)
        self.assertEqual(r["status"], "skipped")
        self.assertEqual(r["reason"], "no_summarizer")
        self.assertEqual(self.read(), before)

    def test_no_infinite_compression_loop(self):
        """
        自限开关：压缩后旧条目归零，即使文件仍超阈值也不能再次触发。
        否则每次读取都调一次大模型，延迟爆炸、结论还会反复漂移。
        """
        self.seed()
        s = FakeSummarizer()
        r1 = file_memory.compress_history_if_needed(summarizer=s, threshold=1000)
        self.assertEqual(r1["status"], "compressed")

        # 再调一次：内容仍可能 >1000，但旧条目不足，不应再次调用模型
        r2 = file_memory.compress_history_if_needed(summarizer=s, threshold=1000)
        self.assertIn(r2["status"], ("skipped",))
        self.assertEqual(len(s.calls), 1, "第二次不应再调用大模型")

    def test_lessons_merge_on_second_compression(self):
        """第二次压缩时，上一轮的经验块应作为旧条目被一起再提炼，而不是堆叠。"""
        self.seed(n=6)
        file_memory.compress_history_if_needed(
            summarizer=FakeSummarizer("- 旧经验1"), threshold=1000, keep_recent=1
        )
        # 再追加若干条，触发第二轮
        for i in range(1, 5):
            file_memory.append_project_history(f"**场景**: 新增{i}\n**核心结论**: {'Y'*400}")
        s2 = FakeSummarizer("- 融合后经验")
        file_memory.compress_history_if_needed(
            summarizer=s2, threshold=1000, keep_recent=1
        )
        self.assertEqual(len(s2.calls), 1)
        self.assertIn("旧经验1", s2.calls[0], "旧经验应作为输入被再次提炼")
        self.assertEqual(self.read().count(file_memory.LESSONS_HEADING), 1,
                         "经验块不能重复堆叠")

    def test_build_memory_context_triggers_compression(self):
        self.seed()
        s = FakeSummarizer("- 上下文经验")
        ctx = file_memory.build_memory_context(
            summarizer=s, compress_threshold=1000
        )
        self.assertEqual(len(s.calls), 1)
        self.assertIn("上下文经验", ctx)

    def test_no_temp_files_left_behind(self):
        self.seed()
        file_memory.compress_history_if_needed(summarizer=FakeSummarizer(), threshold=1000)
        leftovers = [f for f in os.listdir(self._tmp.name) if f.startswith(".history-")]
        self.assertEqual(leftovers, [], "原子写入不应残留临时文件")

    def test_history_needs_compression_reflects_size(self):
        self.seed()
        self.assertTrue(file_memory.history_needs_compression(threshold=1000))
        self.assertFalse(file_memory.history_needs_compression(threshold=10**9))

    def test_threshold_counts_characters_not_bytes(self):
        """
        阈值语义必须是字符数。中文 1 字符 = 3 字节，若用 os.path.getsize
        比较 3000，会在约 1000 汉字时就误触发压缩（早三倍）。
        """
        text = "# 历史\n\n" + "## 2026-01-01 10:00\n" + "汉" * 1200
        self.write_history(text)
        self.assertEqual(len(text), file_memory.history_char_count())
        # 关键前提：字节数已超过 3000（按字节比较会误判），字符数未超（正确判断为不触发）
        self.assertGreater(os.path.getsize(file_memory.PROJECT_HISTORY_PATH), 3000,
                           "该用例必须让字节数超过阈值，否则测不出区别")
        self.assertLess(len(text), 3000)
        # 1200 字 < 3000 字阈值 → 不应触发；若按字节比较则会误判为超阈值
        self.assertFalse(file_memory.history_needs_compression(threshold=3000))
        r = file_memory.compress_history_if_needed(
            summarizer=FakeSummarizer(), threshold=3000
        )
        self.assertEqual(r["status"], "skipped")
        self.assertEqual(r["reason"], "below_threshold")


class TestHistoryHardLimit(FileMemoryTestBase):
    """
    硬上限：不依赖大模型的那道防线。

    为什么它必须单独存在（而不是"有压缩就够了"）：
    压缩要求 ① 传了摘要器 ② 摘要成功 ③ 旧条目 ≥ MIN_ENTRIES_TO_COMPRESS。
    三者任一不满足就静默跳过，而写入是无条件的 —— 文件继续长，且**没有日志
    会报错**。所以写入侧必须自己兜住自己的增长。
    """

    def _entries(self, n, body="x" * 80):
        return "".join(f"## 2026-01-{i:02d} 10:00\n{body}\n\n" for i in range(1, n + 1))

    def test_no_trim_below_limit(self):
        self.write_history("# 历史\n\n" + self._entries(2))
        r = file_memory.trim_history_if_needed(max_chars=10**6)
        self.assertEqual(r["status"], "skipped")
        self.assertEqual(r["reason"], "below_limit")
        self.assertEqual(r["dropped_entries"], 0)

    def test_trims_oldest_and_keeps_newest(self):
        self.write_history("# 历史\n\n" + self._entries(20))
        r = file_memory.trim_history_if_needed(max_chars=200, keep_entries=3)
        self.assertEqual(r["status"], "trimmed", r)
        self.assertEqual(r["dropped_entries"], 17)
        text = self.read_history()
        # 最新的三条还在（01-18 / 01-19 / 01-20），最旧的已经不在了
        self.assertIn("2026-01-20", text)
        self.assertIn("2026-01-18", text)
        self.assertNotIn("2026-01-01", text)
        self.assertLess(r["after_chars"], r["before_chars"])

    def test_lessons_block_survives_trim(self):
        """
        经验块是压缩唯一的沉淀。裁掉它等于每次压缩白做，而且不会有任何报错。
        """
        self.write_history(
            "# 历史\n\n"
            f"## {file_memory.LESSONS_HEADING}\n- 注入压力必须低于破裂压力\n\n"
            + self._entries(20)
        )
        file_memory.trim_history_if_needed(max_chars=200, keep_entries=2)
        text = self.read_history()
        self.assertIn(file_memory.LESSONS_HEADING, text)
        self.assertIn("注入压力必须低于破裂压力", text)

    def test_never_drops_below_keep_floor(self):
        """
        保留下限优先于上限：宁可文件仍然超限，也不能把历史裁成空壳。
        "文件很小"看起来比"文件超限"健康得多，正是它危险的地方。
        """
        self.write_history("# 历史\n\n" + self._entries(3))
        r = file_memory.trim_history_if_needed(max_chars=10, keep_entries=3)
        self.assertEqual(r["status"], "skipped")
        self.assertEqual(r["reason"], "still_over_limit")
        text = self.read_history()
        self.assertEqual(text.count("## 2026-01-"), 3, "一条都不许丢")

    def test_append_triggers_trim(self):
        """
        写入路径必须自己检查上限 —— 否则"压缩一直不生效"时文件会无限长。
        """
        self.write_history("# 历史\n\n" + self._entries(10))
        orig_limit = file_memory.HISTORY_HARD_LIMIT
        orig_keep = file_memory.HISTORY_KEEP_ON_TRIM
        try:
            file_memory.HISTORY_HARD_LIMIT = 300
            file_memory.HISTORY_KEEP_ON_TRIM = 3
            size = file_memory.append_project_history("新的一条记录")
        finally:
            file_memory.HISTORY_HARD_LIMIT = orig_limit
            file_memory.HISTORY_KEEP_ON_TRIM = orig_keep

        text = self.read_history()
        self.assertIn("新的一条记录", text, "最新写入的条目不能被裁掉")
        self.assertNotIn("2026-01-01", text)
        self.assertEqual(size, len(text))

    def test_append_is_not_blocked_by_lock(self):
        """
        append 里先释放 _WRITE_LOCK 再调 trim（trim 内部会再拿一次锁）。
        threading.Lock 不可重入 —— 顺序写错就是永久死锁，而不是报错。
        这条测试的作用就是"如果哪天顺序被改坏，它会挂住超时"。
        """
        import threading

        done = []

        def worker():
            file_memory.append_project_history("并发记录")
            done.append(True)

        t = threading.Thread(target=worker, daemon=True)
        t.start()
        t.join(timeout=5)
        self.assertTrue(done, "append_project_history 疑似死锁（5s 未返回）")


if __name__ == "__main__":
    unittest.main(verbosity=2)
