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

    def test_returns_file_size(self):
        size = file_memory.append_project_history("内容")
        self.assertEqual(size, os.path.getsize(file_memory.PROJECT_HISTORY_PATH))


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


if __name__ == "__main__":
    unittest.main(verbosity=2)
