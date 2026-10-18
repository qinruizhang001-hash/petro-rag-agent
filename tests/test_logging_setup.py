"""
离线单元测试：日志配置
======================
日志是最容易被"看起来没问题"骗过去的一层：格式串少一个字段、handler 装了两遍、
``extra`` 没传时抛异常 —— 这些都不会让业务出错，只会在真正需要排查的那天
才被发现。所以这里逐条钉住。

零第三方依赖：CI 的 unit-core 直接跑。
"""
import logging
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logging_setup import (
    DEFAULT_FORMAT,
    _ContextFormatter,
    llm_extra,
    reset_logging,
    setup_logging,
)


class TestFormatter(unittest.TestCase):
    def setUp(self):
        self.fmt = _ContextFormatter(DEFAULT_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")

    def _record(self, msg="hello", **fields):
        """
        直接构造 LogRecord 时，``model=`` / ``tokens=`` 这类自定义 kwarg
        会被**静默丢弃**（LogRecord.__init__ 收 **kwargs 但只认自己认识的键）。
        真实路径上 ``Logger.makeRecord`` 才会把 extra 逐键塞进 __dict__，
        所以这里也手工塞，否则测的就不是线上那条路径。
        """
        record = logging.LogRecord(
            name="nodes.report_node", level=logging.INFO, pathname=__file__,
            lineno=1, msg=msg, args=(), exc_info=None,
        )
        record.__dict__.update(fields)
        return record

    def test_format_contains_all_required_fields(self):
        """时间戳 / 模块名 / 模型 / token 四样缺一不可。"""
        out = self.fmt.format(self._record(model="cloud:deepseek-chat", tokens=4521))
        self.assertRegex(out, r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}")
        self.assertIn("nodes.report_node", out)
        self.assertIn("model=cloud:deepseek-chat", out)
        self.assertIn("tokens=4521", out)
        self.assertIn("hello", out)

    def test_missing_extras_default_to_dash(self):
        """
        绝大多数日志不涉及模型和 token。不兜默认值的话，
        格式化会抛 KeyError，把"打一条日志"变成"炸掉整条调用栈"。
        """
        out = self.fmt.format(self._record())
        self.assertIn("model=-", out)
        self.assertIn("tokens=-", out)

    def test_local_and_cloud_are_distinguishable(self):
        local = self.fmt.format(self._record(model="local:qwen2.5"))
        cloud = self.fmt.format(self._record(model="cloud:deepseek-chat"))
        self.assertIn("model=local:", local)
        self.assertIn("model=cloud:", cloud)


class TestSetup(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self._tmp.name, "test.log")
        self._prev_level = logging.getLogger().level

    def tearDown(self):
        # 必须先摘掉并关闭文件 handler，再删临时目录 ——
        # Windows 上文件句柄没关时删除会抛 WinError 32。
        reset_logging()
        self._tmp.cleanup()
        # 还原成默认配置，避免影响同进程里的其它测试
        setup_logging(level=self._prev_level, force=True)

    def test_writes_to_file(self):
        setup_logging(log_file=self.path, level=logging.INFO, force=True)
        logging.getLogger("nodes.extract_node").info(
            "提取完成", extra={"model": "local:qwen2.5", "tokens": 320})
        logging.getLogger().handlers[0].flush()

        with open(self.path, encoding="utf-8") as f:
            content = f.read()
        self.assertIn("nodes.extract_node", content)
        self.assertIn("model=local:qwen2.5", content)
        self.assertIn("tokens=320", content)
        self.assertIn("提取完成", content)

    def test_idempotent_no_duplicate_handlers(self):
        """
        重复配置会叠加 handler，每条日志被写两遍。文件体积异常之前
        根本发现不了 —— 所以必须钉住。
        """
        setup_logging(log_file=self.path, force=True)
        before = len(logging.getLogger().handlers)
        setup_logging(log_file=self.path)          # 不带 force
        setup_logging(log_file=self.path)
        self.assertEqual(len(logging.getLogger().handlers), before)

    def test_force_replaces_instead_of_accumulating(self):
        setup_logging(log_file=self.path, force=True)
        before = len(logging.getLogger().handlers)
        setup_logging(log_file=self.path, force=True)
        self.assertEqual(len(logging.getLogger().handlers), before)

    def test_level_from_env(self):
        old = os.environ.get("LOG_LEVEL")
        try:
            os.environ["LOG_LEVEL"] = "WARNING"
            setup_logging(log_file=self.path, force=True)
            self.assertEqual(logging.getLogger().level, logging.WARNING)
            # INFO 级日志不应落盘
            logging.getLogger("nodes.report_node").info("不该出现")
            logging.getLogger().handlers[0].flush()
            with open(self.path, encoding="utf-8") as f:
                self.assertNotIn("不该出现", f.read())
        finally:
            if old is None:
                os.environ.pop("LOG_LEVEL", None)
            else:
                os.environ["LOG_LEVEL"] = old

    def test_unknown_level_falls_back_to_info(self):
        old = os.environ.get("LOG_LEVEL")
        try:
            os.environ["LOG_LEVEL"] = "NOT_A_LEVEL"
            setup_logging(log_file=self.path, force=True)
            self.assertEqual(logging.getLogger().level, logging.INFO)
        finally:
            if old is None:
                os.environ.pop("LOG_LEVEL", None)
            else:
                os.environ["LOG_LEVEL"] = old


class TestLlmExtra(unittest.TestCase):
    def test_label_shape(self):
        self.assertEqual(llm_extra("qwen2.5", "local")["model"], "local:qwen2.5")
        self.assertEqual(llm_extra("deepseek-chat", "cloud")["model"],
                         "cloud:deepseek-chat")

    def test_missing_parts(self):
        self.assertEqual(llm_extra()["model"], "-")
        self.assertEqual(llm_extra()["tokens"], "-")

    def test_tokens_coerced_to_int(self):
        self.assertEqual(llm_extra("qwen2.5", "local", "123")["tokens"], 123)


if __name__ == "__main__":
    unittest.main(verbosity=2)
