"""checkpoint.db 的写者约束与部署一致性。

这里测的三件事都属于**写错了不会立刻报错**的类型：

1. **连接必须带 WAL 与 busy_timeout。** 少了它们不会让测试变红，只会在
   ``/health`` 用独立连接读库时，把"被正在进行的写挡住"读成"0 条断点"。
2. **后端只能跑一个 worker。** ``SqliteSaver`` 跨进程没有互斥。实测（Windows，
   2 进程 × 60 次 checkpoint 写 × 10 轮 = 1,200 次写）：
   默认配置 3 次失败，加 WAL + busy_timeout **仍然** 3 次失败。
   异常是 ``attempt to write a readonly database``，抛在 ``SqliteSaver.put``
   的 INSERT 上 —— 也就是任何节点跑完都可能让整条任务崩掉，不只是 HITL。
3. **compose 里的数字必须与代码一致。** 代码兜底（强制降为 1）不会让读
   compose 的人知道原因；反过来，compose 写 2 而代码降成 1，两边说的不是
   一件事，排查时会互相误导。

第 2 条为什么不是"加个重试就行"：失败点不是某一次写，而是**任意一次写**。
一个 900 秒的评估要写几十次 checkpoint，0.25% 的单次失败率意味着长任务
基本活不到结束。
"""
from __future__ import annotations

import importlib.util
import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent


def _load_gunicorn_conf():
    """按路径加载 ``gunicorn.conf.py`` —— 它不是包内模块，无法 import。

    加载时把 ``WEB_CONCURRENCY`` 临时设成 1：这个文件的模块体里就会调用
    ``resolve_worker_count``，不设的话每加载一次就往 stderr 打一段降级警告，
    把测试输出淹掉（被测的是函数本身，不是模块加载时的那个值）。
    """
    path = ROOT / "gunicorn.conf.py"
    spec = importlib.util.spec_from_file_location("_gunicorn_conf_under_test", path)
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(os.environ, {"WEB_CONCURRENCY": "1"}):
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


class TestWorkerCountIsClamped(unittest.TestCase):
    """``resolve_worker_count`` 是"多 worker 会被降级"这件事的可测形式。"""

    def setUp(self):
        self.conf = _load_gunicorn_conf()

    def test_two_workers_are_clamped_to_one(self):
        self.assertEqual(self.conf.resolve_worker_count(2, allow_multi=False), 1)

    def test_four_workers_are_clamped_to_one(self):
        self.assertEqual(self.conf.resolve_worker_count(4, allow_multi=False), 1)

    def test_one_worker_stays_one(self):
        self.assertEqual(self.conf.resolve_worker_count(1, allow_multi=False), 1)

    def test_zero_is_raised_to_one(self):
        """``WEB_CONCURRENCY=0`` 不是"自动"，是"没有 worker" —— 不能放过。"""
        self.assertEqual(self.conf.resolve_worker_count(0, allow_multi=False), 1)

    def test_opt_in_is_respected(self):
        """换成 Postgres saver 之后要能真的开多 worker。"""
        self.assertEqual(self.conf.resolve_worker_count(3, allow_multi=True), 3)


class TestComposeAgreesWithTheCode(unittest.TestCase):
    """compose 里那个 ``WEB_CONCURRENCY`` 是给人读的，必须与结论一致。"""

    def test_compose_pins_one_worker(self):
        text = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
        self.assertIn("WEB_CONCURRENCY: 1", text)
        self.assertNotIn("WEB_CONCURRENCY: 2", text)

    def test_compose_explains_why(self):
        """数字旁边没有理由，下一个人就会"顺手优化"回 2。"""
        text = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
        self.assertIn("attempt to write a readonly database", text)


class TestConnectionPragmas(unittest.TestCase):
    """连接级的三个 PRAGMA —— 直接问 sqlite，不读源码。

    只 import ``core.checkpoint``（纯 stdlib）。**不要**在这里 import
    ``workflow``：整图测试（``test_negative_gate``）靠"导入 workflow 之前先把
    节点模块换成桩"来装桩，一旦这里先把 workflow 拉进 ``sys.modules``，
    那张图就绑死了真实节点，整图测试会静默失效（表现为闸门不再拦住）。
    """

    def setUp(self):
        from core.checkpoint import open_checkpoint_conn

        self.open_conn = open_checkpoint_conn
        self.tmp = tempfile.TemporaryDirectory()
        self.db = os.path.join(self.tmp.name, "checkpoint.db")
        self.conn = open_checkpoint_conn(self.db)

    def tearDown(self):
        self.conn.close()
        self.tmp.cleanup()

    def test_journal_mode_is_wal(self):
        mode = self.conn.execute("PRAGMA journal_mode").fetchone()[0]
        self.assertEqual(str(mode).lower(), "wal")

    def test_busy_timeout_is_set(self):
        ms = self.conn.execute("PRAGMA busy_timeout").fetchone()[0]
        self.assertGreaterEqual(int(ms), 10_000)

    def test_it_can_actually_write(self):
        """PRAGMA 对了但库写不进去，是最难查的一种"配置正确"。"""
        self.conn.execute("CREATE TABLE t (x INTEGER)")
        self.conn.execute("INSERT INTO t VALUES (1)")
        self.conn.commit()
        self.assertEqual(self.conn.execute("SELECT count(*) FROM t").fetchone()[0], 1)

    def test_second_connection_sees_the_same_file(self):
        """WAL 的价值在这里：另一个连接读得到，且不会被写挡住。"""
        self.conn.execute("CREATE TABLE t (x INTEGER)")
        self.conn.execute("INSERT INTO t VALUES (7)")
        self.conn.commit()
        other = sqlite3.connect(self.db, timeout=5)
        try:
            self.assertEqual(other.execute("SELECT x FROM t").fetchone()[0], 7)
        finally:
            other.close()

    def test_default_path_follows_the_environment(self):
        """容器里靠 CHECKPOINT_DB 指到挂载目录，默认值不能被写死。"""
        from core import checkpoint

        self.assertEqual(
            checkpoint.CHECKPOINT_DB,
            os.getenv("CHECKPOINT_DB", "checkpoint.db"),
        )


if __name__ == "__main__":
    unittest.main()
