"""checkpoint 库的连接配置。

为什么单独一个模块，而不是写在 ``workflow.py`` 里
-------------------------------------------------
``workflow.py`` 一被 import 就编译整张图（langgraph + 全部节点），于是"验证
连接用了哪些 PRAGMA"这种小事也得背上整张图 —— 在 CI 的零依赖 job 里根本跑不了，
只能靠读代码。拆出来之后这里是纯 stdlib（sqlite3 / os），可以真的问 sqlite
"journal_mode 是什么"，而不是 grep 源码。

为什么后端只能跑一个 worker（实测，不是保守）
---------------------------------------------
``SqliteSaver`` 跨进程没有互斥：两个 gunicorn worker 各自 connect 同一个
checkpoint.db，并发写会失败。实测（Windows，2 进程 × 60 次 checkpoint 写 ×
10 轮 = 1,200 次写）：

    默认配置                     3 次失败 / 1,200
    WAL + busy_timeout=30s       3 次失败 / 1,200

失败抛在 ``SqliteSaver.put`` 的 ``INSERT OR REPLACE INTO checkpoints`` 上，
异常是 ``OperationalError: attempt to write a readonly database``。

**WAL 和 busy_timeout 治不了它**，这一点值得单独记：它们解决 ``SQLITE_BUSY``
（锁等待），而这里是 Windows VFS 把共享冲突映射成 ``SQLITE_READONLY``，
busy handler 根本不会被调用。所以"加 WAL 就多进程安全"是个听起来很对、
实测不成立的修复。

真正的约束是**单写者**：一个 worker 进程内部靠 ``SqliteSaver.lock`` 串行化；
并发由 asyncio + 线程池承担，不靠多进程。``gunicorn.conf.py`` 会把
``WEB_CONCURRENCY`` 强制降到 1 并打印原因。

WAL 仍然开着，但它修的是**另一个**问题：``/health`` 用独立连接读库，回滚日志
模式下"读"会被正在进行的"写"挡住，探针就把"库被锁"读成"0 条断点"。

诚实的边界：checkpoint.db 在容器内、**没有挂卷**，所以"进程退出后还能续跑"
只对 worker 崩溃/重启成立，容器被删就没了。要跨容器保留，把 ``CHECKPOINT_DB``
指到挂载目录（例如 ``/app/state/checkpoint.db``）。
"""
from __future__ import annotations

import os
import sqlite3

# 容器里指到挂载目录才能在容器重建后仍能续跑人工审核。
# 默认值保持不变 —— 换路径是运维决定，不该由代码替人做主。
CHECKPOINT_DB = os.getenv("CHECKPOINT_DB", "checkpoint.db")


def open_checkpoint_conn(path: str | None = None) -> sqlite3.Connection:
    """打开 checkpoint 库并设好三个 PRAGMA。

    * ``check_same_thread=False`` 是**必须**的：FastAPI 的同步端点跑在线程池里，
      连接会跨线程用；``SqliteSaver`` 内部有 ``threading.Lock`` 保证互斥。
    * ``timeout=30`` 是连接级 busy timeout（``PRAGMA busy_timeout`` 是同一个东西
      的另一种写法，两处都写是为了不让"只改了其中一处"悄悄降级）。
    * ``journal_mode=WAL`` 让读不挡写：``/health`` 的独立连接去读时，不会把
      "正在写"误读成"0 条断点"。
    * ``synchronous=NORMAL``：checkpoint 是**可重建的**运行状态，不是账本。
      每写一次都 fsync 会让 900 秒的长任务反复等磁盘；断电丢最后一条断点的
      代价，小于每步都等一次刷盘。
    """
    conn = sqlite3.connect(path or CHECKPOINT_DB, check_same_thread=False, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn
