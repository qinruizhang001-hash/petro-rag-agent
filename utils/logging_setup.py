"""
统一日志配置（Observability）
=============================
``print`` 不是工程标准：它没有级别、没有落盘、没有模块归属，线上出问题时
既不能按级别过滤，也不能事后回溯。这个模块把整条运行链路统一到 ``logging``。

日志格式固定为五段，缺一不可：

    2026-10-18 09:12:33 | INFO    | nodes.report_node        | model=cloud:deepseek-chat | tokens=4521 | 报告生成完成

    └── 时间戳 ─────────┘ └级别┘  └── 模块名 ──┘         └── 当前模型 ──┘        └ Token ┘ └ 正文 ┘

为什么把 **模型名** 和 **Token 消耗** 放进格式串而不是正文里：这两个值
决定了"这次调用花了多少钱、走的是本地还是云端"。写进正文时它们只是
一段自由文本，无法被 grep / 聚合；放进字段后，
``grep 'model=cloud' app.log`` 就能一眼看出云端调用发生在哪一步。

字段缺省为 ``-``（由 ``_ContextFormatter`` 补齐）。这一点是必要的：
``logging.Formatter`` 遇到 record 上没有的属性会直接抛异常，
而绝大多数日志本来就不涉及模型和 token —— 不兜住的话，
"没写 extra 的那条日志"会把整个调用栈炸掉。

落盘用 ``RotatingFileHandler``（5 MB × 3）。本项目已经吃过一次
"``project_history.md`` 无界增长"的亏，日志不该重演。
⚠️ Windows 上多进程同时持有同一个日志文件时轮转可能失败 ——
本项目只有后端进程写日志（前端是纯 HTTP 客户端），不受影响。

本模块**只依赖标准库**：CI 的 unit-core 要能零安装地测它。
"""
from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler

DEFAULT_LOG_FILE = "app.log"
DEFAULT_FORMAT = (
    "%(asctime)s | %(levelname)-7s | %(name)-24s | "
    "model=%(model)s | tokens=%(tokens)s | %(message)s"
)
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
MAX_BYTES = 5 * 1024 * 1024
BACKUP_COUNT = 3

# 第三方库默认会把 HTTP 细节、模型加载进度全部打进来，把真正的业务日志淹掉。
_NOISY_LOGGERS = (
    "httpx", "httpcore", "urllib3", "openai",
    "chromadb", "sentence_transformers", "lightrag", "matplotlib", "PIL",
)

_CONFIGURED = False
_LOG_PATH = ""


class _ContextFormatter(logging.Formatter):
    """给缺失的 ``model`` / ``tokens`` 字段补默认值，避免格式化时抛异常。"""

    def format(self, record: logging.LogRecord) -> str:
        if not hasattr(record, "model"):
            record.model = "-"
        if not hasattr(record, "tokens"):
            record.tokens = "-"
        return super().format(record)


def _level_from_env() -> int:
    raw = (os.getenv("LOG_LEVEL") or "INFO").strip().upper()
    return getattr(logging, raw, logging.INFO)


def setup_logging(log_file: str | None = None, level: int | None = None,
                  force: bool = False) -> str:
    """
    配置根 logger，返回日志文件路径。

    幂等：重复调用不会叠加 handler（否则每条日志会被写两遍，
    而"日志重复"这种问题通常要等到文件体积异常才被发现）。
    """
    global _CONFIGURED, _LOG_PATH
    if _CONFIGURED and not force:
        return _LOG_PATH

    path = log_file or os.getenv("APP_LOG_FILE") or DEFAULT_LOG_FILE
    root = logging.getLogger()
    root.setLevel(level if level is not None else _level_from_env())

    for handler in list(root.handlers):
        if getattr(handler, "_petro_rag_handler", False):
            root.removeHandler(handler)
            handler.close()

    formatter = _ContextFormatter(DEFAULT_FORMAT, datefmt=DATE_FORMAT)

    file_handler = RotatingFileHandler(
        path, maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    file_handler._petro_rag_handler = True
    root.addHandler(file_handler)

    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(formatter)
    console._petro_rag_handler = True
    root.addHandler(console)

    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    _CONFIGURED = True
    _LOG_PATH = path
    return path


def get_logger(name: str) -> logging.Logger:
    """取一个已配置好的 logger。首次调用时自动完成全局配置。"""
    setup_logging()
    return logging.getLogger(name)


def reset_logging() -> None:
    """
    摘掉本模块装的 handler 并关闭它们。

    存在的理由很实际：Windows 上只要文件句柄没关，删临时目录就会抛
    ``PermissionError: [WinError 32]``。测试的 tearDown 需要它，
    否则每个用例都会在清理阶段炸一次 —— 而且炸的是"清理"，
    很容易被误认为是被测代码有问题。
    """
    global _CONFIGURED, _LOG_PATH
    root = logging.getLogger()
    for handler in list(root.handlers):
        if getattr(handler, "_petro_rag_handler", False):
            root.removeHandler(handler)
            handler.close()
    _CONFIGURED = False
    _LOG_PATH = ""


def log_path() -> str:
    """当前日志文件路径（未配置则先配置）。"""
    return setup_logging()


def llm_extra(model: str = "", tier: str = "", tokens: int | None = None) -> dict:
    """
    构造 ``extra=`` 参数，供日志格式串里的 ``model`` / ``tokens`` 使用。

    统一从这里生成，是为了让"模型名长什么样"只有一个答案：
    ``tier:model``（如 ``cloud:deepseek-chat``、``local:qwen2.5``）——
    一眼能看出这次调用是本地还是云端。
    """
    label = f"{tier}:{model}" if tier and model else (model or "-")
    return {"model": label or "-", "tokens": "-" if tokens is None else int(tokens)}
