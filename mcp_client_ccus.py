"""
主系统侧的 MCP Client
=====================
把 ``mcp_server_ccus.py`` 暴露的三个工具接进 LangGraph 工作流。上层只需要
两个同步方法::

    client.list_tools()          # [{name, description, properties, required}, ...]
    client.call(name, args)      # dict（工具返回的结构化结果）

为什么默认走"同进程内存传输"而不是 stdio 子进程
-----------------------------------------------
这是本文件最重要的一个决定。``query_ccus_graph`` 会初始化 LightRAG，连带
加载 sentence-transformers 与约 2.3 GB 的 reranker。如果主系统用 stdio 去
spawn 一个子进程，那个子进程会**再加载一份模型**：

  * api_server 已经持有 reranker（地质分支要用），
  * 再起一份 2.3 GB 的常驻内存，在 16 GB 机器上会被系统直接杀掉，
  * 症状是请求中途 ``ConnectionReset`` / 无 traceback 的进程消失 ——
    项目里 ``test_graph_rag_e2e.py`` 之所以"在进程内起 uvicorn"，就是被这个
    问题逼出来的。

所以主系统用 ``create_client_server_memory_streams()``：协议层完全一样
（JSON-RPC 消息、initialize / list_tools / call_tool 一个不少），只是把
"管道"换成进程内的内存流，Server 与 Client 跑在同一个事件循环里。
模型只有一份，工具仍然是通过 MCP 协议调用的。

stdio 子进程模式仍然保留（``transport="stdio"``），因为它才是 Cursor /
Claude Desktop 的用法：那时没有第二个进程占着模型，子进程是安全的。
``test_mcp_client.py`` 走的就是 stdio，用来证明这条路径也没坏。

线程模型
--------
LangGraph 的同步 ``invoke`` 在普通线程里跑节点，而 MCP 的 ClientSession 是
异步的。这里用一个常驻后台线程 + 私有事件循环持有会话，同步方法通过
``run_coroutine_threadsafe`` 投递协程并等待结果。会话只建立一次（工具发现
和后续每次调用共用），避免每次调用都重新 initialize。
"""
from __future__ import annotations

import asyncio
import atexit
import json
import sys
import threading
from concurrent.futures import TimeoutError as FutureTimeout
from contextlib import AsyncExitStack
from pathlib import Path

from utils.logging_setup import get_logger

log = get_logger(__name__)

ROOT = Path(__file__).resolve().parent
SERVER_SCRIPT = ROOT / "mcp_server_ccus.py"

DEFAULT_TIMEOUT = 180.0     # 秒；图谱首次调用要加载模型，留足余量


def payload_of(result) -> dict:
    """把 ``CallToolResult`` 还原成 dict。

    MCP 只保证 ``content`` 里有文本块。服务端返回 dict 时会被序列化成 JSON
    文本，所以这里优先解析文本；``structured_content`` 只有在服务端显式声明
    output schema 时才非空。
    """
    structured = getattr(result, "structured_content", None)
    if structured:
        return structured
    for block in getattr(result, "content", None) or []:
        text = getattr(block, "text", None)
        if not text:
            continue
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return {"_text": text}
        return parsed if isinstance(parsed, dict) else {"_value": parsed}
    return {}


class CCUSToolClient:
    """MCP Client 的同步外观（线程安全的惰性连接）。"""

    def __init__(self, transport: str = "memory", timeout: float = DEFAULT_TIMEOUT):
        if transport not in ("memory", "stdio"):
            raise ValueError(f"transport 只能是 memory / stdio，收到 {transport!r}")
        self.transport = transport
        self.timeout = timeout
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._session = None
        self._stack: AsyncExitStack | None = None
        self._server_task: asyncio.Task | None = None
        self._error: str = ""

    # ─── 生命周期 ──────────────────────────────────────────
    def _ensure_started(self) -> None:
        with self._lock:
            if self._session is not None:
                return
            if self._error:
                raise RuntimeError(self._error)
            try:
                self._loop = asyncio.new_event_loop()
                self._thread = threading.Thread(
                    target=self._run_loop, name="mcp-client", daemon=True
                )
                self._thread.start()
                fut = asyncio.run_coroutine_threadsafe(self._connect(), self._loop)
                fut.result(timeout=self.timeout)
                log.info(f"  [MCP] 已连接 ccus-tools（{self.transport} 传输）")
            except Exception as exc:  # noqa: BLE001 — 连接失败要降级而不是崩
                self._error = f"MCP 连接失败：{type(exc).__name__}: {exc}"
                log.error(f"  [MCP] {self._error}")
                raise RuntimeError(self._error) from exc

    def _run_loop(self) -> None:
        assert self._loop is not None
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    async def _connect(self) -> None:
        """建立会话。内存模式与 stdio 模式的差别只有"管道"这一层。"""
        self._stack = AsyncExitStack()
        if self.transport == "memory":
            # 同进程：不加载第二份模型，见模块文档
            from mcp.shared.memory import create_client_server_memory_streams
            from mcp_server_ccus import server

            client_streams, server_streams = await self._stack.enter_async_context(
                create_client_server_memory_streams()
            )
            low = server._lowlevel_server  # noqa: SLF001 — 官方未暴露公共别名
            self._server_task = asyncio.create_task(
                low.run(
                    server_streams[0], server_streams[1],
                    low.create_initialization_options(),
                )
            )
            from mcp import ClientSession

            self._session = await self._stack.enter_async_context(
                ClientSession(client_streams[0], client_streams[1])
            )
        else:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client

            params = StdioServerParameters(
                command=sys.executable, args=[str(SERVER_SCRIPT)], cwd=str(ROOT)
            )
            read, write = await self._stack.enter_async_context(stdio_client(params))
            self._session = await self._stack.enter_async_context(
                ClientSession(read, write, read_timeout_seconds=self.timeout)
            )

        await self._session.initialize()

    def close(self) -> None:
        if self._loop is None or self._session is None:
            return
        try:
            async def _shutdown():
                if self._server_task is not None:
                    self._server_task.cancel()
                if self._stack is not None:
                    await self._stack.aclose()

            asyncio.run_coroutine_threadsafe(_shutdown(), self._loop).result(timeout=5)
        except Exception:  # noqa: BLE001 — 退出路径，失败不值得打扰调用方
            pass
        finally:
            self._session = None
            if self._loop is not None:
                self._loop.call_soon_threadsafe(self._loop.stop)

    # ─── 工具 ─────────────────────────────────────────────
    def list_tools(self) -> list[dict]:
        """向 Server 请求工具清单（动态发现，不在客户端硬编码）。"""
        self._ensure_started()
        result = self._submit(self._session.list_tools())
        specs = []
        for tool in result.tools:
            schema = tool.input_schema or {}
            specs.append({
                "name": tool.name,
                "description": tool.description or "",
                "properties": schema.get("properties") or {},
                "required": schema.get("required") or [],
            })
        return specs

    def call(self, name: str, arguments: dict | None = None) -> dict:
        """调用一个工具，返回其结构化结果。失败抛 RuntimeError。"""
        self._ensure_started()
        result = self._submit(
            self._session.call_tool(name, arguments or {}, read_timeout_seconds=self.timeout)
        )
        if getattr(result, "is_error", False):
            raise RuntimeError(f"工具 {name} 返回错误：{payload_of(result)}")
        return payload_of(result)

    def _submit(self, coro):
        assert self._loop is not None
        try:
            return asyncio.run_coroutine_threadsafe(coro, self._loop).result(
                timeout=self.timeout
            )
        except FutureTimeout as exc:
            raise RuntimeError(f"MCP 调用超时（{self.timeout:.0f}s）") from exc


_CLIENT: CCUSToolClient | None = None
_CLIENT_LOCK = threading.Lock()


def get_client(transport: str = "memory") -> CCUSToolClient:
    """进程内单例：会话只建立一次，工具发现与后续调用共用。"""
    global _CLIENT
    with _CLIENT_LOCK:
        if _CLIENT is None:
            _CLIENT = CCUSToolClient(transport=transport)
            atexit.register(_CLIENT.close)
        return _CLIENT


def reset_client() -> None:
    """测试用：丢弃单例（下次 get_client 会重新连接）。"""
    global _CLIENT
    with _CLIENT_LOCK:
        if _CLIENT is not None:
            _CLIENT.close()
        _CLIENT = None
