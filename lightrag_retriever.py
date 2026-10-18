"""
LightRAG GraphRAG 检索封装
==========================
把 LightRAG 的异步接口封装成同步函数，供 nodes/retrieve_node.py 等同步节点调用。

设计要点：
- LightRAG 的存储锁绑定其初始化时的 event loop，因此用一个后台线程
  常驻一个 event loop，所有异步调用（初始化/建图/查询）都投递到该 loop 执行。
- 对外暴露同步的 retrieve(query) → 返回结构化片段列表（entities/relations/chunks），
  供上游的 bge-reranker 做精排。
"""
import asyncio
import os
import threading
from functools import partial

# ─── 清理代理环境变量（httpx 会读 http_proxy，导致 localhost Ollama 超时）───
for _k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
    os.environ.pop(_k, None)
os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
os.environ.setdefault("no_proxy", "localhost,127.0.0.1")

from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.utils import EmbeddingFunc

# 片段构建逻辑抽到纯函数模块，便于离线单元测试（CI 无需 Ollama）
from fragment_builder import build_fragments, extract_data

from utils.logging_setup import get_logger

log = get_logger(__name__)

WORKING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lightrag_storage")
OLLAMA_HOST = "http://localhost:11434"
LLM_MODEL = "qwen2.5"
EMBED_MODEL = "nomic-embed-text"


class _LightRAGBridge:
    """后台线程常驻 event loop，封装 LightRAG 的异步生命周期。"""

    def __init__(self):
        self._loop = None
        self._thread = None
        self._rag = None
        self._ready = threading.Event()

    def _start(self):
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop_forever, daemon=True)
        self._thread.start()
        self._ready.wait(timeout=120)

    def _loop_forever(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._init())
        self._ready.set()
        self._loop.run_forever()

    async def _init(self):
        self._rag = LightRAG(
            working_dir=WORKING_DIR,
            llm_model_func=ollama_model_complete,
            llm_model_name=LLM_MODEL,
            llm_model_kwargs={
                "host": OLLAMA_HOST,
                "options": {"num_ctx": 8192},
                "timeout": 600,
            },
            # 适配 qwen2.5 (7.6B, 8192 上下文)
            entity_extraction_use_json=True,
            entity_extract_max_gleaning=0,
            max_entity_tokens=1500,
            max_relation_tokens=1500,
            max_total_tokens=4000,
            chunk_token_size=400,
            embedding_func=EmbeddingFunc(
                embedding_dim=768,
                max_token_size=8192,
                func=partial(
                    ollama_embed.func,
                    embed_model=EMBED_MODEL,
                    host=OLLAMA_HOST,
                ),
            ),
        )
        await self._rag.initialize_storages()

    def _submit(self, coro_factory):
        """把协程投递到后台 loop 执行并同步取回结果。"""
        if self._loop is None or self._loop.is_closed():
            self._start()
        future = asyncio.run_coroutine_threadsafe(coro_factory(), self._loop)
        return future.result(timeout=600)

    # ─── 对外同步接口 ──────────────────────────────

    def insert_text(self, text: str):
        return self._submit(lambda: self._rag.ainsert(text))

    def query_data(self, query: str, mode: str = "hybrid") -> dict:
        """返回结构化检索结果（不经过 LLM 生成），供 rerank 使用。"""
        return self._submit(
            lambda: self._rag.aquery_data(
                query,
                param=QueryParam(mode=mode, enable_rerank=False),
            )
        )

    def query(self, query: str, mode: str = "hybrid") -> str:
        """返回 LLM 生成的最终回答。"""
        return self._submit(
            lambda: self._rag.aquery(
                query,
                param=QueryParam(mode=mode, enable_rerank=False),
            )
        )


# 模块级单例（懒加载，首次调用时才启动后台线程和加载模型）
_bridge = None
_bridge_lock = threading.Lock()


def _get_bridge() -> _LightRAGBridge:
    global _bridge
    if _bridge is None:
        with _bridge_lock:
            if _bridge is None:
                log.info("[LightRAG] 首次初始化，加载知识图谱存储与后台 event loop...")
                b = _LightRAGBridge()
                b._start()
                _bridge = b
                log.info("[LightRAG] 初始化完成。")
    return _bridge


def retrieve_documents(query: str, mode: str = "hybrid") -> list[str]:
    """
    同步检索，返回候选片段列表，供 rerank 精排。

    片段构建策略见 fragment_builder.build_fragments（纯函数，可离线单测）。
    返回空列表表示图谱为空或无相关结果。
    """
    try:
        bridge = _get_bridge()
        result = bridge.query_data(query, mode=mode)
    except Exception as e:
        log.warning(f"  [LightRAG] 检索异常: {e}")
        return []

    return build_fragments(extract_data(result))


def query_graph_answer(query: str, mode: str = "hybrid") -> str:
    """同步查询，返回 LightRAG 的 LLM 最终回答（用于对比/回退）。"""
    try:
        bridge = _get_bridge()
        return bridge.query(query, mode=mode)
    except Exception as e:
        log.warning(f"  [LightRAG] 查询异常: {e}")
        return ""
