"""
LightRAG GraphRAG 检索封装
==========================
把 LightRAG 的异步接口封装成同步函数，供 multi_agent_ccus.py 的同步节点调用。

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
                print("[LightRAG] 首次初始化，加载知识图谱存储与后台 event loop...")
                b = _LightRAGBridge()
                b._start()
                _bridge = b
                print("[LightRAG] 初始化完成。")
    return _bridge


def retrieve_documents(query: str, mode: str = "hybrid") -> list[str]:
    """
    同步检索，返回候选片段列表，供 rerank 精排。

    片段构建策略（提升 Rerank 区分度）：
    1. 增强关系片段：把两端实体的完整描述拼进关系片段，
       形成 "实体A描述 --[关系关键词]--> 实体B描述" 的长语义单元，
       每个片段同时携带实体语义和关系语义（多跳推理上下文完整）。
    2. 孤立实体片段：未被任何关系覆盖到的实体单独保留，避免信息丢失。
    3. 原始 chunk：文本块原样保留。

    返回空列表表示图谱为空或无相关结果。
    """
    try:
        bridge = _get_bridge()
        result = bridge.query_data(query, mode=mode)
    except Exception as e:
        print(f"  [LightRAG] 检索异常: {e}")
        return []

    if not result or result.get("status") != "success":
        return []

    data = result.get("data", {})
    entities = data.get("entities", []) or []
    relationships = data.get("relationships", []) or []
    chunks = data.get("chunks", []) or []

    # ─── 1. 实体索引：name -> 完整描述串 ───────────────
    entity_map = {}
    for ent in entities:
        name = ent.get("entity_name", "")
        if not name:
            continue
        etype = (ent.get("entity_type", "") or "UNKNOWN").strip()
        desc = (ent.get("description", "") or "").strip()
        entity_map[name] = f"[{etype}] {name}: {desc}" if desc else f"[{etype}] {name}"

    # ─── 2. 增强关系片段：拼接两端实体描述，扩充文本长度 ──
    covered = set()
    docs = []
    for rel in relationships:
        src = rel.get("src_id", "")
        tgt = rel.get("tgt_id", "")
        desc = (rel.get("description", "") or "").strip()
        kw = (rel.get("keywords", "") or "").strip()
        if not (src or tgt or desc):
            continue
        src_part = entity_map.get(src) or src
        tgt_part = entity_map.get(tgt) or tgt
        edge = f"--[{kw}]-->" if kw else "-->"
        piece = f"{src_part} {edge} {tgt_part}"
        if desc:
            piece = f"{piece}: {desc}"
        docs.append(f"[关系] {piece}")
        covered.update({src, tgt})

    # ─── 3. 孤立实体片段（未被关系覆盖，避免与关系片段重复）──
    for name, full_desc in entity_map.items():
        if name not in covered:
            docs.append(f"[实体] {full_desc}")

    # ─── 4. 原始文档 chunk ─────────────────────────────
    for chunk in chunks:
        content = (chunk.get("content", "") or "").strip()
        if content:
            docs.append(content)

    # 去重保序
    seen = set()
    unique = []
    for d in docs:
        if d and d not in seen:
            seen.add(d)
            unique.append(d)

    return unique


def query_graph_answer(query: str, mode: str = "hybrid") -> str:
    """同步查询，返回 LightRAG 的 LLM 最终回答（用于对比/回退）。"""
    try:
        bridge = _get_bridge()
        return bridge.query(query, mode=mode)
    except Exception as e:
        print(f"  [LightRAG] 查询异常: {e}")
        return ""
