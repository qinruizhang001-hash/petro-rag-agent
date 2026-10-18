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

# 实体抽取的领域约束（只抽地质/工艺实体）。同样抽成独立模块：
# 它是 prompt 的一部分，值得被单测覆盖，也不该埋在构造参数里。
from utils.kg_prompt import addon_params as kg_addon_params

from utils.logging_setup import get_logger

log = get_logger(__name__)

WORKING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lightrag_storage")
OLLAMA_HOST = "http://localhost:11434"
LLM_MODEL = "qwen2.5"
EMBED_MODEL = "nomic-embed-text"


def _env_int(name: str, default: int) -> int:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        log.info(f"  [Config] {name}={raw!r} 不是合法整数，使用默认值 {default}")
        return default


# 检索广度。这两个数直接决定 rerank 的耗时 —— cross-encoder 是**线性**成本，
# 候选翻倍、延迟就翻倍。
#
# 实测（语料 5 篇 → 23 篇）：LightRAG 默认 top_k=40 / chunk_top_k=20 会展开成
# 120 个片段，本地 rerank 要 76 秒；整条工作流因此逼近 app.py 的 120 秒超时。
# 收到 25 / 12 之后约 50 个片段，延迟回到 30 秒级。
#
# 为什么敢收：rerank 的区分度很高（真实 CCUS 提问的 Top-1 分数 0.97，无关提问
# 0.0001），多出的 70 个候选并没有改变最终 Top-2 的选择。这不是"牺牲质量换速度"，
# 而是去掉了一批本来就不会被选中的候选。
LIGHTRAG_TOP_K = _env_int("LIGHTRAG_TOP_K", 25)
LIGHTRAG_CHUNK_TOP_K = _env_int("LIGHTRAG_CHUNK_TOP_K", 12)


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
                "options": {
                    "num_ctx": 8192,
                    # 生成硬上限。这不是性能调优，是**防挂死**：
                    # num_ctx 满了之后 llama.cpp 会滑动上下文窗口继续生成，
                    # 而 qwen2.5 在 JSON 模式下一旦开头被滑出视野就再也吐不出
                    # EOS —— 实测一次实体抽取生成了 15000+ token 仍在继续，
                    # 整条批量入库因此卡死 5 分钟没有任何进展。
                    # 4096 远高于单块（400 token）合法抽取所需的长度，
                    # 截断时 LightRAG 会解析失败并跳过该块，而不是无限等待。
                    "num_predict": 4096,
                },
                "timeout": 600,
            },
            # 适配 qwen2.5 (7.6B, 8192 上下文)
            entity_extraction_use_json=True,
            entity_extract_max_gleaning=0,
            max_entity_tokens=1500,
            max_relation_tokens=1500,
            max_total_tokens=4000,
            chunk_token_size=400,
            # 领域约束：只抽地质 / 井筒 / 材料 / 工艺 / 技术 / 参数 / 风险 / 法规
            # 八类实体，并显式排除作者单位、年份、街道地址、期刊名、DOI。
            # 这是 LightRAG 官方预留的覆盖点（渲染进 ---Entity Types--- 段），
            # 不需要改包、也不依赖 PROMPT_DIR 环境变量。
            addon_params=kg_addon_params(),
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

    def _submit(self, coro_factory, timeout: float = 600):
        """把协程投递到后台 loop 执行并同步取回结果。"""
        if self._loop is None or self._loop.is_closed():
            self._start()
        future = asyncio.run_coroutine_threadsafe(coro_factory(), self._loop)
        return future.result(timeout=timeout)

    # ─── 对外同步接口 ──────────────────────────────

    def insert_text(self, text: str, timeout: float = 1800):
        """增量建图。超时默认 30 分钟：一个 chunk 的实体抽取在 CPU 上就要几十秒，
        多 chunk 文档累计起来很容易超过检索用的 10 分钟。"""
        return self._submit(lambda: self._rag.ainsert(text), timeout=timeout)

    def query_data(self, query: str, mode: str = "hybrid") -> dict:
        """返回结构化检索结果（不经过 LLM 生成），供 rerank 使用。"""
        return self._submit(
            lambda: self._rag.aquery_data(
                query,
                param=QueryParam(
                    mode=mode,
                    enable_rerank=False,
                    top_k=LIGHTRAG_TOP_K,
                    chunk_top_k=LIGHTRAG_CHUNK_TOP_K,
                ),
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

    def delete_document(self, doc_id: str):
        """删除一篇文档及其在图谱中的贡献（用于清理卡死的半成品）。"""
        return self._submit(lambda: self._rag.adelete_by_doc_id(doc_id))

    def merge_entities(self, sources: list[str], target: str):
        """把一批实体并进 target（图 + 实体向量库 + 关系向量库同步更新）。"""
        return self._submit(lambda: self._rag.amerge_entities(sources, target))

    def delete_entity(self, name: str):
        """删除单个实体及其全部关系。"""
        return self._submit(lambda: self._rag.adelete_by_entity(name))


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


def insert_text(text: str, timeout: float = 1800) -> bool:
    """
    把一段文本**增量**写入知识图谱（建图入口）。

    供 ``ingest_ccus_batch.py`` 等离线入库脚本使用。之所以必须走这个模块、
    而不是在入库脚本里另建一个 LightRAG 实例：建图与检索必须共用同一份配置
    （chunk_token_size / entity 抽取参数 / 同一个 working_dir 与嵌入模型）。
    两套参数建出来的图和查询时的切分粒度对不上，检索会静默退化 —— 不报错，
    只是召回变差，这种问题最难查。

    返回 True 表示成功。失败只记日志并返回 False，不抛异常：批量入库时
    单篇论文失败不应该中断整个流程。
    """
    try:
        bridge = _get_bridge()
        bridge.insert_text(text, timeout=timeout)
        return True
    except Exception as e:
        log.warning(f"  [LightRAG] 增量建图异常: {e}")
        return False


def delete_document(doc_id: str) -> bool:
    """
    删除一篇文档及其对图谱的贡献。

    批量入库必须有这条路径：进程被杀 / 超时中断时，LightRAG 会把文档留在
    ``status=processing``，而**后续所有插入都会被这个状态挡住** —— 重跑
    不会报错，只会安静地什么都不做。所以恢复流程必须是"先删掉半成品，
    再重新插入"，而不是"再插一次试试"。
    """
    try:
        bridge = _get_bridge()
        bridge.delete_document(doc_id)
        return True
    except Exception as e:
        log.warning(f"  [LightRAG] 删除文档异常: {e}")
        return False


def merge_entities(sources: list[str], target: str) -> bool:
    """
    把多个实体并成一个（别名归并的执行端）。

    必须走 LightRAG 自己的 ``amerge_entities``，而不是直接改 graphml ——
    它同时维护图存储、实体向量库、关系向量库和 chunk 追踪表。只改 graphml
    会让向量库里留下指向已消失节点的向量，检索时命中一个"图里不存在的实体"，
    这种不一致不会报错，只会让召回莫名其妙地变差。
    """
    try:
        bridge = _get_bridge()
        bridge.merge_entities(sources, target)
        return True
    except Exception as e:
        log.warning(f"  [LightRAG] 合并实体异常 {target!r}: {e}")
        return False


def delete_entity(name: str) -> bool:
    """
    删除单个实体（脏节点清理的执行端）。

    ``adelete_by_entity`` 会连带删除该实体的所有关系并清理 chunk 追踪，
    比手工删 graphml 里的 ``<node>`` 安全 —— 后者会留下悬空边。
    """
    try:
        bridge = _get_bridge()
        bridge.delete_entity(name)
        return True
    except Exception as e:
        log.warning(f"  [LightRAG] 删除实体异常 {name!r}: {e}")
        return False
