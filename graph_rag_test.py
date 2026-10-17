"""
LightRAG GraphRAG 建图 + 多跳推理测试
=====================================
用本地 Ollama (qwen2.5 + nomic-embed-text) 把 CCUS 文本构建成知识图谱，
再通过图谱路径回答多跳推理问题。

运行:
    python graph_rag_test.py
"""
import asyncio
import logging
import os
from functools import partial

# ─── 清理代理环境变量（关键！）───────────────────────────
# LightRAG 的 ollama 模块用 httpx，会默认读取 http_proxy/https_proxy。
# 本机有代理变量时，访问 localhost:11434 的 Ollama 请求会被路由到代理而超时。
for _k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
    os.environ.pop(_k, None)
# 显式告知 httpx/ollama 不代理本地地址
os.environ["NO_PROXY"] = "localhost,127.0.0.1"
os.environ["no_proxy"] = "localhost,127.0.0.1"

from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.utils import EmbeddingFunc

# ─── 配置 ────────────────────────────────────────────────
WORKING_DIR = "./lightrag_storage"
OLLAMA_HOST = "http://localhost:11434"
LLM_MODEL = "qwen2.5"               # 实体/关系抽取 + 问答
EMBED_MODEL = "nomic-embed-text"    # 768 维本地嵌入
CCUS_TEXT_PATH = "./ccus_data.txt"

# 多跳推理测试问题
TEST_QUERY = "断层活化和盖层完整性之间有什么关联？"


def configure_logging():
    """把 LightRAG 的日志打到控制台，方便看实体抽取进度。"""
    logger = logging.getLogger("lightrag")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.handlers = []
    logger.addHandler(handler)


def build_rag() -> LightRAG:
    """初始化 LightRAG，强制使用本地 Ollama。"""
    print(f"\n[LightRAG] 初始化：LLM={LLM_MODEL}，Embedding={EMBED_MODEL}，host={OLLAMA_HOST}")
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=ollama_model_complete,
        llm_model_name=LLM_MODEL,
        llm_model_kwargs={
            "host": OLLAMA_HOST,
            "options": {"num_ctx": 8192},
            "timeout": 600,          # Ollama 单次请求超时（实体抽取较慢，放宽到 10 分钟）
        },
        # ─── 针对 qwen2.5 (7.6B, 8192 上下文) 的适配 ──────
        entity_extraction_use_json=True,   # JSON 结构化输出，小模型更稳定
        entity_extract_max_gleaning=0,     # 关闭 gleaning（二次补抽），本地模型慢，提速
        max_entity_tokens=1500,            # 缩小实体抽取预算，避免超出 8192 上下文
        max_relation_tokens=1500,
        max_total_tokens=4000,
        chunk_token_size=400,              # 小 chunk，降低单次抽取负载
        embedding_func=EmbeddingFunc(
            embedding_dim=768,
            max_token_size=8192,
            func=partial(
                ollama_embed.func,  # 用未包装的原始函数，避免二次 EmbeddingFunc 包装
                embed_model=EMBED_MODEL,
                host=OLLAMA_HOST,
            ),
        ),
    )
    return rag


def load_ccus_text() -> str:
    """读取 CCUS 测试文本；文件不存在时用内置文本兜底。"""
    if os.path.exists(CCUS_TEXT_PATH):
        with open(CCUS_TEXT_PATH, "r", encoding="utf-8") as f:
            text = f.read().strip()
        print(f"[LightRAG] 已读取本地文本 {CCUS_TEXT_PATH}，长度 {len(text)} 字")
        return text

    # 兜底：手写一段含断层/盖层/注入压力实体的 CCUS 文本
    fallback = (
        "CCUS地质封存的核心风险在于盖层完整性与断层稳定性。"
        "盖层通常为泥岩或膏盐层，具有特定的突破压力；当注入压力超过盖层突破压力时，可能引发CO2泄漏。"
        "断层活化是另一重大风险：长期高压注入会使储层孔隙压力升高，降低断层面有效正应力，诱发微地震或沿断层泄漏。"
        "因此，选址时必须评估盖层突破压力、避开活动断层，并通过时移地震和微地震监测追踪断层活化与盖层破裂信号。"
    )
    print(f"[LightRAG] 未找到 {CCUS_TEXT_PATH}，使用内置测试文本（{len(fallback)} 字）")
    return fallback


async def main():
    configure_logging()

    rag = build_rag()
    await rag.initialize_storages()

    text = load_ccus_text()

    # ─── Step 2: 建图（实体/关系抽取，耗时较长） ──────
    print("\n[LightRAG] 开始抽取实体与关系，构建知识图谱（耗时较长，请耐心等待）...")
    await rag.ainsert(text)
    print("[LightRAG] 知识图谱构建完成，已保存到", os.path.abspath(WORKING_DIR))

    # ─── Step 3: 多跳推理测试 ──────
    print("\n" + "=" * 50)
    print(f"[多跳推理测试] 问题：{TEST_QUERY}")
    print("=" * 50)

    for mode in ("hybrid", "global"):
        print(f"\n───── 查询模式: {mode} ─────")
        # enable_rerank=False：本测试不配内置 rerank，多智能体里会用自己的 bge-reranker
        resp = await rag.aquery(TEST_QUERY, param=QueryParam(mode=mode, enable_rerank=False))
        print(resp)

    await rag.llm_response_cache.index_done_callback()
    await rag.finalize_storages()


if __name__ == "__main__":
    asyncio.run(main())
    print("\n[Done] 图谱查询测试完成！")
