"""
ChromaDB 向量检索（井筒分支专用）
=================================
``lightrag_retriever.py`` 的对称物：那边查知识图谱（实体/关系/chunk），
这边查 ChromaDB 的原文切片。

为什么两条分支要用**不同**的存储
--------------------------------
这不是为了"看起来像架构"，而是两种检索的失效模式不同：

- **LightRAG** 把论文抽成实体与关系，擅长回答"断层活化与注入压力之间是什么
  关系"—— 多跳、跨论文、需要图谱才连得起来的地质问题。
- **ChromaDB** 存的是原文切片，擅长回答"套管腐蚀速率在哪些文献里被提到过"——
  单点、字面接近、不需要跨论文推理的工程问题。

地质问题用向量库会召回一堆零散句子，井筒问题用图谱会被实体抽取的粒度拖累
（"环空带压"在图上可能压根没被抽成实体）。所以按问题性质分库检索，
再在 aggregate 里合并 —— 而不是一个库查两次。

集合与嵌入函数必须与写入端一致
------------------------------
这里**不**自己 new 一个 PersistentClient，而是通过
``tools.knowledge_tools.get_collection()`` 复用同一个集合。原因写在
``utils/embeddings.py`` 里：本项目曾经用 512 维的 ``bge-small-zh-v1.5``
去查一个 384 维的库，查询路径一直是坏的且不报错 —— 只是召回质量悄悄变差。
集合名、路径、嵌入函数只能有**一处**定义。
"""
from __future__ import annotations

from tools.knowledge_tools import get_collection
from utils.logging_setup import get_logger

log = get_logger(__name__)


def retrieve_documents(query: str, k: int = 4) -> list[str]:
    """从 CCUS 原文库召回与 ``query`` 最相近的 ``k`` 个切片。

    返回按相似度降序的文本列表；库里没有数据或查询失败时返回空列表
    （调用方负责把"空"翻译成对用户有意义的说法，这里不替它决定）。
    """
    query = (query or "").strip()
    if not query:
        return []

    try:
        collection = get_collection()
    except Exception as exc:  # noqa: BLE001 — 初始化失败不该炸掉整条分支
        log.warning(f"  [Chroma] 集合初始化失败：{type(exc).__name__}: {exc}")
        return []

    if collection is None:
        log.warning("  [Chroma] 集合未就绪，跳过向量检索")
        return []

    count = collection.count()
    if count == 0:
        log.warning("  [Chroma] 集合为空，跳过向量检索")
        return []

    try:
        result = collection.query(
            query_texts=[query],
            n_results=max(1, min(k, count)),
        )
    except Exception as exc:  # noqa: BLE001 — 检索失败不该炸掉整条分支
        log.warning(f"  [Chroma] 查询失败：{type(exc).__name__}: {exc}")
        return []

    docs = (result.get("documents") or [[]])[0]
    # 过滤空切片：空字符串进了 Prompt 只会浪费 token
    return [d for d in docs if d and d.strip()]


def collection_size() -> int:
    """库里的切片总数（用于日志与自检；失败返回 0）。"""
    try:
        collection = get_collection()
        return int(collection.count()) if collection is not None else 0
    except Exception:  # noqa: BLE001
        return 0
