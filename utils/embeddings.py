"""
ChromaDB 嵌入函数（单一来源）
=============================
CCUS 知识库的写入端（``ingest_ccus_batch.py``）与查询端
（``tools/knowledge_tools.py``）**必须用同一个嵌入模型**。两边不一致的后果
不是报错，而是更糟的两种情形：

  - 维度不同 → 查询时 Chroma 直接抛维度不匹配（至少能被发现）
  - 维度相同但模型不同 → 查询不报错，只是**召回全是噪声**（几乎无法发现）

所以这个选择只允许有一个地方做。

这里默认用 ChromaDB 内置的 ``all-MiniLM-L6-v2``（384 维，ONNX，随 chromadb
一起分发、无需联网）。选它有两个具体原因，不是"默认值最省事"：

  1. **与存量数据一致。** ``ccus_knowledge_v2`` 里 671 条向量都是 384 维，
     说明当初就是用它建的。换成 512 维的 ``bge-small-zh-v1.5`` 会让存量
     数据不可用 —— 事实上这个 bug 真实存在过：``knowledge_tools.py`` 曾
     硬编码 ``BAAI/bge-small-zh-v1.5``（512 维）去查 384 维的库，查询路径
     一直是坏的。
  2. **离线可用。** 模型随 chromadb 分发并已缓存，不依赖 HuggingFace 连通性。
     本机实测 HuggingFace 不可达，任何需要现场下载模型的方案都不可行。

需要换更强的中文模型时，设 ``EMBEDDING_MODEL`` 环境变量即可，但**必须整库
重建**（写入端与查询端一起换），否则就是上面说的第二种情形。
"""
from __future__ import annotations

import os


def embedding_model_name() -> str:
    """当前生效的嵌入模型名。空串表示用 ChromaDB 内置模型。"""
    return os.getenv("EMBEDDING_MODEL", "").strip()


def get_embedding_function(model_name: str | None = None):
    """
    返回 ChromaDB 的嵌入函数。

    ``model_name`` 为空 → 内置 ``all-MiniLM-L6-v2``；
    非空 → 用 SentenceTransformer 加载指定模型（需要本地已缓存或能联网）。
    """
    from chromadb.utils import embedding_functions

    name = (model_name if model_name is not None else embedding_model_name()).strip()
    if not name:
        return embedding_functions.DefaultEmbeddingFunction()
    return embedding_functions.SentenceTransformerEmbeddingFunction(model_name=name)


def describe() -> str:
    """给日志用的一句话描述。"""
    name = embedding_model_name()
    return f"{name} (sentence-transformers)" if name else "all-MiniLM-L6-v2 (chromadb 内置, 384 维)"
