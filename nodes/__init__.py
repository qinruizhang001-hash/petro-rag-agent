"""
nodes —— LangGraph 的节点函数，一个文件一个职责。

导入本包会先触发 ``core/__init__.py`` 的环境预置。这一点是必要的：
``retrieve_node`` 会 import ``lightrag_retriever``，后者连带 lightrag 与
sentence_transformers，必须在这些重依赖之前完成 HF_HUB_OFFLINE 与 sys.path 设置。
"""
import core  # noqa: F401  （仅为触发 core/__init__ 的预置，见其 docstring）
