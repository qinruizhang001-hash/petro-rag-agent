"""
共享的重型单例：bge-reranker-v2-m3
==================================
两个地方要用它 —— 检索节点做精排，token 统计在 tiktoken 不可用时退回到
它的 tokenizer。

**为什么必须共用一份**：这个模型单份约 2.3 GB。在同一个进程里加载第二份会
直接耗尽 Windows 页面文件，抛 ``OSError 1455``，而表面报错是
``Unrecognized processing class`` —— 一个指向完全错误方向的症状。
端到端测试里因此显式复用这个实例，而不是 new 一个。

导入本模块即触发加载（与原实现一致），所以不要把它 import 进纯逻辑模块。
"""
from sentence_transformers import CrossEncoder

from utils.logging_setup import get_logger

log = get_logger(__name__)

# 本地 Rerank 只加载一次，避免重复占内存
log.info("正在加载本地 Rerank 重排模型...")
reranker = CrossEncoder('./models/bge-reranker-v2-m3', max_length=512)
log.info("✅ Rerank 模型加载完成！")
