"""
utils —— 与图结构无关的横切工具：token 统计、上下文压缩、成本核算。

这些模块**不 import 任何节点或图**，因此可以被离线单元测试直接加载，
CI 里零第三方依赖即可验证（见 .github/workflows/ci.yml 的 unit-core）。
本 __init__ 也刻意保持为空：一旦在这里 import 了重依赖，
上面那个"零依赖可测"的前提就没了。
"""
