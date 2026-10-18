"""
CCUS 多智能体风险评估系统 —— 入口文件
=====================================
本文件**不再包含任何逻辑**，只做导入转发。实现已按包结构拆分：

    core/        配置（OpenAI client / API Key / 模型路由）、图状态、共享单例
    nodes/       每个节点一个文件，辅助函数在同目录的 conditions / *_node 里
    utils/       token 统计、上下文压缩、成本核算（零重依赖，可离线单测）
    workflow.py  LangGraph 的节点注册、边、条件路由与编译

工具注册（Function Calling）不在本文件，见 nodes/aggregate_node.py 的
Phase A（plan_prompt → 解析 → 执行 → 兜底），实现散在 pinn_model.py /
tools/physics_tools.py / utils/pinn_query.py。加逻辑到这里会让"入口"变成
第二个实现点 —— CI 有一条 40 行上限的守卫专门防这件事。

两种用法都保持不变：

    from multi_agent_ccus import app    # 程序化调用
    python multi_agent_ccus.py          # CLI 演示（转发到 workflow.main）

新代码建议直接 ``from workflow import app``，不要依赖这个兼容入口。
"""
from workflow import app

__all__ = ["app"]


if __name__ == "__main__":
    from workflow import main

    main()
