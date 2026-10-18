"""
CCUS 多智能体风险评估系统 —— 入口文件
=====================================
本文件**不再包含任何逻辑**，只做导入转发。实现已按包结构拆分：

    core/        配置（OpenAI client / API Key / 模型路由）、图状态、共享单例
    nodes/       每个节点一个文件，辅助函数在同目录的 conditions / *_node 里
    utils/       token 统计、上下文压缩、成本核算（零重依赖，可离线单测）
    workflow.py  LangGraph 的节点注册、边、条件路由与编译

工具注册（Function Calling）在哪里
----------------------------------
不在本文件里。本文件只是入口转发，加逻辑会让"入口"变成第二个实现点。
工具调用分三处，各有分工：

    nodes/aggregate_node.py   Phase A 的 plan_prompt（告诉模型有哪些工具）
                              + 解析 JSON 指令 + 执行 + 确定性兜底
    pinn_model.py             predict_pressure 的实现（PINN，惰性 import）
    utils/pinn_query.py       从问句里抽 (x, t) 的纯函数，供兜底触发

要新增一个工具，三处都要动：prompt 里描述它、dispatch 里执行它、
兜底逻辑里决定"模型没规划时要不要自己调"。漏掉第三处的症状是
"模型心情好才会用"——本地 7B 对上下文长度很敏感，同一句话换个说法
就可能不再规划。

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
