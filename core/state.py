"""
图状态定义（AgentState）
========================
LangGraph 的 StateGraph 以这个 TypedDict 作为通道声明的唯一来源。

单独成模块的理由很实际：**节点、图、API 三层都要引用它，但它不该依赖任何东西**。
放在这里之后，``from core.state import AgentState`` 不会连带加载 OpenAI client、
reranker 或任何重依赖，节点模块之间的循环导入风险也随之消失。

字段按功能分组（记忆 / 联网 / 反思 / 路由 / 文件记忆 / 图表 / Token 经济学），
新增字段时请归到对应分组并写明单位与来源 —— 这些字段最终会原样出现在
API 响应里，注释是唯一的文档。
"""
from __future__ import annotations

from typing import TypedDict


class AgentState(TypedDict):
    user_input: str
    extracted_params: str
    retrieved_knowledge: str
    final_report: str
    score: float
    feedback: str
    token_usage: dict
    # ─── 记忆系统字段 ───
    need_memory: bool              # Memory Router 判断是否需要搜记忆
    personal_memory: str           # 检索到的原始记忆内容
    personal_context: str          # 整理后注入给后续节点的上下文
    # ─── 联网搜索字段 ───
    web_content: str               # 联网搜索+自动沉淀的新内容
    # ─── 反思重试计数 ───
    retry_count: int               # 报告重写次数，用于防止无限循环
    # ─── 多模型路由 ───
    model_routing: dict            # {stage: "tier:model"}，记录每个角色实际用了哪个模型
    # ─── 文件记忆 ───
    file_memory_context: str       # 注入 Prompt 的文件记忆文本
    history_entry: str             # 本次评估压成的一条历史记录
    history_saved: bool            # 幂等标记，防止重跑重复追加
    memory_compression: dict       # 本次是否触发历史压缩及压缩统计
    # ─── 图表 ───
    chart_info: dict               # generate_pressure_curve 的结果（路径/结论）
    # ─── Token 经济学：动态路由 ───
    complexity_mode: str           # 本次请求判定的档位："local" | "cloud"
    complexity_info: dict          # 判定依据（字符数/是否命中/RAG 相关度/日志标签）
    # ─── Token 经济学：上下文压缩 ───
    context_pruning: dict          # 压缩统计（前后字符数/状态/原因）
    cost_estimation: dict          # 本次任务的成本估算（人民币）
    # ─── 物理一致性校验（规则引擎，不是模型打分）───
    physics_check: dict            # {status, blocking, verdict, limits, violations, feedback}
    # ─── 人类在环（HITL）───
    human_decision: dict           # {approved, reviewer, comment, auto}
    human_rejections: int          # 人工驳回次数，用于给"人一直驳回"设上限
