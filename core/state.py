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

并行分支（Fan-out / Fan-in）
---------------------------
``extract`` 之后同时触发 ``geology`` 与 ``wellbore``，两者在同一个 superstep 里
**并发**写状态。这带来两条硬约束，写错了不会报"结果不对"，而是直接抛异常
或静默丢数据：

1. **同一个 key 被两个分支同时写 → LangGraph 抛 ``InvalidUpdateError``**
   （``At key 'x': Can receive only one value per step``）。所以 ``token_usage``
   与 ``model_routing`` 这两个"两条分支都要写"的字段必须带 reducer；
   而分支各自的产物（``geology_report`` / ``wellbore_report``）用**独立字段**，
   从结构上就不存在覆盖问题 —— 这比给一个共享字段写合并规则更可靠。
2. **reducer 只对"可以合并"的字段成立**。``token_usage`` 是
   ``{stage: payload}``，两条分支写的是不同的 stage（``retrieve`` /
   ``chroma_retrieve``），浅合并即可；如果哪天两个分支写同一个 stage，
   后写的会覆盖先写的，这一点必须靠 stage 命名区分来保证。

``final_report`` 仍然存在且仍是唯一对外契约：aggregate 节点负责把两段结论
合成它，API / 前端 / 评审 / 历史记忆全都只读这一个字段。
"""
from __future__ import annotations

from typing import Annotated, TypedDict


def merge_dict(left: dict | None, right: dict | None) -> dict:
    """并行分支写 dict 字段时的合并器：浅合并，右值优先。

    用于 ``token_usage`` / ``model_routing``。两者都是
    ``{key: payload}`` 的扁平结构，且两条并行分支写的 key 不同
    （见模块文档），因此浅合并既够用也不会丢数据。
    """
    return {**(left or {}), **(right or {})}


class AgentState(TypedDict):
    user_input: str
    extracted_params: str
    retrieved_knowledge: str
    final_report: str
    score: float
    feedback: str
    # ─── 并行分支产物（Fan-out / Fan-in）───
    # 为什么拆成三个字段而不是共用一个 final_report：
    # geology 与 wellbore 在同一 superstep 里并发执行，共用一个字段必然互相覆盖
    # （LangGraph 会直接抛 InvalidUpdateError，若加了 reducer 则变成"后写的赢"，
    # 两种结局都是丢数据）。独立字段让"谁写了什么"在状态里永远可分辨。
    geology_report: str            # geology 分支：断层 / 盖层风险评估片段
    wellbore_report: str           # wellbore 分支：井筒腐蚀 / 压力异常评估片段
    geology_evidence: str          # geology 分支实际注入 Prompt 的图谱证据
    wellbore_evidence: str         # wellbore 分支实际注入 Prompt 的向量库证据
    geology_status: dict           # {ok, elapsed_s, error, engine, candidates, top_score}
    wellbore_status: dict          # {ok, elapsed_s, error, engine, candidates}
    parallel_timing: dict          # {geology_s, wellbore_s, wall_s, serial_s, speedup, ...}
    branch_merge: dict             # {degraded, missing[], geology_ok, wellbore_ok, reasons{}}
    # ─── Token / 路由（两条分支并发写，必须带 reducer）───
    token_usage: Annotated[dict, merge_dict]
    model_routing: Annotated[dict, merge_dict]
    # ─── 记忆系统字段 ───
    need_memory: bool              # Memory Router 判断是否需要搜记忆
    personal_memory: str           # 检索到的原始记忆内容
    personal_context: str          # 整理后注入给后续节点的上下文
    # ─── 联网搜索字段 ───
    web_content: str               # 联网搜索+自动沉淀的新内容
    # ─── 自我进化：历史先验 ───
    # extract 节点查 system_experience 后实际拼进 prompt 的文本块（含标题与
    # 使用说明）；没有命中任何先验时为空串。留着它是为了可审计：报告里那句
    # "参考了历史经验"必须有对应的原文，否则无法判断它是不是模型编的。
    system_experience: str
    # ─── 反思重试计数 ───
    retry_count: int               # 报告重写次数，用于防止无限循环
    # ─── 文件记忆 ───
    file_memory_context: str       # 注入 Prompt 的文件记忆文本
    history_entry: str             # 本次评估压成的一条历史记录
    history_saved: bool            # 幂等标记，防止重跑重复追加
    memory_compression: dict       # 本次是否触发历史压缩及压缩统计
    # ─── 图表 ───
    chart_info: dict               # generate_pressure_curve 的结果（路径/结论）
    # ─── PINN 压力预测 ───
    # {x_m, t_days, pressure_bar, scenario, text}；未调用时为空 dict。
    # 只有 aggregate 写这个字段（上游没有并发的写者），所以不需要 reducer。
    pinn_prediction: dict
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
