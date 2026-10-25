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
   ``{stage: payload}``，并行分支写不同 stage；合并后从各 stage 的真实
   ``total_tokens`` 重算 ``llm_total_tokens``，不能直接采用任一分支的标量总数。
   如果哪天两个分支写同一个 stage，后写的 payload 会覆盖先写的，必须靠
   stage 命名区分来保证。

``final_report`` 仍然存在且仍是唯一对外契约：aggregate 节点负责把两段结论
合成它，API / 前端 / 评审 / 历史记忆全都只读这一个字段。
"""
from __future__ import annotations

from typing import Annotated, TypedDict


def merge_dict(left: dict | None, right: dict | None) -> dict:
    """Merge parallel dict updates and recompute the aggregate LLM token count.

    ``token_usage`` and ``model_routing`` share this reducer. Stage keys are
    branch-specific, so right-biased shallow merging preserves each result. The
    scalar ``llm_total_tokens`` is not branch-specific: summing or choosing either
    branch value would be wrong because each branch starts from the same snapshot.
    Recompute it from the merged per-stage LLM payloads instead.
    """
    merged = {**(left or {}), **(right or {})}
    if "llm_total_tokens" in merged:
        merged["llm_total_tokens"] = sum(
            int(payload.get("total_tokens") or 0)
            for key, payload in merged.items()
            if key != "llm_total_tokens" and isinstance(payload, dict)
        )
    return merged


class AgentState(TypedDict):
    user_input: str
    extracted_params: str
    # Query rewriting preserves the original question and creates branch-specific
    # English search terms; these are not evidence and must never replace user_input.
    geology_query: str
    wellbore_query: str
    query_rewrite_status: dict
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
    geology_evidence: str          # 通过 Self-RAG 评估后注入 Prompt 的图谱证据
    wellbore_evidence: str         # 通过 Self-RAG 评估后注入 Prompt 的向量库证据
    supplemental_evidence: str     # 地质分支降级时 aggregate 补取的图谱片段（同样进入 Prompt）
    tool_facts: list               # 注入报告 Prompt 的确定性工具结果（密度 / 压力曲线 / PINN / 风险概率）
    geology_candidates: list        # Grader 评判的地质 Top-K 原始片段
    wellbore_candidates: list       # Grader 评判的井筒 Top-K 原始片段
    retrieval_grade: dict           # Self-RAG 的分支相关性评分与重试状态
    retrieval_retry_count: int      # 有界的替代检索次数（最多一次）
    retrieval_strategy: str         # primary | expanded
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
    # ─── 质量闸门（确定性判定，见 core/gates.py）───
    # 报告到底过没过，只读这一个字段。早期实现靠"score>=8 或重写两次"隐式表达，
    # 于是"重试用尽后的放行"和"真的达标"在状态里无法区分。
    quality_gate: dict             # {status, score, threshold, physics_status, physics_blocking, reason, ...}
    report_status: str             # "ok" | "failed"；失败态由 mark_failed 写入
    draft_report: str              # 物理否决时被隔离的原始正文（可审计，不交付）
    # ─── 时间维度记忆（Temporal Knowledge）───
    # retrieve 节点产出。conflicts 是"同一井位 + 同一参数 + 不同时间 + 数值显著不同"
    # 的结构化列表；context 是注入给下游的硬指令原文（可审计）。
    temporal_conflicts: list
    temporal_context: str
    temporal_stats: dict           # {claims, conflicts, entities}
    # ─── 不确定性量化（Monte Carlo）───
    # aggregate 节点产出：{density: {...}, injection: {...}, sensitivity: {...},
    # decision: {...}}。每项含 mean / p5 / p95 / std / n / rel_sigma /
    # dist_label / text。数值全部来自代码模拟，模型只负责转述，不负责计算。
    # ``dist_label`` 不能省：同样写"±5%"，均匀与正态的 90% 区间差一倍宽。
    uq_summary: dict
    # ─── 风险决策（UQ 概率 × 阈值）───
    # 由 ``core.risk_policy.decide`` 产出：{level, triggered, resolved, p,
    # p_ci, rule, thresholds, notes, text}。与 ``uq_summary`` 分开是因为
    # 它回答的是另一个问题：UQ 说"概率是多少"，这里说"要不要告警"。
    # ``resolved=False`` 表示概率的置信区间跨越阈值边界 —— 等级仍给出，
    # 但不可单独据此下结论（100 次模拟下 5% 与 8% 的区别在噪声里）。
    risk_decision: dict
    # ─── 深层物理判定（断层活化 / 盖层封闭性 / 两相流）───
    # aggregate 节点产出：{status, blocking, geomechanics, caprock, relperm,
    # policy, assumptions, version}。``status="no_data"`` 表示参数不足未算，
    # 与"算了但不合格"是两件事，不能混。evaluate 节点会用同一套规则重算一遍
    # 并据此阻断，所以这里的字段是**展示与审计**用的，不是闸门的唯一依据。
    physics_deep: dict
    # ─── 人类在环（HITL）───
    human_decision: dict           # {approved, reviewer, comment, auto}
    human_rejections: int          # 人工驳回次数，用于给"人一直驳回"设上限
