"""
Agent 1：参数提取
=================
两件事合在一个节点里，因为它们是同一个决策的两面：
从用户描述里抽出评估所需参数，**同时**判定这次请求值不值得用云端模型。

复杂度判定放在这里，是因为这是整个流程里第一次能看到 ``user_input`` 长度的
位置。此时尚未检索，``rag_triggered=False`` 表示"尚未检索"而非"确定没命中"；
``retrieve_node`` 拿到 rerank 分数后会做**单向**升级（只升不降，见
``model_router.escalate_complexity``）。
"""
from __future__ import annotations

from core.config import get_llm, llm_chat
from core.state import AgentState
from model_router import classify_complexity
from utils.tokens import record_llm_usage

from utils.logging_setup import get_logger

log = get_logger(__name__)


def extract_params(state: AgentState):
    log.info("\n[Agent 1] 正在提取 CCUS 项目关键参数...")

    # ─── 动态路由判定（入口）─────────────────────────────
    # 这是整个流程里第一次能看到 user_input 长度的位置，所以复杂度判定放这里。
    # 此时还没检索，rag_triggered=False 表示"尚未检索"而非"确定没命中"；
    # retrieve_knowledge 拿到 rerank 分数后会做单向升级。
    complexity = classify_complexity(state["user_input"])
    mode = complexity["mode"]
    log.info(f"  [Router] {complexity['label']} {complexity['reason']}")

    prompt = f"你是一个CCUS专家。请从以下描述中提取评估碳封存风险所需的关键参数：\n{state['user_input']}"
    _, model, tier, _ = get_llm("extract", mode)
    response = llm_chat(
        "extract",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        mode=mode,
    )
    return {
        "extracted_params": response.choices[0].message.content,
        "complexity_mode": mode,
        "complexity_info": complexity,
        **record_llm_usage(state, "extract", response, model, tier),
    }
