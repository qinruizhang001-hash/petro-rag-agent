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
from evolution_engine import query_system_experience
from model_router import classify_complexity
from utils.tokens import record_llm_usage

from utils.logging_setup import get_logger

log = get_logger(__name__)


def _experience_block(user_input: str) -> tuple[str, int]:
    """查一次 system_experience，返回 (可拼进 prompt 的先验块, 条数)。

    为什么放在 extract 而不是 retrieve：先验要在**抽取参数时**就影响判断
    （例如"缺埋深会导致物理校验失效，应主动追问埋深"），等到检索阶段再注入
    已经晚了。代价是这个节点要快 —— ``query_system_experience`` 只读一个
    小 JSON 文件、用字符二元组打分，不加载任何模型。

    返回条数而不是让调用方数换行：先验文本里本来就可能有换行，靠数 ``\\n``
    推条数迟早会错，而这种错只会表现为日志里一个不对的数字，没人会发现。
    """
    prior = query_system_experience(user_input)
    if not prior:
        return "", 0
    count = sum(1 for line in prior.splitlines() if line.strip())
    block = (
        "【系统经验（历史评估提炼的先验，优先级高于你的常识）】\n"
        f"{prior}\n"
        "以上规律来自本系统过去若干次评估。与本次描述相关时优先采用；"
        "与本次数据明显冲突时以本次数据为准，并在输出中说明冲突。\n\n"
    )
    return block, count


def extract_params(state: AgentState):
    log.info("\n[Agent 1] 正在提取 CCUS 项目关键参数...")

    # ─── 动态路由判定（入口）─────────────────────────────
    # 这是整个流程里第一次能看到 user_input 长度的位置，所以复杂度判定放这里。
    # 此时还没检索，rag_triggered=False 表示"尚未检索"而非"确定没命中"；
    # retrieve_knowledge 拿到 rerank 分数后会做单向升级。
    complexity = classify_complexity(state["user_input"])
    mode = complexity["mode"]
    log.info(f"  [Router] {complexity['label']} {complexity['reason']}")

    prior_block, prior_count = _experience_block(state["user_input"])
    if prior_count:
        log.info(f"  [Evolution] 注入 {prior_count} 条历史先验到抽取 prompt")

    prompt = (
        f"{prior_block}"
        f"你是一个CCUS专家。请从以下描述中提取评估碳封存风险所需的关键参数：\n{state['user_input']}"
    )
    _, model, tier, _ = get_llm("extract", mode)
    response = llm_chat(
        "extract",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        mode=mode,
    )
    return {
        "extracted_params": response.choices[0].message.content,
        "system_experience": prior_block,
        "complexity_mode": mode,
        "complexity_info": complexity,
        **record_llm_usage(state, "extract", response, model, tier),
    }
