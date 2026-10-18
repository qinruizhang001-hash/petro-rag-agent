"""
文件记忆后处理节点：把本次评估写进 project_history.md
====================================================
为什么是独立节点而不是塞进 ``evaluate_report``：
- 评审员会被反思循环多次调用（rewrite 分支回到 report），在里面写文件会把
  中间稿和最终稿都写进去，历史立刻被污染。
- 挂在 ``check_quality`` 的 pass 分支后面，天然保证"每次任务只写一次"。

还有一层幂等保护：state 里的 ``history_saved`` 标记挡住重跑重复追加 ——
LangGraph 的 checkpoint 重放可能让同一节点再次执行。

抽取用的是本地小模型：这一步只是信息压缩，不需要云端智商。
失败时降级为规则拼接，保证历史不断档 —— 宁可记录粗糙，也不要断档。
"""
from __future__ import annotations

from core.config import get_llm, llm_chat
from core.state import AgentState
from file_memory import append_project_history
from utils.tokens import record_llm_usage

from utils.logging_setup import get_logger

log = get_logger(__name__)


def save_file_memory(state: AgentState):
    """
    把本次评估压缩成一条结构化历史记录，追加到 project_history.md。

    为什么单独做成一个节点，而不是塞进 evaluate_report：
    - evaluate_report 会被反思循环多次调用（rewrite 分支回到 report），
      在里面写文件会把中间稿和最终稿都写进去，历史立刻被污染。
    - 单独节点挂在 check_quality 的 pass 分支后面，天然保证"每次任务只写一次"。
    """
    log.info("\n[File Memory] 正在把本次评估写入 project_history.md...")

    if state.get("history_saved"):
        log.info("  -> 本次任务已记录过，跳过（幂等）")
        return {"history_saved": True}

    report = state.get("final_report", "")
    if not report:
        log.info("  -> 报告为空，跳过记录")
        return {"history_saved": True}

    # 用本地小模型做压缩 —— 这一步只是信息抽取，不需要云端智商
    prompt = f"""请从下面这份 CCUS 风险评估报告中，提取一条**极简**的历史记录，供后续评估做纵向对比。

【用户原始需求】：{state['user_input']}
【提取的参数】：{state.get('extracted_params', '')}
【评审分数】：{state.get('score', '-')}
【报告全文】：
{report}

请严格按以下 Markdown 格式输出，不要输出任何额外内容、不要加一级/二级标题：

**场景**: 一句话描述项目场景（储层类型、断层情况等）
**关键参数**: 列出压力/温度/注入速率等具体数值，没有就写"未提供"
**核心结论**: 2-3 句，只写最重要的风险判断和对应缓解措施
**评审分**: {state.get('score', '-')}/10"""

    try:
        _, model, tier, _ = get_llm("history")
        response = llm_chat(
            "history",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        entry = response.choices[0].message.content.strip()
        usage_update = record_llm_usage(state, "history", response, model, tier)
    except Exception as e:
        # 云端/本地都失败时降级为规则拼接，保证历史不断档
        log.warning(f"  -> 结论抽取失败（{e}），改用规则拼接")
        entry = (
            f"**场景**: {state['user_input'][:80]}\n"
            f"**关键参数**: 见提取结果\n"
            f"**核心结论**: {report[:200].replace(chr(10), ' ')}…\n"
            f"**评审分**: {state.get('score', '-')}/10"
        )
        usage_update = {}

    size = append_project_history(entry)
    log.info(f"  -> 已追加到 project_history.md（当前 {size} 字符）")

    return {
        "history_entry": entry,
        "history_saved": True,
        **usage_update,
    }
