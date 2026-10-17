"""
Agent 4：评审员
===============
给报告打分并给出 1~2 句评语，分数驱动反思循环（< 8 分打回重写）。

它**不**写文件、不改报告 —— 只输出 ``score`` 与 ``feedback``。
把"落盘"拆到独立的 ``save_file_memory`` 节点是必要的：评审会被反思循环
多次调用，在这里写文件会把中间稿和最终稿都写进历史，立刻污染记忆。

解析失败时兜底 ``score = 10``（放行）。这个选择看起来危险，但反过来
（默认 0 分）会让一次解析抖动触发两次无意义的重写，成本更高且结果更差。
"""
from __future__ import annotations

from core.config import get_llm, llm_chat
from core.state import AgentState
from utils.tokens import record_llm_usage


def evaluate_report(state: AgentState):
    print("\n[Agent 4] 正在审核报告质量...")
    prompt = f"""你是一个严格的CCUS学术评审员。请根据【检索到的文献】，评估【AI生成的回答】的质量。
【用户问题】：{state['user_input']}
【检索到的文献】：{state['retrieved_knowledge']}
【AI生成的回答】：{state['final_report']}

请严格按以下格式输出，不要输出任何其他内容：
分数：[0-10的整数]
评语：[1-2句简短评语]"""

    _, model, tier, _ = get_llm("evaluate")
    response = llm_chat(
        "evaluate",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    result = response.choices[0].message.content

    try:
        score = int(result.split("分数：")[1].split("\n")[0].strip())
        feedback = result.split("评语：")[1].strip()
    except:
        score = 10
        feedback = result

    print(f"  -> 当前得分：{score} 分 | 评语：{feedback}")
    return {
        "score": score,
        "feedback": feedback,
        **record_llm_usage(state, "evaluate", response, model, tier),
    }
