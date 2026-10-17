"""
Agent 3：报告生成
=================
本模块里最贵的一步，也是整个系统里唯一被流式展示的一步。

两阶段结构（有意为之，不是图省事）：

  Phase A「工具规划」—— 非流式、短输出、低温。只让模型回答"要调哪些工具、
    参数是什么"，把 JSON 指令拿回来。
  Phase B「终稿」—— 流式输出，且**只流这一遍**。

为什么不沿用"一稿带工具指令 + 二次整合"：那种写法必须生成两遍完整报告，
前端若两遍都流式，用户会看到报告重复出现，而且第一遍里的 ```json``` 会原样
显示在屏幕上，成本也翻倍。拆成"短规划 + 单次终稿"后：只流一份、不出现原始
JSON、总 token 更少。

两个确定性兜底，都针对同一个失效模式 —— "模型该调工具却没调"：
密度值用正则兜住 P/T，图表用代码兜底绘制。数值来自工具是核心承诺，
模型"觉得够了自己算"会直接破坏它，所以不让它靠自觉。
"""
from __future__ import annotations

import json
import re

from chart_tools import generate_pressure_curve
from core.config import get_llm, llm_chat, llm_chat_streamed
from core.state import AgentState
from file_memory import (
    build_memory_context,
    compress_history_if_needed,
    history_needs_compression,
)
from memory.memory_manager import save_memory
from tools.physics_tools import calculate_co2_density
from utils.tokens import record_llm_usage, record_usage_payload


def _extract_pressure_temperature(params: str) -> tuple[float | None, float | None]:
    """
    从参数提取结果中抓取压力 / 温度数值。

    这是 Function Calling 的**确定性兜底**：模型有时会认为"信息已够"
    而跳过工具调用，直接凭参数写出报告。此时数值就变成了模型的推测。
    用正则兜住 P/T，保证密度值一定来自物理工具而非语言模型。
    """
    p_match = re.search(r"(?:压力|pressure)[^\d\-]{0,15}(\d+(?:\.\d+)?)", params or "", re.I)
    t_match = re.search(r"(?:温度|temperature)[^\d\-]{0,15}(\d+(?:\.\d+)?)", params or "", re.I)
    p_val = float(p_match.group(1)) if p_match else None
    t_val = float(t_match.group(1)) if t_match else None
    return p_val, t_val


def _extract_depth(params: str) -> float | None:
    """
    从参数里抓储层深度（m）。抓不到返回 None，由 chart_tools 用默认值并标注"估算"。

    兼容 "深度 3000 m" / "埋深：2500米" / "depth 2000 m" 等写法。
    """
    m = re.search(
        r"(?:埋深|深度|depth)[^\d\-]{0,12}(\d+(?:\.\d+)?)\s*(?:m|米|meters?)?",
        params or "", re.I,
    )
    if not m:
        return None
    try:
        val = float(m.group(1))
    except ValueError:
        return None
    # 明显不合理（如把 3000 写成 3）就丢掉，交给默认值
    return val if 100 <= val <= 8000 else None


def _chart_payload(result: dict | None) -> dict:
    """把图表工具结果裁成可 JSON 序列化的精简结构（推前端 + 写状态）。"""
    if not result or result.get("status") != "ok":
        return {}
    keep = (
        "path", "rel_path", "url_path", "summary", "verdict", "verdict_text",
        "depth_m", "injection_pressure_bar", "fracture_bar_at_depth",
        "hydrostatic_bar_at_depth", "admissible_depth_min_m",
        "admissible_depth_max_m", "depth_assumed",
    )
    return {k: result.get(k) for k in keep if k in result}


_TOOL_FENCE = re.compile(r"```(?:json)?\s*\{.*?\}\s*```", re.DOTALL)


def strip_tool_json(text: str) -> str:
    """
    去掉正文里残留的工具 JSON 代码块。

    即便 prompt 明确禁止，模型仍可能把 ```json``` 混进终稿。这类内容一旦
    流到前端，用户看到的就是一串带花括号的机器指令 —— 属于"看起来像 bug"的
    输出，必须在入库/展示前清掉。
    """
    if not text:
        return ""
    cleaned = _TOOL_FENCE.sub("", text)
    # 清理因删除而产生的多余空行（保留段落间的单个空行）
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


SUMMARY_PROMPT = """你是一名 CCUS 项目档案管理员。下面是该项目的历次评估记录（旧记录）。

请把它们提炼成 **3 条最高级别的「项目核心经验」**，供后续评估直接复用。

要求：
1. 只要 3 条，每条一行，以 "- " 开头，不要编号、不要分小标题、不要开场白。
2. 每条必须**具体可复用**：包含触发条件 + 后果 + 应对措施，
   例如"注入压力 > 250 bar 且存在隐伏断层时，断层活化风险显著上升，需布设微地震监测"。
3. 合并重复场景，只保留规律性认识，不要罗列单次结论。
4. 若旧记录信息不足以支撑 3 条，有几条写几条，不要编造。

【旧评估记录】：
{old_text}

直接输出 3 条 "- " 开头的经验，不要输出其他任何内容。"""


def compress_history_with_llm(state: AgentState) -> dict:
    """
    超阈值时调用本地模型压缩历史记忆，返回可并入节点返回值的状态更新。

    为什么单独包一层而不直接把 llm_chat 塞进 file_memory：
    压缩是一次真实的 LLM 调用，token 消耗必须被记账（否则成本统计失真）；
    而 file_memory 模块不允许依赖任何 LLM 客户端（循环导入 + 离线单测）。
    所以这里做"回调注入"：file_memory 只管文件与阈值判断，模型调用留在本模块。
    """
    if not history_needs_compression():
        return {}

    captured: dict = {}

    def _summarizer(old_text: str) -> str:
        _, model, tier, _ = get_llm("summary")
        response = llm_chat(
            "summary",
            messages=[{
                "role": "user",
                "content": SUMMARY_PROMPT.format(old_text=old_text[:12000]),
            }],
            temperature=0.2,
        )
        captured["response"] = response
        captured["model"] = model
        captured["tier"] = tier
        return response.choices[0].message.content

    result = compress_history_if_needed(summarizer=_summarizer)

    if result["status"] == "compressed":
        print(f"  [Memory] 🗜️ 历史压缩：{result['before_chars']} → "
              f"{result['after_chars']} 字，{result['dropped_entries']} 条旧记录已提炼")
        print(f"  [Memory] 核心经验：\n{result['lessons']}")

    if "response" not in captured:
        return {}

    update = record_llm_usage(
        state, "history_compress", captured["response"],
        captured["model"], captured["tier"],
    )
    update["memory_compression"] = {
        "status": result["status"],
        "before_chars": result["before_chars"],
        "after_chars": result["after_chars"],
        "dropped_entries": result["dropped_entries"],
        "lessons": result["lessons"],
    }
    return update


def generate_report(state: AgentState, config=None):
    """
    Agent 3：生成评估报告。

    两阶段结构（这是有意为之，不是为了省事）：

      Phase A「工具规划」——非流式、短输出、低温。
        只让模型回答"需要调哪些工具、参数是什么"，把 JSON 指令拿回来。
      Phase B「终稿」——流式输出，且**只流这一遍**。

    为什么不沿用"一稿带工具指令 + 二次整合"：
    那种写法必须生成两遍完整报告，前端若两遍都流式，用户会看到报告重复出现，
    而且第一遍里的 ```json``` 会原样显示在屏幕上。成本也翻倍。
    拆成"短规划 + 单次终稿"后：只流一份、不出现原始 JSON、总 token 更少。
    """

    print("\n[Agent 3] 正在生成 CCUS 项目综合评估报告...")

    # ─── 文件记忆：读取 user_profile.md + project_history.md ───
    # 首次进入本节点时读取一次并写回 state，重写轮次直接复用（避免每轮重复 IO）
    file_memory_block = state.get("file_memory_context")
    compress_update = {}
    if file_memory_block is None:
        compress_update = compress_history_with_llm(state)
        file_memory_block = build_memory_context()
        if file_memory_block:
            print(f"  -> 文件记忆已加载（{len(file_memory_block)} 字）")
        else:
            print("  -> 文件记忆为空，跳过注入")

    personal_context = state.get("personal_context", "")
    web_context = ""
    if state.get("web_content"):
        web_context = f"\n【联网搜索补充内容（已自动入库）】：\n{state['web_content']}"

    # 所有流式调用共用的 token 推送通道（SSE 用；invoke 时为 None）
    sink = (config or {}).get("configurable", {}).get("token_sink")

    # ─── System Prompt：角色 + 长期文件记忆 + 写作硬约束 ───
    system_prompt = f"""你是一个资深的CCUS首席工程师，负责撰写 CO2 地质封存项目的风险评估报告。

{file_memory_block if file_memory_block else "（暂无文件记忆）"}

【写作硬性约束】：
1. 如果知识库中没有相关记录，必须直接输出"知识库中无相关记录"。
2. 必须遵循上面「用户画像」中的报告风格偏好（结论先行、分点陈述、数值落地）。
3. **禁止使用"综上所述""总而言之""总的来说"等总结性套话**，直接分点陈述。
4. 如果本次场景与「历史评估记录」中的某次相近，应显式做纵向对比
   （例如"相比上次 250bar/80℃ 的评估，本次压力更高，断层活化风险上升"）。
5. 请分点列出：1. 主要地质封存风险 2. 主要井筒风险 3. 缓解与监测建议。
6. **绝对不要输出 JSON、工具指令或图片 Markdown 路径**，只写正文。"""

    # ─── 本次任务的材料（两个阶段共用）───
    materials = f"""【提取的 CCUS 参数】：{state['extracted_params']}
【知识库检索到的标准】：{state['retrieved_knowledge']}
{web_context}
{personal_context}
【上一版报告的修改意见】：{state.get('feedback', '无')}"""

    # ══════════════════════════════════════════════════════
    # Phase A：工具规划（短输出、非流式）
    # ══════════════════════════════════════════════════════
    plan_prompt = materials + """

【任务：只做工具规划，不要写报告】
你有两个工具：
1. `calculate_co2_density` —— 参数中出现注入压力和温度时调用，不要自己口算密度。
2. `generate_pressure_curve` —— 必须调用，生成"地层压力随深度"地质图。
   depth 取参数中的储层埋深；参数里没有就省略该字段（系统会按默认值绘制并标注"估算"），
   **不要编造深度**。

只输出 json 代码块，每个工具一个，不要输出任何解释或正文。数值必须取自【提取的 CCUS 参数】：
```json
{"tool": "calculate_co2_density", "pressure_bar": 250, "temperature_celsius": 80}
```
```json
{"tool": "generate_pressure_curve", "depth": 2500, "injection_pressure_bar": 250}
```"""

    plan_msgs = [
        {"role": "system", "content": "你是工具调用规划器，只输出 JSON 代码块。"},
        {"role": "user", "content": plan_prompt},
    ]
    _, plan_model, plan_tier, _ = get_llm("report_plan")
    plan_resp = llm_chat("report_plan", messages=plan_msgs, temperature=0.0)
    state_update = record_llm_usage(
        {**state, **compress_update}, "report_plan", plan_resp, plan_model, plan_tier
    )
    usage_state = {**state, **compress_update, **state_update}

    # 扫描**所有** json 代码块（一份报告可能同时要密度和图）
    tool_requests = []
    for block in re.findall(r'```json\s*(\{.*?\})\s*```', plan_resp.choices[0].message.content or "", re.DOTALL):
        try:
            data = json.loads(block)
        except Exception as e:
            print(f"  [Warning] 工具指令解析失败: {e}")
            continue
        if isinstance(data, dict) and data.get("tool"):
            tool_requests.append(data)
    print(f"  -> 工具规划返回 {len(tool_requests)} 条指令")

    density_result = None
    chart_result = None
    for req in tool_requests:
        name = req.get("tool")
        if name == "calculate_co2_density":
            density_result = calculate_co2_density(
                req.get("pressure_bar"), req.get("temperature_celsius")
            )
        elif name == "generate_pressure_curve":
            try:
                chart_result = generate_pressure_curve(
                    req.get("depth"), req.get("injection_pressure_bar")
                )
            except Exception as e:
                print(f"  [Warning] 图表生成失败: {e}")

    # ─── 确定性兜底：模型该调没调时，由代码补上 ─────────────
    # 本地 7B 模型在上下文变丰富后会"觉得自己能算"，从而跳过工具调用，
    # 于是密度变成模型口算值 —— 这会直接破坏"数值不可幻觉"的核心承诺。
    # 因此不让它靠自觉：参数里有 P/T 就必须有工具结果。
    p_val, t_val = _extract_pressure_temperature(state.get("extracted_params", ""))
    if density_result is None and p_val is not None and t_val is not None:
        print("  [System] 模型未规划密度计算，但参数含压力/温度 → 确定性兜底")
        density_result = calculate_co2_density(p_val, t_val)

    # 图表同理：报告没有配图会显得业余，且输入几乎总能从参数推出。
    if chart_result is None or chart_result.get("status") != "ok":
        depth_val = _extract_depth(state.get("extracted_params", ""))
        print("  [System] 模型未规划图表 → 确定性兜底绘图")
        try:
            chart_result = generate_pressure_curve(depth_val, p_val)
        except Exception as e:
            print(f"  [Warning] 图表生成失败（不影响报告）: {e}")
            chart_result = None

    chart_ok = bool(chart_result and chart_result.get("status") == "ok")
    if chart_ok:
        print(f"  [System] 图表已生成：{chart_result['rel_path']}")
        # 立刻推给前端：用户不用等报告写完就能看到图
        if sink is not None:
            try:
                sink({"type": "chart", "chart": _chart_payload(chart_result)})
            except Exception:
                pass

    # ══════════════════════════════════════════════════════
    # Phase B：终稿（流式，唯一被流式展示的一遍）
    # ══════════════════════════════════════════════════════
    tool_facts = []
    if density_result is not None:
        tool_facts.append(f"CO2 密度 = {density_result}（由物理工具计算）")
    if chart_ok:
        tool_facts.append(f"地层压力曲线结论：{chart_result['summary']}")

    facts_block = ""
    if tool_facts:
        facts_block = (
            "【工具执行结果 —— 必须写进报告正文】\n- " + "\n- ".join(tool_facts)
            + "\n\n"
        )

    # 工具结果放在**最前面**：实测本地模型对结尾处的约束遵守度很差，
    # 把数值要求前置后，模型更可能把密度直接写进正文而不是靠代码兜底补。
    final_user = (
        facts_block
        + materials
        + "\n\n请撰写完整报告。上面工具结果里的数值必须原样写进正文"
          "（例如在地质风险部分点明该压力/温度下的 CO2 密度），"
          "不要改写、不要省略，也不要单独列成附注；"
          "不要输出 JSON 指令，不要在正文里写图片路径。"
    )
    final_msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": final_user},
    ]

    # ─── 动态路由：沿用本次请求的复杂度档位 ────────────────
    # 档位在 extract_params 里判定、在 retrieve_knowledge 里按 RAG 命中情况
    # 单向升级，这里只是消费它 —— 报告生成是整条链路里最贵的一步，
    # 简单请求不该为它付云端的钱。
    report_mode = state.get("complexity_mode") or "cloud"
    complexity = state.get("complexity_info") or {}
    if complexity.get("label"):
        print(f"  [Router] {complexity['label']} {complexity.get('reason', '')}")

    _, model, tier, _ = get_llm("report", report_mode)
    streamed = llm_chat_streamed("report", final_msgs, 0.3, sink=sink, mode=report_mode)
    report_text = strip_tool_json(streamed.text)
    state_update = record_usage_payload(
        usage_state, "report",
        streamed.prompt_tokens, streamed.completion_tokens, model, tier,
        estimated=streamed.usage_estimated,
    )

    # ─── 数值兜底：工具算出来的数不能"没写进报告"───────────
    # 模型偶尔会漏掉注入的数值。数值来自工具是核心承诺，漏了就等于没算，
    # 所以这里补一行显式数值，而不是让它自生自灭。
    if density_result is not None:
        numeric = density_result.split()[0] if density_result else ""
        if numeric and numeric not in report_text:
            print(f"  [System] 报告未包含工具算出的密度 {numeric} → 追加数值说明")
            report_text = report_text.rstrip() + (
                f"\n\n> CO2 密度 {density_result}，按注入压力与温度由状态方程工具计算。"
            )

    # ─── 报告末尾补齐图片引用（代码保证，不靠模型自觉）──────
    # 报告本体存 Markdown 相对路径，便于归档与其它消费者；
    # 前端用 chart 事件里的真实路径调 st.image，不会重复渲染。
    if chart_ok:
        ref = f"![地层压力随深度变化曲线]({chart_result['rel_path']})"
        if chart_result["rel_path"] not in report_text:
            report_text = report_text.rstrip() + f"\n\n{ref}\n"

    # ─── 自动保存报告到个人长期记忆 ───────────────────────
    if not state.get("feedback"):
        try:
            print("  -> 正在保存报告到个人长期记忆...")
            save_memory(
                title=f"CCUS评估报告: {state['user_input'][:50]}",
                content=report_text,
                tags=["CCUS", "评估报告", "auto-save"]
            )
            print("  -> 报告已保存到个人长期记忆 (Notion + ChromaDB)")
        except Exception as e:
            print(f"  -> 记忆保存失败（不影响主流程）: {e}")

    current_retry = state.get("retry_count", 0)
    return {
        "final_report": report_text,
        "retry_count": current_retry + 1,
        "file_memory_context": file_memory_block,
        "chart_info": _chart_payload(chart_result) if chart_ok else {},
        **compress_update,
        **state_update,
    }
