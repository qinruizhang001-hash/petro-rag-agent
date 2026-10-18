"""
汇总节点（aggregate_agent）：把两条并行分支的结论融合成最终报告
==============================================================
Fan-in 节点。上游是 ``geology`` 与 ``wellbore`` 两条并行分支，LangGraph 保证
**两条都写完才会触发这里**（实测：不等长分支的 join 节点只执行一次，
且看到的是两份完整更新）。

三件事，按顺序：

  ① **防御性合并**（``compose_branch_inputs``）—— 纯函数，不看模型。
     任何一条分支挂掉，这里都要把"缺什么"翻译成用户能看懂的话，
     而不是让 ``KeyError`` 冒到 API 层。
  ② **工具规划 + 确定性兜底** —— 密度与图表不依赖分支结论，只看提取到的参数，
     所以即使两条分支全挂，数值仍然来自物理工具而不是模型猜测。
  ③ **终稿流式输出** —— 唯一被流式展示的一遍，且只流这一遍。

为什么降级说明要**用代码追加**而不是交给模型
--------------------------------------------
"如果地质模块挂了，报告里必须写'地质模块检索超时，请人工复核'"这条要求，
交给模型就是在赌它照做。实测本地 7B 模型对"缺失模块"这类元信息极不敏感：
prompt 里写了它也可能只字不提，而这类遗漏恰恰是最危险的 ——
读者会把"没写"理解成"没风险"。所以这里用代码在报告末尾追加一段
不可省略的「数据完整性提示」。

与 report 节点的关系
--------------------
本文件由 ``nodes/report_node.py`` 迁移而来（工具规划、图表兜底、流式终稿、
历史压缩、报告落记忆等逻辑原样保留）。改名是为了让"节点名 = 文件名 = 职责"
重新对齐：扇出之后，这个节点不再是"生成报告"，而是"**汇总**两段结论并成稿"。
"""
from __future__ import annotations

import json
import re
import time

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
from utils.branch_merge import (
    compose_branch_inputs,
    degradation_block,
    parallel_timing_of,
)
from utils.pinn_query import extract_pinn_query
from utils.tokens import record_llm_usage, record_usage_payload

from utils.logging_setup import get_logger

log = get_logger(__name__)


def _call_pinn(x, t, scenario: str = "injection") -> dict:
    """调用 PINN 压力预测工具。

    惰性 import：``pinn_model`` 会拉起 torch（约 1~2 s、数百 MB 常驻）。API
    进程在别的路径上并不需要它，只有真的规划出这个工具时才值得付这笔钱。
    顺带的好处是：torch 没装时接口照样能启动，工具调用失败而不是整个服务挂掉。

    返回值做成 dict 而不是裸 float，是为了让"用了哪个场景、输入是什么"能和
    数值一起进报告 —— 一个 110 bar 如果不说边界条件，读者无法判断它对不对。
    """
    from pinn_model import predict_pressure

    x_f, t_f = float(x), float(t)
    p = predict_pressure(x_f, t_f, scenario=scenario)
    return {
        "x_m": x_f,
        "t_days": t_f,
        "pressure_bar": p,
        "scenario": scenario,
        "text": (
            f"距井口 {x_f:g} m、注入 {t_f:g} 天后地层压力 = {p:.2f} bar"
            f"（一维扩散方程 PINN 数值解，D = 0.1 m²/s，{scenario} 边界条件）"
        ),
    }


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
        log.info(f"  [Memory] 🗜️ 历史压缩：{result['before_chars']} → "
              f"{result['after_chars']} 字，{result['dropped_entries']} 条旧记录已提炼")
        log.info(f"  [Memory] 核心经验：\n{result['lessons']}")

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


def aggregate_report(state: AgentState, config=None):
    """
    汇总节点：把地质 / 井筒两段分支结论融合成最终报告。

    两阶段结构（这是有意为之，不是为了省事）：

      Phase A「工具规划」——非流式、短输出、低温。
        只让模型回答"需要调哪些工具、参数是什么"，把 JSON 指令拿回来。
      Phase B「终稿」——流式输出，且**只流这一遍**。

    为什么不沿用"一稿带工具指令 + 二次整合"：
    那种写法必须生成两遍完整报告，前端若两遍都流式，用户会看到报告重复出现，
    而且第一遍里的 ```json``` 会原样显示在屏幕上。成本也翻倍。
    拆成"短规划 + 单次终稿"后：只流一份、不出现原始 JSON、总 token 更少。

    扇出之后多了一层：**上游可能缺数据**。所以本节点第一件事不是写报告，
    而是先判断"两段结论齐不齐"，齐了走正常融合，缺了走降级路径 ——
    降级路径仍然出报告，只是多一段代码追加的完整性提示。
    """
    t_agg = time.perf_counter()

    # ─── ① 防御性合并：先看两段结论齐不齐 ──────────────────
    branch = compose_branch_inputs(state)
    timing = parallel_timing_of(state)
    log.info("\n[Aggregate] 正在汇总地质 + 井筒分支结论并生成最终报告...")
    log.info(
        f"  [Aggregate] 分支耗时：地质 {timing['geology_s']}s ∥ 井筒 {timing['wellbore_s']}s"
        f" → 并行墙钟 {timing['wall_s']}s（串行需 {timing['serial_s']}s，"
        f"加速 {timing['speedup']}×）"
    )
    if branch["degraded"]:
        log.warning(
            f"  [Aggregate] ⚠️ 降级运行：缺失分支 {branch['missing']}；"
            f"报告将显式标注并基于剩余数据生成"
        )

    # ─── 文件记忆：读取 user_profile.md + project_history.md ───
    # 首次进入本节点时读取一次并写回 state，重写轮次直接复用（避免每轮重复 IO）
    file_memory_block = state.get("file_memory_context")
    compress_update = {}
    if file_memory_block is None:
        compress_update = compress_history_with_llm(state)
        file_memory_block = build_memory_context()
        if file_memory_block:
            log.info(f"  -> 文件记忆已加载（{len(file_memory_block)} 字）")
        else:
            log.info("  -> 文件记忆为空，跳过注入")

    personal_context = state.get("personal_context", "")

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
6. **绝对不要输出 JSON、工具指令或图片 Markdown 路径**，只写正文。
7. 下面给的是**两条并行分支各自产出的结论**（地质 / 井筒）。你的任务是
   **融合**而不是复述：保留两段中的全部数值与判断，去掉重复表述，
   按上面的 1/2/3 结构重新组织。**不要新增两段里没有的结论。**"""

    # ─── 本次任务的材料（两个阶段共用）───
    # 注意：这里不再拼 state["retrieved_knowledge"]。扇出之后，注入给下游的
    # 是两条分支**已经消化过的结论**（各 150~250 字），而不是原始片段 ——
    # 这正是并行架构省 token 的地方：原始证据只进分支的 prompt，不进终稿的。
    degraded_hint = ""
    if branch["degraded"]:
        degraded_hint = (
            "\n【数据完整性警告】：本次缺失以下分支，"
            "请在报告开头用一句话说明结论不完整、需人工复核：\n"
            + "\n".join(f"- {m}" for m in branch["missing"])
            + "\n"
        )

    materials = f"""【提取的 CCUS 参数】：{state['extracted_params']}
【地质分支结论（LightRAG 图谱证据）】：
{branch['geology']}

【井筒分支结论（ChromaDB 文献证据）】：
{branch['wellbore']}
{degraded_hint}
{personal_context}
【上一版报告的修改意见】：{state.get('feedback', '无')}"""

    # ══════════════════════════════════════════════════════
    # Phase A：工具规划（短输出、非流式）
    # ══════════════════════════════════════════════════════
    plan_prompt = materials + """

【任务：只做工具规划，不要写报告】
你有三个工具：
1. `calculate_co2_density` —— 参数中出现注入压力和温度时调用，不要自己口算密度。
2. `generate_pressure_curve` —— 必须调用，生成"地层压力随深度"地质图。
   depth 取参数中的储层埋深；参数里没有就省略该字段（系统会按默认值绘制并标注"估算"），
   **不要编造深度**。
3. `predict_pressure` —— 当问题问的是"距井口 X 米处、注入 T 天后的压力"时调用，
   由物理信息神经网络解一维扩散方程给出数值，**不要自己推算压力随距离/时间的衰减**。
   x 单位米（0~100），t 单位天（0~30），两者都必须取自问题原文，不要臆造。

只输出 json 代码块，每个工具一个，不要输出任何解释或正文。数值必须取自【提取的 CCUS 参数】：
```json
{"tool": "calculate_co2_density", "pressure_bar": 250, "temperature_celsius": 80}
```
```json
{"tool": "generate_pressure_curve", "depth": 2500, "injection_pressure_bar": 250}
```
```json
{"tool": "predict_pressure", "x": 50, "t": 10}
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
            log.warning(f"  [Warning] 工具指令解析失败: {e}")
            continue
        if isinstance(data, dict) and data.get("tool"):
            tool_requests.append(data)
    log.info(f"  -> 工具规划返回 {len(tool_requests)} 条指令")

    density_result = None
    chart_result = None
    pinn_result = None
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
                log.warning(f"  [Warning] 图表生成失败: {e}")
        elif name == "predict_pressure":
            # 坐标必须能在**用户原话**里核对到，不能信模型给的数字。
            # 原因很具体：plan_prompt 里有一个 {"x": 50, "t": 10} 的示例，
            # 7B 模型很容易把它照抄进一个根本没问距离和时间的问题里，
            # 于是报告多出一个"距井口 50 m = 109.95 bar"——用户没问，
            # 数值还是精确的，这种污染比明显错误更难发现。
            grounded = extract_pinn_query(state.get("user_input", ""))
            if grounded is None:
                log.warning(
                    f"  [Warning] 模型规划了 PINN，但问句里没有可核对的 x/t"
                    f"（{req}）→ 忽略，避免编造坐标"
                )
                continue
            try:
                pinn_result = _call_pinn(*grounded)
                log.info(f"  [Tool 调用] PINN → {pinn_result['text']}")
            except Exception as e:
                log.warning(f"  [Warning] PINN 预测失败（不影响报告）: {e}")

    # ─── 确定性兜底：模型该调没调时，由代码补上 ─────────────
    # 本地 7B 模型在上下文变丰富后会"觉得自己能算"，从而跳过工具调用，
    # 于是密度变成模型口算值 —— 这会直接破坏"数值不可幻觉"的核心承诺。
    # 因此不让它靠自觉：参数里有 P/T 就必须有工具结果。
    p_val, t_val = _extract_pressure_temperature(state.get("extracted_params", ""))
    if density_result is None and p_val is not None and t_val is not None:
        log.info("  [System] 模型未规划密度计算，但参数含压力/温度 → 确定性兜底")
        density_result = calculate_co2_density(p_val, t_val)

    # 图表同理：报告没有配图会显得业余，且输入几乎总能从参数推出。
    if chart_result is None or chart_result.get("status") != "ok":
        depth_val = _extract_depth(state.get("extracted_params", ""))
        log.info("  [System] 模型未规划图表 → 确定性兜底绘图")
        try:
            chart_result = generate_pressure_curve(depth_val, p_val)
        except Exception as e:
            log.warning(f"  [Warning] 图表生成失败（不影响报告）: {e}")
            chart_result = None

    # ─── PINN 兜底：触发条件看**用户原话**，而不是提取出的参数 ───
    # extract_params 会把"距离井口 50 米"改写成"储层埋深 2500 m"这类工程参数，
    # 而 (x, t) 这组空间-时间坐标只存在于原话里。所以这里扫 user_input。
    # 与密度/图表兜底同一个理由：压力随距离和时间的衰减不该由模型口算。
    if pinn_result is None:
        q = extract_pinn_query(state.get("user_input", ""))
        if q is not None:
            log.info(f"  [System] 问题含井口距离/注入时长（x={q[0]:g} m, t={q[1]:g} d）"
                     f"但模型未规划 PINN → 确定性兜底调用")
            try:
                pinn_result = _call_pinn(q[0], q[1])
                log.info(f"  [Tool 调用] PINN → {pinn_result['text']}")
            except Exception as e:
                log.warning(f"  [Warning] PINN 兜底调用失败（不影响报告）: {e}")

    chart_ok = bool(chart_result and chart_result.get("status") == "ok")
    if chart_ok:
        log.info(f"  [System] 图表已生成：{chart_result['rel_path']}")
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
    if pinn_result is not None:
        tool_facts.append(f"PINN 压力预测：{pinn_result['text']}")

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
        log.info(f"  [Router] {complexity['label']} {complexity.get('reason', '')}")

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
            log.info(f"  [System] 报告未包含工具算出的密度 {numeric} → 追加数值说明")
            report_text = report_text.rstrip() + (
                f"\n\n> CO2 密度 {density_result}，按注入压力与温度由状态方程工具计算。"
            )

    # ─── PINN 数值兜底：与密度同理，工具算了但没写进报告就等于没算 ───
    # 实测本地 7B 会忽略 facts_block 里"必须写进正文"的指令（报告里只字未提
    # 109.95 bar），所以这里和密度一样由代码补。判断依据是数值本身是否出现，
    # 而不是"模型有没有提到压力" —— 后者无法可靠判断。
    if pinn_result is not None:
        pnum = f"{pinn_result['pressure_bar']:.2f}"
        if pnum not in report_text:
            log.info(f"  [System] 报告未包含 PINN 预测 {pnum} bar → 追加数值说明")
            report_text = report_text.rstrip() + f"\n\n> {pinn_result['text']}。"

    # ─── 降级说明：代码追加，不依赖模型照做 ────────────────
    # 这段必须放在图片引用**之前**吗？不必 —— 但要放在正文之后、且保证唯一。
    # 重写轮次（feedback 非空）会重新生成本节点，所以这里每次都重新追加，
    # 不会因为上一轮的追加而重复（report_text 是本轮模型输出，不含旧尾巴）。
    note = degradation_block(branch)
    if note:
        report_text = report_text.rstrip() + f"\n\n{note}\n"

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
            log.info("  -> 正在保存报告到个人长期记忆...")
            save_memory(
                title=f"CCUS评估报告: {state['user_input'][:50]}",
                content=report_text,
                tags=["CCUS", "评估报告", "auto-save"]
            )
            log.info("  -> 报告已保存到个人长期记忆 (Notion + ChromaDB)")
        except Exception as e:
            log.warning(f"  -> 记忆保存失败（不影响主流程）: {e}")

    # ─── 证据合并：评审节点仍读 retrieved_knowledge ─────────
    # 评审要回答"报告里的结论有没有证据支撑"，所以它需要的是**证据**，
    # 而不是两段已经消化过的结论（拿结论去评结论，等于让模型自己给自己打分）。
    merged_evidence = (
        f"【地质图谱证据】\n{state.get('geology_evidence') or '（无）'}\n\n"
        f"【井筒文献证据】\n{state.get('wellbore_evidence') or '（无）'}"
    )

    agg_elapsed = time.perf_counter() - t_agg
    timing = {**timing, "aggregate_s": round(agg_elapsed, 2),
              "total_s": round(timing["wall_s"] + agg_elapsed, 2)}
    log.info(
        f"  [Aggregate] ✅ 汇总完成（{agg_elapsed:.1f}s）；"
        f"扇出+汇总总耗时 {timing['total_s']}s"
    )

    current_retry = state.get("retry_count", 0)
    return {
        "final_report": report_text,
        "retrieved_knowledge": merged_evidence,
        "parallel_timing": timing,
        "branch_merge": {
            "degraded": branch["degraded"],
            "missing": branch["missing"],
            "geology_ok": branch["geology_ok"],
            "wellbore_ok": branch["wellbore_ok"],
            "reasons": branch["reasons"],
        },
        "retry_count": current_retry + 1,
        "file_memory_context": file_memory_block,
        "chart_info": _chart_payload(chart_result) if chart_ok else {},
        "pinn_prediction": pinn_result or {},
        **compress_update,
        **state_update,
    }
