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
"如果地质模块没有可用证据，报告里必须提醒人工复核"这条要求，
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
from core.uq import uq_block
from file_memory import (
    build_memory_context,
    compress_history_if_needed,
    history_needs_compression,
)
from memory.memory_manager import save_memory
from nodes.aggregate_finalize import finalize_report
from nodes.aggregate_prompt import build_materials, build_system_prompt
from nodes.aggregate_support import (
    TOOL_POLICY,
    chart_payload,
    example_args,
    extract_depth,
    extract_pressure_temperature,
    get_client_specs,
    strip_tool_json,
    uq_payload,
)
from tools.physics_tools import co2_density_uq, injection_risk_uq
from utils.branch_merge import compose_branch_inputs, parallel_timing_of
from utils.pinn_query import extract_pinn_query
from utils.tokens import record_llm_usage, record_usage_payload

from utils.logging_setup import get_logger

log = get_logger(__name__)


def _execute_tool(name: str, arguments: dict) -> dict:
    """通过 MCP Client 调用一个工具，返回其结构化结果。

    工具不再由本模块 import 具体实现 —— 而是走 MCP 协议向 Server 请求。
    失败返回 ``{}`` 而不是抛异常：工具不可用不该让整份报告生不出来，
    降级路径（缺数值但仍然成稿）本来就是这个节点的既有设计。
    """
    try:
        from mcp_client_ccus import get_client

        return get_client().call(name, arguments)
    except Exception as exc:  # noqa: BLE001
        log.warning(f"  [MCP] 调用 {name} 失败（不影响报告）: {type(exc).__name__}: {exc}")
        return {}


def _call_pinn(x, t, scenario: str = "injection") -> dict:
    """调用 PINN 压力预测工具（经 MCP）。

    返回值做成 dict 而不是裸 float，是为了让"用了哪个场景、输入是什么"能和
    数值一起进报告 —— 一个 110 bar 如果不说边界条件，读者无法判断它对不对。
    """
    payload = _execute_tool(
        "predict_pressure_pinn",
        {"x": float(x), "t": float(t), "scenario": scenario},
    )
    if "pressure_bar" not in payload:
        raise RuntimeError(f"PINN 工具未返回压力值：{payload or '空响应'}")

    x_f, t_f, p = float(x), float(t), float(payload["pressure_bar"])
    out = {
        "x_m": x_f,
        "t_days": t_f,
        "pressure_bar": p,
        "scenario": payload.get("scenario", scenario),
        # 教学演示的定位必须跟着数值走：报告正文里出现 109.95 bar 时，
        # 同一句话里就要有"这不是油藏依据"。
        "usage_scope": payload.get("usage_scope", "teaching-demo"),
        "not_for_safety_verdict": bool(payload.get("not_for_safety_verdict", True)),
        "text": (
            f"距井口 {x_f:g} m、注入 {t_f:g} 天后地层压力 = {p:.2f} bar"
            f"（一维扩散方程 PINN 数值解，D = 0.1 m²/s，{scenario} 边界条件；"
            f"该模型为教学演示，边界条件未标定到实际储层，不参与安全/不安全判定）"
        ),
    }
    if payload.get("analytic_steady_state_bar") is not None:
        out["analytic_bar"] = payload["analytic_steady_state_bar"]
    return out


def _call_density(pressure_bar, temperature_celsius) -> dict | None:
    """
    调用 CO2 密度工具（经 MCP），返回含置信区间的结构化结果。

    返回 dict 而不是裸字符串：报告需要 ``text``（一句话，带区间），
    前端看板需要 ``mean/p5/p95/samples``（画直方图）。只留 text 的话，
    前端就得回头解析中文括号，那是最容易漂移的一种"接口"。

    MCP 不可用时退回本地计算 —— 密度是纯函数、毫秒级，没有理由因为
    传输层问题就让报告失去这个数值。
    """
    payload = _execute_tool(
        "get_co2_density",
        {
            "pressure_bar": float(pressure_bar),
            "temperature_celsius": float(temperature_celsius),
            "uncertainty": 0.05,
            "n_simulations": 100,
        },
    )
    if (
        payload.get("text")
        and payload.get("mean_kg_m3") is not None
        and payload.get("ci90_high_kg_m3") is not None
    ):
        mean = float(payload["mean_kg_m3"])
        std = float(payload.get("std_kg_m3") or 0.0)
        return {
            "text": str(payload["text"]),
            "mean": mean,
            "p5": payload.get("ci90_low_kg_m3"),
            "p95": payload.get("ci90_high_kg_m3"),
            "std": std,
            # 标称值与变异系数由这里补齐：MCP 的响应里没有它们，
            # 而前端要拿 nominal 做"均值 vs 标称"的对照、拿 cv 判断离散程度。
            "nominal": float(pressure_bar),
            "cv": (std / mean) if mean else 0.0,
            "n": payload.get("n_simulations"),
            "rel_sigma": payload.get("input_uncertainty"),
            "unit": "kg/m³",
            "samples": payload.get("samples"),
            "source": "mcp",
        }

    log.info("  [System] MCP 密度结果缺少区间字段（旧版 Server 或测试替身）→ 本地计算")
    try:
        summary = co2_density_uq(
            float(pressure_bar), float(temperature_celsius),
            rel_sigma=0.05, n=100, keep_samples=True,
        )
        summary["source"] = "local"
        return summary
    except Exception as exc:  # noqa: BLE001
        log.warning(f"  [Warning] 本地密度计算失败: {type(exc).__name__}: {exc}")
        return None


def _plan_prompt_tools() -> str:
    """从 MCP Server 动态生成"可用工具"清单，客户端不硬编码工具名。

    为什么还要 ``TOOL_HINTS`` 这一层：Server 的 description 是给**外部**
    客户端（Cursor / Claude Desktop）看的通用说明，而这一阶段需要的是"本节点
    什么时候该调它"。两者不冲突，但职责不同，所以这里补一句上下文相关的提示。

    ``query_ccus_graph`` 刻意不列进规划器：地质分支已经检索过一轮，让 7B
    再规划一次同一批检索只会重复消耗 GPU。它改为**确定性触发**——只有当地质
    分支降级（没产出结论）时才由代码调用，用来补证据（见下方 fallback）。
    """
    try:
        specs = get_client_specs()
    except Exception as exc:  # noqa: BLE001
        log.warning(f"  [MCP] 工具发现失败，退回静态清单：{type(exc).__name__}: {exc}")
        return (
            "1. `get_co2_density` —— 参数中出现注入压力和温度时调用。\n"
            "2. `predict_pressure_pinn` —— 问到\"距井口 X 米、注入 T 天后\"的压力时调用。"
        )

    lines, examples = [], []
    for spec in specs:
        name = spec["name"]
        if name in TOOL_POLICY["skip_in_planner"]:
            continue
        hint = TOOL_HINTS.get(name) or spec.get("description", "")
        lines.append(f"{len(lines) + 1}. `{name}` —— {hint}")
        args = example_args(spec)
        if args:
            examples.append(json.dumps({"tool": name, **args}, ensure_ascii=False))

    body = "\n".join(lines)
    blocks = "\n".join(f"```json\n{ex}\n```" for ex in examples)
    return f"{body}\n\n只输出 json 代码块，每个工具一个，不要输出任何解释或正文。\n{blocks}"


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

    # ─── Prompt 组装：纯函数，见 nodes/aggregate_prompt.py ───
    system_prompt = build_system_prompt(file_memory_block)
    materials = build_materials(state, branch, personal_context)

    # ══════════════════════════════════════════════════════
    # Phase A：工具规划（短输出、非流式）
    # ══════════════════════════════════════════════════════
    plan_prompt = materials + f"""

【任务：只做工具规划，不要写报告】
你有以下工具（清单由 MCP Server 动态提供，本节点不硬编码工具名）：
{_plan_prompt_tools()}
```json
{{"tool": "generate_pressure_curve", "depth": 2500, "injection_pressure_bar": 250}}
```
（`generate_pressure_curve` 是本地图表工具：必须调用，depth 取参数中的储层埋深；
参数里没有就省略该字段，系统会按默认值绘制并标注"估算"，**不要编造深度**。）
数值必须取自【提取的 CCUS 参数】。"""

    plan_msgs = [
        {"role": "system", "content": "你是工具调用规划器，只输出 JSON 代码块。"},
        {"role": "user", "content": plan_prompt},
    ]
    _, plan_model, plan_tier, _ = get_llm("report_plan")
    try:
        plan_resp = llm_chat("report_plan", messages=plan_msgs, temperature=0.0)
    except Exception as exc:
        # 工具规划是**可选项**：它只决定要不要额外算几个数值（密度、压力曲线、
        # PINN），报告本身不依赖它。本地 Ollama 实测会在这个阶段整段不吐 token
        # （服务端 /api/ps 仍正常），客户端超时后如果继续上抛，一次卡顿就会把
        # 整条 case 拖成失败 —— 降级成"不调用任何工具"显然更划算。
        log.warning(
            f"  [Warning] 工具规划失败（{type(exc).__name__}: {exc}）→ 本次不调用任何工具，直接写报告"
        )
        plan_resp = None
    if plan_resp is None:
        state_update = {}
    else:
        state_update = record_llm_usage(
            {**state, **compress_update}, "report_plan", plan_resp, plan_model, plan_tier
        )
    usage_state = {**state, **compress_update, **state_update}

    # 扫描**所有** json 代码块（一份报告可能同时要密度和图）
    tool_requests = []
    plan_text = (plan_resp.choices[0].message.content or "") if plan_resp else ""
    for block in re.findall(r'```json\s*(\{.*?\})\s*```', plan_text, re.DOTALL):
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
    graph_extra = ""
    for req in tool_requests:
        name = req.get("tool")
        if name == "get_co2_density":
            # 压力/温度一律从提取出的参数里重新取，不采用模型给的数字：
            # 规划 prompt 里有 {"pressure_bar": 250, ...} 的示例，7B 会照抄。
            p_arg, t_arg = extract_pressure_temperature(state.get("extracted_params", ""))
            p_arg = req.get("pressure_bar") if p_arg is None else p_arg
            t_arg = req.get("temperature_celsius") if t_arg is None else t_arg
            if p_arg is None or t_arg is None:
                log.warning(f"  [Warning] 密度指令缺压力/温度，且参数里也没有：{req}")
                continue
            density_result = _call_density(p_arg, t_arg)
        elif name == "generate_pressure_curve":
            try:
                chart_result = generate_pressure_curve(
                    req.get("depth"), req.get("injection_pressure_bar")
                )
            except Exception as e:
                log.warning(f"  [Warning] 图表生成失败: {e}")
        elif name == "predict_pressure_pinn":
            # 坐标必须能在**用户原话**里核对到，不能信模型给的数字。
            # 原因很具体：规划 prompt 里有一个 {"x": 50, "t": 10} 的示例，
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
        elif name == "query_ccus_graph":
            # 规划器里没有列出它（见 _plan_prompt_tools 的说明），这里仍然
            # 兼容：模型自己写出来时，用用户原话当 query，避免它编检索词。
            payload = _execute_tool(
                "query_ccus_graph", {"query": state.get("user_input", ""), "top_k": 4}
            )
            frags = payload.get("fragments") or []
            if frags:
                graph_extra = "\n".join(f"- {str(f)[:400]}" for f in frags)
                log.info(f"  [MCP] query_ccus_graph 返回 {len(frags)} 条片段")

    # ─── 确定性兜底：模型该调没调时，由代码补上 ─────────────
    # 本地 7B 模型在上下文变丰富后会"觉得自己能算"，从而跳过工具调用，
    # 于是密度变成模型口算值 —— 这会直接破坏"数值不可幻觉"的核心承诺。
    # 因此不让它靠自觉：参数里有 P/T 就必须有工具结果。
    p_val, t_val = extract_pressure_temperature(state.get("extracted_params", ""))
    if density_result is None and p_val is not None and t_val is not None:
        log.info("  [System] 模型未规划密度计算，但参数含压力/温度 → 确定性兜底")
        density_result = _call_density(p_val, t_val)

    # ─── 不确定性量化（确定性触发，不经模型规划）───────────
    # 这一块和密度/图表兜底同一个理由，但更强：UQ 不是"锦上添花的工具"，
    # 而是报告的**固定组成部分**。交给 7B 决定要不要评估不确定性，
    # 它几乎总会在"信息已经够了"的时候跳过 —— 而恰恰是那种自信最危险。
    depth_val = extract_depth(state.get("extracted_params", ""))
    uq_summary: dict = {}
    if density_result and density_result.get("mean") is not None:
        uq_summary["density"] = {
            "mean": density_result.get("mean"),
            "p5": density_result.get("p5"),
            "p95": density_result.get("p95"),
            "std": density_result.get("std"),
            "n": density_result.get("n"),
            "rel_sigma": density_result.get("rel_sigma"),
            "unit": density_result.get("unit", "kg/m³"),
            "text": density_result.get("text", ""),
            "samples": density_result.get("samples"),
        }

    # 注入风险：深度与压力都从**提取出的参数**里取，不采用模型给的数字
    if p_val is not None and depth_val is not None:
        try:
            uq_summary["injection"] = injection_risk_uq(
                depth_val, p_val, rel_sigma=0.05, n=100
            )
            log.info(f"  [UQ] 注入风险分布：{uq_summary['injection']['text']}")
        except Exception as e:
            log.warning(f"  [Warning] 注入风险蒙特卡洛失败（不影响报告）: {e}")

    if uq_summary:
        log.info(
            f"  [UQ] 不确定性量化完成："
            f"{len(uq_summary)} 项（每项 100 次模拟）"
        )
        # 推给前端：UQ 面板不必等报告写完
        if sink is not None:
            try:
                sink({"type": "uq", "uq": uq_payload(uq_summary)})
            except Exception:
                pass

    # ─── 地质分支降级时，用图谱工具补证据 ────────────────────
    # query_ccus_graph 不交给 7B 规划（会重复检索），而是绑定在这个明确的
    # 条件下：地质分支没产出结论时，才值得再检索一次来补。
    if not branch["geology_ok"] and not graph_extra:
        log.info("  [System] 地质分支降级 → 调用 query_ccus_graph 补充证据")
        payload = _execute_tool(
            "query_ccus_graph", {"query": state.get("user_input", ""), "top_k": 4}
        )
        frags = payload.get("fragments") or []
        if frags:
            graph_extra = "\n".join(f"- {str(f)[:400]}" for f in frags)
            log.info(f"  [MCP] 补到 {len(frags)} 条图谱片段")
        else:
            log.warning(f"  [MCP] 图谱补充检索无结果：{payload or '空响应'}")

    # 图表同理：报告没有配图会显得业余，且输入几乎总能从参数推出。
    if chart_result is None or chart_result.get("status") != "ok":
        depth_val = extract_depth(state.get("extracted_params", ""))
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
                sink({"type": "chart", "chart": chart_payload(chart_result)})
            except Exception:
                pass

    # ══════════════════════════════════════════════════════
    # Phase B：终稿（流式，唯一被流式展示的一遍）
    # ══════════════════════════════════════════════════════
    tool_facts = []
    if density_result and density_result.get("text"):
        tool_facts.append(
            f"CO2 密度 = {density_result.get('text')}（由物理工具 + 蒙特卡洛计算）"
        )
    if chart_ok:
        tool_facts.append(f"地层压力曲线结论：{chart_result['summary']}")
    if pinn_result is not None:
        tool_facts.append(f"PINN 压力预测：{pinn_result['text']}")
    injection_uq = uq_summary.get("injection") or {}
    if injection_uq.get("text"):
        tool_facts.append(f"注入风险概率分布：{injection_uq['text']}")

    facts_block = ""
    if tool_facts:
        facts_block = (
            "【工具执行结果 —— 必须写进报告正文】\n- " + "\n- ".join(tool_facts)
            + "\n\n"
        )

    # 概率性结论的硬约束单独成块（由 core.uq 生成，不是 System Prompt 里的
    # 一句"请包含不确定性"）。放在工具结果后面，因为它是"怎么用这些数"的说明。
    prob_block = uq_block(uq_summary)
    if prob_block:
        facts_block += prob_block + "\n"

    # 工具结果放在**最前面**：实测本地模型对结尾处的约束遵守度很差，
    # 把数值要求前置后，模型更可能把密度直接写进正文而不是靠代码兜底补。
    # 图谱补充证据（只有地质分支降级时才会有）单独成块，不要混进"工具结果"：
    # 那一段的措辞是"数值必须原样写进正文"，而这里是待判断的证据片段。
    graph_block = ""
    if graph_extra:
        graph_block = (
            "\n\n【补充图谱证据（query_ccus_graph 检索，因地质分支缺失而补取）】\n"
            + graph_extra
            + "\n以上为原始证据片段，引用时注明来自知识图谱检索，不要当作已核实的结论。"
        )

    final_user = (
        facts_block
        + materials
        + graph_block
        + "\n\n请撰写完整报告。上面工具结果里的数值必须原样写进正文"
          "（例如在地质风险部分点明该压力/温度下的 CO2 密度），"
          "**并且必须带上 90% 置信区间与风险概率，不能只写单一数值**；"
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

    # ─── 收尾：数值兜底 / 降级说明 / 图表引用 / 概率溯源 ───
    # 全部是确定性文本变换，见 nodes/aggregate_finalize.py
    probability_source = " ".join(
        [
            str(state.get("geology_evidence") or ""),
            str(state.get("wellbore_evidence") or ""),
            str(graph_extra or ""),
            str(state.get("retrieved_knowledge") or ""),
            " ".join(str(fact) for fact in tool_facts),
        ]
    )
    report_text, stripped_probabilities = finalize_report(
        report_text,
        density_result=density_result,
        uq_summary=uq_summary,
        pinn_result=pinn_result,
        branch=branch,
        chart_result=chart_result,
        chart_ok=chart_ok,
        probability_source=probability_source,
    )

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
        # 地质分支降级时补取的图谱片段：它确实进了报告 Prompt，因此必须能被
        # 审计（API / UI / RAGAS 评估都要能看到模型实际依据了什么）。
        "supplemental_evidence": graph_extra,
        # 确定性工具结果（密度 / 压力曲线 / PINN / 注入风险）同样进了 Prompt，
        # 报告里复述这些数值是有依据的，不该在忠实度评估里被算成"编造"。
        "tool_facts": list(tool_facts),
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
        "chart_info": chart_payload(chart_result) if chart_ok else {},
        "pinn_prediction": pinn_result or {},
        # 不确定性量化结果：结构化字段，供 API / 前端 / 评审直接消费。
        # 报告正文里那段话是模型转述，可能被漏写；这个字段才是数值的可靠来源。
        "uq_summary": uq_payload(uq_summary),
        **compress_update,
        **state_update,
    }
