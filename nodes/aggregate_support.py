"""
汇总节点的纯函数支撑（工具清单 / 参数抽取 / 序列化裁剪）
========================================================
这些函数原来散在 ``nodes/aggregate_node.py`` 里，和 MCP 调用、流式输出、
记忆写入混在同一个文件。它们有一个共同点：**不碰模型、不碰网络、不碰文件**，
输入输出都是普通 Python 对象 —— 所以它们可以被单测穷举，而节点里那些
"真的会调模型"的部分不能。

抽出来的另一个好处是可读性：读 ``aggregate_node.py`` 的人关心的是
"报告是怎么一步步产出的"，而不是"图表结果要保留哪几个键"。
"""
from __future__ import annotations

import json
import re


def example_args(spec: dict) -> dict:
    """按 schema 造一个示例参数，缺省值优先用领域里说得通的数字。"""
    out = {}
    for key in spec.get("required") or []:
        if key in EXAMPLE_VALUES:
            out[key] = EXAMPLE_VALUES[key]
            continue
        kind = (spec.get("properties", {}).get(key) or {}).get("type")
        out[key] = 0.0 if kind == "number" else (0 if kind == "integer" else "…")
    return out


def get_client_specs() -> list[dict]:
    """取 MCP 工具清单（薄封装，便于测试替换）。"""
    from mcp_client_ccus import get_client

    return get_client().list_tools()


# 规划器里出现的工具提示：讲"什么时候调"，不是"这个工具是什么"
TOOL_HINTS = {
    "get_co2_density": (
        "参数中出现注入压力和温度时调用，不要自己口算密度。"
        "返回的是均值 + 90% 置信区间，报告里必须把区间一起写出来。"
    ),
    "predict_pressure_pinn": (
        "当问题问的是\"距井口 X 米处、注入 T 天后的压力\"时调用，"
        "由物理信息神经网络解一维扩散方程给出数值，"
        "**不要自己推算压力随距离/时间的衰减**。x 单位米（0~100），"
        "t 单位天（0~30）。"
    ),
}

# 示例值只用于让 7B 看懂参数形状；真实参数一律从用户输入里重新抽取，
# 不采用模型给的数字（见 dispatch 里的接地守卫）。
EXAMPLE_VALUES = {
    "pressure_bar": 250,
    "temperature_celsius": 80,
    "x": 50,
    "t": 10,
    "query": "断层活化与注入压力的关系",
}

# 哪些工具**不交给模型规划**（确定性触发）。理由写在各条注释里。
TOOL_POLICY = {
    "skip_in_planner": {
        "query_ccus_graph",
        # 注入风险的概率分布是**确定性触发**的：只要参数里有深度和压力就一定要算，
        # 不能交给 7B 决定"这次要不要评估不确定性"。不确定性量化是报告的固定项，
        # 不是可选工具 —— 交给模型规划，等于让它自己决定要不要诚实。
        "assess_injection_uncertainty",
        # 断层活化与盖层封闭性同理，而且更严重：它们要的是**埋深 + 注入压力**，
        # 规划 prompt 里一旦出现示例值，7B 会照抄，而埋深被抄错会让断层判定
        # 整体偏掉且不报错（滑移比看起来仍然"合理"）。改为在 aggregate 里
        # 由代码按提取到的参数确定性调用（见 nodes/aggregate_node._assess_deep）。
        "get_fault_stability",
        "get_caprock_integrity",
    }
}


def extract_pressure_temperature(params: str) -> tuple[float | None, float | None]:
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


def extract_depth(params: str) -> float | None:
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


def chart_payload(result: dict | None) -> dict:
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


def uq_payload(uq: dict | None) -> dict:
    """
    把不确定性量化结果裁成可 JSON 序列化的结构。

    保留 ``samples``（100 个浮点，约 1 KB）：前端画分布直方图要靠它。
    只给 mean/p5/p95 的话，看板上只能显示三个数字，"分布"这件事就丢了 ——
    而分布恰恰是 UQ 唯一比点估计多出来的信息。

    v1.7 新增的四类字段各有必须保留的理由：

    - ``dist`` / ``dist_label``：不写分布名，"±5%"在均匀与正态下差一倍宽，
      看板上就没法和报告正文对齐口径；
    - ``corr`` / ``independent`` / ``corr_empirical``：独立抽样是一个**假设**，
      前端必须能显示"这次是按独立算的"还是"按 ρ=0.9 算的"；
    - ``sensitivity``：Sobol 的排名是给读者的行动建议，不是调试信息；
    - ``decision``：告警等级与阈值，看板要能直接做徽标。
    """
    if not uq:
        return {}
    keep = (
        "mean", "p5", "p50", "p95", "std", "cv", "n", "rel_sigma", "unit",
        "text", "nominal", "ci_level", "probabilities", "counts",
        "verdict", "verdict_label", "depth_m", "injection_pressure_bar",
        "dist", "dist_label", "corr", "independent", "corr_empirical", "priors",
        "decision",
    )
    out: dict = {}
    for key in ("density", "injection"):
        item = uq.get(key) or {}
        if not item:
            continue
        # MCP 路径返回的是原始工具输出，里面不一定有 cv，但一定有 std+mean。
        if item.get("cv") is None and item.get("std") is not None and item.get("mean"):
            item = dict(item)
            item["cv"] = item["std"] / item["mean"]
        slim = {k: item.get(k) for k in keep if item.get(k) is not None}
        samples = item.get("samples")
        if samples:
            slim["samples"] = [round(float(v), 4) for v in samples]
        out[key] = slim

    # 敏感性分析：结构是 dict-of-dict，不能套上面的 keep 白名单，
    # 但也不能整包透传 —— ``first_order_raw`` 之类的调试字段没必要过网络。
    sens = uq.get("sensitivity") or {}
    if sens:
        out["sensitivity"] = {
            k: sens.get(k) for k in (
                "n_base", "evaluations", "variance", "mean", "names",
                "first_order", "total_order", "interaction",
                "first_order_se", "total_order_se",
                "ranking", "event", "event_probability", "notes", "text",
            ) if sens.get(k) is not None
        }

    # 决策判定：顶层放一份，前端不必先钻进 injection 里找。
    decision = uq.get("decision") or {}
    if decision:
        out["decision"] = {
            k: decision.get(k) for k in (
                "level", "level_label", "triggered", "resolved", "metric",
                "label", "p", "p_ci", "hits", "n", "rule", "thresholds",
                "notes", "text",
            ) if decision.get(k) is not None
        }
    return out


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
