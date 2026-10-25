"""
深层物理评估：把地应力、盖层、两相流串成一次判定
================================================
``physics_rules`` 回答"注入压力对不对"，这个模块回答"会不会滑、会不会漏"。
两者的关系是**串联而不是替代**：

    physics_rules    P_inj vs P_h / P_frac        快、只看两条线、总能算
    physics_deep     断层活化 + 盖层突破 + 两相流  慢、需要更多输入、可能无数据

## 阻断策略（这是本模块最重要的设计决定）

深层判定依赖**假设参数**（Shmin/SHmax 比例、孔喉半径）。用假设值去阻断报告，
会造成另一类错误：把本来没问题的项目拦下来，而且拦下来的理由是"按默认应力
剖面你会滑"。所以阻断只在两种情况下发生：

1. 输入**明确涉及断层**（出现"断层""断裂"等词），且给了倾角 —— 此时默认
   应力剖面是这个场地唯一可用的信息，用它判定是合理的；
2. 输入**明确给了** CO₂ 柱高或注入速率 —— 这时盖层判定用的是实际数字。

其余情况一律只输出信息（含全部假设与敏感性），不阻断。这个策略写进
``policy`` 字段随结果返回，报告里能看到"为什么这次没拦"。

## 与质量闸门的关系

深层判定同样只产出**事实**，不做评分。阻断时给的是"哪条判据、差多少"，
不是"报告写得不好"。模型分不能覆盖它 —— 与 ``physics_rules`` 同一条原则。
"""
from __future__ import annotations

import re

from utils import caprock as cp
from utils import geomechanics as gm
from utils import relperm as rp
from utils import reservoir_backend as rb

DEEP_VERSION = "2026.10-d1"

# "断层"类词。命中才算"这个场地有断层"，才允许用假设应力剖面做阻断。
_FAULT_RE = re.compile(r"断层|断裂|fault|faults", re.I)
_DIP_RE = re.compile(r"倾角[^\d]{0,8}(\d+(?:\.\d+)?)\s*[°度]")
# 柱高/注入速率的显式提及
_COLUMN_RE = re.compile(r"(?:CO2|CO₂|二氧化碳)?\s*柱高[^\d]{0,8}(\d+(?:\.\d+)?)\s*(m|米)")
_RATE_RE = re.compile(r"注入(?:速率|流量|量)[^\d]{0,8}(\d+(?:\.\d+)?)\s*(万)?\s*(?:m3|m³|方|吨)")


def extract_fault_context(text: str) -> dict:
    """
    从文本里抓断层信息。抓不到就给 None，**不猜**。

    只抓倾角：走向、倾向、断距这些即使抓到了也未必对，而倾角对滑移判据
    影响最大且最容易被写出来（"断层倾角 60°"）。抓不到倾角时用 60°
    （最优产状附近），并在结果里标注这是默认值。
    """
    text = text or ""
    has_fault = bool(_FAULT_RE.search(text))
    dip = None
    m = _DIP_RE.search(text)
    if m:
        value = float(m.group(1))
        if 5.0 <= value <= 89.0:      # 超出范围一定是抽取错误
            dip = value
    return {"has_fault": has_fault, "dip_deg": dip}


def extract_column_context(text: str) -> dict:
    """抓 CO₂ 柱高与注入速率的显式提及。"""
    text = text or ""
    column = None
    m = _COLUMN_RE.search(text)
    if m:
        value = float(m.group(1))
        if m.group(2) == "米" or m.group(2) == "m":
            column = value
    rate = None
    m = _RATE_RE.search(text)
    if m:
        value = float(m.group(1))
        if m.group(2):                # "万方" → ×1e4
            value *= 1e4
        if 1.0 <= value <= 1e6:
            rate = value
    return {"column_m": column, "rate_m3_per_day": rate}


def assess(depth_m: float | None,
           injection_pressure_bar: float | None,
           temperature_celsius: float | None = None,
           text: str = "",
           environ: dict | None = None) -> dict:
    """
    跑一遍深层判定。

    参数不足时返回 ``status="no_data"`` 并说明缺什么，不抛异常 ——
    物理校验"没算过"和"算出来不合格"是两件完全不同的事，必须能区分
    （``physics_rules.render_prompt_block`` 里踩过这个坑：把"没算过"渲染成
    "未通过"，模型会据此扣分）。

    返回：
        status          "ok" | "no_data"
        blocking        bool（策略见模块 docstring）
        blocking_reasons list[str]
        geomechanics    断层活化结果（无断层时为 None）
        caprock         盖层结果
        relperm         两相流特征
        policy          本次阻断策略的说明
        assumptions     汇总的假设参数
        version
    """
    if depth_m is None or injection_pressure_bar is None:
        return {
            "status": "no_data",
            "missing": [k for k, v in (("depth_m", depth_m),
                                       ("injection_pressure_bar", injection_pressure_bar))
                        if v is None],
            "blocking": False,
            "version": DEEP_VERSION,
            "note": "缺少埋深或注入压力，未做深层物理判定",
        }

    temp = float(temperature_celsius) if temperature_celsius is not None else 60.0
    fault = extract_fault_context(text)
    column_ctx = extract_column_context(text)

    # ── 断层活化 ────────────────────────────────────────
    geo = None
    if fault["has_fault"]:
        dip = fault["dip_deg"] or 60.0
        geo = gm.fault_reactivation(
            depth_m=depth_m,
            injection_pressure_bar=injection_pressure_bar,
            dip_deg=dip,
            environ=environ,
        )
        geo["dip_source"] = "input" if fault["dip_deg"] else "default-60"
    elif fault["dip_deg"]:
        # 给了倾角但没提"断层"：多半是在说别的（地层倾角），不判
        geo = None

    # ── 盖层封闭性 ──────────────────────────────────────
    cap = cp.caprock_integrity(
        depth_m=depth_m,
        injection_pressure_bar=injection_pressure_bar,
        temperature_celsius=temp,
        co2_column_height_m=column_ctx["column_m"],
        injection_rate_m3_per_day=column_ctx["rate_m3_per_day"],
        environ=environ,
    )

    # ── 两相流特征（总是可算，纯信息）────────────────────
    rel = rp.bl_front()

    reasons: list[str] = []
    policy = []
    if geo is not None:
        policy.append("输入提及断层 → 用默认应力剖面判定并允许阻断")
        policy.append(
            "断层判定走保守口径（总应力不变）；孔隙弹性应力路径只作参考，"
            "因为它仅对注入层成立、不含盖层与隔层的应力转移"
        )
        if geo.get("blocking"):
            pl = geo["plane"]
            if geo.get("mechanism") == "tensile":
                # 张性开启不能套剪切判据的措辞：这里"余量"还是正的，但面已经
                # 被顶开了。写成"滑移比 inf + 余量 182 bar"会让人以为判据自相
                # 矛盾，实际是两条不同的机制。
                reasons.append(
                    f"断层张性开启：注入压力 {injection_pressure_bar:.0f} bar 已超过"
                    f"断层面（倾角 {pl['dip_deg']:.0f}°）正应力 "
                    f"{pl['sigma_n_mpa'] * 10:.0f} bar，面被顶开、流体获得向上通道；"
                    f"此时摩擦系数与黏聚力都不是控制因素"
                )
            else:
                reasons.append(
                    f"断层活化：注入压力 {injection_pressure_bar:.0f} bar 使断层面"
                    f"（倾角 {pl['dip_deg']:.0f}°）滑移比达到 "
                    f"{geo['slip_ratio']}（≥1 即滑动），临界孔隙压力 "
                    f"{geo['pp_crit_bar']} bar，当前余量 {geo['margin_bar']} bar"
                )
    else:
        policy.append("输入未提及断层 → 不做断层判定，避免用假设应力剖面阻断")

    if cap.get("verdict") not in (None, "no_data") and cap.get("blocking"):
        policy.append("输入给出了柱高或注入速率 → 盖层判定可阻断")
        reasons.append(f"盖层失效：{cap.get('summary')}")
    elif cap.get("verdict") == "no_data":
        policy.append("未给柱高/注入速率 → 盖层只给突破压力与可封柱高，不阻断")

    return {
        "status": "ok",
        "blocking": bool(reasons),
        "blocking_reasons": reasons,
        "geomechanics": geo,
        "caprock": cap,
        "relperm": rel,
        "policy": policy,
        "assumptions": sorted(set(
            (geo or {}).get("assumptions", [])
            + cap.get("assumptions", [])
            + ((geo or {}).get("coupled") or {}).get("assumed", [])
        )),
        "version": DEEP_VERSION,
    }


def render_block(result: dict) -> str:
    """把深层判定渲染成报告段落。"""
    if not result or result.get("status") != "ok":
        return ""
    parts = []
    if result.get("geomechanics"):
        parts.append(gm.render_block(result["geomechanics"]))
    if result.get("caprock"):
        parts.append(cp.render_block(result["caprock"]))
    if result.get("relperm"):
        parts.append(rp.render_block(result["relperm"]))
    if result.get("assumptions"):
        parts.append(
            "【前提】以下参数使用默认假设值，不是实测："
            + "、".join(result["assumptions"])
            + "。这些结论仅在该假设成立时有效。"
        )
    return "\n\n".join(p for p in parts if p)


_GEO_VERDICT_CN = {
    "stable": "未活化",
    "near_critical": "接近活化（余量不足 20%）",
    "reactivated": "已活化",
    "tensile_opening": "张性开启",
}
_CAP_VERDICT_CN = {
    "sealed": "封闭",
    "capillary_breakthrough": "毛细管突破",
    "mechanical_failure": "力学失效",
}


def _fmt(value, nd: int = 0) -> str:
    """数值格式化。

    必须先判字符串：``geomechanics._round`` 会把 ``inf`` 转成 ``"inf"``
    （有效正应力被压到 0 时滑移比是无穷大），直接套 ``:.0f`` 会抛
    ``ValueError``，而这个格式错误会发生在渲染报告的那一刻 —— 即"算对了、
    写不出来"。
    """
    if isinstance(value, str):
        return value
    return f"{float(value):.{nd}f}"


def render_facts(result: dict) -> list[str]:
    """
    报告 Prompt 用的紧凑事实行（每条一行）。

    与 ``render_block`` 的分工：那段是给人读的完整段落（敏感性、局限、逐条
    假设），这里只留**必须原样写进正文的数值**。

    为什么把假设压在同一行而不是另起一行：7B 抄的时候是按行抄的。前提单独
    成行时，正文里只会出现"滑移比 0.87"，于是"按假设应力剖面算出来的"变成
    了"这个场地实测的"——把一个通用剖面的结论写成了场地结论，正是本项目
    一直在防的那类失真。所以前提必须和数字待在同一行里。

    返回空列表表示"没有可写进正文的数值"，调用方据此整块省略。
    """
    if not result or result.get("status") != "ok":
        return []
    facts: list[str] = []

    geo = result.get("geomechanics")
    if geo:
        st, pl = geo["state"], geo["plane"]
        line = (
            f"断层稳定性：倾角 {pl['dip_deg']:.0f}° 的断层面在 {st['depth_m']:.0f} m 处 "
            f"σn = {pl['sigma_n_mpa']:.1f} MPa、τ = {pl['tau_mpa']:.1f} MPa；"
            f"注入后滑移比 {_fmt(geo['slip_ratio'], 2)}（≥1 即滑动），临界孔隙压力 "
            f"{_fmt(geo['pp_crit_bar'])} bar，余量 {_fmt(geo['margin_bar'])} bar，"
            f"判定：{_GEO_VERDICT_CN.get(geo['verdict'], geo['verdict'])}"
            f"（前提：Shmin/SHmax 由 Sv 按比例假设，非实测）"
        )
        # 两个口径结论不同时才加这一句：一致时加只会让正文变长、信息量为零。
        # 必须留在同一行 —— 7B 按行抄，"仅对注入层成立"离开数字就成了免责声明。
        coupled = geo.get("coupled") or {}
        if coupled and coupled.get("verdict") != geo["verdict"]:
            line += (
                f"；计入注入层孔隙弹性应力路径后滑移比 {_fmt(coupled['slip_ratio'], 2)}、"
                f"余量 {_fmt(coupled['margin_bar'])} bar，判定："
                f"{_GEO_VERDICT_CN.get(coupled['verdict'], coupled['verdict'])}"
                f"（该口径只对注入层成立，不含应力转移）"
            )
        facts.append(line)

    cap = result.get("caprock") or {}
    if cap.get("entry_pressure_bar"):
        line = (
            f"盖层封闭性：毛细管突破压力 {cap['entry_pressure_bar']:.0f} bar，"
            f"可封 CO₂ 柱高 {cap['max_column_height_m']:.0f} m，"
            f"密度差 Δρ = {cap['density_difference']:.0f} kg/m³（Peng-Robinson）"
        )
        if cap.get("verdict") not in (None, "no_data"):
            line += (
                f"；本次柱高 {cap['column_height_m']:.0f} m → 浮力 "
                f"{cap['buoyancy_bar']:.1f} bar，余量 {cap['margin_bar']:.1f} bar，"
                f"判定：{_CAP_VERDICT_CN.get(cap['verdict'], cap['verdict'])}"
            )
        trap = cap.get("trap") or {}
        if trap:
            line += (
                f"；圈闭口径（面积 {trap['trap_area_m2']:.0f} m²）同一时刻柱高 "
                f"{trap['column_height_m']:.0f} m（一维口径 "
                f"{trap['plume_column_height_m']:.0f} m，{trap['column_ratio']}×），"
                f"长到可封上限需 {trap['fill_days_early']:.0f} 天（一维口径 "
                f"{trap['fill_days_late']:.0f} 天，早界为保守侧）"
            )
        line += "（前提：孔喉半径与接触角取文献默认值，非 MICP 实测）"
        facts.append(line)

    rel = result.get("relperm") or {}
    if rel.get("co2_saturation"):
        facts.append(
            f"两相流（Buckley-Leverett）：前缘 CO₂ 饱和度 {rel['sg_front']:.2f}，"
            f"波及区平均 CO₂ 饱和度 {rel['co2_saturation']:.2f}，"
            f"流度比 M = {rel['mobility_ratio']:.1f}（>1 说明 CO₂ 指进、波及效率低）"
            f"（前提：Brooks-Corey 默认相渗参数）"
        )
    return facts


def run_simulator_case(depth_m: float, injection_pressure_bar: float,
                       temperature_celsius: float = 60.0,
                       injection_rate_m3_per_day: float | None = None,
                       external_path: str | None = None) -> dict:
    """
    走模拟器适配层跑一次。外部结果优先，缺失时降级到解析并**写明原因**。
    """
    case = rb.SimulationCase(
        name="inline-assessment",
        depth_m=depth_m,
        injection_pressure_bar=injection_pressure_bar,
        temperature_celsius=temperature_celsius,
        injection_rate_m3_per_day=injection_rate_m3_per_day,
    )
    return rb.run_case(case, external_path=external_path)
