"""
物理一致性校验（规则引擎，不是模型打分）
========================================
Agent 4 原本只做一件事：让大模型给报告打个 0~10 分。问题是这个分数是**黑盒**——
模型可能给一份"注入压力已超过地层破裂压力、却仍判定风险可控"的报告打 9 分，
因为它的措辞漂亮、结构完整。这类错误在 CCUS 里是致命的：报告会被当作工程依据。

所以这里把"物理上说不说得通"从模型手里拿走，交给**确定性代码**：

    静水压力   P_h(d)    = ρ_w·g·d / 1e5      ρ_w = 1030 kg/m³
    地层破裂压力 P_frac(d) = 1.5 · P_h(d)      破裂压力梯度常用 1.4–1.8 倍静水

    注入压力 P_inj 与这两条线的关系只有三种，没有第四种：
      P_inj > P_frac  → 盖层被压裂 / CO2 泄漏，**必须**在报告里点明
      P_inj < P_h     → 注不进去，**不能**说"注入顺畅"
      其余             → 落在安全注入窗口内

规则分两级：
  **blocking**  —— 报告与硬物理约束**矛盾**（说安全，实际超压；说顺畅，实际注不进）。
                  直接 0 分打回重写，并给出可执行的修正提示。
  **warning**   —— 数值对不上、单位可疑、遗漏说明。记录但不阻断，
                  因为"报告没写"和"报告写错了"不是一回事。

设计上刻意与 ``chart_tools`` 解耦（那里依赖 matplotlib）：本模块只依赖标准库，
CI 的 unit-core 能零安装地跑它。代价是两条压力公式各写了一份 ——
``tests/test_physics_rules.py`` 会在有 matplotlib 时交叉校验两处结果一致，
防止它们悄悄漂移。
"""
from __future__ import annotations

import re

# ─── 物理常数（必须与 chart_tools 保持一致）───────────────
RHO_WATER = 1030.0          # kg/m³，含盐地层水
G = 9.81                    # m/s²
FRACTURE_RATIO = 1.5        # 破裂压力 / 静水压力
BAR_PER_PA = 1e-5

# 合理区间：超出这个范围几乎一定是单位写错（MPa 当 bar、kPa 当 bar）
MIN_PLAUSIBLE_BAR = 20.0
MAX_PLAUSIBLE_BAR = 1200.0

# 破裂压力数值的容差。取 25% 不是随手定的：工程上破裂梯度常用
# 1.4–1.8 倍静水，相对本模块的 1.5 倍分别是 -6.7% 与 +20%。
# 容差必须覆盖这个合法区间，否则每份用了 1.8 倍梯度的报告都会被误判。
FRACTURE_VALUE_TOLERANCE = 0.25

# 密度容差（简化状态方程 ρ = 1.9P/T，允许模型做四舍五入）
DENSITY_TOLERANCE = 0.15

_CN_NUM = r"(\d+(?:\.\d+)?)"

# ─── 文本抽取用的正则 ────────────────────────────────────
_INJECTION_RE = re.compile(
    r"(?:注入压力|井底压力|注入井底压力|注入压|injection\s*pressure)"
    r"[^\d\-]{0,12}" + _CN_NUM + r"\s*(bar|Bar|BAR|兆帕|MPa|mpa|kPa|psi)?",
    re.I,
)
# 没有"注入"修饰的裸"压力"，用于兜底；后面会排除掉破裂/静水/地层语境
_BARE_PRESSURE_RE = re.compile(
    r"(?<!破裂)(?<!地层)(?<!静水)(?<!孔隙)(?:压力|pressure)"
    r"[^\d\-]{0,10}" + _CN_NUM + r"\s*(bar|Bar|BAR|兆帕|MPa|mpa|kPa|psi)?",
    re.I,
)
_FRACTURE_VALUE_RE = re.compile(
    r"(?:地层破裂压力|破裂压力|fracture\s*pressure)"
    r"[^\d\-]{0,12}" + _CN_NUM + r"\s*(bar|Bar|BAR|兆帕|MPa|mpa|kPa|psi)?",
    re.I,
)
_DEPTH_RE = re.compile(
    r"(?:埋深|储层深度|地层深度|深度|depth)"
    r"[^\d\-]{0,12}" + _CN_NUM + r"\s*(km|千米|公里|m|米|meters?)?",
    re.I,
)
_TEMP_RE = re.compile(
    r"(?:温度|temperature)[^\d\-]{0,12}" + _CN_NUM + r"\s*(℃|°C|C|摄氏)?",
    re.I,
)
_DENSITY_RE = re.compile(r"密度[^\d]{0,12}" + _CN_NUM + r"\s*kg\s*/\s*m", re.I)

_FRACTURE_TERMS = ("破裂压力", "压裂压力", "地层破裂", "压裂", "fracture")
_COMPARISON_RE = re.compile(r"超过|高于|超出|大于|突破")
# 否定词要按"紧邻前缀"匹配。"注入压力未超过破裂压力" 里 "超过" 前面是 "力未"，
# 所以不能用"窗口内包含否定词"来判断 —— 必须看前缀的**结尾**是不是否定词。
# "没有" 单独列出来，是因为它结尾是"有"，不会被单字"没"命中。
_NEGATIONS = ("未", "不", "没", "无", "非", "没有")

# "注入顺畅"类断言：与 P_inj < P_h 直接矛盾
_SMOOTH_CLAIMS = (
    "注入顺畅", "注入顺利", "注入压力充足", "压力充足", "注入能力充足",
    "可顺利注入", "能够顺利注入", "无注入困难", "不存在注入困难",
    "不会出现注入困难", "不会出现注入压力不足",
)


# ══════════════════════════════════════════════════════════
# 压力公式
# ══════════════════════════════════════════════════════════
def hydrostatic_bar(depth_m: float) -> float:
    """深度（m）→ 静水压力（bar）。1 bar = 1e5 Pa。"""
    return RHO_WATER * G * float(depth_m) * BAR_PER_PA


def fracture_bar(depth_m: float) -> float:
    """深度（m）→ 地层破裂压力（bar）。"""
    return hydrostatic_bar(depth_m) * FRACTURE_RATIO


# ══════════════════════════════════════════════════════════
# 单位归一化
# ══════════════════════════════════════════════════════════
def _to_bar(value: float, unit: str | None) -> float:
    """把压力统一成 bar。没有单位时按 bar 处理（国内工程文档的默认写法）。"""
    u = (unit or "bar").strip().lower()
    if u in ("mpa", "兆帕"):
        return value * 10.0
    if u == "kpa":
        return value / 100.0
    if u == "psi":
        return value * 0.0689476
    return value


def _to_meters(value: float, unit: str | None) -> float:
    u = (unit or "m").strip().lower()
    if u in ("km", "千米", "公里"):
        return value * 1000.0
    return value


def _first(pattern: re.Pattern, text: str):
    m = pattern.search(text or "")
    return m if m else None


# ══════════════════════════════════════════════════════════
# 参数抽取
# ══════════════════════════════════════════════════════════
def _extract_from(text: str) -> dict:
    """从一段文本里抓参数。抽不到的字段为 None。"""
    p_inj = None
    m = _first(_INJECTION_RE, text)
    if m:
        p_inj = _to_bar(float(m.group(1)), m.group(2))
    else:
        # 兜底：裸"压力"，但要求前面不是破裂/静水/地层语境
        m = _first(_BARE_PRESSURE_RE, text)
        if m:
            p_inj = _to_bar(float(m.group(1)), m.group(2))

    temperature = None
    m = _first(_TEMP_RE, text)
    if m:
        temperature = float(m.group(1))

    depth = None
    m = _first(_DEPTH_RE, text)
    if m:
        depth = _to_meters(float(m.group(1)), m.group(2))
        # 明显不合理（把 3000 写成 3）就丢掉，避免用错误深度算破裂压力
        if not (50.0 <= depth <= 9000.0):
            depth = None

    fracture_stated = None
    m = _first(_FRACTURE_VALUE_RE, text)
    if m:
        fracture_stated = _to_bar(float(m.group(1)), m.group(2))

    return {
        "injection_pressure_bar": p_inj,
        "temperature_c": temperature,
        "depth_m": depth,
        "fracture_bar_stated": fracture_stated,
    }


def extract_parameters(params_text: str, user_input: str = "") -> dict:
    """
    从 Agent 1 的提取结果里抓出 P / T / 深度，**缺失字段回落到用户原文**。

    返回 ``{"injection_pressure_bar", "temperature_c", "depth_m",
    "fracture_bar_stated"}``，抓不到的字段为 ``None``。

    为什么注入压力优先匹配"注入压力"而不是裸"压力"：参数文本里常常同时出现
    "注入压力 250 bar"和"地层破裂压力 378 bar"，用裸"压力"会抓到后者，
    于是校验逻辑会拿破裂压力去和破裂压力比 —— 永远通过，等于没校验。

    为什么要做**逐字段**回落而不是"参数为空才回落"：Agent 1 的输出是模型的
    改写结果，它经常把埋深写成"储层深度未明确"或干脆省略，而用户原文里明明
    写着"埋深2500m"。只判空就会让物理校验静默地变成"无法验证"——
    看起来像"参数不足"，实际是抽取丢了信息。用户原文对**数值**才是权威。
    """
    primary = _extract_from((params_text or "").strip())
    if not (user_input or "").strip():
        return primary

    missing = [k for k, v in primary.items() if v is None]
    if not missing:
        return primary

    secondary = _extract_from(user_input)
    for key in missing:
        if secondary.get(key) is not None:
            primary[key] = secondary[key]
    return primary


# ══════════════════════════════════════════════════════════
# 报告断言识别
# ══════════════════════════════════════════════════════════
def _negated(sentence: str, pos: int) -> bool:
    """判断 ``pos`` 处的比较词是否被否定（"未超过""不超过""没有超过"）。"""
    prefix = sentence[max(0, pos - 3):pos]
    return any(prefix.endswith(n) for n in _NEGATIONS)


# "破裂压力被突破" 是**被动**表述：比较词在破裂压力之后，但语义上仍是
# "已超压"。方向性检查会漏掉它，所以单独补一条被动式。
_PASSIVE_EXCEEDED_RE = re.compile(
    r"(?:破裂压力|压裂压力|地层破裂)[^。！？；\n]{0,12}"
    r"(?:被|已被|遭到|已经)(?:超过|突破|打破|压穿)"
)


def acknowledges_fracture_exceeded(report: str) -> bool:
    """
    报告是否**明确点明**"注入压力已超过地层破裂压力"。

    判定是**方向性**的，不是关键词共现：比较词必须出现在"破裂压力"之前，
    且不能被否定。否则 "破裂压力 379 bar 高于注入压力" 这种**反向**表述
    （意思是安全的）会被误判成"已点明风险"，规则就形同虚设。
    被动式（"破裂压力被突破"）单独兜一条，因为它的语序正好相反。
    """
    text = report or ""
    if _PASSIVE_EXCEEDED_RE.search(text):
        return True

    for sentence in re.split(r"[。！？；\n]", text):
        if not any(t in sentence for t in _FRACTURE_TERMS):
            continue
        fpos = min(sentence.find(t) for t in _FRACTURE_TERMS if t in sentence)
        for m in _COMPARISON_RE.finditer(sentence):
            if m.start() >= fpos:
                continue            # 比较词在破裂压力之后 → 方向相反
            if _negated(sentence, m.start()):
                continue            # "未超过破裂压力" → 没有点明风险
            return True
    return False


def claims_smooth_injection(report: str) -> str:
    """报告是否断言"注入顺畅"。返回命中的原句（没有则空串）。"""
    for sentence in re.split(r"[。！？；\n]", report or ""):
        for claim in _SMOOTH_CLAIMS:
            if claim in sentence:
                return sentence.strip()
    return ""


# ══════════════════════════════════════════════════════════
# 主校验
# ══════════════════════════════════════════════════════════
def evaluate_physics(
    params_text: str = "",
    report_text: str = "",
    user_input: str = "",
    params: dict | None = None,
) -> dict:
    """
    对一份报告做物理一致性校验，返回结构化结果。

    返回：
        {
          "status": "pass" | "warning" | "violation" | "unverifiable",
          "blocking": bool,
          "verdict": "safe_window" | "above_fracture" | "below_hydrostatic" | "unknown",
          "params": {...}, "limits": {...},
          "violations": [{"code","severity","message"}],
          "checks": [{"code","ok","detail"}],
          "feedback": "给重写用的明确物理错误提示（仅 blocking 时非空）",
          "summary": "一行人类可读结论"
        }

    ``params`` 用于测试时直接注入数值，跳过文本抽取。
    """
    extracted = dict(params) if params else extract_parameters(params_text, user_input)
    p_inj = extracted.get("injection_pressure_bar")
    depth = extracted.get("depth_m")
    temperature = extracted.get("temperature_c")
    report = report_text or ""

    violations: list[dict] = []
    checks: list[dict] = []

    def violation(code, severity, message):
        violations.append({"code": code, "severity": severity, "message": message})

    def check(code, ok, detail):
        checks.append({"code": code, "ok": bool(ok), "detail": detail})

    # ─── 0. 压力合理性 ────────────────────────────────────
    if p_inj is None:
        check("injection_pressure_present", False, "参数中未提取到注入压力")
        return _result("unverifiable", extracted, {}, violations, checks,
                       "未提取到注入压力，无法做物理一致性校验")

    check("injection_pressure_present", True, f"{p_inj:.1f} bar")

    if p_inj > MAX_PLAUSIBLE_BAR or p_inj < MIN_PLAUSIBLE_BAR:
        violation(
            "IMPLAUSIBLE_PRESSURE", "warning",
            f"注入压力 {p_inj:.1f} bar 超出合理区间 "
            f"[{MIN_PLAUSIBLE_BAR:.0f}, {MAX_PLAUSIBLE_BAR:.0f}] bar，疑似单位错误"
            f"（如把 MPa 写成 bar）",
        )

    if depth is None:
        check("depth_present", False, "参数中未提供储层深度，无法计算破裂压力")
        # 深度缺失不是报告的错，只标 unverifiable，不阻断
        status = "warning" if violations else "unverifiable"
        return _result(status, extracted, {}, violations, checks,
                       "未提供储层深度，无法校验注入压力与破裂压力的关系")

    p_h = hydrostatic_bar(depth)
    p_frac = fracture_bar(depth)
    limits = {
        "hydrostatic_bar": round(p_h, 1),
        "fracture_bar": round(p_frac, 1),
        "safety_margin_bar": round(p_frac - p_inj, 1),
        "depth_m": round(float(depth), 0),
    }
    check("depth_present", True, f"{depth:.0f} m")

    # ─── 1. 注入压力 vs 破裂压力（硬约束）─────────────────
    if p_inj > p_frac:
        verdict = "above_fracture"
        over = p_inj - p_frac
        over_pct = over / p_frac * 100
        acknowledged = acknowledges_fracture_exceeded(report)
        check("pressure_vs_fracture", False,
              f"P_inj {p_inj:.1f} bar > P_frac {p_frac:.1f} bar（超 {over:.1f} bar）")
        check("fracture_exceeded_acknowledged", acknowledged,
              "报告已点明超破裂压力" if acknowledged else "报告未点明超破裂压力")
        if not acknowledged:
            violation(
                "P_INJ_ABOVE_FRACTURE_UNACKNOWLEDGED", "blocking",
                f"注入压力 {p_inj:.0f} bar 在 {depth:.0f} m 处超过地层破裂压力 "
                f"{p_frac:.0f} bar（超出 {over:.0f} bar / {over_pct:.0f}%），"
                f"报告却未指出该超压事实与盖层压裂、CO2 泄漏风险",
            )
    else:
        verdict = None
        check("pressure_vs_fracture", True,
              f"P_inj {p_inj:.1f} bar ≤ P_frac {p_frac:.1f} bar")

        # ─── 2. 注入压力 vs 静水压力（硬约束）─────────────
        if p_inj < p_h:
            verdict = "below_hydrostatic"
            claimed = claims_smooth_injection(report)
            check("pressure_vs_hydrostatic", False,
                  f"P_inj {p_inj:.1f} bar < P_h {p_h:.1f} bar（注入困难）")
            check("injection_difficulty_acknowledged", not claimed,
                  f"报告称「{claimed}」" if claimed else "报告未断言注入顺畅")
            if claimed:
                violation(
                    "P_INJ_BELOW_HYDROSTATIC_CLAIMED_SMOOTH", "blocking",
                    f"注入压力 {p_inj:.0f} bar 低于 {depth:.0f} m 处地层压力 "
                    f"{p_h:.0f} bar，CO2 无法自流注入；"
                    f"报告却称「{claimed}」，与物理约束矛盾",
                )
        else:
            verdict = "safe_window"
            check("pressure_vs_hydrostatic", True,
                  f"P_inj {p_inj:.1f} bar ≥ P_h {p_h:.1f} bar")

    # ─── 3. 报告引用的破裂压力是否离谱（warning）──────────
    stated = extracted.get("fracture_bar_stated")
    if stated:
        diff = abs(stated - p_frac) / p_frac
        check("fracture_value_consistent", diff <= FRACTURE_VALUE_TOLERANCE,
              f"报告 {stated:.0f} bar vs 计算 {p_frac:.0f} bar（偏差 {diff*100:.0f}%）")
        if diff > FRACTURE_VALUE_TOLERANCE:
            violation(
                "FRACTURE_VALUE_MISMATCH", "warning",
                f"报告引用破裂压力 {stated:.0f} bar，按 {depth:.0f} m / "
                f"{FRACTURE_RATIO}× 静水计算应为 {p_frac:.0f} bar，"
                f"偏差 {diff*100:.0f}% 已超出合理梯度区间",
            )

    # ─── 4. 报告引用的 CO2 密度是否与工具一致（warning）───
    if temperature is not None and p_inj:
        m = _DENSITY_RE.search(report)
        if m:
            expected = 1.9 * p_inj / (temperature + 273.15)
            stated_density = float(m.group(1))
            diff = abs(stated_density - expected) / expected
            check("density_consistent", diff <= DENSITY_TOLERANCE,
                  f"报告 {stated_density:.2f} vs 计算 {expected:.2f} kg/m³")
            if diff > DENSITY_TOLERANCE:
                violation(
                    "DENSITY_MISMATCH", "warning",
                    f"报告中的 CO2 密度 {stated_density:.2f} kg/m³ 与物理工具结果 "
                    f"{expected:.2f} kg/m³ 不符（偏差 {diff*100:.0f}%）",
                )

    blocking = any(v["severity"] == "blocking" for v in violations)
    if blocking:
        status = "violation"
    elif violations:
        status = "warning"
    else:
        status = "pass"

    feedback = _build_feedback(violations, p_inj, depth, p_h, p_frac, verdict)
    summary = _build_summary(status, p_inj, depth, p_h, p_frac, verdict)
    return _result(status, extracted, limits, violations, checks, summary,
                   feedback=feedback)


def _result(status, params, limits, violations, checks, summary, feedback=""):
    return {
        "status": status,
        "blocking": any(v["severity"] == "blocking" for v in violations),
        "verdict": _verdict_of(limits, params),
        "params": {k: (round(v, 2) if isinstance(v, float) else v)
                   for k, v in params.items()},
        "limits": limits,
        "violations": violations,
        "checks": checks,
        "feedback": feedback,
        "summary": summary,
    }


def _verdict_of(limits, params) -> str:
    p_inj = params.get("injection_pressure_bar")
    p_frac = limits.get("fracture_bar")
    p_h = limits.get("hydrostatic_bar")
    if p_inj is None or p_frac is None:
        return "unknown"
    if p_inj > p_frac:
        return "above_fracture"
    if p_inj < p_h:
        return "below_hydrostatic"
    return "safe_window"


def _build_summary(status, p_inj, depth, p_h, p_frac, verdict) -> str:
    if status == "unverifiable":
        return "参数不足，未做物理一致性校验"
    if verdict == "above_fracture":
        return (f"注入压力 {p_inj:.0f} bar > {depth:.0f} m 处破裂压力 "
                f"{p_frac:.0f} bar")
    if verdict == "below_hydrostatic":
        return (f"注入压力 {p_inj:.0f} bar < {depth:.0f} m 处地层压力 "
                f"{p_h:.0f} bar（注入困难）")
    return (f"注入压力 {p_inj:.0f} bar 落在 {depth:.0f} m 的安全注入窗口 "
            f"[{p_h:.0f}, {p_frac:.0f}] bar 内")


def _build_feedback(violations, p_inj, depth, p_h, p_frac, verdict) -> str:
    """
    把 blocking 违规翻译成"下一版该怎么改"的指令。

    只写"报告有物理错误"没有意义 —— 模型不知道错在哪、该往哪个方向改，
    下一版往往换个说法继续错。所以这里必须给出具体数值与可执行动作。
    """
    blocking = [v for v in violations if v["severity"] == "blocking"]
    if not blocking:
        return ""

    lines = ["【物理一致性校验未通过 —— 必须重写】"]
    for v in blocking:
        lines.append(f"- {v['message']}")

    if verdict == "above_fracture" and depth:
        # 给出两条可执行路径：降压力，或换更深的储层
        max_ok_pressure = p_frac
        min_ok_depth = p_inj / (RHO_WATER * G * BAR_PER_PA * FRACTURE_RATIO)
        lines.append(
            f"修正要求：明确写出注入压力已超过破裂压力 {p_frac:.0f} bar、"
            f"盖层存在压裂与 CO2 泄漏风险；并给出至少一条可执行措施 ——"
            f"将井底压力降至 ≤ {max_ok_pressure:.0f} bar，"
            f"或把储层深度加深至 ≥ {min_ok_depth:.0f} m（该深度下破裂压力 ≥ {p_inj:.0f} bar）。"
        )
    elif verdict == "below_hydrostatic" and depth:
        min_ok_pressure = p_h
        max_ok_depth = p_inj / (RHO_WATER * G * BAR_PER_PA)
        lines.append(
            f"修正要求：明确写出注入压力低于地层压力 {p_h:.0f} bar、需要更高井口压力"
            f"（或压裂改造）才能有效注入；不得使用「注入顺畅」「压力充足」一类表述。"
            f"可执行措施：把井底压力提高至 > {min_ok_pressure:.0f} bar，"
            f"或把储层深度调整到 < {max_ok_depth:.0f} m。"
        )
    lines.append("其余结论可保留，但凡是与上述数值冲突的表述必须一并改掉。")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════
# 结论 → 评审模型可读的事实块
# ══════════════════════════════════════════════════════════
# 各结论对应的"评审要求"。只覆盖**有实际数值结论**的三种状态。
_PROMPT_HINT = {
    "pass": "注入压力落在安全注入窗口内，报告只要不与之矛盾即可。",
    "warning": "存在数值层面的轻微不一致，扣分幅度不超过 1 分。",
    "violation": "报告与物理约束矛盾（已在规则层拦截）。",
}


def render_prompt_block(physics: dict) -> str:
    """
    把 ``evaluate_physics`` 的结论渲染成给评审模型看的事实块。

    **返回空串表示"本次没有可供模型参考的物理结论"**，调用方应据此整块省略，
    而不是塞一句措辞含糊的占位说明。

    为什么 ``unverifiable`` 必须返回空串（实测结论，不是设计偏好）：
    早期版本把它渲染成"结论：unverifiable / 说明：未提供储层深度…"写进 prompt，
    结果同一份报告从 7 分掉到 3 分，评语反复出现"与物理一致性校验结论存在冲突"。
    模型把"没算过"读成了"校验没通过"。事后在 prompt 里补一句"这不是报告的缺陷、
    不要扣分"也压不住 —— "结论 / 校验 / 违规"这些词本身就在暗示失败。
    结论：不需要模型判断的维度，就不要出现在它的输入里。
    """
    status = physics.get("status")
    if status not in _PROMPT_HINT:      # unverifiable / None / 未知状态
        return ""

    limits = physics.get("limits") or {}
    params = physics.get("params") or {}
    lines = [
        "【物理一致性校验（由 Python 规则引擎给出，不是你的判断）】",
        f"- 结论：{status} / {physics.get('verdict')}",
        f"- 说明：{physics.get('summary')}",
        f"- 评审要求：{_PROMPT_HINT[status]}",
    ]
    if limits:
        lines.append(
            f"- 数值：注入压力 {params.get('injection_pressure_bar')} bar，"
            f"储层深度 {limits.get('depth_m')} m，"
            f"地层压力 {limits.get('hydrostatic_bar')} bar，"
            f"破裂压力 {limits.get('fracture_bar')} bar"
        )
    for v in physics.get("violations") or []:
        lines.append(f"- 违规[{v['severity']}]：{v['message']}")
    if not physics.get("violations"):
        lines.append("- 未发现违规")
    return "\n".join(lines)
