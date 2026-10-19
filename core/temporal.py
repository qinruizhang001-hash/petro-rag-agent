"""
时间维度记忆（Temporal Knowledge）
==================================
CCUS 项目的数据是有"保质期"的：同一口井 2025 年实测地层压力 200 bar、
2026 年复测 300 bar，这不是两个"差不多"的数字，而是**一条演化轨迹**。
传统 RAG 把两段文本都塞进上下文，模型看到的是一团互相矛盾的证据，
最终往往取平均、或者只引用先出现的那一段 —— 两种都是错的。

本模块做三件事，全部是纯函数（无 IO、无模型、无第三方依赖）：

1. ``parse_timestamp`` —— 从文本里认出时间点（"2025年6月" / "2026-03-15"）。
2. ``extract_claims`` —— 把"某口井 + 某个参数 + 数值 + 时间"抽成结构化断言。
3. ``detect_conflicts`` / ``build_conflict_notice`` —— 找出同一对象在不同时间点
   的冲突数值，并生成一段**显式指令**注入上下文，告诉模型优先采用最新时间戳。

为什么冲突判定不能交给模型
--------------------------
"这两个数冲突吗"看起来是个语义问题，实际是个算术问题：同一口井的压力从
200 变到 300，变化 50%，超过测量不确定度，就是冲突；从 200 变到 203 不是。
把这件事交给 7B 模型判断，它会稳定地给出"存在一定差异，建议综合考虑"这种
既不错也无用的结论。所以判定用代码（同对象 + 同参数 + 相对偏差 > 阈值 +
时间戳不同），模型只负责在报告里**解释**这个已经被判定的冲突。

单位与量纲
----------
压力统一到 bar（MPa×10、kPa×0.01），温度统一到 °C，长度统一到 m，
注入速率统一到 万吨/年。归一化只做一次，之后所有比较都在同一量纲上，
否则 "2.5 MPa" 和 "25 bar" 会被误判成冲突。
"""
from __future__ import annotations

import re
from typing import Any, Iterable

# ─── 时间戳 ────────────────────────────────────────────────
# 必须带 "年" 或 "-" / "/" / "." 分隔符才算日期。
# 这一条限制是刻意的：裸四位数（"埋深 2000 m"）如果被当成 2000 年，
# 每份报告都会凭空多出一堆"跨世纪冲突"。
_TS_RE = re.compile(
    r"(?P<y>(?:19|20)\d{2})\s*(?:年|[-/.])\s*"
    r"(?:(?P<m>\d{1,2})\s*(?:月|[-/.])\s*"
    r"(?:(?P<d>\d{1,2})\s*日?)?)?"
)

# ─── 实体（井 / 区块 / 构造 / 气田）────────────────────────
# 中文实体：以"井/井位/区块/构造/气田/油田/平台/场地"结尾，
# 前面允许字母数字与连字符（"A-1井"、"长庆A-1井"）。
_ENTITY_RE = re.compile(
    r"(?:"
    r"(?P<zh>[A-Za-z0-9\u4e00-\u9fa5][A-Za-z0-9\u4e00-\u9fa5\-—·]{0,11}?\s*"
    r"(?:井位|井|区块|构造|气田|油田|平台|场地))"
    r"|(?P<en>(?:[Ww]ell|[Bb]lock|[Ff]ield)\s+[A-Za-z0-9\-]+)"
    r")"
)

# ─── 数值 + 单位 ───────────────────────────────────────────
_NUM_UNIT_RE = re.compile(
    r"(?P<num>-?\d+(?:\.\d+)?)\s*"
    r"(?P<unit>万吨/年|t/a|t/d|kg/m³|kg/m3|MPa|kPa|bars|bar|摄氏度|℃|°C|mD|%|米|m\b)"
)

# 单位 → (规范单位, 换算系数)
_UNIT_CANON: dict[str, tuple[str, float]] = {
    "bar": ("bar", 1.0),
    "bars": ("bar", 1.0),
    "MPa": ("bar", 10.0),
    "kPa": ("bar", 0.01),
    "℃": ("°C", 1.0),
    "°C": ("°C", 1.0),
    "摄氏度": ("°C", 1.0),
    "kg/m³": ("kg/m³", 1.0),
    "kg/m3": ("kg/m³", 1.0),
    "mD": ("mD", 1.0),
    "%": ("%", 1.0),
    "万吨/年": ("万吨/年", 1.0),
    "t/a": ("万吨/年", 1e-4),
    "t/d": ("万吨/年", 3.65e-3),
    "m": ("m", 1.0),
    "米": ("m", 1.0),
}

# 单位兜底推断参数类型（关键词优先，见 _metric_of）
_UNIT_METRIC: dict[str, tuple[str, str]] = {
    "bar": ("pressure", "地层压力"),
    "°C": ("temperature", "温度"),
    "kg/m³": ("density", "CO2 密度"),
    "mD": ("permeability", "渗透率"),
    "%": ("porosity", "孔隙度"),
    "万吨/年": ("rate", "注入速率"),
    "m": ("depth", "埋深"),
}

# 关键词 → 参数类型。顺序有意义：先匹配到的赢。
_METRIC_KEYWORDS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("pressure", "地层压力", ("破裂压力", "孔隙压力", "地层压力", "储层压力", "压力", "pressure")),
    ("temperature", "温度", ("温度", "temperature")),
    ("permeability", "渗透率", ("渗透率", "permeability")),
    ("porosity", "孔隙度", ("孔隙度", "porosity")),
    ("density", "密度", ("密度", "density")),
    ("rate", "注入速率", ("注入速率", "注入速度", "注入量", "injection rate")),
    ("depth", "埋深", ("埋深", "深度", "depth")),
)

# 这些词出现在数值附近时，"%" 不代表孔隙度（饱和度 / 采收率是另一个量）
_IGNORE_CONTEXT = ("饱和度", "采收率", "效率", "占比", "占比")

_SENTENCE_SPLIT = re.compile(r"[。；;\n\r]+")


def parse_timestamp(text: str) -> str | None:
    """
    从文本中取**第一个**时间点，返回 ``YYYY`` / ``YYYY-MM`` / ``YYYY-MM-DD``。

    没有时间信息返回 ``None`` —— 调用方据此决定"这个数值能不能进时间轴"。
    宁可漏判，也不要给一个 2000 m 的埋深安上 2000 年的日期。
    """
    if not text:
        return None
    m = _TS_RE.search(text)
    if not m:
        return None
    year = int(m.group("y"))
    if not (1900 <= year <= 2100):
        return None
    month = m.group("m")
    day = m.group("d")
    if month is None:
        return f"{year:04d}"
    month_i = int(month)
    if not (1 <= month_i <= 12):
        return f"{year:04d}"
    if day is None:
        return f"{year:04d}-{month_i:02d}"
    day_i = int(day)
    if not (1 <= day_i <= 31):
        return f"{year:04d}-{month_i:02d}"
    return f"{year:04d}-{month_i:02d}-{day_i:02d}"


def ts_sort_key(ts: str) -> tuple[int, int, int]:
    """时间戳 → 可比较元组。粒度不同（年 / 月 / 日）也能正确排序。"""
    parts = [int(p) for p in re.findall(r"\d+", ts or "")]
    while len(parts) < 3:
        parts.append(0)
    return parts[0], parts[1], parts[2]


def _months_between(old: str, new: str) -> int | None:
    """两个时间戳相差的月数；任一缺月份则返回 None（只报"跨年"）。"""
    o, n = ts_sort_key(old), ts_sort_key(new)
    if o[1] == 0 or n[1] == 0:
        return None
    return (n[0] - o[0]) * 12 + (n[1] - o[1])


def _span_text(old: str, new: str) -> str:
    months = _months_between(old, new)
    if months is None:
        years = ts_sort_key(new)[0] - ts_sort_key(old)[0]
        return f"{years} 年" if years else "同一年内"
    if months < 12:
        return f"{months} 个月"
    years, rem = divmod(months, 12)
    return f"{years} 年" + (f" {rem} 个月" if rem else "")


def _fmt_num(value: float) -> str:
    """数值展示：整数不带小数点，小数保留到有意义的一位。"""
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    if abs(value) >= 100:
        return f"{value:.1f}"
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _metric_of(gap: str, unit: str) -> tuple[str, str] | None:
    """
    判断这个数值属于哪个参数。

    ``gap`` 是**上一个数值之后、这个数值之前**的那段文字，不是整句。
    这个边界是必需的：在"地层压力 200 bar，温度 80℃"里，如果允许搜索整句，
    80℃ 会先撞上"压力"而被记成第二个压力值 —— 然后系统就会自信地报告
    "同一口井压力从 200 变成 80，变化 -60%"，一条完全虚构的冲突。
    用 gap 做边界，等于要求"关键词必须紧挨着它修饰的那个数"。

    关键词都不在 gap 里时，退回单位推断（bar→压力、℃→温度……）；
    ``%`` 这种歧义单位在没有关键词的情况下直接放弃，宁可不抽。
    """
    for key, label, kws in _METRIC_KEYWORDS:
        if any(kw in gap for kw in kws):
            return key, label
    if any(bad in gap for bad in _IGNORE_CONTEXT):
        return None
    return _UNIT_METRIC.get(unit)


def _nearest_entity(entities: list[tuple[int, str]], pos: int, max_dist: int = 60) -> str | None:
    """取离数值最近的实体名；太远就不认（避免把上一段的井名安到这一段）。"""
    if not entities:
        return None
    best = min(entities, key=lambda e: abs(e[0] - pos))
    if abs(best[0] - pos) > max_dist:
        return None
    return best[1]


def extract_claims(text: str) -> list[dict]:
    """
    把一段文本抽成结构化断言列表。

    每条断言形如::

        {"entity": "A-1井", "metric": "pressure", "metric_label": "地层压力",
         "value": 200.0, "unit": "bar", "timestamp": "2025-06", "raw": "..."}

    设计取舍：
    - **没有时间戳的数值直接丢弃。** 时间轴的每一个点都必须能说清"什么时候"，
      否则"200 bar"无法参与冲突判定，混进去只会制造假冲突。
    - **时间戳在句内可以继承。** "2025年6月，A-1井压力 200 bar、温度 80℃"
      后半句没有年份，但它显然属于同一次测量。
    - **实体同样可以继承**，但只在本段文本内。跨文档继承会把不同井的数据
      串到一起，那种错误在报告里看不出来。
    """
    if not text:
        return []

    claims: list[dict] = []
    carry_ts: str | None = None
    carry_entity: str | None = None

    for sentence in _SENTENCE_SPLIT.split(text):
        sentence = sentence.strip()
        if not sentence:
            continue

        own_ts = parse_timestamp(sentence)
        if own_ts:
            carry_ts = own_ts
        ts = own_ts or carry_ts

        entities = [(m.start(), m.group(0).strip()) for m in _ENTITY_RE.finditer(sentence)]
        if entities:
            carry_entity = entities[-1][1]

        if not ts:
            continue

        prev_end = 0
        for m in _NUM_UNIT_RE.finditer(sentence):
            raw_unit = m.group("unit")
            # 关键词的归属边界：上一个数值结束 → 这个数值开始
            gap = sentence[prev_end:m.start("num")]
            prev_end = m.end()

            canon = _UNIT_CANON.get(raw_unit)
            if canon is None:
                continue
            unit, factor = canon
            try:
                value = float(m.group("num")) * factor
            except ValueError:
                continue

            metric = _metric_of(gap, unit)
            if metric is None:
                continue
            key, label = metric

            entity = (
                _nearest_entity(entities, m.start("num"))
                or carry_entity
                or "未指定井位"
            )

            claims.append({
                "entity": entity,
                "metric": key,
                "metric_label": label,
                "value": round(value, 6),
                "unit": unit,
                "timestamp": ts,
                "raw": sentence[:160],
            })

    return claims


def _rel_diff(old: float, new: float) -> float:
    """相对偏差（取绝对值）。分母加 eps，防止 0 值导致除零。"""
    denom = max(abs(old), 1e-9)
    return abs(new - old) / denom


def detect_conflicts(
    claims: Iterable[dict],
    rel_tolerance: float = 0.05,
    max_conflicts: int = 5,
) -> list[dict]:
    """
    找出"同一对象 + 同一参数 + 不同时间 + 数值显著不同"的冲突。

    ``rel_tolerance`` 默认 5%：低于这个幅度视为测量噪声，不报警。
    报出来的每一条都带 ``older`` / ``newer`` 两端，下游据此生成
    "以最新为准"的显式指令。

    为什么只跟**最近的一个不同值**比，而不是全量两两比较：
    三口井十年数据两两比会产出几十条重复冲突，把上下文淹没；
    报告需要的是"最新值 vs 它之前最近的那个不同值"这一条主线。
    """
    groups: dict[tuple[str, str], list[dict]] = {}
    for c in claims:
        groups.setdefault((c["entity"], c["metric"]), []).append(c)

    conflicts: list[dict] = []
    for (entity, metric), items in groups.items():
        if len(items) < 2:
            continue

        ordered = sorted(items, key=lambda c: ts_sort_key(c["timestamp"]))
        newest = ordered[-1]
        older = None
        for cand in reversed(ordered[:-1]):
            # 同一时间点的两个值不算"时间冲突"（那是同一次测量的重复/矛盾）
            if ts_sort_key(cand["timestamp"]) >= ts_sort_key(newest["timestamp"]):
                continue
            if _rel_diff(cand["value"], newest["value"]) > rel_tolerance:
                older = cand
                break
        if older is None:
            continue

        delta_pct = (newest["value"] - older["value"]) / max(abs(older["value"]), 1e-9) * 100.0
        conflicts.append({
            "entity": entity,
            "metric": metric,
            "metric_label": newest["metric_label"],
            "older": {
                "value": older["value"],
                "unit": older["unit"],
                "timestamp": older["timestamp"],
                "display": f"{_fmt_num(older['value'])} {older['unit']}",
            },
            "newer": {
                "value": newest["value"],
                "unit": newest["unit"],
                "timestamp": newest["timestamp"],
                "display": f"{_fmt_num(newest['value'])} {newest['unit']}",
            },
            "delta_pct": round(delta_pct, 1),
            "span": _span_text(older["timestamp"], newest["timestamp"]),
            "message": (
                f"{entity} 的{newest['metric_label']}在 {older['timestamp']} 为 "
                f"{_fmt_num(older['value'])} {older['unit']}，到 {newest['timestamp']} "
                f"变为 {_fmt_num(newest['value'])} {newest['unit']}"
                f"（{delta_pct:+.1f}%，间隔 {_span_text(older['timestamp'], newest['timestamp'])}）"
            ),
        })

    # 最新的冲突排前面：上下文有长度预算，最该被看到的是最近发生的事
    conflicts.sort(key=lambda c: ts_sort_key(c["newer"]["timestamp"]), reverse=True)
    return conflicts[:max_conflicts]


def build_conflict_notice(conflicts: list[dict]) -> str:
    """
    生成注入上下文的"时间维度提示"块。

    措辞是刻意强硬的：``必须`` / ``不要`` / ``禁止``。
    本地 7B 模型对"建议优先参考"这类软约束基本无视，而这段文字存在的唯一目的
    就是改变它的取值行为 —— 写成客气话等于没写。
    """
    if not conflicts:
        return ""

    lines = [
        "## ⏱ 时间维度提示（系统自动检测，必须遵守）",
        "",
        "检索到的证据里，**同一个对象在不同时间点给出了互相冲突的数据**。",
        "处理规则：",
        "1. **必须优先采用时间戳最新的数据**作为当前结论的依据。",
        "2. 必须显式写出你采用的是哪个时间点的数据，以及它与旧数据的差异幅度。",
        "3. **禁止**对冲突数据取平均、取中值，或含糊表述为\"约\"、\"大致\"。",
        "4. 旧数据不是错的，它代表当时的测量结果；应在报告中说明差异的可能原因"
        "（压力回升、新钻井/新测量、测量方式变化、生产制度调整等），不要隐去。",
        "",
        "| 对象 | 参数 | 旧数据 | 最新数据 | 变化 | 间隔 |",
        "|---|---|---|---|---|---|",
    ]
    for c in conflicts:
        lines.append(
            f"| {c['entity']} | {c['metric_label']} | "
            f"{c['older']['display']}（{c['older']['timestamp']}） | "
            f"{c['newer']['display']}（{c['newer']['timestamp']}） | "
            f"{c['delta_pct']:+.1f}% | {c['span']} |"
        )

    lines += ["", "**本次结论一律以「最新数据」列为准。**", ""]
    return "\n".join(lines)


def analyze_temporal(docs: Iterable[str], rel_tolerance: float = 0.05) -> dict:
    """
    检索结果的统一入口：抽断言 → 找冲突 → 生成提示块。

    返回::

        {"claims": 12, "conflicts": [...], "notice": "...", "entities": [...]}

    空结果也返回完整结构（``notice`` 为空串），调用方无需判 None。
    """
    text = "\n".join(d for d in docs if d)
    claims = extract_claims(text)
    conflicts = detect_conflicts(claims, rel_tolerance=rel_tolerance)
    entities = sorted({c["entity"] for c in claims})
    return {
        "claims": len(claims),
        "conflicts": conflicts,
        "notice": build_conflict_notice(conflicts),
        "entities": entities[:8],
    }


def stamp_entity_timestamp(entity: dict[str, Any]) -> str | None:
    """
    给一个 LightRAG 实体补 ``timestamp`` 字段（原地不修改，返回识别结果）。

    实体本身通常不带时间，但描述里常常写着"2025 年测得…"。
    这里只做"有没有"的判断，不改写实体内容 —— LightRAG 的实体对象是共享的，
    往里面塞字段会污染图谱缓存。
    """
    if not entity:
        return None
    for key in ("timestamp", "created_at", "time", "date"):
        raw = entity.get(key)
        if isinstance(raw, str) and raw.strip():
            ts = parse_timestamp(raw) or raw.strip()
            if ts:
                return ts
    return parse_timestamp(str(entity.get("description") or ""))
