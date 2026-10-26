"""
工况问答 Agent
==============
"过去 6 小时井口压力怎么样" —— 这类问题看起来是自然语言，实际只有两个
需要理解的部分：**哪段时间**、**哪个测点**。其余全是算术。

所以本模块把"理解"和"算术"分开，并且都不交给模型：

- 时间窗口与测点指代用关键词表确定性解析。解析不出就**明确说解析不出**，
  按默认窗口回答并标注"未识别到时间范围，已按默认 1 小时"。
- 统计量由代码算，包括覆盖率、缺口、超阈次数、变化速率。
- 模型的唯一用途是把已经算好的事实组织成人话。它不能引入任何新数字 ——
  生成后会跑一遍守卫，把无法在事实里找到出处的数字标出来。

"现在"是数据的最新时刻，不是系统时钟
------------------------------------
这是本模块最重要的一条决定。离线分析、补录数据、历史复盘时，系统时钟
与数据时刻相差几天；若按系统时钟取"过去 6 小时"，窗口会完全落在数据之外，
回答于是变成"无数据" —— 而真正的原因是窗口取错了，不是数据缺失。

所以 ``now`` 默认取 ``max(r.ts)``，并在回答里写明用的是哪个时刻。
调用方要按真实时钟取窗口时必须显式传 ``now``。

覆盖率比均值重要
----------------
6 小时窗口里只有 40 分钟数据，此时"平均注入速率 180 t/d"这个数字本身没错，
但它会让人以为看到了 6 小时的工况。所以每个统计块都带 ``coverage``：
有数据的时间跨度 / 窗口长度，以及缺口清单。缺数据就明说，不插值 ——
插值出来的曲线看不出是插的，而它的每个点都会被当成实测引用。
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Sequence

from ops import alarms as alarms_mod
from ops.contracts import (
    ACTIONABLE,
    DIRECTION_HIGH,
    INDETERMINATE,
    QUALITY_BAD,
    SEVERITY_LABELS,
    Reading,
    SensorSpec,
)
from ops.thresholds import SENSORS, ThresholdRegistry

# ─── 时间窗口 ───
DEFAULT_WINDOW_HOURS = 1.0
MAX_WINDOW_HOURS = 24.0 * 31

_TIME_UNIT_HOURS: dict[str, float] = {
    "秒": 1 / 3600,
    "分钟": 1 / 60,
    "分": 1 / 60,
    "小时": 1.0,
    "钟头": 1.0,
    "天": 24.0,
    "日": 24.0,
    "周": 168.0,
}

#: 中文数字（只到 24，够日常表达；更大的量用阿拉伯数字写）
_CN_NUM = {
    "一": 1, "两": 2, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6,
    "七": 7, "八": 8, "九": 9, "十": 10, "半": 0.5,
}

_WINDOW_PATTERNS = (
    # 过去/最近/近 + 数字 + 单位
    re.compile(
        r"(?:过去|最近|近|前)\s*(?P<num>\d+(?:\.\d+)?|[一两二三四五六七八九十半]+)\s*"
        r"个?\s*(?P<unit>秒|分钟|分|小时|钟头|天|日|周)"
    ),
    # 最近一小时 / 过去半天
    re.compile(
        r"(?:过去|最近|近)\s*(?P<num>一|两|二|三|半)\s*(?P<unit>小时|天|日)"
    ),
)

_RELATIVE_PATTERNS = {
    "今天": 24.0,
    "今日": 24.0,
    "本班": 8.0,
    "当班": 8.0,
    "昨天": 48.0,
    "昨日": 48.0,
    "本周": 168.0,
    "这周": 168.0,
}


def _to_number(raw: str) -> float | None:
    raw = raw.strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        pass
    total = 0.0
    seen = False
    for ch in raw:
        v = _CN_NUM.get(ch)
        if v is None:
            return None
        total += v
        seen = True
    return total if seen else None


@dataclass(frozen=True)
class Window:
    hours: float
    label: str
    explicit: bool          # 是否从问句里解析出来（否则为默认值）
    note: str = ""


def parse_window(question: str) -> Window:
    """从问句里解析时间窗口。解析不出时返回默认窗口并标记 ``explicit=False``。"""
    text = question.strip()
    for pat in _WINDOW_PATTERNS:
        m = pat.search(text)
        if not m:
            continue
        num = _to_number(m.group("num"))
        unit = _TIME_UNIT_HOURS.get(m.group("unit"))
        if num is None or unit is None:
            continue
        hours = num * unit
        if hours <= 0:
            continue
        if hours > MAX_WINDOW_HOURS:
            return Window(
                hours=MAX_WINDOW_HOURS,
                label=f"{MAX_WINDOW_HOURS / 24:.0f} 天",
                explicit=True,
                note=f"问句中的窗口超过 {MAX_WINDOW_HOURS / 24:.0f} 天，已截断",
            )
        return Window(hours=hours, label=m.group(0).strip(), explicit=True)

    for word, hours in _RELATIVE_PATTERNS.items():
        if word in text:
            return Window(hours=hours, label=word, explicit=True)

    return Window(
        hours=DEFAULT_WINDOW_HOURS,
        label=f"最近 {DEFAULT_WINDOW_HOURS:g} 小时",
        explicit=False,
        note=(
            "问句里没有可识别的时间范围（如「过去 6 小时」「今天」），"
            f"已按默认 {DEFAULT_WINDOW_HOURS:g} 小时回答。"
            "如果这不是你想问的范围，请把时间段写进问题"
        ),
    )


# ─── 测点指代 ───
#: 顺序很重要：长词在前，否则"井口压"会先命中"井口"而漏掉完整说法。
#: "MVA" 指向**一组**测点而不是一个 —— MVA 是一套监测体系（深部/浅部/
#: 近地表/井筒），把它当成单一指标会让回答失去层位信息。
SENSOR_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("捕集率", ("capture_rate",)),
    ("再生能耗", ("regenerator_duty",)),
    ("溶剂损耗", ("solvent_loss",)),
    ("管段压差", ("pipeline_dp",)),
    ("压差", ("pipeline_dp",)),
    ("管道出口压力", ("pipeline_outlet_p",)),
    ("管压", ("pipeline_outlet_p",)),
    ("出口压力", ("pipeline_outlet_p",)),
    ("纯度", ("co2_purity",)),
    ("水露点", ("h2o_dewpoint",)),
    ("露点", ("h2o_dewpoint",)),
    ("井口压力", ("wellhead_p",)),
    ("井口压", ("wellhead_p",)),
    ("注入速率", ("injection_rate",)),
    ("注入量", ("injection_rate",)),
    ("注入指数", ("injectivity_index",)),
    ("环空压力", ("annulus_p",)),
    ("环空", ("annulus_p",)),
    ("储层压力", ("reservoir_p",)),
    ("土壤", ("soil_co2_flux",)),
    ("地下水", ("gw_ph", "gw_conductivity")),
    ("pH", ("gw_ph",)),
    ("电导率", ("gw_conductivity",)),
    ("形变", ("surface_heave",)),
    ("沉降", ("surface_heave",)),
    ("抬升", ("surface_heave",)),
    ("MVA", tuple(s.key for s in SENSORS.values() if s.stage == "mva")),
    ("mva", tuple(s.key for s in SENSORS.values() if s.stage == "mva")),
    ("监测", tuple(s.key for s in SENSORS.values() if s.stage == "mva")),
    ("运输", tuple(s.key for s in SENSORS.values() if s.stage == "transport")),
    ("管道", tuple(s.key for s in SENSORS.values() if s.stage == "transport")),
    ("注入", tuple(s.key for s in SENSORS.values() if s.stage == "injection")),
    ("捕集", tuple(s.key for s in SENSORS.values() if s.stage == "capture")),
    ("全部", tuple(s.key for s in SENSORS.values())),
    ("所有", tuple(s.key for s in SENSORS.values())),
    ("全线", tuple(s.key for s in SENSORS.values())),
)


@dataclass(frozen=True)
class Resolved:
    sensors: list[str]
    matched: list[str]
    unresolved: bool
    note: str = ""


def resolve_sensors(question: str) -> Resolved:
    """解析问句指向哪些测点。解析不出时**不猜**，返回空并说明。"""
    hits: list[str] = []
    matched: list[str] = []
    for alias, keys in SENSOR_ALIASES:
        if alias in question:
            matched.append(alias)
            for k in keys:
                if k not in hits:
                    hits.append(k)
    if not hits:
        return Resolved(
            sensors=[],
            matched=[],
            unresolved=True,
            note=(
                "问句里没有识别到测点名称。可用说法见 ops/condition.py 的 "
                "SENSOR_ALIASES（如「井口压力」「管压」「捕集率」「MVA」）。"
                "不猜具体测点：猜错会给出一个数值正确、对象错误的回答"
            ),
        )
    return Resolved(sensors=hits, matched=matched, unresolved=False)


# ─── 统计 ───
@dataclass(frozen=True)
class SeriesStats:
    sensor: str
    label: str
    unit: str
    n: int
    start: datetime
    end: datetime
    first: float
    last: float
    min_v: float
    max_v: float
    mean: float
    p95: float
    p5: float
    slope_per_hour: float
    coverage: float
    gaps: list[dict]
    over_threshold: int
    threshold_source: str
    direction: str

    def focus_value(self) -> float:
        """朝"坏"方向的分位值：越高越坏看 p95，越低越坏看 p5。"""
        return self.p95 if self.direction == DIRECTION_HIGH else self.p5


def _percentile(sorted_vals: Sequence[float], p: float) -> float:
    """nearest-rank 百分位。

    刻意写明算法：不同库（numpy 默认线性插值、Excel PERCENTILE.INC/EXC）
    在同一组数据上会给出不同的 p95，而报告里的"95 分位 118.3 bar"如果没有
    算法说明就无法复核。nearest-rank 只取实测点，永远不会给出一个
    从未出现过的值 —— 对"这口井实际到过多高"这类问题这是必需的。
    """
    if not sorted_vals:
        raise ValueError("空序列没有百分位")
    n = len(sorted_vals)
    rank = max(1, math.ceil(p / 100.0 * n))
    return sorted_vals[min(rank, n) - 1]


def _median_gap(series: Sequence[Reading]) -> float | None:
    gaps = [
        (b.ts - a.ts).total_seconds()
        for a, b in zip(series, series[1:])
    ]
    gaps = [g for g in gaps if g > 0]
    if not gaps:
        return None
    gaps.sort()
    mid = len(gaps) // 2
    return gaps[mid] if len(gaps) % 2 else (gaps[mid - 1] + gaps[mid]) / 2


def compute_stats(
    spec: SensorSpec,
    series: Sequence[Reading],
    window: Window,
    registry: ThresholdRegistry,
    *,
    now: datetime,
) -> SeriesStats:
    values = [r.value for r in series if r.quality != QUALITY_BAD]
    if not values:
        raise ValueError(f"{spec.key}: 窗口内没有可信读数")

    ordered = sorted(values)
    span_h = (series[-1].ts - series[0].ts).total_seconds() / 3600.0
    slope = 0.0
    if span_h > 0:
        slope = (series[-1].value - series[0].value) / span_h

    # 缺口：相邻间隔超过中位间隔 3 倍视为一段缺口
    gaps: list[dict] = []
    med = _median_gap(series)
    if med and med > 0:
        for a, b in zip(series, series[1:]):
            g = (b.ts - a.ts).total_seconds()
            if g > med * 3:
                gaps.append({
                    "from": a.ts.isoformat(),
                    "to": b.ts.isoformat(),
                    "missing_seconds": round(g - med),
                })

    threshold = registry.get(spec.key)
    over = 0
    if threshold is not None:
        warn_pairs = [v for name, v in threshold.levels() if name == "warn"]
        if warn_pairs:
            w = warn_pairs[0]
            for v in values:
                if (v >= w) if spec.direction == DIRECTION_HIGH else (v <= w):
                    over += 1

    return SeriesStats(
        sensor=spec.key,
        label=spec.label,
        unit=spec.unit,
        n=len(values),
        start=series[0].ts,
        end=series[-1].ts,
        first=values[0],
        last=values[-1],
        min_v=ordered[0],
        max_v=ordered[-1],
        mean=sum(values) / len(values),
        p95=_percentile(ordered, 95),
        p5=_percentile(ordered, 5),
        slope_per_hour=slope,
        coverage=min(1.0, span_h / window.hours) if window.hours > 0 else 0.0,
        gaps=gaps,
        over_threshold=over,
        threshold_source=threshold.source if threshold else "未配置",
        direction=spec.direction,
    )


# ─── 回答 ───
@dataclass
class ConditionAnswer:
    question: str
    window: Window
    now: datetime
    sensors: list[str]
    stats: list[SeriesStats] = field(default_factory=list)
    facts: list[str] = field(default_factory=list)
    text: str = ""
    #: 问句解析中未能确定的部分（必须让用户看到，不能静默用默认值顶替）
    unresolved: list[str] = field(default_factory=list)
    #: 回答里无法在 facts 中找到出处的数字（守卫结果，应为空）
    unsourced_numbers: list[str] = field(default_factory=list)
    #: 该窗口内的告警摘要（intent=anomaly 时填充）
    alarm_summary: str = ""

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "window": {
                "hours": self.window.hours,
                "label": self.window.label,
                "explicit": self.window.explicit,
                "note": self.window.note,
            },
            "now": self.now.isoformat(),
            "sensors": self.sensors,
            "stats": [
                {
                    "sensor": s.sensor,
                    "label": s.label,
                    "unit": s.unit,
                    "n": s.n,
                    "start": s.start.isoformat(),
                    "end": s.end.isoformat(),
                    "min": s.min_v,
                    "max": s.max_v,
                    "mean": s.mean,
                    "p95": s.p95,
                    "p5": s.p5,
                    "last": s.last,
                    "slope_per_hour": s.slope_per_hour,
                    "coverage": round(s.coverage, 4),
                    "gaps": s.gaps,
                    "over_threshold": s.over_threshold,
                    "threshold_source": s.threshold_source,
                }
                for s in self.stats
            ],
            "facts": list(self.facts),
            "text": self.text,
            "unresolved": list(self.unresolved),
            "unsourced_numbers": list(self.unsourced_numbers),
            "alarm_summary": self.alarm_summary,
        }


_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def find_unsourced_numbers(text: str, facts: Sequence[str]) -> list[str]:
    """找出回答里无法在事实块中找到的数字。

    用途是守卫模型生成的措辞。数字比对按"字符串出现在某条事实里"判断，
    而不是数值相等 —— 因为 180 和 180.0 在文本里是两种写法，
    但后者通常意味着模型自己重算过。宁可多报一个可疑项，也不放过一个
    凭记忆写下的数字。
    """
    haystack = "\n".join(facts)
    out: list[str] = []
    for m in _NUM_RE.finditer(text):
        token = m.group(0)
        # 形如 2026-10-26T08:00 的时间戳会拆出 2026/10/26/08/00，逐个比对即可
        if token in haystack:
            continue
        if token.lstrip("-") in haystack:
            continue
        out.append(token)
    return out


def _render_template(answer: ConditionAnswer) -> str:
    """无模型时的模板回答。所有数字都直接来自 stats，因此天然可溯源。"""
    lines: list[str] = []
    w = answer.window
    lines.append(
        f"时间范围：{w.label}（以数据最新时刻 {answer.now:%Y-%m-%d %H:%M} 为基准，"
        f"向前 {w.hours:g} 小时）"
    )
    if w.note:
        lines.append(f"注意：{w.note}")
    if not answer.stats:
        lines.append("窗口内没有可用的可信读数，无法给出结论。")
        return "\n".join(lines)

    for s in answer.stats:
        unit = f" {s.unit}" if s.unit and s.unit != "-" else ""
        lines.append("")
        lines.append(f"【{s.label}】")
        lines.append(
            f"- 采样 {s.n} 点，覆盖 {s.coverage:.0%} 的窗口"
            f"（{s.start:%H:%M}–{s.end:%H:%M}）"
        )
        lines.append(
            f"- 范围 {s.min_v:g}–{s.max_v:g}{unit}，均值 {s.mean:g}{unit}，"
            f"末值 {s.last:g}{unit}"
        )
        focus = "95 分位" if s.direction == DIRECTION_HIGH else "5 分位"
        lines.append(
            f"- {focus} {s.focus_value():g}{unit}（nearest-rank，取实测点，不插值）"
        )
        trend = "上升" if s.slope_per_hour > 0 else ("下降" if s.slope_per_hour < 0 else "持平")
        lines.append(f"- 窗口内{trend} {abs(s.slope_per_hour):.3g}{unit}/h（首末点线性）")
        if s.over_threshold:
            lines.append(
                f"- 有 {s.over_threshold} 个采样点越过 warn 阈值"
                f"（阈值出处：{s.threshold_source}）"
            )
        else:
            lines.append(f"- 无采样点越过 warn 阈值（阈值出处：{s.threshold_source}）")
        if s.gaps:
            total_missing = sum(g["missing_seconds"] for g in s.gaps)
            lines.append(
                f"- 数据缺口 {len(s.gaps)} 段，合计约 {total_missing / 60:.0f} 分钟；"
                "缺口内没有数据，本回答未做插值"
            )
    if answer.alarm_summary:
        lines.append("")
        lines.append(f"【同期告警】{answer.alarm_summary}")
    return "\n".join(lines)


_ANOMALY_WORDS = ("异常", "告警", "报警", "超限", "超阈", "有问题", "不正常")
_TREND_WORDS = ("趋势", "走势", "变化", "上升", "下降", "波动")


def detect_intent(question: str) -> str:
    if any(w in question for w in _ANOMALY_WORDS):
        return "anomaly"
    if any(w in question for w in _TREND_WORDS):
        return "trend"
    return "snapshot"


def answer(
    question: str,
    readings: Sequence[Reading],
    registry: ThresholdRegistry,
    *,
    now: datetime | None = None,
    window: Window | None = None,
    llm=None,
) -> ConditionAnswer:
    """回答一个工况问题。

    ``llm`` 为 ``None`` 时用模板；给了可调用对象则用它组织语言，并对输出
    跑数字守卫。**默认不接模型**，因为模板已经能回答，而模型只会增加
    一类失败（措辞里出现没有出处的数字）。
    """
    win = window or parse_window(question)
    resolved = resolve_sensors(question)
    intent = detect_intent(question)

    if not readings:
        return ConditionAnswer(
            question=question,
            window=win,
            now=now or datetime.now(),
            sensors=[],
            unresolved=["没有任何读数传入，无法回答"],
            text="没有收到任何时序数据，无法回答该问题。",
        )

    # "现在"默认取数据最新时刻，理由见模块文档
    anchor = now or max(r.ts for r in readings)
    start = anchor - timedelta(hours=win.hours)

    answer_obj = ConditionAnswer(
        question=question,
        window=win,
        now=anchor,
        sensors=resolved.sensors,
    )
    if win.note:
        answer_obj.unresolved.append(win.note)
    if resolved.unresolved:
        answer_obj.unresolved.append(resolved.note)
        answer_obj.text = (
            "无法确定你问的是哪个测点，因此没有给出数值。\n" + resolved.note
        )
        return answer_obj

    for key in resolved.sensors:
        spec = SENSORS[key]
        series = sorted(
            (r for r in readings if r.sensor == key and start <= r.ts <= anchor),
            key=lambda r: r.ts,
        )
        if not series:
            continue
        try:
            answer_obj.stats.append(
                compute_stats(spec, series, win, registry, now=anchor)
            )
        except ValueError:
            continue

    # 事实块：模型只能引用这里出现过的数字
    facts: list[str] = [
        f"基准时刻 {anchor.isoformat()}，窗口 {win.hours:g} 小时，起点 {start.isoformat()}"
    ]
    for s in answer_obj.stats:
        unit = f" {s.unit}" if s.unit and s.unit != "-" else ""
        facts.append(
            f"{s.label}：采样 {s.n} 点，覆盖 {s.coverage:.0%}；"
            f"范围 {s.min_v:g}–{s.max_v:g}{unit}；均值 {s.mean:g}{unit}；"
            f"末值 {s.last:g}{unit}；95 分位 {s.p95:g}{unit}；5 分位 {s.p5:g}{unit}；"
            f"斜率 {s.slope_per_hour:+.3g}{unit}/h；"
            f"越 warn 阈值 {s.over_threshold} 点（出处：{s.threshold_source}）"
        )
        for g in s.gaps:
            facts.append(
                f"{s.label} 数据缺口：{g['from']} 至 {g['to']}，"
                f"缺约 {g['missing_seconds']} 秒"
            )

    if intent == "anomaly":
        report = alarms_mod.evaluate(readings, registry)
        answer_obj.alarm_summary = report.summary_line()
        facts.append(f"同期告警摘要：{report.summary_line()}")
        for a in report.actionable:
            facts.append(
                f"告警 {a.alarm_id}：{a.label} {a.value:g}，等级 "
                f"{SEVERITY_LABELS.get(a.severity, a.severity)}"
            )

    answer_obj.facts = facts

    if not answer_obj.stats:
        answer_obj.text = (
            f"窗口（{start:%Y-%m-%d %H:%M} 至 {anchor:%Y-%m-%d %H:%M}）内"
            "没有可信读数，无法给出结论。"
            "这不等于工况正常 —— 没有数据与数据正常是两件事。"
        )
        return answer_obj

    template = _render_template(answer_obj)
    if llm is None:
        answer_obj.text = template
        return answer_obj

    prompt = (
        "你是 CCUS 运维助手。只能使用下面「事实」中出现的数字，"
        "不得引入任何新数字、不得推测未列出的测点。\n"
        "如果事实不足以回答问题，直接说明缺什么。\n\n"
        f"问题：{question}\n\n事实：\n" + "\n".join(f"- {f}" for f in facts)
    )
    try:
        generated = llm(prompt)
    except Exception as exc:  # noqa: BLE001
        # 模型失败不能让回答消失：模板是完整答案，不是降级占位
        answer_obj.text = template
        answer_obj.unresolved.append(f"模型组织措辞失败，已回退模板：{exc}")
        return answer_obj

    text = str(generated).strip() or template
    answer_obj.unsourced_numbers = find_unsourced_numbers(text, facts)
    if answer_obj.unsourced_numbers:
        # 有无法溯源的数字就整篇回退模板：逐句删数字会留下语法破碎的句子，
        # 而"看起来通顺但数字没出处"比"格式朴素但每个数字都能回代"更危险
        answer_obj.text = template
        answer_obj.unresolved.append(
            "模型输出含无法在事实中找到的数字（"
            + "、".join(sorted(set(answer_obj.unsourced_numbers)))
            + "），已整篇回退模板回答"
        )
    else:
        answer_obj.text = text
    return answer_obj
