"""
告警 / 异常 Agent
=================
输入一批测点读数，输出分级告警、归因候选、处置建议。**全程不调用语言模型**。

为什么不用模型
--------------
这个 Agent 的输出会直接驱动"要不要停输"和"要不要上报监管"。这类决定必须
满足两个条件：同样的输入永远得到同样的输出；每一条结论都能指回一条规则。
语言模型两条都不满足 —— 它今天说 warn、明天说 alert，而且说不出是哪条
判断让它这么说的。

模型的位置在**报告 Agent**：把已经定好的结论组织成人话。判定本身必须是
确定性的，这与本项目在质量闸门上的一贯做法一致（先有确定性闸门，再谈措辞）。

分级不是一个数字，是四条规则的叠加
----------------------------------
只看"超了多少"会把两种完全不同的情况判成同一级：
读数刚好越过 warn 线一次，与读数连续六小时稳定停在 alert 线以上。
后者才需要连夜处理。所以分级 = 基础等级 + 三条升级规则：

===============  ==========================================================
规则              作用
===============  ==========================================================
``R_THRESHOLD``  按阈值给出基础等级
``R_PERSIST``    连续多个采样点停在同级 → 升一级（稳态越限比瞬时越限危险）
``R_RAPID``      变化速率足以在一小时内跨过一个等级间隔 → 升一级
                 （等到真的越限时已经没有处置窗口了）
``R_SUSPECT``    读数可疑 → 等级封顶在 warn（可疑的读数不该触发停输）
===============  ==========================================================

每条规则命中都会写进 ``Alarm.rationale``。运维问"为什么是 alert"时，
答案是那几行字，不是一个 0.73。

"不可判定"是第四种结果，不是第五级
----------------------------------
数据缺失、仪表不可信、阈值未配置 —— 这三种情况的正确输出是
``indeterminate``：既不是 normal（那会让看板变绿），也不是 critical
（那会触发一次没有依据的应急）。它必须单独计数，并在报告里单独成段，
因为它要求的是**不同的人做不同的事**：查仪表、补阈值，而不是处置工况。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable, Sequence

from ops import chain
from ops.contracts import (
    ACTIONABLE,
    CONFIDENCE_PARTIAL,
    CONFIDENCE_STRONG,
    CONFIDENCE_WEAK,
    DIRECTION_HIGH,
    INDETERMINATE,
    QUALITY_BAD,
    QUALITY_SUSPECT,
    SEVERITIES,
    SEVERITY_LABELS,
    Alarm,
    Attribution,
    Reading,
    SensorSpec,
    Threshold,
)
from ops.thresholds import SENSORS, ThresholdRegistry

# ─── 等级排序（用于升级与比较）───
SEVERITY_RANK: dict[str, int] = {name: i for i, name in enumerate(SEVERITIES)}

# ─── 政策参数（都有出处，且都可通过参数覆盖）───
#: 早期提示线：距 warn 阈值不足 warn→alert 间距的多少比例时进入 notice。
#: 出处：项目政策选择，**无标准依据**。选 10% 的理由是它只用于"下个班次
#: 复核"这一级动作，误报成本接近零；若把它当作告警线，任何比例都缺乏依据。
#:
#: 宽度以 warn→alert 的**间距**为基准，而不是 warn 值的百分比。后者对
#: 有物理上限的测点会失效：`co2_purity` warn=95、上限 100，按 10% 比例算
#: 提示线是 104.5，比量程还高，于是每个合法读数都成了 notice。
DEFAULT_NOTICE_MARGIN = 0.10

#: 连续多少个采样点停在同级视为"稳态越限"。
#: 出处：项目政策选择。3 点对应常见 5 分钟采样的 15 分钟，足以排除单点毛刺，
#: 又不至于让 1 小时内的真实趋势漏判。
DEFAULT_PERSIST_POINTS = 3

#: 变化速率阈值：一小时内跨过多少个"等级间隔"就升级。
#: 出处：项目政策选择。1.0 表示"照这个速度一小时后就会掉进下一级"。
DEFAULT_RAPID_INTERVALS_PER_HOUR = 1.0

#: 政策说明（进报告，让读者知道这些线是怎么来的）
POLICY_PROVENANCE = (
    "分级政策（非标准规定）：notice = 距 warn 阈值不足 warn→alert 间距的 10%；"
    "稳态越限 = 连续 3 个采样点；速率升级 = 1 小时内跨过 1 个等级间隔。"
    "阈值本身另有出处，见每条告警的 provenance 字段"
)


def _rank(severity: str) -> int:
    """``indeterminate`` 不参与排序，调用方必须先处理掉它。"""
    return SEVERITY_RANK[severity]


def _upgrade(severity: str, steps: int = 1) -> str:
    i = min(_rank(severity) + steps, len(SEVERITIES) - 1)
    return SEVERITIES[i]


def _downgrade_cap(severity: str, cap: str) -> str:
    return severity if _rank(severity) <= _rank(cap) else cap


# ============================================================
# 一、单点分类
# ============================================================
@dataclass(frozen=True)
class Classification:
    """单个读数相对于阈值的分类结果。"""

    level: str                    # warn / alert / critical / notice / normal
    threshold_level: str          # 命中的阈值名（notice/normal 时为空）
    threshold_value: float | None
    distance_ratio: float         # 相对阈值的位置，0 = 正好在 warn 上
    rationale: list[str]


def classify(
    spec: SensorSpec,
    value: float,
    threshold: Threshold | None,
    notice_margin: float = DEFAULT_NOTICE_MARGIN,
) -> Classification:
    """把单个读数分类。阈值缺失时返回 ``indeterminate`` 的等价物（level 为 None 由调用方处理）。

    这里刻意**不**处理 quality 与阈值缺失 —— 那是 ``evaluate`` 的职责，
    因为它们不是"这个值有多坏"的问题，而是"这个值能不能用"的问题。
    把两类问题混在一个函数里，会让"不可判定"悄悄变成一个很高的等级或很低。
    """
    if threshold is None:
        raise ValueError(
            f"{spec.key}: classify 需要阈值。未配置的情况由 evaluate 产出 "
            "indeterminate，不要在这里兜底"
        )

    levels = threshold.levels()
    if not levels:
        raise ValueError(f"{spec.key}: 阈值 {threshold.source!r} 三级全为空")

    # 沿严重度从高到低找第一个越过的级别
    hit_name: str | None = None
    hit_value: float | None = None
    for name, tv in reversed(levels):
        crossed = value >= tv if spec.direction == DIRECTION_HIGH else value <= tv
        if crossed:
            hit_name, hit_value = name, tv
            break

    if hit_name is not None:
        rationale = [
            f"R_THRESHOLD：{spec.label} = {value:g} {spec.unit}，"
            f"{'≥' if spec.direction == DIRECTION_HIGH else '≤'} "
            f"{hit_name} 阈值 {hit_value:g}（出处：{threshold.source}）"
        ]
        return Classification(
            level=hit_name,
            threshold_level=hit_name,
            threshold_value=hit_value,
            distance_ratio=0.0,
            rationale=rationale,
        )

    # 未越限：判断是否进入 notice 区
    warn_pairs = [p for p in levels if p[0] == "warn"]
    if warn_pairs:
        warn_value = warn_pairs[0][1]
        # 提示带宽度取 **warn → alert 的间距**，不是 warn 值的百分比。
        #
        # 为什么不能按阈值比例算：`co2_purity` 的 warn = 95，而它的物理上限
        # 就是 100 %。按 10 % 比例算，提示线落在 104.5 —— 比量程上限还高，
        # 于是**每一个合法读数都是 notice**。一条常年亮着的提示等于没有提示，
        # 而且它会教人忽略这一级，之后真正的 notice 也不会被看见。
        #
        # warn → alert 的间距是这个数据里真实存在的一把尺子：它由配置决定、
        # 随项目缩放、与量纲无关，而且三级必须互不相同（见 contracts 的校验），
        # 所以间距恒大于 0。压力类测点也因此得到一个按设计余量缩放的提示带，
        # 而不是一个按绝对值缩放的。
        alert_pairs = [p for p in levels if p[0] == "alert"]
        span = abs(warn_value - alert_pairs[0][1]) if alert_pairs else abs(warn_value)
        if span <= 0:
            # 理论上不可达（三级必须互不相同）。真到了这里就退化成"无提示带"，
            # 因为一个宽度为零的提示带会把**所有**未越限读数都算成 notice。
            span = abs(warn_value) or 1.0
        band = span * notice_margin
        distance = abs(value - warn_value)
        if spec.direction == DIRECTION_HIGH:
            notice_line = warn_value - band
            near = value >= notice_line
        else:
            notice_line = warn_value + band
            near = value <= notice_line
        if near:
            return Classification(
                level="notice",
                threshold_level="notice",
                threshold_value=notice_line,
                distance_ratio=distance / span,
                rationale=[
                    f"R_THRESHOLD：{spec.label} = {value:g} {spec.unit} 未越 warn 阈值 "
                    f"{warn_value:g}，但已进入早期提示带"
                    f"（提示线 {notice_line:g}，宽度 = warn→alert 间距的 "
                    f"{notice_margin:.0%}）。提示带宽度是项目政策选择，无标准依据"
                ],
            )
        return Classification(
            level="normal",
            threshold_level="",
            threshold_value=None,
            distance_ratio=distance / span,
            rationale=[
                f"R_THRESHOLD：{spec.label} = {value:g} {spec.unit} 距 warn 阈值 "
                f"{warn_value:g} 尚有 {distance:g} {spec.unit}"
                f"（超过提示带宽度 {band:g}）"
            ],
        )

    return Classification(
        level="normal",
        threshold_level="",
        threshold_value=None,
        distance_ratio=1.0,
        rationale=[
            f"R_THRESHOLD：{spec.label} = {value:g} {spec.unit} 未越任何已配置阈值"
        ],
    )


# ============================================================
# 二、时序辅助
# ============================================================
def _series(readings: Iterable[Reading], sensor_key: str) -> list[Reading]:
    out = [r for r in readings if r.sensor == sensor_key]
    out.sort(key=lambda r: r.ts)
    return out


def _rate_per_hour(series: Sequence[Reading]) -> float | None:
    """窗口首尾的变化速率。少于两点或时间跨度为 0 时返回 None。

    刻意不用最小二乘：两点足够说明"照这个速度会怎样"，而拟合需要
    至少三点且对毛刺敏感。趋势分析是 condition Agent 的事，
    这里的用途只是"要不要提前升级"。
    """
    if len(series) < 2:
        return None
    span = (series[-1].ts - series[0].ts).total_seconds() / 3600.0
    if span <= 0:
        return None
    return (series[-1].value - series[0].value) / span


def _persist_count(
    series: Sequence[Reading],
    spec: SensorSpec,
    threshold: Threshold,
    level_name: str,
) -> int:
    """从最新一点往回数，连续有多少点达到 ``level_name``（或更高）。

    只数**同一个等级或更高**，不数"越过 warn 但未到 alert" —— 后者会把
    "在 warn 线上抖动"和"稳定停在 warn 上"算成同一件事。
    """
    levels = threshold.levels()
    want = [tv for name, tv in levels if _rank(name) >= _rank(level_name)]
    if not want:
        return 0
    # 取"达到 level_name 所需的最低门槛"。越高越坏时，门槛是这一档里**最小**
    # 的那个数（>= alert 门槛 100 即已进入 alert 及以上，不必到 critical 120）；
    # 越低越坏时阈值递减，门槛是这一档里**最大**的那个数。
    gate = min(want) if spec.direction == DIRECTION_HIGH else max(want)
    count = 0
    for r in reversed(series):
        if r.quality == QUALITY_BAD:
            break
        crossed = r.value >= gate if spec.direction == DIRECTION_HIGH else r.value <= gate
        if not crossed:
            break
        count += 1
    return count


def _interval(threshold: Threshold) -> float | None:
    """相邻两个等级阈值之间的最小间隔，用于把速率换算成"等级间隔/小时"。"""
    levels = threshold.levels()
    if len(levels) < 2:
        return None
    gaps = [abs(b[1] - a[1]) for a, b in zip(levels, levels[1:])]
    gaps = [g for g in gaps if g > 0]
    return min(gaps) if gaps else None


# ============================================================
# 三、单测点评估
# ============================================================
def _evaluate_sensor(
    spec: SensorSpec,
    series: Sequence[Reading],
    threshold: Threshold | None,
    *,
    notice_margin: float,
    persist_points: int,
    rapid_intervals: float,
) -> Alarm:
    latest = series[-1]
    alarm_id = f"A-{latest.ts.strftime('%Y%m%dT%H%M')}-{spec.key}"

    # ── 不可判定的三种来源，优先于一切分级 ──
    if latest.quality == QUALITY_BAD:
        return Alarm(
            alarm_id=alarm_id,
            sensor=spec.key,
            label=spec.label,
            stage=spec.stage,
            severity=INDETERMINATE,
            direction=spec.direction,
            value=latest.value,
            ts=latest.ts,
            rationale=[
                f"R_QUALITY：最新读数 quality=bad（{latest.ts.isoformat()}）。"
                "不可信读数不参与分级 —— 判 normal 会让盲区看起来正常，"
                "判 critical 会触发一次没有依据的应急"
            ],
            actions=chain.actions_for(["verify_instrument"]),
            provenance="不适用（未使用阈值）",
            quality=latest.quality,
        )

    if threshold is None:
        return Alarm(
            alarm_id=alarm_id,
            sensor=spec.key,
            label=spec.label,
            stage=spec.stage,
            severity=INDETERMINATE,
            direction=spec.direction,
            value=latest.value,
            ts=latest.ts,
            rationale=[
                f"R_NO_THRESHOLD：{spec.label} 没有配置阈值。"
                "没有出处的阈值不参与判定，因此本条不可判定，"
                "不得计为正常"
            ],
            actions=chain.actions_for(["configure_threshold"]),
            provenance="未配置",
            quality=latest.quality,
        )

    cls = classify(spec, latest.value, threshold, notice_margin)
    severity = cls.level
    rationale = list(cls.rationale)

    # ── R_PERSIST：稳态越限 ──
    if severity in ACTIONABLE:
        n = _persist_count(series, spec, threshold, cls.threshold_level)
        if n >= persist_points:
            before = severity
            severity = _upgrade(severity, 1)
            if severity != before:
                rationale.append(
                    f"R_PERSIST：连续 {n} 个采样点达到 {cls.threshold_level} 及以上"
                    f"（门槛 {persist_points} 点）→ 由 {before} 升为 {severity}。"
                    "稳态越限比单点越限危险：前者说明工况已经改变，不是毛刺"
                )
            else:
                rationale.append(
                    f"R_PERSIST：连续 {n} 个采样点达到 {cls.threshold_level} 及以上，"
                    f"但 {before} 已是最高等级，不再升级"
                )

    # ── R_RAPID：速率足以在一小时内跨级 ──
    if severity in ACTIONABLE or severity == "notice":
        rate = _rate_per_hour(series)
        gap = _interval(threshold)
        if rate is not None and gap:
            intervals_per_hour = abs(rate) / gap
            # 只有"朝坏的方向"变化才算提前升级。direction=high 表示数值越大
            # 越坏，所以 rate > 0 才是恶化；direction=low 反过来
            worsening = rate > 0 if spec.direction == DIRECTION_HIGH else rate < 0
            if worsening and intervals_per_hour >= rapid_intervals:
                before = severity
                severity = _upgrade(severity, 1)
                if severity != before:
                    rationale.append(
                        f"R_RAPID：窗口内变化 {rate:+.3g} {spec.unit}/h，"
                        f"相当于 {intervals_per_hour:.2f} 个等级间隔/小时"
                        f"（门槛 {rapid_intervals:g}）→ 由 {before} 升为 {severity}。"
                        "照此速度一小时内会进入下一级，等到越限再处置就没有窗口了"
                    )

    # ── R_SUSPECT：读数可疑时封顶 ──
    if latest.quality == QUALITY_SUSPECT and _rank(severity) > _rank("warn"):
        before = severity
        severity = "warn"
        rationale.append(
            f"R_SUSPECT：读数 quality=suspect（校验值漂移/超期未标定）→ "
            f"由 {before} 封顶为 warn。可疑读数不应触发停输或应急，"
            "但也不应因为'不确定'而当作正常"
        )

    actions = chain.severity_actions(severity)
    # 归因给出的原因级动作更具体，追加在后面而不是替换：
    # "立即响应"与"做关井测试"是两件事，前者是纪律，后者是方法
    if severity in ACTIONABLE:
        actions = actions + chain.actions_for(
            [k for rule in chain.CAUSE_RULES if rule.stage == spec.stage
             for k in rule.actions[:1]]
        )[:1]
    return Alarm(
        alarm_id=alarm_id,
        sensor=spec.key,
        label=spec.label,
        stage=spec.stage,
        severity=severity,
        direction=spec.direction,
        value=latest.value,
        ts=latest.ts,
        rationale=rationale,
        actions=actions,
        threshold_level=cls.threshold_level,
        threshold_value=cls.threshold_value,
        provenance=threshold.source,
        quality=latest.quality,
    )


# ============================================================
# 四、全链归因
# ============================================================
def _state_of(alarm: Alarm) -> str:
    """把告警等级折叠成规则表使用的三态。"""
    if alarm.severity == INDETERMINATE:
        return chain.STATE_UNKNOWN
    if alarm.severity in ACTIONABLE or alarm.severity == "notice":
        return chain.STATE_DEVIATED
    return chain.STATE_OK


def attribute(
    alarms: Sequence[Alarm],
    rules: Sequence[chain.CauseRule] = chain.CAUSE_RULES,
) -> list[Attribution]:
    """按规则表给出归因候选（含被排除的）。

    匹配语义：
    - ``requires`` 全部命中 → strong
    - 命中数 ≥ ``min_hits`` 且 ≥2 → partial（缺的读数在证据里注明）
    - ``excludes`` 任一命中 → ruled_out（即使 requires 也命中）
    - 一条都没命中 → 不进入结果

    ``excludes`` 优先于 ``requires`` 是刻意的：一条原因同时"有证据支持"
    和"有证据排除"时，正确的做法是不给出这条原因，并说明为什么排除 ——
    而不是按命中数投票。
    """
    states = {a.sensor: _state_of(a) for a in alarms}
    labels = {a.sensor: a.label for a in alarms}
    out: list[Attribution] = []

    for rule in rules:
        hits: list[str] = []
        missing: list[str] = []
        for sensor_key, want in rule.requires:
            got = states.get(sensor_key, chain.STATE_UNKNOWN)
            if got == chain.STATE_UNKNOWN:
                missing.append(labels.get(sensor_key, sensor_key))
            elif got == want:
                hits.append(
                    f"{labels.get(sensor_key, sensor_key)} "
                    f"{chain.STATE_LABELS.get(want, want)}"
                )

        excluded_by: str = ""
        for (sensor_key, want), reason in zip(rule.excludes, rule.exclude_reasons):
            if states.get(sensor_key) == want:
                excluded_by = (
                    f"{labels.get(sensor_key, sensor_key)} "
                    f"{chain.STATE_LABELS.get(want, want)}：{reason}"
                )
                break

        if excluded_by:
            out.append(
                Attribution(
                    cause=rule.cause,
                    stage=rule.stage,
                    confidence=CONFIDENCE_WEAK,
                    evidence=list(hits),
                    ruled_out=True,
                    ruled_out_reason=excluded_by,
                    basis=[
                        {"sensor": k, "expected": want, "observed": states.get(k)}
                        for k, want in rule.requires
                    ],
                )
            )
            continue

        if not hits:
            continue

        evidence = list(hits)
        if missing:
            evidence.append(
                "未核对（数据缺失/不可判定）：" + "、".join(sorted(set(missing)))
            )

        if len(hits) == len(rule.requires):
            conf = CONFIDENCE_STRONG
        elif len(hits) >= max(rule.min_hits, 2):
            conf = CONFIDENCE_PARTIAL
            evidence.append(
                f"仅 {len(hits)}/{len(rule.requires)} 条判据命中，"
                "结论强度相应降低"
            )
        else:
            continue

        evidence.append(f"判据：{rule.rationale}")
        if rule.caveat:
            evidence.append(f"不可据此排除：{rule.caveat}")

        out.append(
            Attribution(
                cause=rule.cause,
                stage=rule.stage,
                confidence=conf,
                evidence=evidence,
                basis=[
                    {"sensor": k, "expected": want, "observed": states.get(k)}
                    for k, want in rule.requires
                ],
            )
        )

    # 排序：未被排除的在前，强度高的在前。稳定排序保证同样输入同样输出
    order = {CONFIDENCE_STRONG: 0, CONFIDENCE_PARTIAL: 1, CONFIDENCE_WEAK: 2}
    out.sort(key=lambda a: (a.ruled_out, order[a.confidence], a.cause))
    return out


# ============================================================
# 五、对外入口
# ============================================================
@dataclass
class AlarmReport:
    alarms: list[Alarm] = field(default_factory=list)
    attribution: list[Attribution] = field(default_factory=list)
    policy: str = POLICY_PROVENANCE
    threshold_origin: str = ""
    #: 已声明但没配阈值的测点 —— 已知盲区，报告必须列出
    unconfigured: list[str] = field(default_factory=list)
    #: 完全没收到读数的已声明测点
    no_data: list[str] = field(default_factory=list)

    @property
    def actionable(self) -> list[Alarm]:
        return [a for a in self.alarms if a.severity in ACTIONABLE]

    @property
    def indeterminate(self) -> list[Alarm]:
        return [a for a in self.alarms if a.severity == INDETERMINATE]

    def by_severity(self) -> dict[str, int]:
        out = {name: 0 for name in SEVERITIES}
        out[INDETERMINATE] = 0
        for a in self.alarms:
            out[a.severity] = out.get(a.severity, 0) + 1
        return out

    def top_severity(self) -> str:
        """最高等级。**没有任何告警不等于正常** —— 无数据时返回 indeterminate。"""
        if not self.alarms:
            return INDETERMINATE
        ranks = [_rank(a.severity) for a in self.alarms if a.severity != INDETERMINATE]
        if not ranks:
            return INDETERMINATE
        return SEVERITIES[max(ranks)]

    def summary_line(self) -> str:
        counts = self.by_severity()
        parts = [
            f"{SEVERITY_LABELS[k]} {v}"
            for k, v in counts.items()
            if v
        ]
        head = f"最高等级：{SEVERITY_LABELS.get(self.top_severity(), self.top_severity)}"
        if self.unconfigured:
            head += f"｜已知盲区 {len(self.unconfigured)} 个测点无阈值"
        if self.no_data:
            head += f"｜{len(self.no_data)} 个测点无读数"
        return head + ("｜" + "、".join(parts) if parts else "")

    def to_dict(self) -> dict:
        return {
            "summary": self.summary_line(),
            "top_severity": self.top_severity(),
            "counts": self.by_severity(),
            "policy": self.policy,
            "threshold_origin": self.threshold_origin,
            "unconfigured": list(self.unconfigured),
            "no_data": list(self.no_data),
            "alarms": [a.to_dict() for a in self.alarms],
            "attribution": [a.to_dict() for a in self.attribution],
        }


def evaluate(
    readings: Sequence[Reading],
    registry: ThresholdRegistry,
    *,
    notice_margin: float = DEFAULT_NOTICE_MARGIN,
    persist_points: int = DEFAULT_PERSIST_POINTS,
    rapid_intervals: float = DEFAULT_RAPID_INTERVALS_PER_HOUR,
    sensors: Sequence[SensorSpec] | None = None,
) -> AlarmReport:
    """评估一批读数。

    只对**收到了读数**的测点产出告警；对已声明但没读数的测点，记入
    ``no_data`` 而不是静默跳过 —— 后者会让"这个测点根本没接上"
    与"这个测点一直正常"在看板上长得一模一样。
    """
    if persist_points < 1:
        raise ValueError(
            f"persist_points={persist_points} 必须 ≥1。"
            "0 会让 R_PERSIST 永远成立，等于对所有告警无差别升级"
        )
    if not 0.0 <= notice_margin < 1.0:
        raise ValueError(
            f"notice_margin={notice_margin} 必须在 [0, 1) 内。"
            "≥1 会让提示带覆盖整个量程，所有读数都变成 notice"
        )

    specs = list(sensors if sensors is not None else SENSORS.values())
    seen = {r.sensor for r in readings}
    unknown = seen - {s.key for s in specs}
    if unknown:
        raise ValueError(
            f"读数里出现未声明的测点：{sorted(unknown)}。"
            "未声明的测点没有量纲与方向，无法判定；"
            "静默忽略会让它的异常完全不进入任何报告"
        )

    alarms: list[Alarm] = []
    no_data: list[str] = []
    for spec in specs:
        series = _series(readings, spec.key)
        if not series:
            no_data.append(spec.key)
            continue
        alarms.append(
            _evaluate_sensor(
                spec,
                series,
                registry.get(spec.key),
                notice_margin=notice_margin,
                persist_points=persist_points,
                rapid_intervals=rapid_intervals,
            )
        )

    report = AlarmReport(
        alarms=alarms,
        threshold_origin=registry.origin,
        unconfigured=registry.missing(),
        no_data=sorted(no_data),
    )
    report.attribution = attribute(alarms)
    # 把归因挂回单条告警：报告里"这条告警指向什么"要能就近读到，
    # 而不是让读者自己去全链列表里找
    for alarm in report.alarms:
        alarm.attribution = [
            a for a in report.attribution
            if a.stage == alarm.stage and not a.ruled_out
        ]
    return report


def evaluate_rows(
    rows: Sequence[dict],
    registry: ThresholdRegistry,
    **kwargs,
) -> AlarmReport:
    """从 ``{sensor, value, ts, quality}`` 字典列表评估（便于接 API / CSV）。"""
    readings = [
        Reading(
            sensor=str(r["sensor"]),
            value=float(r["value"]),
            ts=r["ts"] if isinstance(r["ts"], datetime) else datetime.fromisoformat(str(r["ts"])),
            quality=str(r.get("quality", "good")),
        )
        for r in rows
    ]
    return evaluate(readings, registry, **kwargs)
