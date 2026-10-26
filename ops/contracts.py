"""
运维层数据契约
==============
四个 Agent 之间只通过这里定义的对象通信。定义集中在一个纯 stdlib 模块里的
理由是实际踩过的：阈值、单位、方向一旦分散在各 Agent 里，就会出现
"告警 Agent 认为 80 bar 是高压、报告 Agent 认为 80 bar 正常"这种分歧，
而两边各自都自洽、各自都有测试。分歧只在汇总时暴露，那时已经很难定位。

为什么 ``direction`` 必须显式
-----------------------------
"超阈值"不是一个方向。"捕集率 65%" 比 "90%" 更糟，而 "管压 120 bar" 比
"80 bar" 更糟。若统一按 "> threshold" 判定，捕集率下降永远不会告警 ——
而且不会报错，只是那条规则安静地不生效。所以每个传感器必须声明
``DIRECTION_HIGH``（越高越坏）或 ``DIRECTION_LOW``（越低越坏），
没有"两个方向都查"的选项：双向告警需要两组阈值，那是两个传感器的语义。

为什么 ``quality`` 是必填的第三态
--------------------------------
仪表有三类状态：读数可信、读数可疑、读数不可信。把后两类折进 ``normal``
是这类系统最常见的静默失败：传感器断线 → 无读数 → 无告警 → 看板全绿。
所以 ``Reading.quality`` 参与判定：``bad`` 的读数**不判定**，直接产出
``indeterminate``，且 ``indeterminate`` 在汇总里按"未知"计数，不按"正常"计数。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

# ─── 链条环节：顺序即物料流向，supervisor 的冲突检测依赖这个顺序 ───
STAGES: tuple[str, ...] = ("capture", "transport", "injection", "mva")

STAGE_LABELS: dict[str, str] = {
    "capture": "捕集",
    "transport": "运输",
    "injection": "注入",
    "mva": "监测核查与核算（MVA）",
}

# ─── 告警等级（从低到高）───
SEVERITIES: tuple[str, ...] = ("normal", "notice", "warn", "alert", "critical")

#: 数据不可用 / 阈值缺失时的等级。**不是** ``normal``，也排在 ``critical`` 之后
#: 只是为了便于"按严重度排序时把未知放在最前"，真正的排序在 alarms 里做。
INDETERMINATE = "indeterminate"

SEVERITY_LABELS: dict[str, str] = {
    "normal": "正常",
    "notice": "关注",
    "warn": "警告",
    "alert": "告警",
    "critical": "严重",
    INDETERMINATE: "不可判定",
}

#: 需要人介入的等级（报告里必须单独成段，不能混在"运行正常"里）
ACTIONABLE: frozenset[str] = frozenset({"warn", "alert", "critical"})

# ─── 方向 ───
DIRECTION_HIGH = "high"   # 越高越坏：压力、通量、pH 偏移量
DIRECTION_LOW = "low"     # 越低越坏：捕集率、注入速率、泵效
DIRECTIONS: tuple[str, ...] = (DIRECTION_HIGH, DIRECTION_LOW)

# ─── 读数质量 ───
QUALITY_GOOD = "good"
QUALITY_SUSPECT = "suspect"   # 可判定但要降级说明（例如校验值漂移）
QUALITY_BAD = "bad"           # 不判定
QUALITIES: tuple[str, ...] = (QUALITY_GOOD, QUALITY_SUSPECT, QUALITY_BAD)

# ─── 归因强度：刻意不用数字 ───
# 用 0.8 这种数会立刻被当成概率引用，而它既不是频率也不是主观概率，
# 只是一个规则命中强度的别名。用等级就没有被误读的空间。
CONFIDENCE_STRONG = "strong"     # 规则要求的读数全部命中，且无矛盾读数
CONFIDENCE_PARTIAL = "partial"   # 命中但存在可解释的矛盾读数
CONFIDENCE_WEAK = "weak"         # 仅单条读数支持
CONFIDENCES: tuple[str, ...] = (CONFIDENCE_STRONG, CONFIDENCE_PARTIAL, CONFIDENCE_WEAK)


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


@dataclass(frozen=True)
class SensorSpec:
    """一个测点的静态描述。

    ``basis`` 不是注释，是判读依据：同一个名字在不同测量基准下结论不同。
    例：井口压力用"表压"还是"绝对压力"差一个大气压，在 100 bar 量级上无所谓，
    但在判断"是否接近临界压力 73.8 bar"时足以改变结论；MVA 的土壤 CO₂ 通量
    用"箱式法瞬时值"还是"累积通量"，其阈值根本不是一个数量级。
    """

    key: str
    label: str
    unit: str
    stage: str
    direction: str
    basis: str

    def __post_init__(self) -> None:
        _require(bool(self.key.strip()), "SensorSpec.key 不能为空")
        _require(bool(self.label.strip()), f"{self.key}: label 不能为空")
        _require(bool(self.unit.strip()), f"{self.key}: unit 不能为空")
        _require(
            self.stage in STAGES,
            f"{self.key}: stage={self.stage!r} 不在 {STAGES} 内。"
            "链条环节是枚举而不是自由字符串，否则 supervisor 的上下游推断会静默失效",
        )
        _require(
            self.direction in DIRECTIONS,
            f"{self.key}: direction={self.direction!r} 必须是 {DIRECTIONS} 之一。"
            "方向缺失会让'越低越坏'的传感器永远不告警，而且不会报错",
        )
        _require(
            bool(self.basis.strip()),
            f"{self.key}: basis 不能为空。读数没有测量基准就无法判读，"
            "也无法解释为什么阈值是这个数",
        )


@dataclass(frozen=True)
class Threshold:
    """一组分级阈值。

    ``source`` 是必填的，且必须描述**这个数字从哪来**。这不是形式要求：
    CCUS 项目的压力限值通常来自三处完全不同的地方 —— 管道设计规范
    （如 DNV-RP-F104）、项目设计文件、运营方的风险接受准则。三者混为一谈时，
    报告里说"超过标准限值"和"超过我们自己设的线"是完全不同的两件事，
    而读者无法从数字本身分辨。

    允许写 "未配置" 之外的一切，但**不允许留空**：留空的阈值会被当作
    "有出处但说不出出处"，比明确写"这是我们的政策选择"更糟。
    """

    warn: float | None = None
    alert: float | None = None
    critical: float | None = None
    source: str = ""
    note: str = ""

    def __post_init__(self) -> None:
        _require(
            bool(self.source.strip()),
            "Threshold.source 不能为空：没有出处的阈值不允许参与判定。"
            "如果这确实是运营方的政策选择，就明确写'项目政策选择（非标准规定）'",
        )
        vals = [v for v in (self.warn, self.alert, self.critical) if v is not None]
        _require(len(vals) > 0, f"阈值 {self.source!r} 三级全为空，等于没有阈值")
        _require(
            len(set(vals)) == len(vals),
            f"阈值 {self.source!r} 存在重复值 {vals}。重复的级别等于少一级，"
            "而告警只会少报、不会报错",
        )

    def validate_for(self, sensor: SensorSpec) -> None:
        """检查分级方向是否与传感器的"坏"方向一致。

        这一步放在这里而不是 ``__post_init__``：``Threshold`` 不知道自己
        属于哪个传感器，而单调性只有结合方向才有意义。
        """
        pairs = [
            (name, getattr(self, name))
            for name in ("warn", "alert", "critical")
            if getattr(self, name) is not None
        ]
        if len(pairs) < 2:
            return
        values = [v for _n, v in pairs]
        ascending = all(a < b for a, b in zip(values, values[1:]))
        descending = all(a > b for a, b in zip(values, values[1:]))
        if sensor.direction == DIRECTION_HIGH:
            _require(
                ascending,
                f"{sensor.key}: direction=high（越高越坏），阈值必须递增 "
                f"warn<alert<critical，实际为 {pairs}。顺序反了会让高级别永远不可达",
            )
        else:
            _require(
                descending,
                f"{sensor.key}: direction=low（越低越坏），阈值必须递减 "
                f"warn>alert>critical，实际为 {pairs}。顺序反了会让高级别永远不可达",
            )

    def is_configured(self) -> bool:
        return any(v is not None for v in (self.warn, self.alert, self.critical))

    def levels(self) -> list[tuple[str, float]]:
        """按严重度从低到高返回已配置的 (等级名, 阈值)。"""
        out: list[tuple[str, float]] = []
        for name in ("warn", "alert", "critical"):
            v = getattr(self, name)
            if v is not None:
                out.append((name, float(v)))
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "warn": self.warn,
            "alert": self.alert,
            "critical": self.critical,
            "source": self.source,
            "note": self.note,
        }


@dataclass(frozen=True)
class Reading:
    """一条测点读数。"""

    sensor: str
    value: float
    ts: datetime
    quality: str = QUALITY_GOOD

    def __post_init__(self) -> None:
        _require(bool(self.sensor.strip()), "Reading.sensor 不能为空")
        _require(
            self.quality in QUALITIES,
            f"Reading.quality={self.quality!r} 必须是 {QUALITIES} 之一",
        )
        # NaN 会静默通过所有比较（NaN > x 恒为 False），必须在这里挡住
        _require(
            self.value == self.value and self.value not in (float("inf"), float("-inf")),
            f"{self.sensor}: 读数为 {self.value!r}，NaN/Inf 无法比较大小，"
            "静默放行会让它看起来像'没超阈值'",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "sensor": self.sensor,
            "value": self.value,
            "ts": self.ts.isoformat(),
            "quality": self.quality,
        }


@dataclass
class Attribution:
    """一条归因候选。

    ``ruled_out`` 为真时仍然保留在列表里，并给出 ``ruled_out_reason`` ——
    这是刻意的：读者需要看到"我们考虑过这个原因、因为某条读数排除了它"，
    而不是只看到剩下的一条结论然后怀疑是不是漏了什么。
    """

    cause: str
    stage: str
    confidence: str
    evidence: list[str] = field(default_factory=list)
    ruled_out: bool = False
    ruled_out_reason: str = ""
    #: 结构化的判据：``[{"sensor": ..., "state": ...}, ...]``。
    #: 与 ``evidence``（给人看的措辞）并存，是因为 supervisor 要核对
    #: "这条归因引用的测点现在还是不是那个状态"。从自然语言里解析状态
    #: 只能靠正则，而正则一改就静默失配 —— 那会让核对变成永远通过。
    basis: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        _require(bool(self.cause.strip()), "Attribution.cause 不能为空")
        _require(
            self.confidence in CONFIDENCES,
            f"confidence={self.confidence!r} 必须是 {CONFIDENCES} 之一",
        )
        _require(
            self.ruled_out or bool(self.evidence),
            f"归因 {self.cause!r} 未被排除却没有证据。"
            "没有证据的归因与猜测无法区分，运维照着它处置只会引入新故障",
        )
        if self.ruled_out:
            _require(
                bool(self.ruled_out_reason.strip()),
                f"归因 {self.cause!r} 被标记排除，但没有写排除理由",
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "cause": self.cause,
            "stage": self.stage,
            "confidence": self.confidence,
            "evidence": list(self.evidence),
            "ruled_out": self.ruled_out,
            "ruled_out_reason": self.ruled_out_reason,
            "basis": [dict(b) for b in self.basis],
        }


@dataclass
class Alarm:
    """一条告警。

    ``rationale`` 是一个**字符串列表**而不是一个分数，因为运维要问的是
    "为什么是这一级"，而不是"严重度 0.73 是什么"。每条字符串对应一条
    实际命中的判定规则。
    """

    alarm_id: str
    sensor: str
    label: str
    stage: str
    severity: str
    direction: str
    value: float | None
    ts: datetime
    rationale: list[str] = field(default_factory=list)
    attribution: list[Attribution] = field(default_factory=list)
    actions: list[dict[str, Any]] = field(default_factory=list)
    threshold_level: str = ""
    threshold_value: float | None = None
    provenance: str = ""
    quality: str = QUALITY_GOOD

    def __post_init__(self) -> None:
        _require(bool(self.alarm_id.strip()), "Alarm.alarm_id 不能为空")
        _require(
            self.severity in SEVERITIES or self.severity == INDETERMINATE,
            f"severity={self.severity!r} 不在 {SEVERITIES + (INDETERMINATE,)} 内",
        )
        _require(
            bool(self.rationale),
            f"{self.alarm_id}: rationale 不能为空。"
            "每条告警都要能回答'为什么是这一级'，否则分级就是不可复核的",
        )

    @property
    def actionable(self) -> bool:
        return self.severity in ACTIONABLE

    def to_dict(self) -> dict[str, Any]:
        return {
            "alarm_id": self.alarm_id,
            "sensor": self.sensor,
            "label": self.label,
            "stage": self.stage,
            "severity": self.severity,
            "severity_label": SEVERITY_LABELS.get(self.severity, self.severity),
            "direction": self.direction,
            "value": self.value,
            "ts": self.ts.isoformat(),
            "rationale": list(self.rationale),
            "attribution": [a.to_dict() for a in self.attribution],
            "actions": list(self.actions),
            "threshold_level": self.threshold_level,
            "threshold_value": self.threshold_value,
            "provenance": self.provenance,
            "quality": self.quality,
        }


@dataclass(frozen=True)
class Citation:
    """报告里一个数字的出处。

    ``kind`` 有三种，区别在于**读者能怎么核对**：

    - ``alarm``：指向一条告警记录，读者可以查到它的 value / 阈值 / 判定规则
    - ``reading``：指向一条原始读数，读者可以回到时序里看这个点
    - ``summary``：指向一个可复算的聚合量（如"告警 3 条"）。
      它不能逐条回代，但必须写明**算法**，读者可以用同样的输入重算出同一个数。
      这是它和"来源：系统"的区别 —— 后者无法复算。

    不允许"来源：系统"这种无法回代的写法，那是把责任推给一个不存在的实体。
    """

    kind: str
    ref: str
    #: summary 类必须给出算法说明；alarm/reading 类可留空
    detail: str = ""

    def __post_init__(self) -> None:
        _require(
            self.kind in ("alarm", "reading", "summary"),
            f"Citation.kind={self.kind!r} 必须是 'alarm' / 'reading' / 'summary'",
        )
        _require(bool(self.ref.strip()), "Citation.ref 不能为空")
        if self.kind == "summary":
            _require(
                bool(self.detail.strip()),
                f"summary 引用 {self.ref!r} 没有算法说明。"
                "聚合量不能逐条回代，唯一能核对的方式是复算，因此算法必须写明",
            )

    def render(self) -> str:
        return f"[{self.kind}:{self.ref}]"
