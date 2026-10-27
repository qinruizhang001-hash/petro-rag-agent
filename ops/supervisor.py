"""
全链条编排 Agent（Supervisor）
==============================
捕集 → 运输 → 注入 → MVA 的任务拆分、调用、核对、汇总。

Supervisor 不做什么
-------------------
**它不算阈值，也不判定等级。** 这两件事分别属于 ``alarms`` 和阈值注册表。
Supervisor 只做三件别处做不了的事：

1. **拆分**：根据请求决定跑哪些子 Agent，并记录为什么跑、跑没跑成。
2. **核对**：子 Agent 的结论放在一起看是否互相矛盾。单个 Agent 各自自洽
   而合起来矛盾，是这类系统最典型的失效模式 —— 每个 Agent 的测试都绿，
   只有把它们并排放在一页上才看得出来。
3. **汇总**：把结果拼成一份可交付的东西，并把矛盾显式写在最前面。

如果 Supervisor 自己也开始判阈值，它就成了第五个有判定权的 Agent，
而它恰恰是唯一一个看不到原始时序的角色。权限与信息必须匹配。

四类冲突，处理方式各不相同
--------------------------
=======================  ==============================================  ============
类型                      含义                                            处理
=======================  ==============================================  ============
``contradiction``         同一测点在同一时刻的两个数值不一致                阻断
``direction``             一个 Agent 说越限、另一个说未越限                  阻断
``orphan_attribution``    归因引用的测点现在已不是那个状态                    阻断
``pressure_ceiling``      井底压力超过破裂压力的 90%（法规上限）             阻断
``coupling_deviation``    上下游变化方向与耦合表相反                        提示
=======================  ==============================================  ============

前四种里，前三种是**错误**：同一事实被两个 Agent 描述成两样，必须查清是谁错了。
第四种 ``pressure_ceiling`` 不是两个 Agent 之间的矛盾，而是**单一事实违反法规**，
所以它单独成一类：它的处置不是"查清谁错了"，是"降量并核实破裂压力来源"。
最后一种 ``coupling_deviation`` 连错误都不是 —— 耦合表说的是"通常如何"，
反向变化往往意味着存在另一个驱动源，那是有价值的信息。把最后一种也阻断，
会让系统在真实工况变化时拒绝出报告，而那正是最需要报告的时候。

所以 ``blocking`` 是 ``Conflict`` 的一个字段，不是靠类型名猜。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Sequence

from ops import alarms as alarms_mod
from ops import ceiling as ceiling_mod
from ops import chain
from ops import condition as condition_mod
from ops import reports as reports_mod
from ops.chain import STAGES
from ops.contracts import (
    ACTIONABLE,
    INDETERMINATE,
    STAGE_LABELS,
    Alarm,
    Reading,
)
from ops.thresholds import SENSORS, ThresholdRegistry

# ─── 冲突类型 ───
CONFLICT_CONTRADICTION = "contradiction"
CONFLICT_DIRECTION = "direction"
CONFLICT_ORPHAN = "orphan_attribution"
CONFLICT_COUPLING = "coupling_deviation"
CONFLICT_CEILING = "pressure_ceiling"

#: 阻断性冲突：出现即不允许把报告提交审批
BLOCKING_KINDS: frozenset[str] = frozenset({
    CONFLICT_CONTRADICTION,
    CONFLICT_DIRECTION,
    CONFLICT_ORPHAN,
    CONFLICT_CEILING,
})

#: 数值比对容差。同一份读数经两条路径处理后应当逐位相同，
#: 所以容差取得极小；它只用来吸收浮点表示差异，不用来"容忍不一致"
_REL_TOL = 1e-9


@dataclass
class Conflict:
    kind: str
    summary: str
    detail: list[str] = field(default_factory=list)
    blocking: bool = True
    #: 建议的下一步（对 coupling_deviation 尤其重要：它需要的是解释而非修错）
    next_step: str = ""

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "summary": self.summary,
            "detail": list(self.detail),
            "blocking": self.blocking,
            "next_step": self.next_step,
        }

    def render(self) -> str:
        head = "【阻断】" if self.blocking else "【提示】"
        out = [f"{head}{self.summary}"]
        out += [f"  - {d}" for d in self.detail]
        if self.next_step:
            out.append(f"  → {self.next_step}")
        return "\n".join(out)


# ============================================================
# 一、任务拆分
# ============================================================
@dataclass(frozen=True)
class Task:
    agent: str
    reason: str


@dataclass
class TaskRun:
    agent: str
    reason: str
    ok: bool
    elapsed_s: float
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "agent": self.agent,
            "reason": self.reason,
            "ok": self.ok,
            "elapsed_s": round(self.elapsed_s, 3),
            "error": self.error,
        }


@dataclass
class OpsRequest:
    """一次运维请求。

    ``want`` 显式列出需要哪些产物。不按"问句里有没有提到"来推断，
    因为"帮我看看井口压力"和"出个日报"的差别不在措辞，在交付物。
    """

    readings: Sequence[Reading]
    period: tuple[datetime, datetime]
    question: str = ""
    want_alarms: bool = True
    want_condition: bool = False
    want_report: bool = False
    report_kind: str = "daily"
    permit_id: str = ""
    #: 周报用的分期区间。日报/监管材料只看 ``period``，不看这里。
    #: 为什么单独一个字段而不是让 ``period`` 变复杂：周报的语义是"多期的分布"，
    #: 而 ``period`` 的语义是"这次看哪一段时间"。把两种语义塞进一个字段，
    #: 调用方永远搞不清该传哪个。
    periods: Sequence[tuple[datetime, datetime]] = ()
    #: 工况问答的"现在"。默认由 ``condition.answer`` 取数据最新时刻；
    #: 离线复盘时必须显式给，否则"最近 1 小时"会以三天后的系统时钟为基准，
    #: 取到一个空窗口并渲染成"没有异常"。
    now: datetime | None = None

    def tasks(self) -> list[Task]:
        """拆分。顺序即依赖顺序：告警是其余两者的基础。"""
        out: list[Task] = []
        if self.want_alarms or self.want_condition or self.want_report:
            out.append(
                Task("alarms", "所有其他产物的判定基础：先确定每个测点处于什么等级")
            )
        if self.want_condition:
            out.append(
                Task("condition", "需要对时序给出带引用的数值回答，由问句解析窗口与测点")
            )
        if self.want_report:
            kind_label = {
                "daily": "日报", "weekly": "周报", "regulatory": "监管材料"
            }.get(self.report_kind, self.report_kind)
            out.append(Task("reports", f"需要生成{kind_label}草稿并走人工审批"))
        return out


# ============================================================
# 二、冲突检测
# ============================================================
def _state_map(alarms: Sequence[Alarm]) -> dict[str, str]:
    out: dict[str, str] = {}
    for a in alarms:
        if a.severity == INDETERMINATE:
            out[a.sensor] = chain.STATE_UNKNOWN
        elif a.severity in ACTIONABLE or a.severity == "notice":
            out[a.sensor] = chain.STATE_DEVIATED
        else:
            out[a.sensor] = chain.STATE_OK
    return out


def detect_value_conflicts(
    alarm_report: alarms_mod.AlarmReport,
    cond: condition_mod.ConditionAnswer,
) -> list[Conflict]:
    """核对"同一条告警的数值"与"工况问答里同一测点的末值"是否一致。

    两者理论上来自同一份读数，但窗口不同：告警取最新点，工况问答取窗口内
    最后一点。若窗口没有覆盖到最新点，两个数字就会不同 —— 这不是 bug，
    而是"报告里的数字取决于窗口选择"这一事实，必须让人看见。
    """
    out: list[Conflict] = []
    by_sensor = {a.sensor: a for a in alarm_report.alarms}
    for st in cond.stats:
        alarm = by_sensor.get(st.sensor)
        if alarm is None or alarm.value is None:
            continue
        a, b = float(alarm.value), float(st.last)
        if abs(a - b) <= _REL_TOL * max(1.0, abs(a), abs(b)):
            continue
        out.append(
            Conflict(
                kind=CONFLICT_CONTRADICTION,
                summary=(
                    f"{st.label}：告警记录 {a:g} 与工况问答末值 {b:g} 不一致"
                ),
                detail=[
                    f"告警 {alarm.alarm_id} 取的是 {alarm.ts.isoformat()} 的读数",
                    f"工况问答窗口 {st.start.isoformat()}–{st.end.isoformat()}，"
                    f"末值取 {st.end.isoformat()}",
                    "两者来自同一批读数但窗口不同；读者无法从任一份文件里"
                    "判断该信哪个",
                ],
                blocking=True,
                next_step=(
                    "统一取数口径：让工况问答的窗口覆盖告警时刻，"
                    "或在报告里同时写明两个时刻"
                ),
            )
        )
    return out


def detect_direction_conflicts(
    alarm_report: alarms_mod.AlarmReport,
    cond: condition_mod.ConditionAnswer,
) -> list[Conflict]:
    """核对"工况问答说没越限"与"告警说越限"。

    这一类比数值不一致更严重：不是同一事实的两个数值，而是同一事实的
    两个相反结论。看板与日报同时存在时，运维会按自己先看到的那个行动。
    """
    out: list[Conflict] = []
    by_sensor = {a.sensor: a for a in alarm_report.alarms}
    for st in cond.stats:
        alarm = by_sensor.get(st.sensor)
        if alarm is None:
            continue
        cond_says_over = st.over_threshold > 0
        alarm_says_over = alarm.severity in ACTIONABLE
        if cond_says_over == alarm_says_over:
            continue
        out.append(
            Conflict(
                kind=CONFLICT_DIRECTION,
                summary=(
                    f"{st.label}：工况问答判定"
                    f"{'有' if cond_says_over else '无'}越限，"
                    f"告警判定为 {alarm.severity}"
                ),
                detail=[
                    f"工况问答：窗口内 {st.n} 点中 {st.over_threshold} 点越过 warn",
                    f"告警：{alarm.alarm_id} 等级 {alarm.severity}"
                    f"（依据 {alarm.threshold_level or '无'} 阈值）",
                ],
                blocking=True,
                next_step="检查两者使用的阈值是否为同一份注册表，以及窗口是否覆盖告警时刻",
            )
        )
    return out


def detect_orphan_attributions(
    alarm_report: alarms_mod.AlarmReport,
) -> list[Conflict]:
    """核对归因**声称命中**的判据，其测点当前是否仍是那个状态。

    只核对 ``observed == expected`` 的判据。``observed`` 为 unknown（数据缺失）
    或与 expected 不同（未命中）的判据，归因本身就没拿它当证据 —— 把它们也算作
    "引用不符"会把每一条 partial 归因都报成冲突，而 partial 正是缺数据时的
    正常输出。

    在单次 ``evaluate`` 内这条检测恒为真（归因与告警同批计算）。它防的是
    **把归因缓存下来跨批次复用**的用法：那时告警状态已经重算、归因还是旧的，
    报告里的因果解释读起来完全合理、实际无依据。这类 bug 不会报错，
    只会让解释越来越不可信，所以留一道可被触发的检查。
    """
    states = _state_map(alarm_report.alarms)
    labels = {a.sensor: a.label for a in alarm_report.alarms}
    out: list[Conflict] = []
    for att in alarm_report.attribution:
        if att.ruled_out:
            continue
        stale: list[str] = []
        for item in att.basis:
            sensor = item.get("sensor", "")
            expected = item.get("expected", "")
            claimed = item.get("observed", chain.STATE_UNKNOWN)
            if claimed != expected:
                # 归因没声称这条命中，不算引用不符
                continue
            current = states.get(sensor, chain.STATE_UNKNOWN)
            if current != expected:
                stale.append(
                    f"{labels.get(sensor, sensor)}：归因假定 {expected}，"
                    f"实际 {current}"
                )
        if not stale:
            continue
        out.append(
            Conflict(
                kind=CONFLICT_ORPHAN,
                summary=f"归因「{att.cause}」引用了与当前状态不符的测点",
                detail=stale,
                blocking=True,
                next_step="重新计算归因；这条因果解释当前无依据，不能写进报告",
            )
        )
    return out


def detect_coupling_deviations(
    alarm_report: alarms_mod.AlarmReport,
) -> list[Conflict]:
    """核对上下游变化方向是否与耦合表一致。

    **这类不是错误**，所以 ``blocking=False``。耦合表写的是"通常如何"；
    反向变化通常意味着有另一个驱动源，那是有价值的信息，不是数据错误。
    """
    states = _state_map(alarm_report.alarms)
    labels = {a.sensor: a.label for a in alarm_report.alarms}
    out: list[Conflict] = []
    for cp in chain.COUPLINGS:
        if cp.expected == chain.STATE_ANY:
            continue
        trig = states.get(cp.trigger, chain.STATE_UNKNOWN)
        resp = states.get(cp.response, chain.STATE_UNKNOWN)
        if trig != cp.trigger_state or resp == chain.STATE_UNKNOWN:
            continue
        if resp == cp.expected:
            continue
        out.append(
            Conflict(
                kind=CONFLICT_COUPLING,
                summary=(
                    f"{labels.get(cp.trigger, cp.trigger)} "
                    f"{'偏离' if cp.trigger_state == chain.STATE_DEVIATED else '正常'}"
                    f"，但 {labels.get(cp.response, cp.response)} 的变化方向与预期相反"
                ),
                detail=[
                    f"耦合预期：{cp.rationale}",
                    f"实测：{labels.get(cp.response, cp.response)} "
                    f"{chain.STATE_LABELS.get(resp, resp)}",
                    "可能的额外驱动源："
                    + ("、".join(cp.alternatives) if cp.alternatives else "（未列出）"),
                ],
                blocking=False,
                next_step="在报告中说明额外的驱动源，不要把它写成数据错误",
            )
        )
    return out


def detect_pressure_ceiling(
    check: ceiling_mod.CeilingCheck | None,
) -> list[Conflict]:
    """把注入压力上限校核的结果转成冲突条目。

    为什么 ``ok`` 时返回空、``indeterminate`` 时返回提示
    ----------------------------------------------------
    ``ok`` 是"查过了，没问题"，不需要占一行；``indeterminate`` 是"没查成"，
    必须占一行 —— 否则一个从来没接井底压力计的场地，界面上会和
    "校核通过"的场地长得一模一样。这类"缺席被渲染成合格"的失败不会报错，
    只会让一份监管材料带着一个未被核实的合规声明发出去。
    """
    if check is None or check.state == ceiling_mod.STATE_OK:
        return []
    if check.state == ceiling_mod.STATE_EXCEEDED:
        return [
            Conflict(
                kind=CONFLICT_CEILING,
                summary=ceiling_mod.render(check),
                detail=list(check.basis),
                blocking=True,
                next_step=check.next_step,
            )
        ]
    if check.state == ceiling_mod.STATE_NEAR:
        return [
            Conflict(
                kind=CONFLICT_CEILING,
                summary=ceiling_mod.render(check),
                detail=list(check.basis),
                # 接近上限不是违规，不阻断；但必须让人看见余量只剩多少
                blocking=False,
                next_step=check.next_step,
            )
        ]
    return [
        Conflict(
            kind=CONFLICT_CEILING,
            summary=ceiling_mod.render(check),
            detail=list(check.basis),
            blocking=False,
            next_step=check.next_step,
        )
    ]


def detect_conflicts(
    alarm_report: alarms_mod.AlarmReport,
    cond: condition_mod.ConditionAnswer | None = None,
    ceiling_check: ceiling_mod.CeilingCheck | None = None,
) -> list[Conflict]:
    out: list[Conflict] = []
    if cond is not None:
        out += detect_value_conflicts(alarm_report, cond)
        out += detect_direction_conflicts(alarm_report, cond)
    out += detect_orphan_attributions(alarm_report)
    out += detect_pressure_ceiling(ceiling_check)
    out += detect_coupling_deviations(alarm_report)
    # 阻断性冲突排前面：报告读者最该先看到的是"这份材料目前不能发"
    out.sort(key=lambda c: (not c.blocking, c.kind, c.summary))
    return out


# ============================================================
# 三、编排执行
# ============================================================
@dataclass
class OpsResult:
    request_question: str
    period: tuple[datetime, datetime]
    alarms: alarms_mod.AlarmReport | None = None
    condition: condition_mod.ConditionAnswer | None = None
    report: reports_mod.ReportDraft | None = None
    conflicts: list[Conflict] = field(default_factory=list)
    runs: list[TaskRun] = field(default_factory=list)
    #: 注入压力上限校核（见 ops/ceiling.py）。恒有值：``indeterminate`` 也是结论，
    #: 它表示"这项没查成"，而不是"没这一项"。
    ceiling: ceiling_mod.CeilingCheck | None = None

    @property
    def blocking(self) -> list[Conflict]:
        return [c for c in self.conflicts if c.blocking]

    @property
    def can_submit(self) -> bool:
        """现在能不能把报告提交审批。

        三个条件缺一不可：没有阻断冲突、**确实有报告**、引用检查通过。

        中间那条不是形式主义：报告 Agent 失败时 ``report is None``，如果这里
        返回 ``True``，响应里就会出现 ``can_submit: true`` 与
        ``draft_id: ""`` 并存 —— 前端据此渲染出一个"提交审批"按钮，
        点下去 404。没有东西可提交，就是不能提交。
        """
        if self.blocking:
            return False
        if self.report is None:
            return False
        return not self.report.problems

    def summary_line(self) -> str:
        parts: list[str] = []
        if self.alarms is not None:
            parts.append(self.alarms.summary_line())
        if self.ceiling is not None:
            parts.append(ceiling_mod.render(self.ceiling))
        if self.conflicts:
            n_block = len(self.blocking)
            parts.append(
                f"冲突 {len(self.conflicts)} 项"
                + (f"（其中 {n_block} 项阻断）" if n_block else "（均非阻断）")
            )
        else:
            parts.append("未检出跨 Agent 冲突")
        if self.report is not None:
            parts.append(f"报告状态：{reports_mod.STATUS_LABELS[self.report.status]}")
        return "｜".join(parts)

    def to_dict(self) -> dict:
        return {
            "question": self.request_question,
            "period": [self.period[0].isoformat(), self.period[1].isoformat()],
            "summary": self.summary_line(),
            "can_submit": self.can_submit,
            "runs": [r.to_dict() for r in self.runs],
            "conflicts": [c.to_dict() for c in self.conflicts],
            "ceiling": self.ceiling.to_dict() if self.ceiling else None,
            "alarms": self.alarms.to_dict() if self.alarms else None,
            "condition": self.condition.to_dict() if self.condition else None,
            "report": self.report.to_dict() if self.report else None,
        }


def _run(agent: str, reason: str, fn) -> tuple[TaskRun, object]:
    t0 = time.perf_counter()
    try:
        result = fn()
    except Exception as exc:  # noqa: BLE001
        return (
            TaskRun(
                agent=agent,
                reason=reason,
                ok=False,
                elapsed_s=time.perf_counter() - t0,
                error=f"{type(exc).__name__}: {exc}",
            ),
            None,
        )
    return (
        TaskRun(agent=agent, reason=reason, ok=True, elapsed_s=time.perf_counter() - t0),
        result,
    )


def run(request: OpsRequest, registry: ThresholdRegistry) -> OpsResult:
    """按拆分结果依次执行，最后做跨 Agent 核对。

    子 Agent 失败时**继续执行其余任务**，但把失败记进 ``runs`` 并让
    ``can_submit`` 为假。理由：报告写不出来时，告警清单本身仍有价值；
    而"因为周报生成失败所以什么都不输出"会让运维在最需要信息时看不到信息。
    """
    result = OpsResult(
        request_question=request.question, period=request.period
    )
    # 上限校核先算：它不依赖任何子 Agent，而监管材料要用到它。
    # 放在循环里会让"没勾选生成报告"时这一项凭空消失 —— 而"这次没查上限"
    # 与"上限没问题"在界面上必须能区分。
    result.ceiling = ceiling_mod.check(request.readings)
    tasks = request.tasks()

    for task in tasks:
        if task.agent == "alarms":
            run_rec, alarms = _run(
                "alarms",
                task.reason,
                lambda: alarms_mod.evaluate(request.readings, registry),
            )
            result.runs.append(run_rec)
            result.alarms = alarms  # type: ignore[assignment]
        elif task.agent == "condition":
            run_rec, cond = _run(
                "condition",
                task.reason,
                lambda: condition_mod.answer(
                    request.question, request.readings, registry, now=request.now
                ),
            )
            result.runs.append(run_rec)
            result.condition = cond  # type: ignore[assignment]
        elif task.agent == "reports":
            if result.alarms is None:
                result.runs.append(
                    TaskRun(
                        agent="reports",
                        reason=task.reason,
                        ok=False,
                        elapsed_s=0.0,
                        error="上游告警评估未成功，报告没有可引用的判定结果",
                    )
                )
                continue
            report = None
            if request.report_kind == "regulatory":
                run_rec, report = _run(
                    "reports",
                    task.reason,
                    lambda: reports_mod.build_regulatory(
                        result.alarms,  # type: ignore[arg-type]
                        registry,
                        period=request.period,
                        permit_id=request.permit_id,
                        ceiling=result.ceiling,
                    ),
                )
            elif request.report_kind == "weekly":
                # 周报需要多期。这里**不**用一期冒充一周 —— 那会让"各期最高
                # 等级分布"这个核心字段只有一个样本，读起来却像一周的结论。
                periods = list(request.periods)
                if len(periods) < 2:
                    result.runs.append(
                        TaskRun(
                            agent="reports",
                            reason=task.reason,
                            ok=False,
                            elapsed_s=0.0,
                            error=(
                                f"周报需要至少 2 期数据，本次收到 {len(periods)} 期。"
                                "只给一期时报告里只有一条样本，"
                                "分不清'这周一直 warn'与'这周某天 warn 一次'"
                            ),
                        )
                    )
                    continue

                def _build_weekly() -> reports_mod.ReportDraft:
                    per_period = [
                        alarms_mod.evaluate(
                            [r for r in request.readings if lo <= r.ts <= hi],
                            registry,
                        )
                        for lo, hi in periods
                    ]
                    return reports_mod.build_weekly(
                        per_period, registry, period=request.period
                    )

                run_rec, report = _run("reports", task.reason, _build_weekly)
            else:
                run_rec, report = _run(
                    "reports",
                    task.reason,
                    lambda: reports_mod.build_daily(
                        result.alarms,  # type: ignore[arg-type]
                        registry,
                        period=request.period,
                    ),
                )
            result.runs.append(run_rec)
            result.report = report  # type: ignore[assignment]

    if result.alarms is not None:
        result.conflicts = detect_conflicts(
            result.alarms, result.condition, result.ceiling
        )
    else:
        # 告警 Agent 挂了也要把上限校核的结论说出来：它不依赖告警，
        # 而且"压力超没超破裂压力"是整份材料里最不能缺的一项
        result.conflicts = detect_pressure_ceiling(result.ceiling)
    return result


def render_brief(result: OpsResult) -> str:
    """把结果渲染成一段可读的简报。

    顺序是刻意的：**先矛盾，后结论**。读者如果只看前三行，看到的应该是
    "这份材料目前不能发"以及为什么，而不是一段看起来很确定的工况描述。
    """
    lines: list[str] = []
    start, end = result.period
    lines.append(f"# 运维简报（{start:%Y-%m-%d %H:%M} 至 {end:%Y-%m-%d %H:%M}）")
    lines.append("")

    blocking = result.blocking
    if blocking:
        lines.append("## 需要先处理：阻断性冲突")
        lines.append("")
        for c in blocking:
            lines.append(c.render())
            lines.append("")
    elif result.conflicts:
        lines.append("## 需要说明：非阻断性偏离")
        lines.append("")
        for c in result.conflicts:
            lines.append(c.render())
            lines.append("")

    if result.alarms is not None:
        lines.append("## 告警")
        lines.append("")
        lines.append(result.alarms.summary_line())
        lines.append("")
        for a in result.alarms.actionable:
            lines.append(
                f"- {a.ts:%H:%M} **{a.label}** "
                f"{a.severity}（{a.value:g}）{a.alarm_id}"
            )
        lines.append("")

        # 归因摘要。只列未排除的、且强度为 strong/partial 的 ——
        # 简报的读者要的是"这些告警指向什么"，不是规则库的全集。
        # 被排除的原因留在报告里：那里有空间写排除理由，而简报没有。
        active = [a for a in result.alarms.attribution if not a.ruled_out]
        if active:
            lines.append("## 归因（候选，非结论）")
            lines.append("")
            for att in active[:5]:
                lines.append(f"- **{att.cause}**（{att.confidence}）")
                for ev in att.evidence[:2]:
                    lines.append(f"  - {ev}")
            ruled = [a for a in result.alarms.attribution if a.ruled_out]
            if ruled:
                lines.append("")
                lines.append(
                    f"已排除 {len(ruled)} 条候选原因（理由见报告）："
                    + "、".join(a.cause.split("（")[0] for a in ruled[:3])
                )
            lines.append("")

    if result.condition is not None:
        lines.append("## 工况问答")
        lines.append("")
        lines.append(result.condition.text)
        lines.append("")

    if result.report is not None:
        lines.append("## 报告")
        lines.append("")
        lines.append(
            f"{result.report.title}｜状态："
            f"{reports_mod.STATUS_LABELS[result.report.status]}｜"
            f"引用 {len(result.report.citations)} 处"
        )
        if result.report.problems:
            lines.append("")
            lines.append("引用检查未通过：")
            for p in result.report.problems:
                lines.append(f"- {p.render()}")
        lines.append("")

    lines.append("## 任务执行记录")
    lines.append("")
    for r in result.runs:
        mark = "OK" if r.ok else "FAIL"
        tail = f"｜{r.error}" if r.error else ""
        lines.append(f"- [{mark}] {r.agent}（{r.elapsed_s:.2f}s）：{r.reason}{tail}")
    return "\n".join(lines)
