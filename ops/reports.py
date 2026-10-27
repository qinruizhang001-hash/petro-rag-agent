"""
报告 Agent
==========
日报、周报、监管材料的起草。核心不是"写得好"，是**每个数字都能回代**。

为什么报告里的数字必须带引用标记
--------------------------------
告警 Agent 已经把每个数字的出处算出来了，报告 Agent 的工作只是搬运。
一旦允许它"组织语言"，模型就会在措辞里顺手加一个看起来合理的数字 ——
"压力持续偏高约 15%" 里的 15% 可能来自任何地方，也可能不来自任何地方。
读者无法分辨，而监管材料里的无法回代的数字是实质性缺陷，不是笔误。

所以这里的做法是：模板里每个数字都通过 ``_num()`` 输出，它强制要求一个
``Citation``，并在正文里渲染成 ``106.5 bar [alarm:A-20261026T1110-wellhead_p]``。
生成后跑一遍检查：正文里出现的每个数字，要么紧跟引用标记，要么落在
显式豁免区（时间戳、章节编号）。有落单的数字就不允许提交审批。

HITL 状态机
-----------
::

    draft ──submit──→ pending_review ──approve──→ approved
                              │
                              └──reject（必须写意见）──→ rejected
                                                          │
                                            重新生成 ────────┘

三条硬规则，都是为了让"谁批准了什么"可追溯：

1. ``submit`` 前必须通过引用检查。不过就不能提交 —— 否则审批人看到的
   是一份自己无法核对的稿子，"批准"于是变成"背书"。
2. ``approve`` 只能从 ``pending_review`` 出发。草稿直接批准等于没有审批环节，
   而这正是最容易发生的"省一步"。
3. ``reject`` 必须写意见。没有意见的驳回，起草方无法知道要改什么，
   下一稿会原样再来一遍。

监管材料额外要求
----------------
- 必须写明许可编号（没有编号的材料无法归入任何档案）
- 必须单独列出**监测盲区**（未配置阈值的测点、无读数的测点）。
  隐瞒盲区比报告异常更严重：异常可以解释，盲区是"我们没看见"。
- 必须写明本报告**不构成泄漏认定**。筛查级监测的结论不能写成场地级验证。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Sequence

from ops.alarms import AlarmReport
from ops.contracts import (
    ACTIONABLE,
    INDETERMINATE,
    SEVERITY_LABELS,
    STAGE_LABELS,
    STAGES,
    Alarm,
    Citation,
)
from ops.thresholds import ThresholdRegistry

# ─── 状态 ───
STATUS_DRAFT = "draft"
STATUS_PENDING = "pending_review"
STATUS_APPROVED = "approved"
STATUS_REJECTED = "rejected"

STATUS_LABELS = {
    STATUS_DRAFT: "草稿",
    STATUS_PENDING: "待审批",
    STATUS_APPROVED: "已批准",
    STATUS_REJECTED: "已驳回",
}

#: 允许的状态迁移。写成表而不是散落的 if，是因为漏一条分支的后果是
#: "某个状态无法离开"，而那种 bug 只会在真实审批时才发现
TRANSITIONS: dict[str, tuple[str, ...]] = {
    STATUS_DRAFT: (STATUS_PENDING,),
    STATUS_PENDING: (STATUS_APPROVED, STATUS_REJECTED),
    STATUS_REJECTED: (STATUS_DRAFT,),
    STATUS_APPROVED: (),
}


# ─── 引用检查 ───
#: 豁免区：这些片段里的数字是**标识**不是**测量值**，不需要引用。
#:
#: 判定标准只有一条：这个数字能不能通过"回到原始记录"核对？日期不能
#: （它是坐标不是观测量），章节号不能，化学式里的 2 不能。而"106.5 bar"
#: 能 —— 所以它必须带引用。
_EXEMPT_PATTERNS: tuple[re.Pattern, ...] = (
    re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(?::\d{2})?"),   # ISO 时间戳
    re.compile(r"\d{4}-\d{2}-\d{2}"),                            # 日期
    # 顺序要紧：先匹配更长的时间表达，否则 \d{2}:\d{2} 会把后半段挖掉，
    # 让 "10-26 08:10" 剩下一个孤立的 "10" 被当成无引用数字
    re.compile(r"\b\d{2}-\d{2}\s+\d{2}:\d{2}\b"),               # 月-日 时:分
    re.compile(r"\d{2}:\d{2}"),                                  # 时刻
    re.compile(r"^#{1,6}\s*\d+[.、]?\s", re.M),                  # 章节编号
    re.compile(r"^\s*\d+[.、]\s", re.M),                        # 列表序号
    re.compile(r"§\d+(?:\.\d+)*"),                               # 文件章节号
    re.compile(r"\bCO2\b"),                                      # 化学式
    re.compile(r"\bv\d+(?:\.\d+)*\b"),                           # 版本号
)

#: 记录原文行（以 ``>`` 开头）。这些行转述判定记录，数字归属同一条目的引用，
#: 因此不单独要求行内引用。用 blockquote 而不是自定义标记，是因为它在渲染后
#: 仍然看得出"这是引用的原文"，而自定义标记渲染后不可见、读者无从判断。
_QUOTE_LINE = re.compile(r"^\s*>")

_CITATION_RE = re.compile(r"\[(alarm|reading|summary):([^\]]+)\]")
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")

#: 严格区块标记。区块内的每个数字都必须绑定引用；区块外（说明文字、
#: 政策参数、处置时限、测点 ID）不做逐数字核对。
#:
#: 为什么需要这个区分：一份真实报告里大部分数字不是测量值 ——
#: "24 小时内复采样"里的 24 是处置时限，"notice 留 10% 余量"里的 10 是
#: 政策参数，"co2_purity" 里的 2 是标识符的一部分。对它们一律要求引用，
#: 只会得到一屏误报，然后所有人学会忽略这个检查 —— 那比没有检查更糟。
#:
#: 区块标记用 HTML 注释，渲染后不可见，但在源码里一眼能看出哪几段受约束。
STRICT_OPEN = "<!-- strict -->"
STRICT_CLOSE = "<!-- /strict -->"


def _strict(lines: list[str]) -> list[str]:
    """把若干行包进严格区块。"""
    return [STRICT_OPEN] + list(lines) + [STRICT_CLOSE]


def _strict_line_numbers(text: str) -> set[int]:
    """算出落在严格区块内的行号（1-based）。"""
    inside: set[int] = set()
    depth = 0
    for i, line in enumerate(text.split("\n"), start=1):
        stripped = line.strip()
        if stripped == STRICT_OPEN:
            depth += 1
            continue
        if stripped == STRICT_CLOSE:
            depth = max(0, depth - 1)
            continue
        if depth:
            inside.add(i)
    return inside


def _mask_exempt(text: str) -> str:
    """把豁免片段与引用标记替换成等长占位，保持位置不变以便报行号。

    引用标记本身也要挖掉：``A-20261026T0810-wellhead_p`` 里的数字是告警 ID
    的一部分，它是标识，不是被引用的测量值。
    """
    masked = text
    for pat in _EXEMPT_PATTERNS:
        masked = pat.sub(lambda m: " " * len(m.group(0)), masked)
    masked = _CITATION_RE.sub(lambda m: " " * len(m.group(0)), masked)
    return masked


@dataclass(frozen=True)
class CitationProblem:
    kind: str          # unsourced / dangling
    token: str
    line: int
    context: str

    def render(self) -> str:
        if self.kind == "unsourced":
            return (
                f"第 {self.line} 行数字 {self.token!r} 没有引用标记"
                f"（上下文：…{self.context}…）"
            )
        return (
            f"第 {self.line} 行引用标记指向未登记的来源 {self.token!r}"
            f"（上下文：…{self.context}…）"
        )


def check_citations(text: str, citations: dict[str, Citation]) -> list[CitationProblem]:
    """检查严格区块内的数字是否都有引用，以及所有引用是否指向已登记来源。

    两类问题都要查。只查前者会漏掉"引用了不存在的告警"——那种稿子看起来
    每个数字都有出处，而所有出处都是空的。

    绑定规则：数字之后、下一个数字之前，必须出现一个引用标记。
    用"下一个数字之前"而不是固定字符窗口，是因为单位串长度不定
    （``bar`` 与 ``t/(d·bar)`` 差很多），固定窗口会把长单位的数字误判为无引用。
    """
    problems: list[CitationProblem] = []
    strict = _strict_line_numbers(text)
    masked_lines = _mask_exempt(text).split("\n")
    raw_lines = text.split("\n")

    for lineno, (line, raw) in enumerate(zip(masked_lines, raw_lines), start=1):
        marks = [(m.start(), m.end(), m.group(0)) for m in _CITATION_RE.finditer(raw)]
        # 悬空引用在任何行上都要查：引用了不存在的告警比缺引用更严重
        for _s, _e, token in marks:
            if token not in citations:
                problems.append(
                    CitationProblem("dangling", token, lineno, raw.strip()[:60])
                )
        if lineno not in strict:
            continue
        nums = [(m.start(), m.end()) for m in _NUMBER_RE.finditer(line)]
        for idx, (_ns, ne) in enumerate(nums):
            next_num_start = nums[idx + 1][0] if idx + 1 < len(nums) else len(line) + 1
            bound = any(ne <= ms < next_num_start for ms, _me, _t in marks)
            if not bound:
                problems.append(
                    CitationProblem(
                        "unsourced", line[_ns:ne].strip(), lineno, raw.strip()[:60]
                    )
                )
    return problems


def _num(value: float, unit: str, cite: Citation, *, fmt: str = "g") -> str:
    """把一个数字连同它的出处渲染成正文片段。

    模板里**不允许**直接写 ``f"{value}"`` —— 那个数字会立刻失去出处。
    所有数字都必须经过这里，这是一条可以静态检查的约定。
    """
    unit_txt = f" {unit}" if unit and unit != "-" else ""
    return f"{format(value, fmt)}{unit_txt} {cite.render()}"


# ─── 草稿 ───
@dataclass
class ReportDraft:
    kind: str
    title: str
    period: tuple[datetime, datetime]
    body: str
    citations: dict[str, Citation] = field(default_factory=dict)
    status: str = STATUS_DRAFT
    #: 审批轨迹，只追加不修改
    trail: list[dict] = field(default_factory=list)
    #: 引用检查结果（提交前必须为空）
    problems: list[CitationProblem] = field(default_factory=list)
    #: 监管材料必填
    permit_id: str = ""
    #: 生成时已知的盲区（必须进正文，也留一份结构化副本供程序读取）
    blind_spots: list[str] = field(default_factory=list)

    @property
    def ready_to_submit(self) -> bool:
        return self.status == STATUS_DRAFT and not self.problems

    def register(self, cite: Citation) -> Citation:
        self.citations[cite.render()] = cite
        return cite

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "title": self.title,
            "period": [self.period[0].isoformat(), self.period[1].isoformat()],
            "body": self.body,
            "status": self.status,
            "status_label": STATUS_LABELS[self.status],
            "trail": list(self.trail),
            "problems": [p.render() for p in self.problems],
            "permit_id": self.permit_id,
            "blind_spots": list(self.blind_spots),
            "citation_count": len(self.citations),
        }


def _stage_section(draft: ReportDraft, report: AlarmReport, stage: str) -> list[str]:
    lines: list[str] = []
    alarms = [a for a in report.alarms if a.stage == stage]
    if not alarms:
        return lines
    lines.append(f"### {STAGE_LABELS[stage]}")
    for a in alarms:
        sev = SEVERITY_LABELS.get(a.severity, a.severity)
        cite = draft.register(
            Citation("alarm", a.alarm_id, f"{a.label} 在 {a.ts.isoformat()} 的判定")
        )
        if a.value is None:
            lines += _strict(
                [f"- **{a.label}**：{sev}（无有效读数）{cite.render()}"]
            )
        else:
            from ops.thresholds import SENSORS as _S
            unit = _S[a.sensor].unit if a.sensor in _S else ""
            lines += _strict(
                [f"- **{a.label}**：{_num(a.value, unit, cite)}，等级 {sev}"]
            )
            lines.append(f"  > 阈值出处：{a.provenance}")
        for reason in a.rationale:
            lines.append(f"  > 判定依据：{reason}")
        active = [x for x in a.attribution if not x.ruled_out]
        for att in active:
            lines.append(f"  - 归因（{att.confidence}）：{att.cause}")
        for act in a.actions:
            lines.append(f"  - 处置：{act['action']}（{act['owner']}）")
    return lines


def build_daily(
    report: AlarmReport,
    registry: ThresholdRegistry,
    *,
    period: tuple[datetime, datetime],
    title: str | None = None,
) -> ReportDraft:
    """生成日报草稿。"""
    start, end = period
    draft = ReportDraft(
        kind="daily",
        title=title or f"CCUS 全链条运行日报（{start:%Y-%m-%d}）",
        period=period,
        body="",
    )

    top = report.top_severity()
    counts = report.by_severity()
    total_cite = draft.register(
        Citation(
            "summary",
            "告警统计",
            "按 alarms.AlarmReport.by_severity() 对本期每条告警的 severity 计数",
        )
    )
    actionable_n = len(report.actionable)

    lines: list[str] = [
        f"# {draft.title}",
        "",
        f"报告期：{start:%Y-%m-%d %H:%M} 至 {end:%Y-%m-%d %H:%M}",
        "",
        "## 一、总体结论",
        "",
    ]
    if not report.alarms:
        lines.append(
            "本期**未收到任何测点读数**。这不构成「运行正常」的结论："
            "没有数据与数据正常是两件事，请先确认采集链路。"
        )
    elif top == INDETERMINATE:
        lines.append(
            "本期所有测点均**不可判定**（仪表不可信或阈值未配置），"
            "因此无法给出运行结论。"
        )
    elif actionable_n:
        lines += _strict([
            f"本期最高等级为 **{SEVERITY_LABELS[top]}**，"
            f"需人工介入的告警 {actionable_n} 条 {total_cite.render()}。"
            "详见第三节。"
        ])
    else:
        lines += _strict([
            f"本期最高等级为 **{SEVERITY_LABELS[top]}**，"
            f"无需要人工介入的告警 {total_cite.render()}。"
        ])

    # ── 分级计数 ──
    lines += ["", "### 分级计数", ""]
    count_lines = [
        f"- {SEVERITY_LABELS.get(name, name)}：{n} {total_cite.render()}"
        for name, n in counts.items()
        if n
    ]
    lines += _strict(count_lines)

    # ── 分环节 ──
    lines += ["", "## 二、分环节明细", ""]
    any_stage = False
    for stage in STAGES:
        sec = _stage_section(draft, report, stage)
        if sec:
            any_stage = True
            lines += sec
            lines.append("")
    if not any_stage:
        lines.append("本期无分环节明细（未收到读数）。")

    # ── 归因与处置 ──
    active = [a for a in report.attribution if not a.ruled_out]
    ruled = [a for a in report.attribution if a.ruled_out]
    lines += ["", "## 三、归因与处置", ""]
    if active:
        for att in active:
            lines.append(f"- **{att.cause}**（{STAGE_LABELS.get(att.stage, att.stage)}，"
                         f"强度 {att.confidence}）")
            for ev in att.evidence:
                lines.append(f"  - {ev}")
    else:
        lines.append("本期没有命中的归因规则。")
    if ruled:
        lines += ["", "### 已排除的原因", ""]
        for att in ruled:
            lines.append(f"- ~~{att.cause}~~：{att.ruled_out_reason}")

    # ── 数据完整性（盲区必须披露）──
    lines += ["", "## 四、数据完整性", ""]
    blind: list[str] = []
    if report.unconfigured:
        blind.append(
            "未配置阈值的测点（无法判定，非正常）：" + "、".join(report.unconfigured)
        )
    if report.no_data:
        blind.append("本期无读数的测点：" + "、".join(report.no_data))
    if blind:
        for b in blind:
            lines.append(f"- {b}")
        draft.blind_spots = blind
    else:
        lines.append("- 所有已声明测点均配置了阈值并收到读数。")
    lines += [
        "",
        f"阈值注册表来源：{report.threshold_origin}",
        f"分级政策：{report.policy}",
    ]

    lines += [
        "",
        "## 五、审批",
        "",
        f"状态：{STATUS_LABELS[draft.status]}",
        "",
    ]

    draft.body = "\n".join(lines)
    draft.problems = check_citations(draft.body, draft.citations)
    return draft


def build_weekly(
    reports: Sequence[AlarmReport],
    registry: ThresholdRegistry,
    *,
    period: tuple[datetime, datetime],
    title: str | None = None,
) -> ReportDraft:
    """周报：把若干期日报聚合成趋势。

    刻意不做"平均严重度"这类指标 —— 它会把"一天 critical、六天正常"
    与"七天都 warn"算成同一件事，而这两种情况的处置完全不同。
    周报只报**计数分布**与**出现过的最高等级**。
    """
    if not reports:
        raise ValueError("周报需要至少一期日报数据；空列表会让周报看起来'一切正常'")

    start, end = period
    draft = ReportDraft(
        kind="weekly",
        title=title or f"CCUS 全链条运行周报（{start:%Y-%m-%d} 至 {end:%Y-%m-%d}）",
        period=period,
        body="",
    )
    count_cite = draft.register(
        Citation(
            "summary",
            "周报聚合",
            "对本周期内每期 AlarmReport 的最高等级与 actionable 计数做累加",
        )
    )

    # 每周各等级出现次数（按"本期最高等级"统计，而不是按告警条数）
    top_counts: dict[str, int] = {}
    for r in reports:
        t = r.top_severity()
        top_counts[t] = top_counts.get(t, 0) + 1

    lines: list[str] = [
        f"# {draft.title}",
        "",
        f"报告期：{start:%Y-%m-%d} 至 {end:%Y-%m-%d}，共 {len(reports)} 期"
        f" {count_cite.render()}",
        "",
        "## 一、各期最高等级分布",
        "",
    ]
    dist_lines = [
        f"- {SEVERITY_LABELS[name]}：{top_counts.get(name, 0)} 期 {count_cite.render()}"
        for name in SEVERITY_LABELS
        if top_counts.get(name, 0)
    ]
    lines += _strict(dist_lines)

    lines += ["", "## 二、本周期出现过的告警", ""]
    seen: dict[str, Alarm] = {}
    for r in reports:
        for a in r.alarms:
            if a.severity in ACTIONABLE:
                seen.setdefault(a.alarm_id, a)
    if seen:
        for a in sorted(seen.values(), key=lambda x: x.ts):
            cite = draft.register(
                Citation("alarm", a.alarm_id, f"{a.label} 在 {a.ts.isoformat()} 的判定")
            )
            lines += _strict([
                f"- {a.ts:%m-%d %H:%M} **{a.label}** "
                f"{SEVERITY_LABELS.get(a.severity, a.severity)} {cite.render()}"
            ])
    else:
        lines.append("- 本周期无需要人工介入的告警。")

    blind: list[str] = []
    for r in reports:
        for key in r.unconfigured:
            item = f"测点 {key} 在本周期内始终没有配置阈值"
            if item not in blind:
                blind.append(item)
    lines += ["", "## 三、数据完整性", ""]
    if blind:
        lines += [f"- {b}" for b in blind]
        draft.blind_spots = blind
    else:
        lines.append("- 本周期内所有测点均配置了阈值。")

    lines += ["", "## 四、审批", "", f"状态：{STATUS_LABELS[draft.status]}", ""]
    draft.body = "\n".join(lines)
    draft.problems = check_citations(draft.body, draft.citations)
    return draft


# ─── 监管材料：注入压力上限校核 ───
def _ceiling_section(draft: ReportDraft, ceiling) -> list[str]:
    """把上限校核渲染成报告正文。

    数字进严格区块（必须带引用），法规比例与处置动作留在区块外 ——
    "90 %" 是法规给定的常数，不是本期的测量值，给它挂一条"引用"反而
    会让读者以为它可以回代到某条记录。
    """
    if ceiling is None:
        return [
            "本期**未执行**注入压力上限校核。",
            "",
            "这不是「校核通过」，而是「本项缺失」：没有井底压力或破裂压力读数时，"
            "校核无法进行。缺失原因须在本节补记，否则许可条件中的这一项"
            "在档案里是无记录的。",
        ]

    out: list[str] = []
    # 用 citation 而不是 source：后者含「换辖区时须同时替换本常量与出处」，
    # 那是写给维护者的，出现在监管正文里等于把内部备忘抄进了合规文件
    out.append(f"结论：**{ceiling.label}**（依据 {ceiling.citation}）")
    out.append("")

    if ceiling.bhp is not None and ceiling.bhp_ts is not None:
        cite_bhp = draft.register(
            Citation(
                "reading",
                f"bhp@{ceiling.bhp_ts.isoformat()}",
                "井底注入压力读数（本报告期峰值）",
            )
        )
        out += _strict([f"- 井底压力峰值：{_num(ceiling.bhp, 'bar', cite_bhp)}"])
    if ceiling.frac_p is not None and ceiling.frac_ts is not None:
        cite_frac = draft.register(
            Citation(
                "reading",
                f"frac_p@{ceiling.frac_ts.isoformat()}",
                "注入层破裂压力（峰值时刻生效的那次 LOT/XLOT 实测值，或标注为估算的推算值）",
            )
        )
        out += _strict([f"- 破裂压力：{_num(ceiling.frac_p, 'bar', cite_frac)}"])
    if ceiling.ratio is not None:
        cite_ratio = draft.register(
            Citation(
                "summary",
                "ceiling-ratio",
                "井底压力峰值 / 破裂压力。按 ops/ceiling.py 的 check() 复算："
                "取本报告期内 bhp 的最大值，除以**时间戳不晚于该峰值**的最后一次 frac_p。"
                "不是时间上最新的 frac_p —— 峰值之后完成的复测不得用来改写峰值当时的结论",
            )
        )
        out += _strict(
            [f"- 占破裂压力比例：{_num(ceiling.ratio * 100, '%', cite_ratio, fmt='.4g')}"]
        )

    out.append("")
    for line in ceiling.basis:
        out.append(f"> {line}")
    if ceiling.next_step:
        out += ["", f"处置：{ceiling.next_step}"]

    if ceiling.state == "indeterminate":
        # 不在这里指定补正动作。不可判定的成因有五种（缺 bhp、缺 frac_p、
        # 读数非正、峰值时刻没有生效的测定……），正确的补正动作各不相同，
        # 而 `next_step` 已经给出了针对性的那一条。在这里写"补齐读数"
        # 会把"需要补做 LOT"说成"补录一个数"，读者照着做只会填一个假数进来。
        out += [
            "",
            "**本项未通过校核。** 本期不得声明注入压力处于许可上限以内；"
            "须按缺失处理，按上述处置补正后重新校核。",
        ]
    return out


def build_regulatory(
    report: AlarmReport,
    registry: ThresholdRegistry,
    *,
    period: tuple[datetime, datetime],
    permit_id: str,
    title: str | None = None,
    ceiling=None,
) -> ReportDraft:
    """监管材料。比日报多三样东西：许可编号、MVA 专节、免责边界。

    ``ceiling`` 是注入压力上限校核（``ops.ceiling.CeilingCheck``）。它是
    **唯一一项法规直接给定的判据**，所以单独成节并且排在 MVA 之前 ——
    MVA 回答"有没有漏"，上限校核回答"注入本身合不合规"，后者是许可的
    前提条件，不是运行优化项。

    省略 ``ceiling`` 时本节会明确写"未执行"，而不是留空：一份监管材料里
    缺一节，读者无法分辨是"不需要"还是"没做"。
    """
    if not permit_id.strip():
        raise ValueError(
            "监管材料必须写明许可编号。没有编号的材料无法归入档案，"
            "也无法在后续核查中被引用"
        )

    start, end = period
    draft = ReportDraft(
        kind="regulatory",
        title=title or f"CCUS 封存监测报告（{start:%Y-%m-%d} 至 {end:%Y-%m-%d}）",
        period=period,
        body="",
        permit_id=permit_id,
    )

    mva_alarms = [a for a in report.alarms if a.stage == "mva"]
    lines: list[str] = [
        f"# {draft.title}",
        "",
        f"许可编号：{permit_id}",
        f"报告期：{start:%Y-%m-%d %H:%M} 至 {end:%Y-%m-%d %H:%M}",
        "",
        "## 一、监测范围与数据可用性",
        "",
    ]

    blind: list[str] = []
    if report.unconfigured:
        blind.append("未配置判定阈值的监测项：" + "、".join(report.unconfigured))
    if report.no_data:
        blind.append("本期无有效读数的监测项：" + "、".join(report.no_data))
    if blind:
        lines.append("本报告期内存在以下**监测盲区**，须一并披露：")
        for b in blind:
            lines.append(f"- {b}")
        draft.blind_spots = blind
    else:
        lines.append("所有已声明监测项均配置阈值并取得有效读数。")

    lines += ["", "## 二、注入压力上限校核", ""]
    lines += _ceiling_section(draft, ceiling)

    lines += ["", "## 三、MVA 监测结果", ""]
    if not mva_alarms:
        lines.append("本期未收到 MVA 层监测数据。**不能据此认为未发生泄漏**。")
    else:
        for a in mva_alarms:
            from ops.thresholds import SENSORS as _S
            sev = SEVERITY_LABELS.get(a.severity, a.severity)
            cite = draft.register(
                Citation("alarm", a.alarm_id, f"{a.label} 在 {a.ts.isoformat()} 的判定")
            )
            unit = _S[a.sensor].unit if a.sensor in _S else ""
            if a.value is None:
                lines += _strict([f"- {a.label}：{sev}（无有效读数）{cite.render()}"])
            else:
                lines += _strict([
                    f"- {a.label}：{_num(a.value, unit, cite)}，等级 {sev}"
                ])
                lines.append(f"  > 阈值出处：{a.provenance}")
            for att in a.attribution:
                if not att.ruled_out:
                    lines.append(f"  - 归因：{att.cause}（{att.confidence}）")

    lines += [
        "",
        "## 四、异常与处置",
        "",
    ]
    actionable = report.actionable
    if not actionable:
        lines.append("本期无达到告警级别的异常。")
    else:
        for a in actionable:
            cite = draft.register(
                Citation("alarm", a.alarm_id, f"{a.label} 在 {a.ts.isoformat()} 的判定")
            )
            lines += _strict([
                f"- {a.ts:%m-%d %H:%M} {a.label}："
                f"{SEVERITY_LABELS.get(a.severity, a.severity)} {cite.render()}"
            ])
            for act in a.actions:
                lines.append(f"  - 处置：{act['action']}（{act['owner']}）")

    lines += [
        "",
        "## 五、结论边界",
        "",
        "1. 本报告的判定基于在线监测数据与项目配置的判定阈值，"
        "阈值出处逐条列于各条目。",
        "2. 本报告**不构成泄漏认定**，也不构成「未发生泄漏」的证明。"
        "筛查级监测只能提示是否需要进一步核查，"
        "泄漏认定须依据许可条件规定的核查程序。",
        "3. 监测盲区已在第一节披露；盲区范围内的工况本报告无法评价。",
        "4. 注入压力上限校核只覆盖「井底压力 / 破裂压力」这一条法规判据，"
        "不覆盖套管抗内压、水泥环密封、盖层力学完整性等其他完整性要素。",
        "",
        "## 六、审批",
        "",
        f"状态：{STATUS_LABELS[draft.status]}",
        "",
    ]

    draft.body = "\n".join(lines)
    draft.problems = check_citations(draft.body, draft.citations)
    return draft


# ─── 状态机 ───
def _advance(draft: ReportDraft, to: str, actor: str, note: str = "") -> None:
    allowed = TRANSITIONS.get(draft.status, ())
    if to not in allowed:
        raise ValueError(
            f"不允许从 {STATUS_LABELS.get(draft.status, draft.status)} "
            f"迁移到 {STATUS_LABELS.get(to, to)}。"
            f"当前状态允许的下一状态：{allowed or '（终态，无）'}"
        )
    draft.trail.append({
        "at": datetime.now().isoformat(timespec="seconds"),
        "from": draft.status,
        "to": to,
        "actor": actor,
        "note": note,
    })
    draft.status = to


def submit(draft: ReportDraft, *, actor: str = "起草方") -> ReportDraft:
    """提交审批。引用检查不通过时**拒绝提交**。"""
    if draft.status != STATUS_DRAFT:
        raise ValueError(
            f"只有草稿可以提交，当前状态：{STATUS_LABELS.get(draft.status, draft.status)}"
        )
    if draft.problems:
        detail = "\n  - ".join(p.render() for p in draft.problems)
        raise ValueError(
            "引用检查未通过，拒绝提交审批：\n  - " + detail
            + "\n审批人无法核对无出处的数字，批准这样的稿子等于背书"
        )
    _advance(draft, STATUS_PENDING, actor)
    return draft


def approve(draft: ReportDraft, *, reviewer: str, note: str = "") -> ReportDraft:
    if not reviewer.strip():
        raise ValueError("批准必须记录审批人；无署名的批准无法追溯")
    _advance(draft, STATUS_APPROVED, reviewer, note)
    return draft


def reject(draft: ReportDraft, *, reviewer: str, note: str) -> ReportDraft:
    if not reviewer.strip():
        raise ValueError("驳回必须记录审批人")
    if not note.strip():
        raise ValueError(
            "驳回必须写意见。没有意见的驳回，起草方无法知道要改什么，"
            "下一稿会原样再来一遍"
        )
    _advance(draft, STATUS_REJECTED, reviewer, note)
    return draft


def reopen(draft: ReportDraft, *, actor: str = "起草方") -> ReportDraft:
    """驳回后重新打开为草稿，保留完整审批轨迹。"""
    _advance(draft, STATUS_DRAFT, actor, "按驳回意见修改后重新起草")
    return draft
