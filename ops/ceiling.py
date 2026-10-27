"""
注入压力上限校核
================
回答一个问题：**井底注入压力有没有越过地层破裂压力的 90%？**

为什么单独一个模块，而不是做成一条阈值告警
------------------------------------------
阈值告警的语义是"这个测点偏离了配置的运行窗口"，它是**项目自定**的；
90% 破裂压力是**法规**给定的上限，两者性质不同：

* 前者可以按项目调，后者调不了 —— 调了就是不合规；
* 前者的对象是单个测点，后者的对象是**两个读数之比**；
* 前者越限要人判断，后者越限本身就是一条违规事实。

把法规上限塞进测点阈值里，会出现"井口压力正常、井底已经超破裂压力"这种
组合，而单测点告警看不出来。

为什么不能拿井口压力代替
------------------------
井口压力与井底压力之间隔着**液柱**与**摩阻**，而这两项都随井深、注入量、
流体密度变化。同一个 120 bar 井口压力：

* 在 800 m 浅井，静液柱约 80 bar，井底约 200 bar —— 对 250 bar 的破裂压力
  已经接近上限；
* 在 2500 m 深井，静液柱约 250 bar，井底约 370 bar —— 对 450 bar 的破裂压力
  才用掉 82 %。

所以"井口压力 120 bar 是否安全"这个问题**没有与井深无关的答案**。用井口阈值
代替破裂压力校核，在浅井上会漏报，在深井上会误报，而两种错误都不会报错。

为什么取窗口内的**最大值**而不是末值
------------------------------------
合规判据是"任何时刻都不得超过"，不是"现在不得超过"。用末值会让一次发生在
班初的压力尖峰在交班时完全消失 —— 而那正是要报给监管的那一次。

为什么取"峰值时刻**生效**的那次破裂压力"，而不是"最近一次"
------------------------------------------------------------
破裂压力是周期性作业结果，一个窗口里可能出现两次（窗口跨越了一次复测）。
若取时间上最后一条，就会出现这种情形：峰值发生在旧测定的有效期内、旧测定
说 430 bar，而峰值之后的新测定说 470 bar —— 用新测定算，比值从 97.7 %
降到 89.4 %，一次**真实的超压**被事后抹掉。

反方向同样不成立：新测定更低时，会把当时合规的操作追认成违规。
两个方向都不是"更准确"，而是**用事后才知道的数去改写当时的结论**。

所以规则是：取 ``ts <= 峰值时刻`` 的最后一次测定。若读数里**没有**任何
一次测定早于峰值，则峰值时刻根本没有生效的依据 —— 输出 ``indeterminate``，
并说明这不是"缺一个数"，而是"那一时段没有依据"（UIC Class VI 要求注入前
就取得破裂压力测定）。

"无法校核"是第四种结论
----------------------
缺井底压力、缺破裂压力、破裂压力读数非正、峰值时刻没有生效的测定 ——
这几种情况下正确输出是 ``indeterminate``，既不是 ``ok``（那会把"没查"
写成"查过了没问题"），也不是 ``exceeded``（那会触发一次没有依据的停注）。
它要求的是**补数据**，与"压力超了"要求的**停注**是两件不同的事。

每一条结论都带出处
------------------
包括 ``indeterminate`` 那几条。读者即使拿到的是一条"没算成"，也需要知道
本来要跟哪个数比 —— 否则"未通过校核"读起来像系统坏了，而不是像一条
待补的合规项。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable, Sequence

from ops.contracts import QUALITY_BAD, QUALITY_GOOD, QUALITY_SUSPECT, Reading

# ─── 法规上限 ───
#: 井底注入压力不得超过破裂压力的比例。
#:
#: 出处：40 CFR §146.68(b)（美国 UIC Class VI 注入井）。这是**法规值**，
#: 不是项目政策值，因此可以内置；但内置的同时必须写明辖区 —— 换成 EU
#: CCS Directive、中国《二氧化碳捕集、利用与封存项目环境影响评价技术导则》
#: 或任何地方性许可条件时，这个数与出处必须一起改。
CEILING_RATIO = 0.90

#: 可引用的出处（给读者看的形态）。报告正文与结论里只用这一句。
CEILING_CITATION = "40 CFR §146.68(b)（美国 UIC Class VI）"

#: 完整的出处说明，含换辖区时的处置要求。
#:
#: 「换辖区时须同时替换本常量与出处」是写给**维护者**的，不是写给监管读者的 ——
#: 所以它只出现在这个常量里，不进报告正文。正文用 ``CEILING_CITATION``。
CEILING_SOURCE = (
    CEILING_CITATION + "：注入层内井底注入压力不得超过该层破裂压力的 90%。"
    "换辖区时须同时替换本常量与出处"
)

#: 预警带宽度（占破裂压力的比例）。**项目政策选择，无标准依据** ——
#: 选 5 个百分点是因为它只用于"提前安排复核"，误报成本接近零。
#: 与 ``CEILING_RATIO`` 分开命名，是为了让读者一眼看出哪个数有法规依据、
#: 哪个数是本项目自己定的。
NEAR_BAND = 0.05

#: 破裂压力读数与井底压力读数之间允许的最大时间差。
#:
#: 破裂压力来自 LOT/XLOT，是**周期性**作业结果，与逐时井底压力天然不同步。
#: 超过这个跨度时比值仍然算得出来，但必须在结论里声明，否则"70 %"会被
#: 读成"以昨天的破裂压力核对了今天的压力"，而实际可能是两年前的 LOT。
MAX_SKEW_DAYS = 30

STATE_OK = "ok"
STATE_NEAR = "near"
STATE_EXCEEDED = "exceeded"
STATE_UNKNOWN = "indeterminate"

STATE_LABELS = {
    STATE_OK: "未超上限",
    STATE_NEAR: "接近上限",
    STATE_EXCEEDED: "超过上限",
    STATE_UNKNOWN: "无法校核",
}


@dataclass(frozen=True)
class CeilingCheck:
    """一次上限校核的结果。

    ``ratio`` 是**窗口内最大井底压力**与**最新破裂压力**之比。两个读数的
    时间戳都保留在 ``bhp_ts`` / ``frac_ts`` 里 —— 比值离开这两个时刻就无法
    复核，而"比值 0.70"单独看和"比值 0.70，用的是 2024 年的 LOT"是两回事。
    """

    state: str
    bhp: float | None = None
    bhp_ts: datetime | None = None
    frac_p: float | None = None
    frac_ts: datetime | None = None
    ratio: float | None = None
    headroom_bar: float | None = None
    #: 井底压力峰值与破裂压力读数的时间差（天）
    skew_days: float | None = None
    basis: list[str] = field(default_factory=list)
    next_step: str = ""
    ceiling_ratio: float = CEILING_RATIO
    source: str = CEILING_SOURCE
    #: 正文里引用的出处（``source`` 含维护提示，不适合进报告）
    citation: str = CEILING_CITATION

    def __post_init__(self) -> None:
        if self.state not in STATE_LABELS:
            raise ValueError(f"ceiling.state={self.state!r} 必须是 {tuple(STATE_LABELS)} 之一")
        # 有结论就必须有依据：一个没有 basis 的"未超上限"与"没查过"无法区分
        if not self.basis:
            raise ValueError(
                "ceiling.basis 不能为空。校核结论必须能说明用了哪两个读数、"
                "哪个时刻、按什么比例算的"
            )

    @property
    def label(self) -> str:
        return STATE_LABELS[self.state]

    @property
    def blocks(self) -> bool:
        """是否应当阻断报告提交。只有**确证超限**才阻断。

        ``indeterminate`` 不阻断：缺数据要补数据，而不是把人拦在审批门口。
        但它会以提示形式出现在冲突列表里，不会被静默忽略。
        """
        return self.state == STATE_EXCEEDED

    def to_dict(self) -> dict:
        return {
            "state": self.state,
            "label": self.label,
            "blocks": self.blocks,
            "bhp": self.bhp,
            "bhp_ts": self.bhp_ts.isoformat() if self.bhp_ts else None,
            "frac_p": self.frac_p,
            "frac_ts": self.frac_ts.isoformat() if self.frac_ts else None,
            "ratio": self.ratio,
            "headroom_bar": self.headroom_bar,
            "skew_days": self.skew_days,
            "ceiling_ratio": self.ceiling_ratio,
            "source": self.source,
            "citation": self.citation,
            "basis": list(self.basis),
            "next_step": self.next_step,
        }


def _usable(readings: Iterable[Reading], sensor: str) -> list[Reading]:
    """取某个测点**可用**的读数。

    ``quality=bad`` 的读数不参与校核：用它算出来的比值会被当成合规证据，
    而它连"这个数是多少"都不可信。``suspect`` 保留 —— 契约里它的语义是
    "可判定但要降级说明"，所以由调用方在结论里补一句，而不是在这里丢掉。
    """
    out = [r for r in readings if r.sensor == sensor and r.quality != QUALITY_BAD]
    out.sort(key=lambda r: r.ts)
    return out


def _frac_in_force(frac_rows: Sequence[Reading], at: datetime) -> Reading | None:
    """取 ``at`` 时刻**生效**的破裂压力测定（``ts <= at`` 的最后一条）。

    不是"最近一次" —— 理由见模块文档：取最后一条会让峰值之后的一次复测
    改写峰值当时的合规结论，两个方向都不成立。

    返回 ``None`` 表示 ``at`` 时刻没有任何已生效的测定，由调用方判
    ``indeterminate``。这里**不**退化成"用最早的一条顶上"：那等于默许
    用未来数据补过去结论。
    """
    prior = [r for r in frac_rows if r.ts <= at]
    return prior[-1] if prior else None


def _source_line(ratio: float) -> str:
    """出处行。``indeterminate`` 分支也要带 —— 读者需要知道本来要跟哪个数比。"""
    return f"依据：{CEILING_CITATION}；本次按破裂压力的 {ratio:.0%} 判定"


def check(
    readings: Sequence[Reading],
    *,
    ratio: float = CEILING_RATIO,
    near_band: float = NEAR_BAND,
    max_skew_days: float = MAX_SKEW_DAYS,
) -> CeilingCheck:
    """校核窗口内的注入压力上限。

    ``readings`` 是**整个窗口**的读数，不是末值 —— 内部自己取井底压力峰值。

    但 ``frac_p`` 是例外：它是周期性作业结果，最近一次 LOT 往往在窗口之前
    几十天甚至几百天。调用方必须把**最近一次实测值一并传进来**，否则本项
    只会得到 ``indeterminate``。传进来的时间戳必须是**测定作业的日期**，
    不是录入系统的日期 —— 录入日期晚于峰值会让本项判为"峰值时刻没有依据"。
    """
    if not 0.0 < ratio < 1.0:
        raise ValueError(
            f"ceiling.ratio={ratio} 必须在 (0, 1) 内。"
            "1.0 表示允许达到破裂压力本身，那已经会压开地层；"
            "0 会让每一次注入都判超限"
        )
    if not 0.0 <= near_band < 1.0 - ratio + 1e-9:
        raise ValueError(
            f"ceiling.near_band={near_band} 越界。预警带不得宽到覆盖整个量程，"
            f"否则每个合法读数都成了「接近上限」"
        )

    bhp_rows = _usable(readings, "bhp")
    frac_rows = _usable(readings, "frac_p")

    if not bhp_rows:
        return CeilingCheck(
            state=STATE_UNKNOWN,
            basis=[
                "没有可用的井底压力（bhp）读数，无法校核注入压力上限。",
                "**不能用井口压力代替**：井口与井底之间隔着液柱与摩阻，"
                "同一井口压力在不同井深对应完全不同的井底压力，"
                "用井口阈值代替会在浅井漏报、在深井误报。",
                _source_line(ratio),
            ],
            next_step=(
                "接入井下压力计，或由井口压力折算井底压力并在报告中标注为折算值；"
                "折算误差在注入量波动时可达十几 bar，而破裂压力余量往往只有几十 bar"
            ),
            ceiling_ratio=ratio,
        )

    if not frac_rows:
        return CeilingCheck(
            state=STATE_UNKNOWN,
            basis=[
                "没有可用的破裂压力（frac_p）读数，无法校核注入压力上限。",
                "破裂压力是地层属性，不能由任何运行参数推出。"
                "缺少实测值时，本项**未通过校核**，不等于通过。",
                _source_line(ratio),
            ],
            next_step=(
                "补做漏失试验（LOT）或微型压裂（XLOT）取得实测破裂压力；"
                "由压力梯度估算得到的值须在报告中标注为估算，且不得用于合规判定"
            ),
            ceiling_ratio=ratio,
        )

    # 峰值：合规判据是"任何时刻都不得超过"，末值会让班初的尖峰在交班时消失
    peak = max(bhp_rows, key=lambda r: r.value)
    frac = _frac_in_force(frac_rows, peak.ts)

    if frac is None:
        # 峰值时刻没有任何已生效的测定。退化成"用最早一条顶上"会默认
        # 允许用事后数据补当时结论，那正是本节要挡的那件事。
        first = frac_rows[0]
        gap_days = (first.ts - peak.ts).total_seconds() / 86400.0
        return CeilingCheck(
            state=STATE_UNKNOWN,
            bhp=peak.value,
            bhp_ts=peak.ts,
            basis=[
                f"井底压力峰值 {peak.value:g} bar（{peak.ts:%Y-%m-%d %H:%M}），"
                "但读数中没有任何一次破裂压力测定早于这个时刻 —— "
                "峰值时刻没有生效的破裂压力依据。",
                f"最早的一次测定是 {first.value:g} bar（{first.ts:%Y-%m-%d %H:%M}），"
                f"比峰值晚 {gap_days:.0f} 天。",
                "不能用晚于峰值的测定回溯判定该时段：那等于用事后才知道的数"
                "证明当时的操作合规。复测值更高时，一次真实超压会被抹掉；"
                "更低时，当时合规的操作会被追认为违规 —— 两个方向都不成立。",
                "按 UIC Class VI，注入前就必须取得破裂压力测定，所以"
                "「峰值时刻没有依据」本身是要说明的问题，不只是缺一个数。",
                _source_line(ratio),
            ],
            next_step=(
                "先确认 LOT/XLOT 的作业日期是否录入正确（日期录晚是常见录入错误）；"
                "若确实没有更早的测定，本期不得声明注入压力在上限以内，"
                "须补做 LOT 后重新校核"
            ),
            ceiling_ratio=ratio,
        )

    if frac.value <= 0:
        return CeilingCheck(
            state=STATE_UNKNOWN,
            bhp=peak.value,
            bhp_ts=peak.ts,
            frac_p=frac.value,
            frac_ts=frac.ts,
            basis=[
                f"破裂压力读数为 {frac.value:g} bar（≤ 0），不是有效的压力值。",
                "用它作分母得到的比值没有物理意义，因此本项不判定。",
                "最常见的原因是单位错误（把 MPa 当 bar 录入会得到原值的十分之一）"
                "或测点未标定 —— 两者都不会让系统报错，只会让比值凭空放大。",
                _source_line(ratio),
            ],
            next_step="核实破裂压力测点：是录入错误、单位错误（MPa 当 bar），还是测点未标定",
            ceiling_ratio=ratio,
        )

    if peak.value <= 0:
        return CeilingCheck(
            state=STATE_UNKNOWN,
            bhp=peak.value,
            bhp_ts=peak.ts,
            frac_p=frac.value,
            frac_ts=frac.ts,
            basis=[
                f"井底压力峰值为 {peak.value:g} bar（≤ 0）。注入井的井底压力必须为正，"
                "该读数说明测点异常或数据被错误换算。",
                _source_line(ratio),
            ],
            next_step="核实井底压力测点与单位；在修正前本项不判定",
            ceiling_ratio=ratio,
        )

    value = peak.value / frac.value
    headroom = frac.value * ratio - peak.value
    skew = (peak.ts - frac.ts).total_seconds() / 86400.0

    basis = [
        f"井底压力峰值 {peak.value:g} bar（{peak.ts:%Y-%m-%d %H:%M}），"
        f"破裂压力 {frac.value:g} bar（{frac.ts:%Y-%m-%d %H:%M}），"
        f"比值 {value:.1%}",
        f"上限 {ratio:.0%} 对应 {frac.value * ratio:.4g} bar，"
        + (
            f"余量 {headroom:.4g} bar"
            if headroom >= 0
            else f"已超出 {abs(headroom):.4g} bar"
        ),
        _source_line(ratio),
    ]

    # 多次测定必须交代清楚用了哪一次、为什么不用别的 —— 否则读者无法判断
    # 这个比值是不是挑出来的
    later = [r for r in frac_rows if r.ts > peak.ts]
    if len(frac_rows) > 1:
        basis.append(
            f"读数中共 {len(frac_rows)} 次破裂压力测定，采用峰值时刻之前最近的一次"
            f"（{frac.ts:%Y-%m-%d}）"
            + (
                f"；另有 {len(later)} 次晚于峰值，按「以峰值时刻生效的测定为准」未采用"
                if later
                else "；没有晚于峰值的测定"
            )
        )
        weakest = min(later, key=lambda r: r.value)
        if later and weakest.value < frac.value:
            basis.append(
                f"其中 {weakest.value:g} bar（{weakest.ts:%Y-%m-%d}）低于本次采用值，"
                f"低 {1 - weakest.value / frac.value:.1%} —— 若该测定成立，说明地层承压"
                "能力下降，那是 frac_p 测点阈值该报的事，不改变本期判定"
            )

    if frac.quality == QUALITY_SUSPECT:
        basis.append(
            "破裂压力读数的质量为「可疑」：该比值只能作为参考，"
            "提交前须核对 LOT/XLOT 原始记录"
        )
    if peak.quality == QUALITY_SUSPECT:
        basis.append(
            "井底压力峰值读数的质量为「可疑」：该比值只能作为参考，"
            "提交前须核对压力计校验记录"
        )

    if skew > max_skew_days:
        basis.append(
            f"注意：破裂压力读数与井底压力峰值相隔 {skew:.0f} 天，"
            "期间地层压力与注入历史都可能改变破裂压力。本比值应读作"
            "「按峰值时刻之前最近一次实测破裂压力的估算」，不是同期实测对照"
        )

    if value > ratio:
        return CeilingCheck(
            state=STATE_EXCEEDED,
            bhp=peak.value,
            bhp_ts=peak.ts,
            frac_p=frac.value,
            frac_ts=frac.ts,
            ratio=value,
            headroom_bar=headroom,
            skew_days=skew,
            basis=basis,
            next_step=(
                "立即降量或停注，把井底压力压回上限以下；同时核实破裂压力来源 —— "
                "若 frac_p 为梯度估算值而非 LOT/XLOT 实测值，本条应降级为提示并先补实测"
            ),
            ceiling_ratio=ratio,
        )

    if value > ratio - near_band:
        return CeilingCheck(
            state=STATE_NEAR,
            bhp=peak.value,
            bhp_ts=peak.ts,
            frac_p=frac.value,
            frac_ts=frac.ts,
            ratio=value,
            headroom_bar=headroom,
            skew_days=skew,
            basis=basis
            + [
                f"距上限还剩 {ratio - value:.1%}（占预警带宽的 "
                f"{(ratio - value) / near_band:.0%}）；"
                f"预警带宽 {near_band:.0%}（占破裂压力）为项目政策选择，无标准依据"
            ],
            next_step="安排复核：确认破裂压力取值是否仍适用，并核对配产是否需要下调",
            ceiling_ratio=ratio,
        )

    return CeilingCheck(
        state=STATE_OK,
        bhp=peak.value,
        bhp_ts=peak.ts,
        frac_p=frac.value,
        frac_ts=frac.ts,
        ratio=value,
        headroom_bar=headroom,
        skew_days=skew,
        basis=basis,
        next_step="无需处置；下个报告期重新校核",
        ceiling_ratio=ratio,
    )


def render(check_result: CeilingCheck) -> str:
    """一句话摘要（给报告与冲突列表用）。"""
    head = f"注入压力上限校核：{check_result.label}"
    if check_result.ratio is not None:
        head += f"（{check_result.ratio:.1%} / 上限 {check_result.ceiling_ratio:.0%}）"
    return head
