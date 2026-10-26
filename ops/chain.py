"""
链条拓扑与物理耦合
==================
捕集 → 运输 → 注入 → MVA 是一条**物料串联**的链：上游的量决定下游的量，
下游的阻力反过来决定上游的压力。这个单向+反馈的结构是本模块的全部内容。

为什么耦合关系要写成规则表而不是交给模型
--------------------------------------
"井口压力高、注入速率低"意味着注入性恶化 —— 这是渗流力学里的一条
确定结论（注入指数下降），不是语言模型的判断。把它交给模型，等于把一个
可以复核的推理变成一段无法复核的措辞。而且模型会在数据缺失时把这条
关系"补全"，于是归因看起来更有道理、实际更没有依据。

所以本模块的规则表有两个用途：
1. ``alarms`` 用它做归因（给出候选与排除理由）
2. ``supervisor`` 用它做跨 Agent 的方向核对

"耦合异常" 不等于 "矛盾"
------------------------
这是本模块最容易被误用的一点。``pipeline_dp`` 上升时，``pipeline_outlet_p``
按耦合表**预期**下降；如果实测上升，正确的结论是"存在另一个驱动源"，
而不是"数据错了"。把它标成矛盾会导致运维去查仪表，而真正的信息
（上游来量增加）被忽略。

因此本模块把这类情况命名为 ``coupling_deviation``（耦合偏离）并附上
"可能的额外驱动源"，与 ``contradiction``（同一量的两个值不一致，属于
必须查清的错误）严格分开。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ops.contracts import STAGES

# ─── 状态字面量（由 alarms 依据阈值与方向算出）───
STATE_HIGH = "high"       # 朝"坏"的方向越限（对 direction=low 的传感器，指偏低）
STATE_LOW = "low"         # 朝"好"的方向（未越限）
STATE_NORMAL = "normal"   # 未越限，且远离阈值
STATE_UNKNOWN = "unknown"  # 数据缺失 / 仪表不可信 / 阈值未配置
STATE_ANY = "any"         # 规则通配

#: 与方向无关的**偏离**语义。规则表统一用它，避免"high 对捕集率意味着偏低"
#: 这种反直觉表达。判定时 alarms 负责把方向折叠成偏离。
STATE_DEVIATED = "deviated"
STATE_OK = "ok"

STATES = (STATE_HIGH, STATE_LOW, STATE_NORMAL, STATE_UNKNOWN, STATE_ANY)


def upstream_of(stage: str) -> str | None:
    i = STAGES.index(stage)
    return STAGES[i - 1] if i > 0 else None


def downstream_of(stage: str) -> str | None:
    i = STAGES.index(stage)
    return STAGES[i + 1] if i + 1 < len(STAGES) else None


# ============================================================
# 一、预期耦合方向
# ============================================================
@dataclass(frozen=True)
class Coupling:
    """一条"上游变化 → 下游预期响应"。

    ``expected`` 描述下游**朝坏方向还是朝好方向**变化，取值 ``deviated`` /
    ``ok``。用这个语义而不是涨跌，是因为"管压降"对压力是朝好方向、
    对流量是朝坏方向，直接写涨跌会在不同传感器上互相矛盾。
    """

    trigger: str            # 触发变化的测点
    trigger_state: str      # deviated / ok
    response: str           # 预期响应的测点
    expected: str           # deviated / ok
    rationale: str
    #: 观测到反向时**可能的额外驱动源**（写在这里，冲突报告才能给出下一步）
    alternatives: tuple[str, ...] = ()


COUPLINGS: tuple[Coupling, ...] = (
    Coupling(
        trigger="capture_rate",
        trigger_state=STATE_DEVIATED,
        response="pipeline_outlet_p",
        expected=STATE_OK,
        rationale="捕集率下降 → 进入管道的 CO₂ 质量流量下降 → 沿线压降减小、"
        "出口压力降低（在注入端抽吸不变的条件下）",
        alternatives=("注入端同时降量，掩盖了来量下降", "管道积存 CO₂ 缓慢释放"),
    ),
    Coupling(
        trigger="capture_rate",
        trigger_state=STATE_DEVIATED,
        response="injection_rate",
        expected=STATE_DEVIATED,
        rationale="来量下降会沿链传导到注入量。若注入量不降，说明中间有缓冲"
        "（管存 / 储罐），此时压力信号比流量信号更早反映问题",
        alternatives=("管道管存缓冲", "注入站从其他气源补充"),
    ),
    Coupling(
        trigger="pipeline_dp",
        trigger_state=STATE_DEVIATED,
        response="pipeline_outlet_p",
        expected=STATE_DEVIATED,
        rationale="管段阻力上升 → 相同入口压力下出口压力下降（或需更高入口压力"
        "维持出口压力）",
        alternatives=("入口压力同时被提高，掩盖了阻力上升", "水合物局部节流"),
    ),
    Coupling(
        trigger="pipeline_outlet_p",
        trigger_state=STATE_DEVIATED,
        response="wellhead_p",
        expected=STATE_DEVIATED,
        rationale="注入站入口压力高 → 井口压力高（中间只有节流与增压设备）",
        alternatives=("节流阀主动降压", "增压泵降速运行"),
    ),
    Coupling(
        trigger="injection_rate",
        trigger_state=STATE_DEVIATED,
        response="reservoir_p",
        expected=STATE_OK,
        rationale="注入量下降 → 储层压力增长放缓或回落（滞后数小时到数天，"
        "取决于储层压缩性与边界）",
        alternatives=("邻井同时增注", "储层压力本身处于自然恢复期"),
    ),
    Coupling(
        trigger="wellhead_p",
        trigger_state=STATE_DEVIATED,
        response="reservoir_p",
        expected=STATE_DEVIATED,
        rationale="井口压力高而注入量不增，通常意味着井底到储层的通路受阻；"
        "若储层压力同步上升，则是储层整体超压而非近井堵塞",
        alternatives=("油管结垢造成附加压降", "封隔器失封导致压力串通"),
    ),
    Coupling(
        trigger="annulus_p",
        trigger_state=STATE_DEVIATED,
        response="wellhead_p",
        expected=STATE_ANY,
        rationale="环空带压与油管压力**没有固定方向关系**：热膨胀型环空压力"
        "与油管压力同向，窜流型则与油管压力反向。因此这条耦合只用于提示"
        "「需要区分热效应与窜流」，不用于判定",
        alternatives=(),
    ),
    Coupling(
        trigger="reservoir_p",
        trigger_state=STATE_DEVIATED,
        response="soil_co2_flux",
        expected=STATE_ANY,
        rationale="储层超压会提高泄漏驱动力，但通量是否上升取决于盖层完整性"
        "与运移路径。储层压力高 + 通量正常是一个**可以接受**的组合，"
        "不能据此排除泄漏，也不能据此认定泄漏",
        alternatives=(),
    ),
)

COUPLING_INDEX: dict[tuple[str, str], Coupling] = {
    (c.trigger, c.response): c for c in COUPLINGS
}


# ============================================================
# 二、归因规则
# ============================================================
@dataclass(frozen=True)
class CauseRule:
    """一条"哪些读数同向 → 指向哪个原因"的规则。

    ``requires`` 是全部命中的理想条件；``min_hits`` 允许部分命中时仍给出
    候选，但强度降为 ``partial``。为什么允许部分命中：运维现场经常缺测点，
    要求全部命中会让最需要归因的时候（数据不全）恰好给不出任何候选。

    ``excludes`` 是排除条件。它比 ``requires`` 更重要：一条原因被明确排除，
    运维才不会去查一个不存在的故障。排除条件同样要写理由。
    """

    key: str
    cause: str
    stage: str
    requires: tuple[tuple[str, str], ...]
    rationale: str
    actions: tuple[str, ...]
    min_hits: int = 0
    excludes: tuple[tuple[str, str], ...] = ()
    #: 排除条件的理由（与 excludes 一一对应）
    exclude_reasons: tuple[str, ...] = ()
    #: 该原因成立时，报告中必须提醒的"不能因此排除什么"
    caveat: str = ""

    def __post_init__(self) -> None:
        if self.excludes and len(self.exclude_reasons) != len(self.excludes):
            raise ValueError(
                f"规则 {self.key}: excludes 有 {len(self.excludes)} 条，"
                f"exclude_reasons 有 {len(self.exclude_reasons)} 条。"
                "缺理由的排除等于凭直觉排除"
            )
        if self.min_hits and self.min_hits < 2:
            raise ValueError(
                f"规则 {self.key}: min_hits={self.min_hits}。"
                "只命中一条读数不能构成归因，那是猜测；单条读数走 notice 即可"
            )
        if self.min_hits > len(self.requires):
            raise ValueError(
                f"规则 {self.key}: min_hits={self.min_hits} 大于 requires 条数 "
                f"{len(self.requires)}，这条规则永远不会命中"
            )


CAUSE_RULES: tuple[CauseRule, ...] = (
    # ── 注入 ──
    CauseRule(
        key="injectivity_loss",
        cause="注入层吸水能力下降（近井堵塞 / 结垢 / 盐析 / 水锁）",
        stage="injection",
        requires=(
            ("wellhead_p", STATE_DEVIATED),
            ("injection_rate", STATE_DEVIATED),
            ("injectivity_index", STATE_DEVIATED),
        ),
        min_hits=2,
        rationale="注入指数 = 注入速率 /（井底压力 − 储层静压）。井口压力升高的"
        "同时注入量与注入指数同时下降，意味着单位压差能推入的量变少，"
        "这是近井渗流条件恶化的定义性特征，而不是「推得多所以压力高」",
        actions=("injection_check_skin", "injection_reduce_rate", "injection_sample_water"),
        caveat="注入指数下降也可能由储层压力上升造成（分母变大）。"
        "若 reservoir_p 同步上升，应先按储层超压处置，不要直接酸化",
    ),
    CauseRule(
        key="reservoir_overpressure",
        cause="储层整体超压（注入总量超过储层容纳能力）",
        stage="injection",
        requires=(
            ("wellhead_p", STATE_DEVIATED),
            ("reservoir_p", STATE_DEVIATED),
            ("injection_rate", STATE_OK),
        ),
        min_hits=2,
        rationale="井口压力与储层压力同步上升、而注入量并未下降，说明压差仍能"
        "推动同样的流量，阻力不是集中在近井，而是整个储层压力水平抬升",
        actions=("injection_reduce_rate", "injection_review_alloc"),
        caveat="储层压力测点若在观测井而非注入井附近，滞后可达数天，"
        "此时两者的「同步」可能只是巧合",
    ),
    CauseRule(
        key="wellbore_integrity",
        cause="井筒完整性失效嫌疑（环空带压）",
        stage="injection",
        requires=(
            ("annulus_p", STATE_DEVIATED),
            ("wellhead_p", STATE_DEVIATED),
        ),
        min_hits=2,
        excludes=(("annulus_p", STATE_OK),),
        exclude_reasons=(
            "环空压力未越限，则不存在「带压」这一前提，"
            "此时讨论完整性失效属于无对象推理",
        ),
        rationale="环空压力持续上升，同时油管压力也处于高位，是窜流型带压的"
        "典型组合。需与热膨胀区分：关井后环空压力自行回落即为热效应",
        actions=("wellbore_shutin_test", "wellbore_notify_integrity", "wellbore_prepare_workover"),
        caveat="本判定只能提出「需要做关井测试」，不能给出完整性结论。"
        "完整性结论需要 MIT/噪声测井/温度测井等实测手段",
    ),
    CauseRule(
        key="annulus_thermal",
        cause="环空带压（热膨胀型，非窜流）",
        stage="injection",
        requires=(
            ("annulus_p", STATE_DEVIATED),
            ("wellhead_p", STATE_OK),
        ),
        min_hits=2,
        rationale="环空压力上升而油管压力正常，符合注入热流体后环空流体受热"
        "膨胀的特征。这一类不需要修井，但需要确认环空泄压阀可用",
        actions=("wellbore_annulus_bleed_check",),
    ),
    # ── 运输 ──
    CauseRule(
        key="pipeline_restriction",
        cause="管道阻力上升（水合物 / 杂质沉积 / 阀门节流）",
        stage="transport",
        requires=(
            ("pipeline_dp", STATE_DEVIATED),
            ("pipeline_outlet_p", STATE_DEVIATED),
        ),
        min_hits=2,
        excludes=(("pipeline_dp", STATE_OK),),
        exclude_reasons=(
            "压差未越限时，出口压力高只能由入口压力高解释，"
            "指向的是上游供气而不是管道本身",
        ),
        rationale="压差是管道阻力的直接观测量。压差与出口压力同时升高，"
        "说明入口压力被抬得更高才维持住出口压力，阻力在管内而不在管外",
        actions=("transport_pigging", "transport_dewpoint_check", "transport_throttle_inspect"),
    ),
    CauseRule(
        key="upstream_overfeed",
        cause="上游来量超过下游注入能力（压力沿链正向传导）",
        stage="transport",
        requires=(
            ("pipeline_outlet_p", STATE_DEVIATED),
            ("wellhead_p", STATE_DEVIATED),
            ("pipeline_dp", STATE_OK),
        ),
        min_hits=2,
        rationale="出口压力与井口压力同时升高而压差正常，说明管道本身没有额外"
        "阻力，压力是「推」过来的。此时应在上游降量或在下游增注能力",
        actions=("transport_coordinate_capture", "injection_add_capacity"),
    ),
    CauseRule(
        key="purity_corrosion",
        cause="杂质与水含量同时偏高（碳酸腐蚀风险）",
        stage="transport",
        requires=(
            ("co2_purity", STATE_DEVIATED),
            ("h2o_dewpoint", STATE_DEVIATED),
        ),
        min_hits=2,
        rationale="CO₂ 纯度下降意味着非凝析气（N₂/Ar/O₂/H₂S）比例上升，"
        "水露点升高意味着游离水更可能出现。两者叠加才是腐蚀条件，"
        "单独任一项都不足以判定",
        actions=("transport_purity_check", "transport_dehydration_check", "transport_corrosion_coupon"),
        caveat="腐蚀速率取决于材质与温度，本判定只指出条件成立，"
        "不给出腐蚀速率",
    ),
    # ── 捕集 ──
    CauseRule(
        key="solvent_degradation",
        cause="溶剂降解 / 再沸器结垢（捕集率与能耗同向恶化）",
        stage="capture",
        requires=(
            ("capture_rate", STATE_DEVIATED),
            ("regenerator_duty", STATE_DEVIATED),
            ("solvent_loss", STATE_DEVIATED),
        ),
        min_hits=2,
        rationale="捕集率下降同时再生能耗与溶剂损耗上升，是溶剂劣化或换热面"
        "结垢的典型组合。若只有捕集率下降而能耗正常，则应先查入口烟气条件",
        actions=("capture_solvent_analysis", "capture_reboiler_inspect"),
    ),
    CauseRule(
        key="inlet_condition_change",
        cause="入口烟气条件变化或分析仪漂移（捕集率下降但能耗正常）",
        stage="capture",
        requires=(
            ("capture_rate", STATE_DEVIATED),
            ("regenerator_duty", STATE_OK),
        ),
        min_hits=2,
        rationale="捕集率是进出口浓度/流量的比值。比值下降而单位能耗不变，"
        "说明系统本身效率没变，变的是入口条件或测量基准",
        actions=("capture_inlet_analyzer_check", "capture_gas_flow_recheck"),
        caveat="分析仪漂移与真实入口变化从数据上无法区分，必须用标气复测",
    ),
    # ── MVA ──
    CauseRule(
        key="shallow_intrusion",
        cause="浅部 CO₂ 侵入嫌疑（土壤气 + 地下水同向异常）",
        stage="mva",
        requires=(
            ("soil_co2_flux", STATE_DEVIATED),
            ("gw_ph", STATE_DEVIATED),
            ("gw_conductivity", STATE_DEVIATED),
        ),
        min_hits=2,
        rationale="CO₂ 溶解生成碳酸：pH 下降与电导率上升必然同时发生。"
        "土壤通量上升则给出气相证据。三者同向才构成「浅部侵入」的完整证据链",
        actions=("mva_resample", "mva_notify_regulator", "mva_extend_survey"),
        caveat="土壤通量受温度、湿度、气压与土壤呼吸影响极大，"
        "单次高值不构成泄漏证据；必须与同点位基线及空白点对比",
    ),
    CauseRule(
        key="sensor_drift",
        cause="仪表漂移嫌疑（单项异常，缺少关联证据）",
        stage="mva",
        requires=(
            ("gw_ph", STATE_DEVIATED),
            ("gw_conductivity", STATE_OK),
        ),
        min_hits=2,
        rationale="pH 下降而电导率不变，与 CO₂ 溶解的物理后果不一致。"
        "更可能是探头漂移或取样污染。这类异常的正确处置是复测，不是上报",
        actions=("mva_recalibrate_probe", "mva_resample"),
        caveat="电导率测点缺失时本条不成立，此时应按「证据不足」处理，"
        "而不是「排除侵入」",
    ),
    CauseRule(
        key="plume_expansion",
        cause="羽流正常扩展（储层压力与形变同向，无浅部证据）",
        stage="mva",
        requires=(
            ("reservoir_p", STATE_DEVIATED),
            ("surface_heave", STATE_DEVIATED),
        ),
        min_hits=2,
        rationale="注入引起的储层压力抬升与地表微抬升是**预期**响应，"
        "在未出现浅部异常时属于正常扩展，不构成泄漏证据",
        actions=("mva_trend_confirm",),
        caveat="本条是「符合预期」的判定，不是「安全」的判定。"
        "形变速率是否可接受取决于许可条件，不是本模块能给的结论",
    ),
)

RULE_INDEX: dict[str, CauseRule] = {r.key: r for r in CAUSE_RULES}


# ============================================================
# 三、处置动作字典
# ============================================================
#: 动作必须写成"可执行的下一步"，而不是"加强监控"这种无法验收的说法。
#: ``owner`` 是执行角色，``sla_minutes`` 是期望响应时长 —— 后者是运营政策，
#: 因此**没有内置默认值**：不同项目的值班制度不同，写死一个 30 会让
#: 报告里的"应在 30 分钟内响应"看起来像标准要求。
ACTIONS: dict[str, dict[str, str]] = {
    "injection_check_skin": {
        "action": "调取该井最近一次压降试井/注入剖面，计算表皮系数变化",
        "owner": "油藏工程师",
    },
    "injection_reduce_rate": {
        "action": "按当前井底压力上限反算允许注入量，阶梯降量至压力回落",
        "owner": "中控操作",
    },
    "injection_sample_water": {
        "action": "井口取样做水质全分析（Ca/Mg/Ba/SO4），判断结垢倾向",
        "owner": "化验",
    },
    "injection_review_alloc": {
        "action": "复核全井组注入配产，确认是否超出储层许可注入量",
        "owner": "油藏工程师",
    },
    "injection_add_capacity": {
        "action": "评估启用备用注入井或提高单井注入能力",
        "owner": "生产调度",
    },
    "wellbore_shutin_test": {
        "action": "执行关井稳压测试：关井后记录环空压力衰减曲线，区分热效应与窜流",
        "owner": "完井工程师",
    },
    "wellbore_notify_integrity": {
        "action": "通知井完整性责任人，启动完整性事件记录",
        "owner": "井完整性责任人",
    },
    "wellbore_prepare_workover": {
        "action": "准备 MIT/噪声测井方案，评估修井必要性",
        "owner": "完井工程师",
    },
    "wellbore_annulus_bleed_check": {
        "action": "检查环空泄压阀可用性，按程序泄压并记录恢复速率",
        "owner": "中控操作",
    },
    "transport_pigging": {
        "action": "安排清管作业，作业前后对比同流量下的压差",
        "owner": "管道运维",
    },
    "transport_dewpoint_check": {
        "action": "复核脱水单元运行参数，在线水露点与取样比对",
        "owner": "管道运维",
    },
    "transport_throttle_inspect": {
        "action": "检查沿线截断阀/节流阀开度记录，排除误操作",
        "owner": "中控操作",
    },
    "transport_coordinate_capture": {
        "action": "与捕集单元协调降量，使来量与注入能力匹配",
        "owner": "生产调度",
    },
    "transport_purity_check": {
        "action": "复核在线色谱，与实验室取样比对，确认非凝析气比例",
        "owner": "化验",
    },
    "transport_dehydration_check": {
        "action": "检查脱水装置（TEG/分子筛）运行状态与再生温度",
        "owner": "管道运维",
    },
    "transport_corrosion_coupon": {
        "action": "调取腐蚀挂片/探针数据，评估腐蚀速率趋势",
        "owner": "腐蚀工程师",
    },
    "capture_solvent_analysis": {
        "action": "取贫/富液样做胺浓度、热稳定盐与降解产物分析",
        "owner": "工艺工程师",
    },
    "capture_reboiler_inspect": {
        "action": "核算再沸器传热系数，与原始设计值对比判断结垢程度",
        "owner": "工艺工程师",
    },
    "capture_inlet_analyzer_check": {
        "action": "用标气校验入口 CO₂ 分析仪，同时核对烟气流量计",
        "owner": "仪表",
    },
    "capture_gas_flow_recheck": {
        "action": "复核烟气流量测量（皮托管/超声），检查取压孔堵塞",
        "owner": "仪表",
    },
    "mva_resample": {
        "action": "24 小时内同点位复采样，同步测空白点与基线点",
        "owner": "环境监测",
    },
    "mva_recalibrate_probe": {
        "action": "用标准缓冲液标定 pH 探头，比对实验室测定值",
        "owner": "仪表",
    },
    "mva_notify_regulator": {
        "action": "按许可条件要求向监管方报告异常监测结果",
        "owner": "HSE",
    },
    "mva_extend_survey": {
        "action": "加密监测网（扩大土壤气与地下水采样点密度）",
        "owner": "环境监测",
    },
    "mva_trend_confirm": {
        "action": "与历史趋势比对，确认处于预期包络内并记录",
        "owner": "环境监测",
    },
    "verify_instrument": {
        "action": "检查该测点供电、通信与就地显示，确认是断线还是漂移",
        "owner": "仪表",
    },
    "configure_threshold": {
        "action": "补齐该测点阈值并写明出处（标准条款 / 设计文件 / 政策选择）",
        "owner": "工艺工程师",
    },
}

#: 分级对应的通用处置（当原因未明确时使用）。
#: 注意 ``notice`` 与 ``normal`` 的动作刻意不同：notice 要求"下个班次复核"，
#: 这是让"看起来没事但有点偏"的读数有人再看一眼，而不是就此消失。
#:
#: 这里的值是**完整动作描述**，不是 ``ACTIONS`` 的 key —— 因为它描述的是
#: "这个等级该做什么"，与"这个原因该做什么"是两个维度。展开成条目由
#: ``severity_actions()`` 负责，key 前缀 ``severity:`` 使它在报告里可回代。
SEVERITY_ACTIONS: dict[str, tuple[str, ...]] = {
    "normal": (),
    "notice": ("下个班次复核该测点趋势，确认是否继续偏离",),
    "warn": ("通知值班工程师，2 小时内确认读数真实性（含仪表自检）",),
    "alert": ("立即响应：按原因对应的处置执行，必要时阶梯降量",),
    "critical": ("立即停输/停注并启动应急预案，同时通知 HSE 与生产调度",),
    "indeterminate": ("确认仪表与通信状态；阈值未配置时先补齐阈值",),
}


def severity_actions(severity: str) -> list[dict[str, str]]:
    """把分级对应的通用处置展开成可执行条目。"""
    return [
        {
            "key": f"severity:{severity}",
            "action": text,
            "owner": "值班工程师" if severity in ("warn", "alert") else "中控操作",
        }
        for text in SEVERITY_ACTIONS.get(severity, ())
    ]


def actions_for(keys: tuple[str, ...] | list[str]) -> list[dict[str, str]]:
    """把动作 key 展开成可执行条目；未知 key 直接抛错。

    未知 key 静默跳过会让归因说"建议处置"却列不出任何动作 —— 报告读起来
    像是"已给出建议"，实际什么都没给。
    """
    out: list[dict[str, str]] = []
    for k in keys:
        if k not in ACTIONS:
            raise ValueError(
                f"未定义的处置动作 {k!r}。请在 chain.ACTIONS 里补齐，"
                "否则归因会输出一条空建议"
            )
        out.append({"key": k, **ACTIONS[k]})
    return out
