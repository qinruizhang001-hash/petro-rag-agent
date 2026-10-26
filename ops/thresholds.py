"""
测点清单与阈值注册表
====================
本模块只做两件事：**声明有哪些测点**、**从外部装载阈值**。

刻意不做的事：内置任何阈值数字。
--------------------------------
这不是洁癖。CCUS 的压力限值来自三个互不相同的地方 —— 管道设计规范
（DNV-RP-F104 / ISO 27913）、项目设计文件、运营方的风险接受准则。
同一个 80 bar，在一份项目里是"设计压力 60%"，在另一份项目里是"规范要求
的致密相下限"。代码里写死一个数，等于把某一份项目文件的结论伪装成
通用物理常数，而读者看不出这件事。

所以：``DEFAULT_SENSORS`` 里的量纲、方向、测量基准是**描述性**的（它们是
工程共识，例如"井口压力越高越危险"），可以内置；阈值是**规范性**的，
必须外部给，并且必须带出处。

MVA 不是一个传感器
------------------
MVA（Monitoring, Verification and Accounting，监测、核查与核算）是
NETL 定义的一整套体系，覆盖深部（储层压力、地震）、浅部（地下水、土壤气）、
近地表（通量、形变）与井筒完整性四层监测。它的首要目的**不是报警**，
而是"证明没有泄漏 + 量化封存量"。

把它当成一个叫 MVA 的测点是常见的建模错误：那样只会得到一个没有量纲的
指标，超阈值时无法归因到任何具体层位。所以这里把 MVA 展开成四个测点，
并在 ``SensorSpec.basis`` 里写明各自能证明什么、不能证明什么。
"""
from __future__ import annotations

import json
import os
from typing import Any, Mapping

from ops.contracts import (
    DIRECTION_HIGH,
    DIRECTION_LOW,
    SensorSpec,
    Threshold,
)

# ─── 测点清单（描述性，可内置；阈值见文件头说明）───
DEFAULT_SENSORS: tuple[SensorSpec, ...] = (
    # ── 捕集 ──
    SensorSpec(
        key="capture_rate",
        label="捕集率",
        unit="%",
        stage="capture",
        direction=DIRECTION_LOW,
        basis="捕集单元进出口 CO₂ 流量差 / 入口 CO₂ 流量，按小时均值",
    ),
    SensorSpec(
        key="regenerator_duty",
        label="再生能耗",
        unit="GJ/tCO₂",
        stage="capture",
        direction=DIRECTION_HIGH,
        basis="再沸器蒸汽热量 / 捕集 CO₂ 质量，含公用工程分摊",
    ),
    SensorSpec(
        key="solvent_loss",
        label="溶剂损耗",
        unit="kg/tCO₂",
        stage="capture",
        direction=DIRECTION_HIGH,
        basis="补充溶剂量 / 捕集 CO₂ 质量，按周累计折算",
    ),
    # ── 运输 ──
    SensorSpec(
        key="pipeline_dp",
        label="管段压差",
        unit="bar",
        stage="transport",
        direction=DIRECTION_HIGH,
        basis="同一直管段进出口表压差，扣除高程差静水项",
    ),
    SensorSpec(
        key="pipeline_outlet_p",
        label="管道出口压力",
        unit="bar",
        stage="transport",
        direction=DIRECTION_HIGH,
        basis="进注入站前的表压。低于致密相下限会导致两相流，"
        "但下限是设计参数不是通用常数，因此不在此内置",
    ),
    SensorSpec(
        key="co2_purity",
        label="CO₂ 纯度",
        unit="%",
        stage="transport",
        direction=DIRECTION_LOW,
        basis="在线气相色谱，体积分数，扣除 N₂/Ar 后的 CO₂ 占比",
    ),
    SensorSpec(
        key="h2o_dewpoint",
        label="水露点",
        unit="°C",
        stage="transport",
        direction=DIRECTION_HIGH,
        basis="露点仪实测。与水合物生成、碳酸腐蚀直接相关，"
        "但多高算高取决于运行压力与杂质组成，须按项目给",
    ),
    # ── 注入 ──
    SensorSpec(
        key="wellhead_p",
        label="井口压力",
        unit="bar",
        stage="injection",
        direction=DIRECTION_HIGH,
        basis="注入井口表压，井口装置下游、油管头上游",
    ),
    SensorSpec(
        key="injection_rate",
        label="注入速率",
        unit="t/d",
        stage="injection",
        direction=DIRECTION_LOW,
        basis="单井日注入质量，由井口质量流量计累计差得出",
    ),
    SensorSpec(
        key="injectivity_index",
        label="注入指数",
        unit="t/(d·bar)",
        stage="injection",
        direction=DIRECTION_LOW,
        basis="注入速率 /（井底压力 − 储层静压）。比单看压力更能区分"
        "「推不动」与「不想推」：前者指数下降，后者指数不变",
    ),
    SensorSpec(
        key="annulus_p",
        label="环空压力",
        unit="bar",
        stage="injection",
        direction=DIRECTION_HIGH,
        basis="油套环空表压。持续上升是井筒完整性失效的早期信号，"
        "但需先排除热膨胀（关井后回落即为热效应）",
    ),
    # ── MVA（四层监测，见文件头）──
    SensorSpec(
        key="reservoir_p",
        label="储层监测压力",
        unit="bar",
        stage="mva",
        direction=DIRECTION_HIGH,
        basis="观测井井底压力。深部层，反映羽流扩展与压力前缘，"
        "不能证明浅部没有泄漏",
    ),
    SensorSpec(
        key="soil_co2_flux",
        label="土壤 CO₂ 通量",
        unit="g/(m²·d)",
        stage="mva",
        direction=DIRECTION_HIGH,
        basis="累积通量箱法，单点 30 min 积分。受土壤呼吸与气象强烈影响，"
        "必须与同点位基线对比，绝对阈值几乎不可迁移",
    ),
    SensorSpec(
        key="gw_ph",
        label="浅层地下水 pH",
        # pH 无量纲；写 "-" 而不是空串，因为空单位会被契约层拒绝（见 SensorSpec）
        unit="-",
        stage="mva",
        direction=DIRECTION_LOW,
        basis="监测井取样实验室测定（现场探头漂移大，不单独用于判定）",
    ),
    SensorSpec(
        key="gw_conductivity",
        label="浅层地下水电导率",
        unit="uS/cm",
        stage="mva",
        direction=DIRECTION_HIGH,
        basis="同上。CO₂ 溶解生成碳酸会同时压低 pH、抬高电导率，"
        "两者同向变化才构成证据，单看一个容易误判",
    ),
    SensorSpec(
        key="surface_heave",
        label="地表形变速率",
        unit="mm/yr",
        stage="mva",
        direction=DIRECTION_HIGH,
        basis="InSAR 或水准测量。取绝对值：抬升（注入）与沉降（压力衰竭）"
        "都是异常，方向信息在原始时序里保留",
    ),
)

SENSORS: dict[str, SensorSpec] = {s.key: s for s in DEFAULT_SENSORS}

# ─── 环境变量前缀 ───
ENV_PREFIX = "OPS_TH_"
ENV_CONFIG_PATH = "OPS_THRESHOLDS_PATH"

#: 默认配置文件位置（相对项目根）
DEFAULT_CONFIG_RELPATH = os.path.join("config", "ops_thresholds.json")

#: env 覆盖时 source 的追加标记
ENV_SOURCE_SUFFIX = "（env 覆盖）"


def sensor(key: str) -> SensorSpec:
    """按名字取测点；未知名直接抛错而不是返回 None。

    拼错一个测点名如果只返回 None，结果是"这个测点没有阈值 → 不判定 →
    看板正常"。一次拼写错误就变成一处永久盲区，且没有任何输出会提示。
    """
    try:
        return SENSORS[key]
    except KeyError:
        raise ValueError(
            f"未知测点 {key!r}。已注册：{sorted(SENSORS)}。"
            "新增测点请在 DEFAULT_SENSORS 里声明量纲与方向，"
            "否则它的读数永远无法被判定"
        ) from None


def _read_env_float(environ: Mapping[str, str], name: str) -> float | None:
    raw = environ.get(name)
    if raw is None or not raw.strip():
        return None
    try:
        return float(raw)
    except ValueError:
        raise ValueError(
            f"{name}={raw!r} 不是数字。环境变量写错必须硬失败："
            "静默忽略会让阈值悄悄回到配置文件里的旧值，而报告仍声称用的是 env 值"
        ) from None


class ThresholdRegistry:
    """测点 → 阈值的只读映射。

    未配置的测点 ``get()`` 返回 ``None``，由调用方产出 ``indeterminate``。
    这里**不**提供"默认阈值"兜底 —— 兜底值就是无出处的魔法数。
    """

    __slots__ = ("_items", "_origin")

    def __init__(self, items: dict[str, Threshold], origin: str) -> None:
        self._items = dict(items)
        self._origin = origin

    @property
    def origin(self) -> str:
        """这份注册表从哪来（配置文件路径 / "内置空表" / "测试注入"）。"""
        return self._origin

    def get(self, key: str) -> Threshold | None:
        return self._items.get(key)

    def require(self, key: str) -> Threshold:
        th = self._items.get(key)
        if th is None:
            raise KeyError(
                f"测点 {key!r} 没有配置阈值（注册表来源：{self._origin}）。"
                "需要判定时必须先配置；不要在这里兜底一个默认值"
            )
        return th

    def configured(self) -> list[str]:
        return sorted(self._items)

    def missing(self) -> list[str]:
        """已声明但没配阈值的测点 —— 这些是**已知盲区**，报告里必须列出。"""
        return sorted(k for k in SENSORS if k not in self._items)

    def to_dict(self) -> dict[str, Any]:
        return {
            "origin": self._origin,
            "configured": {k: v.to_dict() for k, v in sorted(self._items.items())},
            "missing": self.missing(),
        }


def build_registry(
    data: Mapping[str, Mapping[str, Any]],
    *,
    origin: str = "调用方注入",
) -> ThresholdRegistry:
    """从已解析的映射构建注册表（便于测试与 API 传参）。"""
    items: dict[str, Threshold] = {}
    problems: list[str] = []
    for key, raw in data.items():
        if key.startswith("_"):
            # 下划线开头的键是配置文件里的注释（JSON 不支持注释），
            # 跳过而不是报"未知测点" —— 否则想加一句说明就会让整份配置加载失败
            continue
        if key not in SENSORS:
            problems.append(f"未知测点 {key!r}")
            continue
        spec = SENSORS[key]
        try:
            th = Threshold(
                warn=raw.get("warn"),
                alert=raw.get("alert"),
                critical=raw.get("critical"),
                source=str(raw.get("source") or ""),
                note=str(raw.get("note") or ""),
            )
            th.validate_for(spec)
        except ValueError as exc:
            # 必须带测点名。注册表有十几个测点时，"source 不能为空"
            # 这条消息本身无法定位 —— 而它恰恰是最常出现的一条
            problems.append(f"[{key}] {exc}")
            continue
        items[key] = th
    if problems:
        # 一次性报全部问题：逐个报会让人改一轮跑一轮
        raise ValueError(
            "阈值注册表有问题，拒绝加载：\n  - " + "\n  - ".join(problems)
        )
    return ThresholdRegistry(items, origin)


def apply_env_overrides(
    items: dict[str, Threshold],
    environ: Mapping[str, str] | None = None,
) -> dict[str, Threshold]:
    """把 ``OPS_TH_<SENSOR>_<LEVEL>`` 形式的覆盖应用到阈值上。

    覆盖后会改写 ``source``，追加 ``（env 覆盖）``。这一步不能省：报告里
    引用的阈值必须与实际判定用的阈值同源，否则"报告说 80 bar 告警、
    实际按 90 bar 判定"这种错无法从任何单个文件里看出来。
    """
    env = os.environ if environ is None else environ
    out = dict(items)
    for key, th in list(items.items()):
        spec = SENSORS[key]
        changed = False
        values: dict[str, float | None] = {}
        for level in ("warn", "alert", "critical"):
            name = f"{ENV_PREFIX}{key.upper()}_{level.upper()}"
            got = _read_env_float(env, name)
            if got is not None:
                values[level] = got
                changed = True
        if not changed:
            continue
        merged = {
            "warn": values.get("warn", th.warn),
            "alert": values.get("alert", th.alert),
            "critical": values.get("critical", th.critical),
        }
        new = Threshold(
            warn=merged["warn"],
            alert=merged["alert"],
            critical=merged["critical"],
            source=th.source + ENV_SOURCE_SUFFIX,
            note=th.note,
        )
        new.validate_for(spec)
        out[key] = new
    return out


def load_registry(
    path: str | None = None,
    environ: Mapping[str, str] | None = None,
    *,
    root: str | None = None,
) -> ThresholdRegistry:
    """装载阈值：配置文件 → env 覆盖。

    文件不存在**不是错误**，返回空表并在 ``origin`` 里写明"文件不存在"。
    理由是运维系统常常先上线、后配阈值；此时每条告警都应是 ``indeterminate``
    并明确说"阈值未配置"，而不是启动失败、也不是全部 normal。
    """
    env = os.environ if environ is None else environ
    root = root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if path is None:
        path = env.get(ENV_CONFIG_PATH) or os.path.join(root, DEFAULT_CONFIG_RELPATH)

    if not os.path.exists(path):
        origin = f"内置空表（配置文件不存在：{path}）"
        return ThresholdRegistry({}, origin)

    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except json.JSONDecodeError as exc:
        raise ValueError(f"阈值配置文件 {path} 不是合法 JSON：{exc}") from None

    if not isinstance(data, dict):
        raise ValueError(
            f"阈值配置文件 {path} 顶层必须是对象（测点名 → 阈值），"
            f"实际是 {type(data).__name__}"
        )

    registry = build_registry(data, origin=path)
    overridden = apply_env_overrides(
        {k: v for k, v in ((key, registry.get(key)) for key in registry.configured()) if v},
        env,
    )
    if overridden != {k: registry.get(k) for k in registry.configured()}:
        return ThresholdRegistry(overridden, path + " + env 覆盖")
    return registry


def registry_from_env(environ: Mapping[str, str] | None = None) -> ThresholdRegistry:
    """只从环境变量建表（每条阈值仍须自带 source，故仅供内部诊断用）。"""
    return load_registry(environ=environ)
