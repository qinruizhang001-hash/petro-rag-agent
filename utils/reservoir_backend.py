"""
储层模拟器适配协议（契约 + 参考实现）
=====================================
项目里一直缺的不是"接一个商业模拟器"，而是**接什么、怎么判断接得对**。
没有契约就谈不上对接：ECLIPSE 输出的压力单位可能是 psi、时间可能是天或月、
饱和度可能是小数或百分数。这类不一致不会报错，只会让结论错一个量级。

所以这个模块定义三件事：

    1. ``SimulationCase``   输入契约：算例必须包含哪些量，缺一个就拒绝执行
    2. ``SimulationResult`` 输出契约：结果必须包含哪些量，单位必须显式声明
    3. ``ReservoirBackend`` 协议：validate / run / summarise

## 两个实现，一个是真的

``AnalyticalScreeningBackend`` —— 参考实现。它组合 ``relperm`` + ``flow1d`` +
``caprock`` + ``geomechanics``，**真的能算出数**，用于证明这份契约是可满足的，
也让整条链路在没有商业模拟器时仍然可用。

``ExternalResultBackend`` —— 消费端。它不运行任何模拟器，只做两件事：
按契约校验外部模拟器（ECLIPSE / CMG / TOUGH / OpenGeoSys）导出的 JSON，
然后把它转成 ``SimulationResult``。

**它不是空壳**：校验逻辑是真的（单位、量级、单调性、收敛性），并且
``AnalyticalScreeningBackend`` 的输出会被同一套校验检查 —— 契约被两侧同时
使用，而不是只有文档在声称。

## 为什么"不收敛"必须拒绝

模拟器返回 ``converged: false`` 时结果里的数字仍然存在、看起来完全正常，
但那是迭代失败后的残值。把它当结论写进报告，等于用一堆噪声做安全判定。
所以校验直接拒绝，而不是打个警告 —— 这是 fail-closed，与质量闸门同一个原则。
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field
from typing import Protocol

from utils import caprock as cp
from utils import flow1d
from utils import relperm as rp

BACKEND_VERSION = "2026.10-b2"
# Breaking change: /1 did not bind the result to a case. A stale JSON could be
# consumed as the answer to another injection scenario without any warning.
SCHEMA = "ccus-sim-result/2"

# 允许的单位写法。不做单位换算 —— 要求外部适配器自己转成这套单位，
# 因为"猜单位"是这类集成里最危险的一步（psi 与 bar 差 14.5 倍，
# 但两个数字都"看起来合理"）。
REQUIRED_UNITS = {
    "pressure": "bar",
    "length": "m",
    "time": "day",
    "saturation": "fraction",
}


class BackendUnavailable(RuntimeError):
    """外部模拟器结果不可用。消息必须说清楚缺什么、怎么补。"""


# ══════════════════════════════════════════════════════════
# 契约
# ══════════════════════════════════════════════════════════
@dataclass(frozen=True)
class SimulationCase:
    """一次模拟的输入。缺关键量时 ``validate`` 会列出全部问题，而不是抛第一个。"""

    name: str
    depth_m: float
    injection_pressure_bar: float
    temperature_celsius: float
    thickness_m: float = 50.0
    porosity: float = 0.2
    permeability_md: float = 100.0
    plume_radius_m: float = 500.0
    injection_rate_m3_per_day: float | None = None
    rho_brine: float = 1030.0
    fault_dip_deg: float = 60.0
    fault_azimuth_deg: float = 90.0

    def validate(self) -> list[str]:
        """返回全部问题（空列表表示可用）。一次列全，避免改一个跑一次。"""
        problems = []
        if not (self.name or "").strip():
            problems.append("算例缺少名称")
        if not 50.0 <= self.depth_m <= 9000.0:
            problems.append(f"埋深 {self.depth_m} m 超出 [50, 9000]")
        if not math.isfinite(self.injection_pressure_bar) or self.injection_pressure_bar <= 0:
            problems.append("注入压力必须为有限正数")
        if not -20.0 <= self.temperature_celsius <= 200.0:
            problems.append(f"温度 {self.temperature_celsius}℃ 超出 [-20, 200]")
        if not 1.0 <= self.thickness_m <= 1000.0:
            problems.append(f"储层厚度 {self.thickness_m} m 超出 [1, 1000]")
        if not 0.01 <= self.porosity <= 0.45:
            problems.append(f"孔隙度 {self.porosity} 超出 [0.01, 0.45]")
        if not math.isfinite(self.permeability_md) or self.permeability_md <= 0:
            problems.append("渗透率必须为有限正数")
        if not math.isfinite(self.plume_radius_m) or self.plume_radius_m <= 0:
            problems.append("羽流半径必须为有限正数")
        if self.injection_rate_m3_per_day is not None and (
            not math.isfinite(self.injection_rate_m3_per_day)
            or self.injection_rate_m3_per_day <= 0
        ):
            problems.append("注入速率必须为有限正数")
        if not math.isfinite(self.rho_brine) or self.rho_brine <= 0:
            problems.append("盐水密度必须为有限正数")
        if not math.isfinite(self.fault_azimuth_deg):
            problems.append("断层方位必须为有限数")
        if not 0.0 < self.fault_dip_deg < 90.0:
            problems.append(f"断层倾角 {self.fault_dip_deg}° 超出 (0, 90)")
        return problems


@dataclass(frozen=True)
class SimulationResult:
    """统一结果。两个后端返回同一种结构，下游不需要知道是谁算的。"""

    backend: str
    kind: str                       # "analytical" | "external"
    converged: bool
    units: dict
    time_days: list[float] = field(default_factory=list)
    co2_saturation: list[float] = field(default_factory=list)
    pressure_bar: list[float] = field(default_factory=list)
    co2_column_height_m: float | None = None
    entry_pressure_bar: float | None = None
    notes: list[str] = field(default_factory=list)
    version: str = BACKEND_VERSION

    def validate(self) -> list[str]:
        """结果自检。两个后端都要过这一关 —— 契约必须双向生效。"""
        problems = []
        if not self.converged:
            problems.append("模拟未收敛：结果不可用于判定")
        for key, want in REQUIRED_UNITS.items():
            got = (self.units or {}).get(key)
            if got != want:
                problems.append(f"单位 {key} 必须显式声明为 {want!r}，实际 {got!r}")
        if not self.time_days:
            problems.append("缺少时间序列")
            return problems
        n = len(self.time_days)
        if len(self.co2_saturation) != n or len(self.pressure_bar) != n:
            problems.append(
                f"序列长度不一致：时间 {n} / 饱和度 {len(self.co2_saturation)} / "
                f"压力 {len(self.pressure_bar)}"
            )
        if any(not math.isfinite(t) or t < 0 for t in self.time_days):
            problems.append("时间必须为有限非负数")
        if any(b <= a for a, b in zip(self.time_days, self.time_days[1:])):
            problems.append("时间序列必须严格递增")
        if any(not math.isfinite(s) or not 0.0 <= s <= 1.0 for s in self.co2_saturation):
            problems.append("CO₂ 饱和度必须是 [0, 1] 内的有限数（百分数没有换算？）")
        if any(not math.isfinite(p) or p <= 0 for p in self.pressure_bar):
            problems.append("压力必须为有限正数")
        for label, value in (
            ("CO₂ 柱高", self.co2_column_height_m),
            ("盖层突破压力", self.entry_pressure_bar),
        ):
            if value is not None and (not math.isfinite(value) or value < 0):
                problems.append(f"{label} 必须为有限非负数")
        return problems

    def as_dict(self) -> dict:
        return asdict(self)


# ══════════════════════════════════════════════════════════
# 协议
# ══════════════════════════════════════════════════════════
class ReservoirBackend(Protocol):
    """模拟器后端。``run`` 前必须调用 ``validate``，不可用时抛 ``BackendUnavailable``。"""

    name: str
    kind: str

    def validate(self, case: SimulationCase) -> list[str]:
        ...

    def run(self, case: SimulationCase) -> SimulationResult:
        ...

    def summarise(self, result: SimulationResult) -> str:
        ...


# ══════════════════════════════════════════════════════════
# 参考实现：解析筛查
# ══════════════════════════════════════════════════════════
class AnalyticalScreeningBackend:
    """
    组合相渗 + 一维驱替 + 盖层 + 地应力，产出一条时间序列。

    它**不是**储层模拟器：没有二维/三维、没有溶解、没有重力分异、没有相间传质。
    它的定位是"在跑不动商业模拟器时给出量级与趋势"，以及"证明契约可满足"。
    这个定位写在 ``notes`` 里随结果返回，避免被引用成模拟器结论。
    """

    name = "analytical-screening"
    kind = "analytical"

    def __init__(self, n_steps: int = 6):
        self.n_steps = n_steps

    def validate(self, case: SimulationCase) -> list[str]:
        return case.validate()

    def run(self, case: SimulationCase) -> SimulationResult:
        problems = self.validate(case)
        if problems:
            raise ValueError("算例不合法：" + "；".join(problems))

        front = rp.bl_front()
        t_bt = 1.0 / front["dfg_front"] if front["dfg_front"] > 0 else 1.0
        area = math.pi * case.plume_radius_m ** 2
        pore_volume = area * case.thickness_m * case.porosity
        rate = case.injection_rate_m3_per_day or 1000.0
        days_bt = t_bt * pore_volume / rate

        times, sats, press = [], [], []
        for i in range(1, self.n_steps + 1):
            frac = i / self.n_steps
            t_d = t_bt * frac * 0.95          # 停在突破前，避免剖面失真
            prof = flow1d.solve(n_cells=120, t_end=t_d)
            times.append(round(days_bt * frac * 0.95, 1))
            sats.append(round(sum(prof["sg"]) / len(prof["sg"]), 6))
            # 压力按注入压力近似恒定（不可压缩、稳态假设）
            press.append(round(case.injection_pressure_bar, 2))

        h = flow1d.co2_column_height(
            prof["sg"], 1.0 / 120, case.thickness_m, case.porosity
        )
        cap = cp.caprock_integrity(
            depth_m=case.depth_m,
            injection_pressure_bar=case.injection_pressure_bar,
            temperature_celsius=case.temperature_celsius,
            rho_brine=case.rho_brine,
            co2_column_height_m=h,
        )

        return SimulationResult(
            backend=self.name,
            kind=self.kind,
            converged=True,
            units=dict(REQUIRED_UNITS),
            time_days=times,
            co2_saturation=sats,
            pressure_bar=press,
            co2_column_height_m=round(h, 2),
            entry_pressure_bar=cap.get("entry_pressure_bar"),
            notes=[
                "解析筛查后端：一维不可混溶驱替，非储层模拟器",
                "未含重力分异、溶解、相间传质；柱高为下限，突破时间为上限",
                f"盖层判定：{cap.get('verdict')}（{cap.get('summary', '')}）",
            ],
        )

    def summarise(self, result: SimulationResult) -> str:
        if not result.time_days:
            return "无结果"
        return (
            f"[{result.backend}] 末时刻平均 CO₂ 饱和度 "
            f"{result.co2_saturation[-1]:.3f}，等效柱高 "
            f"{result.co2_column_height_m} m，盖层突破压力 "
            f"{result.entry_pressure_bar} bar"
        )


# ══════════════════════════════════════════════════════════
# 外部模拟器结果消费端
# ══════════════════════════════════════════════════════════
class ExternalResultBackend:
    """
    读取商业/开源模拟器导出的 JSON，按契约校验后转成 ``SimulationResult``。

    它**不运行**模拟器。要在本机跑 CMG/ECLIPSE，需要另写一个适配器把
    模拟器输出转成本模块的 JSON 契约（见 ``SCHEMA`` 与 ``docs/simulator-adapter.md``）。
    这里负责的是"转进来之后不许出错"：单位、量级、单调性、收敛性逐项校验。

    构造时若文件不存在，``validate`` 会返回明确原因而不是抛异常；
    ``run`` 才抛 ``BackendUnavailable``。这样调用方可以先探测再决定。
    """

    name = "external-json"
    kind = "external"

    def __init__(self, path: str):
        self.path = path

    # ── 读取 ─────────────────────────────────────────────
    def _load(self) -> dict:
        if not os.path.exists(self.path):
            raise BackendUnavailable(
                f"找不到模拟器结果文件 {self.path}。"
                "需要先用适配器把 ECLIPSE/CMG/TOUGH 的输出转成 "
                f"{SCHEMA} 契约（字段见 docs/simulator-adapter.md）。"
            )
        try:
            with open(self.path, encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, UnicodeError, json.JSONDecodeError) as e:
            raise BackendUnavailable(f"{self.path} 无法读取或不是合法 JSON：{e}") from e
        if not isinstance(data, dict):
            raise BackendUnavailable(f"{self.path} 根节点必须是 JSON 对象")
        # No instance cache: the file can be replaced between requests. A run
        # validates and consumes the SAME snapshot (not validate old/read new).
        return data

    def _check(self, data: dict, case: SimulationCase) -> list[str]:
        problems = list(case.validate())
        if data.get("schema") != SCHEMA:
            problems.append(
                f"schema 必须是 {SCHEMA!r}，实际 {data.get('schema')!r}"
            )
        if not isinstance(data.get("case"), dict) or data["case"] != asdict(case):
            problems.append("外部结果的 case 输入与当前算例不一致；拒绝把旧算例当成当前结论")
        if data.get("converged") is not True:
            problems.append("converged 必须是 JSON 布尔值 true；字符串 'false' 不能当作收敛")
        if not isinstance(data.get("backend"), str) or not data["backend"].strip():
            problems.append("backend 必须是非空的模拟器标识")
        try:
            result = self._to_result(data)
        except (TypeError, ValueError, OverflowError) as e:
            problems.append(f"模拟器结果字段格式错误：{e}")
        else:
            problems.extend(result.validate())
        return problems

    def validate(self, case: SimulationCase) -> list[str]:
        try:
            data = self._load()
        except BackendUnavailable as e:
            return list(case.validate()) + [str(e)]
        return self._check(data, case)

    def run(self, case: SimulationCase) -> SimulationResult:
        try:
            data = self._load()
        except BackendUnavailable as e:
            raise BackendUnavailable("模拟器结果不可用：" + str(e)) from e
        problems = self._check(data, case)
        if problems:
            raise BackendUnavailable("模拟器结果不可用：" + "；".join(problems))
        return self._to_result(data)

    def _to_result(self, data: dict) -> SimulationResult:
        def series(name: str) -> list[float]:
            values = data.get(name)
            if not isinstance(values, list):
                raise TypeError(f"{name} 必须是数组")
            if any(isinstance(x, bool) for x in values):
                raise TypeError(f"{name} 不接受布尔值")
            return [float(x) for x in values]

        notes = data.get("notes", [])
        if not isinstance(notes, list) or any(not isinstance(s, str) for s in notes):
            raise TypeError("notes 必须是字符串数组")
        units = data.get("units")
        if not isinstance(units, dict):
            raise TypeError("units 必须是对象")
        return SimulationResult(
            backend=data.get("backend", ""),
            kind=self.kind,
            converged=data.get("converged") is True,
            units=units,
            time_days=series("time_days"),
            co2_saturation=series("co2_saturation"),
            pressure_bar=series("pressure_bar"),
            co2_column_height_m=(
                float(data["co2_column_height_m"])
                if data.get("co2_column_height_m") is not None else None
            ),
            entry_pressure_bar=(
                float(data["entry_pressure_bar"])
                if data.get("entry_pressure_bar") is not None else None
            ),
            notes=notes,
        )

    def summarise(self, result: SimulationResult) -> str:
        if not result.time_days:
            return "无结果"
        return (
            f"[{result.backend}] {len(result.time_days)} 个时间步，"
            f"末时刻平均 CO₂ 饱和度 {result.co2_saturation[-1]:.3f}，"
            f"压力 {result.pressure_bar[-1]:.1f} bar"
        )


# ══════════════════════════════════════════════════════════
# 注册表
# ══════════════════════════════════════════════════════════
def available_backends(
    external_path: str | None = None, case: SimulationCase | None = None
) -> dict:
    """按**真实算例**探测后端。未给 case 时不能用假算例证明外部结果可用。"""
    out = {
        "analytical-screening": {
            "kind": "analytical",
            "available": True,
            "note": "一维解析筛查，随时可用；不是储层模拟器",
        }
    }
    if external_path:
        backend = ExternalResultBackend(external_path)
        problems = (
            backend.validate(case) if case is not None
            else ["未提供 SimulationCase：不能确认结果是否属于当前算例"]
        )
        out["external-json"] = {
            "kind": "external",
            "available": not problems,
            "path": external_path,
            "problems": problems,
        }
    return out


def compare_screening_to_external(
    case: SimulationCase, external_path: str, *,
    pressure_tolerance_bar: float, saturation_tolerance: float,
) -> dict:
    """对齐时间轴并量化筛查模型与外部模拟结果的差异；**不是现场验证**。

    输入必须已绑定相同 case，并声明压力与饱和度的观测位置/空间支撑一致；
    缺声明或时间不覆盖筛查时间轴时返回 not_comparable，绝不外推或假装验证。
    容差由调用者根据数据质量预先指定，不在代码里发明及格线。
    """
    for label, value in (
        ("pressure_tolerance_bar", pressure_tolerance_bar),
        ("saturation_tolerance", saturation_tolerance),
    ):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{label} 必须为有限非负数")
    backend = ExternalResultBackend(external_path)
    try:
        data = backend._load()
    except BackendUnavailable as e:
        return {"status": "rejected", "problems": [str(e)]}
    problems = backend._check(data, case)
    if problems:
        return {"status": "rejected", "problems": problems}
    if data.get("comparison_basis") != {
        "pressure_bar": "injection_well_bottomhole",
        "co2_saturation": "same_control_volume_mean",
    }:
        return {
            "status": "not_comparable",
            "reason": "未声明两条曲线对应同一井底观测点及同一体积的平均饱和度",
        }
    external = backend._to_result(data)
    screening = AnalyticalScreeningBackend().run(case)
    if len(external.time_days) < 2 or (
        external.time_days[0] > screening.time_days[0]
        or external.time_days[-1] < screening.time_days[-1]
    ):
        return {
            "status": "not_comparable",
            "reason": "外部时间轴未覆盖筛查模型全部采样时刻；不允许外推或只挑重叠段",
        }

    def interpolate(times: list[float], values: list[float], at: float) -> float:
        from bisect import bisect_left

        pos = bisect_left(times, at)
        if pos < len(times) and times[pos] == at:
            return values[pos]
        left, right = pos - 1, pos
        weight = (at - times[left]) / (times[right] - times[left])
        return values[left] * (1.0 - weight) + values[right] * weight

    metrics = {}
    for key, predicted, observed, tolerance in (
        ("pressure_bar", screening.pressure_bar, external.pressure_bar, pressure_tolerance_bar),
        ("co2_saturation", screening.co2_saturation, external.co2_saturation, saturation_tolerance),
    ):
        errors = [
            abs(p - interpolate(external.time_days, observed, t))
            for t, p in zip(screening.time_days, predicted)
        ]
        metrics[key] = {
            "mae": sum(errors) / len(errors), "max_abs_error": max(errors),
            "tolerance": tolerance, "within_tolerance": max(errors) <= tolerance,
        }
    return {
        "status": "compared", "time_days": screening.time_days,
        "metrics": metrics,
        "within_user_tolerances": all(m["within_tolerance"] for m in metrics.values()),
        "external_backend": external.backend,
        "source_identity_verified": False,
        "limitations": (
            "仅量化声明同支撑区域的两条曲线差异；case 与比较口径均由文件作者声明，"
            "未独立核验来源、网格或真实油田观测，不构成场地级验证"
        ),
    }


def run_case(
    case: SimulationCase, external_path: str | None = None, *,
    allow_screening_fallback: bool = False,
) -> dict:
    """运行外部结果或解析筛查；指定外部结果失败时默认**拒绝**。

    旧版即使外部结果的算例或字段不匹配也会返回 ``ok=True`` 的解析结果，
    下游若只检查 ok 就会误认为外部验证成功。现在只有显式选择
    ``allow_screening_fallback=True`` 才会退回筛查，并在结果里写明原因。
    """
    problems = case.validate()
    if problems:
        return {"ok": False, "problems": problems, "backend": None}

    if external_path:
        backend = ExternalResultBackend(external_path)
        try:
            result = backend.run(case)
            return {
                "ok": True,
                # backend 是"谁算出的数"（可能是 eclipse-2021），
                # adapter 是"走哪条通路进来的"。两者都写，读者才知道
                # 结论来自模拟器还是来自解析筛查。
                "backend": result.backend,
                "adapter": backend.name,
                "degraded": False,
                "result": result.as_dict(),
                "summary": backend.summarise(result),
            }
        except BackendUnavailable as e:
            if not allow_screening_fallback:
                return {
                    "ok": False, "backend": None, "adapter": backend.name,
                    "degraded": False, "problems": [str(e)], "result": None,
                }
            fallback = AnalyticalScreeningBackend()
            result = fallback.run(case)
            return {
                "ok": True,
                "backend": result.backend,
                "adapter": fallback.name,
                "degraded": True,
                "degraded_reason": str(e),
                "result": result.as_dict(),
                "summary": fallback.summarise(result),
            }

    backend = AnalyticalScreeningBackend()
    result = backend.run(case)
    return {
        "ok": True,
        "backend": result.backend,
        "adapter": backend.name,
        "degraded": False,
        "result": result.as_dict(),
        "summary": backend.summarise(result),
    }
