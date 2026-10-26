"""
告警 Agent 的测试
=================
重点不是"分级算得对不对"，而是**每条规则在什么情况下会静默失效**：

- 方向写反 → "越低越坏"的传感器永远不告警，而且不报错
- 稳态越限的门槛取错 → 该升级的不升级，告警看起来正常工作
- 仪表不可信被折进 normal → 看板全绿，盲区无人知晓
- 未配阈值被折进 normal → 同上，但更隐蔽（配置错误看起来像工况良好）

每一条都有对应的测试，并且断言消息写明"这条错了会怎样"。
"""
from __future__ import annotations

import os
import sys
import unittest
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ops import alarms
from ops.contracts import (
    CONFIDENCE_PARTIAL,
    CONFIDENCE_STRONG,
    INDETERMINATE,
    QUALITY_BAD,
    QUALITY_SUSPECT,
    Reading,
)
from ops.thresholds import build_registry

T0 = datetime(2026, 10, 26, 8, 0)


def _reg(**extra):
    base = {
        "wellhead_p": {"warn": 80, "alert": 100, "critical": 120, "source": "设计文件 §4.2"},
        "injection_rate": {"warn": 200, "alert": 150, "critical": 100, "source": "设计文件 §4.2"},
        "injectivity_index": {"warn": 5.0, "alert": 4.0, "critical": 3.0, "source": "设计文件 §4.3"},
        "annulus_p": {"warn": 20, "alert": 30, "critical": 40, "source": "设计文件 §4.4"},
        "reservoir_p": {"warn": 300, "alert": 340, "critical": 380, "source": "设计文件 §4.5"},
        "capture_rate": {"warn": 88, "alert": 82, "critical": 75, "source": "可研 §4.1"},
        "regenerator_duty": {"warn": 3.0, "alert": 3.6, "critical": 4.2, "source": "可研 §4.2"},
        "solvent_loss": {"warn": 1.5, "alert": 2.5, "critical": 4.0, "source": "可研 §4.3"},
        "soil_co2_flux": {"warn": 5.0, "alert": 10.0, "critical": 20.0, "source": "基线报告"},
        "gw_ph": {"warn": 6.5, "alert": 6.0, "critical": 5.5, "source": "基线报告"},
        "gw_conductivity": {"warn": 800, "alert": 1200, "critical": 1800, "source": "基线报告"},
        "pipeline_dp": {"warn": 15, "alert": 25, "critical": 35, "source": "管道设计"},
        "pipeline_outlet_p": {"warn": 90, "alert": 110, "critical": 130, "source": "管道设计"},
    }
    base.update(extra)
    return build_registry(base, origin="测试注册表")


def _series(sensor, values, minutes=5, quality="good"):
    return [
        Reading(sensor, v, T0 + timedelta(minutes=minutes * i), quality=quality)
        for i, v in enumerate(values)
    ]


class ThresholdLevelTests(unittest.TestCase):
    def test_high_direction_levels(self):
        # warn=80，提示线 = 80 × 0.9 = 72。70 在提示带外，74 在带内
        reg = _reg()
        for value, expect in [
            (70.0, "normal"), (74.0, "notice"), (85.0, "warn"),
            (100.0, "alert"), (119.0, "alert"), (120.0, "critical"),
        ]:
            with self.subTest(value=value):
                rep = alarms.evaluate(_series("wellhead_p", [value]), reg)
                self.assertEqual(rep.alarms[0].severity, expect)

    def test_low_direction_levels(self):
        # 捕集率越低越坏。若按 "> threshold" 判定，这条规则永远不生效。
        # warn=88，提示线 = 88 × 1.1 = 96.8：97 在带外，90 在带内
        reg = _reg()
        for value, expect in [
            (97.0, "normal"), (90.0, "notice"), (88.0, "warn"),
            (82.0, "alert"), (75.0, "critical"),
        ]:
            with self.subTest(value=value):
                rep = alarms.evaluate(_series("capture_rate", [value]), reg)
                self.assertEqual(rep.alarms[0].severity, expect)

    def test_notice_band_before_warn(self):
        # 提示带是项目政策选择，但必须存在：否则读数从"完全正常"一步跳到 warn，
        # 中间的偏移没有任何人看到
        reg = _reg()
        rep = alarms.evaluate(_series("wellhead_p", [74.0]), reg)   # 80 * 0.9 = 72
        self.assertEqual(rep.alarms[0].severity, "notice")

    def test_threshold_provenance_is_carried_into_alarm(self):
        reg = _reg()
        rep = alarms.evaluate(_series("wellhead_p", [110.0]), reg)
        self.assertIn("设计文件 §4.2", rep.alarms[0].provenance)


class PersistenceRuleTests(unittest.TestCase):
    def test_persistent_exceedance_upgrades(self):
        reg = _reg()
        rep = alarms.evaluate(_series("wellhead_p", [105, 106, 107]), reg)
        self.assertEqual(rep.alarms[0].severity, "critical")
        self.assertTrue(any("R_PERSIST" in r for r in rep.alarms[0].rationale))

    def test_single_point_does_not_upgrade(self):
        # 单点越限可能只是毛刺；升到 critical 会让运维半夜被叫起来查一个不存在的问题。
        # 采样间隔取 60 分钟，是为了让 R_RAPID 不介入 —— 否则这条测试
        # 测的是速率规则而不是持久性规则（两者都会升级，结果一样但原因不同）
        reg = _reg()
        rep = alarms.evaluate(_series("wellhead_p", [98, 99, 105], minutes=60), reg)
        self.assertEqual(rep.alarms[0].severity, "alert")
        self.assertFalse(any("R_PERSIST" in r for r in rep.alarms[0].rationale))

    def test_persist_gate_uses_the_entry_threshold_not_the_top(self):
        # 门槛必须取"进入该档所需的最小值"。若取成 critical 门槛 120，
        # 三个 105 的读数会被判为"未达到 alert"，于是不升级 —— 而告警
        # 本身看起来完全正常，只是少升了一级
        reg = _reg()
        rep = alarms.evaluate(_series("wellhead_p", [105, 105, 105]), reg)
        self.assertEqual(rep.alarms[0].severity, "critical")

    def test_persist_respects_point_count(self):
        reg = _reg()
        rep = alarms.evaluate(_series("wellhead_p", [105, 106]), reg)
        self.assertEqual(rep.alarms[0].severity, "alert")   # 只有 2 点，门槛 3

    def test_zero_persist_points_is_rejected(self):
        # 0 会让 R_PERSIST 永远成立，等于对所有告警无差别升级
        with self.assertRaises(ValueError):
            alarms.evaluate(_series("wellhead_p", [105]), _reg(), persist_points=0)


class RapidRuleTests(unittest.TestCase):
    def test_fast_rise_upgrades(self):
        # 85 → 99 用 10 分钟，折合 84 bar/h，等级间隔 20 bar
        reg = _reg()
        rep = alarms.evaluate(_series("wellhead_p", [85, 99], minutes=10), reg)
        self.assertEqual(rep.alarms[0].severity, "alert")
        self.assertTrue(any("R_RAPID" in r for r in rep.alarms[0].rationale))

    def test_fast_fall_of_low_direction_sensor_upgrades(self):
        # 方向必须反过来：捕集率下降才是恶化。
        # 若沿用 rate > 0 判定，捕集率崩掉时不会提前升级
        reg = _reg()
        rep = alarms.evaluate(_series("capture_rate", [87, 83], minutes=10), reg)
        self.assertIn(rep.alarms[0].severity, ("warn", "alert"))
        self.assertTrue(any("R_RAPID" in r for r in rep.alarms[0].rationale))

    def test_improving_trend_does_not_upgrade(self):
        reg = _reg()
        rep = alarms.evaluate(_series("wellhead_p", [99, 85], minutes=10), reg)
        self.assertEqual(rep.alarms[0].severity, "warn")
        self.assertFalse(any("R_RAPID" in r for r in rep.alarms[0].rationale))


class IndeterminateTests(unittest.TestCase):
    def test_bad_quality_is_indeterminate_not_normal(self):
        # 判 normal 会让盲区看起来正常；判 critical 会触发一次没有依据的应急
        reg = _reg()
        rep = alarms.evaluate(
            _series("wellhead_p", [110], quality=QUALITY_BAD), reg
        )
        self.assertEqual(rep.alarms[0].severity, INDETERMINATE)
        self.assertNotIn(INDETERMINATE, ("normal",))

    def test_bad_quality_keeps_its_own_action(self):
        reg = _reg()
        rep = alarms.evaluate(
            _series("wellhead_p", [110], quality=QUALITY_BAD), reg
        )
        self.assertTrue(
            any("供电" in a["action"] for a in rep.alarms[0].actions),
            "仪表不可信时应给仪表动作，而不是工况处置动作",
        )

    def test_missing_threshold_is_indeterminate(self):
        # 未配阈值必须显式不可判定：静默按 normal 处理会让配置错误
        # 看起来像工况良好
        reg = build_registry(
            {"wellhead_p": {"warn": 80, "source": "x"}}, origin="只有一条"
        )
        rep = alarms.evaluate(_series("capture_rate", [50]), reg)
        a = [x for x in rep.alarms if x.sensor == "capture_rate"][0]
        self.assertEqual(a.severity, INDETERMINATE)
        self.assertIn("未配置", a.provenance)
        self.assertTrue(any("阈值" in act["action"] for act in a.actions))

    def test_unconfigured_list_is_reported(self):
        reg = build_registry({"wellhead_p": {"warn": 80, "source": "x"}})
        rep = alarms.evaluate(_series("wellhead_p", [70]), reg)
        self.assertIn("capture_rate", rep.unconfigured)
        self.assertIn("已知盲区", rep.summary_line())

    def test_suspect_quality_caps_at_warn(self):
        # 可疑读数不该触发停输，但也不该因为"不确定"而当作正常
        reg = _reg()
        rep = alarms.evaluate(
            _series("wellhead_p", [130], quality=QUALITY_SUSPECT), reg
        )
        self.assertEqual(rep.alarms[0].severity, "warn")
        self.assertTrue(any("R_SUSPECT" in r for r in rep.alarms[0].rationale))

    def test_no_readings_at_all_is_indeterminate(self):
        # 没有告警 ≠ 正常
        rep = alarms.evaluate([], _reg())
        self.assertEqual(rep.top_severity(), INDETERMINATE)
        self.assertIn("无读数", rep.summary_line())


class AttributionTests(unittest.TestCase):
    def test_full_match_is_strong(self):
        reg = _reg()
        rs = (
            _series("wellhead_p", [110])
            + _series("injection_rate", [140])
            + _series("injectivity_index", [3.5])
        )
        rep = alarms.evaluate(rs, reg)
        injectivity = [
            a for a in rep.attribution if "吸水能力" in a.cause and not a.ruled_out
        ]
        self.assertTrue(injectivity, "三条同向读数应指向注入性恶化")
        self.assertEqual(injectivity[0].confidence, CONFIDENCE_STRONG)

    def test_partial_match_is_partial(self):
        # 缺一条判据时结论强度必须降级，而不是照旧报 strong
        reg = _reg()
        rs = _series("wellhead_p", [110]) + _series("injection_rate", [140])
        rep = alarms.evaluate(rs, reg)
        hits = [a for a in rep.attribution if "吸水能力" in a.cause]
        self.assertTrue(hits)
        self.assertEqual(hits[0].confidence, CONFIDENCE_PARTIAL)
        self.assertTrue(
            any("未核对" in e for e in hits[0].evidence),
            "缺数据的判据必须写明未核对，否则读者会以为全部判据都支持这条归因",
        )

    def test_exclusion_beats_evidence(self):
        # 环空压力正常时，完整性失效这条原因必须被**排除并写明理由**，
        # 而不是按命中数投票留下它
        reg = _reg()
        rs = _series("wellhead_p", [110]) + _series("annulus_p", [5])
        rep = alarms.evaluate(rs, reg)
        integrity = [a for a in rep.attribution if "完整性" in a.cause]
        self.assertTrue(integrity)
        self.assertTrue(integrity[0].ruled_out)
        self.assertIn("环空压力", integrity[0].ruled_out_reason)

    def test_ruled_out_attributions_are_not_attached_to_alarms(self):
        reg = _reg()
        rs = _series("wellhead_p", [110]) + _series("annulus_p", [5])
        rep = alarms.evaluate(rs, reg)
        for a in rep.alarms:
            for att in a.attribution:
                self.assertFalse(att.ruled_out)

    def test_single_sensor_alone_cannot_attribute(self):
        # 一条读数不能构成归因
        reg = _reg()
        rep = alarms.evaluate(_series("wellhead_p", [110]), reg)
        self.assertEqual([a for a in rep.attribution if not a.ruled_out], [])

    def test_attribution_is_stable_across_runs(self):
        # 同样输入必须同样输出，否则报告无法复核
        reg = _reg()
        rs = (
            _series("wellhead_p", [110])
            + _series("injection_rate", [140])
            + _series("injectivity_index", [3.5])
        )
        a = [x.to_dict() for x in alarms.evaluate(rs, reg).attribution]
        b = [x.to_dict() for x in alarms.evaluate(rs, reg).attribution]
        self.assertEqual(a, b)


class InputValidationTests(unittest.TestCase):
    def test_undeclared_sensor_raises(self):
        # 未声明的测点没有量纲与方向，无法判定；
        # 静默忽略会让它的异常完全不进入任何报告
        with self.assertRaises(ValueError) as ctx:
            alarms.evaluate(_series("wellhead_pressure", [1.0]), _reg())
        self.assertIn("未声明", str(ctx.exception))

    def test_notice_margin_bounds(self):
        with self.assertRaises(ValueError):
            alarms.evaluate(_series("wellhead_p", [70]), _reg(), notice_margin=1.0)
        with self.assertRaises(ValueError):
            alarms.evaluate(_series("wellhead_p", [70]), _reg(), notice_margin=-0.1)

    def test_rows_entry_point(self):
        rows = [{"sensor": "wellhead_p", "value": 110, "ts": T0.isoformat()}]
        rep = alarms.evaluate_rows(rows, _reg())
        self.assertEqual(rep.alarms[0].severity, "alert")


class ReportShapeTests(unittest.TestCase):
    def test_counts_include_every_level(self):
        rep = alarms.evaluate(_series("wellhead_p", [110]), _reg())
        counts = rep.by_severity()
        self.assertIn(INDETERMINATE, counts)
        self.assertEqual(sum(counts.values()), len(rep.alarms))

    def test_to_dict_is_json_safe(self):
        import json
        rep = alarms.evaluate(_series("wellhead_p", [110]), _reg())
        json.dumps(rep.to_dict(), ensure_ascii=False)

    def test_top_severity_ignores_indeterminate(self):
        reg = _reg()
        rs = _series("wellhead_p", [110]) + _series(
            "capture_rate", [50], quality=QUALITY_BAD
        )
        rep = alarms.evaluate(rs, reg)
        self.assertEqual(rep.top_severity(), "alert")


if __name__ == "__main__":
    unittest.main(verbosity=2)
