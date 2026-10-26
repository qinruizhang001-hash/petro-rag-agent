"""
运维层契约与规则的测试
======================
这一组测试守的不是"函数能跑"，而是**契约层有没有真的挡住错误输入**。
每一条都对应一个具体的失效后果，写在断言消息里 —— 因为删掉一条校验
不会让任何东西变红，只会让某类错误静默通过，而那正是这类系统最危险的状态。
"""
from __future__ import annotations

import os
import sys
import unittest
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ops import chain
from ops.contracts import (
    CONFIDENCE_STRONG,
    DIRECTION_HIGH,
    DIRECTION_LOW,
    INDETERMINATE,
    Alarm,
    Attribution,
    Citation,
    Reading,
    SensorSpec,
    Threshold,
)
from ops.thresholds import (
    DEFAULT_SENSORS,
    ENV_PREFIX,
    SENSORS,
    apply_env_overrides,
    build_registry,
    load_registry,
    sensor,
)


class ThresholdContractTests(unittest.TestCase):
    def test_empty_source_is_rejected(self):
        # 没有出处的阈值不能参与判定：报告里"超过标准限值"与"超过我们自己设的线"
        # 是两件完全不同的事，而读者无法从数字本身分辨
        with self.assertRaises(ValueError) as ctx:
            Threshold(warn=80.0, source="")
        self.assertIn("source", str(ctx.exception))

    def test_whitespace_source_is_rejected(self):
        with self.assertRaises(ValueError):
            Threshold(warn=80.0, source="   ")

    def test_all_levels_empty_is_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            Threshold(source="某文件")
        self.assertIn("三级全为空", str(ctx.exception))

    def test_duplicate_levels_are_rejected(self):
        # 重复的级别等于少一级，而告警只会少报、不会报错
        with self.assertRaises(ValueError) as ctx:
            Threshold(warn=80.0, alert=80.0, source="某文件")
        self.assertIn("重复", str(ctx.exception))

    def test_direction_high_requires_ascending(self):
        spec = sensor("wellhead_p")   # direction=high
        self.assertEqual(spec.direction, DIRECTION_HIGH)
        Threshold(warn=80, alert=100, critical=120, source="x").validate_for(spec)
        with self.assertRaises(ValueError) as ctx:
            Threshold(warn=120, alert=100, critical=80, source="x").validate_for(spec)
        self.assertIn("递增", str(ctx.exception))

    def test_direction_low_requires_descending(self):
        spec = sensor("capture_rate")  # direction=low
        self.assertEqual(spec.direction, DIRECTION_LOW)
        Threshold(warn=88, alert=82, critical=75, source="x").validate_for(spec)
        with self.assertRaises(ValueError) as ctx:
            Threshold(warn=75, alert=82, critical=88, source="x").validate_for(spec)
        self.assertIn("递减", str(ctx.exception))

    def test_single_level_skips_monotonic_check(self):
        # 只配了一级时没有单调性可言，不该被拒
        spec = sensor("wellhead_p")
        Threshold(warn=80, source="x").validate_for(spec)

    def test_levels_returns_in_ascending_severity(self):
        th = Threshold(warn=80, alert=100, critical=120, source="x")
        self.assertEqual([n for n, _ in th.levels()], ["warn", "alert", "critical"])


class SensorSpecTests(unittest.TestCase):
    def test_unknown_stage_is_rejected(self):
        # stage 是枚举而不是自由字符串：写错会让 supervisor 的上下游推断静默失效
        with self.assertRaises(ValueError) as ctx:
            SensorSpec(
                key="x", label="X", unit="bar", stage="storage",
                direction=DIRECTION_HIGH, basis="某基准",
            )
        self.assertIn("stage", str(ctx.exception))

    def test_empty_direction_is_rejected(self):
        with self.assertRaises(ValueError):
            SensorSpec(
                key="x", label="X", unit="bar", stage="capture",
                direction="", basis="某基准",
            )

    def test_empty_basis_is_rejected(self):
        # 没有测量基准就无法判读：表压与绝对压力差一个大气压，
        # 在判断"是否接近临界压力"时足以改变结论
        with self.assertRaises(ValueError) as ctx:
            SensorSpec(
                key="x", label="X", unit="bar", stage="capture",
                direction=DIRECTION_HIGH, basis="",
            )
        self.assertIn("basis", str(ctx.exception))

    def test_every_default_sensor_is_well_formed(self):
        for spec in DEFAULT_SENSORS:
            with self.subTest(sensor=spec.key):
                self.assertIn(spec.stage, ("capture", "transport", "injection", "mva"))
                self.assertIn(spec.direction, (DIRECTION_HIGH, DIRECTION_LOW))
                self.assertTrue(spec.basis.strip())

    def test_mva_is_expanded_into_several_sensors(self):
        # MVA 是一套监测体系（深部/浅部/近地表），不是一个测点。
        # 把它当成单一指标，超阈值时就无法归因到任何具体层位
        mva = [s for s in DEFAULT_SENSORS if s.stage == "mva"]
        self.assertGreaterEqual(len(mva), 4)
        keys = {s.key for s in mva}
        self.assertIn("reservoir_p", keys)      # 深部
        self.assertIn("gw_ph", keys)            # 浅部
        self.assertIn("soil_co2_flux", keys)    # 近地表


class ReadingTests(unittest.TestCase):
    def test_nan_is_rejected(self):
        # NaN 与任何数比较都是 False，静默放行会让它看起来像"没超阈值"
        with self.assertRaises(ValueError) as ctx:
            Reading(sensor="wellhead_p", value=float("nan"), ts=datetime(2026, 1, 1))
        self.assertIn("NaN", str(ctx.exception))

    def test_inf_is_rejected(self):
        with self.assertRaises(ValueError):
            Reading(sensor="wellhead_p", value=float("inf"), ts=datetime(2026, 1, 1))

    def test_unknown_quality_is_rejected(self):
        with self.assertRaises(ValueError):
            Reading(
                sensor="wellhead_p", value=1.0, ts=datetime(2026, 1, 1),
                quality="maybe",
            )


class AttributionTests(unittest.TestCase):
    def test_evidence_required_unless_ruled_out(self):
        # 没有证据的归因与猜测无法区分，运维照着它处置只会引入新故障
        with self.assertRaises(ValueError) as ctx:
            Attribution(cause="某原因", stage="injection", confidence=CONFIDENCE_STRONG)
        self.assertIn("没有证据", str(ctx.exception))

    def test_ruled_out_requires_reason(self):
        with self.assertRaises(ValueError) as ctx:
            Attribution(
                cause="某原因", stage="injection",
                confidence=CONFIDENCE_STRONG, ruled_out=True,
            )
        self.assertIn("排除理由", str(ctx.exception))

    def test_unknown_confidence_is_rejected(self):
        with self.assertRaises(ValueError):
            Attribution(
                cause="某原因", stage="injection",
                confidence="0.8", evidence=["某读数"],
            )


class AlarmTests(unittest.TestCase):
    def _alarm(self, **kw):
        base = dict(
            alarm_id="A-1", sensor="wellhead_p", label="井口压力",
            stage="injection", severity="warn", direction=DIRECTION_HIGH,
            value=85.0, ts=datetime(2026, 1, 1), rationale=["R_THRESHOLD：…"],
        )
        base.update(kw)
        return Alarm(**base)

    def test_empty_rationale_is_rejected(self):
        # 每条告警都要能回答"为什么是这一级"，否则分级不可复核
        with self.assertRaises(ValueError) as ctx:
            self._alarm(rationale=[])
        self.assertIn("为什么是这一级", str(ctx.exception))

    def test_unknown_severity_is_rejected(self):
        with self.assertRaises(ValueError):
            self._alarm(severity="very-bad")

    def test_indeterminate_is_accepted(self):
        a = self._alarm(severity=INDETERMINATE)
        self.assertFalse(a.actionable)

    def test_actionable_flag_matches_severity(self):
        self.assertTrue(self._alarm(severity="alert").actionable)
        self.assertTrue(self._alarm(severity="critical").actionable)
        self.assertFalse(self._alarm(severity="notice").actionable)


class CitationTests(unittest.TestCase):
    def test_summary_requires_algorithm(self):
        # 聚合量不能逐条回代，唯一能核对的方式是复算，因此算法必须写明
        with self.assertRaises(ValueError) as ctx:
            Citation(kind="summary", ref="告警统计")
        self.assertIn("算法", str(ctx.exception))

    def test_summary_with_detail_is_accepted(self):
        c = Citation(kind="summary", ref="告警统计", detail="按 severity 计数")
        self.assertEqual(c.render(), "[summary:告警统计]")

    def test_unknown_kind_is_rejected(self):
        with self.assertRaises(ValueError):
            Citation(kind="system", ref="某系统")


class RegistryTests(unittest.TestCase):
    def test_unknown_sensor_raises_instead_of_returning_none(self):
        # 拼错测点名如果只返回 None，结果是"没有阈值 → 不判定 → 看板正常"，
        # 一次拼写错误变成一处永久盲区
        with self.assertRaises(ValueError) as ctx:
            sensor("wellhead_pressure")
        self.assertIn("未知测点", str(ctx.exception))
        self.assertIn("wellhead_p", str(ctx.exception))

    def test_missing_config_file_is_not_an_error(self):
        # 运维系统常常先上线后配阈值：此时应逐条 indeterminate 并说明，
        # 而不是启动失败、也不是全部 normal
        reg = load_registry(path="/definitely/not/here.json", environ={})
        self.assertEqual(reg.configured(), [])
        self.assertIn("不存在", reg.origin)
        self.assertEqual(len(reg.missing()), len(SENSORS))

    def test_unknown_sensor_in_config_is_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            build_registry({"wellhead_pressure": {"warn": 1, "source": "x"}})
        self.assertIn("未知测点", str(ctx.exception))

    def test_all_problems_reported_at_once(self):
        # 逐个报会让人改一轮跑一轮
        with self.assertRaises(ValueError) as ctx:
            build_registry({
                "wellhead_p": {"warn": 1, "source": ""},
                "capture_rate": {"warn": 80, "alert": 90, "source": "x"},
                "nope": {"warn": 1, "source": "x"},
            })
        msg = str(ctx.exception)
        # 每条问题都要带测点名：注册表有十几个测点时，"source 不能为空"
        # 这条消息本身无法定位
        self.assertIn("[wellhead_p]", msg)
        self.assertIn("[capture_rate]", msg)
        self.assertIn("nope", msg)

    def test_env_override_changes_source(self):
        # 覆盖后必须改写 source：否则报告说 80 bar 告警、实际按 90 bar 判定，
        # 这个错无法从任何单个文件里看出来
        reg = build_registry(
            {"wellhead_p": {"warn": 80, "source": "设计文件"}}, origin="文件"
        )
        overridden = apply_env_overrides(
            {"wellhead_p": reg.get("wellhead_p")},
            {f"{ENV_PREFIX}WELLHEAD_P_WARN": "90"},
        )
        th = overridden["wellhead_p"]
        self.assertEqual(th.warn, 90.0)
        self.assertIn("env 覆盖", th.source)
        self.assertIn("设计文件", th.source)

    def test_bad_env_value_fails_hard(self):
        # 静默忽略会让阈值悄悄回到文件里的旧值，而报告仍声称用的是 env 值
        reg = build_registry(
            {"wellhead_p": {"warn": 80, "source": "x"}}, origin="文件"
        )
        with self.assertRaises(ValueError) as ctx:
            apply_env_overrides(
                {"wellhead_p": reg.get("wellhead_p")},
                {f"{ENV_PREFIX}WELLHEAD_P_WARN": "eighty"},
            )
        self.assertIn("不是数字", str(ctx.exception))

    def test_env_override_still_checks_monotonicity(self):
        reg = build_registry(
            {"wellhead_p": {"warn": 80, "alert": 100, "source": "x"}}, origin="文件"
        )
        with self.assertRaises(ValueError):
            apply_env_overrides(
                {"wellhead_p": reg.get("wellhead_p")},
                {f"{ENV_PREFIX}WELLHEAD_P_ALERT": "70"},
            )

    def test_require_raises_for_unconfigured(self):
        reg = build_registry({"wellhead_p": {"warn": 1, "source": "x"}})
        with self.assertRaises(KeyError):
            reg.require("capture_rate")


class ChainRuleTests(unittest.TestCase):
    def test_every_rule_action_is_defined(self):
        # 未定义的 action key 会让归因输出一条空建议，
        # 报告读起来像"已给出建议"，实际什么都没给
        for rule in chain.CAUSE_RULES:
            for key in rule.actions:
                with self.subTest(rule=rule.key, action=key):
                    self.assertIn(key, chain.ACTIONS)

    def test_every_rule_sensor_exists(self):
        for rule in chain.CAUSE_RULES:
            for key, _state in rule.requires + rule.excludes:
                with self.subTest(rule=rule.key, sensor=key):
                    self.assertIn(key, SENSORS)

    def test_every_coupling_sensor_exists(self):
        for cp in chain.COUPLINGS:
            with self.subTest(trigger=cp.trigger, response=cp.response):
                self.assertIn(cp.trigger, SENSORS)
                self.assertIn(cp.response, SENSORS)

    def test_min_hits_below_two_is_rejected(self):
        # 只命中一条读数不能构成归因，那是猜测
        with self.assertRaises(ValueError) as ctx:
            chain.CauseRule(
                key="x", cause="c", stage="injection",
                requires=(("wellhead_p", "deviated"), ("injection_rate", "deviated")),
                rationale="r", actions=(), min_hits=1,
            )
        self.assertIn("min_hits", str(ctx.exception))

    def test_min_hits_above_requires_is_rejected(self):
        # 这条规则永远不会命中，属于"写了但没生效"
        with self.assertRaises(ValueError):
            chain.CauseRule(
                key="x", cause="c", stage="injection",
                requires=(("wellhead_p", "deviated"),),
                rationale="r", actions=(), min_hits=3,
            )

    def test_excludes_must_have_matching_reasons(self):
        # 缺理由的排除等于凭直觉排除
        with self.assertRaises(ValueError) as ctx:
            chain.CauseRule(
                key="x", cause="c", stage="injection",
                requires=(("wellhead_p", "deviated"),),
                excludes=(("annulus_p", "ok"),),
                rationale="r", actions=(),
            )
        self.assertIn("凭直觉排除", str(ctx.exception))

    def test_every_rule_has_rationale(self):
        for rule in chain.CAUSE_RULES:
            with self.subTest(rule=rule.key):
                self.assertTrue(rule.rationale.strip())

    def test_severity_actions_non_empty_for_all_levels(self):
        for level in ("notice", "warn", "alert", "critical", INDETERMINATE):
            with self.subTest(level=level):
                self.assertTrue(chain.severity_actions(level))

    def test_actions_for_unknown_key_raises(self):
        with self.assertRaises(ValueError) as ctx:
            chain.actions_for(["no_such_action"])
        self.assertIn("空建议", str(ctx.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)
