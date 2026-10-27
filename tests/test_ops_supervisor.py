"""
全链编排 Agent（Supervisor）的测试
==================================
这个模块此前没有测试。它的失败方式和别的模块不同 —— 它不"算错"，
而是**编排错**：

- 子 Agent 失败时整条链一起哑掉 → 运维在最需要信息时看不到任何信息
- 周报只有一期却照常出报告 → "分布"字段只有一个样本，读起来像一周的结论
- 耦合偏离被当成阻断 → 真实工况变化时系统拒绝出报告，而那正是最该出报告的时候
- 简报把结论排在矛盾前面 → 读者只看前三行，看到的是一段确定的口吻

每一条都有对应测试，断言消息写明"这条错了会怎样"。
"""
from __future__ import annotations

import os
import sys
import unittest
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ops import supervisor
from ops.contracts import Reading
from ops.thresholds import build_registry

T0 = datetime(2026, 10, 27, 8, 0)

REG = build_registry(
    {
        "annulus_p": {"warn": 9.5, "alert": 10.0, "critical": 12.0, "source": "设计文件 §4.3"},
        "wellhead_p": {"warn": 80.0, "alert": 85.0, "critical": 90.0, "source": "设计文件 §4.2"},
        "capture_rate": {"warn": 90.0, "alert": 88.0, "critical": 85.0, "source": "能效考核指标"},
    },
    origin="测试注入",
)


def _series(sensor: str, values, *, step_min: int = 10, quality: str = "good"):
    return [
        Reading(sensor, float(v), T0 + timedelta(minutes=step_min * i), quality)
        for i, v in enumerate(values)
    ]


class TestTaskSplitting(unittest.TestCase):
    def test_alarms_is_always_the_base(self):
        req = supervisor.OpsRequest(
            readings=_series("annulus_p", [11]),
            period=(T0, T0),
            want_report=True,
        )
        agents = [t.agent for t in req.tasks()]
        self.assertEqual(agents[0], "alarms", "告警是其余产物的判定基础，必须排第一")
        self.assertIn("reports", agents)

    def test_no_work_when_nothing_wanted(self):
        req = supervisor.OpsRequest(
            readings=_series("annulus_p", [11]),
            period=(T0, T0),
            want_alarms=False,
            want_condition=False,
            want_report=False,
        )
        self.assertEqual(req.tasks(), [])


class TestRun(unittest.TestCase):
    def _run(self, **kw):
        base = dict(
            readings=_series("annulus_p", [11.0, 11.5, 12.4]),
            period=(T0, T0 + timedelta(minutes=20)),
        )
        base.update(kw)
        return supervisor.run(supervisor.OpsRequest(**base), REG)

    def test_alarms_and_report_both_produced(self):
        result = self._run(want_report=True)
        self.assertIsNotNone(result.alarms)
        self.assertIsNotNone(result.report)
        self.assertTrue(all(r.ok for r in result.runs))

    def test_report_failure_does_not_take_down_alarms(self):
        """报告写不出来时，告警清单本身仍有价值。"""
        result = self._run(want_report=True, report_kind="weekly")  # 只给一期
        self.assertIsNotNone(result.alarms, "周报失败不应连带丢掉告警结果")
        self.assertIsNone(result.report)
        report_run = next(r for r in result.runs if r.agent == "reports")
        self.assertFalse(report_run.ok)
        self.assertFalse(result.can_submit)

    def test_weekly_needs_at_least_two_periods(self):
        one = [(T0, T0 + timedelta(minutes=20))]
        result = self._run(want_report=True, report_kind="weekly", periods=one)
        self.assertIsNone(result.report)
        self.assertIn("2 期", next(r for r in result.runs if r.agent == "reports").error)

    def test_weekly_with_two_periods(self):
        periods = [
            (T0, T0 + timedelta(minutes=20)),
            (T0 + timedelta(hours=3), T0 + timedelta(hours=3, minutes=20)),
        ]
        result = self._run(want_report=True, report_kind="weekly", periods=periods)
        self.assertIsNotNone(result.report)
        self.assertEqual(result.report.kind, "weekly")

    def test_regulatory_report_carries_permit(self):
        result = self._run(want_report=True, report_kind="regulatory", permit_id="CCS-2026-0417")
        self.assertEqual(result.report.permit_id, "CCS-2026-0417")


class TestDirectionConflicts(unittest.TestCase):
    """``detect_direction_conflicts`` 此前没有任何测试。

    它比较的两个数来自两个模块：工况问答的 ``over_threshold``（数"窗口内有几个点
    越过 warn"）与告警的 ``severity``（判"这一条算不算 actionable"）。两者口径一旦
    分开，失败方向有两个，**都不会报错**：

    - 一边变严（例如把 ``>=`` 改成 ``>``）：每个恰好压线的班次都产生一条
      **阻断**冲突，系统拒绝出报告，而所有人都认为工况正常；
    - 一边变松：真正的矛盾不再被发现，看板与日报给出相反结论，没人知道。

    所以这里同时锁"压线一致"与"notice 不算越限"两件事。
    """

    def _conflicts(self, alarm_readings, cond_readings):
        from ops import alarms, condition

        alarm_report = alarms.evaluate(alarm_readings, REG)
        cond = condition.answer("环空压力怎么样", cond_readings, REG)
        return supervisor.detect_direction_conflicts(alarm_report, cond)

    def test_exactly_at_warn_is_not_a_conflict(self):
        # 9.5 恰好等于 warn。告警侧是 `>=` 判 warn，工况侧是 `>=` 计 over，
        # 两边都算"越限" → 一致。任一侧改成 `>` 就会在这里造出一条阻断冲突
        self.assertEqual(
            self._conflicts(_series("annulus_p", [9.5]), _series("annulus_p", [9.5])),
            [],
            "恰好压线时两侧口径必须一致，否则每个压线的班次都拒绝出报告",
        )

    def test_below_warn_on_both_sides_is_not_a_conflict(self):
        self.assertEqual(
            self._conflicts(_series("annulus_p", [9.0]), _series("annulus_p", [9.0])),
            [],
        )

    def test_notice_is_not_counted_as_over(self):
        # 9.4 落在提示带里（提示线 9.35），告警是 notice —— 不 action，
        # 工况侧 over_threshold 也必须是 0。把 notice 算进 ACTIONABLE，
        # 或让 over_threshold 数 notice，都会让正常班次出现阻断冲突
        self.assertEqual(
            self._conflicts(_series("annulus_p", [9.4]), _series("annulus_p", [9.4])),
            [],
            "notice 是「关注」不是「越限」，两侧都不能把它当成越限",
        )

    def test_real_disagreement_blocks_and_names_both_sides(self):
        # 两个窗口不同：告警看的是最新点（9.0，正常），工况问答看的是
        # 一个窗口内有 3 点越限的序列。同一时刻两个相反结论，必须阻断
        conflicts = self._conflicts(
            _series("annulus_p", [9.0]),
            _series("annulus_p", [9.6, 9.6, 9.6]),
        )
        self.assertEqual(len(conflicts), 1)
        c = conflicts[0]
        self.assertEqual(c.kind, supervisor.CONFLICT_DIRECTION)
        self.assertTrue(c.blocking, "看板与日报结论相反时必须阻断：读者只会看其中一个")
        joined = " ".join(c.detail)
        self.assertIn("3", joined, "必须写出工况侧有几个点越限，否则无法复核")
        self.assertIn("warn", joined, "必须写出告警侧命中的是哪一级")

    def test_sensor_absent_from_alarms_is_skipped(self):
        # 告警里没有这个测点 = 没评估过，不是"判定为正常"。
        # 把它当成 False 会造出一条假冲突
        from ops import alarms, condition

        cond = condition.answer("环空压力怎么样", _series("annulus_p", [9.6]), REG)
        other_only = alarms.evaluate(_series("wellhead_p", [50.0]), REG)
        self.assertEqual(
            supervisor.detect_direction_conflicts(other_only, cond),
            [],
            "没评估过的测点不能被当成「判定为正常」",
        )


class TestConflicts(unittest.TestCase):
    def test_value_mismatch_blocks(self):
        """告警取最新点、工况问答取窗口末值；窗口不覆盖最新点时两者必然不同。"""
        readings = _series("annulus_p", [11.0, 11.5, 12.4])
        result = supervisor.run(
            supervisor.OpsRequest(
                readings=readings,
                period=(T0, T0 + timedelta(minutes=20)),
                question="环空压力怎么样",
                want_condition=True,
                want_report=True,
                now=T0 + timedelta(minutes=5),
            ),
            REG,
        )
        kinds = {c.kind for c in result.conflicts}
        self.assertIn(supervisor.CONFLICT_CONTRADICTION, kinds)
        self.assertFalse(result.can_submit)
        self.assertTrue(result.blocking)

    def test_aligned_window_has_no_conflict(self):
        """窗口对齐时不该有**矛盾**类冲突。

        这里刻意只筛矛盾类，不断言 ``conflicts == []``：上限校核在本用例里
        必然返回 ``indeterminate``（没有 bhp/frac_p 读数），那是一条**真实的
        提示**，不是噪声 —— 一个没接井底压力计的场地，就该在冲突列表里
        看到"这项没查成"。把它一起断言掉，等于要求系统把缺席渲染成合格。
        """
        readings = _series("annulus_p", [11.0, 11.5, 12.4])
        result = supervisor.run(
            supervisor.OpsRequest(
                readings=readings,
                period=(T0, T0 + timedelta(minutes=20)),
                question="环空压力怎么样",
                want_condition=True,
            ),
            REG,
        )
        contradictions = [
            c for c in result.conflicts if c.kind != supervisor.CONFLICT_CEILING
        ]
        self.assertEqual(contradictions, [])
        self.assertFalse(result.blocking, "没有矛盾类冲突时不应有阻断项")

    def test_missing_wellbore_pressure_is_reported_not_ignored(self):
        """没接井底压力计 → 上限校核必须出现在冲突列表里。

        ``indeterminate`` 不阻断（缺数据要补数据，不是把人拦在审批门口），
        但它必须可见：否则"没查"与"查过没问题"在界面上完全一样。
        """
        readings = _series("annulus_p", [11.0, 11.5, 12.4])
        result = supervisor.run(
            supervisor.OpsRequest(
                readings=readings,
                period=(T0, T0 + timedelta(minutes=20)),
            ),
            REG,
        )
        ceilings = [
            c for c in result.conflicts if c.kind == supervisor.CONFLICT_CEILING
        ]
        self.assertEqual(len(ceilings), 1)
        self.assertFalse(ceilings[0].blocking)
        self.assertEqual(result.ceiling.state, "indeterminate")

    def test_exceeded_ceiling_blocks_submission(self):
        """井底压力超过破裂压力 90% 是法规违规，必须阻断。"""
        readings = (
            _series("bhp", [410.0])
            + _series("frac_p", [452.0])
        )
        result = supervisor.run(
            supervisor.OpsRequest(
                readings=readings,
                period=(T0, T0),
            ),
            REG,
        )
        self.assertTrue(result.blocking, "超限必须阻断：这是法规判据，不是运行偏好")
        self.assertFalse(result.can_submit)
        self.assertEqual(result.ceiling.state, "exceeded")

    def test_coupling_deviation_does_not_block(self):
        """耦合表说的是"通常如何"，反向变化往往是另一个驱动源 —— 是信息不是错误。

        阻断它会让系统在真实工况变化时拒绝出报告，而那正是最需要报告的时候。
        """
        conflict = supervisor.Conflict(
            kind=supervisor.CONFLICT_COUPLING,
            summary="注入速率与井口压力反向变化",
            blocking=False,
        )
        self.assertFalse(conflict.blocking)
        self.assertNotIn(
            supervisor.CONFLICT_COUPLING,
            supervisor.BLOCKING_KINDS,
            "耦合偏离一旦列入阻断，工况变化时系统就再也出不了报告",
        )


class TestBrief(unittest.TestCase):
    def test_blocking_conflict_appears_before_conclusions(self):
        """读者只看前三行时，应该看到"这份材料不能发"，而不是一段确定的工况描述。"""
        readings = _series("annulus_p", [11.0, 11.5, 12.4])
        result = supervisor.run(
            supervisor.OpsRequest(
                readings=readings,
                period=(T0, T0 + timedelta(minutes=20)),
                question="环空压力怎么样",
                want_condition=True,
                want_report=True,
                now=T0 + timedelta(minutes=5),
            ),
            REG,
        )
        brief = supervisor.render_brief(result)
        head = "\n".join(brief.split("\n")[:8])
        self.assertIn("阻断", head)
        self.assertLess(
            brief.index("阻断"), brief.index("告警"),
            "矛盾必须排在结论之前；顺序反了，读者会先读到一个确定的结论",
        )

    def test_brief_always_renders(self):
        result = supervisor.run(
            supervisor.OpsRequest(
                readings=_series("annulus_p", [11.0]),
                period=(T0, T0),
            ),
            REG,
        )
        self.assertTrue(supervisor.render_brief(result).strip())


if __name__ == "__main__":
    unittest.main(verbosity=2)
