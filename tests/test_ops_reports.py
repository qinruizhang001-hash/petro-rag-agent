"""
报告 Agent 与 Supervisor 的测试
===============================
这两个 Agent 的失效方式是"看起来更专业"：

- 报告里一个没有出处的数字，读者无法分辨，而监管材料里的无法回代数字
  是实质性缺陷
- 审批状态机能被跳过（草稿直接批准），于是审批环节变成形式
- 跨 Agent 的矛盾被"取平均"或"选一个"处理掉，于是最该被看到的信息消失
- 把非阻断的耦合偏离当成错误，系统会在真实工况变化时拒绝出报告 ——
  而那正是最需要报告的时候

每一条都对应一个测试。
"""
from __future__ import annotations

import os
import sys
import unittest
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ops import alarms, reports, supervisor
from ops.contracts import Citation, Reading
from ops.thresholds import build_registry

T0 = datetime(2026, 10, 26, 8, 0)
PERIOD = (T0, T0 + timedelta(hours=8))


def _reg():
    return build_registry(
        {
            "wellhead_p": {"warn": 80, "alert": 100, "critical": 120, "source": "设计文件 §4.2"},
            "injection_rate": {"warn": 200, "alert": 150, "critical": 100, "source": "设计文件 §4.2"},
            "injectivity_index": {"warn": 5.0, "alert": 4.0, "critical": 3.0, "source": "设计文件 §4.3"},
            "capture_rate": {"warn": 88, "alert": 82, "critical": 75, "source": "可研 §4.1"},
            "pipeline_dp": {"warn": 15, "alert": 25, "critical": 35, "source": "管道设计"},
            "pipeline_outlet_p": {"warn": 90, "alert": 110, "critical": 130, "source": "管道设计"},
            "soil_co2_flux": {"warn": 5.0, "alert": 10.0, "critical": 20.0, "source": "基线报告 2025 版"},
        },
        origin="config/ops_thresholds.json",
    )


def _series(sensor, values, minutes=5, start=T0):
    return [
        Reading(sensor, v, start + timedelta(minutes=minutes * i))
        for i, v in enumerate(values)
    ]


def _alarm_report():
    rs = (
        _series("wellhead_p", [105, 106, 107])
        + _series("injection_rate", [140])
        + _series("injectivity_index", [3.5])
    )
    return alarms.evaluate(rs, _reg())


class CitationCheckTests(unittest.TestCase):
    def test_number_without_citation_is_flagged(self):
        problems = reports.check_citations(
            "<!-- strict -->\n井口压力 106.5 bar\n<!-- /strict -->", {}
        )
        self.assertEqual(len(problems), 1)
        self.assertEqual(problems[0].kind, "unsourced")

    def test_number_with_citation_passes(self):
        cite = Citation("alarm", "A-1")
        problems = reports.check_citations(
            "<!-- strict -->\n井口压力 106.5 bar [alarm:A-1]\n<!-- /strict -->",
            {cite.render(): cite},
        )
        self.assertEqual(problems, [])

    def test_long_unit_does_not_break_binding(self):
        # 绑定规则用"下一个数字之前"而不是固定字符窗口：
        # t/(d·bar) 比 bar 长得多，固定窗口会把长单位的数字误判为无引用
        cite = Citation("alarm", "A-1")
        problems = reports.check_citations(
            "<!-- strict -->\n注入指数 3.5 t/(d·bar) [alarm:A-1]\n<!-- /strict -->",
            {cite.render(): cite},
        )
        self.assertEqual(problems, [])

    def test_dangling_citation_is_flagged(self):
        # 引用了不存在的告警比缺引用更严重：稿子看起来每个数字都有出处，
        # 而所有出处都是空的
        problems = reports.check_citations(
            "<!-- strict -->\n井口压力 106.5 bar [alarm:A-does-not-exist]\n<!-- /strict -->",
            {},
        )
        self.assertEqual([p.kind for p in problems], ["dangling"])

    def test_outside_strict_block_is_not_checked(self):
        # "24 小时内复采样"里的 24 是处置时限，不是测量值。
        # 对它也要求引用只会得到一屏误报，然后所有人学会忽略这个检查
        problems = reports.check_citations(
            "- 处置：24 小时内同点位复采样\n"
            "- 政策：notice 留 10% 余量\n"
            "- 测点 co2_purity 未配置阈值\n",
            {},
        )
        self.assertEqual(problems, [])

    def test_timestamp_and_list_number_are_exempt(self):
        problems = reports.check_citations(
            "<!-- strict -->\n1. 2026-10-26 08:00 起\n<!-- /strict -->", {}
        )
        self.assertEqual(problems, [])

    def test_citation_id_digits_are_exempt(self):
        # A-20261026T0810-wellhead_p 里的数字是告警 ID 的一部分
        cite = Citation("alarm", "A-20261026T0810-wellhead_p")
        problems = reports.check_citations(
            "<!-- strict -->\n井口压力 107 bar [alarm:A-20261026T0810-wellhead_p]\n<!-- /strict -->",
            {cite.render(): cite},
        )
        self.assertEqual(problems, [])


class DailyReportTests(unittest.TestCase):
    def test_generated_report_passes_its_own_check(self):
        d = reports.build_daily(_alarm_report(), _reg(), period=PERIOD)
        self.assertEqual(d.problems, [], [p.render() for p in d.problems])
        self.assertTrue(d.ready_to_submit)

    def test_manual_number_breaks_the_check(self):
        # 模拟"有人手工加了一句带数字的话"：必须被抓出来
        d = reports.build_daily(_alarm_report(), _reg(), period=PERIOD)
        d.body = d.body.replace(
            reports.STRICT_CLOSE,
            "井口压力较昨日上升 15%\n" + reports.STRICT_CLOSE,
            1,
        )
        problems = reports.check_citations(d.body, d.citations)
        self.assertTrue(any(p.token == "15" for p in problems))

    def test_blind_spots_are_disclosed(self):
        # 隐瞒盲区比报告异常更严重：异常可以解释，盲区是"我们没看见"
        d = reports.build_daily(_alarm_report(), _reg(), period=PERIOD)
        self.assertTrue(d.blind_spots)
        self.assertIn("未配置阈值", d.body)
        self.assertIn("无读数", d.body)

    def test_no_readings_is_not_reported_as_normal(self):
        empty = alarms.evaluate([], _reg())
        d = reports.build_daily(empty, _reg(), period=PERIOD)
        self.assertIn("未收到任何测点读数", d.body)
        self.assertIn("不构成", d.body)

    def test_all_indeterminate_is_reported_as_such(self):
        # 注册表为空 → 有读数的测点也拿不到阈值 → 全部 indeterminate。
        # 此时不能写"运行正常"：那是把配置缺失渲染成工况良好
        rep = alarms.evaluate(_series("wellhead_p", [70]), build_registry({}, origin="空表"))
        d = reports.build_daily(rep, build_registry({}, origin="空表"), period=PERIOD)
        self.assertIn("不可判定", d.body)
        self.assertNotIn("运行正常", d.body)

    def test_ruled_out_causes_are_listed_with_reasons(self):
        reg = _reg()
        rs = _series("wellhead_p", [110]) + _series("annulus_p", [5]) if False else None
        d = reports.build_daily(_alarm_report(), reg, period=PERIOD)
        self.assertIn("阈值注册表来源", d.body)


class RegulatoryReportTests(unittest.TestCase):
    def test_permit_id_is_required(self):
        # 没有编号的材料无法归入档案，也无法在后续核查中被引用
        with self.assertRaises(ValueError) as ctx:
            reports.build_regulatory(
                _alarm_report(), _reg(), period=PERIOD, permit_id="  "
            )
        self.assertIn("许可编号", str(ctx.exception))

    def test_mva_section_exists_even_without_mva_data(self):
        # 没有 MVA 数据不等于没有泄漏
        d = reports.build_regulatory(
            _alarm_report(), _reg(), period=PERIOD, permit_id="CCS-2026-001"
        )
        self.assertIn("MVA", d.body)
        self.assertIn("不能据此认为未发生泄漏", d.body)

    def test_conclusion_boundary_is_stated(self):
        # 筛查级监测的结论不能写成场地级验证
        d = reports.build_regulatory(
            _alarm_report(), _reg(), period=PERIOD, permit_id="CCS-2026-001"
        )
        self.assertIn("不构成泄漏认定", d.body)
        self.assertIn("筛查级", d.body)

    def test_generated_report_passes_its_own_check(self):
        d = reports.build_regulatory(
            _alarm_report(), _reg(), period=PERIOD, permit_id="CCS-2026-001"
        )
        self.assertEqual(d.problems, [], [p.render() for p in d.problems])


class ApprovalFlowTests(unittest.TestCase):
    def _draft(self):
        return reports.build_daily(_alarm_report(), _reg(), period=PERIOD)

    def test_happy_path(self):
        d = self._draft()
        reports.submit(d, actor="值班员")
        self.assertEqual(d.status, reports.STATUS_PENDING)
        reports.approve(d, reviewer="张工", note="已核对")
        self.assertEqual(d.status, reports.STATUS_APPROVED)
        self.assertEqual(len(d.trail), 2)

    def test_draft_cannot_be_approved_directly(self):
        # 草稿直接批准等于没有审批环节，而这正是最容易发生的"省一步"
        d = self._draft()
        with self.assertRaises(ValueError) as ctx:
            reports.approve(d, reviewer="张工")
        self.assertIn("不允许", str(ctx.exception))

    def test_submit_is_blocked_by_citation_problems(self):
        d = self._draft()
        d.problems = [reports.CitationProblem("unsourced", "15", 1, "…15%…")]
        with self.assertRaises(ValueError) as ctx:
            reports.submit(d)
        self.assertIn("拒绝提交", str(ctx.exception))
        self.assertEqual(d.status, reports.STATUS_DRAFT)

    def test_approve_requires_reviewer(self):
        d = self._draft()
        reports.submit(d)
        with self.assertRaises(ValueError):
            reports.approve(d, reviewer="")

    def test_reject_requires_note(self):
        # 没有意见的驳回，起草方无法知道要改什么，下一稿会原样再来一遍
        d = self._draft()
        reports.submit(d)
        with self.assertRaises(ValueError) as ctx:
            reports.reject(d, reviewer="张工", note="  ")
        self.assertIn("驳回必须写意见", str(ctx.exception))

    def test_reject_then_reopen_keeps_trail(self):
        d = self._draft()
        reports.submit(d)
        reports.reject(d, reviewer="张工", note="第三节归因缺少排除理由")
        self.assertEqual(d.status, reports.STATUS_REJECTED)
        reports.reopen(d)
        self.assertEqual(d.status, reports.STATUS_DRAFT)
        self.assertEqual(len(d.trail), 3)
        self.assertIn("第三节", d.trail[1]["note"])

    def test_approved_is_terminal(self):
        d = self._draft()
        reports.submit(d)
        reports.approve(d, reviewer="张工")
        with self.assertRaises(ValueError):
            reports.submit(d)

    def test_transition_table_covers_every_status(self):
        # 漏一条分支的后果是"某个状态无法离开"，而那种 bug 只会在真实审批时才发现
        for status in (reports.STATUS_DRAFT, reports.STATUS_PENDING,
                       reports.STATUS_REJECTED, reports.STATUS_APPROVED):
            with self.subTest(status=status):
                self.assertIn(status, reports.TRANSITIONS)


class WeeklyReportTests(unittest.TestCase):
    def test_requires_at_least_one_period(self):
        # 空列表会让周报看起来"一切正常"
        with self.assertRaises(ValueError):
            reports.build_weekly([], _reg(), period=PERIOD)

    def test_uses_count_distribution_not_average_severity(self):
        # "一天 critical、六天正常" 与 "七天都 warn" 的平均严重度一样，
        # 但处置完全不同。所以周报只报分布
        rep = _alarm_report()
        d = reports.build_weekly([rep, rep], _reg(), period=PERIOD)
        self.assertIn("最高等级分布", d.body)
        self.assertEqual(d.problems, [], [p.render() for p in d.problems])


class TaskSplitTests(unittest.TestCase):
    def test_minimal_request_only_runs_alarms(self):
        req = supervisor.OpsRequest(
            readings=_series("wellhead_p", [70]), period=PERIOD,
            want_condition=False, want_report=False,
        )
        self.assertEqual([t.agent for t in req.tasks()], ["alarms"])

    def test_condition_and_report_add_tasks(self):
        req = supervisor.OpsRequest(
            readings=_series("wellhead_p", [70]), period=PERIOD,
            question="最近 1 小时井口压力", want_condition=True, want_report=True,
        )
        self.assertEqual(
            [t.agent for t in req.tasks()], ["alarms", "condition", "reports"]
        )

    def test_every_task_has_a_reason(self):
        req = supervisor.OpsRequest(
            readings=_series("wellhead_p", [70]), period=PERIOD,
            question="q", want_condition=True, want_report=True,
        )
        for t in req.tasks():
            self.assertTrue(t.reason.strip())


class ConflictTests(unittest.TestCase):
    def test_value_conflict_detected(self):
        # 告警取最新点（107，T0+5min），工况问答的窗口截止在 T0（末值 105）。
        # 这不是 bug，而是"报告里的数字取决于窗口选择"这一事实，
        # 必须让人看见 —— 否则两份文件各说各话而读者无从判断
        rs = _series("wellhead_p", [105, 107])
        rep = alarms.evaluate(rs, _reg())
        cond = __import__("ops.condition", fromlist=["x"]).answer(
            "最近 1 小时井口压力", rs, _reg(), now=T0,
        )
        self.assertEqual(rep.alarms[0].value, 107.0)
        self.assertEqual(cond.stats[0].last, 105.0)
        conflicts = supervisor.detect_value_conflicts(rep, cond)
        self.assertTrue(conflicts)
        self.assertTrue(conflicts[0].blocking)

    def test_no_value_conflict_when_same_window(self):
        rs = _series("wellhead_p", [105, 107])
        rep = alarms.evaluate(rs, _reg())
        cond = __import__("ops.condition", fromlist=["x"]).answer(
            "最近 1 小时井口压力", rs, _reg()
        )
        self.assertEqual(supervisor.detect_value_conflicts(rep, cond), [])

    def test_coupling_deviation_is_not_blocking(self):
        # 耦合表写的是"通常如何"。反向变化往往意味着有另一个驱动源，
        # 那是有价值的信息，不是错误。把这类也阻断，会让系统在真实工况
        # 变化时拒绝出报告 —— 而那正是最需要报告的时候
        reg = build_registry(
            {
                "capture_rate": {"warn": 88, "alert": 82, "critical": 75, "source": "可研"},
                "pipeline_outlet_p": {"warn": 90, "alert": 110, "critical": 130, "source": "管道"},
            },
            origin="测试",
        )
        rs = _series("capture_rate", [78]) + _series("pipeline_outlet_p", [115])
        conflicts = supervisor.detect_conflicts(alarms.evaluate(rs, reg))
        coupling = [c for c in conflicts if c.kind == supervisor.CONFLICT_COUPLING]
        self.assertTrue(coupling)
        self.assertFalse(coupling[0].blocking)
        self.assertIn("额外驱动源", coupling[0].render())

    def test_blocking_conflicts_sort_first(self):
        reg = build_registry(
            {
                "capture_rate": {"warn": 88, "alert": 82, "critical": 75, "source": "可研"},
                "pipeline_outlet_p": {"warn": 90, "alert": 110, "critical": 130, "source": "管道"},
                "wellhead_p": {"warn": 80, "alert": 100, "critical": 120, "source": "设计"},
            },
            origin="测试",
        )
        rs = (
            _series("capture_rate", [78])
            + _series("pipeline_outlet_p", [115])
            + _series("wellhead_p", [110])
        )
        conflicts = supervisor.detect_conflicts(alarms.evaluate(rs, reg))
        if len(conflicts) >= 2:
            flags = [c.blocking for c in conflicts]
            self.assertEqual(flags, sorted(flags, reverse=True))

    def test_orphan_attribution_detects_stale_cache(self):
        # 单次运行内这条检测恒为真；它防的是把归因缓存下来跨批次复用 ——
        # 那时告警状态已重算、归因还是旧的，解释读起来合理但无依据
        reg = build_registry(
            {
                "capture_rate": {"warn": 88, "alert": 82, "critical": 75, "source": "可研"},
                "pipeline_outlet_p": {"warn": 90, "alert": 110, "critical": 130, "source": "管道"},
                "pipeline_dp": {"warn": 15, "alert": 25, "critical": 35, "source": "管道"},
            },
            origin="测试",
        )
        rs = (
            _series("capture_rate", [78])
            + _series("pipeline_outlet_p", [115])
            + _series("pipeline_dp", [12])
        )
        rep = alarms.evaluate(rs, reg)
        self.assertEqual(supervisor.detect_orphan_attributions(rep), [])
        # 模拟陈旧：把 pipeline_dp 的告警丢掉，归因仍声称它 ok
        rep.alarms = [a for a in rep.alarms if a.sensor != "pipeline_dp"]
        stale = supervisor.detect_orphan_attributions(rep)
        self.assertTrue(stale)
        self.assertTrue(stale[0].blocking)


class OrchestrationTests(unittest.TestCase):
    def _request(self, **kw):
        rs = (
            _series("wellhead_p", [105, 106, 107])
            + _series("injection_rate", [140])
            + _series("injectivity_index", [3.5])
        )
        base = dict(readings=rs, period=PERIOD)
        base.update(kw)
        return supervisor.OpsRequest(**base)

    def test_full_run_produces_all_outputs(self):
        res = supervisor.run(
            self._request(
                question="最近 1 小时井口压力", want_condition=True,
                want_report=True, report_kind="daily",
            ),
            _reg(),
        )
        self.assertIsNotNone(res.alarms)
        self.assertIsNotNone(res.condition)
        self.assertIsNotNone(res.report)
        self.assertTrue(all(r.ok for r in res.runs))

    def test_can_submit_requires_no_blocking_conflict(self):
        res = supervisor.run(self._request(want_report=True), _reg())
        if res.blocking:
            self.assertFalse(res.can_submit)
        else:
            self.assertTrue(res.can_submit)

    def test_weekly_kind_fails_explicitly(self):
        # 用一期冒充一周会让周报的"分布"字段只有一个样本
        res = supervisor.run(
            self._request(want_report=True, report_kind="weekly"), _reg()
        )
        self.assertIsNone(res.report)
        report_run = [r for r in res.runs if r.agent == "reports"][0]
        self.assertFalse(report_run.ok)
        self.assertIn("多期", report_run.error)

    def test_report_skipped_when_alarms_failed(self):
        bad = supervisor.OpsRequest(
            readings=[Reading("nonexistent", 1.0, T0)], period=PERIOD,
            want_report=True,
        )
        res = supervisor.run(bad, _reg())
        self.assertIsNone(res.alarms)
        self.assertIsNone(res.report)
        self.assertFalse([r for r in res.runs if r.agent == "alarms"][0].ok)

    def test_brief_puts_conflicts_before_conclusions(self):
        # 读者如果只看前三行，看到的应该是"这份材料目前不能发"以及为什么
        res = supervisor.run(self._request(want_report=True), _reg())
        brief = supervisor.render_brief(res)
        self.assertIn("运维简报", brief)
        if res.blocking:
            self.assertLess(brief.index("阻断性冲突"), brief.index("## 告警"))

    def test_result_is_json_safe(self):
        import json
        res = supervisor.run(self._request(want_report=True), _reg())
        json.dumps(res.to_dict(), ensure_ascii=False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
