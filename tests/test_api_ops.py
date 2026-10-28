"""
运维接口（api_ops）的测试
=========================
这一组测试要守住的东西，按重要性排：

1. **鉴权没漏**。``api_ops.router`` 自己不带鉴权（不能反向导入
   ``require_token``，否则成环），鉴权靠 ``api_server`` 的 ``include_router``
   施加。漏写 ``dependencies`` 不会报错，只会让 ``/api/ops/*`` 裸奔 ——
   所以这里有一条静态断言直接读源码。
2. **失败不伪装成成功**。未知测点、NaN、无数据、未署名的批准、无意见的驳回，
   都必须是非 2xx，且消息能告诉调用方怎么改。
3. **审批能跨请求**。报告落盘之后，换一个连接、换一个进程都还读得到。
   内存里的草稿在真实审批流里等于没有。
4. **不可判定不是正常**。没配阈值的测点必须出现在 ``missing`` 里，
   而不是被折进 normal。

测试不导入 ``api_server``：那会拉起 LangGraph 全图与 2.2 GB 重排模型，
在 CI 的零依赖 job 里跑不了。这也是 ``api_ops`` 单独成模块的原因。
"""
from __future__ import annotations

import os
import re
import shutil
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from fastapi.testclient import TestClient

import api_ops
from ops import reports as ops_reports
from ops import store as ops_store
from ops.thresholds import build_registry

T0 = datetime(2026, 10, 27, 9, 0)

#: 只配三个测点，故意留一大片盲区 —— 用来验证 missing 不是空话
REG = build_registry(
    {
        "annulus_p": {"warn": 9.5, "alert": 10.0, "critical": 12.0, "source": "设计文件 §4.3"},
        "wellhead_p": {"warn": 80.0, "alert": 85.0, "critical": 90.0, "source": "设计文件 §4.2"},
        "capture_rate": {"warn": 90.0, "alert": 88.0, "critical": 85.0, "source": "能效考核指标"},
    },
    origin="测试注入",
)


def _app() -> FastAPI:
    app = FastAPI()
    app.include_router(api_ops.router)
    return app


def _rows(sensor: str, values, *, step_min: int = 5, quality: str = "good"):
    return [
        {
            "sensor": sensor,
            "value": v,
            "ts": (T0 + timedelta(minutes=step_min * i)).isoformat(),
            "quality": quality,
        }
        for i, v in enumerate(values)
    ]


class OpsApiTestBase(unittest.TestCase):
    """把报告库指到临时目录，并固定阈值表。

    每个测试一份独立的 db 文件：审批链路的测试会写库，共用一个文件时
    前一个测试的草稿会出现在后一个的 ``list`` 结果里。
    """

    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="ops_api_")
        self._old_db = ops_store.OPS_DB
        ops_store.OPS_DB = os.path.join(self._tmp, "ops_reports.db")
        self._patch = mock.patch("ops.thresholds.load_registry", return_value=REG)
        self._patch.start()
        self.client = TestClient(_app())

    def tearDown(self):
        self._patch.stop()
        ops_store.OPS_DB = self._old_db
        shutil.rmtree(self._tmp, ignore_errors=True)


# ══════════════════════════════════════════════════════════
# 一、鉴权没漏
# ══════════════════════════════════════════════════════════
class TestAuthWiring(unittest.TestCase):
    def setUp(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(root, "api_server.py"), encoding="utf-8") as fh:
            self.source = fh.read()

    def test_ops_router_is_included_with_token_dependency(self):
        """漏掉 dependencies 不会报错，只会让运维接口裸奔。"""
        lines = [ln for ln in self.source.split("\n") if "include_router(ops_router" in ln]
        self.assertEqual(
            len(lines), 1,
            "api_server.py 应当只有一处挂载 ops_router；多处挂载会让鉴权口径不一致",
        )
        self.assertIn(
            "Depends(require_token)", lines[0],
            "ops_router 必须带鉴权依赖。不带也能跑，但 /api/ops/* 就变成匿名可写了",
        )

    def test_ops_module_does_not_import_api_server(self):
        """反向导入会成环；这条断言把环挡在导入期之外。"""
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(root, "api_ops.py"), encoding="utf-8") as fh:
            src = fh.read()
        self.assertIsNone(
            re.search(r"^\s*(from|import)\s+api_server", src, re.M),
            "api_ops.py 不能导入 api_server（会成环，且会把 2.2 GB 模型拖进接口测试）",
        )


# ══════════════════════════════════════════════════════════
# 二、阈值与盲区
# ══════════════════════════════════════════════════════════
class TestThresholdsEndpoint(OpsApiTestBase):
    def test_missing_lists_unconfigured_sensors(self):
        r = self.client.get("/api/ops/thresholds")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertIn("annulus_p", body["configured"])
        self.assertIn(
            "soil_co2_flux", body["missing"],
            "没配阈值的测点必须出现在 missing 里；折进 configured 会让它看起来可判定",
        )
        self.assertEqual(body["origin"], "测试注入")

    def test_sensor_catalog_carries_unit_and_direction(self):
        body = self.client.get("/api/ops/thresholds").json()
        by_key = {s["key"]: s for s in body["sensors"]}
        self.assertEqual(by_key["capture_rate"]["direction"], "low")
        self.assertEqual(by_key["annulus_p"]["unit"], "bar")
        self.assertIsNotNone(by_key["annulus_p"]["threshold"])
        self.assertIsNone(
            by_key["soil_co2_flux"]["threshold"],
            "未配阈值的测点 threshold 必须是 null，不能给个默认值",
        )


# ══════════════════════════════════════════════════════════
# 三、告警
# ══════════════════════════════════════════════════════════
class TestAlarmsEndpoint(OpsApiTestBase):
    def test_critical_alarm_returned(self):
        r = self.client.post(
            "/api/ops/alarms", json={"readings": _rows("annulus_p", [11.0, 11.5, 12.4])}
        )
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertEqual(body["top_severity"], "critical")
        alarm = next(a for a in body["alarms"] if a["sensor"] == "annulus_p")
        self.assertEqual(alarm["severity"], "critical")
        self.assertTrue(alarm["rationale"], "每条告警都要能回答'为什么是这一级'")
        self.assertTrue(alarm["provenance"], "阈值出处必须随告警返回")
        self.assertIn("period", body, "响应要带回时间范围，否则读者不知道看的是哪一段")

    def test_period_inferred_from_readings(self):
        body = self.client.post(
            "/api/ops/alarms", json={"readings": _rows("annulus_p", [1.0, 2.0], step_min=10)}
        ).json()
        self.assertEqual(body["period"][0], T0.isoformat())
        self.assertEqual(body["period"][1], (T0 + timedelta(minutes=10)).isoformat())

    def test_unknown_sensor_is_rejected_with_its_name(self):
        r = self.client.post(
            "/api/ops/alarms", json={"readings": _rows("no_such_sensor", [1.0])}
        )
        self.assertEqual(r.status_code, 400)
        self.assertIn(
            "no_such_sensor", r.json()["detail"],
            "报错必须点出是哪个测点：只说'未知测点'等于让人一个个试",
        )

    def test_nan_is_rejected(self):
        """NaN 与任何数比较都为假，会被当成'没超阈值'。必须在入口拦掉。"""
        r = self.client.post(
            "/api/ops/alarms",
            json={"readings": [{"sensor": "annulus_p", "value": "NaN",
                                "ts": T0.isoformat(), "quality": "good"}]},
        )
        self.assertEqual(r.status_code, 400)

    def test_bad_quality_is_rejected(self):
        r = self.client.post(
            "/api/ops/alarms", json={"readings": _rows("annulus_p", [1.0], quality="ok")}
        )
        self.assertEqual(r.status_code, 400)
        self.assertIn("quality", r.json()["detail"])

    def test_empty_readings_without_period_is_rejected(self):
        r = self.client.post("/api/ops/alarms", json={"readings": []})
        self.assertEqual(
            r.status_code, 400,
            "没数据时必须报错。返回 200 + 空告警会被渲染成'运行正常'",
        )

    def test_reversed_period_is_rejected(self):
        r = self.client.post(
            "/api/ops/alarms",
            json={
                "readings": _rows("annulus_p", [1.0]),
                "period": {
                    "start": (T0 + timedelta(hours=1)).isoformat(),
                    "end": T0.isoformat(),
                },
            },
        )
        self.assertEqual(r.status_code, 400)

    def test_policy_parameters_are_validated(self):
        r = self.client.post(
            "/api/ops/alarms",
            json={"readings": _rows("annulus_p", [1.0]), "persist_points": 0},
        )
        self.assertEqual(
            r.status_code, 400,
            "persist_points=0 会让 R_PERSIST 永远成立，等于对所有告警无差别升级",
        )

    def test_misspelled_field_is_rejected_not_ignored(self):
        """拼错字段名必须 422，不能变成"200 + 空读数"。

        Pydantic 默认**忽略**未知字段，于是 ``{"readngs": [...]}`` 会通过校验、
        读数列表为空，接着被 ``_ops_period`` 判成"没有数据" —— 一个 400，
        消息还指着数据源。人就去查采集器了，而问题在请求体里少了一个 ``i``。

        这条测试盯的是 ``model_config = ConfigDict(extra="forbid")``。
        它是一行**没有调用点**的配置：删掉之后所有别的测试照样绿，
        只有这条会红。而它恰恰是最容易在"简化模型"时被顺手删掉的那种。
        """
        r = self.client.post(
            "/api/ops/alarms",
            json={
                "readngs": [
                    {"sensor": "wellhead_p", "value": 90.0, "ts": T0.isoformat()}
                ]
            },
        )
        self.assertEqual(r.status_code, 422, r.text)
        self.assertIn(
            "readngs", r.text,
            "422 的 detail 里要出现那个拼错的字段名，否则调用方仍不知道错在哪",
        )

    def test_unknown_field_inside_a_reading_is_rejected(self):
        """嵌套模型同样 forbid：读数的字段拼错不该被静默丢弃。"""
        r = self.client.post(
            "/api/ops/alarms",
            json={
                "readings": [
                    {"sensor": "wellhead_p", "value": 90.0,
                     "ts": T0.isoformat(), "qualty": "good"}
                ]
            },
        )
        self.assertEqual(
            r.status_code, 422,
            "quality 拼错若被忽略，会退回默认 good —— 一条坏读数被当成可信读数",
        )


# ══════════════════════════════════════════════════════════
# 四、工况问答
# ══════════════════════════════════════════════════════════
class TestConditionEndpoint(OpsApiTestBase):
    def test_answer_has_stats_and_template_text(self):
        r = self.client.post(
            "/api/ops/condition",
            json={
                "question": "最近一小时环空压力怎么样",
                "readings": _rows("annulus_p", [11.0, 11.5, 12.4], step_min=20),
            },
        )
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertEqual(body["sensors"], ["annulus_p"])
        self.assertTrue(body["stats"], "答不出数值等于没答")
        self.assertTrue(body["text"])
        self.assertEqual(
            body["unsourced_numbers"], [],
            "模板回答的数字都直接来自 stats，不应出现无出处数字",
        )

    def test_unresolvable_question_is_reported_not_guessed(self):
        body = self.client.post(
            "/api/ops/condition",
            json={"question": "今天天气怎么样", "readings": _rows("annulus_p", [1.0])},
        ).json()
        self.assertEqual(body["sensors"], [])
        self.assertTrue(
            body["unresolved"],
            "解析不出测点时必须写进 unresolved，不能静默套用全部测点",
        )


# ══════════════════════════════════════════════════════════
# 五、全链编排与报告落盘
# ══════════════════════════════════════════════════════════
class TestRunEndpoint(OpsApiTestBase):
    def _run(self, **kw):
        payload = {"readings": _rows("annulus_p", [11.0, 11.5, 12.4], step_min=20)}
        payload.update(kw)
        return self.client.post("/api/ops/run", json=payload)

    def test_daily_report_is_persisted_and_readable(self):
        r = self._run(want_report=True, report_kind="daily")
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertTrue(body["draft_id"], "报告必须落盘并返回草稿号，否则无法跨请求审批")
        self.assertIn("brief", body)

        detail = self.client.get(f"/api/ops/reports/{body['draft_id']}")
        self.assertEqual(detail.status_code, 200)
        self.assertEqual(detail.json()["status"], "draft")

    def test_detail_carries_its_own_id_and_allowed_actions(self):
        """详情必须自带草稿号与可执行动作。

        两个都是**前端算不出来**的：

        - 没有 `draft_id`，调用方只能"记住自己刚才问了哪个号"。详情响应
          拿去做一次刷新，号就丢了，而后续每个审批请求都按号寻址。
        - `allowed` 如果让前端自己抄一张表，就有了两份真值。后端将来加一条
          迁移时前端不会跟着变，表现是按钮点不动或点下去 409 —— 都不报错。
        """
        draft_id = self._run(want_report=True, report_kind="daily").json()["draft_id"]
        detail = self.client.get(f"/api/ops/reports/{draft_id}").json()
        self.assertEqual(detail["draft_id"], draft_id)
        self.assertEqual(detail["allowed"], ["submit"])

        # 走完一次完整流转，`allowed` 必须跟着状态变 —— 否则前端会一直
        # 显示"提交审批"，而它在待审批状态下必然失败。
        self.client.post(f"/api/ops/reports/{draft_id}/submit", json={"actor": "早班"})
        pending = self.client.get(f"/api/ops/reports/{draft_id}").json()
        self.assertEqual(pending["status"], "pending_review")
        self.assertEqual(sorted(pending["allowed"]), ["approve", "reject"])

        self.client.post(
            f"/api/ops/reports/{draft_id}/approve", json={"reviewer": "张工"}
        )
        approved = self.client.get(f"/api/ops/reports/{draft_id}").json()
        self.assertEqual(approved["allowed"], [], "已批准是终态，不该还有可执行动作")

    def test_post_endpoints_also_return_id_and_allowed(self):
        """四个 POST 的返回值与 GET 详情同构。

        不同构的话，前端每做完一个动作就得再发一次 GET 才知道新状态下的
        按钮该长什么样 —— 而那一次 GET 失败时，界面会停在"刚点完但按钮没变"，
        看起来像点击没生效。
        """
        draft_id = self._run(want_report=True, report_kind="daily").json()["draft_id"]
        submitted = self.client.post(
            f"/api/ops/reports/{draft_id}/submit", json={"actor": "早班"}
        ).json()
        self.assertEqual(submitted["draft_id"], draft_id)
        self.assertEqual(sorted(submitted["allowed"]), ["approve", "reject"])
        self.assertEqual(len(submitted["trail"]), 1)

    def test_report_without_permit_id_is_rejected(self):
        r = self._run(want_report=True, report_kind="regulatory")
        self.assertEqual(r.status_code, 400)
        self.assertIn("permit_id", r.json()["detail"])

    def test_regulatory_with_permit_id_works(self):
        r = self._run(want_report=True, report_kind="regulatory", permit_id="CCS-2026-0417")
        self.assertEqual(r.status_code, 200, r.text)
        detail = self.client.get(f"/api/ops/reports/{r.json()['draft_id']}").json()
        self.assertEqual(detail["permit_id"], "CCS-2026-0417")

    def test_weekly_with_one_period_does_not_fake_a_report(self):
        r = self._run(want_report=True, report_kind="weekly")
        body = r.json()
        self.assertEqual(body["draft_id"], "")
        report_run = next(x for x in body["runs"] if x["agent"] == "reports")
        self.assertFalse(report_run["ok"])
        self.assertIn("2 期", report_run["error"])

    def test_weekly_with_two_periods_builds_a_report(self):
        p1 = {"start": T0.isoformat(), "end": (T0 + timedelta(minutes=20)).isoformat()}
        p2 = {"start": (T0 + timedelta(hours=2)).isoformat(),
              "end": (T0 + timedelta(hours=2, minutes=20)).isoformat()}
        r = self._run(want_report=True, report_kind="weekly", periods=[p1, p2])
        self.assertEqual(r.status_code, 200, r.text)
        self.assertTrue(r.json()["draft_id"], "两期数据应当能生成周报")

    def test_unknown_report_kind_is_rejected(self):
        r = self._run(want_report=True, report_kind="monthly")
        self.assertEqual(r.status_code, 400)

    def test_blocking_conflict_stops_submission(self):
        """两路读数给出不同值时必须阻断，且 can_submit 为假。"""
        rows = _rows("annulus_p", [11.0, 11.5, 12.4], step_min=20)
        r = self.client.post(
            "/api/ops/run",
            json={
                "readings": rows,
                "question": "环空压力现在多少",
                "want_condition": True,
                "want_report": True,
                "now": (T0 + timedelta(minutes=5)).isoformat(),
            },
        )
        body = r.json()
        kinds = {c["kind"] for c in body["conflicts"]}
        self.assertIn(
            "contradiction", kinds,
            "告警取最新点、工况问答取窗口末值，两者不一致时必须作为阻断冲突报出",
        )
        self.assertFalse(body["can_submit"], "有阻断冲突时不允许提交审批")


# ══════════════════════════════════════════════════════════
# 六、审批状态机（跨请求）
# ══════════════════════════════════════════════════════════
class TestApprovalFlow(OpsApiTestBase):
    def _make_draft(self) -> str:
        r = self.client.post(
            "/api/ops/run",
            json={
                "readings": _rows("annulus_p", [11.0, 11.5, 12.4], step_min=20),
                "want_report": True,
                "report_kind": "daily",
            },
        )
        draft_id = r.json()["draft_id"]
        self.assertTrue(draft_id)
        return draft_id

    def test_submit_then_approve_survives_reload(self):
        draft_id = self._make_draft()
        r = self.client.post(f"/api/ops/reports/{draft_id}/submit", json={"actor": "早班"})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["status"], "pending_review")

        r = self.client.post(
            f"/api/ops/reports/{draft_id}/approve",
            json={"reviewer": "工艺主管 张", "note": "已核对环空压力趋势"},
        )
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["status"], "approved")

        # 模拟"服务重启"：绕过接口直接读库
        reloaded = ops_store.load(draft_id)
        self.assertIsNotNone(reloaded)
        self.assertEqual(reloaded.status, "approved")
        self.assertEqual(len(reloaded.trail), 2, "提交与批准两条轨迹都必须留下")
        self.assertEqual(reloaded.trail[-1]["actor"], "工艺主管 张")

    def test_approve_without_reviewer_is_rejected(self):
        draft_id = self._make_draft()
        self.client.post(f"/api/ops/reports/{draft_id}/submit", json={})
        r = self.client.post(
            f"/api/ops/reports/{draft_id}/approve", json={"reviewer": "  "}
        )
        self.assertEqual(r.status_code, 400)
        self.assertIn("审批人", r.json()["detail"])

    def test_approve_before_submit_is_conflict(self):
        draft_id = self._make_draft()
        r = self.client.post(
            f"/api/ops/reports/{draft_id}/approve", json={"reviewer": "张"}
        )
        self.assertEqual(
            r.status_code, 409,
            "草稿不能直接批准。允许的话，'待审批'这个状态就形同虚设",
        )

    def test_reject_requires_comment(self):
        draft_id = self._make_draft()
        self.client.post(f"/api/ops/reports/{draft_id}/submit", json={})
        r = self.client.post(
            f"/api/ops/reports/{draft_id}/reject", json={"reviewer": "张", "note": ""}
        )
        self.assertEqual(r.status_code, 400)
        self.assertIn("意见", r.json()["detail"])

    def test_reject_then_reopen(self):
        draft_id = self._make_draft()
        self.client.post(f"/api/ops/reports/{draft_id}/submit", json={})
        r = self.client.post(
            f"/api/ops/reports/{draft_id}/reject",
            json={"reviewer": "张", "note": "缺少盲区说明"},
        )
        self.assertEqual(r.json()["status"], "rejected")

        r = self.client.post(f"/api/ops/reports/{draft_id}/reopen", json={"actor": "早班"})
        self.assertEqual(r.json()["status"], "draft")
        self.assertEqual(len(r.json()["trail"]), 3)

    def test_submit_blocked_by_citation_problems(self):
        """引用检查不通过时必须拒绝提交，并把问题逐条返回。"""
        draft_id = self._make_draft()
        draft = ops_store.load(draft_id)
        draft.body += "\n<!-- strict -->\n- 环空压力 999.9 bar\n<!-- /strict -->\n"
        draft.problems = ops_reports.check_citations(draft.body, draft.citations)
        ops_store.save(draft_id, draft)
        self.assertTrue(draft.problems, "构造的坏稿子应当产生引用问题")

        r = self.client.post(f"/api/ops/reports/{draft_id}/submit", json={})
        self.assertEqual(r.status_code, 409)
        detail = r.json()["detail"]
        self.assertIn("problems", detail)
        self.assertTrue(detail["problems"])
        self.assertIn("999.9", " ".join(detail["problems"]))

    def test_missing_draft_is_404(self):
        self.assertEqual(self.client.get("/api/ops/reports/nope").status_code, 404)
        self.assertEqual(
            self.client.post("/api/ops/reports/nope/submit", json={}).status_code, 404
        )

    def test_list_returns_summaries(self):
        draft_id = self._make_draft()
        body = self.client.get("/api/ops/reports?limit=5").json()
        self.assertGreaterEqual(body["count"], 1)
        row = next(r for r in body["reports"] if r["id"] == draft_id)
        self.assertEqual(row["kind"], "daily")
        self.assertNotIn("body", row, "列表不返回正文，正文可能几十 KB")


if __name__ == "__main__":
    unittest.main(verbosity=2)
