# -*- coding: utf-8 -*-
"""
``utils/reservoir_backend.py`` 的回归测试。

这个模块最容易变成"空壳"——写一个 ``NotImplementedError`` 的适配器，文档上
看起来支持 ECLIPSE，实际一行也算不出来。测试因此重点钉两件事：

1. **契约真的被用**：解析后端产出的结果必须过 ``SimulationResult.validate()``，
   也就是契约不是只有文档在声称，而是两侧都在执行。
2. **不收敛必须拒绝**：模拟器失败时数字仍然存在、看起来完全正常，
   把它当结论写进报告等于用噪声做安全判定。
"""
import json
import os
import re
import sys
import tempfile
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils import reservoir_backend as rb  # noqa: E402


def _case(**kw):
    base = dict(name="A1", depth_m=2500, injection_pressure_bar=300,
                temperature_celsius=80, injection_rate_m3_per_day=1000)
    base.update(kw)
    return rb.SimulationCase(**base)


def _valid_payload(**kw):
    base = {
        "schema": rb.SCHEMA,
        "case": rb.asdict(_case()),
        "backend": "eclipse-2021",
        "converged": True,
        "units": dict(rb.REQUIRED_UNITS),
        "time_days": [0, 30, 60],
        "co2_saturation": [0.0, 0.1, 0.2],
        "pressure_bar": [260, 270, 280],
        "co2_column_height_m": 12.0,
        "entry_pressure_bar": 43.3,
        "notes": ["demo"],
    }
    base.update(kw)
    return base


class _TempJson:
    def __init__(self, payload):
        self.dir = tempfile.mkdtemp(prefix="simres-")
        self.path = os.path.join(self.dir, "result.json")
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False)


class CaseValidationTests(unittest.TestCase):
    def test_valid_case_has_no_problems(self):
        self.assertEqual(_case().validate(), [])

    def test_all_problems_are_listed_at_once(self):
        """一次列全，避免改一个跑一次。"""
        case = rb.SimulationCase(
            name="", depth_m=5, injection_pressure_bar=-1, temperature_celsius=500,
            porosity=0.9, permeability_md=0, plume_radius_m=0,
        )
        problems = case.validate()
        self.assertGreaterEqual(len(problems), 6)

    def test_depth_outside_range_is_rejected(self):
        self.assertTrue(_case(depth_m=20000).validate())


class ContractTests(unittest.TestCase):
    """结果契约本身 —— 两个后端共用。"""

    def _result(self, **kw):
        base = dict(
            backend="x", kind="external", converged=True,
            units=dict(rb.REQUIRED_UNITS),
            time_days=[0, 1, 2], co2_saturation=[0.0, 0.1, 0.2],
            pressure_bar=[200, 210, 220],
        )
        base.update(kw)
        return rb.SimulationResult(**base)

    def test_clean_result_passes(self):
        self.assertEqual(self._result().validate(), [])

    def test_non_converged_is_rejected(self):
        """
        不收敛时数字仍在、看起来正常 —— 必须拒绝而不是警告。
        这是 fail-closed，与质量闸门同一个原则。
        """
        problems = self._result(converged=False).validate()
        self.assertTrue(any("未收敛" in p for p in problems))

    def test_units_must_be_declared_explicitly(self):
        """缺单位必须拒绝：psi 与 bar 差 14.5 倍，而两个数字都"看起来合理"。"""
        problems = self._result(units={}).validate()
        self.assertTrue(any("单位" in p for p in problems))

    def test_wrong_pressure_unit_is_caught(self):
        units = dict(rb.REQUIRED_UNITS, pressure="psi")
        problems = self._result(units=units).validate()
        self.assertTrue(any("pressure" in p for p in problems))

    def test_percent_saturation_is_caught(self):
        problems = self._result(co2_saturation=[0.0, 10.0, 20.0]).validate()
        self.assertTrue(any("饱和度" in p for p in problems))

    def test_time_must_be_strictly_increasing(self):
        problems = self._result(time_days=[0, 60, 30]).validate()
        self.assertTrue(any("递增" in p for p in problems))

    def test_length_mismatch_is_caught(self):
        problems = self._result(co2_saturation=[0.0, 0.1]).validate()
        self.assertTrue(any("长度" in p for p in problems))

    def test_negative_pressure_is_caught(self):
        problems = self._result(pressure_bar=[200, -5, 220]).validate()
        self.assertTrue(any("压力" in p for p in problems))


class ExternalResultSafetyTests(unittest.TestCase):
    def test_result_is_bound_to_the_exact_case(self):
        tmp = _TempJson(_valid_payload())
        other = _case(injection_pressure_bar=400)
        backend = rb.ExternalResultBackend(tmp.path)
        self.assertTrue(any("case" in s for s in backend.validate(other)))
        with self.assertRaises(rb.BackendUnavailable):
            backend.run(other)
        self.assertFalse(rb.run_case(other, external_path=tmp.path)["ok"])

    def test_old_schema_cannot_bypass_case_binding(self):
        tmp = _TempJson(_valid_payload(schema="ccus-sim-result/1"))
        self.assertTrue(rb.ExternalResultBackend(tmp.path).validate(_case()))

    def test_converged_string_is_not_true(self):
        tmp = _TempJson(_valid_payload(converged="false"))
        self.assertTrue(any("converged" in s for s in rb.ExternalResultBackend(tmp.path).validate(_case())))

    def test_nonfinite_numbers_and_malformed_arrays_are_rejected(self):
        for bad in ([300, float("nan")], [300, float("inf")], "bad-array", [True, 300]):
            with self.subTest(bad=bad):
                tmp = _TempJson(_valid_payload(pressure_bar=bad))
                self.assertTrue(rb.ExternalResultBackend(tmp.path).validate(_case()))
        self.assertTrue(_case(permeability_md=float("nan")).validate())

    def test_replaced_file_is_not_served_from_cache(self):
        tmp = _TempJson(_valid_payload())
        backend = rb.ExternalResultBackend(tmp.path)
        self.assertEqual(backend.validate(_case()), [])
        with open(tmp.path, "w", encoding="utf-8") as fh:
            json.dump(_valid_payload(converged=False), fh)
        self.assertTrue(backend.validate(_case()))
        with self.assertRaises(rb.BackendUnavailable):
            backend.run(_case())

    def test_probe_cannot_use_a_fabricated_case(self):
        tmp = _TempJson(_valid_payload())
        self.assertFalse(rb.available_backends(tmp.path)["external-json"]["available"])
        self.assertTrue(rb.available_backends(tmp.path, _case())["external-json"]["available"])


class AnalyticalBackendTests(unittest.TestCase):
    def setUp(self):
        self.backend = rb.AnalyticalScreeningBackend()
        self.case = _case()

    def test_output_satisfies_its_own_contract(self):
        """
        契约必须双向生效：参考实现自己产出的结果也要过校验。

        这条测试是"契约不是空话"的证据 —— 如果校验只对外部结果生效，
        那它就只是一段装饰。
        """
        result = self.backend.run(self.case)
        self.assertEqual(result.validate(), [])

    def test_saturation_grows_over_time(self):
        result = self.backend.run(self.case)
        sats = result.co2_saturation
        self.assertEqual(sats, sorted(sats))
        self.assertGreater(sats[-1], sats[0])

    def test_reports_what_it_is_not(self):
        """结果必须自带"我不是储层模拟器"的声明，否则会被引用成模拟器结论。"""
        result = self.backend.run(self.case)
        joined = " ".join(result.notes)
        self.assertIn("非储层模拟器", joined)
        self.assertIn("重力分异", joined)

    def test_invalid_case_raises_with_all_reasons(self):
        bad = rb.SimulationCase(name="", depth_m=1, injection_pressure_bar=0,
                                temperature_celsius=60)
        with self.assertRaises(ValueError) as ctx:
            self.backend.run(bad)
        self.assertIn("算例不合法", str(ctx.exception))

    def test_summary_mentions_backend(self):
        result = self.backend.run(self.case)
        self.assertIn(self.backend.name, self.backend.summarise(result))


class ExternalBackendTests(unittest.TestCase):
    def test_missing_file_says_what_to_do(self):
        backend = rb.ExternalResultBackend(os.path.join(tempfile.mkdtemp(), "nope.json"))
        problems = backend.validate(_case())
        joined = " ".join(problems)
        self.assertIn("找不到", joined)
        self.assertIn(rb.SCHEMA, joined)
        self.assertIn("simulator-adapter", joined)

    def test_valid_file_is_accepted(self):
        tmp = _TempJson(_valid_payload())
        backend = rb.ExternalResultBackend(tmp.path)
        self.assertEqual(backend.validate(_case()), [])
        result = backend.run(_case())
        self.assertEqual(result.backend, "eclipse-2021")
        self.assertEqual(result.kind, "external")

    def test_wrong_schema_is_rejected(self):
        tmp = _TempJson(_valid_payload(schema="something-else/9"))
        problems = rb.ExternalResultBackend(tmp.path).validate(_case())
        self.assertTrue(any("schema" in p for p in problems))

    def test_non_converged_external_result_is_rejected(self):
        tmp = _TempJson(_valid_payload(converged=False))
        problems = rb.ExternalResultBackend(tmp.path).validate(_case())
        self.assertTrue(any("未收敛" in p for p in problems))
        with self.assertRaises(rb.BackendUnavailable):
            rb.ExternalResultBackend(tmp.path).run(_case())

    def test_broken_json_is_reported_not_crashed(self):
        d = tempfile.mkdtemp()
        path = os.path.join(d, "bad.json")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("{not json")
        problems = rb.ExternalResultBackend(path).validate(_case())
        self.assertTrue(any("JSON" in p for p in problems))


class RunCaseTests(unittest.TestCase):
    def test_uses_analytical_when_no_external_path(self):
        out = rb.run_case(_case())
        self.assertTrue(out["ok"])
        self.assertEqual(out["backend"], "analytical-screening")
        self.assertFalse(out["degraded"])

    def test_requested_external_missing_fails_closed(self):
        missing = os.path.join(tempfile.mkdtemp(), "absent.json")
        out = rb.run_case(_case(), external_path=missing)
        self.assertFalse(out["ok"])
        self.assertIsNone(out["result"])
        self.assertIn("找不到", out["problems"][0])

    def test_explicit_fallback_names_the_degradation(self):
        missing = os.path.join(tempfile.mkdtemp(), "absent.json")
        out = rb.run_case(
            _case(), external_path=missing, allow_screening_fallback=True
        )
        self.assertTrue(out["ok"])
        self.assertTrue(out["degraded"])
        self.assertIn("找不到", out["degraded_reason"])
        self.assertEqual(out["backend"], "analytical-screening")

    def test_prefers_external_when_available(self):
        tmp = _TempJson(_valid_payload())
        out = rb.run_case(_case(), external_path=tmp.path)
        self.assertEqual(out["backend"], "eclipse-2021")
        self.assertFalse(out["degraded"])

    def test_invalid_case_is_rejected_before_any_backend_runs(self):
        out = rb.run_case(rb.SimulationCase(name="", depth_m=1,
                                            injection_pressure_bar=0,
                                            temperature_celsius=60))
        self.assertFalse(out["ok"])
        self.assertIsNone(out["backend"])
        self.assertTrue(out["problems"])


class RegistryTests(unittest.TestCase):
    def test_analytical_is_always_available(self):
        out = rb.available_backends()
        self.assertIn("analytical-screening", out)
        self.assertTrue(out["analytical-screening"]["available"])

    def test_external_reports_why_unavailable(self):
        missing = os.path.join(tempfile.mkdtemp(), "absent.json")
        out = rb.available_backends(missing)
        self.assertIn("external-json", out)
        self.assertFalse(out["external-json"]["available"])
        self.assertTrue(out["external-json"]["problems"])


class CrossModelComparisonTests(unittest.TestCase):
    """合成输入只测试机制，绝不计作真实油田或专业模拟器验证。"""

    def _payload(self, **overrides):
        result = rb.AnalyticalScreeningBackend().run(_case())
        payload = _valid_payload(
            time_days=[0.0] + result.time_days,
            pressure_bar=[result.pressure_bar[0]] + result.pressure_bar,
            co2_saturation=[result.co2_saturation[0]] + result.co2_saturation,
            comparison_basis={
                "pressure_bar": "injection_well_bottomhole",
                "co2_saturation": "same_control_volume_mean",
            },
        )
        payload.update(overrides)
        return payload

    def test_equal_synthetic_curves_compare_but_do_not_certify(self):
        tmp = _TempJson(self._payload())
        out = rb.compare_screening_to_external(
            _case(), tmp.path, pressure_tolerance_bar=0,
            saturation_tolerance=0,
        )
        self.assertEqual(out["status"], "compared")
        self.assertTrue(out["within_user_tolerances"])
        self.assertFalse(out["source_identity_verified"])
        self.assertIn("不构成场地级验证", out["limitations"])

    def test_disagreement_exceeds_declared_tolerance(self):
        payload = self._payload()
        payload["pressure_bar"] = [p + 5.0 for p in payload["pressure_bar"]]
        tmp = _TempJson(payload)
        out = rb.compare_screening_to_external(
            _case(), tmp.path, pressure_tolerance_bar=2.0,
            saturation_tolerance=0.1,
        )
        self.assertFalse(out["within_user_tolerances"])
        self.assertAlmostEqual(out["metrics"]["pressure_bar"]["max_abs_error"], 5.0)

    def test_time_extrapolation_and_missing_support_are_rejected(self):
        payload = self._payload()
        payload["time_days"] = payload["time_days"][:-1]
        payload["pressure_bar"] = payload["pressure_bar"][:-1]
        payload["co2_saturation"] = payload["co2_saturation"][:-1]
        tmp = _TempJson(payload)
        out = rb.compare_screening_to_external(
            _case(), tmp.path, pressure_tolerance_bar=1,
            saturation_tolerance=0.01,
        )
        self.assertEqual(out["status"], "not_comparable")
        tmp = _TempJson(self._payload(comparison_basis={}))
        out = rb.compare_screening_to_external(
            _case(), tmp.path, pressure_tolerance_bar=1,
            saturation_tolerance=0.01,
        )
        self.assertEqual(out["status"], "not_comparable")

    def test_wrong_case_and_missing_reference_do_not_compare(self):
        tmp = _TempJson(self._payload())
        out = rb.compare_screening_to_external(
            _case(injection_pressure_bar=400), tmp.path,
            pressure_tolerance_bar=1, saturation_tolerance=0.01,
        )
        self.assertEqual(out["status"], "rejected")
        with self.assertRaises(ValueError):
            rb.compare_screening_to_external(
                _case(), tmp.path, pressure_tolerance_bar=-1,
                saturation_tolerance=0.01,
            )


class DocumentedExampleTests(unittest.TestCase):
    """
    ``docs/simulator-adapter.md`` 里的示例 JSON 必须真的能通过契约。

    文档示例是最容易被复制的东西：适配器作者会照着它写。示例一旦与代码漂移，
    每个照着写的人都会撞上同一个错误，而错误信息会指向这份文档 —— 循环。
    """

    DOC_EXAMPLE = {
        "schema": rb.SCHEMA,
        "case": rb.asdict(_case()),
        "backend": "eclipse-2021",
        "converged": True,
        "units": {"pressure": "bar", "length": "m",
                  "time": "day", "saturation": "fraction"},
        "time_days": [0, 30, 90, 180, 365],
        "co2_saturation": [0.0, 0.12, 0.24, 0.31, 0.36],
        "pressure_bar": [253.0, 262.4, 271.9, 278.5, 283.1],
        "co2_column_height_m": 18.5,
        "entry_pressure_bar": 43.3,
        "notes": ["网格 50×50×10，相对渗透率取 Brooks-Corey"],
    }

    def test_documented_payload_passes_validation(self):
        tmp = _TempJson(self.DOC_EXAMPLE)
        self.assertEqual(rb.ExternalResultBackend(tmp.path).validate(_case()), [])

    def test_documented_payload_runs_end_to_end(self):
        tmp = _TempJson(self.DOC_EXAMPLE)
        out = rb.run_case(_case(), external_path=tmp.path)
        self.assertTrue(out["ok"])
        self.assertEqual(out["backend"], "eclipse-2021")
        self.assertEqual(out["adapter"], "external-json")
        self.assertFalse(out["degraded"])

    def test_doc_file_references_the_schema_it_documents(self):
        """文档存在且写的是同一个 schema —— 否则报错信息指向一个错的地方。"""
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "docs", "simulator-adapter.md")
        self.assertTrue(os.path.exists(path), "docs/simulator-adapter.md 不存在")
        with open(path, encoding="utf-8") as fh:
            body = fh.read()
        self.assertIn(rb.SCHEMA, body)
        match = re.search(r"```json\s*\n(.*?)\n```", body, flags=re.DOTALL)
        self.assertIsNotNone(match, "文档里缺可执行的 JSON 样例")
        payload = json.loads(match.group(1))
        case = rb.SimulationCase(**payload["case"])
        tmp = _TempJson(payload)
        self.assertEqual(rb.ExternalResultBackend(tmp.path).validate(case), [])


if __name__ == "__main__":
    unittest.main()
