"""
离线单元测试：模型路由解析
==========================
不需要 Ollama / 模型文件 / 网络，纯逻辑断言，CI 秒级跑完。

运行:
    python -m unittest discover -s tests -v
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_router import build_routing, parse_route, resolve_route


DEFAULTS = {
    "extract": "local",
    "report": "cloud",
    "evaluate": "local",
    "history": "local",
    "summary": "local",
}


class TestParseRoute(unittest.TestCase):
    def test_plain_tier(self):
        self.assertEqual(parse_route("cloud"), ("cloud", ""))
        self.assertEqual(parse_route("local"), ("local", ""))

    def test_tier_with_model(self):
        self.assertEqual(
            parse_route("cloud:deepseek/deepseek-chat"),
            ("cloud", "deepseek/deepseek-chat"),
        )

    def test_model_name_may_contain_slash_and_dash(self):
        """网关用 provider/model 命名，不能因为斜杠被切错。"""
        tier, model = parse_route("cloud:kimi/kimi-k2.5")
        self.assertEqual(tier, "cloud")
        self.assertEqual(model, "kimi/kimi-k2.5")

    def test_case_and_whitespace_insensitive(self):
        self.assertEqual(parse_route("  CLOUD  "), ("cloud", ""))
        self.assertEqual(parse_route(" Cloud : openai/gpt-4o-mini "),
                         ("cloud", "openai/gpt-4o-mini"))

    def test_empty_spec_falls_back_to_local(self):
        self.assertEqual(parse_route(""), ("local", ""))
        self.assertEqual(parse_route(None), ("local", ""))

    def test_unknown_tier_is_preserved_for_reason(self):
        """未知档位不在解析阶段报错，交由 resolve_route 给出可追踪原因。"""
        self.assertEqual(parse_route("gpu"), ("gpu", ""))


class TestBuildRouting(unittest.TestCase):
    def test_no_env_returns_defaults(self):
        self.assertEqual(build_routing(DEFAULTS, {}), DEFAULTS)

    def test_env_overrides_single_stage(self):
        r = build_routing(DEFAULTS, {"ROUTE_REPORT": "local"})
        self.assertEqual(r["report"], "local")
        self.assertEqual(r["extract"], "local")   # 其余保持不变

    def test_env_can_pin_model(self):
        r = build_routing(DEFAULTS, {"ROUTE_REPORT": "cloud:deepseek/deepseek-reasoner"})
        self.assertEqual(r["report"], "cloud:deepseek/deepseek-reasoner")

    def test_blank_env_does_not_wipe_default(self):
        """
        .env 里写 `ROUTE_REPORT=` （留空）时不能把默认值清成空字符串——
        否则该角色会静默退化为 local，且没有任何提示。
        """
        r = build_routing(DEFAULTS, {"ROUTE_REPORT": "", "ROUTE_EVALUATE": "   "})
        self.assertEqual(r["report"], "cloud")
        self.assertEqual(r["evaluate"], "local")

    def test_unknown_stage_in_env_is_ignored(self):
        r = build_routing(DEFAULTS, {"ROUTE_NONEXISTENT": "cloud"})
        self.assertEqual(set(r), set(DEFAULTS))


class TestResolveRoute(unittest.TestCase):
    def test_cloud_route_passes_when_available(self):
        tier, model, reason = resolve_route(
            {"report": "cloud:deepseek/deepseek-chat"}, "report", True
        )
        self.assertEqual(tier, "cloud")
        self.assertEqual(model, "deepseek/deepseek-chat")
        self.assertEqual(reason, "")

    def test_cloud_route_degrades_when_key_missing(self):
        tier, model, reason = resolve_route({"report": "cloud"}, "report", False)
        self.assertEqual(tier, "local")
        self.assertEqual(model, "")
        self.assertIn("CLOUD_API_KEY", reason, "降级必须说明原因，否则极难排查")

    def test_degrade_keeps_model_override(self):
        """降级到本地时仍保留用户指定的模型名，便于从日志看出意图。"""
        tier, model, reason = resolve_route(
            {"report": "cloud:deepseek/deepseek-chat"}, "report", False
        )
        self.assertEqual(tier, "local")
        self.assertEqual(model, "deepseek/deepseek-chat")
        self.assertTrue(reason)

    def test_local_route_never_degrades(self):
        tier, _, reason = resolve_route({"extract": "local"}, "extract", False)
        self.assertEqual(tier, "local")
        self.assertEqual(reason, "")

    def test_local_with_explicit_model(self):
        tier, model, reason = resolve_route(
            {"extract": "local:qwen2.5:14b"}, "extract", True
        )
        self.assertEqual((tier, model, reason), ("local", "qwen2.5:14b", ""))

    def test_invalid_tier_degrades_with_reason(self):
        tier, _, reason = resolve_route({"report": "gpu"}, "report", True)
        self.assertEqual(tier, "local")
        self.assertIn("gpu", reason)

    def test_missing_stage_defaults_to_local(self):
        tier, model, reason = resolve_route({}, "anything", True)
        self.assertEqual((tier, model, reason), ("local", "", ""))


class TestGatewayScenario(unittest.TestCase):
    """端到端意图：把 CLOUD_BASE_URL 指向聚合网关时，各角色各用各的模型。"""

    def setUp(self):
        self.routing = build_routing(DEFAULTS, {
            "ROUTE_REPORT": "cloud:deepseek/deepseek-chat",
            "ROUTE_EVALUATE": "cloud:kimi/kimi-k2",
            "ROUTE_EXTRACT": "local",
        })

    def test_each_stage_gets_its_own_model(self):
        cases = {
            "report": ("cloud", "deepseek/deepseek-chat"),
            "evaluate": ("cloud", "kimi/kimi-k2"),
            "extract": ("local", ""),
        }
        for stage, (want_tier, want_model) in cases.items():
            tier, model, _ = resolve_route(self.routing, stage, True)
            self.assertEqual((tier, model), (want_tier, want_model),
                             f"{stage} 路由不符")

    def test_gateway_down_degrades_all_cloud_stages(self):
        for stage in ("report", "evaluate"):
            tier, _, reason = resolve_route(self.routing, stage, False)
            self.assertEqual(tier, "local")
            self.assertTrue(reason)


if __name__ == "__main__":
    unittest.main(verbosity=2)
