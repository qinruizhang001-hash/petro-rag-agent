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

from model_router import (
    build_routing,
    parse_route,
    resolve_route,
    classify_complexity,
    escalate_complexity,
    COMPLEXITY_CHAR_THRESHOLD,
    RAG_MIN_SCORE,
    LOCAL_MODE_LABEL,
    CLOUD_MODE_LABEL,
    DYNAMIC_STAGES,
)


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


class TestClassifyComplexity(unittest.TestCase):
    """动态路由：按输入长度 + RAG 是否命中决定档位。"""

    def test_short_input_stays_local(self):
        c = classify_complexity("250bar 80度 有断层")
        self.assertEqual(c["mode"], "local")
        self.assertEqual(c["label"], LOCAL_MODE_LABEL)

    def test_long_input_goes_cloud(self):
        c = classify_complexity("某" * 80)
        self.assertEqual(c["mode"], "cloud")
        self.assertEqual(c["label"], CLOUD_MODE_LABEL)

    def test_exactly_threshold_stays_local(self):
        """边界用 >：恰好 50 字仍算简单查询。"""
        c = classify_complexity("a" * COMPLEXITY_CHAR_THRESHOLD)
        self.assertEqual(c["mode"], "local")

    def test_one_over_threshold_goes_cloud(self):
        c = classify_complexity("a" * (COMPLEXITY_CHAR_THRESHOLD + 1))
        self.assertEqual(c["mode"], "cloud")

    def test_rag_hit_escalates_short_input(self):
        """短输入但知识库确实命中 → 云端。"""
        c = classify_complexity("断层活化风险", rag_triggered=True, top_score=0.74)
        self.assertEqual(c["mode"], "cloud")
        self.assertIn("知识库命中", c["reason"])

    def test_rag_false_hit_keeps_local(self):
        """
        图谱对无关提问也会返回片段，但相关度接近 0。
        这种"假命中"不该把请求推到云端。
        """
        c = classify_complexity("今天天气怎么样", rag_triggered=False, top_score=0.0001)
        self.assertEqual(c["mode"], "local")
        self.assertIn("0.0001", c["reason"])

    def test_rag_threshold_boundary(self):
        c_low = classify_complexity("x", rag_triggered=False, top_score=RAG_MIN_SCORE - 0.001)
        c_eq = classify_complexity("x", rag_triggered=True, top_score=RAG_MIN_SCORE)
        self.assertEqual(c_low["mode"], "local")
        self.assertEqual(c_eq["mode"], "cloud")

    def test_top_score_overrides_inconsistent_flag(self):
        """
        分数与布尔值冲突时以分数为准。
        否则日志会打出"未命中（相关度 0.7400 < 0.05）"这种自相矛盾的话。
        """
        c = classify_complexity("短问题", rag_triggered=False, top_score=0.74)
        self.assertEqual(c["mode"], "cloud")
        self.assertTrue(c["rag_triggered"])
        self.assertIn("命中", c["reason"])
        self.assertNotIn("未命中", c["reason"])

    def test_flag_used_when_score_unknown(self):
        c = classify_complexity("短问题", rag_triggered=True, top_score=None)
        self.assertEqual(c["mode"], "cloud")
        self.assertIn("知识库命中", c["reason"])

    def test_reasons_are_explainable(self):
        """判定必须能解释：日志里要能看出为什么选了这个档位。"""
        c = classify_complexity("a" * 60, rag_triggered=True, top_score=0.3)
        self.assertGreaterEqual(len(c["reasons"]), 2)
        self.assertIn("→ 云端", c["reason"])
        self.assertIn("60 字", c["reason"])

    def test_chars_recorded(self):
        c = classify_complexity("一二三四五")
        self.assertEqual(c["chars"], 5)

    def test_label_constant_wording(self):
        """日志标签是需求明确要求的字符串，不能改。"""
        self.assertEqual(LOCAL_MODE_LABEL, "[Cost-Saving: Local Mode]")
        self.assertEqual(CLOUD_MODE_LABEL, "[Premium: Cloud Mode]")

    def test_dynamic_stages_only_extract_and_report(self):
        """
        report_plan 不在动态路由管辖内：它只输出几十个 token 的 JSON，
        无论请求多复杂都该留在本地。
        """
        self.assertEqual(set(DYNAMIC_STAGES), {"extract", "report"})


class TestEscalate(unittest.TestCase):
    def test_local_to_cloud_upgrade(self):
        prev = classify_complexity("短问题")
        new = classify_complexity("短问题", rag_triggered=True, top_score=0.5)
        merged = escalate_complexity(prev, new)
        self.assertEqual(merged["mode"], "cloud")
        self.assertIn("retrieve 阶段升级", merged["reason"])

    def test_cloud_never_downgrades(self):
        """
        extract 已按长输入付过云端的钱，retrieve 不该再降回本地。

        用手工构造的判定结果来测，而不是走 classify_complexity ——
        因为输入长度在同一次请求里是常量，"长输入在 retrieve 阶段变短"
        不可能发生，走真实函数永远测不到这个防御分支。
        """
        prev = {"mode": "cloud", "reason": "输入 80 字 > 50 字阈值", "chars": 80}
        new = {"mode": "local", "reason": "知识库未命中", "chars": 80}
        merged = escalate_complexity(prev, new)
        self.assertEqual(merged["mode"], "cloud")
        self.assertIn("不再降级", merged["reason"])

    def test_no_prev_returns_new(self):
        new = classify_complexity("x")
        self.assertEqual(escalate_complexity(None, new)["mode"], new["mode"])

    def test_same_mode_returns_new(self):
        prev = classify_complexity("x")
        new = classify_complexity("x", rag_triggered=False, top_score=0.0)
        self.assertEqual(escalate_complexity(prev, new)["mode"], "local")


class TestLocalClientTimeouts(unittest.TestCase):
    """本地客户端必须有明确的超时/重试上限。

    OpenAI SDK 默认 600s × 3 次 = 一次卡住的生成能占住半小时，且日志里没有
    任何异常。实测 Ollama 在 report_plan 阶段就那样静默卡死过（服务端
    /api/ps 仍正常）。这条测试防的是"把显式配置悄悄改回默认值"。
    """

    def test_local_client_does_not_use_the_sdk_default_timeout(self):
        from core import config

        self.assertEqual(config.local_client.timeout, config.LOCAL_LLM_TIMEOUT)
        self.assertLessEqual(config.local_client.timeout, 300)
        self.assertLessEqual(config.local_client.max_retries, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
