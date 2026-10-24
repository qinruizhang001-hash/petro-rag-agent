"""
离线单元测试：成本核算（Token Cost Model）
==========================================
纯算术，但算错的后果是"前端显示一个漂亮而虚假的省钱数字"，且不会报错。
所以每个口径都要有断言钉住。

运行:
    python -m unittest tests.test_cost_model -v
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.cost_model import (
    DEFAULT_CLOUD_PRICING,
    LOCAL_PRICING,
    PRICING_CNY_PER_MTOK,
    caption_text,
    estimate_cost,
    format_cny,
    is_local_model,
    resolve_pricing,
)


def usage(**stages):
    """构造 token_usage 结构。"""
    return dict(stages)


class TestPricingLookup(unittest.TestCase):
    def test_local_models_by_marker(self):
        for name in ("qwen2.5", "ollama/qwen2.5:latest", "llama3", "local:qwen2.5"):
            self.assertTrue(is_local_model(name), name)

    def test_cloud_models_not_local(self):
        for name in ("deepseek-chat", "deepseek/deepseek-chat", "gpt-4o"):
            self.assertFalse(is_local_model(name), name)

    def test_ollama_prefixed_deepseek_is_local(self):
        """
        网关上的 'ollama/deepseek-r1' 名字含 deepseek，但算力在本机，
        真实成本是电费。本地标记必须优先于云端关键字。
        """
        pricing, key, fell_back = resolve_pricing("ollama/deepseek-r1")
        self.assertEqual(key, "local")
        self.assertEqual(pricing, LOCAL_PRICING)
        self.assertFalse(fell_back)

    def test_exact_match(self):
        pricing, key, fell_back = resolve_pricing("deepseek-chat", "cloud")
        self.assertEqual(key, "deepseek-chat")
        self.assertEqual(pricing["input"], 2.0)
        self.assertFalse(fell_back)

    def test_provider_prefix_stripped(self):
        """'deepseek/deepseek-chat' 要能命中 deepseek-chat 的价格。"""
        pricing, key, fell_back = resolve_pricing("deepseek/deepseek-chat", "cloud")
        self.assertEqual(key, "deepseek-chat")
        self.assertFalse(fell_back)

    def test_unknown_cloud_model_falls_back_high(self):
        """未知模型按最高档估，宁可高估账单。"""
        pricing, key, fell_back = resolve_pricing("some-new-model", "cloud")
        self.assertTrue(fell_back)
        self.assertEqual(pricing, DEFAULT_CLOUD_PRICING)

    def test_unknown_local_tier_falls_back_local(self):
        pricing, key, fell_back = resolve_pricing("mystery-model", "local")
        self.assertEqual(pricing, LOCAL_PRICING)
        self.assertFalse(fell_back)


class TestEstimateCost(unittest.TestCase):
    def test_single_local_stage_costs_electricity(self):
        """本地模型不是零成本：输入 1000 + 输出 1000 应有一个小正数。"""
        est = estimate_cost(usage(
            extract={"prompt_tokens": 1000, "completion_tokens": 1000,
                     "total_tokens": 2000, "model": "qwen2.5", "tier": "local"},
        ))
        # 1000/1e6*0.1 + 1000/1e6*0.8 = 0.0009
        self.assertAlmostEqual(est["total_cny"], 0.0009, places=8)
        self.assertEqual(est["stages_priced"], 1)

    def test_deepseek_cost_matches_hand_calc(self):
        """10k 输入 + 2k 输出 @ 2/8 元每百万 = 0.02 + 0.016 = 0.036"""
        est = estimate_cost(usage(
            report={"prompt_tokens": 10000, "completion_tokens": 2000,
                    "total_tokens": 12000, "model": "deepseek-chat", "tier": "cloud"},
        ))
        self.assertAlmostEqual(est["total_cny"], 0.036, places=8)

    def test_baseline_is_all_cloud(self):
        """
        基线 = 全部 token 都按 baseline_model 计费。
        本地阶段越多，节省越大。
        """
        est = estimate_cost(
            usage(
                extract={"prompt_tokens": 1000, "completion_tokens": 500,
                         "model": "qwen2.5", "tier": "local"},
                report={"prompt_tokens": 1000, "completion_tokens": 500,
                        "model": "deepseek-chat", "tier": "cloud"},
            ),
            baseline_model="deepseek-chat",
        )
        # 基线 = 2000 输入 * 2 + 1000 输出 * 8 = 0.004 + 0.008 = 0.012
        self.assertAlmostEqual(est["baseline_cny"], 0.012, places=8)
        # 实际 = 本地(0.0001+0.0004) + 云端(0.002+0.004) = 0.0065
        self.assertAlmostEqual(est["total_cny"], 0.0065, places=8)
        self.assertAlmostEqual(est["saved_cny"], 0.0055, places=8)

    def test_all_local_equals_full_saving(self):
        """全本地运行时，节省额应等于基线（实际成本≈电费，不为负）。"""
        est = estimate_cost(usage(
            report={"prompt_tokens": 10000, "completion_tokens": 10000,
                    "model": "qwen2.5", "tier": "local"},
        ))
        self.assertGreater(est["saved_cny"], 0)
        self.assertLessEqual(est["total_cny"], est["baseline_cny"])

    def test_saved_pct(self):
        est = estimate_cost(usage(
            extract={"prompt_tokens": 10000, "completion_tokens": 0,
                     "model": "qwen2.5", "tier": "local"},
        ))
        # 实际 10000/1e6*0.1=0.001；基线 10000/1e6*2=0.02 → 省 95%
        self.assertAlmostEqual(est["saved_pct"], 0.95, places=4)

    def test_retrieve_stats_not_priced(self):
        """检索统计不是 LLM 调用，不能按 token 计价。"""
        est = estimate_cost(usage(
            retrieve={"query_tokens": 100, "candidate_tokens": 9999,
                      "injected_context_tokens": 500},
        ))
        self.assertEqual(est["stages_priced"], 0)
        self.assertEqual(est["total_cny"], 0.0)

    def test_llm_total_tokens_not_double_counted(self):
        """llm_total_tokens 是汇总字段，再计一次就翻倍。"""
        est = estimate_cost(usage(
            extract={"prompt_tokens": 1000, "completion_tokens": 0,
                     "model": "deepseek-chat", "tier": "cloud"},
            llm_total_tokens=1000,
        ))
        self.assertEqual(est["stages_priced"], 1)
        self.assertAlmostEqual(est["total_cny"], 0.002, places=8)

    def test_zero_token_stage_skipped(self):
        est = estimate_cost(usage(
            extract={"prompt_tokens": 0, "completion_tokens": 0,
                     "model": "deepseek-chat"},
        ))
        self.assertEqual(est["stages_priced"], 0)

    def test_model_from_routing_snapshot(self):
        """payload 缺 model 时，用 model_routing 的 'tier:model' 兜底计价。"""
        est = estimate_cost(
            usage(report={"prompt_tokens": 10000, "completion_tokens": 0}),
            routing={"report": "cloud:deepseek-chat"},
        )
        self.assertEqual(est["breakdown"][0]["model"], "deepseek-chat")
        self.assertAlmostEqual(est["total_cny"], 0.02, places=8)

    def test_estimated_flag_propagates(self):
        est = estimate_cost(usage(
            report={"prompt_tokens": 10, "completion_tokens": 10,
                    "model": "deepseek-chat", "usage_estimated": True},
        ))
        self.assertTrue(est["estimated"])
        self.assertTrue(est["breakdown"][0]["estimated"])

    def test_unpriced_models_reported(self):
        est = estimate_cost(usage(
            report={"prompt_tokens": 10, "completion_tokens": 10,
                    "model": "brand-new-cloud-model", "tier": "cloud"},
        ))
        self.assertIn("brand-new-cloud-model", est["unpriced_models"])

    def test_empty_usage(self):
        est = estimate_cost({})
        self.assertEqual(est["total_cny"], 0.0)
        self.assertEqual(est["saved_cny"], 0.0)
        self.assertEqual(est["saved_pct"], 0.0)
        self.assertEqual(est["breakdown"], [])

    def test_saved_never_negative(self):
        """实际用了比基线更贵的模型时，节省额按 0 计并给出说明。"""
        est = estimate_cost(
            usage(report={"prompt_tokens": 10000, "completion_tokens": 0,
                          "model": "deepseek-v4-pro", "tier": "cloud"}),
            baseline_model="deepseek-chat",
        )
        self.assertEqual(est["saved_cny"], 0.0)
        self.assertTrue(est["note"])

    def test_breakdown_fields(self):
        est = estimate_cost(usage(
            report={"prompt_tokens": 10, "completion_tokens": 5,
                    "model": "deepseek-chat", "tier": "cloud"},
        ))
        b = est["breakdown"][0]
        for key in ("stage", "tier", "model", "pricing_key", "input_tokens",
                    "output_tokens", "cny", "baseline_cny", "estimated"):
            self.assertIn(key, b)
        self.assertEqual(b["stage"], "report")
        self.assertEqual(b["input_tokens"], 10)
        self.assertEqual(b["output_tokens"], 5)

    def test_pricing_table_is_positive(self):
        for name, p in PRICING_CNY_PER_MTOK.items():
            self.assertGreater(p["input"], 0, name)
            self.assertGreater(p["output"], 0, name)
            # 输出单价必然高于输入（decode 比 prefill 贵）
            self.assertGreater(p["output"], p["input"], name)


class TestFormatting(unittest.TestCase):
    def test_small_amounts_keep_significant_digits(self):
        self.assertEqual(format_cny(0.0021), "0.0021")
        # 0.015 若按两位小数会变成 "0.01"，误差 33% —— 必须保留四位
        self.assertEqual(format_cny(0.015), "0.0150")

    def test_under_one_yuan_keeps_four_decimals(self):
        """本项目单次成本落在 0.001~0.05 元，四位小数才看得出差异。"""
        self.assertEqual(format_cny(0.5), "0.5000")
        self.assertEqual(format_cny(0.015966), "0.0160")

    def test_tiny_amounts_use_six_decimals(self):
        self.assertEqual(format_cny(0.000012), "0.000012")

    def test_large_amounts_two_decimals(self):
        self.assertEqual(format_cny(1.2345), "1.23")

    def test_zero(self):
        self.assertEqual(format_cny(0), "0.00")

    def test_caption_states_the_counterfactual_instead_of_claiming_savings(self):
        # 措辞必须把"反事实"写在脸上：基线是把同一批 token 按云端列表价重算的
        # 估算，不是任何客户的真实账单。写成"已为你节省"会让人以为存在一笔
        # 可以报销的差额 —— 我们并不知道对方的采购价。
        est = estimate_cost(usage(
            extract={"prompt_tokens": 10000, "completion_tokens": 0,
                     "model": "qwen2.5", "tier": "local"},
        ))
        text = caption_text(est)
        self.assertTrue(text.startswith("本次评估预估成本：¥"), text)
        self.assertIn("列表价", text)
        self.assertIn("反事实", text)
        self.assertIn("不是真实账单", text)
        self.assertNotIn("已为你节省", text)

    def test_caption_without_saving(self):
        est = {"stages_priced": 1, "total_cny": 0.002, "saved_cny": 0}
        self.assertEqual(caption_text(est), "本次评估预估成本：¥0.0020")

    def test_caption_without_usage(self):
        self.assertIn("未采集到", caption_text({"stages_priced": 0}))


if __name__ == "__main__":
    unittest.main(verbosity=2)
