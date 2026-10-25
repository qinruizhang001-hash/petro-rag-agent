"""
决策阈值（UQ 概率 → 告警等级）的离线单元测试
=============================================
零依赖、零 IO、零模型。

这一层最容易出的不是算错，而是**悄悄失效**：

- 阈值顺序写反（warn > alert）→ 告警等级永远不可达，报告只会少一段，
  不会报错；
- 环境变量写错（``UQ_WARN_P_LEAK=5`` 想表达 5%，实际是 500%）→ 永远不告警；
- 概率是噪声却按点估计硬判 → "恰好 5%" 被当成确定结论。

所以这里的断言集中在"出错时必须响"：非法配置抛异常、区间跨阈值时
``resolved=False``、报告引用的阈值来路必须是政策而不是标准。

运行::

    python -m unittest -v tests.test_risk_policy
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.risk_policy import (
    DEFAULT_ALERT_P_LEAK,
    DEFAULT_WARN_P_LEAK,
    ENV_ALERT,
    ENV_WARN,
    decide,
    load_thresholds,
    samples_needed,
)


class TestLoadThresholds(unittest.TestCase):
    def test_defaults(self):
        th = load_thresholds({})
        self.assertEqual(th["warn"], DEFAULT_WARN_P_LEAK)
        self.assertEqual(th["alert"], DEFAULT_ALERT_P_LEAK)
        self.assertEqual(th["source"], "default")

    def test_provenance_says_it_is_not_a_standard(self):
        """阈值来路必须写明"非标准规定"，否则会被当成标准引用出去。"""
        th = load_thresholds({})
        self.assertIn("非标准", th["provenance"])
        self.assertIn("ISO 27914", th["provenance"])

    def test_env_override(self):
        th = load_thresholds({ENV_WARN: "0.10", ENV_ALERT: "0.30"})
        self.assertEqual((th["warn"], th["alert"]), (0.10, 0.30))
        self.assertEqual(th["source"], "env")
        self.assertIn("环境变量", th["provenance"])

    def test_partial_override_still_flips_the_source(self):
        # 只覆盖一个也是覆盖：报告里引用的就不再是默认值
        th = load_thresholds({ENV_ALERT: "0.5"})
        self.assertEqual(th["source"], "env")
        self.assertEqual(th["warn"], DEFAULT_WARN_P_LEAK)

    def test_non_numeric_env_raises(self):
        with self.assertRaises(ValueError) as ctx:
            load_thresholds({ENV_WARN: "five-percent"})
        self.assertIn(ENV_WARN, str(ctx.exception))

    def test_out_of_range_env_raises(self):
        # 写成 5（想表达 5%）是最常见的错，必须当场拒绝
        with self.assertRaises(ValueError):
            load_thresholds({ENV_WARN: "5"})

    def test_negative_env_raises(self):
        with self.assertRaises(ValueError):
            load_thresholds({ENV_WARN: "-0.1"})

    def test_inverted_thresholds_raise(self):
        with self.assertRaises(ValueError) as ctx:
            load_thresholds({ENV_WARN: "0.5", ENV_ALERT: "0.2"})
        self.assertIn("必须大于", str(ctx.exception))

    def test_equal_thresholds_raise(self):
        # warn == alert 时 alert 那一档的区间宽度为 0，等价于取消告警
        with self.assertRaises(ValueError):
            load_thresholds({ENV_WARN: "0.2", ENV_ALERT: "0.2"})

    def test_real_environment_is_read_when_not_injected(self):
        # 不传 environ 时读进程环境：默认环境下应当仍是 default
        old_warn, old_alert = os.environ.get(ENV_WARN), os.environ.get(ENV_ALERT)
        os.environ.pop(ENV_WARN, None)
        os.environ.pop(ENV_ALERT, None)
        try:
            self.assertEqual(load_thresholds()["source"], "default")
        finally:
            if old_warn is not None:
                os.environ[ENV_WARN] = old_warn
            if old_alert is not None:
                os.environ[ENV_ALERT] = old_alert


class TestDecideLevels(unittest.TestCase):
    def setUp(self):
        self.th = load_thresholds({})

    def test_below_warn_is_ok(self):
        d = decide({"ok": 0.99, "risky": 0.01}, 100, self.th)
        self.assertEqual(d["level"], "ok")
        self.assertFalse(d["triggered"])

    def test_between_warn_and_alert_is_warn(self):
        d = decide({"ok": 0.90, "risky": 0.10}, 100, self.th)
        self.assertEqual(d["level"], "warn")
        self.assertTrue(d["triggered"])

    def test_above_alert_is_alert(self):
        d = decide({"ok": 0.70, "risky": 0.30}, 100, self.th)
        self.assertEqual(d["level"], "alert")
        self.assertTrue(d["triggered"])

    def test_missing_metric_means_zero(self):
        d = decide({"ok": 1.0}, 100, self.th)
        self.assertEqual(d["p"], 0.0)
        self.assertEqual(d["level"], "ok")

    def test_empty_probabilities(self):
        d = decide({}, 0, self.th)
        self.assertEqual(d["level"], "ok")
        self.assertFalse(d["resolved"])

    def test_custom_metric_name(self):
        d = decide({"ok": 0.8, "under": 0.2}, 100, self.th, metric="under",
                   label="注不进")
        self.assertEqual(d["level"], "alert")
        self.assertIn("注不进", d["text"])


class TestResolution(unittest.TestCase):
    """区间跨阈值时必须标记"不可区分"，而不是假装点估计是确定的。"""

    def setUp(self):
        self.th = load_thresholds({})

    def test_exactly_at_the_boundary_is_unresolved(self):
        # k=5 / n=100：95% 区间 2.2%–11.2%，跨越 5% 这条线
        d = decide({"ok": 0.95, "risky": 0.05}, 100, self.th)
        self.assertEqual(d["level"], "warn")
        self.assertFalse(d["resolved"])
        self.assertTrue(any("不可区分" in n for n in d["notes"]))
        self.assertIn("不可区分", d["text"])

    def test_far_from_the_boundary_is_resolved(self):
        d = decide({"ok": 0.70, "risky": 0.30}, 100, self.th)
        self.assertTrue(d["resolved"])

    def test_zero_hits_is_resolved_when_interval_stays_below_warn(self):
        d = decide({"ok": 1.0}, 100, self.th)
        self.assertTrue(d["resolved"])

    def test_more_samples_resolve_a_borderline_case(self):
        # 同样的 5%，样本量上去以后区间收窄，仍跨线（5% 恰在线上），
        # 但 6% 在 5000 次模拟下应当被判为"明确高于 5%"
        d = decide({"ok": 0.94, "risky": 0.06}, 5000, self.th)
        self.assertEqual(d["level"], "warn")
        self.assertTrue(d["resolved"])

    def test_notes_suggest_more_samples_when_needed(self):
        d = decide({"ok": 0.95, "risky": 0.05}, 100, self.th)
        self.assertTrue(any("模拟" in n for n in d["notes"]))


class TestProbabilityInterval(unittest.TestCase):
    def setUp(self):
        self.th = load_thresholds({})

    def test_interval_brackets_the_point_estimate(self):
        d = decide({"ok": 0.88, "risky": 0.12}, 100, self.th)
        lo, hi = d["p_ci"]
        self.assertLessEqual(lo, d["p"])
        self.assertLessEqual(d["p"], hi)

    def test_hits_match_the_probability(self):
        d = decide({"ok": 0.88, "risky": 0.12}, 100, self.th)
        self.assertEqual(d["hits"], 12)

    def test_text_contains_the_interval(self):
        d = decide({"ok": 0.88, "risky": 0.12}, 100, self.th)
        self.assertIn("95% 区间", d["text"])
        self.assertIn("100 次模拟", d["text"])

    def test_no_simulation_count_is_flagged(self):
        d = decide({"ok": 1.0}, 0, self.th)
        self.assertFalse(d["resolved"])
        self.assertTrue(any("模拟次数" in n for n in d["notes"]))


class TestTextAndRule(unittest.TestCase):
    def setUp(self):
        self.th = load_thresholds({})

    def test_rule_mentions_both_thresholds(self):
        d = decide({"risky": 0.1}, 100, self.th)
        self.assertIn("20%", d["rule"])
        self.assertIn("5%", d["rule"])
        self.assertIn("告警", d["rule"])

    def test_text_states_the_rule_and_the_level(self):
        d = decide({"risky": 0.3}, 100, self.th)
        self.assertIn("告警", d["text"])
        self.assertIn(d["rule"], d["text"])

    def test_env_override_is_noted(self):
        th = load_thresholds({ENV_WARN: "0.01", ENV_ALERT: "0.02"})
        d = decide({"risky": 0.03}, 1000, th)
        self.assertTrue(any("环境变量" in n for n in d["notes"]))


class TestSamplesNeeded(unittest.TestCase):
    def test_matches_the_normal_approximation(self):
        # n = z²p(1−p)/h²；p=0.05、h=0.01 → 1.96²·0.0475/1e-4 ≈ 1825
        self.assertEqual(samples_needed(0.05, 0.01), 1825)

    def test_smaller_half_width_needs_more_samples(self):
        self.assertGreater(samples_needed(0.1, 0.01), samples_needed(0.1, 0.05))

    def test_extreme_probabilities_need_fewer_samples(self):
        self.assertLess(samples_needed(0.01, 0.01), samples_needed(0.5, 0.01))

    def test_degenerate_probability_returns_zero(self):
        self.assertEqual(samples_needed(0.0, 0.01), 0)
        self.assertEqual(samples_needed(1.0, 0.01), 0)


class TestDecideIsConsistentWithToolLayer(unittest.TestCase):
    """工具层产出的概率必须能直接喂给 decide，不需要中间转换。"""

    def test_injection_risk_uq_output_feeds_decide(self):
        from tools.physics_tools import injection_risk_uq

        out = injection_risk_uq(2500, 380, seed=42)
        self.assertIn("decision", out)
        decision = out["decision"]
        self.assertIn(decision["level"], ("ok", "warn", "alert"))
        self.assertAlmostEqual(
            sum(out["probabilities"].values()), 1.0, places=6
        )
        # 决策里的概率必须与概率分布里的一致（不能各算一遍）
        self.assertAlmostEqual(decision["p"],
                               out["probabilities"].get("risky", 0.0), places=9)

    def test_correlation_changes_the_probability_but_not_the_contract(self):
        from tools.physics_tools import injection_risk_uq

        ind = injection_risk_uq(2500, 375, seed=42)
        cor = injection_risk_uq(2500, 375, seed=42, corr=0.9)
        self.assertTrue(ind["independent"])
        self.assertFalse(cor["independent"])
        # 正相关是"保护性"的：深度偏大时破裂压力也偏大，越界反而更少
        self.assertLess(cor["probabilities"].get("risky", 0.0),
                        ind["probabilities"].get("risky", 0.0))
        for out in (ind, cor):
            self.assertAlmostEqual(sum(out["probabilities"].values()), 1.0, places=6)

    def test_corr_out_of_range_is_rejected(self):
        from tools.physics_tools import injection_risk_uq

        with self.assertRaises(ValueError):
            injection_risk_uq(2500, 300, corr=1.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
