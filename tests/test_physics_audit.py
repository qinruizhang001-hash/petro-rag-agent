"""物理规则的可审计性：参数有出处、可覆盖、版本可追溯，且 PINN 不参与裁决。

为什么这几个测试重要：规则引擎的输出会被当成工程依据。如果 ``ρ=1030`` 与
``1.5× 静水`` 是硬编码在代码里的两个数，那么"物理规则说超压了"这句话就无法
被第三方复核 —— 它只是另一种形式的魔法数，只不过穿着代码的外衣。
"""
from __future__ import annotations

import unittest
from pathlib import Path

from utils.physics_rules import (
    RULES_VERSION,
    RuleParams,
    evaluate_physics,
    load_rule_params,
    rules_manifest,
)

PARAMS = {"injection_pressure_bar": 400.0, "depth_m": 2500.0, "temperature_c": 80.0}
SAFE_REPORT = "注入压力 400 bar，风险可控。"


class RulesManifestTests(unittest.TestCase):
    def test_result_carries_version_and_parameters(self):
        result = evaluate_physics(params=PARAMS, report_text=SAFE_REPORT)
        self.assertEqual(result["rules_version"], RULES_VERSION)
        self.assertIn("rho_water", result["rules"])
        self.assertEqual(result["rules"]["fracture_ratio"]["value"], 1.5)

    def test_every_parameter_has_a_source(self):
        manifest = rules_manifest()
        for name, meta in manifest["parameters"].items():
            with self.subTest(parameter=name):
                self.assertTrue(meta["source"].strip(),
                                msg=f"{name} 没有出处，等于魔法数")

    def test_version_is_a_string_not_a_float(self):
        # 版本号要能表达"2026.10-r1"这种带修订次的形态，浮点数不行。
        self.assertIsInstance(RULES_VERSION, str)
        self.assertGreaterEqual(len(RULES_VERSION), 6)


class RulesAreWiredNotDecorativeTests(unittest.TestCase):
    """改了参数，结论必须跟着变 —— 否则参数只是写在报告里的装饰。"""

    def test_default_ratio_flags_400_bar_at_2500_m(self):
        result = evaluate_physics(params=PARAMS, report_text=SAFE_REPORT)
        self.assertTrue(result["blocking"])
        self.assertEqual(result["verdict"], "above_fracture")

    def test_higher_fracture_ratio_makes_the_same_case_safe(self):
        # 1.5× → P_frac≈379 bar（超压）；2.0× → P_frac≈505 bar（安全窗口内）。
        result = evaluate_physics(
            params=PARAMS, report_text=SAFE_REPORT,
            rules=RuleParams(fracture_ratio=2.0),
        )
        self.assertFalse(result["blocking"])
        self.assertEqual(result["verdict"], "safe_window")
        # 生效值必须写进结果，而不是回显默认值。
        self.assertEqual(result["rules"]["fracture_ratio"]["value"], 2.0)


class EnvOverrideTests(unittest.TestCase):
    def test_valid_env_values_are_used(self):
        rules = load_rule_params({"CCUS_RHO_WATER": "1025", "CCUS_FRACTURE_RATIO": "1.7"})
        self.assertEqual(rules.rho_water, 1025.0)
        self.assertEqual(rules.fracture_ratio, 1.7)

    def test_out_of_range_values_fall_back_to_defaults(self):
        # 多打一个 0 的 15 会让所有报告都判成超压 —— 这种手误必须被挡住。
        rules = load_rule_params({"CCUS_FRACTURE_RATIO": "15", "CCUS_RHO_WATER": "abc"})
        self.assertEqual(rules.fracture_ratio, 1.5)
        self.assertEqual(rules.rho_water, 1030.0)

    def test_missing_env_keeps_defaults(self):
        rules = load_rule_params({})
        self.assertEqual(rules, RuleParams())


class PinnStaysOutOfTheVerdictTests(unittest.TestCase):
    """1-D PINN 是教学演示，不能参与"安全/不安全"的最终裁决。"""

    def test_physics_result_has_no_pinn_input(self):
        result = evaluate_physics(params=PARAMS, report_text=SAFE_REPORT)
        for key in result:
            self.assertNotIn("pinn", key.lower())

    def test_rules_module_does_not_reference_pinn(self):
        # 结构性保证：裁决代码里根本没有 PINN 这个词，就不会有"顺手引用一下"。
        source = (Path(__file__).resolve().parent.parent
                  / "utils" / "physics_rules.py").read_text(encoding="utf-8")
        self.assertNotIn("pinn", source.lower())


class PhysicsLimitsIndexTests(unittest.TestCase):
    """限制汇总页必须与真实模块对齐，否则它就是一份会过期的免责声明。

    `docs/physics-limits.md` 的价值全在"能查到"，一旦某个模块改名、删掉，
    表里那一行就变成指向不存在的文件的承诺 —— 而没有任何东西会红。
    """

    DOC = Path(__file__).resolve().parent.parent / "docs" / "physics-limits.md"

    # 会对外输出结论、因而必须出现在索引里的模块。新增同类模块时应补进来，
    # 这里不自动扫描：目录里还有一堆不产生物理结论的工具模块。
    REQUIRED = (
        "pressure_model.py",
        "utils/physics_rules.py",
        "utils/geomechanics.py",
        "utils/poroelastic.py",
        "utils/caprock.py",
        "utils/co2_properties.py",
        "utils/relperm.py",
        "utils/flow1d.py",
        "utils/reservoir_backend.py",
        "core/optimize.py",
        "utils/physics_deep.py",
    )

    def setUp(self):
        self.text = self.DOC.read_text(encoding="utf-8")
        self.root = self.DOC.parent.parent

    def test_every_referenced_module_exists(self):
        import re

        for rel in re.findall(r"`([\w/]+\.py)`", self.text):
            with self.subTest(module=rel):
                self.assertTrue(
                    (self.root / rel).is_file(),
                    f"docs/physics-limits.md 提到的 {rel} 不存在（改名后忘了同步？）",
                )

    def test_every_conclusion_module_is_indexed(self):
        for rel in self.REQUIRED:
            with self.subTest(module=rel):
                self.assertIn(rel, self.text)

    def test_it_says_it_is_not_a_simulator(self):
        # 这份文档唯一的目的就是防止被当成完整地质力学，所以这两句必须在
        self.assertIn("没有一个是数值模拟器", self.text)
        self.assertIn("筛查级", self.text)


if __name__ == "__main__":
    unittest.main()
