"""
utils.entity_aliases 的离线单测（纯标准库，CI 直接跑）
=====================================================
这些测试锁的是**两种相反方向的错误**，而且都很贵：

  漏合并 → 同一个概念在图谱上是两个节点，跨论文永远连不起来，
           连通分量数虚高。代价是"知识网络"退化成"知识碎片"。
  错合并 → 两个不同的地质概念被永久并成一个节点，描述被拼接、
           关系被串味，而且**无法自动还原**。代价比漏合并高一个量级。

所以测试里两类断言都要有：
  - 必须合并的（大小写、单复数、括号注释、引文后缀、显式同义词）；
  - 必须**不**合并的（有细微差别的概念，例如 Pre-Existing Fault vs Fault）。

另外锁住一个曾经写错、且不报错的地方：``井筒``/``储层``/``断层``/``盖层``
都只有两个字，被"名字太短就丢弃"的规则误杀时不会有任何异常，只会在报告里
表现为"地质实体少了四个"。
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.entity_aliases import (
    build_merge_plan,
    canonical_name,
    noise_reason,
    noise_rule_names,
    normalize_key,
    plan_summary,
    protected_names,
)


class TestNormalizeKey(unittest.TestCase):
    def test_case_and_plural_collapse(self):
        self.assertEqual(normalize_key("CO2 Plumes"), normalize_key("CO2 plume"))
        self.assertEqual(normalize_key("Saline Aquifers"), normalize_key("saline aquifer"))

    def test_title_case_variants_collapse(self):
        self.assertEqual(
            normalize_key("Depleted Hydrocarbon Fields"),
            normalize_key("Depleted hydrocarbon fields"),
        )

    def test_parenthetical_gloss_removed(self):
        self.assertEqual(normalize_key("Calcite (Calcium Carbonate)"), "calcite")
        self.assertEqual(normalize_key("光纤传感技术(DAS/DTS)"), "光纤传感技术")

    def test_citation_suffix_removed(self):
        self.assertEqual(
            normalize_key("Digital Shadow (Gahlot, Orozco, et al. 2024)"),
            "digital shadow",
        )
        self.assertEqual(normalize_key("Bayesian FWM (2023; Herrmann 2023)"), "bayesian fwm")

    def test_cap_rock_rewritten(self):
        self.assertEqual(normalize_key("Cap Rock Fracturing"), normalize_key("Caprock fracturing"))

    def test_gas_is_not_singularized(self):
        # "Gas" 去掉 s 就成了 "Ga"。三字母保护 + ss/us/is 保护必须生效。
        self.assertEqual(normalize_key("Gas"), "gas")
        self.assertEqual(normalize_key("Stress"), "stress")
        self.assertEqual(normalize_key("Analysis"), "analysis")

    def test_acronym_not_singularized(self):
        self.assertEqual(normalize_key("CCUS"), "ccus")

    def test_processes_singularized(self):
        self.assertEqual(normalize_key("Processes"), "process")


class TestCanonicalName(unittest.TestCase):
    def test_known_variants_map_to_canonical(self):
        self.assertEqual(canonical_name("Carbon dioxide (CO2)"), "CO2")
        self.assertEqual(canonical_name("Carbon Dioxide"), "CO2")
        self.assertEqual(canonical_name("Carbon Capture and Storage (CCS)"), "CCS")
        self.assertEqual(canonical_name("Faults"), "Fault")

    def test_unknown_name_kept_as_is(self):
        self.assertEqual(canonical_name("Halite Precipitation"), "Halite Precipitation")


class TestNoiseReason(unittest.TestCase):
    def test_affiliations_are_noise(self):
        for name in (
            "University of Oslo",
            "AGH University of Krakow",
            "Institute of Mechanics, Moscow State University",
            "Department of Petroleum Engineering",
        ):
            self.assertIsNotNone(noise_reason(name), name)

    def test_years_and_dates_are_noise(self):
        for name in ("2050", "2020", "2040s", "Received Date: July 21, 2021"):
            self.assertIsNotNone(noise_reason(name), name)

    def test_addresses_and_contacts_are_noise(self):
        self.assertIsNotNone(
            noise_reason("10100 Burnet Rd., Bldg. 130, Austin, TX 78758, USA")
        )
        self.assertIsNotNone(noise_reason("mohammad.nooraiepour@geo.uio.no"))
        self.assertIsNotNone(noise_reason("DOI: 10.23880/ppej-16000269"))

    def test_citation_fragments_are_noise(self):
        for name in ("Arts et al. 2008", "Chadwick Et Al", "Chen et al.,"):
            self.assertIsNotNone(noise_reason(name), name)

    def test_paper_titles_are_noise(self):
        self.assertIsNotNone(
            noise_reason(
                "A Deep Learning-Accelerated Data Assimilation and Forecasting Workflow "
                "for Commercial-Scale Geologic Carbon Storage"
            )
        )

    def test_person_type_is_noise(self):
        self.assertIsNotNone(noise_reason("Abhinav Prakash Gahlot", "person"))

    def test_chinese_two_char_geology_is_kept(self):
        # 最关键的一条：这四个是最核心的地质实体，绝不能被"太短"规则杀掉
        for name in ("井筒", "储层", "断层", "盖层"):
            self.assertIsNone(noise_reason(name), name)

    def test_chemical_formula_kept(self):
        self.assertIsNone(noise_reason("H2"))

    def test_symbols_are_noise(self):
        for name in ("Xk", "xk", "κk"):
            self.assertIsNotNone(noise_reason(name), name)

    def test_policy_bodies_are_not_noise(self):
        # organization 类型不整类丢弃：IPCC 是监管实体，删掉就是丢知识
        self.assertIsNone(noise_reason("IPCC", "organization"))

    def test_funding_bodies_are_noise(self):
        """
        资助 / 行政机构是噪声：一篇论文的资助方会把几十篇不相关的论文连成
        假枢纽，连的是经费关系而不是地质知识。

        这四个名字是实测从当前语料里取出来的，不是编的 —— 规则必须真的
        命中它们，否则"已处理机构名"就只是文档里的一句话。
        """
        for name in (
            "Georgia Research Alliance",
            "German Federal Ministry of Research, Technology, and Space",
            "German Ministry of Economic Affairs and Climate Protection (BMWK)",
            "Ministry of Economic Affairs (The Netherlands)",
        ):
            self.assertEqual(noise_reason(name, "organization"), "funding-body", name)

    def test_alias_table_beats_name_rules(self):
        """
        别名表声明过的实体，任何名字规则都不许删。

        这条修的是一个真实矛盾：`funding-body` 规则含 `agency`，而
        `International Energy Agency` 在别名表里（并成 IEA，政策主体要保留）。
        没有 `_PROTECTED` 前置判断的话，两条各自正确的规则撞在一起，
        结果就是静默删掉一个刻意保留的实体 —— 而且删得"有理有据"。
        """
        for name in (
            "International Energy Agency",
            "IEA",
            "Intergovernmental Panel on Climate Change",
            "IPCC",
            "Geological Carbon Storage",
        ):
            self.assertIsNone(noise_reason(name, "organization"), name)

    def test_every_rule_name_is_registered(self):
        """
        `noise_rule_names()` 与 `noise_reason()` 必须给出同一套原因。

        报告靠前者列"零命中的规则"，靠后者写实际命中原因。两边不一致时，
        报告会漏掉某条规则（看起来像"没有这条规则"），而这正是这份报告
        存在要解决的问题。
        """
        from utils.entity_aliases import _NOISE_RULES

        declared = set(noise_rule_names())
        for reason, _pattern in _NOISE_RULES:
            self.assertIn(reason, declared)
        self.assertEqual(len(declared), len(noise_rule_names()), "原因名有重复")

    def test_protected_names_covers_alias_variants(self):
        names = protected_names()
        self.assertGreaterEqual(len(names), 20, names)
        # 规范名与变体都要在保护名单里，只护规范名等于没护
        for probe in ("iea", "international energy agency", "ipcc", "gcs"):
            self.assertIn(probe, names)

    def test_real_entities_kept(self):
        for name in (
            "Caprock",
            "CO2 Plume",
            "Saline Aquifer",
            "Wellbore Integrity",
            "Reinforcement Learning Agent",
        ):
            self.assertIsNone(noise_reason(name), name)


class TestBuildMergePlan(unittest.TestCase):
    def test_co2_variants_merge_into_shortest(self):
        plan = build_merge_plan(["CO2", "Carbon Dioxide (CO2)", "Carbon dioxide (CO2)"])
        self.assertEqual(len(plan["merges"]), 1)
        self.assertEqual(plan["merges"][0]["target"], "CO2")
        self.assertEqual(len(plan["merges"][0]["sources"]), 3)

    def test_ccs_variants_merge(self):
        names = [
            "CCS",
            "Carbon Capture and Storage",
            "Carbon Capture and Storage (CCS)",
            "Carbon capture and storage (CCS)",
        ]
        plan = build_merge_plan(names)
        self.assertEqual(len(plan["merges"]), 1)
        self.assertEqual(plan["merges"][0]["target"], "CCS")
        self.assertEqual(len(plan["merges"][0]["sources"]), 4)

    def test_target_always_in_sources(self):
        plan = build_merge_plan(["Fault", "Faults"])
        merge = plan["merges"][0]
        self.assertIn(merge["target"], merge["sources"])

    def test_different_concepts_not_merged(self):
        plan = build_merge_plan(
            ["Fault", "Pre-Existing Faults", "Caprock", "Caprock Fracturing"]
        )
        self.assertEqual(plan["merges"], [])
        self.assertEqual(plan["kept"], 4)

    def test_noise_excluded_from_groups(self):
        # 带尾随空格的是**另一个**图节点（LightRAG 按 id 精确匹配），
        # 两条都要删，不能因为 strip 后相同就合并成一条计划
        plan = build_merge_plan(["University of Oslo", "University of Oslo ", "CO2"])
        self.assertEqual(len(plan["noise"]), 2)
        self.assertEqual(plan["merges"], [])

    def test_accepts_attr_dict_as_type(self):
        # 调用方常直接传 graphml 读出来的属性字典
        plan = build_merge_plan(["Abhinav Prakash Gahlot"], {"Abhinav Prakash Gahlot": {"entity_type": "person"}})
        self.assertEqual(plan["noise"][0]["reason"], "type:person")

    def test_plan_is_deterministic(self):
        names = ["CO2 Plume", "CO2 plume", "CO2 Plumes", "Fault", "Faults"]
        first = build_merge_plan(names)
        second = build_merge_plan(list(reversed(names)))
        self.assertEqual(first, second)

    def test_summary_counts(self):
        plan = build_merge_plan(["CO2", "Carbon Dioxide (CO2)", "Fault", "2050"])
        text = plan_summary(plan)
        self.assertIn("合并组 1 个", text)
        self.assertIn("脏节点 1 个", text)


if __name__ == "__main__":
    unittest.main()
