"""
utils.graph_analysis 与 normalize_graph.project 的离线单测
==========================================================
连通分量数是我们用来回答"知识图谱到底连不连"的唯一指标。这个数算错了
不会报错 —— 它只会给出一个看起来很合理、但完全错误的结论。所以这里用
**手算得出的小图**把它钉死。

同时锁住 ``project()`` 的预演语义：归一化之后分量会怎么变，必须能在
真正执行（十几分钟）之前算出来。
"""
import os
import sys
import tempfile
import textwrap
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.graph_analysis import (
    connected_components,
    format_summary,
    load_graph,
    summarize,
)

_GRAPHML = textwrap.dedent(
    """\
    <?xml version='1.0' encoding='utf-8'?>
    <graphml xmlns="http://graphml.graphdrawing.org/xmlns">
      <key id="d1" for="node" attr.name="entity_type" attr.type="string" />
      <graph edgedefault="undirected">
        <node id="A"><data key="d1">concept</data></node>
        <node id="B"><data key="d1">method</data></node>
        <node id="C"><data key="d1">concept</data></node>
        <node id="D"><data key="d1">concept</data></node>
        <edge source="A" target="B" />
        <edge source="B" target="C" />
      </graph>
    </graphml>
    """
)


class TestConnectedComponents(unittest.TestCase):
    def test_chain_is_one_component(self):
        comps = connected_components(["A", "B", "C"], [("A", "B"), ("B", "C")])
        self.assertEqual(len(comps), 1)
        self.assertEqual(sorted(comps[0]), ["A", "B", "C"])

    def test_isolated_nodes_are_their_own_components(self):
        comps = connected_components(["A", "B", "C"], [("A", "B")])
        self.assertEqual(len(comps), 2)
        self.assertEqual(comps[0], ["A", "B"])
        self.assertEqual(comps[1], ["C"])

    def test_no_edges(self):
        comps = connected_components(["A", "B"], [])
        self.assertEqual(len(comps), 2)

    def test_empty_graph(self):
        self.assertEqual(connected_components([], []), [])

    def test_duplicate_edges_do_not_inflate(self):
        comps = connected_components(["A", "B"], [("A", "B"), ("A", "B"), ("B", "A")])
        self.assertEqual(len(comps), 1)

    def test_edge_endpoint_outside_node_list_is_counted(self):
        # 手工改过图时会出现悬空边。丢弃它会让分量数偏小 —— 正是要测量的指标
        comps = connected_components(["A"], [("A", "GHOST")])
        self.assertEqual(len(comps), 1)
        self.assertEqual(sorted(comps[0]), ["A", "GHOST"])

    def test_sorted_by_size_desc(self):
        comps = connected_components(
            ["A", "B", "C", "D", "E"], [("A", "B"), ("B", "C"), ("D", "E")]
        )
        self.assertEqual([len(c) for c in comps], [3, 2])


class TestLoadAndSummarize(unittest.TestCase):
    def setUp(self):
        fd, self.path = tempfile.mkstemp(suffix=".graphml")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(_GRAPHML)

    def tearDown(self):
        os.unlink(self.path)

    def test_load_graph(self):
        nodes, edges, attrs = load_graph(self.path)
        self.assertEqual(sorted(nodes), ["A", "B", "C", "D"])
        self.assertEqual(len(edges), 2)
        self.assertEqual(attrs["A"]["entity_type"], "concept")

    def test_summarize(self):
        stats = summarize(self.path)
        self.assertEqual(stats["nodes"], 4)
        self.assertEqual(stats["edges"], 2)
        self.assertEqual(stats["components"], 2)
        self.assertEqual(stats["largest"], 3)
        self.assertEqual(stats["isolated"], 1)
        self.assertAlmostEqual(stats["largest_ratio"], 0.75)
        self.assertEqual(stats["types"]["concept"], 3)

    def test_format_summary_contains_key_numbers(self):
        text = format_summary(summarize(self.path))
        self.assertIn("4 节点", text)
        self.assertIn("连通分量 2", text)


class TestProject(unittest.TestCase):
    """归一化预演：合并与删除之后，连通分量应该怎么变。"""

    def _write(self, body: str) -> str:
        fd, path = tempfile.mkstemp(suffix=".graphml")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(
                '<?xml version="1.0" encoding="utf-8"?>\n'
                '<graphml xmlns="http://graphml.graphdrawing.org/xmlns">\n'
                '<key id="d1" for="node" attr.name="entity_type" attr.type="string" />\n'
                f'<graph edgedefault="undirected">{body}</graph></graphml>'
            )
        return path

    def test_merge_joins_two_components(self):
        from normalize_graph import project

        # 旧图：CO2-A 与 Carbon Dioxide (CO2)-B 是两个分量；合并后应连成一个
        path = self._write(
            '<node id="CO2"><data key="d1">material</data></node>'
            '<node id="Reservoir"><data key="d1">geologicalbody</data></node>'
            '<node id="Carbon Dioxide (CO2)"><data key="d1">material</data></node>'
            '<node id="Caprock"><data key="d1">geologicalbody</data></node>'
            '<edge source="CO2" target="Reservoir" />'
            '<edge source="Carbon Dioxide (CO2)" target="Caprock" />'
        )
        try:
            nodes, edges, attrs = load_graph(path)
            from utils.entity_aliases import build_merge_plan

            plan = build_merge_plan(nodes, attrs)
            result = project(plan, nodes, edges)
            self.assertEqual(result["components"], 1)
            self.assertEqual(result["nodes"], 3)
        finally:
            os.unlink(path)

    def test_noise_removal_drops_its_edges(self):
        from normalize_graph import project

        path = self._write(
            '<node id="Fault"><data key="d1">geologicalbody</data></node>'
            '<node id="Reservoir"><data key="d1">geologicalbody</data></node>'
            '<node id="University of Oslo"><data key="d1">organization</data></node>'
            '<edge source="Fault" target="Reservoir" />'
            '<edge source="Fault" target="University of Oslo" />'
        )
        try:
            nodes, edges, attrs = load_graph(path)
            from utils.entity_aliases import build_merge_plan

            plan = build_merge_plan(nodes, attrs)
            result = project(plan, nodes, edges)
            self.assertEqual(result["nodes"], 2)
            self.assertEqual(result["edges"], 1)
        finally:
            os.unlink(path)

    def test_self_loop_after_merge_is_dropped(self):
        from normalize_graph import project
        from utils.entity_aliases import build_merge_plan

        path = self._write(
            '<node id="CO2"><data key="d1">material</data></node>'
            '<node id="Carbon Dioxide (CO2)"><data key="d1">material</data></node>'
            '<edge source="CO2" target="Carbon Dioxide (CO2)" />'
        )
        try:
            nodes, edges, attrs = load_graph(path)
            result = project(build_merge_plan(nodes, attrs), nodes, edges)
            self.assertEqual(result["nodes"], 1)
            self.assertEqual(result["edges"], 0)
            self.assertEqual(result["components"], 1)
        finally:
            os.unlink(path)


class TestDomainClassification(unittest.TestCase):
    """领域维度的词法分类。

    这一组测试的起因是一个真实 bug：``cement`` 不带词边界时，
    ``Reinforcement`` 与 ``Displacement`` 里的 ``-cement`` 会命中，
    把两个强化学习算法和一组位移场算成井筒 —— 井筒维度因此虚高 33%
    （142 vs 实际 107），而盖层维度看起来更加单薄。误判方向不是随机的：
    它总是把本来就不缺的维度撑大、把缺的维度显得更缺。
    """

    def test_cement_substring_is_not_wellbore(self):
        from utils.graph_analysis import classify_domains

        for name in ("Deep Reinforcement Learning", "Displacement u1(x1, 0)",
                     "Model-Free Reinforcement Learning", "Displacement Profile"):
            self.assertNotIn("wellbore", classify_domains(name, "technology"),
                             f"{name} 被误判为井筒")

    def test_real_cement_entities_are_wellbore(self):
        from utils.graph_analysis import classify_domains

        for name in ("Cement", "Cement Sheath", "Class G Cement",
                     "Cement-Sandstone/Shale Formation Interface"):
            self.assertIn("wellbore", classify_domains(name, "material"),
                          f"{name} 应当算井筒")

    def test_cement_plant_is_not_wellbore(self):
        """水泥厂/水泥产量属于 CCS 产业链，不是井筒完整性。"""
        from utils.graph_analysis import classify_domains

        for name in ("Kujawy Cement Plant", "Cement Production",
                     "Cement and Lime Production"):
            self.assertNotIn("wellbore", classify_domains(name, "technology"))

    def test_rl_policy_is_not_regulation(self):
        from utils.graph_analysis import classify_domains

        for name in ("Multi-Agent Deep Deterministic Policy Gradient (MADDPG)",
                     "Alternative Policy Mechanisms"):
            hit = classify_domains(name, "technology")
            if "Alternative" in name:  # 真政策，应当命中
                self.assertIn("regulation", hit)
            else:
                self.assertNotIn("regulation", hit, f"{name} 是 RL 算法不是法规")

    def test_standard_deviation_is_not_regulation(self):
        from utils.graph_analysis import classify_domains

        self.assertNotIn("regulation",
                         classify_domains("Posterior Standard Deviation", "parameter"))
        self.assertIn("regulation", classify_domains("Clear Regulatory Standards", ""))

    def test_author_names_typed_as_regulation_are_dropped(self):
        """抽取阶段把作者名标成 regulation，类型直通分支不能照收。"""
        from utils.graph_analysis import classify_domains

        for name in ("Borello et al.", "Cheung et al. (2012)",
                     "Bolourinejad and Herber"):
            self.assertNotIn("regulation", classify_domains(name, "regulation"),
                             f"{name} 是作者名")

    def test_entity_can_belong_to_two_dimensions(self):
        from utils.graph_analysis import classify_domains

        hit = classify_domains("Monitoring Well", "wellborecomponent")
        self.assertIn("wellbore", hit)
        self.assertIn("monitoring", hit)

    def test_reservoir_simulation_is_not_a_reservoir_type(self):
        from utils.graph_analysis import classify_domains

        self.assertNotIn("reservoir",
                         classify_domains("Reservoir Simulation", "technology"))
        self.assertIn("reservoir", classify_domains("Saline Aquifer", ""))

    def test_caprock_vocabulary(self):
        from utils.graph_analysis import classify_domains

        for name in ("Zechstein Caprock", "Nordland Shale", "Top Seals",
                     "Opalinus Clay", "Overburden"):
            self.assertIn("caprock", classify_domains(name, "geologicalbody"))


class TestDomainCoverage(unittest.TestCase):
    """覆盖统计的口径：声明维度优先，且必须能分清"抽的"和"策展的"。"""

    def _graph(self):
        nodes = ["Caprock", "Fault", "Shale Caprock", "Nordland Shale",
                 "Injection Well", "Unrelated Thing"]
        edges = [("Caprock", "Fault"), ("Injection Well", "Caprock")]
        attrs = {n: {"entity_type": "geologicalbody"} for n in nodes}
        return nodes, edges, attrs

    def test_lexical_only_counts_name_hits(self):
        from utils.graph_analysis import domain_coverage

        nodes, edges, attrs = self._graph()
        cov = domain_coverage(nodes, edges, attrs)
        # 命中三个：Caprock 本身、Shale Caprock（caprock+shale）、
        # Nordland Shale（shale）。Fault 与 Injection Well 不该进盖层维度。
        self.assertEqual(cov["dimensions"]["caprock"]["entities"], 3)
        self.assertEqual(cov["dimensions"]["caprock"]["declared"], 0)
        self.assertNotIn("Fault", cov["dimensions"]["caprock"]["examples"])

    def test_declared_dimension_overrides_lexical(self):
        """本体声明优先：名字猜不出来时也必须算对。

        "Zechstein Halite" 不含 caprock/seal/shale，靠正则永远进不了盖层维度；
        如果统计只靠正则，策展补的 40 个盖层实体会有大半不计数，
        报告就会低估自己补了什么。
        """
        from utils.graph_analysis import domain_coverage

        nodes = ["Zechstein Halite", "Unrelated Thing"]
        edges = [("Zechstein Halite", "Unrelated Thing")]
        attrs = {n: {"entity_type": "geologicalbody"} for n in nodes}
        plain = domain_coverage(nodes, edges, attrs)
        self.assertEqual(plain["dimensions"]["caprock"]["entities"], 0)

        declared = domain_coverage(nodes, edges, attrs,
                                   {"Zechstein Halite": ["caprock"]})
        self.assertEqual(declared["dimensions"]["caprock"]["entities"], 1)
        self.assertEqual(declared["dimensions"]["caprock"]["declared"], 1)
        self.assertEqual(declared["dimensions"]["caprock"]["connected_ratio"], 1.0)

    def test_isolated_ratio_is_reported(self):
        from utils.graph_analysis import domain_coverage

        nodes, edges, attrs = self._graph()
        cov = domain_coverage(nodes, edges, attrs,
                              {"Unrelated Thing": ["caprock"]})
        d = cov["dimensions"]["caprock"]
        # 盖层成员 4 个：Caprock / Shale Caprock / Nordland Shale / Unrelated Thing。
        # 只有 Caprock 有边（连 Fault 与 Injection Well），其余 3 个是孤立的。
        self.assertEqual(d["entities"], 4)
        self.assertEqual(d["isolated"], 3)
        self.assertLess(d["connected_ratio"], 1.0)

    def test_table_has_curated_column(self):
        from utils.graph_analysis import domain_coverage, format_domain_table

        nodes, edges, attrs = self._graph()
        table = format_domain_table(domain_coverage(nodes, edges, attrs))
        self.assertIn("策展", table)
        for label in ("储层类型", "断层体系", "盖层", "井筒", "监测", "法规"):
            self.assertIn(label, table)


if __name__ == "__main__":
    unittest.main()
