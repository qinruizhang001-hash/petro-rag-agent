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


if __name__ == "__main__":
    unittest.main()
