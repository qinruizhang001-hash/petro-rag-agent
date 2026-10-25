"""
策展领域本体的单测
==================
本体是**人工数据**，所以它最容易出的错不是崩溃，而是悄悄退化：
某个维度的实体被删到 5 个、某条关系的端点写错名字被静默跳过、
重复执行合并脚本每次都多写几条边。这些都不会报错，只会让
"覆盖度 3.2 倍提升"这句话变成假的。

所以这里锁四件事：
1. 本体文件本身合格（字段齐、类型在抽取 schema 内、维度不塌方）；
2. 合并是幂等的（跑两次第二次必须新增 0）；
3. 大小写去重生效（``Zechstein caprock`` 与 ``Zechstein Caprock`` 已是两个节点，
   本体不能造出第三个）；
4. 端点解析不到时必须**报出来**，不能默默丢掉。
"""
import json
import os
import sys
import tempfile
import textwrap
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.augment_graph import (  # noqa: E402
    ALLOWED_TYPES,
    DEFAULT_ONTOLOGY,
    load_ontology,
    merge,
    write_augmented,
)

DIMENSIONS = {"reservoir", "fault", "caprock", "wellbore", "monitoring", "regulation"}

# 关系里引用的**图谱既有节点**（不是本体实体）。改动这张表等于承认
# 本体依赖抽取图谱里的具体节点名，必须同步改本体。
GRAPH_BRIDGES = {
    "Zechstein Caprock", "Fault", "Caprock", "Caprock Integrity Failure",
    "Overburden", "Cement Sheath", "Injection Well", "Monitoring Well",
    "Saline Aquifer",
}

_GRAPHML_TMPL = textwrap.dedent(
    """\
    <?xml version='1.0' encoding='utf-8'?>
    <graphml xmlns="http://graphml.graphdrawing.org/xmlns">
      <key id="d0" for="node" attr.name="entity_id" attr.type="string" />
      <key id="d1" for="node" attr.name="entity_type" attr.type="string" />
      <key id="d2" for="node" attr.name="description" attr.type="string" />
      <key id="d3" for="node" attr.name="source_id" attr.type="string" />
      <key id="d4" for="node" attr.name="file_path" attr.type="string" />
      <key id="d7" for="edge" attr.name="weight" attr.type="double" />
      <key id="d8" for="edge" attr.name="description" attr.type="string" />
      <key id="d9" for="edge" attr.name="keywords" attr.type="string" />
      <key id="d10" for="edge" attr.name="source_id" attr.type="string" />
      <key id="d11" for="edge" attr.name="file_path" attr.type="string" />
      <graph edgedefault="undirected">
    {body}
      </graph>
    </graphml>
    """
)


def _node(name: str, etype: str = "geologicalbody") -> str:
    return (
        f'    <node id="{name}">'
        f'<data key="d0">{name}</data><data key="d1">{etype}</data>'
        f'<data key="d2">d</data><data key="d3">doc-1-chunk-000</data>'
        f'<data key="d4">a.pdf</data></node>'
    )


def _edge(a: str, b: str) -> str:
    return (
        f'    <edge source="{a}" target="{b}">'
        f'<data key="d7">1.0</data><data key="d8">d</data>'
        f'<data key="d9">k</data><data key="d10">doc-1-chunk-000</data>'
        f'<data key="d11">a.pdf</data></edge>'
    )


class TestOntologyFile(unittest.TestCase):
    def setUp(self):
        self.onto = load_ontology()

    def test_has_version(self):
        self.assertTrue(self.onto["version"])

    def test_every_entity_has_a_reference(self):
        """没有出处的实体不许进本体 —— 否则就是把模型的记忆当知识。"""
        missing = [e["name"] for e in self.onto["entities"] if not e.get("reference")]
        self.assertEqual(missing, [])

    def test_every_relation_has_a_reference(self):
        missing = [f"{r['from']}->{r['to']}" for r in self.onto["relations"]
                   if not r.get("reference")]
        self.assertEqual(missing, [])

    def test_types_stay_inside_the_extraction_schema(self):
        """本体不能引入抽取阶段不允许的类型。

        ``utils/kg_prompt.py`` 只允许 8 类；本体加一个 ``concept``，
        两边对同一张图的类型统计就不可比了。
        """
        bad = {e["name"]: e["entity_type"] for e in self.onto["entities"]
               if e["entity_type"] not in ALLOWED_TYPES}
        self.assertEqual(bad, {})

    def test_dimensions_are_the_six_named_by_the_audit(self):
        used = {e["dimension"] for e in self.onto["entities"]}
        self.assertEqual(used, DIMENSIONS)

    def test_no_dimension_collapses(self):
        """每个维度至少 20 个实体。

        门槛不高，但能挡住"某个维度被删到 3 个而没人发现"——
        覆盖度提升正是靠这几个维度撑起来的。
        """
        counts = {}
        for e in self.onto["entities"]:
            counts[e["dimension"]] = counts.get(e["dimension"], 0) + 1
        thin = {k: v for k, v in counts.items() if v < 20}
        self.assertEqual(thin, {}, f"维度实体过少: {thin}")

    def test_no_duplicate_names_ignoring_case(self):
        seen, dups = {}, []
        for e in self.onto["entities"]:
            low = e["name"].lower()
            if low in seen:
                dups.append((seen[low], e["name"]))
            seen[low] = e["name"]
        self.assertEqual(dups, [])

    def test_relation_endpoints_resolve(self):
        names = {e["name"] for e in self.onto["entities"]}
        dangling = []
        for r in self.onto["relations"]:
            for side in (r["from"], r["to"]):
                if side not in names and side not in GRAPH_BRIDGES:
                    dangling.append(f"{r['from']}->{r['to']} ({side})")
        self.assertEqual(dangling, [], f"端点解析不到: {dangling}")

    def test_no_duplicate_relation_pairs(self):
        pairs, dups = set(), []
        for r in self.onto["relations"]:
            key = frozenset((r["from"], r["to"]))
            if key in pairs:
                dups.append(f"{r['from']}->{r['to']}")
            pairs.add(key)
        self.assertEqual(dups, [], f"重复关系: {dups}")


class TestValidation(unittest.TestCase):
    """校验必须炸，不能静默放行。"""

    def _write(self, payload: dict) -> str:
        fd, path = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False)
        return path

    def test_rejects_type_outside_schema(self):
        path = self._write({"version": "t", "entities": [
            {"name": "X", "entity_type": "concept", "dimension": "caprock",
             "description": "d", "reference": "r"}], "relations": []})
        try:
            with self.assertRaises(ValueError):
                load_ontology(path)
        finally:
            os.unlink(path)

    def test_rejects_entity_without_reference(self):
        path = self._write({"version": "t", "entities": [
            {"name": "X", "entity_type": "process", "dimension": "caprock",
             "description": "d"}], "relations": []})
        try:
            with self.assertRaises(ValueError):
                load_ontology(path)
        finally:
            os.unlink(path)

    def test_rejects_duplicate_entity(self):
        ent = {"name": "X", "entity_type": "process", "dimension": "caprock",
               "description": "d", "reference": "r"}
        path = self._write({"version": "t", "entities": [ent, dict(ent)],
                            "relations": []})
        try:
            with self.assertRaises(ValueError):
                load_ontology(path)
        finally:
            os.unlink(path)


class TestMerge(unittest.TestCase):
    def setUp(self):
        self.onto = load_ontology()

    def test_merge_adds_nodes_and_edges(self):
        nodes = ["Fault", "Caprock"]
        edges = [("Fault", "Caprock")]
        attrs = {n: {"entity_type": "geologicalbody"} for n in nodes}
        out = merge(nodes, edges, attrs, self.onto)
        self.assertGreater(out["stats"]["added_nodes"], 100)
        self.assertGreater(out["stats"]["added_edges"], 80)
        self.assertEqual(out["stats"]["nodes_before"], 2)
        self.assertEqual(len(out["nodes"]), out["stats"]["nodes_after"])

    def test_merge_is_idempotent(self):
        """跑第二次必须新增 0 —— 否则每次部署都会往图里灌重复节点。"""
        nodes = ["Fault", "Caprock"]
        edges = [("Fault", "Caprock")]
        attrs = {n: {"entity_type": "geologicalbody"} for n in nodes}
        first = merge(nodes, edges, attrs, self.onto)
        second = merge(first["nodes"], first["edges"], first["attrs"], self.onto)
        self.assertEqual(second["stats"]["added_nodes"], 0)
        self.assertEqual(second["stats"]["added_edges"], 0)

    def test_case_insensitive_dedup(self):
        """图里已有 ``Zechstein caprock`` 时，本体不能再造一个 ``Zechstein Caprock``。"""
        nodes = ["Zechstein caprock", "Fault", "Caprock"]
        edges = [("Fault", "Caprock")]
        attrs = {n: {"entity_type": "geologicalbody"} for n in nodes}
        out = merge(nodes, edges, attrs, self.onto)
        lowered = [n.lower() for n in out["nodes"]]
        self.assertEqual(lowered.count("zechstein caprock"), 1)

    def test_unresolved_relations_are_reported_not_dropped(self):
        """桥接节点不在图里时，关系必须计入 unresolved，而不是静默消失。"""
        out = merge([], [], {}, self.onto)
        self.assertGreater(out["stats"]["unresolved_edges"], 0)
        self.assertTrue(out["stats"]["unresolved_examples"])

    def test_bridge_nodes_come_from_the_graph(self):
        """空图上合并时，未解析的端点必须正好是那批图谱桥接节点。"""
        out = merge([], [], {}, self.onto)
        expected = sum(1 for r in self.onto["relations"]
                       if r["from"] in GRAPH_BRIDGES or r["to"] in GRAPH_BRIDGES)
        self.assertEqual(out["stats"]["unresolved_edges"], expected)

    def test_overrides_cover_every_added_node(self):
        """每个新增节点都要有维度声明，否则审计会漏算它。"""
        out = merge([], [], {}, self.onto)
        missing = [n for n in out["stats"]["added_node_names"]
                   if n not in out["overrides"]]
        self.assertEqual(missing, [])

    def test_deduped_node_inherits_dimension(self):
        """本体实体撞上既有节点时，既有节点也要拿到维度声明。"""
        nodes = ["Storage Permit"]
        out = merge(nodes, [], {"Storage Permit": {"entity_type": "regulation"}},
                    self.onto)
        self.assertIn("regulation", out["overrides"]["Storage Permit"])


class TestSurvivesNormalization(unittest.TestCase):
    """``normalize_graph.py --apply`` 不能把本体当噪声删掉。

    本体节点的 ``file_path`` 是 ``domain_ontology``，抽取侧的噪声规则
    （太短、引用、标题句、裸数字、机构、人名、年份、期刊、街道地址）
    本来跟它无关。但规则是会改的，一旦有人把 ``title-like`` 的判定
    放宽到"超过 N 个词"，``Australia Offshore Petroleum and Greenhouse
    Gas Storage Act`` 这类长名字就会被清掉 —— 而且不会报错，
    只会让覆盖度悄悄退回原样。
    """

    def test_no_entity_is_classified_as_noise(self):
        from utils.entity_aliases import noise_reason

        flagged = []
        for e in load_ontology()["entities"]:
            reason = noise_reason(e["name"], e["entity_type"])
            if reason:
                flagged.append(f"{e['name']} → {reason}")
        self.assertEqual(flagged, [], f"本体实体被噪声规则命中: {flagged}")

    def test_no_entity_is_merged_away_by_alias_rules(self):
        """别名规则若把两个本体实体归成一组，说明本体内有重复概念。"""
        from utils.entity_aliases import canonical_name

        groups: dict[str, list[str]] = {}
        for e in load_ontology()["entities"]:
            groups.setdefault(canonical_name(e["name"]), []).append(e["name"])
        clashes = {k: v for k, v in groups.items() if len(v) > 1}
        self.assertEqual(clashes, {}, f"本体内概念重复: {clashes}")


class TestWriteRoundTrip(unittest.TestCase):
    def setUp(self):
        self.onto = load_ontology()

    def _graph(self, body: str) -> str:
        fd, path = tempfile.mkstemp(suffix=".graphml")
        os.close(fd)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(_GRAPHML_TMPL.format(body=body))
        return path

    def test_write_then_read_back(self):
        from utils.graph_analysis import load_graph

        src = self._graph("\n".join([
            _node("Fault"), _node("Caprock"),
            _edge("Fault", "Caprock"),
        ]))
        try:
            stats = write_augmented(src, self.onto, out_path=src, backup=False)
            self.assertTrue(stats["written"])
            nodes, edges, attrs = load_graph(src)
            self.assertEqual(len(nodes), stats["nodes_after"])
            self.assertEqual(len(edges), stats["edges_after"])
            # 新节点的 entity_id 必须等于节点 id，LightRAG 靠它反查
            self.assertEqual(attrs["Zechstein Halite"]["entity_id"],
                             "Zechstein Halite")
            self.assertEqual(attrs["Zechstein Halite"]["file_path"],
                             "domain_ontology")

            again = write_augmented(src, self.onto, out_path=src, backup=False)
            self.assertFalse(again["written"])  # 幂等：没有可写的东西
        finally:
            os.unlink(src)

    def test_write_preserves_original_attributes(self):
        from utils.graph_analysis import load_graph

        src = self._graph("\n".join([
            _node("Fault"), _node("Caprock"),
            _edge("Fault", "Caprock"),
        ]))
        try:
            write_augmented(src, self.onto, out_path=src, backup=False)
            _, _, attrs = load_graph(src)
            self.assertEqual(attrs["Fault"]["file_path"], "a.pdf")
            self.assertEqual(attrs["Fault"]["source_id"], "doc-1-chunk-000")
        finally:
            os.unlink(src)

    def test_backup_created_for_in_place_write(self):
        src = self._graph("\n".join([_node("Fault"), _node("Caprock")]))
        try:
            stats = write_augmented(src, self.onto, out_path=None, backup=True)
            self.assertTrue(stats.get("backup"))
            self.assertTrue(os.path.exists(stats["backup"]))
            os.unlink(stats["backup"])
        finally:
            os.unlink(src)

    def test_case_variant_nodes_do_not_break_idempotency(self):
        """图里同时存在 ``Zechstein Caprock`` 与 ``Zechstein caprock`` 时，
        两次写入不能产生不同的解析结果。

        这是实际踩到的 bug：写入路径曾用 set 建"小写名 → 节点名"映射，
        而 merge 用 list 建。set 的迭代顺序不稳定，同一个本体关系
        两次运行可能解析到不同的大小写变体，于是每次重跑都多写一条边
        （实测第二次多出 2 条）。幂等性就这样在"看起来成功"的情况下失效。
        """
        from utils.graph_analysis import load_graph

        src = self._graph("\n".join([
            _node("Zechstein Caprock"), _node("Zechstein caprock"),
            _node("Fault"),
            _edge("Fault", "Zechstein Caprock"),
        ]))
        try:
            write_augmented(src, self.onto, out_path=src, backup=False)
            _, edges_1, _ = load_graph(src)
            again = write_augmented(src, self.onto, out_path=src, backup=False)
            _, edges_2, _ = load_graph(src)
            self.assertEqual(again["added_edges"], 0)
            self.assertEqual(len(edges_1), len(edges_2))
        finally:
            os.unlink(src)

    def test_missing_required_node_key_raises(self):
        """graphml 缺 entity_type 键时必须炸，不能写出一张没类型的图。"""
        fd, path = tempfile.mkstemp(suffix=".graphml")
        os.close(fd)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(textwrap.dedent(
                """\
                <?xml version='1.0' encoding='utf-8'?>
                <graphml xmlns="http://graphml.graphdrawing.org/xmlns">
                  <key id="d0" for="node" attr.name="entity_id" attr.type="string" />
                  <graph edgedefault="undirected">
                    <node id="Fault"><data key="d0">Fault</data></node>
                  </graph>
                </graphml>
                """
            ))
        try:
            with self.assertRaises(ValueError):
                write_augmented(path, self.onto, out_path=path, backup=False)
        finally:
            os.unlink(path)


class TestRealOntologyPath(unittest.TestCase):
    def test_default_path_exists(self):
        self.assertTrue(os.path.exists(DEFAULT_ONTOLOGY),
                        f"本体文件缺失: {DEFAULT_ONTOLOGY}")


if __name__ == "__main__":
    unittest.main()
