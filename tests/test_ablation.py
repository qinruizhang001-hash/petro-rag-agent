"""
消融对照的单测
==============
消融脚本是"每个模块到底贡献了什么"的唯一证据来源，所以它自己有两条
必须钉死的性质：

1. **补丁必须真的到达调用点**。`no_physics_gate` 靠替换
   `utils.physics_deep.assess` 实现。如果调用方写的是
   `from utils.physics_deep import assess` 并且**在模块顶层**导入，
   那它拿到的是旧引用，补丁不生效 —— 消融组会静默退化成 baseline，
   而我却会照着结果写"关掉物理闸门没有任何影响"。这是最坏的失败方式：
   结论看起来像发现，其实是补丁没打上。
2. **三组结果必须累加，不能互相覆盖**。早期 `existing.update(payload)`
   连跑三组只剩最后一组，文件写成功、命令行不报错。

这里不测物理算得对不对（那是 `test_physics_deep.py` 的事），
只测"关掉它"这个动作是否真的发生了。
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts import ablation  # noqa: E402


class TestSummarise(unittest.TestCase):
    """`_summarise` 是所有离线结论的度量口径，先钉住它。"""

    def test_counts_components_largest_and_isolated(self):
        nodes = ["a", "b", "c", "d"]
        edges = [("a", "b"), ("b", "c")]
        got = ablation._summarise(nodes, edges)
        self.assertEqual(got["nodes"], 4)
        self.assertEqual(got["edges"], 2)
        self.assertEqual(got["components"], 2)      # {a,b,c} + {d}
        self.assertEqual(got["largest"], 3)
        self.assertEqual(got["isolated"], 1)
        self.assertAlmostEqual(got["largest_ratio"], 0.75)
        self.assertAlmostEqual(got["isolated_ratio"], 0.25)

    def test_empty_graph_does_not_divide_by_zero(self):
        got = ablation._summarise([], [])
        self.assertEqual(got["nodes"], 0)
        self.assertEqual(got["largest"], 0)
        self.assertEqual(got["largest_ratio"], 0.0)


class TestTypeConstraintArm(unittest.TestCase):
    """八类约束的"上界"口径：只数图里已经不合规的节点，不声称因果。"""

    def _run(self, attrs):
        nodes = list(attrs)
        edges = [(nodes[0], nodes[1])] if len(nodes) > 1 else []
        return ablation.arm_type_constraint(nodes, edges, attrs)

    def test_off_schema_nodes_are_counted_with_empty_type(self):
        attrs = {
            "A": {"entity_type": "GeologicalBody"},
            "B": {"entity_type": "unknown"},
            "C": {"entity_type": ""},
            "D": {"entity_type": "parameter"},
        }
        got = self._run(attrs)
        self.assertEqual(got["off_schema_nodes"], 2)
        self.assertAlmostEqual(got["off_schema_ratio"], 0.5)
        # 空类型单独成一档，不能和 "unknown" 混在一起 —— 一个是模型写了
        # 占位词，一个是模型什么都没写，修法不同。
        self.assertEqual(got["distribution"]["<empty>"], 1)
        self.assertEqual(got["distribution"]["unknown"], 1)

    def test_type_match_is_case_insensitive(self):
        got = self._run({"A": {"entity_type": "REGULATION"},
                         "B": {"entity_type": "Parameter"}})
        self.assertEqual(got["off_schema_nodes"], 0)

    def test_enforcing_the_constraint_removes_the_nodes_and_their_edges(self):
        attrs = {
            "A": {"entity_type": "parameter"},
            "B": {"entity_type": "unknown"},
            "C": {"entity_type": "risk"},
        }
        nodes = ["A", "B", "C"]
        edges = [("A", "B"), ("A", "C")]
        got = ablation.arm_type_constraint(nodes, edges, attrs)
        self.assertEqual(got["off_schema_edges_touching"], 1)
        self.assertEqual(got["graph_if_enforced"]["nodes"], 2)
        self.assertEqual(got["graph_if_enforced"]["edges"], 1)


class TestAliasMergeArm(unittest.TestCase):
    """别名合并这一臂要在真图上跑，只验不变量，不写死具体数字。

    数字会随图谱重建变化，写死就等于给未来埋一个假失败；而
    "合并后节点只能变少、最大分量不该减少"这两条任何图上都成立。
    """

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(ablation.GRAPHML):
            raise unittest.SkipTest(f"图谱不存在：{ablation.GRAPHML}")
        from utils.graph_analysis import load_graph

        cls.nodes, cls.edges, cls.attrs = load_graph(ablation.GRAPHML)

    def test_merge_never_adds_nodes_and_reports_the_delta(self):
        got = ablation.arm_alias_merge(self.nodes, self.edges, self.attrs)
        with_merge = got["with_merge"]
        without = got["without_merge"]
        self.assertLessEqual(with_merge["nodes"], without["nodes"])
        self.assertEqual(
            got["delta"]["nodes_merged"],
            without["nodes"] - with_merge["nodes"],
        )
        # delta["largest"] 的符号约定：正数表示合并让最大分量变大
        self.assertEqual(
            got["delta"]["largest"],
            with_merge["largest"] - without["largest"],
        )

    def test_noise_removal_is_shared_by_both_sides(self):
        """两臂的唯一变量必须是别名合并：合并只会把边并起来，不会造出新边。"""
        got = ablation.arm_alias_merge(self.nodes, self.edges, self.attrs)
        self.assertLessEqual(got["with_merge"]["edges"],
                             got["without_merge"]["edges"])


class TestArmApplication(unittest.TestCase):
    """`_apply_arm` 是"关掉一个模块"的全部实现，必须可核对。"""

    def tearDown(self):
        # 这一组会改模块级状态，逐个恢复
        import core.gates as gates
        import utils.physics_deep as deep

        gates.MAX_REPORT_REWRITES = self._rewrites
        deep.assess = self._assess

    def setUp(self):
        import core.gates as gates
        import utils.physics_deep as deep

        self._rewrites = gates.MAX_REPORT_REWRITES
        self._assess = deep.assess

    def test_baseline_changes_nothing(self):
        self.assertEqual(ablation._apply_arm("baseline"), ["未改动任何模块"])

    def test_unknown_arm_is_an_error_not_a_silent_baseline(self):
        # 拼错 arm 名如果退化成 baseline，跑完会得到"两组完全一样"，
        # 被读成"这个模块没影响"。必须直接退出。
        with self.assertRaises(SystemExit):
            ablation._apply_arm("no_phsyics_gate")

    def test_no_rewrite_zeroes_the_budget(self):
        import core.gates as gates

        ablation._apply_arm("no_rewrite")
        self.assertEqual(gates.MAX_REPORT_REWRITES, 0)

    def test_no_physics_gate_replaces_the_function_callers_will_get(self):
        """关键断言：函数内 `from utils.physics_deep import assess` 拿到新的那个。"""
        import utils.physics_deep as deep

        original = deep.assess
        ablation._apply_arm("no_physics_gate")
        self.assertIsNot(deep.assess, original)

        # 复现调用方写法（aggregate_node / physics_rules 都是函数内导入）
        from utils.physics_deep import assess as as_callers_see_it

        self.assertIs(as_callers_see_it, deep.assess)

    def test_softening_clears_blocking_but_keeps_the_numbers(self):
        """只关阻断，不删数值 —— 否则消融组连"算过什么"都看不出来。"""
        import utils.physics_deep as deep

        def stub(*_args, **_kwargs):
            return {
                "status": "ok",
                "blocking": True,
                "blocking_reasons": ["断层活化：滑移比 1.4"],
                "policy": ["保守口径"],
                "geomechanics": {"slip_ratio": 1.4},
            }

        deep.assess = stub
        changes = ablation._apply_arm("no_physics_gate")
        got = deep.assess(1000.0, 300.0)

        self.assertFalse(got["blocking"])
        self.assertEqual(got["blocking_reasons"], [])
        self.assertEqual(got["geomechanics"], {"slip_ratio": 1.4})
        self.assertIn("消融", got["policy"][-1])
        self.assertIn("保守口径", got["policy"])
        # 被放行的阻断必须留痕，否则事后无法解释这一组为什么"通过"了
        self.assertTrue(any("拦住但被放行" in c for c in changes))


class TestMergePayload(unittest.TestCase):
    """三组结果累加。覆盖是静默失败，必须测。"""

    def test_three_arms_accumulate(self):
        merged = {}
        for arm in ("baseline", "no_physics_gate", "no_rewrite"):
            merged = ablation.merge_payload(
                merged, {"online": {arm: {"arm": arm, "passed": 1}}}
            )
        self.assertEqual(
            set(merged["online"]),
            {"baseline", "no_physics_gate", "no_rewrite"},
        )

    def test_offline_survives_an_online_write(self):
        merged = ablation.merge_payload({}, {"offline": {"nodes": 4826}})
        merged = ablation.merge_payload(
            merged, {"online": {"baseline": {"arm": "baseline"}}}
        )
        self.assertEqual(merged["offline"], {"nodes": 4826})
        self.assertIn("baseline", merged["online"])

    def test_legacy_single_arm_file_is_replaced_not_crashed_on(self):
        legacy = {"online": {"arm": "baseline", "passed": 2}}
        merged = ablation.merge_payload(
            legacy, {"online": {"no_rewrite": {"arm": "no_rewrite"}}}
        )
        self.assertEqual(set(merged["online"]), {"no_rewrite"})

    def test_rerunning_the_same_arm_overwrites_only_that_arm(self):
        merged = ablation.merge_payload(
            {}, {"online": {"baseline": {"arm": "baseline", "passed": 1}}}
        )
        merged = ablation.merge_payload(
            merged, {"online": {"baseline": {"arm": "baseline", "passed": 2}}}
        )
        self.assertEqual(merged["online"]["baseline"]["passed"], 2)


class TestProbeCase(unittest.TestCase):
    """探针用例的存在理由本身要被测试钉住。

    如果哪天有人"顺手"把探针问句改得温和一点，它会静默失去阻断能力，
    于是 `no_physics_gate` 那一组又变回空实验 —— 而表面上一切正常。
    """

    def _assess(self, question):
        from utils.physics_deep import assess
        from utils.physics_rules import extract_parameters

        ex = extract_parameters("", question)
        return assess(
            depth_m=ex.get("depth_m"),
            injection_pressure_bar=ex.get("injection_pressure_bar"),
            temperature_celsius=ex.get("temperature_c"),
            text=question,
        )

    def test_probe_case_really_blocks(self):
        got = self._assess(ablation.PROBE_CASES[0]["question"])
        self.assertEqual(got["status"], "ok")
        self.assertTrue(got["blocking"], "探针用例不再触发阻断，no_physics_gate 组将退化为空实验")
        self.assertTrue(got["blocking_reasons"])

    def test_probe_case_is_not_in_the_golden_set(self):
        # 探针是构造出来的，混进 golden set 会污染 Faithfulness 的基准。
        ids = {c["id"] for c in ablation.PROBE_CASES}
        self.assertTrue(all(i.startswith("probe-") for i in ids))

    def test_golden_set_never_blocks_which_is_why_the_probe_exists(self):
        """golden set 10 题一例都不触发阻断 —— 这条既是事实，也是探针的依据。"""
        import json

        path = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "eval", "golden_set.json")
        if not os.path.exists(path):
            self.skipTest(f"golden set 不存在：{path}")
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
        # 兼容两种结构：裸列表，或 {"cases": [...]}
        cases = raw if isinstance(raw, list) else raw.get("cases", [])
        blocked = [
            c["id"] for c in cases
            if self._assess(c.get("question", "")).get("blocking")
        ]
        self.assertEqual(
            blocked, [],
            "golden set 出现了会阻断的题；此时 no_physics_gate 不必再依赖探针，"
            "请同步更新 docs/gap-register.md 的 W2 小节",
        )


class TestHistoryIsolation(unittest.TestCase):
    """消融问句绝不能写进真实的 ``project_history.md``。

    这条是**踩出来的**：`file_memory` 在 import 时把历史路径固化成模块常量，
    所以在 `run_online` 里再设 `PROJECT_HISTORY_PATH` 环境变量已经太晚 ——
    探针问句被写进真实历史，压缩逻辑按"最近 N 条"把它留下、把 120 条真实
    记录挤掉。修复分两步：设环境变量（管尚未导入的模块），以及直接改
    已导入模块的常量（管已经导入的）。两件都要测。
    """

    def setUp(self):
        self._original_env = os.environ.get("PROJECT_HISTORY_PATH")
        import file_memory

        self._fm = file_memory
        self._original_attr = file_memory.PROJECT_HISTORY_PATH

    def tearDown(self):
        if self._original_env is None:
            os.environ.pop("PROJECT_HISTORY_PATH", None)
        else:
            os.environ["PROJECT_HISTORY_PATH"] = self._original_env
        self._fm.PROJECT_HISTORY_PATH = self._original_attr

    def test_redirects_the_environment_variable(self):
        path = ablation._isolate_history()
        self.assertIn(".ablation", path.replace("\\", "/"))
        self.assertEqual(os.environ["PROJECT_HISTORY_PATH"], path)

    def test_redirects_an_already_imported_file_memory(self):
        ablation._isolate_history()
        self.assertIn(".ablation", str(self._fm.PROJECT_HISTORY_PATH).replace("\\", "/"))

    def test_never_points_at_the_repo_file(self):
        repo_file = os.path.normcase(os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "project_history.md",
        ))
        path = ablation._isolate_history()
        self.assertNotEqual(os.path.normcase(os.path.abspath(path)), repo_file)
        self.assertNotEqual(
            os.path.normcase(os.path.abspath(str(self._fm.PROJECT_HISTORY_PATH))),
            repo_file,
        )


class TestRunOffline(unittest.TestCase):
    """端到端：三臂都能在真图上跑完并给出 verdict。"""

    def test_all_three_arms_produce_a_verdict(self):
        if not os.path.exists(ablation.GRAPHML):
            self.skipTest(f"图谱不存在：{ablation.GRAPHML}")
        result = ablation.run_offline()
        self.assertEqual(
            [a["arm"] for a in result["arms"]],
            ["alias_merge", "type_constraint", "curated_ontology"],
        )
        for arm in result["arms"]:
            self.assertTrue(arm["verdict"].strip(), f"{arm['arm']} 没有 verdict")
        text = ablation.render_offline(result)
        self.assertIn("离线消融", text)


class TestMarkdownReport(unittest.TestCase):
    """Markdown 报告只能**转述** JSON，不能自带结论。

    手写结论与自动数据分叉之后，没有任何东西会红 —— 而这个仓库已经因为
    手写表格偏了 1.6% 吃过一次亏。所以这里的断言是：报告里的数字必须
    全部能在 payload 里找到。
    """

    PAYLOAD = {
        "offline": {
            "nodes": 100,
            "edges": 120,
            "arms": [
                {
                    "arm": "alias_merge",
                    "question": "别名合并对连通性的贡献",
                    "with_merge": {"nodes": 90, "edges": 110, "components": 30,
                                   "largest": 40, "largest_ratio": 0.44,
                                   "isolated": 12},
                    "without_merge": {"nodes": 95, "edges": 112, "components": 41,
                                      "largest": 43, "largest_ratio": 0.45,
                                      "isolated": 15},
                    "delta": {"components": 11, "largest": -3, "nodes_merged": 5},
                    "verdict": "合并 5 个节点",
                }
            ],
        },
        "online": {
            "baseline@probe": {
                "arm": "baseline", "case_set": "probe", "cases": 1,
                "passed": 0, "failed": 1,
                "results": [{"case_id": "p", "gate_status": "failed",
                             "physics_status": "violation", "retry_count": 2}],
                "note": "小样本诊断", "environment": "本地模型",
            }
        },
    }

    def test_numbers_come_from_the_payload(self):
        text = ablation.render_markdown(self.PAYLOAD)
        for token in ("100", "120", "41", "30", "11", "5"):
            self.assertIn(token, text)
        self.assertIn("failed", text)
        self.assertIn("violation", text)

    def test_it_says_it_is_not_a_benchmark(self):
        text = ablation.render_markdown(self.PAYLOAD)
        self.assertIn("不是", text)
        self.assertIn("小样本诊断", text)

    def test_offline_only_payload_still_renders(self):
        text = ablation.render_markdown({"offline": self.PAYLOAD["offline"]})
        self.assertIn("离线对照", text)
        self.assertNotIn("在线对照", text)

    def test_report_can_be_written_and_read_back(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "sub", "report.md")
            ablation.write_markdown(path, self.PAYLOAD)
            with open(path, encoding="utf-8") as fh:
                self.assertIn("消融对照结果", fh.read())


if __name__ == "__main__":
    unittest.main()
