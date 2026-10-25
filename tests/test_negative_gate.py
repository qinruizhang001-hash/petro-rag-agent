"""整图负向测试：超破裂压力的输入，最终报告不允许是"安全/可通过"。

为什么需要这个测试
------------------
``tests/test_quality_gate.py`` 钉住的是**判定函数**。但"函数判失败"和"图真的
拒绝了这份报告"之间还隔着路由：条件边接错一个分支名，失败态就会静默走进
``save_file_memory`` —— 报告被写进历史记忆、API 返回 200，而单测全绿。

所以这里跑**真实的图**（``workflow.py`` 里注册的节点与边，含
``evaluate → check_quality → {pass,rewrite,fail}`` 三条边），只有两处是假的：

* **上游节点**（检索 / 抽取 / 生成）：它们需要 Ollama + 向量库，与本测试要验证的
  东西无关，桩掉。
* **LLM 输出**：固定返回一份"注入顺畅、风险可控"的报告 —— 正是物理规则要拦的
  那种措辞。

``utils/physics_rules.py``、``nodes/evaluate_node.py``、``core/gates.py``、
``nodes/gate_node.py``、``nodes/conditions.py`` 全部是真实代码，一行没替。

为什么必须挂在 push CI 上
------------------------
这个负例只在 e2e.yml（自建 runner、手动触发）里跑过一版，结果是连续几周没人
跑它。闸门是这类项目唯一"错了会出事"的部分，它的回归测试不能依赖"记得手动点
一下"。
"""
from __future__ import annotations

import contextlib
import os
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from tests import _graph_stubs

_graph_stubs.install_leaf_stubs()

# 文件记忆必须指到临时目录：否则一次测试就会往真实的 project_history.md 里
# 写记录（这个坑在 e2e 里踩过，测试问句混进了真实历史）。
_TMP = tempfile.mkdtemp(prefix="ccus-gate-")
_HISTORY_PATH = os.path.join(_TMP, "project_history.md")
_PROFILE_PATH = os.path.join(_TMP, "user_profile.md")

# **必须无条件覆盖，不能用 setdefault。** `.env` 里写着 `PROJECT_HISTORY_PATH=`
# （留空），`load_dotenv()` 会把这个键设成**空字符串** —— 键存在、值为空，
# setdefault 于是什么也不做。随后若照抄环境变量写进模块常量，历史路径就成了
# ``''``，直到图跑到 ``ensure_files()`` 才炸成 ``FileNotFoundError: ''``，
# 离现场已经隔了七八层 langgraph 栈。
os.environ["PROJECT_HISTORY_PATH"] = _HISTORY_PATH
os.environ["USER_PROFILE_PATH"] = _PROFILE_PATH

# 环境变量只对**尚未导入**的模块有效：``file_memory`` 在 import 时就把路径
# 固化成了模块常量（``PROJECT_HISTORY_PATH = _env_path(...)``）。discover 顺序下
# 它通常已经被前面的测试导入过，于是上面两行形同虚设 —— 必须直接改已导入模块
# 的常量。与 ``eval_ragas._redirect_loaded_file_memory`` 是同一个坑、同一个解法。
_file_memory = sys.modules.get("file_memory")
if _file_memory is not None:
    _file_memory.PROJECT_HISTORY_PATH = _HISTORY_PATH
    _file_memory.USER_PROFILE_PATH = _PROFILE_PATH

OVER_PRESSURE_PARAMS = "注入压力 400 bar；埋深 2500 m；温度 80℃"
SAFE_PARAMS = "注入压力 300 bar；埋深 2500 m；温度 80℃"

# 物理规则按 ρ=1030 kg/m³ / 1.5× 静水计算：2500 m 处 P_h≈253 bar、P_frac≈379 bar。
# 400 bar > 379 bar → blocking；300 bar 落在 [253, 379] → safe_window。
UNSAFE_REPORT = (
    "注入压力 400 bar 远低于地层破裂压力，注入顺畅，风险可控，"
    "各项指标均在安全范围内，可通过验收。"
)
SAFE_REPORT = (
    "注入压力 300 bar 落在 2500 m 处的安全注入窗口内，"
    "但仍需监测盖层完整性。"
)


def _resp(content: str):
    class _Message:
        def __init__(self):
            self.content = content

    class _Choice:
        def __init__(self):
            self.message = _Message()

    class _Response:
        def __init__(self):
            self.choices = [_Choice()]
            self.usage = None

    return _Response()


class _FakeStream:
    def __init__(self, text: str):
        self.text = text
        self.prompt_tokens = 120
        self.completion_tokens = 60
        self.usage_estimated = False


# ─── 上游节点桩（检索 / 抽取 / 生成 / 落盘）───────────────
# 返回的字段与真实节点一致；不产出结论以外的副作用。
def _memory_router(_state):
    return {"need_memory": False}


def _noop(_state):
    return {}


# LangGraph 只保留 AgentState 里声明过的键，测试用的"传参"不能塞进 state ——
# 塞了会被静默丢掉（这里踩过一次：正例拿着负例的 400 bar 跑）。
_CURRENT_PARAMS = {"text": OVER_PRESSURE_PARAMS}


def _extract(_state):
    return {
        "extracted_params": _CURRENT_PARAMS["text"],
        "complexity_mode": "local",
        "complexity_info": {"label": "本地", "reason": "test"},
    }


def _retrieve_geology(_state):
    return {
        "geology_evidence": "某区块盖层为泥岩，破裂压力梯度 1.5–1.8 倍静水。",
        "geology_status": {"ok": True, "elapsed_s": 0.0},
    }


def _retrieve_wellbore(_state):
    return {
        "wellbore_evidence": "注入压力需低于地层破裂压力，否则存在盖层压裂风险。",
        "wellbore_status": {"ok": True, "elapsed_s": 0.0},
    }


def _grade(_state):
    return {
        "retrieval_grade": {"status": "graded", "geology": {"score": 0.9}, "wellbore": {"score": 0.9}},
        "retrieval_retry_count": 0,
        "retrieval_strategy": "primary",
    }


def _generate_geology(_state):
    return {"geology_report": "盖层为泥岩，破裂压力梯度约 1.5 倍静水。"}


def _generate_wellbore(_state):
    return {"wellbore_report": "注入压力应低于破裂压力。"}


def _save_file_memory(_state):
    return {"history_saved": True}


_STUB_NODES = {
    "nodes.memory_node": {
        "memory_router": _memory_router,
        "search_memory": _noop,
        "inject_memory": _noop,
    },
    "nodes.extract_node": {"extract_params": _extract},
    "nodes.query_nodes": {"rewrite_node": _noop, "grade_node": _grade},
    "nodes.geology_node": {
        "retrieve_geology": _retrieve_geology,
        "generate_geology": _generate_geology,
    },
    "nodes.wellbore_node": {
        "retrieve_wellbore": _retrieve_wellbore,
        "generate_wellbore": _generate_wellbore,
    },
    "nodes.history_node": {"save_file_memory": _save_file_memory},
}


def _load_graph():
    """导入真实的 ``workflow`` 模块，只在导入期间替换上游节点。

    导入完立刻把真实模块还原回 ``sys.modules``：否则后面对
    ``nodes.query_nodes`` 等模块的测试会拿到桩（顺序依赖的坑）。

    **导入之后必须验证桩真的生效。** 这套替换只在"workflow 第一次被导入"时
    起作用：若别的测试模块先 ``import workflow``，``sys.modules`` 里已经有
    缓存，这里拿到的就是**绑定真实节点**的那张图 —— 而 ``save_file_memory``
    是真的，于是每跑一次整图就往 ``project_history.md`` 里写 4-5 条测试问句。
    这个后果不是"测试变慢"，是把真实记忆污染掉，而且没有任何断言会失败。
    实测发生过一次（一个新测试模块在模块顶层 import 了 workflow）。
    """
    saved = {name: sys.modules.get(name) for name in _STUB_NODES}
    for name, attrs in _STUB_NODES.items():
        module = types.ModuleType(name)
        module.__dict__.update(attrs)
        sys.modules[name] = module
    try:
        import workflow  # noqa: WPS433  真实图定义
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
    if getattr(workflow, "save_file_memory", None) is not _save_file_memory:
        raise AssertionError(
            "workflow 里的 save_file_memory 不是桩：说明 workflow 在本次导入前"
            "已经被别的测试模块导入过，图绑定了真实节点。继续跑会往真实的 "
            "project_history.md 写测试问句。请检查哪个测试模块在顶层 import 了 "
            "workflow，把它改成在用例内部导入。"
        )
    _assert_memory_is_isolated()
    return workflow


def _assert_memory_is_isolated():
    """历史路径必须是本模块的临时目录，既不能为空、也不能指回仓库。

    空串会等到图跑到 ``ensure_files()`` 才炸（``FileNotFoundError: ''``），
    错误现场离原因隔了七八层栈；指回仓库则更糟 —— 不报错，只是把测试问句
    静默写进真实历史。两种都在这里提前拦下。
    """
    module = sys.modules.get("file_memory")
    if module is None:
        return
    for attr, expected in (
        ("PROJECT_HISTORY_PATH", _HISTORY_PATH),
        ("USER_PROFILE_PATH", _PROFILE_PATH),
    ):
        actual = str(getattr(module, attr, "") or "")
        if not actual.strip():
            raise AssertionError(
                f"file_memory.{attr} 是空串。多半是照抄了 os.environ 里的空值"
                f"（.env 里该键留空），而没走 file_memory._env_path 的空值兜底。"
            )
        if os.path.normcase(os.path.abspath(actual)) != os.path.normcase(
            os.path.abspath(expected)
        ):
            raise AssertionError(
                f"file_memory.{attr} 没指向本测试的临时目录：{actual!r}。"
                f"继续跑会把测试数据写进真实文件。"
            )



@contextlib.contextmanager
def _llm(report_text: str, score_text: str = "分数：9\n评语：结构清晰，证据充分。"):
    with patch(
        "nodes.aggregate_node.get_llm", return_value=(None, "local:qwen2.5", "local", None)
    ), patch("nodes.aggregate_node.llm_chat", side_effect=lambda *_a, **_k: _resp("")), patch(
        "nodes.aggregate_node.llm_chat_streamed", return_value=_FakeStream(report_text)
    ), patch(
        "nodes.evaluate_node.get_llm", return_value=(None, "local:qwen2.5", "local", None)
    ), patch("nodes.evaluate_node.llm_chat", side_effect=lambda *_a, **_k: _resp(score_text)):
        yield


class GraphNegativeGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workflow = _load_graph()

    def _run(self, params: str, report: str):
        _CURRENT_PARAMS["text"] = params
        with _llm(report):
            return self.workflow.app_stream.invoke(
                {"user_input": "某 CCUS 项目注入压力与埋深如下，评估风险。"}
            )

    def test_over_fracture_pressure_never_reaches_the_delivery_path(self):
        final = self._run(OVER_PRESSURE_PARAMS, UNSAFE_REPORT)
        gate = final.get("quality_gate") or {}

        self.assertEqual(gate.get("status"), "failed")
        self.assertTrue(gate.get("physics_blocking"))
        self.assertTrue(gate.get("quarantined"))
        self.assertEqual(final.get("report_status"), "failed")
        # 失败路径不写历史记忆：被拒的报告不能成为后续评估的先验。
        self.assertFalse(final.get("history_saved"))
        # 没有经过人工闸门 —— 说明确实走的是 fail 边，而不是"重试用尽后放行"。
        self.assertNotIn("human_decision", final)

    def test_quarantined_report_cannot_claim_safety(self):
        final = self._run(OVER_PRESSURE_PARAMS, UNSAFE_REPORT)
        delivered = final.get("final_report") or ""
        for claim in ("风险可控", "注入顺畅", "可通过验收", "安全范围内"):
            self.assertNotIn(claim, delivered, msg=f"交付字段里仍有「{claim}」")
        self.assertIn("不得作为工程依据", delivered)
        # 原稿必须留档：否则无法判断模型到底写了什么。
        self.assertIn("风险可控", final.get("draft_report") or "")

    def test_good_input_still_passes(self):
        final = self._run(SAFE_PARAMS, SAFE_REPORT)
        gate = final.get("quality_gate") or {}

        self.assertEqual(gate.get("status"), "passed")
        self.assertFalse(gate.get("physics_blocking"))
        self.assertIn(SAFE_REPORT, final.get("final_report") or "")
        # 通过路径必经人工闸门（自动模式下自放行并留痕）。
        self.assertTrue((final.get("human_decision") or {}).get("approved"))

    def test_rewrite_budget_is_spent_before_failing(self):
        final = self._run(OVER_PRESSURE_PARAMS, UNSAFE_REPORT)
        # 初稿 + 两次重写：物理否决在预算内一直是 rewrite，用尽才 fail。
        self.assertEqual(final.get("retry_count"), 2)

    def test_injection_plan_is_produced_by_the_graph_not_just_imported(self):
        """
        闭环优化的接线：图里必须真的产出 ``injection_plan``。

        这个断言看起来平淡，防的却是这个项目已经踩过一次的坑 ——
        "模块写好了、import 也加了、但没人调用"。只测 ``optimize_plan``
        本身发现不了这种情况：它单测全绿，而报告里根本没有建议。
        所以这里从整图的**最终状态**读，并确认它进了 ``tool_facts``
        （没进 tool_facts 的内容在忠实度评估眼里就是"无出处"）。
        """
        final = self._run(SAFE_PARAMS, SAFE_REPORT)
        plan = final.get("injection_plan") or {}
        self.assertTrue(plan, "图跑完了却没有 injection_plan")
        self.assertEqual(plan.get("status"), "ok")
        self.assertTrue(plan.get("recommended"))
        self.assertTrue(plan.get("text"))
        facts = " ".join(final.get("tool_facts") or [])
        self.assertIn("注入策略优化", facts)
        # 策略建议与 UQ 摘要必须指向同一份数据，否则前端两处会显示不同结论
        self.assertEqual((final.get("uq_summary") or {}).get("plan"), plan)


if __name__ == "__main__":
    unittest.main()
