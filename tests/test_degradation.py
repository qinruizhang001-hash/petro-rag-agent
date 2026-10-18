"""
降级路径的端到端验证（真跑节点，不只看纯函数）
==============================================
``tests/test_branch_merge.py`` 只证明了 ``compose_branch_inputs`` /
``degradation_block`` 这两个纯函数的行为。但需求里"极其重要"的那条是：

    如果 geology_agent 报错挂了，状态里没有 geology_report，
    系统不能崩，而是要在**最终报告里**写"地质模块检索超时，请人工复核"，
    并基于井筒的数据继续生成。

"最终报告里"这四个字，纯函数测试证明不了 —— 从 ``degradation_block()``
返回一段文字，到它真的出现在 ``final_report`` 里，中间还有三步可能出错：
追加的时机（必须在正文之后）、被 ``strip_tool_json`` 清掉、被图表引用覆盖。
所以这里直接调用**真实的节点函数**，用桩替换掉模型和 IO，断言最终产物。

为什么用桩而不是打真接口
------------------------
这里要验证的是**控制流**（缺数据时走哪条路、走完之后产物长什么样），
不是模型质量。真接口会让这个测试变成一个要跑两分钟、结果还随模型波动的
东西；而控制流的分支必须在每次提交时都能被检查。

依赖说明：为什么这组测试不在 CI 里
----------------------------------
不是懒得装依赖，是**装不动**。真实的 import 链是：

    nodes.aggregate_node → core.config → utils.tokens → core.reranker
                                                          → sentence_transformers → torch
    nodes.aggregate_node → memory.memory_manager → chromadb

``utils/tokens.py`` 顶部为了一个 tokenizer 兜底就 import 了 reranker，于是
**任何节点模块**都会把整个模型栈拖进来。这不是本测试的问题，是依赖结构的
问题；为了让它进 CI 去改生产代码的 import 时机，会让当时那份 E2E 结果不再
覆盖 HEAD，代价大于收益。

所以这组测试与 ``test_graph_rag_e2e.py`` 同级：**本地/完整环境跑**。
CI 里跑的是 ``tests/test_branch_merge.py``（同一套降级逻辑的纯函数部分，
零依赖），两者合起来才是完整覆盖 —— 纯函数负责"文案对不对"，
本文件负责"它真的进了 final_report"。

跑法::

    python -m unittest -v tests.test_degradation
"""
from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

# 两组 import 分开写：任一失败只 skip 对应那组，而不是把整个文件拖下水。
try:
    from nodes.aggregate_node import aggregate_report

    HAVE_AGG, AGG_ERR = True, ""
except Exception as exc:  # noqa: BLE001 — import 期异常只意味着"环境不满足"
    HAVE_AGG, AGG_ERR = False, f"{type(exc).__name__}: {exc}"

try:
    from nodes.geology_node import geology_agent

    HAVE_GEO, GEO_ERR = True, ""
except Exception as exc:  # noqa: BLE001
    HAVE_GEO, GEO_ERR = False, f"{type(exc).__name__}: {exc}"


GEOLOGY_FALLBACK = "地质模块检索超时，请人工复核"
WELLBORE_FALLBACK = "井筒模块检索超时，请人工复核"

WELLBORE_TEXT = (
    "井筒风险评估：环空带压达 12 MPa，超过 API RP 90 一级阈值；"
    "套管在 80℃、CO2 分压 3.2 MPa 条件下腐蚀速率 0.12 mm/a，"
    "建议更换为 13Cr 材质并加装井下压力计。"
)
GEOLOGY_TEXT = (
    "地质风险评估：注入压力 250 bar 对应储层破裂压力 310 bar，"
    "安全系数 1.24；盖层突破压力 18 MPa，未见贯通性断层。"
)


class _Msg:
    def __init__(self, content: str):
        self.content = content


class _Choice:
    def __init__(self, content: str):
        self.message = _Msg(content)


class _Resp:
    """伪造一次非流式 chat 返回（只提供节点真正读到的字段）。"""

    def __init__(self, content: str = "", pt: int = 10, ct: int = 20):
        self.choices = [_Choice(content)]
        self.usage = SimpleNamespace(prompt_tokens=pt, completion_tokens=ct)


class _Streamed:
    """伪造一次流式终稿返回。"""

    def __init__(self, text: str, pt: int = 10, ct: int = 20):
        self.text = text
        self.prompt_tokens = pt
        self.completion_tokens = ct
        self.usage_estimated = False


def _state(**over) -> dict:
    """构造一份"扇出之后、进入 aggregate 之前"的状态。

    ``file_memory_context`` 显式给空串而不是缺省：缺省会触发历史压缩那次
    LLM 调用和两次文件 IO，跟本测试无关，还会让测试依赖 project_history.md
    的当前内容。
    """
    base = {
        "user_input": "某 CCUS 项目注入压力 250 bar，温度 80℃，深度 2500 m",
        "extracted_params": "注入压力 250 bar；温度 80℃；储层深度 2500 m；目标层砂岩",
        "feedback": "",
        "retry_count": 0,
        "complexity_mode": "cloud",
        "complexity_info": {},
        "file_memory_context": "",
        "personal_context": "",
    }
    base.update(over)
    return base


def _fake_tool(name: str, arguments: dict) -> dict:
    """替身 MCP 工具：在 _execute_tool 这个边界上打桩。

    工具实现现在通过 MCP 协议取（见 mcp_client_ccus.py），所以桩要打在
    "客户端 → 服务端"这一层，而不是打在某个具体函数上。这样测试既不需要
    torch，也不需要启动传输层，而被测的降级逻辑一行都没被绕过。
    """
    if name == "get_co2_density":
        return {"density_kg_m3": 0.785, "text": "0.785 g/cm3"}
    if name == "predict_pressure_pinn":
        return {"pressure_bar": 109.951, "x_m": arguments.get("x"),
                "t_days": arguments.get("t"), "scenario": "injection",
                "analytic_steady_state_bar": 110.0}
    if name == "query_ccus_graph":
        return {"fragments": [], "count": 0}
    return {}


def _run_aggregate(state: dict) -> tuple[dict, list[dict]]:
    """跑一次真实的 aggregate_report，返回 (节点输出, 终稿 prompt 快照)。

    patch 掉的几样东西都与本测试无关：模型、MCP 工具边界、绘图、长期记忆落库。
    绘图尤其要挡 —— 它会往磁盘写 PNG，让测试变成有副作用的。
    """
    captured: dict = {}

    def _fake_streamed(stage, messages, temperature=0.3, sink=None, mode=None):
        captured["messages"] = messages
        return _Streamed("报告正文：地质与井筒风险如下……")

    with patch("nodes.aggregate_node.llm_chat", return_value=_Resp("")), \
         patch("nodes.aggregate_node.llm_chat_streamed", side_effect=_fake_streamed), \
         patch("nodes.aggregate_node._execute_tool", side_effect=_fake_tool), \
         patch("nodes.aggregate_node.generate_pressure_curve",
               side_effect=RuntimeError("绘图在测试中被跳过")), \
         patch("nodes.aggregate_node.save_memory", return_value=None), \
         patch("nodes.aggregate_node.record_llm_usage", return_value={}), \
         patch("nodes.aggregate_node.record_usage_payload", return_value={}):
        out = aggregate_report(state, config=None)

    return out, captured.get("messages", [])


def _prompt_text(messages: list[dict]) -> str:
    return "\n".join(str(m.get("content", "")) for m in messages)


@unittest.skipUnless(HAVE_GEO, f"缺少 lightrag / sentence-transformers（{GEO_ERR}）")
class TestGeologyBranchFailure(unittest.TestCase):
    """分支层：异常必须变成状态，而不是冒到 LangGraph 里。"""

    def test_branch_returns_instead_of_raising(self):
        """检索超时时 geology_agent 正常返回，把错误写进 geology_status。"""
        with patch("nodes.geology_node.retrieve_knowledge",
                   side_effect=TimeoutError("LightRAG 检索超时 120s")):
            out = geology_agent(_state())

        self.assertIsInstance(out, dict)
        self.assertEqual(out["geology_report"], "", "失败分支不得产出结论文本")
        self.assertFalse(out["geology_status"]["ok"])
        self.assertIn("TimeoutError", out["geology_status"]["error"])
        self.assertIn("120s", out["geology_status"]["error"])
        self.assertIn("elapsed_s", out["geology_status"])
        self.assertGreaterEqual(out["geology_status"]["elapsed_s"], 0.0)

    def test_llm_failure_also_degrades(self):
        """检索成功但生成失败（限流重试耗尽）同样走降级，而不是抛异常。"""
        with patch("nodes.geology_node.retrieve_knowledge",
                   return_value={"retrieved_knowledge": "断层证据：F1 断层距注入井 800 m"}), \
             patch("nodes.geology_node.should_search", return_value="no_search"), \
             patch("nodes.geology_node.call_with_retry",
                   side_effect=RuntimeError("RateLimitError: 429 after 3 attempts")):
            out = geology_agent(_state())

        self.assertEqual(out["geology_report"], "")
        self.assertFalse(out["geology_status"]["ok"])
        self.assertIn("429", out["geology_status"]["error"])
        # 证据仍然被保留：失败的是生成，不是检索
        self.assertIn("F1 断层", out["geology_evidence"])


@unittest.skipUnless(HAVE_AGG, f"缺少 openai / matplotlib（{AGG_ERR}）")
class TestAggregateDegradation(unittest.TestCase):
    """汇总层：缺分支时最终报告长什么样。"""

    def test_geology_dead_report_contains_exact_wording(self):
        """geology 挂了 → final_report 里逐字出现降级文案，且不抛异常。"""
        out, messages = _run_aggregate(_state(
            geology_report="",
            geology_status={
                "ok": False,
                "error": "TimeoutError: LightRAG 检索超时 120s",
                "elapsed_s": 120.0,
                "started_at": 1000.0,
                "ended_at": 1120.0,
            },
            wellbore_report=WELLBORE_TEXT,
            wellbore_status={"ok": True, "elapsed_s": 18.0,
                             "started_at": 1000.0, "ended_at": 1018.0},
        ))

        report = out["final_report"]
        self.assertIn(GEOLOGY_FALLBACK, report,
                      "最终报告里必须逐字出现「地质模块检索超时，请人工复核」")
        # 这条要求不能只是一句孤立的话：必须说明"没写 ≠ 没风险"
        self.assertIn("数据完整性提示", report)
        self.assertIn("不代表该维度无风险", report)
        # 原因要带出来，否则人工复核的人不知道去哪儿查
        self.assertIn("TimeoutError", report)
        # 井筒那条不能也被标成缺失
        self.assertNotIn(WELLBORE_FALLBACK, report)

        self.assertTrue(out["branch_merge"]["degraded"])
        self.assertEqual(out["branch_merge"]["missing"], ["geology"])
        self.assertTrue(out["branch_merge"]["wellbore_ok"])

    def test_wellbore_data_still_reaches_generation(self):
        """"基于井筒的数据继续生成"要能被验证：井筒结论必须进终稿 prompt。"""
        out, messages = _run_aggregate(_state(
            geology_report="",
            geology_status={"ok": False, "error": "TimeoutError: 120s"},
            wellbore_report=WELLBORE_TEXT,
            wellbore_status={"ok": True, "elapsed_s": 18.0},
        ))

        prompt = _prompt_text(messages)
        self.assertIn("环空带压达 12 MPa", prompt, "井筒结论没有进终稿 prompt")
        self.assertIn("13Cr", prompt)
        # 地质侧交给模型的是"降级文案"，不是空串 —— 空串会被模型读成"没这条要求"
        self.assertIn(GEOLOGY_FALLBACK, prompt)
        # 并且要明确告诉模型结论不完整
        self.assertIn("数据完整性警告", prompt)
        # 报告本身照常产出
        self.assertGreater(len(out["final_report"]), 0)

    def test_both_branches_dead_still_produces_report(self):
        """两条分支全挂：仍要出报告（含数值兜底），不能崩。"""
        out, _ = _run_aggregate(_state(
            geology_report="",
            geology_status={"ok": False, "error": "TimeoutError"},
            wellbore_report="",
            wellbore_status={"ok": False, "error": "ChromaError: collection empty"},
        ))

        report = out["final_report"]
        self.assertIn(GEOLOGY_FALLBACK, report)
        self.assertIn(WELLBORE_FALLBACK, report)
        self.assertEqual(sorted(out["branch_merge"]["missing"]), ["geology", "wellbore"])
        # 密度来自物理工具，与分支结论无关 —— 全挂也必须还在
        self.assertIn("0.785", report, "数值兜底失效：密度值没有进报告")

    def test_healthy_run_has_no_degradation_text(self):
        """两条都成功时不得出现降级文案（否则等于天天误报"需人工复核"）。"""
        out, _ = _run_aggregate(_state(
            geology_report=GEOLOGY_TEXT,
            geology_status={"ok": True, "elapsed_s": 70.0,
                            "started_at": 1000.0, "ended_at": 1070.0},
            wellbore_report=WELLBORE_TEXT,
            wellbore_status={"ok": True, "elapsed_s": 18.0,
                             "started_at": 1000.0, "ended_at": 1018.0},
        ))

        report = out["final_report"]
        self.assertNotIn(GEOLOGY_FALLBACK, report)
        self.assertNotIn(WELLBORE_FALLBACK, report)
        self.assertNotIn("数据完整性提示", report)
        self.assertFalse(out["branch_merge"]["degraded"])
        self.assertEqual(out["branch_merge"]["missing"], [])

    def test_parallel_timing_recorded_in_output(self):
        """耗时是实测的：两条分支时间戳重叠 18s，墙钟应远小于串行和。"""
        out, _ = _run_aggregate(_state(
            geology_report=GEOLOGY_TEXT,
            geology_status={"ok": True, "elapsed_s": 70.0,
                            "started_at": 1000.0, "ended_at": 1070.0},
            wellbore_report=WELLBORE_TEXT,
            wellbore_status={"ok": True, "elapsed_s": 18.0,
                             "started_at": 1000.0, "ended_at": 1018.0},
        ))

        t = out["parallel_timing"]
        self.assertAlmostEqual(t["wall_s"], 70.0, places=1)   # 1070 - 1000
        self.assertAlmostEqual(t["serial_s"], 88.0, places=1)  # 70 + 18
        self.assertGreater(t["speedup"], 1.0)
        self.assertAlmostEqual(t["overlap_s"], 18.0, places=1)
        self.assertIn("aggregate_s", t)

    def test_whitespace_only_report_counts_as_missing(self):
        """只有空白的"结论"等于没结论 —— 否则报告里会出现一个空段落。"""
        out, _ = _run_aggregate(_state(
            geology_report="   \n\t  ",
            geology_status={"ok": True, "elapsed_s": 5.0},
            wellbore_report=WELLBORE_TEXT,
            wellbore_status={"ok": True, "elapsed_s": 5.0},
        ))

        self.assertEqual(out["branch_merge"]["missing"], ["geology"])
        self.assertIn(GEOLOGY_FALLBACK, out["final_report"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
