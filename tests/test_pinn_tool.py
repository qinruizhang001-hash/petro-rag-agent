"""
PINN 工具的接线测试（模型 + 工具规划 + 报告兜底）
================================================
需求要的是"Agent 能够调用这个模型进行计算并输出"。"调用"和"输出"是两件事，
而且实测会分叉：本地 7B 确实规划出了 ``predict_pressure``、也确实算出了
109.95 bar，但**最终报告里一个字都没提**——它忽略了 facts_block 里
"必须写进正文"的指令。

所以这个文件测三段链路，缺一段都算没跑通：

  ① 模型规划出工具 → 工具被执行 → ``pinn_prediction`` 有值
  ② 模型没规划 → 由 ``utils/pinn_query`` 兜底触发（否则问句一换就不灵）
  ③ 报告漏写 → 代码把数值补进去（否则"输出了"是假的）

依赖：torch（模型）+ openai/matplotlib（aggregate 节点）。不进 CI 的
unit-core job；零依赖的那部分（正则抽取）在 ``tests/test_pinn_query.py``。
"""
from __future__ import annotations

import unittest
from unittest.mock import patch

try:
    from nodes.aggregate_node import aggregate_report

    from tests.test_degradation import _Resp, _Streamed, _state

    HAVE_DEPS = True
    ERR = ""
except Exception as exc:  # noqa: BLE001
    HAVE_DEPS = False
    ERR = f"{type(exc).__name__}: {exc}"

QUESTION = "预测距离井口 50 米处，注入 10 天后的压力"
PLAN_WITH_TOOL = '```json\n{"tool": "predict_pressure", "x": 50, "t": 10}\n```'
EXPECTED = 109.951          # injection 场景 x=50 m, t=10 d（解析解 110.000）


def _run(state: dict, plan: str, report_text: str = "报告正文：地质与井筒风险如下。") -> dict:
    """跑一次真实的 aggregate_report，模型输出与图表用桩替换。"""

    def _fake_streamed(stage, messages, temperature=0.3, sink=None, mode=None):
        return _Streamed(report_text)

    with patch("nodes.aggregate_node.llm_chat", return_value=_Resp(plan)), \
         patch("nodes.aggregate_node.llm_chat_streamed", side_effect=_fake_streamed), \
         patch("nodes.aggregate_node.calculate_co2_density", return_value="0.785 g/cm3"), \
         patch("nodes.aggregate_node.generate_pressure_curve",
               side_effect=RuntimeError("绘图在测试中跳过")), \
         patch("nodes.aggregate_node.save_memory", return_value=None), \
         patch("nodes.aggregate_node.record_llm_usage", return_value={}), \
         patch("nodes.aggregate_node.record_usage_payload", return_value={}):
        return aggregate_report(state, config=None)


@unittest.skipUnless(HAVE_DEPS, f"缺少运行时依赖（{ERR}）")
class TestToolWiring(unittest.TestCase):

    def test_model_planned_call_is_executed(self):
        """① 模型规划出工具时，工具必须真的被执行并写入状态。"""
        out = _run(_state(user_input=QUESTION), PLAN_WITH_TOOL)

        pinn = out["pinn_prediction"]
        self.assertTrue(pinn, "pinn_prediction 为空：工具没有被执行")
        self.assertAlmostEqual(pinn["pressure_bar"], EXPECTED, delta=0.1)
        self.assertEqual((pinn["x_m"], pinn["t_days"]), (50.0, 10.0))
        self.assertEqual(pinn["scenario"], "injection")

    def test_report_missing_value_is_appended_by_code(self):
        """③ 报告没写这个数时，代码必须补 —— 否则"调用成功"对读者不可见。

        实测就是这样：模型规划了工具、算出 109.95，正文里却只字未提。
        """
        out = _run(_state(user_input=QUESTION), PLAN_WITH_TOOL,
                   report_text="报告正文：地质风险与井筒风险如下。")

        self.assertIn("109.95", out["final_report"], "工具算出的压力没有出现在报告里")
        self.assertIn("PINN", out["final_report"])

    def test_no_duplicate_when_model_did_write_it(self):
        """模型已经写了这个数，就不该再追加一遍（重复数值看起来像 bug）。"""
        out = _run(_state(user_input=QUESTION), PLAN_WITH_TOOL,
                   report_text="报告正文：距井口 50 m、注入 10 天后压力 109.95 bar。")

        self.assertEqual(out["final_report"].count("109.95"), 1)

    def test_fallback_fires_when_model_plans_nothing(self):
        """② 模型没规划时，问句里的 x/t 仍要触发工具。

        这是"问句一换就不灵"的保险：工具规划依赖 7B 模型，而它对上下文长度
        很敏感，同一句话换个说法就可能不再规划。
        """
        out = _run(_state(user_input=QUESTION), plan="")  # 规划阶段返回空

        self.assertTrue(out["pinn_prediction"], "模型未规划，兜底也没有触发")
        self.assertAlmostEqual(out["pinn_prediction"]["pressure_bar"], EXPECTED, delta=0.1)
        self.assertIn("109.95", out["final_report"])

    def test_no_pinn_call_for_unrelated_question(self):
        """不相关的问题不该凭空多出一个压力值。"""
        out = _run(_state(user_input="某CCUS项目注入压力250bar，温度80℃，请评估风险。"), plan="")

        self.assertEqual(out["pinn_prediction"], {})
        self.assertNotIn("PINN", out["final_report"])

    def test_planned_coordinates_must_be_grounded_in_the_question(self):
        """模型规划了 PINN，但问句里没有 x/t → 必须忽略，不能用它给的坐标。

        这条防的是一个很隐蔽的污染：plan_prompt 里有 ``{"x": 50, "t": 10}``
        的示例，7B 模型会把它照抄到一个根本没问距离/时间的问题里。结果报告
        多出一句"距井口 50 m = 109.95 bar"——数值精确、用户没问、极难发现。
        """
        out = _run(
            _state(user_input="某CCUS项目注入压力250bar，温度80℃，请评估风险。"),
            PLAN_WITH_TOOL,   # 模型硬塞了一条 PINN 指令
        )

        self.assertEqual(out["pinn_prediction"], {}, "模型编造的坐标被采用了")
        self.assertNotIn("109.95", out["final_report"])

    def test_failure_degrades_instead_of_crashing(self):
        """权重缺失/模型损坏时，报告照出，只是没有 PINN 数值。"""
        with patch("nodes.aggregate_node._call_pinn",
                   side_effect=FileNotFoundError("缺少权重")):
            out = _run(_state(user_input=QUESTION), PLAN_WITH_TOOL)

        self.assertEqual(out["pinn_prediction"], {})
        self.assertGreater(len(out["final_report"]), 0, "PINN 失败不该拖垮报告")


if __name__ == "__main__":
    unittest.main(verbosity=2)
