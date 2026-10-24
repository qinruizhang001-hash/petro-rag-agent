"""
轻量桩：让「真实图 + 真实闸门」在没有 Ollama / 向量库 / torch 的机器上跑起来
================================================================================
``tests/test_quality_gate.py`` 与 ``tests/test_negative_gate.py`` 要验证的是
**判定逻辑与图拓扑**，不是检索质量。但这两件事被重依赖绑住了：

``nodes/evaluate_node.py`` → ``utils/tokens.py`` → ``core/reranker.py``
→ ``sentence_transformers`` → **torch（2 GB）**

也就是说，只要 import 评审节点，就自动背上一份 2.3 GB 的 reranker。CI 的
``unit-core`` 一个三方包都不装，于是这类测试只能靠人工跑 —— 而"只在人工环境
里跑过的闸门"正是上一版把"重试用尽"写成"放行"却没被发现的原因。

所以这里只替换**叶子**（模型加载、向量库、MCP 传输、图表渲染），
``nodes/`` 下的判定代码、``utils/physics_rules.py``、``core/gates.py``
以及 ``workflow.py`` 的节点与边全部是真实的。

导入顺序很重要：本模块必须在 ``import nodes.*`` / ``import workflow`` **之前**
被导入，否则真实模块已经把 torch 拉起来了。
"""
from __future__ import annotations

import sys
import types

# 已被替换的模块名（供测试断言"确实没加载重依赖"）
STUBBED: list[str] = []


def _install(name: str, **attrs) -> None:
    if name in sys.modules:
        return
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    sys.modules[name] = module
    STUBBED.append(name)


class _FakeReranker:
    """``core.reranker.reranker`` 的最小替身：只提供被用到的两个属性。"""

    tokenizer = None

    def predict(self, pairs, **_kwargs):
        # 分数不重要（检索在本测试里是桩），但必须与入参等长 —— 调用方会 zip。
        return [0.0 for _ in pairs]


def install_leaf_stubs() -> list[str]:
    """替换所有会拉起重依赖的**叶子**模块。可重复调用（幂等）。"""
    _install("core.reranker", reranker=_FakeReranker())

    _install("chart_tools", generate_pressure_curve=_no_chart)
    _install("lightrag_retriever", retrieve_documents=_no_documents)
    _install("chroma_retriever", retrieve_documents=_no_documents)

    _install(
        "tools.physics_tools",
        co2_density_uq=_fake_density_uq,
        injection_risk_uq=_fake_injection_uq,
    )
    _install(
        "tools.knowledge_tools",
        is_tavily_configured=lambda: False,
        search_and_save=lambda *_a, **_k: {"status": "disabled"},
    )

    _install("memory.memory_manager", save_memory=lambda **_k: None)
    _install("memory.memory_tools", search_personal_memory=lambda *_a, **_k: [])

    _install("mcp_client_ccus", get_client=_fake_mcp_client)

    _install(
        "evolution_engine",
        query_system_experience=lambda *_a, **_k: "",
        record_experience=lambda *_a, **_k: None,
    )
    return list(STUBBED)


# ─── 具体替身 ────────────────────────────────────────────
def _no_chart(*_args, **_kwargs) -> dict:
    """图表工具：本测试不关心出图，直接报"不可用"。"""
    return {"status": "unavailable", "error": "chart stub"}


def _no_documents(*_args, **_kwargs) -> list:
    return []


def _fake_density_uq(pressure_bar, temperature_celsius, **_kwargs) -> dict:
    mean = 1.9 * float(pressure_bar) / (float(temperature_celsius) + 273.15)
    return {
        "mean": mean,
        "p5": mean * 0.97,
        "p95": mean * 1.03,
        "std": mean * 0.02,
        "n": 100,
        "rel_sigma": 0.05,
        "unit": "kg/m³",
        "text": f"CO2 密度 {mean:.4f} kg/m³（90% 区间 {mean*0.97:.4f}–{mean*1.03:.4f}）",
        "samples": [mean] * 10,
    }


def _fake_injection_uq(depth_m, pressure_bar, **_kwargs) -> dict:
    return {
        "text": "注入可行 70%，压裂风险 20%，压力不足 10%",
        "probabilities": {"可行": 0.7, "压裂": 0.2, "不足": 0.1},
        "verdict": "feasible",
        "verdict_label": "注入可行",
        "depth_m": depth_m,
        "injection_pressure_bar": pressure_bar,
    }


class _FakeMCPClient:
    """MCP 客户端：没有工具，也不发网络请求。"""

    def list_tools(self) -> list:
        return []

    def call(self, name, arguments):  # noqa: ARG002
        return {}


def _fake_mcp_client() -> _FakeMCPClient:
    return _FakeMCPClient()
