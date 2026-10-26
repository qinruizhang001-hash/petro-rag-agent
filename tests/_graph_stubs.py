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

import importlib
import sys
import types

# 已被替换的模块名（供测试断言"确实没加载重依赖"）
STUBBED: list[str] = []

# 被 ``release()`` 还回去的模块名。与 ``STUBBED`` 分开记，是为了让
# "装过哪些桩"这个历史事实保持可查，同时不让人误以为它还生效。
RELEASED: list[str] = []


def _install(name: str, **attrs) -> None:
    """
    把 ``name`` 换成只有 ``attrs`` 的假模块。已被真实导入过则**跳过**。

    那个 ``return`` 是"本地全绿、CI 必红"的来源，值得写清楚：本地跑
    ``unittest discover`` 时，``tests/test_degradation.py``（字母序在
    ``test_quality_gate`` 之前）先一步 import 了真实的 ``tools/physics_tools.py``，
    于是这里跳过，图测试用的是真实模块 —— 替身少了名字也看不出来
    （实测：discover 顺序下 ``sys.modules['tools.physics_tools'].__file__``
    指向真实文件）。CI 的 graph job 只跑 ``test_quality_gate`` +
    ``test_negative_gate``，替身真的生效，缺名字当场 ImportError。
    所以替身的完整性不能靠"本地跑一遍"来保证，
    见 ``tests/test_graph_stubs.py``（静态解析，不依赖导入顺序）。
    """
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
        # 替身是**契约**，不是"够用就行"：`aggregate_node` 从这个模块 import
        # 哪几个名字，这里就必须有哪几个，否则真实模块里的新增导入会让
        # `workflow.py` 在 import 阶段就崩，而报错信息是
        # "cannot import name X from 'tools.physics_tools' (unknown location)" ——
        # "unknown location" 说的就是替身没有 __file__，很容易被误读成打包问题。
        injection_risk_sensitivity=_fake_sensitivity,
        # 注入策略优化同理：aggregate 会确定性调用它（闭环的唯一环节），
        # 替身缺这个属性时整图 import 就崩，而报错是 "(unknown location)"。
        optimize_injection_strategy=_fake_optimize,
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
        # 返回值必须是 dict，不能是 None：``nodes/history_node.py`` 会读
        # ``evo.get("recorded")``。桩返回 None 时，只要这个桩**真的生效**
        # （也就是 evolution_engine 还没被真实导入过），整图测试就会在
        # save_file_memory 上崩 —— 而它在原来的导入顺序下一直是"碰巧没事"。
        record_experience=lambda *_a, **_k: {"recorded": False},
    )
    return list(STUBBED)


def release(name: str) -> bool:
    """
    把 ``name`` 从替身还原成**真实模块**。返回是否真的做了还原。

    为什么需要它：``install_leaf_stubs()`` 是在 ``test_quality_gate`` /
    ``test_negative_gate`` 的**模块顶层**调用的（桩必须早于 ``import workflow``，
    见上），而 ``unittest discover`` 是"先导入全部测试模块，再跑"。于是桩的
    存活期覆盖了所有测试模块的导入期 —— 任何在顶层写
    ``from tools.physics_tools import ...`` 的模块，只要加载顺序排在这两个之后，
    拿到的就是替身里的假实现。

    这个缺陷已经真实发生过：手工按 ``tests.test_negative_gate`` →
    ``tests.test_optimize`` 的顺序跑时，``injection_risk_uq`` 是
    ``_fake_injection_uq``，它返回的 dict 没有 ``counts``，于是 4 项
    ``KeyError: 'counts'``。全量 discover 不复现，只是因为
    ``test_optimize`` 恰好先被加载 —— 那是**碰巧**，不是设计。

    所以需要真实实现的测试模块要显式调用 ``release(...)``，
    把"我依赖真实实现"写在代码里，而不是依赖模块加载顺序。
    """
    if name not in STUBBED:
        return False
    # 必须先 pop 再 import，否则 importlib 会把替身原样还回来。
    sys.modules.pop(name, None)
    STUBBED.remove(name)
    RELEASED.append(name)
    importlib.import_module(name)
    return True


# ─── 具体替身 ────────────────────────────────────────────
def _no_chart(*_args, **_kwargs) -> dict:
    """图表工具：本测试不关心出图，直接报"不可用"。"""
    return {"status": "unavailable", "error": "chart stub"}


def _no_documents(*_args, **_kwargs) -> list:
    return []


def _fake_density_uq(pressure_bar, temperature_celsius, **_kwargs) -> dict:
    from utils.co2_properties import co2_density_kg_m3

    mean = co2_density_kg_m3(pressure_bar, temperature_celsius)
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
        "dist_label": "±5% 均匀扰动",
        "independent": True,
    }


def _fake_sensitivity(depth_m, pressure_bar, **_kwargs) -> dict:
    """
    Sobol 替身：数字是编的，结构是真的。

    这里只需要"有排名、有 text"，因为被测的是 aggregate 会不会把它写进
    tool_facts 与 Prompt —— 指数算得准不准由 tests/test_sensitivity.py 用
    解析基准负责，不需要在整图测试里重跑一遍 8192 次模型评估。
    """
    return {
        "text": "深度 一阶 60%／总效应 65%；注入压力 一阶 35%／总效应 40%",
        "names": ["深度", "注入压力"],
        "first_order": {"深度": 0.60, "注入压力": 0.35},
        "total_order": {"深度": 0.65, "注入压力": 0.40},
        "interaction": {"深度": 0.05, "注入压力": 0.05},
        "first_order_se": {"深度": 0.01, "注入压力": 0.01},
        "total_order_se": {"深度": 0.01, "注入压力": 0.01},
        "ranking": ["深度", "注入压力"],
        "n_base": 2048,
        "evaluations": 8192,
        "variance": 1.0,
        "mean": 0.2,
        "notes": [],
    }


def _fake_optimize(depth_m, pressure_bar=None, **_kwargs) -> dict:
    """
    注入策略优化替身：结构真、数字假。

    被测的是"aggregate 会不会把推荐方案写进 tool_facts 与 Prompt"，
    不是"优化器解得对不对"—— 后者由 tests/test_optimize.py 用可回代的
    等式（q = n·J·(p−P_h)）与解析基准负责。在这里跑 1000 次抽样只会让
    整图测试慢一个量级，还测不到任何新东西。
    """
    p_h = 0.101 * float(depth_m)
    return {
        "status": "ok",
        "depth_m": float(depth_m),
        "n": 1000,
        "rel_sigma": 0.05,
        "dist": "uniform",
        "dist_label": "±5% 均匀扰动",
        "corr": [[1.0, 0.0], [0.0, 1.0]],
        "independent": True,
        "p_target": 0.05,
        "years": 20.0,
        "binding": "risk",
        "text": (
            f"当前井底压力 {pressure_bar} bar 对应 1.00 Mt/yr，"
            "未观测到压裂（0/1000 次，95% 上界 0.4%）；"
            "推荐 1 口井、井底压力 280 bar，速率 1.00 Mt/yr"
        ),
        "notes": ["注入指数按无表皮计算（skin=0），这是乐观假设"],
        "provenance": {"fracture_ratio": "1.5× 静水压力", "constraint": "Wilson 上界"},
        "physics": {
            "hydrostatic_bar": p_h,
            "fracture_bar": 1.5 * p_h,
            "cap_bar": 1.5 * p_h,
            "injectivity_m3_per_day_per_bar": 500.0,
            "density_kg_m3": 700.0,
        },
        "current": {
            "pressure_bar": pressure_bar,
            "n_wells": 1,
            "rate_m3_per_day": 1000.0,
            "rate_mt_per_year": 1.0,
            "verdict": "ok",
            "risk": {"hits": 0, "n": 1000, "p": 0.0, "p_ci": [0.0, 0.004],
                     "p_upper": 0.004},
        },
        "limit": {"pressure_bar": 280.0, "k_max": 36, "hits": 36,
                  "p_upper": 0.049, "binding": "risk", "feasible": True},
        "well_options": [],
        "recommended": {
            "n_wells": 1, "pressure_bar": 280.0, "rate_m3_per_day": 1000.0,
            "rate_mt_per_year": 1.0, "per_well_mt_per_year": 1.0,
            "binding": "risk",
            "risk": {"hits": 36, "n": 1000, "p": 0.036, "p_ci": [0.026, 0.049],
                     "p_upper": 0.049},
        },
        "headroom": {"fraction": 0.0, "rate_mt_per_year": 0.0},
        "verification": {"hits": 40, "n": 1000, "p": 0.04,
                         "p_ci": [0.029, 0.054], "p_upper": 0.054,
                         "seed": 7920, "agrees": True},
        "frontier": [],
    }


class _FakeMCPClient:
    """MCP 客户端：没有工具，也不发网络请求。"""

    def list_tools(self) -> list:
        return []

    def call(self, name, arguments):  # noqa: ARG002
        return {}


def _fake_mcp_client() -> _FakeMCPClient:
    return _FakeMCPClient()
