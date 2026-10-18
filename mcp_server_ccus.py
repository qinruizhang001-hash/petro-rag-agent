"""
CCUS 工具 MCP Server
====================
把系统里三个"算出来的事实"封装成标准 MCP（Model Context Protocol）工具，
让 Cursor / Claude Desktop 这类外部客户端也能用自然语言调用它们::

    get_co2_density(pressure_bar, temperature_celsius)
    predict_pressure_pinn(x, t)
    query_ccus_graph(query)

设计约束（三条，都是踩过才知道的）
----------------------------------
1. **stdout 必须干净。** stdio 传输把 JSON-RPC 消息直接写在 stdout 上，
   任何一句 ``print()`` 或往 stdout 的日志都会让客户端解析失败，且报错信息
   通常只说"无效的 JSON"，很难定位。本项目的 ``utils/logging_setup`` 只往
   stderr 和 app.log 写，所以这里可以放心 import 业务模块。

2. **重依赖必须惰性导入。** ``list_tools`` 在客户端连接时就调用，如果模块
   顶层 import torch（1~2 s）和 lightrag（连带 sentence-transformers，
   再加 2.3 GB 的 reranker），那么 Cursor 每次启动 MCP 都要等十几秒，
   而且 torch 根本不该为"列出工具名"付钱。所以三个工具的函数体内部才 import。

3. **本进程可能加载第二份模型。** ``query_ccus_graph`` 会初始化 LightRAG，
   而 LightRAG 的 reranker 常驻约 2.3 GB。作为 stdio 子进程被 Cursor 拉起时
   没问题（此时没有别的进程占着模型）；但如果主系统（api_server）已经加载
   了一份，再 spawn 一个带模型的子进程会被系统杀掉 —— 这也是主系统那边改用
   同进程内存传输、而不是 stdio 子进程的原因（见 mcp_client_ccus.py）。

为什么 ``query_ccus_graph`` 只返回证据片段、不做 LLM 生成
---------------------------------------------------------
调用方（Cursor 里的模型、或主系统的 aggregate 节点）自己就有 LLM。在这里
再生成一遍是双重生成：既浪费 token，又把 MCP Server 变成一个必须有 Ollama
才能用的东西。工具负责"取事实"，生成留给调用方。

运行::

    python mcp_server_ccus.py          # stdio 传输，供 MCP 客户端连接

自检（不经过 MCP，直接调函数）::

    python mcp_server_ccus.py --selftest
"""
from __future__ import annotations

import sys

from mcp.server.mcpserver import MCPServer

SERVER_NAME = "ccus-tools"
SERVER_VERSION = "1.0.0"

server = MCPServer(
    name=SERVER_NAME,
    version=SERVER_VERSION,
    instructions=(
        "CCUS（碳捕集利用与封存）数值与检索工具。三个工具都返回由代码计算或"
        "从知识图谱检索得到的事实，而不是模型生成的估计值："
        "get_co2_density 用状态方程算 CO2 密度；"
        "predict_pressure_pinn 用一维扩散方程的物理信息神经网络算地层压力；"
        "query_ccus_graph 从 22 篇 CCUS 文献构建的知识图谱里检索证据片段。"
        "涉及数值时优先调用工具，不要自行推算。"
    ),
)


@server.tool(
    name="get_co2_density",
    description=(
        "计算给定注入压力与温度下 CO2 的密度（kg/m³），使用简化状态方程 "
        "rho = 1.9 * P / T(K)。这是确定性计算，不要用模型记忆的密度值替代。"
        "典型值：250 bar / 80℃ → 约 1.35 kg/m³。"
    ),
)
def get_co2_density(pressure_bar: float, temperature_celsius: float) -> dict:
    """CO2 密度（状态方程，非 Span-Wagner）。"""
    from tools.physics_tools import calculate_co2_density

    text = calculate_co2_density(pressure_bar, temperature_celsius)
    # 底层返回 "1.35 kg/m³"，这里拆出数值，让调用方不必解析字符串
    try:
        value = float(str(text).split()[0])
    except (ValueError, IndexError):
        value = None
    return {
        "density_kg_m3": value,
        "text": str(text),
        "formula": "rho = 1.9 * P[bar] / T[K]",
        "pressure_bar": float(pressure_bar),
        "temperature_celsius": float(temperature_celsius),
        "note": "简化状态方程，用于演示工具调用契约，不是工程精度模型",
    }


@server.tool(
    name="predict_pressure_pinn",
    description=(
        "用一维扩散方程 dP/dt = D·d²P/dx²（D = 0.1 m²/s）的物理信息神经网络，"
        "预测距井口 x 米、注入 t 天后的地层压力（bar）。"
        "适用域：x ∈ [0, 100] m，t ∈ [0, 30] 天；越界输入会被裁剪到边界。"
        "scenario='injection'（默认）为井口 120 bar、远端 100 bar、初值 100 bar，"
        "是非平凡解；scenario='baseline' 是初值与两端恒为 100 bar 的退化算例，"
        "其解析解恒为 100 bar（保留用于对照）。"
    ),
)
def predict_pressure_pinn(x: float, t: float, scenario: str = "injection") -> dict:
    """PINN 压力预测（单位：x 米，t 天，返回 bar）。"""
    from pinn_model import predict_pressure, steady_state_pressure

    p = predict_pressure(x, t, scenario=scenario)
    out = {
        "pressure_bar": p,
        "x_m": float(x),
        "t_days": float(t),
        "scenario": scenario,
        "model": "1-D PINN, 3-layer MLP (2-64-64-1), tanh",
        "pde": "dP/dt = D * d2P/dx2, D = 0.1 m^2/s",
    }
    if scenario == "injection":
        # 附上解析稳态解，让调用方能判断这个数离"物理上应该的答案"有多远。
        # t=10 天已接近稳态，两者应相差不到 0.1 bar（实测 0.049）。
        out["analytic_steady_state_bar"] = round(steady_state_pressure(float(x)), 3)
    return out


@server.tool(
    name="query_ccus_graph",
    description=(
        "从 CCUS 知识图谱（22 篇文献构建，389 实体 / 400 关系）检索与问题相关的"
        "证据片段，返回原文段落。适合问'断层活化与注入压力有什么关系'这类需要"
        "跨文献证据的问题。只检索、不生成结论 —— 请基于返回片段自行作答。"
        "注意：首次调用会加载本地模型，可能耗时数十秒。"
    ),
)
def query_ccus_graph(query: str, top_k: int = 6) -> dict:
    """LightRAG 图谱检索（返回证据片段，不做 LLM 生成）。"""
    import core  # noqa: F401  必须先于 lightrag：HF_HUB_OFFLINE 与 sys.path 预置
    from lightrag_retriever import retrieve_documents

    docs = retrieve_documents(query)
    docs = [d for d in docs if d and d.strip()]
    return {
        "query": query,
        "count": len(docs),
        "fragments": docs[: max(1, int(top_k))],
        "engine": "lightrag (hybrid mode, rerank disabled inside lightrag)",
        "note": "原始证据片段，未经模型改写",
    }


def _selftest() -> int:
    """不经过 MCP 协议，直接调三个函数 —— 用来区分"工具坏了"和"传输坏了"。"""
    print("=== get_co2_density ===")
    print(get_co2_density(250, 80))

    print("\n=== predict_pressure_pinn ===")
    print(predict_pressure_pinn(50, 10))
    print(predict_pressure_pinn(50, 10, "baseline"))

    print("\n=== query_ccus_graph ===")
    try:
        r = query_ccus_graph("断层活化与注入压力的关系", top_k=2)
        print(f"count={r['count']}")
        for i, frag in enumerate(r["fragments"], 1):
            print(f"  [{i}] {frag[:160]}...")
    except Exception as exc:  # noqa: BLE001 — 图谱可能没建好，自检不该因此中断
        print(f"检索失败：{type(exc).__name__}: {exc}")
        return 1
    return 0


def main() -> int:
    if "--selftest" in sys.argv:
        return _selftest()
    # stdio：由客户端（Cursor / Claude Desktop / mcp_client_ccus）拉起
    server.run("stdio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
