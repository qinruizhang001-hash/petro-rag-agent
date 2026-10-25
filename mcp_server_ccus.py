"""
CCUS 工具 MCP Server
====================
把系统里三个"算出来的事实"封装成标准 MCP（Model Context Protocol）工具，
让 Cursor / Claude Desktop 这类外部客户端也能用自然语言调用它们::

    get_co2_density(pressure_bar, temperature_celsius)
    assess_injection_uncertainty(depth_m, injection_pressure_bar)
    get_fault_stability(depth_m, injection_pressure_bar, dip_deg)
    get_caprock_integrity(depth_m, injection_pressure_bar, ...)
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
        "CCUS（碳捕集利用与封存）数值与检索工具。所有工具都返回由代码计算或"
        "从知识图谱检索得到的事实，而不是模型生成的估计值："
        "get_co2_density 用 Peng-Robinson 状态方程算 CO2 密度；"
        "assess_injection_uncertainty 用蒙特卡洛算注入可行性的概率分布；"
        "get_fault_stability 用地应力判据算断层会不会活化；"
        "get_caprock_integrity 算盖层突破压力与可封 CO2 柱高；"
        "predict_pressure_pinn 用一维扩散方程的物理信息神经网络算地层压力；"
        "query_ccus_graph 从 22 篇 CCUS 文献构建的知识图谱里检索证据片段。"
        "涉及数值时优先调用工具，不要自行推算。"
        "注意：地应力与孔喉参数是**假设值**，返回结果里会一并给出，"
        "引用数值时必须同时说明它依赖的假设。"
    ),
)


@server.tool(
    name="get_co2_density",
    description=(
        "计算给定注入压力与温度下 CO2 的密度（kg/m³），使用 Peng-Robinson "
        "状态方程（CO2 临界参数取 Span & Wagner 1996），并对压力输入做 100 次"
        "蒙特卡洛传播，返回均值与 90% 置信区间（p5~p95）。"
        "这是确定性计算 + 统计传播，不要用模型记忆的密度值替代。"
        "典型值：250 bar / 80℃ → 约 660 kg/m³（超临界）。"
    ),
)
def get_co2_density(
    pressure_bar: float,
    temperature_celsius: float,
    uncertainty: float = 0.05,
    n_simulations: int = 100,
) -> dict:
    """CO2 密度（状态方程 + 蒙特卡洛，非 Span-Wagner）。"""
    from tools.physics_tools import co2_density_uq

    summary = co2_density_uq(
        pressure_bar, temperature_celsius,
        rel_sigma=uncertainty, n=n_simulations, keep_samples=True,
    )
    return {
        # 保留标量字段名，老调用方（float(text.split()[0]) 或读 density_kg_m3）不会坏
        "density_kg_m3": round(summary["mean"], 4),
        "mean_kg_m3": round(summary["mean"], 4),
        "ci90_low_kg_m3": round(summary["p5"], 4),
        "ci90_high_kg_m3": round(summary["p95"], 4),
        "std_kg_m3": round(summary["std"], 4),
        "n_simulations": summary["n"],
        "input_uncertainty": summary["rel_sigma"],
        "ci_level": summary["ci_level"],
        # 100 个样本随结果一起返回：前端画分布直方图要用，重算会与
        # 报告里的区间不一致（两次抽样不是同一批随机数）
        "samples": summary.get("samples", []),
        "text": summary["text"],
        "formula": "Peng-Robinson EOS（见 utils/co2_properties.py）",
        "pressure_bar": float(pressure_bar),
        "temperature_celsius": float(temperature_celsius),
        "note": "Peng-Robinson 状态方程 + 100 次蒙特卡洛传播；"
                "临界点附近误差可达 10–15%（Zc 固定 0.307），远离临界点通常 <5%",
    }


@server.tool(
    name="assess_injection_uncertainty",
    description=(
        "给定储层埋深与注入压力，用 100 次蒙特卡洛（深度与压力各 ±5% 均匀扰动，"
        "即 ±5% 是误差边界而非 1σ）评估注入可行性的概率分布，"
        "返回 ok / risky / under 三类结局的概率，以及按阈值判定的告警等级"
        "（level / resolved / p_ci，阈值是风险接受准则、可用环境变量覆盖）。"
        "ok = 压力介于静水压力与破裂压力之间；risky = 超过破裂压力（有压裂风险）；"
        "under = 低于静水压力（注不进）。"
        "corr 可给深度与压力的相关系数（默认 None = 独立，这是一个假设）；"
        "include_sensitivity=True 时额外返回 Sobol 一阶/总效应指数，"
        "回答'压裂概率主要是深度还是压力造成的'。"
    ),
)
def assess_injection_uncertainty(
    depth_m: float,
    injection_pressure_bar: float,
    uncertainty: float = 0.05,
    n_simulations: int = 100,
    corr: float | None = None,
    include_sensitivity: bool = False,
) -> dict:
    """注入风险的概率分布（蒙特卡洛）+ 决策判定 + 可选敏感性分析。"""
    from tools.physics_tools import injection_risk_sensitivity, injection_risk_uq

    out = injection_risk_uq(
        depth_m, injection_pressure_bar,
        rel_sigma=uncertainty, n=n_simulations, corr=corr,
    )
    if include_sensitivity:
        out["sensitivity"] = injection_risk_sensitivity(
            depth_m, injection_pressure_bar, rel_sigma=uncertainty
        )
    return out


@server.tool(
    name="optimize_injection_plan",
    description=(
        "从'诊断'到'处方'：给定储层埋深与当前井底压力（或目标速率），"
        "在风险预算内反解注入方案 —— 推荐井数、井底压力上限、对应速率，"
        "并与当前方案逐项对比。约束用概率的 95% Wilson 上界（不是点估计），"
        "因此'P=4.9% 低于 5%'这种在 n=100 下站不住的说法会被拒绝。"
        "默认 1000 次抽样（约束判定需要比报告更高的分辨率）。"
        "返回 status：ok / infeasible（风险可行但注入能力不够）/ "
        "no_injection（连注都注不进），以及 current / recommended / "
        "well_options / frontier / notes / provenance。"
        "速率是**压力约束下的上限**（未计井筒摩阻、非达西、井间干扰），"
        "引用时必须带这句限定。"
    ),
)
def optimize_injection_plan(
    depth_m: float,
    injection_pressure_bar: float | None = None,
    storage_rate_mt_per_year: float | None = None,
    risk_budget: float = 0.05,
    n_simulations: int = 1000,
    corr: float | None = None,
    skin: float = 0.0,
) -> dict:
    """注入策略优化（井数 / 井底压力上限 / 速率）。"""
    from tools.physics_tools import optimize_injection_strategy

    return optimize_injection_strategy(
        depth_m,
        injection_pressure_bar=injection_pressure_bar,
        storage_rate_mt_per_year=storage_rate_mt_per_year,
        p_target=risk_budget,
        n=n_simulations,
        corr=corr,
        skin=skin,
    )


@server.tool(
    name="predict_pressure_pinn",
    description=(
        "用一维扩散方程 dP/dt = D·d²P/dx²（D = 0.1 m²/s）的物理信息神经网络，"
        "预测距井口 x 米、注入 t 天后的地层压力（bar）。"
        "适用域：x ∈ [0, 100] m，t ∈ [0, 30] 天；越界输入会被裁剪到边界。"
        "scenario='injection'（默认）为井口 120 bar、远端 100 bar、初值 100 bar，"
        "是非平凡解；scenario='baseline' 是初值与两端恒为 100 bar 的退化算例，"
        "其解析解恒为 100 bar（保留用于对照）。"
        "**定位：教学演示。** 边界条件与 D=0.1 m²/s 是示意值，未标定到任何实际"
        "储层；输出不得用于安全/不安全判定，也不参与物理一致性校验。"
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
        # 定位声明随数值一起返回，而不是只写在 README 里：调用方（报告 / 前端）
        # 拿到 109.95 bar 时，必须同时拿到"这是教学演示"这句话。
        "usage_scope": "teaching-demo",
        "not_for_safety_verdict": True,
    }
    if scenario == "injection":
        # 附上解析稳态解，让调用方能判断这个数离"物理上应该的答案"有多远。
        # t=10 天已接近稳态，两者应相差不到 0.1 bar（实测 0.049）。
        out["analytic_steady_state_bar"] = round(steady_state_pressure(float(x)), 3)
    return out


@server.tool(
    name="query_ccus_graph",
    description=(
        "从 CCUS 知识图谱（22 篇文献全文构建，4,826 实体 / 5,008 关系）检索与问题相关的"
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


@server.tool(
    name="get_fault_stability",
    description=(
        "判定给定埋深与注入压力下，一条断层会不会滑动（Mohr-Coulomb 滑动判据 + "
        "Terzaghi 有效应力）。返回滑移比 τ/(μσ'n)（≥1 即滑动）、临界孔隙压力、"
        "余量，以及张性开启（孔隙压力顶开断层面）是否发生。"
        "这是比'超破裂压力'更早触发的风险：断层在 1.5 倍静水以下就可能滑移，"
        "此时注入压力离破裂压力还有很大余量。"
        "**应力剖面是假设值**（Sv=23 MPa/km、Shmin=0.75Sv、SHmax=0.90Sv、μ=0.6），"
        "不是场地实测，结论只在该假设成立时有效。"
    ),
)
def get_fault_stability(
    depth_m: float,
    injection_pressure_bar: float,
    dip_deg: float = 60.0,
    azimuth_deg: float = 90.0,
) -> dict:
    """断层活化判定（假设应力剖面 + 两口径交叉验证）。"""
    from tools.physics_tools import fault_stability

    return fault_stability(
        depth_m, injection_pressure_bar,
        dip_deg=dip_deg, azimuth_deg=azimuth_deg,
    )


@server.tool(
    name="get_caprock_integrity",
    description=(
        "判定盖层能不能封住 CO2：毛细管突破压力（Laplace/Washburn，"
        "Pc = 2γcosθ/r）、可封 CO2 柱高（H = Pc/(Δρ·g)），"
        "以及力学失效界限（注入压力 ≥ σhmin + 抗张强度）。"
        "给 co2_column_height_m 或 injection_rate_m3_per_day 时给出判定结论，"
        "都不给时只返回突破压力与可封柱高（verdict='no_data'）。"
        "Δρ 由 Peng-Robinson 状态方程算出，不是常数。"
        "给 trap_area_m2 时同时算圈闭聚集口径：同一批物质按封闭构造面积聚集，"
        "柱高更高、突破更早，与一维口径构成区间（早界为保守侧）。"
        "**孔喉半径与接触角取文献默认值**（10 nm / 30°），非 MICP 实测。"
    ),
)
def get_caprock_integrity(
    depth_m: float,
    injection_pressure_bar: float,
    temperature_celsius: float = 60.0,
    co2_column_height_m: float | None = None,
    injection_rate_m3_per_day: float | None = None,
    trap_area_m2: float | None = None,
) -> dict:
    """盖层封闭性判定（毛细管 + 力学两条路径）。"""
    from tools.physics_tools import caprock_check

    return caprock_check(
        depth_m, injection_pressure_bar,
        temperature_celsius=temperature_celsius,
        co2_column_height_m=co2_column_height_m,
        injection_rate_m3_per_day=injection_rate_m3_per_day,
        trap_area_m2=trap_area_m2,
    )


def _selftest() -> int:
    """不经过 MCP 协议，直接调各函数 —— 用来区分"工具坏了"和"传输坏了"。"""
    print("=== get_co2_density ===")
    print(get_co2_density(250, 80))

    print("\n=== assess_injection_uncertainty ===")
    print(assess_injection_uncertainty(2500, 250))

    print("\n=== optimize_injection_plan ===")
    print(optimize_injection_plan(2500, 380)["text"])

    print("\n=== get_fault_stability ===")
    print(get_fault_stability(2500, 250)["text"])

    print("\n=== get_caprock_integrity ===")
    print(get_caprock_integrity(2500, 250, 60, co2_column_height_m=50)["text"])

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
