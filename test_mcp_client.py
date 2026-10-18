"""
MCP Client 测试脚本
===================
以真实 MCP 客户端身份（stdio 传输、JSON-RPC 消息、独立子进程）连接
``mcp_server_ccus.py``，列出工具并逐个调用，验证返回值是否正确。

和"直接 import 函数"的区别
--------------------------
直接调函数只能证明**函数**是对的。这里要证明的是**协议层**也对：
工具名与 schema 有没有正确暴露、参数能不能被序列化过去、返回值能不能
被反序列化回来、错误工具名会不会被正确拒绝。这四件事任何一件错了，
Cursor / Claude Desktop 都连不上，而函数本身仍然"测试通过"。

为什么图谱那条要单独给长超时
----------------------------
``query_ccus_graph`` 首次调用要初始化 LightRAG 并加载本地模型（约 2.3 GB），
在 CPU/冷缓存下可能几十秒。默认的读超时会把它误判成失败，所以这条单独放宽。

用法::

    python test_mcp_client.py            # 全部三条
    python test_mcp_client.py --no-graph # 跳过图谱（不加载模型，几秒跑完）
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = Path(__file__).resolve().parent
SERVER = "mcp_server_ccus.py"
EXPECTED_TOOLS = {"get_co2_density", "predict_pressure_pinn", "query_ccus_graph"}

_results: list[tuple[bool, str, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    _results.append((bool(ok), name, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def payload(result) -> dict:
    """从 CallToolResult 里取回字典。

    MCP 只保证 ``content`` 里有文本块；服务端返回 dict 时会被序列化成 JSON
    文本，所以这里解析文本而不是依赖 structured_content —— 后者需要服务端
    显式声明 output schema，对 Cursor 这类客户端反而不是必需的。
    """
    if result.structured_content:
        return result.structured_content
    for block in result.content or []:
        text = getattr(block, "text", None)
        if text:
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return {"_text": text}
    return {}


async def run(with_graph: bool) -> int:
    params = StdioServerParameters(
        command=sys.executable,
        args=[str(ROOT / SERVER)],
        cwd=str(ROOT),
    )

    print("\n=== 连接 MCP Server（stdio 子进程）===")
    async with stdio_client(params) as (read, write):
        # 图谱那条要等模型加载，超时放宽到 5 分钟
        async with ClientSession(read, write, read_timeout_seconds=300) as session:
            t0 = time.perf_counter()
            info = await session.initialize()
            print(f"  已连接：{info.server_info.name} v{info.server_info.version}"
                  f"（{(time.perf_counter() - t0) * 1000:.0f} ms）")

            # ─── ① 工具发现：schema 必须能被客户端看到 ───
            print("\n=== ① 工具发现 ===")
            listed = (await session.list_tools()).tools
            names = {t.name for t in listed}
            check("服务端暴露了 3 个工具", len(listed) == 3, f"{sorted(names)}")
            check("工具名与需求一致", names == EXPECTED_TOOLS,
                  f"缺失 {EXPECTED_TOOLS - names}" if names != EXPECTED_TOOLS else "")

            by_name = {t.name: t for t in listed}
            check("get_co2_density 的 schema 含 pressure_bar / temperature_celsius",
                  {"pressure_bar", "temperature_celsius"} <=
                  set((by_name["get_co2_density"].input_schema or {}).get("properties", {})))
            check("predict_pressure_pinn 的 schema 含 x / t",
                  {"x", "t"} <=
                  set((by_name["predict_pressure_pinn"].input_schema or {}).get("properties", {})))
            check("每个工具都有 description（否则模型不知道何时调用）",
                  all(t.description for t in listed))

            # ─── ② 密度：结果必须与状态方程一致 ───
            print("\n=== ② get_co2_density(250 bar, 80℃) ===")
            res = await session.call_tool(
                "get_co2_density", {"pressure_bar": 250, "temperature_celsius": 80}
            )
            d = payload(res)
            expected = 1.9 * 250 / (80 + 273.15)
            check("调用未报错", not res.is_error, str(d)[:80])
            check("返回结构化密度值", isinstance(d.get("density_kg_m3"), (int, float)))
            check(f"密度 ≈ {expected:.2f} kg/m³（状态方程 1.9P/T）",
                  abs(d.get("density_kg_m3", 0) - expected) < 0.01,
                  f"得到 {d.get('density_kg_m3')}")

            # ─── ③ PINN：必须与解析稳态解对得上 ───
            print("\n=== ③ predict_pressure_pinn(50 m, 10 d) ===")
            res = await session.call_tool(
                "predict_pressure_pinn", {"x": 50, "t": 10}
            )
            p = payload(res)
            check("调用未报错", not res.is_error)
            check("压力 ≈ 110 bar（injection 稳态解析解）",
                  abs(p.get("pressure_bar", 0) - 110.0) < 1.0,
                  f"得到 {p.get('pressure_bar')} bar")
            check("PINN 与解析解差值 < 0.5 bar",
                  abs(p.get("pressure_bar", 0) - p.get("analytic_steady_state_bar", 0)) < 0.5,
                  f"PINN {p.get('pressure_bar')} vs 解析 {p.get('analytic_steady_state_bar')}")

            res = await session.call_tool(
                "predict_pressure_pinn", {"x": 50, "t": 10, "scenario": "baseline"}
            )
            b = payload(res)
            check("baseline 场景恒为 100 bar（退化算例，见 README）",
                  abs(b.get("pressure_bar", 0) - 100.0) < 0.5,
                  f"得到 {b.get('pressure_bar')} bar")

            # ─── ④ 图谱检索：证据片段必须非空 ───
            if with_graph:
                print("\n=== ④ query_ccus_graph（首次调用会加载模型，请稍候）===")
                t1 = time.perf_counter()
                res = await session.call_tool(
                    "query_ccus_graph",
                    {"query": "断层活化与注入压力的关系", "top_k": 3},
                )
                g = payload(res)
                elapsed = time.perf_counter() - t1
                check("调用未报错", not res.is_error, f"{elapsed:.1f}s")
                check("返回了证据片段", int(g.get("count", 0)) > 0,
                      f"count={g.get('count')}")
                frags = g.get("fragments") or []
                check("片段内容非空", bool(frags) and all(len(f) > 50 for f in frags),
                      f"首段 {len(frags[0]) if frags else 0} 字")
            else:
                print("\n=== ④ query_ccus_graph（已跳过：--no-graph）===")

            # ─── ⑤ 错误路径：不存在的工具必须被拒绝 ───
            print("\n=== ⑤ 错误路径 ===")
            try:
                bad = await session.call_tool("no_such_tool", {})
                check("未知工具被拒绝", bool(bad.is_error), str(payload(bad))[:80])
            except Exception as exc:  # noqa: BLE001 — 协议层直接抛错也算拒绝
                check("未知工具被拒绝", True, f"{type(exc).__name__}")

            try:
                bad = await session.call_tool("get_co2_density", {"pressure_bar": 250})
                check("缺参数被拒绝", bool(bad.is_error), str(payload(bad))[:80])
            except Exception as exc:  # noqa: BLE001
                check("缺参数被拒绝", True, f"{type(exc).__name__}")

    passed = sum(1 for ok, _, _ in _results if ok)
    total = len(_results)
    print("\n" + "=" * 62)
    print(f"测试总结：{passed} PASS / {total - passed} FAIL")
    if passed != total:
        print("失败项：")
        for ok, name, detail in _results:
            if not ok:
                print(f"  - {name}  {detail}")
        return 1
    print("✅ MCP Server 三个工具均按预期返回")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-graph", action="store_true",
                    help="跳过 query_ccus_graph（不加载本地模型）")
    args = ap.parse_args()
    return asyncio.run(run(with_graph=not args.no_graph))


if __name__ == "__main__":
    raise SystemExit(main())
