"""
GraphRAG + 多智能体 E2E 回归测试
=================================
一键验证完整链路：
  环境检查 → 图谱存储 → LightRAG 检索 → 完整工作流
  （参数提取 → 图谱检索 + Rerank 精排 → Function Calling 报告生成 → 评审达标）

运行:
    python test_graph_rag_e2e.py

全部通过时退出码 0，任何 FAIL 退出码 1（可接入 CI）。
"""
import os
import sys
import time
import json
import urllib.request

# ─── 前置：清代理（httpx 会读 http_proxy 导致 localhost Ollama 超时）───
for _k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
    os.environ.pop(_k, None)
os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
os.environ.setdefault("no_proxy", "localhost,127.0.0.1")

PASS, FAIL = [], []


def check(name: str, cond: bool, detail: str = ""):
    mark = "PASS" if cond else "FAIL"
    print(f"  [{mark}] {name}" + (f"  ({detail})" if detail else ""))
    (PASS if cond else FAIL).append(name)


def ollama_get(path: str, payload: dict = None):
    req = urllib.request.Request(
        f"http://localhost:11434{path}",
        data=json.dumps(payload).encode() if payload else None,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read().decode())


def main():
    t0 = time.time()
    print("=" * 60)
    print("GraphRAG + 多智能体 E2E 回归测试")
    print("=" * 60)

    # ─── Step 0: 环境检查 ──────────────────────────────
    print("\n[Step 0] 环境检查")
    try:
        tags = ollama_get("/api/tags")
        models = [m["name"] for m in tags.get("models", [])]
        check("Ollama 服务运行", True, f"{len(models)} 个模型")
    except Exception as e:
        check("Ollama 服务运行", False, str(e))
        print("\n❌ Ollama 未运行，测试终止。请先启动 Ollama。")
        sys.exit(1)
    check("qwen2.5 模型可用", any(m.startswith("qwen2.5") for m in models))
    check("nomic-embed-text 模型可用", any(m.startswith("nomic-embed-text") for m in models))

    graphml = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "lightrag_storage", "graph_chunk_entity_relation.graphml")
    check("知识图谱存储存在", os.path.exists(graphml),
          f"{os.path.getsize(graphml)} bytes" if os.path.exists(graphml) else "缺失，请先跑 graph_rag_test.py 建图")

    # ─── Step 1: LightRAG 检索模块 ─────────────────────
    print("\n[Step 1] LightRAG 检索模块")
    try:
        from lightrag_retriever import retrieve_documents
        docs = retrieve_documents("断层活化 盖层完整性 注入压力", mode="hybrid")
        check("图谱检索返回片段", len(docs) > 0, f"{len(docs)} 个片段")
        if docs:
            avg_len = sum(len(d) for d in docs) / len(docs)
            check("片段平均长度（Rerank 区分度）", avg_len >= 80, f"平均 {avg_len:.0f} 字")
            has_rel = any(d.startswith("[关系]") for d in docs)
            check("片段包含关系语义（增强拼接）", has_rel)
    except Exception as e:
        check("LightRAG 检索模块", False, str(e))

    # ─── Step 2: 完整多智能体工作流 ─────────────────────
    print("\n[Step 2] 完整多智能体工作流（含反思循环，约 1-3 分钟）")
    try:
        from multi_agent_ccus import app as agent_app
        test_input = "某CCUS项目注入压力250bar，温度80℃，存在隐伏碳盐酸岩储层断层，请评估风险。"
        config = {"configurable": {"thread_id": f"e2e_test_{int(time.time())}"}}
        final_state = agent_app.invoke({"user_input": test_input}, config)
        check("工作流完整执行", True)
    except Exception as e:
        check("工作流完整执行", False, str(e))
        summary(t0)
        sys.exit(1)

    # ─── Step 3: 结果断言 ─────────────────────────────
    print("\n[Step 3] 结果断言")
    report = final_state.get("final_report", "")
    check("最终报告非空", len(report) > 200, f"{len(report)} 字")
    check("报告包含 CO2 密度计算值（Function Calling）", "kg/m³" in report or "kg/m3" in report)
    check("评审分数达标（score >= 8）", final_state.get("score", 0) >= 8,
          f"score={final_state.get('score')}")
    check("检索走的是 LightRAG 图谱引擎",
          final_state.get("token_usage", {}).get("retrieve", {}).get("engine") == "lightrag")
    check("检索到候选片段", final_state.get("token_usage", {})
          .get("retrieve", {}).get("candidate_docs", 0) > 0)

    summary(t0)


def summary(t0: float):
    print("\n" + "=" * 60)
    print(f"测试总结：{len(PASS)} PASS / {len(FAIL)} FAIL，耗时 {time.time() - t0:.0f}s")
    if FAIL:
        print("失败项：")
        for f in FAIL:
            print(f"  ✗ {f}")
        print("=" * 60)
        sys.exit(1)
    print("✅ 全部通过：多智能体 + GraphRAG 链路正常")
    print("=" * 60)


if __name__ == "__main__":
    main()
