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

    # ─── Step 1-B: 文件记忆模块 ────────────────────────
    print("\n[Step 1-B] 文件记忆（File-Based Memory）")
    try:
        import file_memory
        file_memory.ensure_files()
        check("user_profile.md 存在",
              os.path.exists(file_memory.USER_PROFILE_PATH))
        check("project_history.md 存在",
              os.path.exists(file_memory.PROJECT_HISTORY_PATH))
        ctx = file_memory.build_memory_context()
        check("文件记忆可组装为 Prompt 上下文", len(ctx) > 0, f"{len(ctx)} 字")

        # 边界检查：历史增长后必须只取最近若干条。
        # 写临时文件而不是真实历史 —— 测试绝不能毁掉用户已积累的记录。
        history_before = os.path.getsize(file_memory.PROJECT_HISTORY_PATH)
        orig_path = file_memory.PROJECT_HISTORY_PATH
        tmp_history = orig_path + ".e2e-tmp"
        try:
            many = "\n\n".join(f"## 2026-01-01 00:{i:02d}\n- 条目{i}" for i in range(30))
            with open(tmp_history, "w", encoding="utf-8") as f:
                f.write(many)
            file_memory.PROJECT_HISTORY_PATH = tmp_history
            recent = file_memory.read_project_history(max_entries=2)
            check("历史读取只取最近 N 条（防上下文膨胀）",
                  "条目29" in recent and "条目0" not in recent)
        finally:
            file_memory.PROJECT_HISTORY_PATH = orig_path
            if os.path.exists(tmp_history):
                os.remove(tmp_history)
    except Exception as e:
        check("文件记忆模块", False, str(e))
        history_before = 0

    # ─── Step 1-D: 记忆压缩（Memory Summarization）──────
    print("\n[Step 1-D] 记忆压缩（真实大模型 + 临时文件，不碰真实历史）")
    try:
        import file_memory
        from multi_agent_ccus import get_llm, llm_chat, SUMMARY_PROMPT

        orig_path = file_memory.PROJECT_HISTORY_PATH
        tmp_path = orig_path + ".e2e-compress-tmp"
        try:
            # 造一份真正超过生产阈值（3000 字符）的历史。
            # 10 个差异明显的场景 × 2~3 个压力/温度变体 ≈ 24 条 —— 既贴近真实项目
            # 长期积累的规模，又保证参数各异，模型才能提炼出多条不同规律
            # （复制同一句话只能提炼出一条，那验证的是数据而不是压缩能力）。
            scenarios = [
                ("含烃碳酸盐岩储层，隐伏断层，注入速率偏高", 250, 80,
                 "断层活化风险高，注入压力接近盖层突破压力，必须布设微地震监测并控制注入速率"),
                ("砂岩储层，无断层构造，盖层为厚层泥岩", 180, 55,
                 "盖层完整性良好，主要风险转移到井筒腐蚀与长期密封性，建议定期井中地球物理测井"),
                ("高盐度地层水，碳酸盐岩储层，含少量 H2S", 300, 90,
                 "CO2 密度升高叠加高盐度使井筒腐蚀显著加剧，需采用耐腐蚀合金油管并加装缓蚀剂"),
                ("枯竭气藏改建封存库，地层压力已显著亏空", 120, 45,
                 "注入能力不足，需评估增压方案与相态变化，同时警惕低压条件下的相变导致注入失败"),
                ("含隐伏断层且注入速率高，储层非均质性强", 280, 85,
                 "压力积聚速度快，盖层突破压力是首要控制因素，需实时压力监测与分注调控"),
                ("咸水层封存，厚泥岩盖层，构造简单", 200, 65,
                 "封存潜力好，风险集中在长期泄漏监测，建议时移地震与土壤气体通量测量组合"),
                ("海上咸水层，井筒穿越多套含水层", 220, 70,
                 "井筒完整性是多层系风险源，需重点核查固井质量与套管腐蚀余量"),
                ("含天然裂缝的碳酸盐岩，构造应力场复杂", 260, 82,
                 "天然裂缝可能成为泄漏通道，需结合 InSAR 与微地震联合判别裂缝活化"),
                ("低渗致密砂岩，需压裂改造后注入", 240, 75,
                 "压裂可能沟通隐伏断层，需先做断层三维刻画再决定压裂规模"),
                ("老井密集区改建封存，存在废弃井", 190, 60,
                 "废弃井是最大泄漏风险点，需逐井排查封堵质量并建立长期完整性档案"),
            ]
            blocks = ["# 项目评估历史（Project History）\n\n> 说明行\n\n"]
            n = 0
            for sc, p, t, c in scenarios:
                for dp, dt in ((0, 0), (20, 5), (40, 8)):
                    n += 1
                    blocks.append(
                        f"## 2026-04-{n:02d} 10:00\n"
                        f"**场景**: {sc}\n"
                        f"**关键参数**: 注入压力 {p + dp} bar / 温度 {t + dt} ℃\n"
                        f"**核心结论**: {c}；本工况压力 {p + dp} bar 下该结论依然成立。\n"
                        f"**评审分**: {8 + (n % 2)}/10\n\n"
                    )
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write("".join(blocks))
            file_memory.PROJECT_HISTORY_PATH = tmp_path

            size_before = len(open(tmp_path, encoding="utf-8").read())
            check("构造的临时历史超过压缩阈值（按字符）",
                  size_before > file_memory.HISTORY_COMPRESS_THRESHOLD,
                  f"{size_before} 字符 > {file_memory.HISTORY_COMPRESS_THRESHOLD}")
            check("阈值确实按字符而非字节比较",
                  size_before == file_memory.history_char_count())

            # 用真实本地模型做摘要（与生产路径同一个 llm_chat 入口）
            def _real_summarizer(old_text: str) -> str:
                _, model, tier, _ = get_llm("summary")
                resp = llm_chat(
                    "summary",
                    messages=[{"role": "user",
                               "content": SUMMARY_PROMPT.format(old_text=old_text[:12000])}],
                    temperature=0.2,
                )
                return resp.choices[0].message.content

            result = file_memory.compress_history_if_needed(
                summarizer=_real_summarizer, threshold=file_memory.HISTORY_COMPRESS_THRESHOLD
            )
            check("压缩被触发并成功", result["status"] == "compressed",
                  f"{result['status']} {result.get('reason', '')}")
            check("压缩后文件变小", result["after_chars"] < result["before_chars"],
                  f"{result['before_chars']} → {result['after_chars']} 字")
            check("有旧条目被提炼", result["dropped_entries"] > 0,
                  f"{result['dropped_entries']} 条")

            with open(tmp_path, encoding="utf-8") as f:
                text = f.read()
            check("生成了核心经验块", "项目核心经验" in text)
            lessons = [ln for ln in text.splitlines() if ln.strip().startswith("- ")]
            check("提炼了多条可复用经验", 2 <= len(lessons) <= 4, f"{len(lessons)} 条")
            check("最新记录原样保留", "老井密集区改建封存" in text, "keep_recent=2")
            check("最旧记录已被压缩移除", "含烃碳酸盐岩储层，隐伏断层，注入速率偏高" not in text)
            check("文件头未被破坏", text.startswith("# 项目评估历史"))

            # 自限验证：此时旧条目已归零，即便阈值调到极低也不应再次压缩。
            # 否则单条超长记录会导致每次读取都调一次大模型。
            size_after_first = os.path.getsize(tmp_path)
            second = file_memory.compress_history_if_needed(
                summarizer=lambda _old: "- 不应该被调用", threshold=10
            )
            check("不会无限重复压缩", second["status"] == "skipped",
                  f"{second['status']}/{second.get('reason', '')}")
            check("第二次调用未改动文件",
                  os.path.getsize(tmp_path) == size_after_first)
        finally:
            file_memory.PROJECT_HISTORY_PATH = orig_path
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            leftover = [f for f in os.listdir(os.path.dirname(orig_path) or ".")
                        if f.startswith(".history-")]
            check("原子写入无临时文件残留", not leftover, str(leftover))
    except Exception as e:
        check("记忆压缩模块", False, str(e))

    # ─── Step 1-C: 多模型路由配置 ──────────────────────
    print("\n[Step 1-C] 多模型路由（Model Routing）")
    try:
        from multi_agent_ccus import (
            MODEL_ROUTING, LOCAL_MODEL, CLOUD_MODEL,
            cloud_available, resolve_route,
        )
        check("路由表覆盖核心角色",
              all(k in MODEL_ROUTING for k in ("extract", "report", "evaluate")))
        check("参数提取走本地模型", MODEL_ROUTING.get("extract") == "local")
        check("评审员走本地模型", MODEL_ROUTING.get("evaluate") == "local")
        # 只判断档位，不比对字符串：配了 ROUTE_REPORT=cloud:xxx 时值是
        # "cloud:xxx" 而非 "cloud"，硬比对会假失败。
        check("报告生成配置为云端档位",
              MODEL_ROUTING.get("report", "").split(":")[0] == "cloud",
              MODEL_ROUTING.get("report", "-"))

        if cloud_available():
            check("云端 API 已配置", True, f"{CLOUD_MODEL} 可用")
        else:
            # 没有 key 时必须显式降级，而不是静默假装走了云端
            resolved, reason = resolve_route("report")
            check("未配置云端时显式降级（不静默）",
                  resolved == "local" and bool(reason),
                  f"→ {resolved}:{LOCAL_MODEL}（{reason}）")

        # 模拟"已配置 key"的场景，验证云端路由真的会切过去。
        # 只替换模块级 client，不发任何网络请求。
        import multi_agent_ccus as _m
        _orig = _m.cloud_client
        try:
            _m.cloud_client = object()
            r2, _ = _m.resolve_route("report")
            _, mdl2, tier2, _ = _m.get_llm("report")
            # 不拿 CLOUD_MODEL 比对：配了 ROUTE_REPORT=cloud:xxx 时模型来自覆盖值，
            # 断言"档位切到 cloud 且模型名非空"才不会因环境不同而假失败。
            check("配置 key 后 report 切到云端模型",
                  r2 == "cloud" and tier2 == "cloud" and bool(mdl2),
                  f"{tier2}:{mdl2}")
            r3, _ = _m.resolve_route("extract")
            check("配置 key 后 extract 仍留在本地", r3 == "local")
        finally:
            _m.cloud_client = _orig
    except Exception as e:
        check("多模型路由配置", False, str(e))

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

    # ─── Step 4: 新特性断言 ───────────────────────────
    print("\n[Step 4] 新特性断言（多模型路由 + 文件记忆）")

    routing = final_state.get("model_routing", {})
    check("运行时记录了模型路由", len(routing) >= 3, str(routing))
    check("extract 使用本地模型",
          routing.get("extract", "").startswith("local:"), routing.get("extract", "-"))
    check("evaluate 使用本地模型",
          routing.get("evaluate", "").startswith("local:"), routing.get("evaluate", "-"))
    check("report 使用了预期档位",
          routing.get("report", "").startswith(("cloud:", "local:")),
          routing.get("report", "-"))

    check("文件记忆被读取并注入 Prompt",
          bool(final_state.get("file_memory_context")),
          f"{len(final_state.get('file_memory_context') or '')} 字")
    check("本次评估写入了 project_history.md",
          bool(final_state.get("history_entry")),
          f"{len(final_state.get('history_entry') or '')} 字")

    try:
        history_after = os.path.getsize(file_memory.PROJECT_HISTORY_PATH)
        check("project_history.md 确实增长了",
              history_after > history_before,
              f"{history_before} → {history_after} 字节")
        tail = file_memory.read_project_history(max_entries=1)
        check("历史文件内容可被重新读回",
              bool(tail) and ("核心结论" in tail or "场景" in tail))
    except Exception as e:
        check("历史文件落盘校验", False, str(e))

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
