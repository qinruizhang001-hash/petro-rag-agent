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
import tempfile
import urllib.request

# ─── 前置：清代理（httpx 会读 http_proxy 导致 localhost Ollama 超时）───
for _k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
    os.environ.pop(_k, None)
os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
os.environ.setdefault("no_proxy", "localhost,127.0.0.1")

# ─── 前置：把自我进化引擎隔离到临时目录 ───
# 每次评估结束时 history_node 会沉淀一条经验（这是被验证的功能，不该在测试里
# 关掉）。但测试数据不能进真实的 history/：攒够 20 条就会触发一次真实反思，
# 把测试问句当成"经验"去提炼规律，而它看起来完全正常。
# 同时关掉图谱写入 —— 测试不该改动语料图谱。
os.environ.setdefault("EVOLVE_HOME", tempfile.mkdtemp(prefix="ccus_e2e_evolve_"))
os.environ.setdefault("EVOLVE_WRITE_GRAPH", "0")
# 文件记忆（project_history.md）同样必须隔离，否则测试问句会混进真实历史
_tmp_state = tempfile.mkdtemp(prefix="ccus_e2e_state_")
os.makedirs(_tmp_state, exist_ok=True)
os.environ.setdefault("PROJECT_HISTORY_PATH", os.path.join(_tmp_state, "project_history.md"))

PASS, FAIL = [], []


def check(name: str, cond: bool, detail: str = ""):
    mark = "PASS" if cond else "FAIL"
    print(f"  [{mark}] {name}" + (f"  ({detail})" if detail else ""))
    (PASS if cond else FAIL).append(name)


def ollama_get(path: str, payload: dict = None):
    # 127.0.0.1 而不是 localhost：Windows 上 localhost 先解析到 ::1，而 Ollama
    # 只监听 IPv4，每次请求要多等约 2 秒的 IPv6 失败回退。
    req = urllib.request.Request(
        f"http://127.0.0.1:11434{path}",
        data=json.dumps(payload).encode() if payload else None,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read().decode())


# ─── 进程内 API 服务（Step 6 与 Step 8 共用）──────────────
_API_SERVER = None
_API_PORT = int(os.environ.get("E2E_API_PORT", "8000"))
API_BASE = f"http://127.0.0.1:{_API_PORT}"


def _port_open(port, timeout=0.3):
    """端口有没有人监听（不看是谁）。"""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        return s.connect_ex(("127.0.0.1", port)) == 0


def ensure_api_server(check):
    """
    在**本进程内**起一个 uvicorn 线程，并把 API_BASE 指向它。

    为什么必须是自己起的，而不是"端口上有人就复用"：
    `docker compose up` 的后端常驻 127.0.0.1:8000，它读的是 .env.docker，
    API_TOKEN 有值。复用它会得到一连串 401 —— 实测一次刷出 34 条假失败
    （SSE、成本仪表盘、HITL 全挂），而代码一行没坏。更麻烦的是 Step 7-B
    靠"改 api_server.API_TOKEN 这个模块级全局"来切鉴权，那改的是**本进程**
    的变量，对另一个进程里的服务毫无作用，于是"开启鉴权"这条也必然失败。
    结论：被测对象必须和测试同进程，否则测的是别人的配置。

    为什么同进程而不是另起一个 python：bge-reranker-v2-m3 单份约 3.95 GB
    commit，本机可用 commit 只有 4.5 GB 左右，两个进程各持一份必然有一个被
    系统杀掉（ConnectionReset / 裸 segfault，且没有任何 traceback）。Step 2
    的工作流本来就在本进程加载了模型，服务挂进来 share 同一份，增量接近 0。
    请求仍走真 socket、SSE 仍真分帧，没有绕过 HTTP。

    端口策略：默认 8000（可用 E2E_API_PORT 覆盖）；被占用就顺延到下一个空闲
    端口，并把 API_BASE 指向它。
    """
    global _API_SERVER, _API_PORT, API_BASE

    def _ready(port, timeout=5.0):
        import requests as _rq

        end = time.time() + timeout
        while time.time() < end:
            try:
                r = _rq.get(f"http://127.0.0.1:{port}/health", timeout=1)
                # 只认 200 还不够：要确认应答的确实是**我们起的**这一份。
                # 鉴权关闭是判据之一 —— 本进程读的是 .env（无 token）。
                if r.status_code == 200 and r.json().get("auth") == "disabled":
                    return True
            except Exception:
                pass
            time.sleep(0.2)
        return False

    if _port_open(_API_PORT):
        for cand in range(_API_PORT + 11, _API_PORT + 60):
            if not _port_open(cand):
                check("首选端口被占用，改用备用端口（避免测到容器里的配置）",
                      True, f"{_API_PORT} → {cand}")
                _API_PORT = cand
                API_BASE = f"http://127.0.0.1:{cand}"
                break
        else:
            check("找不到可用端口启动 API 服务", False, f"从 {_API_PORT} 起探测 49 个端口")
            return

    import threading
    import uvicorn
    from api_server import api as _api

    _API_SERVER = uvicorn.Server(
        uvicorn.Config(_api, host="127.0.0.1", port=_API_PORT, log_level="warning")
    )
    threading.Thread(target=_API_SERVER.run, daemon=True).start()
    check("进程内启动 API 服务（与被测配置同源，且不额外占一份 reranker）",
          _ready(_API_PORT, timeout=30.0), API_BASE)


def shutdown_api_server():
    global _API_SERVER
    if _API_SERVER is not None:
        _API_SERVER.should_exit = True
        _API_SERVER = None


def step_pinn():
    """PINN 工具端到端（经 MCP Server）。

    为什么必须单独跑一问带坐标的：其余测试问句都不含距离/时间，接地守卫会把
    模型规划的 PINN 调用**全部**丢掉 —— 那样只验证了"该拦的拦住了"，没验证
    "该出数的出得来"，守卫即使退化成永久关闭也照样全绿。这一问是需求原话。

    为什么做成独立函数而不是写进 main：整套 E2E 约 10 分钟，调这一步时不该
    为此重跑一遍。单独跑::

        python -c "import time, test_graph_rag_e2e as t; t.step_pinn(); t.summary(time.time())"
    """
    try:
        from workflow import app as agent_app
        pinn_input = "预测距离井口 50 米处，注入 10 天后的压力。"
        pinn_state = agent_app.invoke(
            {"user_input": pinn_input},
            {"configurable": {"thread_id": f"e2e_pinn_{int(time.time())}"}},
        )
        pinn = pinn_state.get("pinn_prediction") or {}
        check("问句带坐标时 PINN 被调用", bool(pinn), str(list(pinn)[:5]) if pinn else "未调用")
        p = pinn.get("pressure_bar")
        check("工具返回了压力值", isinstance(p, (int, float)), str(p))
        if isinstance(p, (int, float)):
            # injection 场景 10 天 ≈ 8.6τ，解析稳态解 P(50 m) = 110.000 bar
            check("与解析稳态解一致（110 bar ± 1）", abs(p - 110.0) <= 1.0,
                  f"{p:.3f} bar vs 110.000")
            check("报告正文写出了该数值", f"{p:.2f}" in (pinn_state.get("final_report") or ""),
                  f"{p:.2f}")
    except Exception as e:
        check("PINN 端到端调用", False, f"{type(e).__name__}: {e}")


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

    # ─── Step 1-F: 时间维度记忆（跨时间冲突检测，纯函数）──
    print("\n[Step 1-F] 时间维度记忆（跨时间冲突检测）")
    try:
        from core.temporal import analyze_temporal

        docs = [
            "2025年6月，A-1井实测地层压力为200 bar。",
            "2026年3月，A-1井地层压力升至300 bar。",
        ]
        tm = analyze_temporal(docs)
        check("抽取到带时间戳的断言", tm["claims"] >= 2, f"{tm['claims']} 条")
        check("检测到跨时间冲突", len(tm["conflicts"]) == 1,
              str(tm["conflicts"][:1]))

        c0 = (tm["conflicts"] or [{}])[0]
        check("冲突以最新时间戳为准",
              c0.get("newer", {}).get("timestamp") == "2026-03"
              and c0.get("newer", {}).get("value") == 300.0,
              str(c0.get("newer")))
        check("冲突提示包含强制性指令（优先最新）",
              "必须优先采用时间戳最新的数据" in tm["notice"])
        check("冲突提示禁止取平均", "禁止" in tm["notice"])

        agree = analyze_temporal([
            "2025年6月，A-1井地层压力 200 bar。",
            "2026年3月，A-1井地层压力 201 bar。",
        ])
        check("数值一致时不误报冲突", agree["conflicts"] == [] and agree["notice"] == "")

        # 埋深 2000 m 不能被当成 2000 年（误报会让每份报告都多出假冲突）
        check("裸四位数不被误认为年份",
              analyze_temporal(["埋深 2000 m，注入压力 250 bar。"])["claims"] == 0)
    except Exception as e:
        check("时间维度模块可用", False, f"{type(e).__name__}: {e}")

    # ─── Step 1-D: 记忆压缩（Memory Summarization）──────
    print("\n[Step 1-D] 记忆压缩（真实大模型 + 临时文件，不碰真实历史）")
    try:
        import file_memory
        from core.config import get_llm, llm_chat
        from nodes.aggregate_node import SUMMARY_PROMPT

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

    # ─── Step 1-E: 上下文压缩（真实大模型）──────────────
    print("\n[Step 1-E] 上下文压缩（真实本地模型，超阈值路径）")
    try:
        from core.config import get_llm, llm_chat
        from utils.context_pruner import (
            prune_context, needs_pruning, context_char_count,
            split_into_chunks, DEFAULT_PRUNE_THRESHOLD, DEFAULT_CHUNK_CHARS,
        )

        # 造一段真实的 CCUS 长上下文（> 4000 字符）。
        # 为什么要造而不是用真实检索结果：实测 Top-2 只有 600~960 字
        # （见下面 1-E-2 的断言），根本碰不到 4000 阈值，
        # 那样就永远验证不到压缩这条代码路径。
        # 每段写足 4~5 句（约 450 字），10 段合计约 5000 字 —— 必须显著超过
        # 阈值，否则"构造的上下文超阈值"这条断言自己就会先失败（踩过）。
        passages = [
            ("断层活化", 250, 80,
             "注入压力接近盖层突破压力时，隐伏断层可能沿优势方位重新活化，"
             "诱发微震事件并沿断层面形成垂向泄漏通道。断层活化的判据通常取"
             "莫尔-库仑破坏准则，当有效正应力下降、剪应力上升至破坏包络线时即判定为活化。"
             "工程上需布设微地震监测台网，控制注入速率不超过设计值的 80%，"
             "并在注入前完成断层三维精细刻画，避免在断层上盘 500 m 范围内布井。"),
            ("盖层完整性", 180, 55,
             "厚层泥岩盖层的突破压力通常为静水压力的 1.5~1.8 倍，当注入压力超过该阈值时"
             "盖层发生张性破坏，CO2 沿新生裂缝逸散。盖层评价需同时考虑厚度、脆性矿物含量"
             "与现今地应力方向；脆性指数高于 0.5 的盖层在热应力循环下更易产生裂缝。"
             "建议以声波测井结合岩心突破压力实验确定实际承载上限，"
             "并把运行压力控制在突破压力的 70% 以内留出安全余量。"),
            ("井筒腐蚀", 300, 90,
             "CO2 溶于地层水生成碳酸，pH 降至 4 以下时对碳钢的均匀腐蚀速率可达 1 mm/a 以上，"
             "并可能叠加点蚀与应力腐蚀开裂。腐蚀速率随温度升高先增后减，"
             "在 60~90 ℃ 区间最为严酷；同时高盐度地层水会加剧电化学腐蚀。"
             "工程上需采用耐蚀合金油管或内衬涂层，加装缓蚀剂并建立井下腐蚀监测点，"
             "定期用电磁探伤与超声测厚评估剩余壁厚。"),
            ("固井质量", 220, 70,
             "水泥石碳酸化会导致强度衰退与渗透率上升，形成微环空泄漏通道；"
             "注入引起的温度骤变产生的热应力与压力循环产生的机械应力还会引起套管变形。"
             "固井质量评价以声幅变密度测井为主，水泥胶结指数低于 0.8 即需补救。"
             "对于穿越多套含水层的井段，应使用防腐水泥浆体系并延长封固段，"
             "必要时采用挤水泥修复，确保各层系之间水力隔离。"),
            ("储层非均质性", 260, 82,
             "高渗条带会形成优势流通道，导致 CO2 波及效率下降、压力局部积聚，"
             "同时使预测的封存容量被显著高估。非均质性评价需结合岩心渗透率、"
             "测井解释与地震反演，建立三维地质模型并做多方案数值模拟。"
             "工程上可采用分层注入与流量调配，配合分布式光纤温度传感监测各层吸气剖面，"
             "及时识别窜流通道并调整注入策略。"),
            ("相态变化", 120, 45,
             "枯竭气藏压力亏空条件下，CO2 可能由超临界态转为气态，密度骤降、"
             "注入能力不足，井底可能出现两相流并导致注入压力异常升高。"
             "需评估增压方案与相态包络线，必要时先期回注垫气恢复地层压力。"
             "同时低压条件会加剧焦耳-汤姆逊效引起的井筒温降，"
             "存在水合物生成与冻堵风险，应加注抑制剂或采用加热油管。"),
            ("泄漏监测", 200, 65,
             "时移地震可识别 CO2 羽流运移范围，InSAR 可探测地表形变，"
             "土壤气体通量测量用于判别浅层泄漏，井中地球物理测井用于评价井筒完整性，"
             "四者组合的覆盖率最高。监测方案应覆盖注入期与闭场后至少 20 年，"
             "并以基线调查为参照建立背景值数据库，避免把自然波动误判为泄漏。"),
            ("老井风险", 190, 60,
             "废弃井是封存场地的最大泄漏风险点，历史井往往缺少完整钻井与固井记录，"
             "套管腐蚀与水泥环窜槽难以直接观测。需逐井排查封堵质量，"
             "采用井筒电磁探伤与声波测井定位漏点，建立长期完整性档案并定期复测。"
             "对于无法确认完整性的老井，应在封存场地边界外设置缓冲距离或实施重新封堵。"),
            ("注入能力", 240, 75,
             "注入能力取决于储层渗透率、有效厚度与地层破裂压力三者的共同约束。"
             "低渗储层需评估压裂改造的必要性，但压裂可能沟通隐伏断层，"
             "必须先完成断层三维刻画再决定改造规模与裂缝走向。"
             "现场以阶梯式注入测试确定破裂压力与闭合压力，"
             "并据此设定注入压力上限，避免长期超压导致盖层疲劳损伤。"),
            ("经济性", 210, 68,
             "封存项目的经济性主要由捕集成本、输送距离与监测年限决定。"
             "管道输送在 100 km 以内单位成本最低，超过 200 km 后船舶与槽车方案需重新比较。"
             "监测费用在闭场后仍将持续 20 年以上，需在项目初期计入全生命周期成本。"
             "此外，注入压力越高单位井注入率越大、井数越少，"
             "但井口设备与管材等级要求随之提高，存在明确的经济平衡点。"),
        ]
        blocks = []
        n = 0
        # 每个场景派生 3 个压力/温度变体 → 30 段、约 7000 字。
        # 之所以要这么长：单块上限 3000 字（本地模型 4096 tokens 上下文），
        # 只有超过它才能真正走到**分块压缩**这条路径。
        for topic, p, t, text in passages:
            for dp, dt in ((0, 0), (20, 5), (40, 8)):
                n += 1
                blocks.append(
                    f"[片段{n}·{topic}] 在注入压力 {p + dp} bar、温度 {t + dt} ℃ 的工况下，{text}"
                    f"该判据已在多个咸水层封存项目中验证，适用于埋深 1500~3500 m 的储层条件。"
                )
        long_context = "\n\n".join(blocks)

        n_chars = context_char_count(long_context)
        check("构造的长上下文超过压缩阈值", n_chars > DEFAULT_PRUNE_THRESHOLD,
              f"{n_chars} 字 > {DEFAULT_PRUNE_THRESHOLD}")
        check("needs_pruning 判定为需要压缩", needs_pruning(long_context))

        n_chunks = len(split_into_chunks(long_context))
        check("长上下文会被切成多块（本地模型仅 4096 tokens）",
              n_chunks >= 2, f"{n_chunks} 块")
        check("每块都在单次调用容量内",
              all(len(c) <= DEFAULT_CHUNK_CHARS
                  for c in split_into_chunks(long_context)),
              f"单块上限 {DEFAULT_CHUNK_CHARS} 字")

        captured = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                    "model": "", "tier": ""}

        def _real_pruner(prompt: str) -> str:
            _, pm, pt, _ = get_llm("prune")
            resp = llm_chat(
                "prune",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            u = getattr(resp, "usage", None)
            captured["calls"] += 1
            captured["prompt_tokens"] += int(getattr(u, "prompt_tokens", 0) or 0)
            captured["completion_tokens"] += int(getattr(u, "completion_tokens", 0) or 0)
            captured["model"] = pm
            captured["tier"] = pt
            return resp.choices[0].message.content

        result = prune_context(long_context, summarizer=_real_pruner)
        check("压缩被真实模型执行并成功", result["status"] == "pruned",
              f"{result['status']} {result.get('reason', '')}")
        if result["status"] == "pruned":
            check("压缩后显著变短", result["after_chars"] < result["before_chars"] * 0.85,
                  f"{result['before_chars']} → {result['after_chars']} 字"
                  f"（{result['saved_ratio']*100:.0f}%）")
            check("压缩结果是要点列表",
                  any(ln.strip().startswith("- ")
                      for ln in result["compressed"].splitlines()))
            # 数值必须活下来 —— 压缩最怕的是把阈值/压力丢掉，那就等于毁掉上下文
            nums_kept = [n for n in ("250", "80", "180", "300", "1.5")
                         if n in result["compressed"]]
            check("压缩后保留关键数值", len(nums_kept) >= 2, f"保留 {nums_kept}")
            check("实际走了分块压缩（不是单次吞下 7000 字）",
                  result["chunks"] >= 2 and result["calls"] >= 2,
                  f"{result['chunks']} 块 / {result['calls']} 次调用")
            check("每块调用都真的发出去了",
                  captured["calls"] == result["calls"],
                  f"实际 {captured['calls']} 次")
            check("压缩调用走了本地档位",
                  captured.get("tier") == "local", str(captured.get("tier")))
            check("压缩消耗了真实 token（跨块累加）",
                  captured["prompt_tokens"] > 0 and captured["completion_tokens"] > 0,
                  f"prompt={captured['prompt_tokens']} "
                  f"completion={captured['completion_tokens']}")

        # 失败必须回退原文（模型返回散文而不是要点）
        bad = prune_context(long_context, summarizer=lambda _p: "这是一段散文总结。")
        check("压缩失败时回退原文（不丢上下文）",
              bad["status"] == "failed" and bad["compressed"] == long_context,
              bad.get("reason", "")[:60])

        # ─── 1-E-2：真实检索结果的实际注入量 ───
        # 这是必须如实记录的一条：Top-2 远低于阈值，所以 4000 字压缩
        # 在当前语料下是**安全阀**而不是日常路径。
        #
        # ⚠️ 必须复用进程内已加载的 reranker 单例（core.reranker），不能 new 一个：
        # bge-reranker-v2-m3 单份约 2.3 GB，第二份会直接耗尽 Windows 页面文件
        # （OSError 1455），表现却是 "Unrecognized processing class" 这种
        # 完全指向错误方向的报错。踩过一次，记在这里。
        from lightrag_retriever import retrieve_documents
        from core.reranker import reranker as _rk
        _q = "断层活化 盖层完整性 注入压力"
        _docs = retrieve_documents(_q, mode="hybrid")
        if _docs:
            _sc = _rk.predict([[_q, d] for d in _docs])
            _top2 = [d for _, d in sorted(zip(_sc, _docs), key=lambda x: x[0],
                                          reverse=True)[:2]]
            _inj = len("\n\n".join(_top2))
            _all_chars = len("\n\n".join(_docs))
            check("Top-2 注入量远低于 4000 字阈值（压缩是安全阀）",
                  _inj < DEFAULT_PRUNE_THRESHOLD,
                  f"Top-2 = {_inj} 字；全量候选 = {_all_chars} 字")
            check("Top-1 rerank 分数可用于 RAG 命中判定",
                  float(max(_sc)) > 0.05, f"{float(max(_sc)):.4f}")
    except Exception as e:
        check("上下文压缩（真实模型）", False, str(e))

    # ─── Step 1-C: 多模型路由配置 ──────────────────────
    print("\n[Step 1-C] 多模型路由（Model Routing）")
    try:
        from core.config import (
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
        import core.config as _m
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

            # ─── 复杂度路由与静态表冲突时的模型名处理 ───
            # 跨档位切换时必须丢掉对端专有的模型名，否则会把本地模型名
            # 发给云端端点（或反之），直接 404。
            _orig_report = _m.MODEL_ROUTING["report"]
            try:
                # 静态配的是云端模型，复杂度判定要求本地 → 不能把云端名发给 Ollama
                _m.MODEL_ROUTING["report"] = "cloud:ollama/qwen2.5:latest"
                _, mdl_local, tier_local, _ = _m.get_llm("report", "local")
                check("降到本地时丢弃云端模型名",
                      tier_local == "local" and "ollama/" not in (mdl_local or ""),
                      f"{tier_local}:{mdl_local}")

                # 静态配的是本地模型，复杂度判定要求云端 → 不能把本地名发给云端
                _m.MODEL_ROUTING["report"] = "local:qwen2.5:14b"
                _, mdl_cloud, tier_cloud, _ = _m.get_llm("report", "cloud")
                check("升到云端时丢弃本地模型名",
                      tier_cloud == "cloud" and "14b" not in (mdl_cloud or ""),
                      f"{tier_cloud}:{mdl_cloud}")

                # 静态本身就是云端 spec，升级时应保留其模型覆盖
                _m.MODEL_ROUTING["report"] = "cloud:kimi/kimi-k2"
                _, mdl_keep, _, _ = _m.get_llm("report", "cloud")
                check("同为云端时保留指定模型",
                      mdl_keep == "kimi/kimi-k2", str(mdl_keep))
            finally:
                _m.MODEL_ROUTING["report"] = _orig_report
        finally:
            _m.cloud_client = _orig
    except Exception as e:
        check("多模型路由配置", False, str(e))

    # ─── Step 2: 完整多智能体工作流 ─────────────────────
    print("\n[Step 2] 完整多智能体工作流（含反思循环，约 1-3 分钟）")
    try:
        from workflow import app as agent_app
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

    # ─── 并行扇出 / 扇入 ─────────────────────────────
    # 这一段验证的不是"报告写得好不好"，而是"两条分支是否真的并发跑完并合流"。
    # 数字全部来自分支自己打的时间戳，不是按节点名推算的。
    geo_rep = final_state.get("geology_report") or ""
    well_rep = final_state.get("wellbore_report") or ""
    check("地质分支产出了独立结论", len(geo_rep) > 50, f"{len(geo_rep)} 字")
    check("井筒分支产出了独立结论", len(well_rep) > 50, f"{len(well_rep)} 字")
    check("两条分支结论互不相同（没有串写同一字段）",
          bool(geo_rep) and bool(well_rep) and geo_rep != well_rep)

    merge = final_state.get("branch_merge") or {}
    check("本次未降级（两条分支都成功）", merge.get("degraded") is False, str(merge.get("missing")))

    timing = final_state.get("parallel_timing") or {}
    check("记录了并行耗时", bool(timing.get("wall_s")), str(timing))
    if timing:
        print(f"      地质 {timing.get('geology_s')}s ∥ 井筒 {timing.get('wellbore_s')}s"
              f" → 并行 {timing.get('wall_s')}s"
              f"（串行需 {timing.get('serial_s')}s，加速 {timing.get('speedup')}×，"
              f"重叠 {timing.get('overlap_s')}s）")
        check("并行墙钟小于两分支耗时之和（确实并发了）",
              float(timing.get("wall_s", 0)) < float(timing.get("serial_s", 0)),
              f"wall={timing.get('wall_s')} serial={timing.get('serial_s')}")
        check("加速比 >= 1.0", float(timing.get("speedup", 0)) >= 1.0,
              f"{timing.get('speedup')}×")
        check("汇总节点耗时被单独记录",
              float(timing.get("aggregate_s", 0)) > 0,
              f"{timing.get('aggregate_s')}s")

    # ─── 物理一致性校验（规则引擎，不是模型打分）───
    phys = final_state.get("physics_check") or {}
    check("评审节点产出了物理校验结果", bool(phys), str(phys.get("status")))
    check("物理校验状态合法",
          phys.get("status") in ("pass", "warning", "violation", "unverifiable"),
          str(phys.get("status")))
    check("物理校验未误判为矛盾（本次工况不该被拦）",
          not phys.get("blocking"), str(phys.get("summary")))
    check("物理校验可解释（checks 逐条留痕）",
          bool(phys.get("checks")) and all("detail" in c for c in phys["checks"]),
          f"{len(phys.get('checks') or [])} 条")

    # ─── 不确定性量化（工作流实测）─────────────────────
    uqs = final_state.get("uq_summary") or {}
    check("工作流产出了不确定性量化结果", bool(uqs), str(sorted(uqs)))
    dens_uq = uqs.get("density") or {}
    check("密度 UQ 含 90% 区间",
          dens_uq.get("p5") is not None and dens_uq.get("p95") is not None,
          str(dens_uq.get("text")))
    check("密度 UQ 记录了模拟次数", int(dens_uq.get("n") or 0) == 100,
          str(dens_uq.get("n")))
    # 注入风险判定依赖埋深。本次问句里**没有**给埋深，所以正确行为是
    # "不判" —— 拿默认 2000 m 编一个概率比不给更危险：读者会把它当成
    # 针对本项目的结论。这里正反两个方向都钉住，防止哪天有人"顺手补个默认值"。
    has_depth = "埋深" in test_input or "深度" in test_input
    inj_probs = (uqs.get("injection") or {}).get("probabilities")
    if has_depth:
        check("注入风险概率已计算", bool(inj_probs), str(inj_probs))
    else:
        check("无埋深时不编造注入风险概率", not inj_probs,
              str((uqs.get("injection") or {}).get("text") or "（未计算）"))
    # 报告里的概率性结论由 Prompt 约束 + 代码兜底共同保证：
    # 模型漏写时 aggregate 会追加"置信区间"那句，所以这里可以硬断言。
    check("报告写入了概率性结论（含置信区间）",
          "置信区间" in report or "90%" in report)

    # ─── 时间维度记忆（工作流实测）─────────────────────
    tstats = final_state.get("temporal_stats") or {}
    check("工作流记录了时间维度统计",
          isinstance(tstats, dict) and "claims" in tstats, str(tstats))
    check("时间冲突字段存在且为列表",
          isinstance(final_state.get("temporal_conflicts"), list),
          str(type(final_state.get("temporal_conflicts")).__name__))

    # ─── Step 4: 新特性断言 ───────────────────────────
    print("\n[Step 4] 新特性断言（多模型路由 + 文件记忆）")

    routing = final_state.get("model_routing", {})
    check("运行时记录了模型路由", len(routing) >= 5, str(routing))
    # 回归：retrieve 节点曾用空字典覆盖 model_routing，把 extract 那条抹掉
    # （LangGraph 对 dict 字段是替换语义）。日志一切正常，只有审计字段丢信息。
    check("路由快照未被后续节点清空（含 extract）",
          "extract" in routing, str(list(routing)))
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

    # ─── Step 4-B: 动态路由与上下文压缩（工作流实测）────
    print("\n[Step 4-B] Token 经济学：动态路由 + 上下文压缩（工作流实测）")
    try:
        from model_router import LOCAL_MODE_LABEL, CLOUD_MODE_LABEL

        complexity = final_state.get("complexity_info") or {}
        check("工作流记录了复杂度判定", bool(complexity), str(complexity.get("mode")))
        check("判定带有明确日志标签",
              complexity.get("label") in (LOCAL_MODE_LABEL, CLOUD_MODE_LABEL),
              complexity.get("label", "-"))
        check("判定理由可解释",
              bool(complexity.get("reason")) and len(complexity.get("reasons") or []) >= 2,
              complexity.get("reason", "-"))

        # 本次测试输入约 40 字（< 50），但知识库确实命中 → 应升级到云端。
        # 这条断言验证的正是"短输入 + RAG 命中 → 唤醒云端"这条规则。
        check("短输入 + RAG 命中 → 升级到云端",
              complexity.get("mode") == "cloud",
              f"chars={complexity.get('chars')} "
              f"top_score={complexity.get('top_score')} → {complexity.get('mode')}")

        # report 的实际档位必须与判定一致，否则"判定"只是日志装饰
        report_tier = routing.get("report", "").split(":")[0]
        check("report 实际档位与判定一致",
              report_tier == complexity.get("mode"),
              f"判定={complexity.get('mode')} 实际={report_tier}")

        pruning = final_state.get("context_pruning") or {}
        check("工作流记录了上下文压缩状态", bool(pruning), str(pruning.get("status")))
        check("压缩状态合法", pruning.get("status") in ("pruned", "skipped", "failed"),
              str(pruning.get("status")))
        # 实测 Top-2 只有 600~960 字，所以真实运行必然是 skipped；
        # 若哪天变成 pruned，说明语料变长到确实需要压缩了 —— 同样算通过。
        check("未超阈值时不浪费一次压缩调用",
              pruning.get("status") == "pruned"
              or "阈值" in (pruning.get("reason") or ""),
              f"{pruning.get('before_chars')} 字 / {pruning.get('reason')}")
        if pruning.get("status") == "pruned":
            check("压缩后字符数下降",
                  pruning["after_chars"] < pruning["before_chars"],
                  f"{pruning['before_chars']} → {pruning['after_chars']}")
    except Exception as e:
        check("动态路由与上下文压缩（工作流实测）", False, str(e))

    # ─── Step 4-C: 自我进化的经验沉淀 ─────────────────
    # 只验证"沉淀"这一环：每次评估结束必须留下一条经验，否则反思永远不会被
    # 触发，而"没沉淀"和"还没攒够 20 条"在日志里长得一样。
    # 反思本身需要 20 条，不适合放进单次 E2E；它由 evolution_engine --simulate 覆盖。
    print("\n[Step 4-C] 自我进化：经验沉淀")
    try:
        from utils.experience_memory import EXPERIENCES, count_experiences, read_experiences

        exp_path = os.path.join(os.environ["EVOLVE_HOME"], EXPERIENCES)
        n = count_experiences(exp_path)
        check("评估结束后写入了经验文件", n >= 1, f"{exp_path}（{n} 条）")
        if n:
            last = read_experiences(exp_path)[-1]
            check("经验含评分与检索质量", "score" in last and "retrieval" in last,
                  f"score={last.get('score')} top={last['retrieval'].get('top_score')}")
            check("经验不含报告全文（避免上下文膨胀）", "final_report" not in last)
        check("E2E 未写入真实进化库",
              os.path.abspath(os.environ["EVOLVE_HOME"]).startswith(tempfile.gettempdir()),
              os.environ["EVOLVE_HOME"])
    except Exception as e:
        check("自我进化经验沉淀", False, str(e))

    # ─── Step 5: 图表生成 ─────────────────────────────
    print("\n[Step 5] 图表生成（Function Calling）")
    chart = final_state.get("chart_info") or {}
    check("工作流产出了图表信息", bool(chart), list(chart)[:4] if chart else "缺失")
    if chart:
        cpath = chart.get("path") or ""
        check("图表文件真实存在", os.path.exists(cpath), cpath)
        if os.path.exists(cpath):
            size = os.path.getsize(cpath)
            check("图表文件大小合理（非空图）", size > 15000, f"{size/1024:.0f} KB")
            with open(cpath, "rb") as f:
                check("是合法 PNG", f.read(8) == b"\x89PNG\r\n\x1a\n")
        check("图表带工程结论", bool(chart.get("summary")), (chart.get("summary") or "")[:48])
        check("结论在三种判定内", chart.get("verdict") in ("ok", "risky", "under"),
              str(chart.get("verdict")))
        check("图表有 URL 路径（可静态访问）",
              str(chart.get("url_path", "")).startswith("/charts/"),
              chart.get("url_path", "-"))
        # 报告末尾必须带 Markdown 图片引用
        check("报告含 Markdown 图片引用",
              "![地层压力随深度变化曲线]" in report,
              chart.get("rel_path", "-"))

    # ─── Step 5-B: PINN 工具（经 MCP，端到端）─────────
    print("\n[Step 5-B] PINN 压力预测（MCP 工具，端到端约 1-2 分钟）")
    step_pinn()

    # ─── Step 5-C: 不确定性量化（蒙特卡洛）────────────
    print("\n[Step 5-C] 不确定性量化（蒙特卡洛，100 次）")
    try:
        from core.uq import uq_block
        from tools.physics_tools import (
            calculate_co2_density,
            co2_density_uq,
            injection_risk_uq,
        )

        text = calculate_co2_density(250, 80, seed=42)
        check("密度输出含 90% 置信区间", "90% 置信区间" in text, text)
        check("密度输出含模拟次数", "100 次蒙特卡洛" in text)

        uq = co2_density_uq(250, 80, seed=42, keep_samples=True)
        check("置信区间包住均值", uq["p5"] < uq["mean"] < uq["p95"],
              f"{uq['p5']:.3f} < {uq['mean']:.3f} < {uq['p95']:.3f}")
        check("生成了 100 个样本", len(uq.get("samples") or []) == 100,
              f"{len(uq.get('samples') or [])} 个")
        check("同种子可复现",
              co2_density_uq(250, 80, seed=42)["mean"] == uq["mean"])
        check("区间宽度与 ±5% 扰动同量级",
              0.04 < (uq["p95"] - uq["p5"]) / uq["mean"] < 0.11,
              f"{(uq['p95'] - uq['p5']) / uq['mean']:.3f}")

        inj = injection_risk_uq(2500, 300, seed=42)
        check("注入风险概率归一",
              abs(sum(inj["probabilities"].values()) - 1.0) < 1e-9,
              str(inj["probabilities"]))
        check("注入风险文案带概率", "%" in inj["text"], inj["text"])
        check("远超破裂压力时判定为压裂风险",
              injection_risk_uq(2500, 600, seed=42)["probabilities"].get("risky") == 1.0)

        blk = uq_block({"density": uq, "injection": inj})
        check("报告 Prompt 注入概率性硬约束", "禁止只写单一确定值" in blk)
    except Exception as e:
        check("UQ 模块可用", False, f"{type(e).__name__}: {e}")

    # ─── Step 6: 流式输出 ─────────────────────────────
    print("\n[Step 6] 流式输出（SSE token 级推送）")
    try:
        import json as _json
        import requests as _rq

        # 前置：确保 8000 端口上有服务（外部已有则复用，否则进程内起一个）
        ensure_api_server(check)

        tokens, node_names, chart_events = [], [], []
        done_ev = None
        first_token_at = None
        t_start = time.time()

        with _rq.post(
            f"{API_BASE}/api/analyze",
            json={"user_input": "某CCUS项目埋深2000m，注入压力250bar，温度80℃，请评估风险。"},
            stream=True, timeout=(5, 600),
            headers={"Accept": "text/event-stream"},
        ) as resp:
            check("SSE 端点返回 200", resp.status_code == 200, f"HTTP {resp.status_code}")
            check("Content-Type 是 event-stream",
                  "text/event-stream" in (resp.headers.get("content-type") or ""),
                  resp.headers.get("content-type", "-"))
            for raw in resp.iter_lines(decode_unicode=True):
                if not raw or not raw.startswith("data:"):
                    continue
                ev = _json.loads(raw[5:].strip())
                et = ev.get("type")
                if et == "token":
                    if first_token_at is None:
                        first_token_at = time.time() - t_start
                    tokens.append(ev.get("text", ""))
                elif et == "node" and ev.get("status") == "start":
                    node_names.append(ev.get("node"))
                elif et == "chart":
                    chart_events.append(ev.get("chart") or {})
                elif et == "done":
                    done_ev = ev

        # 注意：这里**不**关服务 —— Step 8 的人工审核还要用它，最后统一收尾

        check("收到 token 流", len(tokens) > 100, f"{len(tokens)} 个 token")
        streamed = "".join(tokens)
        check("流式文本长度合理", len(streamed) > 300, f"{len(streamed)} 字")
        check("token 是分批到达的（真流式）", len(tokens) > 50,
              f"{len(tokens)} 批，若为 1 则说明被缓冲了")
        check("首 token 延迟可测", first_token_at is not None,
              f"{first_token_at:.1f}s" if first_token_at else "无")
        check("推送了节点进度事件", len(node_names) >= 4, " → ".join(node_names))
        check("流式过程中推送了图表", bool(chart_events),
              (chart_events[0].get("rel_path") if chart_events else "-"))
        check("收到 done 事件", done_ev is not None)
        if done_ev:
            check("done 带评审分", done_ev.get("score") is not None, str(done_ev.get("score")))
            check("done 带路由信息", bool(done_ev.get("model_routing")),
                  str(list((done_ev.get("model_routing") or {}).keys())))
            check("done 的报告含密度值",
                  "kg/m" in (done_ev.get("report") or ""))
            # 流式正文不应出现原始工具 JSON（会以机器指令形式暴露给用户）
            check("流式正文不含裸 JSON 工具块",
                  '"tool"' not in streamed and "```json" not in streamed)
    except Exception as e:
        check("流式输出（SSE）", False, str(e))

    # ─── Step 7: 成本仪表盘 ───────────────────────────
    print("\n[Step 7] 成本仪表盘（Token Cost Dashboard）")
    try:
        from utils.cost_model import PRICING_CNY_PER_MTOK, resolve_pricing, format_cny
        from workflow import compute_cost_estimation

        est = (done_ev or {}).get("cost_estimation") or {}
        check("done 事件带 cost_estimation", bool(est),
              str(list(est)[:5]) if est else "缺失")
        check("币种为人民币", est.get("currency") == "CNY", str(est.get("currency")))
        check("有阶段被计价", est.get("stages_priced", 0) >= 3,
              f"{est.get('stages_priced')} 个阶段")

        # ─── 逐阶段复算：金额必须等于 token × 单价 ───
        recompute_ok = True
        bad_rows = []
        for row in est.get("breakdown") or []:
            unit, _key, _fb = resolve_pricing(row["model"], row["tier"])
            want = (row["input_tokens"] / 1_000_000 * unit["input"]
                    + row["output_tokens"] / 1_000_000 * unit["output"])
            if abs(want - row["cny"]) > 1e-7:
                recompute_ok = False
                bad_rows.append(f"{row['stage']}: 期望 {want:.8f} 实得 {row['cny']:.8f}")
        check("每个阶段的金额可用 token×单价复算", recompute_ok,
              "；".join(bad_rows) or f"{len(est.get('breakdown') or [])} 行全部一致")

        total_from_rows = sum(r["cny"] for r in (est.get("breakdown") or []))
        check("总额等于各阶段之和",
              abs(total_from_rows - est.get("total_cny", -1)) < 1e-7,
              f"{total_from_rows:.8f} vs {est.get('total_cny'):.8f}")

        # ─── 基线复算：全部 token 按 baseline_model 列表价 ───
        base_unit, _k, _f = resolve_pricing(est.get("baseline_model", ""), "cloud")
        want_base = sum(
            r["input_tokens"] / 1_000_000 * base_unit["input"]
            + r["output_tokens"] / 1_000_000 * base_unit["output"]
            for r in (est.get("breakdown") or [])
        )
        check("全云端基线可复算",
              abs(want_base - est.get("baseline_cny", -1)) < 1e-7,
              f"{want_base:.8f} vs {est.get('baseline_cny'):.8f}")
        check("节省额 = 基线 - 实际（不为负）",
              est.get("saved_cny", -1) >= 0
              and abs(est.get("saved_cny", -1)
                      - max(0.0, est.get("baseline_cny", 0) - est.get("total_cny", 0))) < 1e-7,
              f"节省 ¥{est.get('saved_cny'):.6f}")

        # ─── 前端文案 ───
        caption = est.get("caption") or ""
        check("生成了前端文案", caption.startswith("本次评估预估成本：¥"), caption)
        check("文案含节省金额",
              ("已为你节省 ¥" in caption) or est.get("saved_cny", 0) <= 0, caption)
        check("文案金额与实际成本一致",
              format_cny(est.get("total_cny", 0)) in caption, caption)

        # ─── 本地与云端混合计价的真实性 ───
        tiers = {r["tier"] for r in (est.get("breakdown") or [])}
        check("本地阶段按电费计价（非零）",
              any(r["cny"] > 0 for r in (est.get("breakdown") or [])
                  if r["tier"] == "local"),
              str({r["stage"] for r in (est.get("breakdown") or [])
                   if r["tier"] == "local"}))
        check("同时存在本地与云端阶段（路由真的分流了）",
              len(tiers) >= 2 or est.get("stages_priced", 0) <= 2, str(tiers))

        # ─── 同步路径共用同一份核算逻辑 ───
        est_sync = compute_cost_estimation(final_state)
        check("同步路径（compute_cost_estimation）可算出成本",
              est_sync.get("stages_priced", 0) >= 3,
              f"¥{est_sync.get('total_cny'):.6f} / {est_sync.get('stages_priced')} 阶段")
        check("两条路径的成本口径一致",
              abs(est_sync.get("total_cny", 0) - est.get("total_cny", 0))
              < max(1e-6, est.get("total_cny", 0) * 0.5),
              f"sync ¥{est_sync.get('total_cny'):.6f} vs sse ¥{est.get('total_cny'):.6f}")

        # ─── API 契约：两个端点都必须返回该字段 ───
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "api_server.py"), encoding="utf-8") as f:
            api_src = f.read()
        check("同步端点返回 cost_estimation",
              '"cost_estimation": compute_cost_estimation(final_state)' in api_src)
        check("流式端点返回 cost_estimation",
              '"cost_estimation": compute_cost_estimation(node_outputs)' in api_src)
    except Exception as e:
        check("成本仪表盘", False, str(e))

    # ─── Step 7-B: Bearer 鉴权 + 真实健康检查 ──────────
    print("\n[Step 7-B] Bearer Token 鉴权 + 健康检查探针")
    try:
        import requests as _rq

        import api_server as _api_mod

        base = API_BASE

        # ── A. 健康检查报告的是依赖，不是"进程还活着" ──
        h = _rq.get(f"{base}/health", timeout=30)
        check("health 返回 200（依赖挂了也不该 503）", h.status_code == 200,
              f"HTTP {h.status_code}")
        hbody = h.json()
        hchecks = hbody.get("checks") or {}
        check("health 覆盖 ollama / chroma / sqlite",
              {"ollama", "chroma", "sqlite"} <= set(hchecks))
        for dep in ("ollama", "chroma", "sqlite"):
            c = hchecks.get(dep) or {}
            check(f"health.{dep} 真实连通", bool(c.get("ok")),
                  str(c.get("detail"))[:70])
            check(f"health.{dep} 带实测耗时",
                  isinstance(c.get("latency_ms"), int), f"{c.get('latency_ms')} ms")
        # 本地 .env 没配 token → 鉴权关闭，且必须如实上报，不能假装已保护
        check("未配置 token 时 health 上报 auth=disabled",
              hbody.get("auth") == "disabled", str(hbody.get("auth")))

        # ── B. 打开鉴权：改模块级全局，无需重启服务 ──
        # require_token 每次调用都读 api_server.API_TOKEN，所以改属性等于
        # 运行期切配置。比"另起一个带 token 的进程"省一份 3.95 GB 的 reranker。
        _api_mod.API_TOKEN = "e2e-secret-token"
        try:
            h2 = _rq.get(f"{base}/health", timeout=30)
            check("开启鉴权后 health 仍免 token（探针不带业务凭证）",
                  h2.status_code == 200 and h2.json().get("auth") == "enabled",
                  f"HTTP {h2.status_code} auth={h2.json().get('auth')}")

            no_tok = _rq.get(f"{base}/api/review/e2e-probe", timeout=30)
            check("无 token 访问 /api/* 返回 401", no_tok.status_code == 401,
                  f"HTTP {no_tok.status_code}")
            check("401 带 WWW-Authenticate: Bearer",
                  "bearer" in (no_tok.headers.get("WWW-Authenticate") or "").lower(),
                  str(no_tok.headers.get("WWW-Authenticate")))

            wrong = _rq.get(f"{base}/api/review/e2e-probe",
                            headers={"Authorization": "Bearer wrong-token"}, timeout=30)
            check("错误 token 返回 401", wrong.status_code == 401,
                  f"HTTP {wrong.status_code}")

            basic = _rq.get(f"{base}/api/review/e2e-probe",
                            headers={"Authorization": "Basic dXNlcjpwYXNz"}, timeout=30)
            check("非 Bearer 方案返回 401", basic.status_code == 401,
                  f"HTTP {basic.status_code}")

            good_tok = _rq.get(f"{base}/api/review/e2e-probe",
                               headers={"Authorization": "Bearer e2e-secret-token"},
                               timeout=30)
            # 放行的证据不是 200，而是 404 —— 这个 thread 本来就不存在。
            # 只有 404 才同时证明"鉴权没拦"且"确实进了路由"。
            check("正确 token 放行（进到路由 → 404 而非 401）",
                  good_tok.status_code == 404, f"HTTP {good_tok.status_code}")

            spec = _rq.get(f"{base}/openapi.json", timeout=30)
            schemes = (spec.json().get("components") or {}).get("securitySchemes") or {}
            check("OpenAPI 声明 Bearer 方案（/docs 的 Authorize 按钮依赖它）",
                  spec.status_code == 200
                  and any("/api/analyze" in p for p in (spec.json().get("paths") or {}))
                  and bool(schemes),
                  f"HTTP {spec.status_code} schemes={list(schemes)}")
        finally:
            _api_mod.API_TOKEN = ""

        after = _rq.get(f"{base}/api/review/e2e-probe", timeout=30)
        check("撤掉 token 后恢复放行（配置是唯一开关）",
              after.status_code == 404, f"HTTP {after.status_code}")
    except Exception as e:
        check("Bearer 鉴权 / 健康检查", False, f"{type(e).__name__}: {e}")

    # ─── Step 8: 物理一致性校验 + 人工审核（HITL）───────
    print("\n[Step 8] 物理一致性校验 + 人工审核（HITL）")
    try:
        import requests as _rq
        from utils.physics_rules import evaluate_physics

        # ─── A. 规则层：纯函数，不花一分钱 ───
        # 这一段的意义是"确定性可验证"：同样的输入永远得到同样的判定，
        # 不依赖模型的措辞偏好。
        bad = evaluate_physics(
            params_text="埋深1500m，注入压力250bar，温度80℃",
            report_text="本项目管理规范，风险可控，注入作业安全。",
        )
        check("规则层拦住『超破裂压力却称安全』", bad["blocking"], bad["summary"])
        check("拦截理由指向破裂压力超限",
              "P_INJ_ABOVE_FRACTURE_UNACKNOWLEDGED"
              in [v["code"] for v in bad["violations"]])
        check("打回时给出可执行的修正数值",
              "≤ 227" in bad["feedback"] and "≥ 1649" in bad["feedback"])

        good = evaluate_physics(
            params_text="埋深1500m，注入压力250bar，温度80℃",
            report_text="注入压力 250 bar 已超过 1500 m 处地层破裂压力 227 bar，"
                        "存在盖层压裂与 CO2 泄漏风险。",
        )
        check("报告点明风险后不再拦截", not good["blocking"], good["status"])

        smooth = evaluate_physics(
            params_text="埋深2500m，注入压力250bar，温度80℃",
            report_text="注入顺畅，注入压力充足。",
        )
        check("规则层拦住『注不进却称注入顺畅』", smooth["blocking"], smooth["verdict"])

        no_depth = evaluate_physics(params_text="注入压力250bar", report_text="安全。")
        check("缺深度时判为『无法验证』而非放行或误拦",
              no_depth["status"] == "unverifiable" and not no_depth["blocking"])

        # ─── B. HITL：HTTP 往返（暂停 → 驳回重写 → 通过落盘）───
        base = API_BASE
        r = _rq.post(
            f"{base}/api/analyze/hitl",
            json={"user_input": "某CCUS项目埋深2000m，注入压力250bar，温度80℃，请评估风险。"},
            timeout=(5, 600),
        )
        check("HITL 端点返回 200", r.status_code == 200, f"HTTP {r.status_code}")
        first = r.json()
        tid = first.get("thread_id", "")
        check("HITL 在人工闸门前暂停",
              first.get("status") == "awaiting_approval", str(first.get("status")))
        check("暂停时已产出完整报告",
              len(first.get("report") or "") > 200,
              f"{len(first.get('report') or '')} 字")
        check("暂停时带物理校验结论", bool(first.get("physics_check")),
              str((first.get("physics_check") or {}).get("status")))
        check("human_review 标记为待确认",
              (first.get("human_review") or {}).get("approved") is None
              and (first.get("human_review") or {}).get("auto") is False,
              str(first.get("human_review")))
        check("审核前不写入历史（未落盘）", not first.get("history_entry"))

        rv = _rq.get(f"{base}/api/review/{tid}", timeout=30)
        check("review 接口可查到待审状态",
              rv.status_code == 200
              and rv.json().get("status") == "awaiting_approval",
              f"HTTP {rv.status_code}")

        rj = _rq.post(
            f"{base}/api/approve",
            json={"thread_id": tid, "approved": False, "reviewer": "e2e",
                  "comment": "请补充断层活化对盖层完整性的定量影响。"},
            timeout=(5, 600),
        )
        check("驳回请求返回 200", rj.status_code == 200, f"HTTP {rj.status_code}")
        rejected = rj.json()
        check("驳回后带意见重写并再次停在闸门前",
              rejected.get("status") == "awaiting_approval",
              str(rejected.get("status")))
        check("驳回次数被记录",
              (rejected.get("human_review") or {}).get("rejections") == 1,
              str((rejected.get("human_review") or {}).get("rejections")))
        check("重写后的报告与首版不同",
              (rejected.get("report") or "") != (first.get("report") or ""))

        ap = _rq.post(
            f"{base}/api/approve",
            json={"thread_id": tid, "approved": True, "reviewer": "e2e",
                  "comment": "确认风险结论。"},
            timeout=(5, 600),
        )
        check("通过请求返回 200", ap.status_code == 200, f"HTTP {ap.status_code}")
        done = ap.json()
        check("通过后流程结束", done.get("status") == "success", str(done.get("status")))
        check("human_review 记录为人工通过（auto=false）",
              (done.get("human_review") or {}).get("approved") is True
              and (done.get("human_review") or {}).get("auto") is False,
              str(done.get("human_review")))
        check("通过后才写入历史", bool(done.get("history_entry")))
        check("人工链路同样带成本核算",
              (done.get("cost_estimation") or {}).get("stages_priced", 0) >= 3,
              f"¥{(done.get('cost_estimation') or {}).get('total_cny', 0):.6f}")

        # ─── 错误路径 ───
        again = _rq.post(f"{base}/api/approve",
                         json={"thread_id": tid, "approved": True}, timeout=30)
        check("已结束的 thread 再 approve 返回 409",
              again.status_code == 409, f"HTTP {again.status_code}")
        missing = _rq.get(f"{base}/api/review/does_not_exist", timeout=30)
        check("未知 thread 返回 404", missing.status_code == 404,
              f"HTTP {missing.status_code}")
    except Exception as e:
        check("物理校验 / 人工审核（HITL）", False, str(e))
    finally:
        shutdown_api_server()

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
