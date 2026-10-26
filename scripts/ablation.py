"""
消融对照：每个模块到底贡献了什么
================================
项目里所有「更好」原本都是零对照的断言 —— 类型约束更好、别名合并更好、
物理闸门更好、Self-RAG 更好。没有对照，这些话无法证伪，也就不该出现在
给教授的材料里。本脚本把每个模块单独关掉，量出它的贡献。

两类对照，区别必须说清楚
------------------------
**离线对照**（不需要模型，几秒）：作用在**已建好的图**上，量的是
"这个后处理步骤改变了什么"。因为图已经固定，所以是精确的因果，不是抽样。

**在线对照**（需要 Ollama，每例十几分钟）：作用在**生成过程**上，量的是
"关掉这个模块，报告会变成什么样"。样本小、有随机性，只能当诊断，不能当基准。

两类的可信度不同，所以分开报，不合并成一张表。

用法::

    python scripts/ablation.py --offline        # 秒级，跑离线对照
    python scripts/ablation.py --online --cases 2 --arm no_physics_gate
    python scripts/ablation.py --write          # 结果写 eval/ablation.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _isolate_history() -> str:
    """把项目历史重定向到 ``.ablation/``，返回目标路径。

    **能早设就早设**：``file_memory`` 在 import 时就把历史路径固化成模块常量，
    之后再设环境变量对它无效。这个坑真的踩了 —— 探针问句被写进
    ``project_history.md``，压缩逻辑按「最近 N 条」把它留下、把 120 条真实历史
    挤掉。测试问句混进个人历史是不可逆的。

    所以这里做两件事：设环境变量（管尚未导入的模块），以及直接改已导入模块
    的常量（管已经导入的）。只做前一件不够。
    """
    path = os.path.join(_ROOT, ".ablation", "project_history.md")
    os.environ["PROJECT_HISTORY_PATH"] = path
    loaded = sys.modules.get("file_memory")
    if loaded is not None:
        loaded.PROJECT_HISTORY_PATH = path
    return path


GRAPHML = os.path.join(_ROOT, "lightrag_storage", "graph_chunk_entity_relation.graphml")
OUT_JSON = os.path.join(_ROOT, "eval", "ablation.json")

# 探针用例：为**触发阻断**而构造，不参与 golden set，也不用于算分数。
#
# 为什么必须有它：golden set 的 10 题没有一题给出断层倾角，也没有一题给出
# CO₂ 柱高或注入速率。而 ``physics_deep`` 的阻断策略写得很清楚 —— 只有这两类
# 输入才允许阻断，其余一律只给信息。所以在 golden set 上关掉阻断能力，
# 结果**必然**与基线逐字相同。那不是"物理闸门没有贡献"，是这套题根本
# 测不出它的贡献。拿它当证据会得到一个看起来像发现、实际是空实验的结论。
#
# 这一题离线验证过：2500 m / 420 bar / 倾角 60° → 滑移比 2.199、阻断成立。
PROBE_CASES = [
    {
        "id": "probe-fault-blocking",
        "question": ("某盐水层封存场地，埋深 2500 m，注入压力 420 bar，"
                     "存在一条倾角 60° 的构造断层。注入是否会活化该断层？"
                     "应如何调整注入方案？"),
    },
]

# 与 utils/kg_prompt.py 的 ---Allowed Entity Types--- 一致。
# 这八类是**提示词**层面的约束，不是代码里的白名单 —— 所以图里仍然会有
# 违反它的节点，下面要量的正是这个缺口。
ALLOWED_TYPES = {
    "geologicalbody", "wellborecomponent", "material", "process",
    "technology", "parameter", "risk", "regulation",
}


# ─── 离线对照 ────────────────────────────────────────────────


def _summarise(nodes, edges) -> dict:
    from utils.graph_analysis import connected_components

    comps = connected_components(nodes, edges)
    total = len(nodes) or 1
    largest = len(comps[0]) if comps else 0
    isolated = sum(1 for c in comps if len(c) == 1)
    return {
        "nodes": len(nodes),
        "edges": len(edges),
        "components": len(comps),
        "largest": largest,
        "largest_ratio": round(largest / total, 4),
        "isolated": isolated,
        "isolated_ratio": round(isolated / total, 4),
    }


def arm_alias_merge(nodes, edges, attrs) -> dict:
    """关掉别名合并，只剩噪声删除，看连通性差多少。

    别名合并的作用一直被引用（"91 → 84 分量"），但那是 12 篇论文时的小图。
    在现在这张图上重算，才知道它还值不值这几百行规则。
    """
    from normalize_graph import project
    from utils.entity_aliases import build_merge_plan

    plan = build_merge_plan(nodes, attrs)
    merged = project(plan, nodes, edges)

    # 同一份噪声计划，但把合并全部清空 —— 唯一变量就是别名合并
    no_merge_plan = {"merges": [], "noise": plan["noise"], "kept": plan["kept"]}
    unmerged = project(no_merge_plan, nodes, edges)

    return {
        "arm": "alias_merge",
        "question": "别名合并对连通性的贡献",
        "with_merge": merged,
        "without_merge": unmerged,
        "delta": {
            "components": unmerged["components"] - merged["components"],
            "largest": merged["largest"] - unmerged["largest"],
            "nodes_merged": unmerged["nodes"] - merged["nodes"],
        },
        "verdict": (
            f"合并 {unmerged['nodes'] - merged['nodes']} 个节点，"
            f"分量 {unmerged['components']} → {merged['components']}，"
            f"最大分量 {unmerged['largest']} → {merged['largest']}"
        ),
    }


def arm_type_constraint(nodes, edges, attrs) -> dict:
    """八类约束是提示词，不是代码闸门 —— 量它漏了多少。

    这里量的是**上界**：图里已经存在的不合规节点。真正的对照需要重新抽取
    （换个类型表跑一遍建图，几小时），所以这个数字只能说明"约束没被完全遵守"，
    不能说明"约束没有用"。
    """
    from utils.graph_analysis import connected_components

    types = {n: (attrs.get(n, {}).get("entity_type") or "").strip().lower() for n in nodes}
    bad = [n for n, t in types.items() if t not in ALLOWED_TYPES]
    bad_set = set(bad)
    bad_edges = [(a, b) for a, b in edges if a in bad_set or b in bad_set]
    total = len(nodes) or 1

    # 去掉不合规节点之后，图会变成什么样
    kept = [n for n in nodes if n not in bad_set]
    kept_set = set(kept)
    kept_edges = [(a, b) for a, b in edges if a in kept_set and b in kept_set]
    after = _summarise(kept, kept_edges)
    before = _summarise(nodes, edges)

    dist = Counter(t or "<empty>" for n, t in types.items() if t not in ALLOWED_TYPES)
    return {
        "arm": "type_constraint",
        "question": "八类实体约束的遵守率（上界，非因果）",
        "off_schema_nodes": len(bad),
        "off_schema_ratio": round(len(bad) / total, 4),
        "off_schema_edges_touching": len(bad_edges),
        "distribution": dict(dist.most_common()),
        "graph_if_enforced": after,
        "graph_now": before,
        "verdict": (
            f"{len(bad)} 个节点（{len(bad) / total:.1%}）不在八类之内，"
            f"牵动 {len(bad_edges)} 条边；约束是提示词，没有代码闸门"
        ),
    }


def arm_ontology(nodes, edges, attrs) -> dict:
    """策展本体对薄维度与连通性的贡献（已在 augment_graph 里实现，这里复用）。"""
    from scripts.augment_graph import load_ontology, merge
    from utils.graph_analysis import domain_coverage

    onto = load_ontology()
    before = domain_coverage(
        [n for n in nodes if attrs.get(n, {}).get("file_path") != "domain_ontology"],
        edges, attrs,
    )
    merged = merge(nodes, edges, attrs, onto)
    after = domain_coverage(merged["nodes"], merged["edges"], merged["attrs"],
                            merged["overrides"])
    dims = {}
    for key, d in after["dimensions"].items():
        prev = before["dimensions"][key]["entities"]
        dims[key] = {"before": prev, "after": d["entities"],
                     "ratio": round(d["entities"] / prev, 2) if prev else None}
    return {
        "arm": "curated_ontology",
        "question": "策展本体对六个维度的贡献",
        "dimensions": dims,
        "verdict": " · ".join(
            f"{k} {v['before']}→{v['after']}" for k, v in dims.items()
        ),
    }


def run_offline() -> dict:
    from utils.graph_analysis import load_graph

    nodes, edges, attrs = load_graph(GRAPHML)
    arms = [
        arm_alias_merge(nodes, edges, attrs),
        arm_type_constraint(nodes, edges, attrs),
        arm_ontology(nodes, edges, attrs),
    ]
    return {"graph": GRAPHML, "nodes": len(nodes), "edges": len(edges), "arms": arms}


def render_offline(result: dict) -> str:
    lines = [
        f"离线消融（图 {result['nodes']} 节点 / {result['edges']} 边）",
        "═" * 78,
    ]
    for arm in result["arms"]:
        lines.append(f"· {arm['question']}")
        lines.append(f"  {arm['verdict']}")
        if arm["arm"] == "alias_merge":
            for label, key in (("开", "with_merge"), ("关", "without_merge")):
                d = arm[key]
                lines.append(
                    f"    {label}: {d['nodes']} 节点 / {d['components']} 分量 / "
                    f"最大 {d['largest']}（{d['largest_ratio']:.1%}）/ "
                    f"孤立 {d['isolated']}（{d['isolated_ratio']:.1%}）"
                )
        elif arm["arm"] == "type_constraint":
            top = list(arm["distribution"].items())[:5]
            lines.append("    分布: " + " · ".join(f"{k}={v}" for k, v in top))
            now, aft = arm["graph_now"], arm["graph_if_enforced"]
            lines.append(
                f"    若严格执行: {now['components']} → {aft['components']} 分量，"
                f"最大分量 {now['largest']} → {aft['largest']}"
            )
        lines.append("")
    return "\n".join(lines)


def render_markdown(payload: dict) -> str:
    """
    把 ``eval/ablation.json`` 渲染成一份能直接给人看的短报告。

    为什么要单独一份 Markdown：JSON 是给脚本的，但"这个模块到底有没有用"
    这个问题，我自己半年后回来看也要先读一遍结构。报告里的每个数字都来自
    payload，**没有一句是手写的** —— 手写结论与自动数据一旦分叉，谁也不会
    发现（这个仓库已经因为手写表格吃过一次 1.6% 的亏）。

    ``payload`` 可以只含 ``offline`` 或 ``online`` 之一：两者触发条件不同
    （离线不需要模型），分开跑、分开写是常态。
    """
    lines: list[str] = [
        "# 消融对照结果",
        "",
        "> 由 `scripts/ablation.py --report` 生成，数字全部来自 `eval/ablation.json`。",
        "> 这份报告回答的是「关掉某个模块之后，什么变了」，**不是**「系统有多准」。",
        "",
    ]

    offline = payload.get("offline") or {}
    if offline:
        lines += [
            "## 离线对照（不需要模型）",
            "",
            f"图：{offline.get('nodes')} 节点 / {offline.get('edges')} 边。",
            "",
            "| 对照项 | 问题 | 判读 |",
            "|---|---|---|",
        ]
        for arm in offline.get("arms") or []:
            lines.append(
                f"| `{arm.get('arm')}` | {arm.get('question', '')} | "
                f"{arm.get('verdict', '')} |"
            )
        lines.append("")

        # 别名合并是唯一有"开/关两套完整指标"的对照，单独摊开，
        # 因为它是这几个模块里唯一能被量化成"连通性变化"的。
        for arm in offline.get("arms") or []:
            if arm.get("arm") != "alias_merge":
                continue
            lines += [
                "### 别名合并的连通性影响",
                "",
                "| 口径 | 节点 | 边 | 连通分量 | 最大分量 | 占比 | 孤立节点 |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
            for label, key in (("合并后（当前）", "with_merge"), ("不合并", "without_merge")):
                d = arm.get(key) or {}
                lines.append(
                    f"| {label} | {d.get('nodes')} | {d.get('edges')} | "
                    f"{d.get('components')} | {d.get('largest')} | "
                    f"{d.get('largest_ratio', 0):.1%} | {d.get('isolated')} |"
                )
            delta = arm.get("delta") or {}
            lines += [
                "",
                f"差：分量 {delta.get('components')}、最大分量 {delta.get('largest')}、"
                f"合并节点 {delta.get('nodes_merged')}。",
                "",
            ]

    online = payload.get("online") or {}
    if online:
        lines += [
            "## 在线对照（需要 Ollama）",
            "",
            "| 组 | arm | 用例集 | 用例数 | 通过 | 失败 | 闸门状态 | 物理状态 | 重写次数 |",
            "|---|---|---|---:|---:|---:|---|---|---:|",
        ]
        for tag, group in sorted(online.items()):
            rows = group.get("results") or []
            if not rows:
                lines.append(
                    f"| `{tag}` | `{group.get('arm')}` | {group.get('case_set', '')} | "
                    f"{group.get('cases')} | {group.get('passed')} | "
                    f"{group.get('failed')} | — | — | — |"
                )
                continue
            for row in rows:
                lines.append(
                    f"| `{tag}` | `{group.get('arm')}` | "
                    f"{group.get('case_set', '')} | {group.get('cases')} | "
                    f"{group.get('passed')} | {group.get('failed')} | "
                    f"{row.get('gate_status', '')} | {row.get('physics_status', '')} | "
                    f"{row.get('retry_count', '')} |"
                )
        notes = {g.get("note") for g in online.values() if g.get("note")}
        envs = {g.get("environment") for g in online.values() if g.get("environment")}
        lines.append("")
        for env in sorted(envs):
            lines.append(f"环境：{env}")
        for note in sorted(notes):
            lines.append(f"\n注意：{note}")

    lines += [
        "",
        "## 复现",
        "",
        "```bash",
        "python scripts/ablation.py --offline --write          # 离线三组",
        "python scripts/ablation.py --online --probe --arm baseline --write",
        "python scripts/ablation.py --online --probe --arm no_physics_gate --write",
        "python scripts/ablation.py --online --probe --arm no_rewrite --write",
        "python scripts/ablation.py --report docs/ablation-report.md",
        "```",
        "",
    ]
    return "\n".join(lines)


def write_markdown(path: str, payload: dict) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    text = render_markdown(payload)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return text


# ─── 在线对照（需要 Ollama）─────────────────────────────────


def _apply_arm(arm: str) -> list[str]:
    """按名字关掉一个模块，返回实际改动了什么（要能被人核对）。"""
    changes: list[str] = []
    if arm == "baseline":
        return ["未改动任何模块"]

    if arm == "no_physics_gate":
        import utils.physics_deep as deep

        original = deep.assess

        def softened(*args, **kwargs):
            result = original(*args, **kwargs)
            if result.get("blocking"):
                changes.append(
                    "拦住但被放行: " + "；".join(result.get("blocking_reasons") or [])
                )
            result = dict(result)
            result["blocking"] = False
            result["blocking_reasons"] = []
            # policy 在 physics_deep 里是 list（每行一条策略说明），
            # 早期这里按字符串拼接，一旦真的拦下过东西就 TypeError ——
            # 而异常会被 run_online 记成"这一例出错"，看起来像环境问题。
            note = "（消融：阻断已关闭）"
            policy = result.get("policy")
            if isinstance(policy, list):
                result["policy"] = policy + [note]
            else:
                result["policy"] = (policy or "") + note
            return result

        deep.assess = softened
        changes.append("physics_deep.assess 的 blocking 恒为 False（数值保留）")

    elif arm == "no_rewrite":
        import core.gates as gates

        gates.MAX_REPORT_REWRITES = 0
        changes.append("core.gates.MAX_REPORT_REWRITES = 0（低分直接 fail，不重写）")

    else:
        raise SystemExit(f"未知的 arm：{arm}")

    return changes


def run_online(arm: str, cases_n: int, timeout: int, probe: bool = False) -> dict:
    """跑真实 Agent，记录**闸门结果**与报告长度。

    不在这里跑 RAGAS：判分要额外调模型，一例多十几分钟，而这一组对照
    要回答的问题不是"分数多少"，而是"关掉闸门后，本该被拦下的报告
    是不是直接出去了"。闸门状态是确定性字段，比分数更适合这个对照。

    三组必须跑在同一套模型上。``.env`` 里的 ``CLOUD_API_KEY`` 会把
    report / geology / wellbore 路由到云端端点；本机上那个端点不可达，
    于是每一例都以 ``APIConnectionError`` 收场，结果表显示
    「0 通过 / 0 失败」—— 读起来像"闸门一个都没拦住"，实际是模型没跑起来。
    这个坑踩过一次，所以直接复用 ``eval_ragas`` 的隔离函数（它同时把
    个人历史、图表、经验库都挪到临时目录，且处理了"core.config 已被导入"
    的情形）。
    """
    # 先隔离历史，再导入 eval_ragas —— 顺序反了，file_memory 可能已经被
    # 导入并把路径固化掉（见 _isolate_history 里的代价）。
    _isolate_history()
    import eval_ragas as er
    from nodes import aggregate_node

    isolated = er._prepare_isolated_agent_environment(allow_cloud=False, allow_web=False)
    cases = (PROBE_CASES if probe else er.load_golden_set(er.DEFAULT_GOLDEN_SET))[:cases_n]
    changes = _apply_arm(arm)

    # 与 eval_ragas 一致：不写个人记忆。少了这一步，消融问句会混进
    # project_history.md，之后再也分不清哪些是真实使用、哪些是测试。
    original_save = aggregate_node.save_memory
    aggregate_node.save_memory = lambda **_kwargs: None
    partial = os.path.join(
        _ROOT, ".ablation",
        f"online_{arm}{'_probe' if probe else ''}.jsonl",
    )
    os.makedirs(os.path.dirname(partial), exist_ok=True)
    if os.path.exists(partial):
        os.remove(partial)
    out = []
    try:
        from workflow import workflow as agent_graph

        app = agent_graph.compile()
        for index, case in enumerate(cases, start=1):
            print(f"[{index}/{len(cases)}] {arm}: {case['id']}", flush=True)
            try:
                state = er._invoke_with_watchdog(app, case, timeout)
                gate = state.get("quality_gate") or {}
                row = {
                    "case_id": case["id"],
                    "question": case["question"],
                    "gate_status": gate.get("status"),
                    "physics_status": gate.get("physics_status"),
                    "physics_blocking": gate.get("physics_blocking"),
                    "retry_count": state.get("retry_count"),
                    "response_chars": len(state.get("final_report") or ""),
                    "error": "",
                }
            except Exception as exc:  # 一例失败不能掩盖其他例
                row = {"case_id": case["id"],
                       "error": f"{type(exc).__name__}: {exc}"}
            out.append(row)
            # 逐例落盘：看门狗超时是**杀进程**（线程杀不掉），不落盘的话
            # 第 2 例挂掉会把第 1 例的结果一起带走。
            with open(partial, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    finally:
        aggregate_node.save_memory = original_save

    passed = sum(1 for r in out if r.get("gate_status") == "passed")
    failed = sum(1 for r in out if r.get("gate_status") in ("failed", "blocked_physics"))
    return {
        "arm": arm,
        "case_set": "probe" if probe else "golden",
        "changes": changes,
        "cases": len(cases),
        "passed": passed,
        "failed": failed,
        "results": out,
        "environment": "本地模型（cloud 客户端已禁用，三组同口径）",
        "note": ("小样本诊断，不是基准。看的是闸门是否仍然拦得住，"
                 "不是 Faithfulness 的绝对值。"),
        "partial": os.path.relpath(partial, _ROOT),
    }


def merge_payload(existing: dict, payload: dict) -> dict:
    """把本次结果并进已有结果，**按 arm 名累加在线组**。

    这里必须单独抽出来并测试：早期版本用 ``existing.update(payload)``，
    连跑三组时每一组都把前一组顶掉，最后只剩最后一组。命令行不报错、
    文件也写成功了，只是内容少了两组 —— 而"少了两组"会被读成
    "另外两组没跑出差异"。静默丢结果比报错难发现得多。
    """
    merged = dict(existing)
    if "online" in payload:
        prev = existing.get("online")
        # 兼容早期格式：online 直接就是单组结果（带 "arm" 键）
        if not isinstance(prev, dict) or "arm" in prev:
            prev = {}
        prev = dict(prev)
        prev.update(payload["online"])
        merged["online"] = prev
    for key, value in payload.items():
        if key != "online":
            merged[key] = value
    return merged


def main() -> int:
    ap = argparse.ArgumentParser(description="模块消融对照")
    ap.add_argument("--offline", action="store_true", help="跑不需要模型的对照")
    ap.add_argument("--online", action="store_true", help="跑需要 Ollama 的对照")
    ap.add_argument("--arm", default="baseline",
                    choices=["baseline", "no_physics_gate", "no_rewrite"])
    ap.add_argument("--cases", type=int, default=2)
    ap.add_argument("--case-timeout", type=int, default=1200)
    ap.add_argument("--probe", action="store_true",
                    help="用内置探针用例（唯一能触发物理阻断的那一题），不用 golden set")
    ap.add_argument("--write", action="store_true", help=f"写 {OUT_JSON}")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--report", default=None, metavar="PATH",
                    help="把 eval/ablation.json 渲染成 Markdown（不跑任何对照）")
    args = ap.parse_args()

    if args.report:
        # 只渲染已存的结果：渲染不该触发一次 20 分钟的在线对照。
        if not os.path.exists(OUT_JSON):
            raise SystemExit(f"没有可渲染的结果：{OUT_JSON} 不存在。先跑 --offline/--online --write")
        with open(OUT_JSON, encoding="utf-8") as fh:
            stored = json.load(fh)
        write_markdown(args.report, stored)
        print(f"[written] {args.report}")
        return 0

    payload = {}
    if args.offline or not args.online:
        payload["offline"] = run_offline()
        print(json.dumps(payload["offline"], ensure_ascii=False, indent=2)
              if args.json else render_offline(payload["offline"]))
    if args.online:
        # 按 arm 名存，不覆盖：三组要一起看才成立，前一组被后一组顶掉
        # 就等于只做了一组对照。探针组单独一个键，避免与 golden set 混在一起。
        tag = f"{args.arm}@probe" if args.probe else args.arm
        payload["online"] = {
            tag: run_online(args.arm, args.cases, args.case_timeout, probe=args.probe)
        }
        print(json.dumps(payload["online"][tag], ensure_ascii=False, indent=2))

    if args.write:
        os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
        existing = {}
        if os.path.exists(OUT_JSON):
            try:
                existing = json.load(open(OUT_JSON, encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                existing = {}
        existing = merge_payload(existing, payload)
        with open(OUT_JSON, "w", encoding="utf-8") as fh:
            json.dump(existing, fh, ensure_ascii=False, indent=2)
        print(f"[written] {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
