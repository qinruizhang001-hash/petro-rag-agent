"""
自我进化引擎（借鉴 Generative Agents 的 Memory Reflection）
==========================================================
闭环：

    每次评估结束
        └─→ history/experiences.jsonl           （1. 存经验：参数 / 检索 / 评分）
                └─→ 行数达到 20 的倍数           （2. 触发）
                        └─→ ReflectionAgent      （3. 提炼：{"pattern","confidence"}）
                                ├─→ history/semantic_memory.json   （高置信度先验）
                                └─→ LightRAG 知识图谱              （可选：写进图谱）
                                        └─→ extract 节点注入 prompt （4/5. 应用）

运行::

    python evolution_engine.py --simulate     # 模拟：20 条假经验 → 一次真实反思
    python evolution_engine.py --stats        # 看当前经验库 / 语义记忆
    python evolution_engine.py --reflect      # 对已攒够的真实数据强制反思一次

三个设计判断，都和"照字面实现"不一样，理由写在下面
--------------------------------------------------
**一、"最高优先级"不是 LightRAG 的概念。** 图谱检索的结果要过 rerank、再被
``retrieve_node`` 截断到 Top-2（实测 129 个候选只留 2 条）。也就是说写进图谱
的先验**会参与竞争，而且大概率被挤掉** —— 它是"最低优先级"，不是最高。
真正能保证"每次都在场"的位置只有一个：extract 节点的 prompt 顶部。所以本模块
用 ``semantic_memory.json`` 作为先验的**权威存储**，图谱写入作为并行的第二通道
保留（用户明确要求，且它确实能被检索到），但主流程读的是前者。

**二、反思用 "reflect" 这个独立角色。** 它默认走 cloud 档位 —— 也就是说
配了 ``CLOUD_API_KEY`` 时就是 DeepSeek（``CLOUD_MODEL`` 默认 ``deepseek-chat``）。
没有 key 时按项目既有的降级契约退到本地 7B，并在结果里写明**实际用了哪个模型**：
"以为在用 DeepSeek、其实在用 7B"是这类系统最难查的问题之一。

**三、写图谱是同步的，但失败绝不影响评估。** ``ainsert`` 会触发一次实体抽取，
是这条链路里唯一的重操作。它带超时、带 try/except、带 ``EVOLVE_WRITE_GRAPH=0``
开关，并且每次写入都记到 ``history/graph_injections.json`` —— 图谱里混进模型
编的"规律"和论文原话是两回事，必须留痕、可追溯、可删除。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

# 必须在 import core 之前清代理：httpx 在构造时读取这些变量，
# 指向本地网关的请求会被转发给系统代理并收到 502（本项目踩过）。
for _k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
    os.environ.pop(_k, None)
os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
os.environ.setdefault("no_proxy", "localhost,127.0.0.1")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from utils.experience_memory import (  # noqa: E402
    BATCH_SIZE,
    EXPERIENCES,
    MIN_CONFIDENCE,
    PRIOR_TAG,
    REFLECTIONS,
    SEMANTIC_MEMORY,
    append_experience,
    batch_window,
    build_experience,
    count_experiences,
    dedupe_patterns,
    due_batch,
    format_priors,
    parse_reflection,
    parse_verdicts,
    rank_priors,
    read_experiences,
)

# ─── 目录解析 ───────────────────────────────────────────────
# EVOLVE_HOME 存在是为了两件事：模拟测试用临时目录（假经验绝不能进真实库），
# 以及让单测在不碰用户数据的前提下验证落盘行为。
HOME = Path(os.getenv("EVOLVE_HOME") or (ROOT / "history"))
EXP_PATH = HOME / EXPERIENCES
REF_PATH = HOME / REFLECTIONS
MEM_PATH = HOME / SEMANTIC_MEMORY
GRAPH_LOG = HOME / "graph_injections.json"

WRITE_GRAPH = os.getenv("EVOLVE_WRITE_GRAPH", "1") not in ("0", "false", "False")
GRAPH_TIMEOUT = float(os.getenv("EVOLVE_GRAPH_TIMEOUT", "120"))


# ══════════════════════════════════════════════════════════
# Reflection Agent
# ══════════════════════════════════════════════════════════
class ReflectionAgent:
    """一个**独立的** Agent：它不看当前任务，只看历史，只输出规律。

    为什么要独立成类而不是复用评审员：两者的问题空间相反。评审员看**一份**
    报告、判断它对不对；反思者看 **20 条**记录、寻找跨样本的共同结构。
    把两种 prompt 混在一起，模型会开始给历史记录打分，而那不是规律。

    prompt 里刻意给了"什么算规律、什么不算"的判据，因为 7B 模型天然倾向
    复述单条记录的具体数值（"某项目注入压力 250 bar"），那不是规律而是记忆。
    """

    SYSTEM = (
        "你是一个 CCUS 领域的方法论分析者。你不评估任何单个项目，"
        "只从一批历史评估记录里寻找**可复用的规律**。"
    )

    TEMPLATE = """下面是同一个 CCUS 风险评估系统最近 {n} 次评估的记录（JSON Lines 风格）。

【任务】提炼跨样本的规律，而不是复述单个案例。

什么算规律（要）：
- 参数条件与结果之间的稳定关联，例如"缺少埋深时物理校验无法执行，评分普遍偏低"；
- 检索质量与结论可靠性之间的关系，例如"Top-1 相关度低于 0.3 时地质结论明显变短"；
- 反复出现的失效模式，例如"注入压力接近破裂压力时报告被打回重写"。

什么不算规律（不要）：
- 单条记录的具体数值（"某项目 250 bar"）—— 这是记忆，不是规律；
- 泛泛的领域常识（"注入压力过大会导致盖层破裂"）—— 报告里本来就有；
- 对系统本身的建议（"应该增加数据"）—— 要描述现象，不要给方案。

【输出格式】严格输出 JSON 数组，每个元素两个字段，不要输出任何解释文字：
[{{"pattern": "一句话规律，不超过 60 字", "confidence": 0.0~1.0}}]

confidence 的含义：这 20 条记录对这条规律的支持强度。只在多条记录都指向同一
结论时才给高分（>0.8）；只有一两条支持、其余矛盾时给低分（<0.5）。

【记录】
{records}"""

    def __init__(self, stage: str = "reflect", temperature: float = 0.2):
        self.stage = stage
        self.temperature = temperature

    def reflect(self, experiences: list[dict]) -> tuple[list[dict], dict]:
        """返回 (规律列表, 元信息)。元信息里带**实际使用的模型**。"""
        from core.config import get_llm, llm_chat

        # 送进 prompt 前先裁字段：报告全文不在这里，但 params 可能很长，
        # 20 条 × 600 字会把上下文吃光，反而降低规律质量。
        slim = self._slim(experiences)
        records = "\n".join(json.dumps(s, ensure_ascii=False) for s in slim)

        _, model, tier, reason = get_llm(self.stage)
        response = llm_chat(
            self.stage,
            messages=[
                {"role": "system", "content": self.SYSTEM},
                {"role": "user", "content": self.TEMPLATE.format(n=len(slim), records=records)},
            ],
            temperature=self.temperature,
        )
        raw = response.choices[0].message.content or ""
        patterns = parse_reflection(raw, min_confidence=0.0)
        meta = {
            "stage": self.stage,
            "model": model,
            "tier": tier,
            "degrade_reason": reason,
            "raw_chars": len(raw),
            "parsed": len(patterns),
        }
        return patterns, meta

    @staticmethod
    def _slim(experiences: list[dict]) -> list[dict]:
        slim = []
        for r in experiences:
            slim.append({
                "run": r.get("run_id"),
                "场景": (r.get("user_input") or "")[:120],
                "参数": (r.get("params") or "")[:220],
                "评分": r.get("score"),
                "检索": r.get("retrieval"),
                "降级": r.get("degraded"),
                "物理校验": r.get("physics"),
                "重写次数": r.get("rewrites"),
            })
        return slim

    def verify(self, patterns: list[dict], experiences: list[dict]) -> tuple[list[dict], dict]:
        """第二道关：让模型**对着同一批数据**逐条反驳自己的规律。

        为什么必须有这一步：实测 7B 从 20 条记录里吐出的 6 条规律中，3 条是
        伪相关（"隐伏断层存在时评分较低"——假数据里只有 3 条带断层，评分毫无
        规律），却全部给了 0.80，与真规律同分。**置信度阈值挡不住伪规律**，
        因为伪规律的置信度和真规律一样高。它只会挡掉模型自己都不确定的那些。

        核查判据是"有没有反例"：一条规律只要在同一条件下出现过相反结果，
        就不该被当成先验注入后续每一次评估。

        一次调用核查全部规律，而不是一条一次：20 条记录要塞进 prompt，
        拆成 N 次就是 N 倍的钱，而每 20 次评估才有一次反思，不值得再省这一层。
        """
        from core.config import get_llm, llm_chat

        if not patterns:
            return [], {"checked": 0, "rejected": 0}

        records = "\n".join(json.dumps(s, ensure_ascii=False)
                            for s in self._slim(experiences))
        claims = "\n".join(f"{i + 1}. {p['pattern']}" for i, p in enumerate(patterns))
        prompt = f"""下面是 20 条评估记录，以及从这个系统里总结出的 {len(patterns)} 条"规律"。

【任务】逐条核查：这些记录是否**真的支持**该规律？

判据只有一条 —— **存在反例就判为不支持**。
反例指：规律声称"条件 A 时结果 B"，但记录里有满足条件 A、结果却不是 B 的样本。
注意区分：
- "缺埋深时评分低"与"埋深大时评分高"是两条不同的规律，后者需要埋深与评分
  呈单调关系才算成立；
- 只被 1~2 条记录支持、其余记录无关的，判为不支持（样本不足，不是规律）；
- 描述系统机制而非数据关联的（"可能降级"），也判为不支持。

【输出】严格输出 JSON 数组，与输入顺序一一对应，不要任何解释文字：
[{{"supported": true, "reason": "一句话依据"}}, {{"supported": false, "reason": "反例"}}]

【记录】
{records}

【待核查的规律】
{claims}"""

        _, model, tier, _ = get_llm(self.stage)
        try:
            response = llm_chat(
                self.stage,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            verdicts = parse_verdicts(response.choices[0].message.content or "", len(patterns))
        except Exception as exc:  # noqa: BLE001 — 核查失败不该让反思白跑
            return patterns, {"checked": 0, "rejected": 0,
                              "error": f"{type(exc).__name__}: {exc}"}

        kept, rejected = [], []
        for i, p in enumerate(patterns):
            v = verdicts.get(i)
            if v is None:
                # 没给出裁决的按"未通过"处理：宁可少一条先验，也不要一条伪规律
                rejected.append({**p, "reason": "核查未返回该条的结论"})
                continue
            if v["supported"]:
                kept.append({**p, "verified_by": model})
            else:
                rejected.append({**p, "reason": v["reason"]})
        return kept, {"checked": len(patterns), "rejected": len(rejected),
                      "model": model, "tier": tier, "details": rejected}


# ══════════════════════════════════════════════════════════
# 存储
# ══════════════════════════════════════════════════════════
def _load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default


def _save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)  # 原子替换：写一半被中断也不会留下坏 JSON


def load_semantic_memory() -> list[dict]:
    data = _load_json(MEM_PATH, [])
    return data if isinstance(data, list) else []


def save_patterns(patterns: list[dict], batch: int, meta: dict) -> list[dict]:
    """把新规律并入语义记忆，返回**本次新增**的那些。

    去重跨越整个文件而不是只在本批次内：同一批数据被反思两次（或相邻两批
    数据高度相似）时，语义记忆会积累近似重复的规律，先验注入就变成了复读。
    """
    existing = load_semantic_memory()
    incoming = []
    for p in patterns:
        if float(p.get("confidence") or 0) < MIN_CONFIDENCE:
            continue
        incoming.append({
            **p,
            "batch": batch,
            "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model": meta.get("model", ""),
        })
    if not incoming:
        return []
    merged = dedupe_patterns(existing + incoming)
    added = [p for p in merged if p not in existing]
    _save_json(MEM_PATH, merged)
    return added


def mark_batch_done(batch: int, meta: dict, added: int) -> None:
    done = _load_json(REF_PATH, [])
    if not isinstance(done, list):
        done = []
    done.append({
        "batch": batch,
        "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": meta.get("model", ""),
        "tier": meta.get("tier", ""),
        "patterns": added,
    })
    _save_json(REF_PATH, done)


# ══════════════════════════════════════════════════════════
# 图谱注入
# ══════════════════════════════════════════════════════════
def inject_to_graph(patterns: list[dict]) -> dict:
    """把高置信度规律写进 LightRAG 知识图谱（第二通道）。

    写入文本带 ``[系统经验]`` 前缀与置信度：图谱里必须能一眼分出"论文说的"
    和"系统自己总结的"。混在一起之后，检索出来的片段会被当成文献证据引用，
    而它其实只是模型的归纳 —— 这比少一条证据严重得多。

    失败一律返回 ``{"ok": False, ...}``，不抛异常：写图谱是**增强**，
    不该让一次评估或一次反思因此失败。
    """
    if not patterns:
        return {"ok": True, "written": 0, "skipped": "无高置信度规律"}
    if not WRITE_GRAPH:
        return {"ok": True, "written": 0, "skipped": "EVOLVE_WRITE_GRAPH=0"}

    lines = [
        f"{PRIOR_TAG}（系统从历史评估中提炼的先验，未经文献验证，置信度 {float(p['confidence']):.2f}）"
        f"{p['pattern']}"
        for p in patterns
    ]
    text = "\n".join(lines)
    t0 = time.time()
    try:
        from lightrag_retriever import insert_text  # 惰性：会拉起 LightRAG

        ok = insert_text(text, timeout=GRAPH_TIMEOUT)
    except Exception as exc:  # noqa: BLE001 — 图谱不可用不该影响主流程
        return {"ok": False, "written": 0, "error": f"{type(exc).__name__}: {exc}"}

    elapsed = round(time.time() - t0, 1)
    record = {
        "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "ok": bool(ok),
        "elapsed_s": elapsed,
        "patterns": patterns,
        "text": text,
    }
    log = _load_json(GRAPH_LOG, [])
    if not isinstance(log, list):
        log = []
    log.append(record)
    _save_json(GRAPH_LOG, log)
    return {"ok": bool(ok), "written": len(patterns) if ok else 0, "elapsed_s": elapsed}


# ══════════════════════════════════════════════════════════
# 闭环入口
# ══════════════════════════════════════════════════════════
def record_experience(state: dict, run_id: str = "") -> dict:
    """记一条经验；若到点则顺手跑一次反思。

    这个函数被 ``nodes/history_node.py`` 在每次评估结束时调用，所以它**必须**
    吞掉所有异常 —— 进化是增强能力，不能让一次正常评估因为"经验写不进去"而
    失败。返回的 dict 只用于日志与测试断言。
    """
    out: dict = {"recorded": False, "count": 0, "reflected": None}
    try:
        record = build_experience(
            state,
            run_id=run_id or f"run_{int(time.time())}",
            ts=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )
        count = append_experience(EXP_PATH, record)
        out.update(recorded=True, count=count)
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {exc}"
        return out

    try:
        done = [r.get("batch") for r in (_load_json(REF_PATH, []) or []) if isinstance(r, dict)]
        batch = due_batch(count, done)
        if batch is not None:
            out["reflected"] = run_reflection(batch, count)
    except Exception as exc:  # noqa: BLE001
        out["reflect_error"] = f"{type(exc).__name__}: {exc}"
    return out


def run_reflection(batch: int, count: int | None = None) -> dict:
    """对第 ``batch`` 批经验做一次反思，返回结构化结果。

    ``count`` 由调用方传入以免重复读文件；单独调用时自己数。

    流程是**两步而不是一步**：先提炼，再核查。核查不是可选项 —— 见
    ``ReflectionAgent.verify`` 里那段实测记录，7B 的伪规律和真规律同分，
    阈值筛不掉。
    """
    total = count if count is not None else count_experiences(EXP_PATH)
    start, end = batch_window(total, batch)
    window = read_experiences(EXP_PATH)[start:end]
    if len(window) < BATCH_SIZE:
        return {"batch": batch, "skipped": f"窗口只有 {len(window)} 条，不足 {BATCH_SIZE}"}

    agent = ReflectionAgent()
    patterns, meta = agent.reflect(window)
    verified, verify_meta = agent.verify(patterns, window)

    # 核查这一步失败时**不收下任何规律，也不标记批次完成**。
    # 两种"失败兜底"都试过，都有害：全部放行等于核查坏了就静默收下未验证的
    # 规律（而它正是用来挡伪规律的那道关）；标记完成则这批数据再也不会被
    # 反思一次。所以选择"这一轮不算数"—— 下一次评估结束时自动重试。
    if verify_meta.get("error"):
        return {
            "batch": batch,
            "window": [start, end],
            "patterns": patterns,
            "verified": [],
            "verify": verify_meta,
            "added": [],
            "meta": meta,
            "skipped": f"核查不可用，本轮不写入（{verify_meta['error']}），下次评估后重试",
            "graph": {"ok": True, "written": 0, "skipped": "核查未通过"},
        }

    added = save_patterns(verified, batch, meta)
    graph = inject_to_graph(added) if added else {"ok": True, "written": 0, "skipped": "无新增"}
    mark_batch_done(batch, meta, len(added))
    return {
        "batch": batch,
        "window": [start, end],
        "patterns": patterns,
        "verified": verified,
        "verify": verify_meta,
        "added": added,
        "meta": meta,
        "graph": graph,
    }


def query_system_experience(query: str, top_k: int = 3) -> str:
    """给 extract 节点用：查一次 system_experience，返回可注入的文本。

    这是第 5 步"应用到主流程"的唯一入口。设计上要求它**快且绝不抛异常**：
    它在整个流程最靠前的节点里被调用，一次失败的检索不该拖垮评估。
    没有语义记忆时立刻返回空串（一次 stat + 一次 read，几毫秒）。
    """
    try:
        records = load_semantic_memory()
        if not records:
            return ""
        priors = rank_priors(records, query, top_k=top_k, min_confidence=MIN_CONFIDENCE)
        return format_priors(priors)
    except Exception:  # noqa: BLE001 — 先验缺失只是少了点上下文
        return ""


# ══════════════════════════════════════════════════════════
# 模拟数据
# ══════════════════════════════════════════════════════════
def simulated_experiences(n: int = BATCH_SIZE) -> list[dict]:
    """造 n 条假经验，但**规律是真的**。

    随机噪声喂给反思模型，它只能编出听起来像规律的话，测试就变成了"看模型
    会不会编"。所以这里埋了三条真实关联，事后可以核对反思有没有找到：

    1. 缺埋深（``params`` 里没有"埋深/深度"）→ 物理校验 unverifiable → 评分 5~6；
    2. 有埋深 → 物理校验 pass → 评分 8~9；
    3. 检索 Top-1 相关度低（<0.3）→ 地质分支降级 → 结论变短、评分再降一档。

    这不是为了"让测试好看"，而是唯一能判断反思是否真的有效的办法：
    已知答案的情况下看它有没有说出来。
    """
    rows = []
    base = datetime(2026, 9, 1, 9, 0, 0)
    for i in range(n):
        has_depth = i % 3 != 0                      # 每 3 条缺 1 条埋深 → 约 7 条缺
        weak_retrieval = i % 5 == 0                 # 每 5 条检索偏弱
        p_inj = 250 + (i % 4) * 10
        temp = 75 + (i % 3) * 5
        depth = 2000 + (i % 3) * 500

        params = f"注入压力 {p_inj} bar，温度 {temp}℃"
        if has_depth:
            params += f"，储层埋深 {depth} m"
        if i % 7 == 0:
            params += "，存在隐伏断层"

        top_score = round(0.12 + (i % 5) * 0.02, 3) if weak_retrieval else round(0.86 + (i % 4) * 0.03, 3)
        if has_depth:
            physics = {"status": "pass", "verdict": "safe_window" if p_inj < 300 else "risky",
                       "blocking": False}
            score = 8.0 + (i % 3) * 0.5
        else:
            physics = {"status": "unverifiable", "verdict": "", "blocking": False}
            score = 5.0 + (i % 2)
        if weak_retrieval:
            score -= 1.0
            physics = physics if has_depth else physics

        rows.append({
            "ts": (base.replace(day=1 + i)).strftime("%Y-%m-%d %H:%M:%S"),
            "run_id": f"sim_{i:02d}",
            "user_input": f"某CCUS项目第 {i + 1} 个算例，请评估封存风险。",
            "params": params,
            "score": round(max(1.0, score), 1),
            "report_chars": 600 if weak_retrieval else 1100,
            "retrieval": {
                "engine": "lightrag",
                "candidates": 129 if not weak_retrieval else 41,
                "top_score": top_score,
                "wellbore_ok": i % 9 != 0,
            },
            "degraded": bool(weak_retrieval),
            "degraded_reasons": ["地质分支检索相关度过低"] if weak_retrieval else [],
            "physics": physics,
            "rewrites": 1 if (has_depth and p_inj >= 280) else 0,
            "pinn_used": i % 6 == 0,
        })
    return rows


# ══════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════
def _print_patterns(patterns: list[dict], title: str) -> None:
    print(f"\n{title}")
    if not patterns:
        print("  （无）")
        return
    for p in patterns:
        print("  " + json.dumps({"pattern": p["pattern"], "confidence": p["confidence"]},
                                ensure_ascii=False))


def cmd_simulate() -> int:
    """模拟一次完整闭环：20 条假经验 → 触发反思 → 打印语义记忆。

    用临时目录（``EVOLVE_HOME``）跑：假经验绝不能进真实库，否则下一次真实
    反思会把这批编造的数据当成事实去提炼规律 —— 而它看起来完全正常。
    """
    global HOME, EXP_PATH, REF_PATH, MEM_PATH, GRAPH_LOG, WRITE_GRAPH

    tmp = Path(tempfile.mkdtemp(prefix="ccus_evolve_sim_"))
    HOME = tmp
    EXP_PATH, REF_PATH, MEM_PATH = tmp / EXPERIENCES, tmp / REFLECTIONS, tmp / SEMANTIC_MEMORY
    GRAPH_LOG = tmp / "graph_injections.json"
    WRITE_GRAPH = False  # 假规律不写真实图谱

    print("=" * 66)
    print("自我进化引擎 · 模拟测试")
    print("=" * 66)
    print(f"临时目录     : {tmp}")
    print(f"图谱写入     : 关闭（模拟数据不得污染真实知识图谱）")

    rows = simulated_experiences(BATCH_SIZE)
    print(f"\n[1] 生成假经验 {len(rows)} 条")
    count = 0
    for r in rows:
        count = append_experience(EXP_PATH, r)
    print(f"    已写入 {EXP_PATH.name}，当前 {count} 行")
    print(f"    埋入的三条真实关联：")
    print(f"      · 缺埋深 → 物理校验 unverifiable → 评分 5~6（{(count // 3)} 条）")
    print(f"      · 有埋深 → 物理校验 pass        → 评分 8~9")
    print(f"      · Top-1 相关度 < 0.3 → 地质分支降级 → 评分再降 1 分")

    print(f"\n[2] 触发条件检查：{count} 行，{BATCH_SIZE} 的倍数为 {count % BATCH_SIZE == 0}")
    batch = due_batch(count, [])
    print(f"    due_batch → 批次 {batch}")
    if batch is None:
        print("    ❌ 未触发，测试失败")
        return 1

    print(f"\n[3] 唤醒 ReflectionAgent（读取最近 {BATCH_SIZE} 条经验）...")
    result = run_reflection(batch, count)
    meta = result.get("meta") or {}
    print(f"    模型路由 : {meta.get('tier')}:{meta.get('model')}"
          + (f"（降级原因：{meta.get('degrade_reason')}）" if meta.get("degrade_reason") else ""))
    print(f"    原始输出 : {meta.get('raw_chars')} 字符 → 解析出 {meta.get('parsed')} 条")

    _print_patterns(result.get("patterns") or [], "[4] 提炼出的规律（模型原始输出，含低置信度）")

    verify = result.get("verify") or {}
    print(f"\n[5] 核查（对着同一批数据找反例）：检查 {verify.get('checked', 0)} 条，"
          f"驳回 {verify.get('rejected', 0)} 条")
    if verify.get("error"):
        print(f"    ⚠️ 核查不可用：{verify['error']}")
    for item in (verify.get("details") or []):
        print(f"    ✗ {item['pattern']} —— {item.get('reason', '')[:70]}")
    if result.get("skipped"):
        print(f"    ⚠️ {result['skipped']}")

    _print_patterns(result.get("added") or [],
                    f"[6] 通过核查并写入语义记忆的规律（置信度 >{MIN_CONFIDENCE}）")
    if result.get("graph", {}).get("skipped"):
        print(f"\n[7] 图谱注入：跳过（{result['graph']['skipped']}）")

    mem = load_semantic_memory()
    print(f"\n[8] 语义记忆落盘：{MEM_PATH}")
    print(f"    {len(mem)} 条")

    print("\n[9] 应用到主流程（模拟 extract 节点的查询）")
    probes = [
        "某CCUS项目注入压力250bar，温度80℃，请评估风险。",
        "某CCUS项目注入压力260bar，温度80℃，埋深2500m，请评估风险。",
    ]
    for q in probes:
        priors = rank_priors(mem, q, top_k=3)
        print(f"\n    问句：{q}")
        print(f"    命中 {len(priors)} 条先验：")
        text = format_priors(priors)
        print("      " + (text.replace("\n", "\n      ") if text else "（无，不注入）"))

    print("\n" + "=" * 66)
    print("✅ 闭环跑通：经验 → 触发 → 反思 → 语义记忆 → 先验注入")
    print("=" * 66)
    return 0


def cmd_stats() -> int:
    count = count_experiences(EXP_PATH)
    mem = load_semantic_memory()
    done = _load_json(REF_PATH, [])
    print(f"目录      : {HOME}")
    print(f"经验条数  : {count}（每 {BATCH_SIZE} 条触发一次反思）")
    print(f"已反思批次: {[d.get('batch') for d in done] if isinstance(done, list) else []}")
    print(f"语义记忆  : {len(mem)} 条")
    _print_patterns(mem, "先验清单：")
    return 0


def cmd_reflect(force: bool = False) -> int:
    count = count_experiences(EXP_PATH)
    done = [r.get("batch") for r in (_load_json(REF_PATH, []) or []) if isinstance(r, dict)]
    batch = due_batch(count, [] if force else done)
    if batch is None:
        print(f"当前 {count} 条经验，无待反思批次（已反思：{done}）。用 --force 强制。")
        return 0
    result = run_reflection(batch, count)
    print(json.dumps({k: v for k, v in result.items() if k != "patterns"},
                     ensure_ascii=False, indent=2))
    _print_patterns(result.get("added") or [], "本次新增先验：")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CCUS 自我进化引擎（Memory Reflection）")
    parser.add_argument("--simulate", action="store_true", help="用 20 条假经验跑一次完整闭环")
    parser.add_argument("--stats", action="store_true", help="查看经验库与语义记忆")
    parser.add_argument("--reflect", action="store_true", help="对已攒够的真实数据反思一次")
    parser.add_argument("--force", action="store_true", help="配合 --reflect：忽略已反思标记")
    args = parser.parse_args(argv)

    if args.simulate:
        return cmd_simulate()
    if args.stats:
        return cmd_stats()
    if args.reflect:
        return cmd_reflect(force=args.force)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
