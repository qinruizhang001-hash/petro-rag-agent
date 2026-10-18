"""
经验库的纯逻辑层（零依赖，可离线单测）
======================================
「自我进化」闭环里所有**不含模型、不含文件系统副作用**的判断都放在这里：

    经验记录怎么构建  →  build_experience
    何时该反思        →  due_batch
    模型输出怎么解析  →  parse_reflection
    哪条先验该被注入  →  rank_priors / format_priors

为什么单独成模块，而不是全塞进 ``evolution_engine.py``：这个仓库里决定质量
的逻辑（片段构建、分支降级、成本核算、PINN 问句抽取）都被抽成纯函数放在
``utils/``，CI 不装三方包也能验证。反思这条链路的风险恰恰全在"解析"和
"时机"上，而不是在调模型那一步：

- 7B 模型输出的 JSON 经常带 ``` 围栏、带解释、把 confidence 写成 "0.95"
  字符串、甚至一次给一个对象而不是数组。解析器要能吃掉这些，否则一条都
  提炼不出来，而"反思没产出"和"反思还没触发"从外面看完全一样。
- 「行数达到 20 的倍数」这条规则如果不记批次，重跑一次脚本就会对同一批
  数据反复反思，语义记忆里堆满重复规律，而且每次都花钱。

边界约定
--------
本模块**不** import ``core.config`` / ``lightrag`` / ``torch``。它只认
dict 和字符串，所以 ``tests/test_experience_memory.py`` 可以纯 stdlib 跑完。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

# ─── 常量：与 evolution_engine 共用的唯一来源 ───────────────
BATCH_SIZE = 20          # 每多少条经验触发一次反思
MIN_CONFIDENCE = 0.8     # 高于此值才允许注入（图谱 / 先验）
MAX_PATTERN_CHARS = 160  # 单条规律的长度上限，防止模型写成一整段

EXPERIENCES = "experiences.jsonl"
REFLECTIONS = "reflections.json"
SEMANTIC_MEMORY = "semantic_memory.json"

# 规律文本里的溯源标记。写进图谱后，检索出来的片段能一眼看出这是系统自己
# 提炼的先验、而不是某篇论文的原话 —— 两者混在一起会让"证据"这个词失去意义。
PRIOR_TAG = "[系统经验]"


# ══════════════════════════════════════════════════════════
# 1. 经验记录
# ══════════════════════════════════════════════════════════
def build_experience(state: dict, run_id: str = "", ts: str = "") -> dict:
    """把一次完整运行的 state 压成一条经验。

    存的是**可比较的字段**，不是报告全文：反思要做的是找跨样本的规律，
    而报告全文会把上下文窗口塞满、还诱导模型去复述某一篇的具体结论。
    报告只保留长度，作为"这次答得详不详细"的弱信号。

    所有取值都做防御性转换 —— 这个函数会被节点在异常路径上调用，它自己
    抛异常就会把"记录失败"变成"整次评估失败"。
    """
    def _num(value, default=None):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    geology = state.get("geology_status") or {}
    wellbore = state.get("wellbore_status") or {}
    physics = state.get("physics_check") or {}
    merge = state.get("branch_merge") or {}
    report = state.get("final_report") or ""

    record = {
        "ts": ts,
        "run_id": run_id,
        "user_input": (state.get("user_input") or "")[:400],
        "params": (state.get("extracted_params") or "")[:600],
        "score": _num(state.get("score")),
        "report_chars": len(report),
        # 检索质量：候选数与 Top-1 分数。这两个数是"这次检索好不好"的唯一
        # 客观痕迹，也是反思能不能得出"什么问法容易检索失败"的依据。
        "retrieval": {
            "engine": geology.get("engine") or "",
            "candidates": _num(geology.get("candidates")),
            "top_score": _num(geology.get("top_score")),
            "wellbore_ok": bool(wellbore.get("ok")),
        },
        "degraded": bool(merge.get("degraded")),
        "degraded_reasons": list((merge.get("reasons") or {}).values())[:3],
        "physics": {
            "status": physics.get("status") or "",
            "verdict": physics.get("verdict") or "",
            "blocking": bool(physics.get("blocking")),
        },
        "rewrites": int(_num(state.get("retry_count"), 0) or 0),
        "pinn_used": bool(state.get("pinn_prediction")),
    }
    return record


def append_experience(path: str | Path, record: dict) -> int:
    """追加一条经验，返回追加后的**总行数**（空行不算）。

    返回总行数而不是 bool：调用方要用它判断是否到了反思点。让追加和计数
    在同一次遍历里完成，避免"写完再数一遍"时被并发写入插队。
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return count_experiences(p)


def count_experiences(path: str | Path) -> int:
    """有效行数。坏行不算 —— 文件被中途打断时会留半行。"""
    p = Path(path)
    if not p.exists():
        return 0
    with p.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def read_experiences(path: str | Path, limit: int | None = None) -> list[dict]:
    """读取经验，坏行跳过（一条坏行不该让整次反思失败）。

    ``limit`` 取**最近** N 条，与 ``project_history`` 的读取语义一致。
    """
    p = Path(path)
    if not p.exists():
        return []
    rows: list[dict] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows[-limit:] if limit else rows


# ══════════════════════════════════════════════════════════
# 2. 反思时机
# ══════════════════════════════════════════════════════════
def due_batch(count: int, done_batches) -> int | None:
    """行数到了 20 的倍数、且该批次没反思过 → 返回批次号（从 1 开始）。

    为什么必须记批次：只看 ``count % 20 == 0`` 的话，脚本重跑一次、或者
    40 条时连跑两次，就会对同一批数据反复反思。语义记忆里堆满重复规律，
    而且每跑一次都要花一次云端调用 —— 这种"看起来在工作"的重复最难发现。

    行数跨过多个批次（例如上次反思失败、又攒了 45 条）时只返回**最新**
    那一批：一次反思 20 条已经够提炼规律，回头补做旧批次的价值远低于成本。
    """
    if count < BATCH_SIZE:
        return None
    batch = count // BATCH_SIZE
    done = set(done_batches or ())
    return None if batch in done else batch


def batch_window(count: int, batch: int) -> tuple[int, int]:
    """批次 ``batch`` 对应的经验下标区间 ``[start, end)``。

    取"第 batch 批完整结束"的那 20 条，而不是"最后 20 条"：后者在攒了
    45 条时会跨进第 3 批，导致同一批数据被两批反思各读一次。
    """
    end = min(batch * BATCH_SIZE, count)
    start = max(0, end - BATCH_SIZE)
    return start, end


# ══════════════════════════════════════════════════════════
# 3. 反思输出解析
# ══════════════════════════════════════════════════════════
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _candidate_json(raw: str) -> list:
    """从模型输出里抠出所有可解析的 JSON 值（数组或对象）。

    三种真实输出都要吃得下：纯 JSON、```json 围栏包着的 JSON、JSON 前后
    还带了一段"好的，我分析了这 20 条经验……"的散文。
    """
    blocks = _FENCE_RE.findall(raw or "") or [raw or ""]
    found: list = []
    for block in blocks:
        text = block.strip()
        if not text:
            continue
        try:
            found.append(json.loads(text))
            continue
        except json.JSONDecodeError:
            pass
        # 围栏里还有散文：截取第一个 [ 或 { 到最后一个 ] 或 }
        for open_ch, close_ch in (("[", "]"), ("{", "}")):
            i, j = text.find(open_ch), text.rfind(close_ch)
            if i != -1 and j > i:
                try:
                    found.append(json.loads(text[i:j + 1]))
                    break
                except json.JSONDecodeError:
                    continue
    return found


def parse_reflection(raw: str, min_confidence: float = 0.0) -> list[dict]:
    """把模型输出解析成 ``[{"pattern": str, "confidence": float}]``。

    约定输出是单个对象 ``{"pattern":..., "confidence":...}``，但一次读 20 条
    经验通常能提炼出多条规律，所以数组也接受；单对象自动包装成一条。

    丢弃规则（宁可少一条，也不要脏一条）：
    - ``pattern`` 为空或长度 < 6：一条三个字的"规律"没有信息量；
    - ``confidence`` 不是数字（含 ``"0.95"`` 这种字符串 → 转成数字保留）；
    - ``confidence`` 不在 [0, 1]：越界说明模型在编，夹紧不如丢弃；
    - ``confidence < min_confidence``。
    """
    items: list = []
    for value in _candidate_json(raw):
        if isinstance(value, list):
            items.extend(value)
        elif isinstance(value, dict):
            # 兼容 {"patterns": [...]} 这种包了一层的外壳
            nested = value.get("patterns")
            if isinstance(nested, list):
                items.extend(nested)
            else:
                items.append(value)

    out: list[dict] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        pattern = item.get("pattern") or item.get("规律") or ""
        if not isinstance(pattern, str):
            continue
        pattern = " ".join(pattern.split()).strip()
        if len(pattern) < 6:
            continue
        if len(pattern) > MAX_PATTERN_CHARS:
            pattern = pattern[:MAX_PATTERN_CHARS].rstrip() + "…"

        raw_conf = item.get("confidence", item.get("置信度"))
        try:
            confidence = float(raw_conf)
        except (TypeError, ValueError):
            continue
        if not 0.0 <= confidence <= 1.0:
            continue
        if confidence < min_confidence:
            continue

        out.append({"pattern": pattern, "confidence": round(confidence, 4)})
    return out


def parse_verdicts(raw: str, expected: int) -> dict[int, dict]:
    """解析核查结果，返回 ``{下标: {"supported": bool, "reason": str}}``。

    按**顺序**对齐待核查的规律，而不是按文本匹配：核查 prompt 要求"与输入
    顺序一一对应"，靠文本对齐会因为模型改写措辞而全部错位，错位比缺失更危险
    —— 它会把"不支持"的裁决安到另一条规律头上。

    长度对不上时按位置取交集：多出来的裁决丢弃，缺的位置留空（调用方按
    "未通过"处理）。这里不做补全猜测。
    """
    items: list = []
    for value in _candidate_json(raw):
        if isinstance(value, list):
            items.extend(value)
        elif isinstance(value, dict):
            for key in ("verdicts", "results", "patterns"):
                if isinstance(value.get(key), list):
                    items.extend(value[key])
                    break
            else:
                items.append(value)

    out: dict[int, dict] = {}
    for i, item in enumerate(items[:max(0, expected)]):
        if not isinstance(item, dict):
            continue
        flag = item.get("supported")
        if flag is None:
            flag = item.get("ok")
        if isinstance(flag, str):
            flag = flag.strip().lower() in ("true", "yes", "1", "支持", "是")
        if not isinstance(flag, bool):
            continue
        reason = item.get("reason") or item.get("依据") or ""
        out[i] = {"supported": flag, "reason": " ".join(str(reason).split())[:200]}
    return out


def dedupe_patterns(patterns: list[dict]) -> list[dict]:
    """去掉重复规律（保留置信度更高的那条）。

    模型对同一批数据跑两次，很容易换几个字再说一遍同一个意思。完全相同的
    直接去重；近似重复（字符集高度重叠）也合并 —— 语义记忆是"先验"，重复
    注入只会让 prompt 变长，不会让判断变准。
    """
    kept: list[dict] = []
    for item in sorted(patterns, key=lambda x: -x.get("confidence", 0)):
        if any(similarity(item["pattern"], k["pattern"]) >= 0.85 for k in kept):
            continue
        kept.append(item)
    return kept


# ══════════════════════════════════════════════════════════
# 4. 先验的检索与格式化
# ══════════════════════════════════════════════════════════
def _bigrams(text: str) -> set[str]:
    """字符二元组集合。中文按字切，不做分词 —— 这一步不该引入词典依赖。"""
    s = re.sub(r"\s+", "", text or "")
    if len(s) < 2:
        return {s} if s else set()
    return {s[i:i + 2] for i in range(len(s) - 1)}


def similarity(a: str, b: str) -> float:
    """两段文本的二元组 Jaccard 相似度（0~1）。纯函数，无模型、无 embedding。

    为什么不用向量检索：语义记忆只有几条到几十条，向量化要拉嵌入模型
    （nomic-embed 走 Ollama，或本地 2.3 GB 的 reranker），为几条文本付这个
    代价没有意义；而且这一步发生在 extract 节点里，是整个流程最该快的节点。
    """
    ba, bb = _bigrams(a), _bigrams(b)
    if not ba or not bb:
        return 0.0
    return len(ba & bb) / len(ba | bb)


# ─── 属性感知的相关度 ────────────────────────────────────
# 纯字面匹配有个致命盲区：规律说"缺少储层埋深时物理校验无法执行"，
# 而问句里**根本不会出现"埋深"两个字** —— 那正是这条规律要提醒的事。
# 字面重叠为 0，先验在最该出现的场合反而被丢掉。实测踩到过。
#
# 所以补一层领域属性匹配：规律提到的属性出现在问句里 → 相关；
# 规律说的是"缺少 X"、而问句里确实没有 X → **更相关**。
ATTRIBUTES = (
    "埋深", "深度", "注入压力", "压力", "温度", "注入速率", "注入量",
    "断层", "盖层", "储层", "井筒", "套管", "渗透率", "孔隙度", "破裂压力",
)
_ABSENCE_MARKERS = ("缺少", "缺失", "未提供", "未给出", "没有提供", "不含", "无")

MIN_RELEVANCE = 0.1   # 低于此值不注入：一个二元组的偶然重叠不是"相关"


def relevance(pattern: str, query: str) -> float:
    """规律与问句的相关度（0~1）。字面重叠 + 属性感知加成。

    加成取值有讲究：
    - 属性在问句里出现（+0.15）：确认相关，但不如字面重叠有说服力；
    - 规律说"缺少 X"且问句里确实没有 X（+0.25）：这正是该提醒的场景，
      给最高加成，否则这类最有价值的规律永远命中不了。
    """
    base = similarity(pattern, query)
    bonus = 0.0
    claims_absence = any(marker in pattern for marker in _ABSENCE_MARKERS)
    for attr in ATTRIBUTES:
        if attr not in pattern:
            continue
        if attr in query:
            bonus += 0.15
        elif claims_absence:
            bonus += 0.25
    return min(1.0, base + bonus)


def rank_priors(records: list[dict], query: str, top_k: int = 3,
                min_confidence: float = MIN_CONFIDENCE) -> list[dict]:
    """按与 query 的相关度给先验排序，只返回高于置信度阈值的。

    排序键是 (相关度, 置信度)：相关度决定"这条规律跟本次问题有没有关系"，
    置信度只在相关度相同时用来打破平局。反过来排会让高置信度但不相关的
    规律挤掉真正对口的那条 —— 而"把无关的先验塞进 prompt"比"不塞"更糟，
    模型会当真。
    """
    rows = [r for r in records if isinstance(r, dict)]
    rows = [r for r in rows if float(r.get("confidence") or 0) >= min_confidence]
    if not rows:
        return []
    scored = [(relevance(r.get("pattern", ""), query), r) for r in rows]
    scored.sort(key=lambda pair: (-pair[0], -float(pair[1].get("confidence") or 0)))
    # 相关度不足的先验不注入：它跟本次问题没有实质关联，塞进去只是噪音
    return [r for score, r in scored if score >= MIN_RELEVANCE][:max(1, int(top_k))]


def format_priors(priors: list[dict]) -> str:
    """把先验渲染成可注入 prompt 的文本块；无先验时返回空串。

    返回空串而不是"（暂无历史经验）"，是因为调用方用 ``if text:`` 判断是否
    拼进 prompt —— 一个占位说明会让每份 prompt 都多出一段无信息量的内容。
    """
    if not priors:
        return ""
    lines = []
    for i, p in enumerate(priors, 1):
        lines.append(f"{i}. {p['pattern']}（置信度 {float(p['confidence']):.2f}）")
    return "\n".join(lines)
