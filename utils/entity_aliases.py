"""
实体别名归并与脏节点识别（纯标准库）
====================================
LightRAG 按**实体名完全相等**合并节点。这意味着 "CO2" 和 "Carbon dioxide (CO2)"
是两条互不相连的节点 —— 同一篇论文里反复出现的核心概念，在图谱上被切成两半。

实测（23 篇语料，765 节点）：同一概念常有 3-5 种写法
    CO2 / Carbon Dioxide / Carbon dioxide (CO2) / CO₂
    Fault / Faults / fault
    Cap Rock Fracturing / Caprock fracturing
    Digital Shadow / Digital Shadows (DS) / 3D Digital Shadow framework
         / 3D digital shadow framework (Gahlot, Orozco, et al. 2024)
这种"同物异名"是连通分量数量暴涨（765 节点散成 247 块）的直接原因之一。

本模块做两件事，都是**确定性**的，不依赖模型：
1. ``build_merge_plan`` —— 把归一化后同键的节点归成一组，给出合并计划；
2. ``noise_reason`` —— 认出"不是知识"的节点（作者单位、年份、街道地址、
   期刊名、DOI、``et al.`` 引文残片），交给调用方删除。

设计约束
--------
- **失败方向必须是"漏合并"，不能是"错合并"**。归一化做错了顶多少连一条边；
  合并做错了会把两个不同的地质概念永久并成一个节点，且描述被拼接、
  无法自动还原。所以这里不做模糊匹配、不做编辑距离、不做 embedding 相似度
  —— 只做大小写、单复数、括号注释、引文后缀这四种**可解释**的归约。
- 规则全部可离线单测（``tests/test_entity_aliases.py``），不需要 LightRAG/Ollama。
"""
from __future__ import annotations

import re

# ─── 规范化 ────────────────────────────────────────────────

# 行尾引用后缀：`(Gahlot, Orozco, et al. 2024)`、`(2023)`、`(2023; Herrmann 2023)`
# 只匹配**末尾**的括号，避免误伤 `Calcite (Calcium Carbonate)` 这类真正写在
# 名字中间的括号 —— 那种由下面的 _PAREN_GLOSS 单独处理。
_CITATION_SUFFIX = re.compile(
    r"\s*[\(（][^()（）]*?(?:\bet\s+al\b|\d{4})[^()（）]*[\)）]\s*$", re.I
)

# 括号注释/缩写：`Calcite (Calcium Carbonate)`、`光纤传感技术(DAS/DTS)`
# 中英文括号都吃，长度上限 60 字符（超过就不像缩写注释，更像正文）
_PAREN_GLOSS = re.compile(r"\s*[\(（][^()（）]{1,60}[\)）]\s*")

# 单复数归约。**只处理以辅音+ies / s / es 结尾的普通复数**，
# 不处理 -us / -is / -ss 结尾（analysis、stress、gas 都不是复数）。
_PLURAL_RULES = (
    (re.compile(r"(?<=[a-z])ies$"), "y"),
    (re.compile(r"(?<=[a-z])(?:sses|shes|ches|xes|zes)$"), "es"),   # 去掉 es
    (re.compile(r"(?<=[a-z])ses$"), "s"),                            # processes → process
    (re.compile(r"(?<=[a-z])s$"), ""),
)

# 键层面的改写：在归一化之后、分组之前应用。
# 这些是"同一个词的不同拼写"，不是同义词表 —— 同义词在 CANONICAL_GROUPS 里。
_KEY_REWRITES = (
    ("cap rock", "caprock"),
    ("cap-rock", "caprock"),
    ("co 2", "co2"),
    ("co₂", "co2"),
)

# 显式同义词组：规范名 → 所有已知写法。
# 只收**领域内确认等价**的写法；有细微差别的概念（Pre-Existing Fault vs Fault）
# 故意不收 —— 错合并的代价远大于漏合并。
CANONICAL_GROUPS: dict[str, tuple[str, ...]] = {
    "CO2": ("CO2", "Carbon Dioxide", "Carbon dioxide", "CO₂", "CO2 Gas"),
    "CCS": (
        "CCS",
        "Carbon Capture and Storage",
        "Carbon capture and storage",
        "Carbon Capture and Sequestration",
        "Carbon Dioxide Capture and Storage",
        "Carbon Capture and Sequestration Technology",
    ),
    "CCUS": (
        "CCUS",
        "Carbon Capture, Utilization, and Storage",
        "Carbon Capture, Utilisation and Storage",
        "Carbon Capture and Storage or Utilization",
    ),
    "Caprock": ("Caprock", "Cap rock", "Cap Rock"),
    "Caprock Fracturing": ("Caprock Fracturing", "Cap Rock Fracturing"),
    "Fault": ("Fault", "Faults"),
    "Saline Aquifer": ("Saline Aquifer", "Saline Aquifers"),
    "CO2 Leakage": ("CO2 Leakage",),
    "CO2 Plume": ("CO2 Plume", "CO2 Plumes"),
    "CO2 Saturation": ("CO2 Saturation",),
    "CO2 Storage": ("CO2 Storage",),
    "Wellbore Integrity": ("Wellbore Integrity",),
    "Digital Twin": ("Digital Twin", "Digital Twins"),
    "Digital Shadow": ("Digital Shadow", "Digital Shadows"),
    "Deep Learning": ("Deep Learning",),
    "Reinforcement Learning": ("Reinforcement Learning",),
    "Calcite": ("Calcite", "Calcium Carbonate", "Calcium carbonate"),
    "Calcium Hydroxide": ("Calcium Hydroxide", "Calcium hydroxide"),
    "Reservoir": ("Reservoir", "Reservoirs"),
    "Porosity": ("Porosity",),
    "Permeability": ("Permeability",),
    # 政策主体：它们是真正的监管实体，但同一机构有 4-5 种写法。
    # 并成一个节点比按 organization 类型删掉更有价值。
    "IPCC": (
        "IPCC",
        "Intergovernmental Panel on Climate Change",
        "International Panel on Climate Change",
        "IPCC Panel on Climate Change",
        "Panel on Climate Change",
    ),
    "IEA": ("IEA", "International Energy Agency"),
    "Geological Carbon Storage": (
        "Geological Carbon Storage",
        "Geologic Carbon Storage",
        "Geologic carbon storage",
        "GCS",
    ),
}

# ─── 脏节点识别 ────────────────────────────────────────────
#
# 每条规则对应一类**明确不是知识**的字符串。刻意保守：宁可放过，不可错杀。
# 命中即删，且删除原因会写进报告（不允许静默消失）。
_NOISE_RULES: tuple[tuple[str, re.Pattern], ...] = (
    ("doi", re.compile(r"^(?:doi\s*:|https?://|www\.)", re.I)),
    ("email", re.compile(r"[\w.\-+]+@[\w\-]+\.[\w.]+")),
    ("date-label", re.compile(r"^(?:received|accepted|published|revised)\s+date\b", re.I)),
    ("volume-issue", re.compile(r"^(?:volume|vol\.?|issue|no\.)\s*\d+", re.I)),
    ("section-label", re.compile(
        r"^(?:a\s+r\s+t\s+i\s+c\s+l\s+e\b|abstract|references|acknowledg|"
        r"keywords|introduction|conclusions?)$", re.I)),
    ("citation", re.compile(r"\bet\s+al\b", re.I)),
    ("year", re.compile(r"^\d{4}[a-z]?$")),
    ("decade", re.compile(r"^\d{4}s$")),
    ("bare-quantity", re.compile(
        r"^\d+(?:\.\d+)?\s*(?:m|km|cm|mm|mpa|kpa|pa|°c|k|yr|years?|ma|%)$", re.I)),
    ("street-address", re.compile(
        r"\b(?:bldg|rd\.|street|ave\.?|avenue|blvd|suite|p\.?o\.?\s*box)\b", re.I)),
    ("postal-code", re.compile(r"\b[A-Z]{2}\s+\d{5}(?:-\d{4})?\b")),
    ("affiliation", re.compile(
        r"\b(?:universit(?:y|é|e)|institute|institut|college\s+of|department\s+of|"
        r"dept\.?\s+of|faculty\s+of|laborator(?:y|ies)|academy\s+of|"
        r"school\s+of|cent(?:er|re)\s+for)\b", re.I)),
    ("publisher", re.compile(
        r"\b(?:elsevier|springer|wiley|ieee|mdpi|frontiers|publishers?)\b", re.I)),
    ("journal", re.compile(r"\bjournal\b", re.I)),
    ("meta-filler", re.compile(
        r"^(?:this\s+study|this\s+paper|current\s+models?|empirical\s+results?|"
        r"additional\s+measurements?|unreliable\s+results?|co-?author|"
        r"corresponding\s+author|author)$", re.I)),
    # 资助 / 行政机构。它们和"作者单位"是同一类噪声：一篇论文的资助方会把
    # 几十篇不相关的论文连成一个假枢纽，连的是**经费关系**，不是地质知识。
    #
    # 为什么单独一条而不是把 `organization` 整类删：`IPCC`、`IEA` 也是
    # organization，但它们是论文里被讨论的**监管主体**。区分标准是"资助者
    # 还是被讨论者"，名字规则能分开这两者（ministry / alliance 只出现在前者），
    # 类型不能。
    #
    # 命中面里有一个必须保护的例外：`International Energy Agency` 含 agency，
    # 但它在别名表里（并成 IEA）。所以 `noise_reason` 里有一道 `_PROTECTED`
    # 前置判断 —— 别名表声明过的实体，任何名字规则都不许删。
    ("funding-body", re.compile(
        r"\b(?:federal\s+ministry|ministry\s+of|research\s+alliance|"
        r"economic\s+affairs|funding\s+agency)\b", re.I)),
)

# 国家/地区/城市。它们会变成高连接度"假枢纽"：几乎每篇论文都提到 Poland 或
# China，于是这个节点把不同论文连在一起 —— 连的是**案例地点**，不是地质知识。
# 只收语料里实际出现过的，不做全量地名表。
_GEO_POLITICAL = frozenset({
    "poland", "polands", "china", "germany", "usa", "united states", "u.s.",
    "thailand", "europe", "european union", "texas", "asia", "east asia",
    "college station", "beijing", "krakow", "oslo", "austin", "berlin",
    "rome", "turin", "padua", "manchester", "north dakota", "louisiana",
})

# 按实体类型整类丢弃。只收**新 prompt 根本不会产出**的类型：
# person / creature / content / publication 在 CCUS 语料里没有知识价值。
#
# `organization` 故意不在其中：它 95% 是作者单位，但 `IPCC`、`International
# Energy Agency` 这类**政策主体**也被标成 organization —— 那是真知识，
# 一刀切会连同监管实体一起删掉。机构类噪音交给上面的 affiliation / publisher
# 名字规则处理，剩下的少数（IPCC、IEA）用别名表并成单节点。
#
# `location` 同理不收：地层名（Upper Rotliegend Group）也可能被标成 location。
# `creature` 也不收：模型会把它当垃圾桶用 —— 实测把 `Reinforcement Learning
# Agent` 标成了 creature。类型不可信时按类型删就会误杀真知识，所以只信名字规则。
_NOISE_TYPES = frozenset({"person", "publication", "content"})

# CJK 单字信息密度高：`井筒`、`储层`、`断层`、`盖层` 都只有两个字，
# 却整张图最核心的实体。长度阈值只对拉丁字母生效。
_HAS_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
# 化学式白名单：H2 / CO / O2 这类两字符名字是合法实体。
_FORMULA = re.compile(r"^[A-Z][A-Za-z]?\d$")

# 超过这个词数基本就是论文标题而不是概念。
_MAX_NAME_WORDS = 10


def _singularize_word(word: str) -> str:
    """把单个英文词归约成单数。只处理低风险后缀，其余原样返回。"""
    if len(word) <= 3 or word.isupper():
        return word
    if word.endswith(("ss", "us", "is")):
        return word
    for pattern, repl in _PLURAL_RULES:
        if pattern.search(word):
            if repl == "y":
                return word[:-3] + "y"
            if repl == "es":
                return word[:-2]
            if repl == "s":
                return word[:-2] if word.endswith("ses") else word[:-1]
            return word[:-1]
    return word


def normalize_key(name: str) -> str:
    """
    把实体名压成一个用于分组的键。

    顺序很重要：先剥引文后缀，再剥括号注释 —— 反过来的话
    ``Digital Shadow (Gahlot, Orozco, et al. 2024)`` 会先被括号规则吃掉
    ``(Gahlot, Orozco, et al. 2024)`` 而留下正确结果，看似等价；
    但 ``Calcite (Calcium Carbonate)`` 里没有年份，两条规则结果相同，
    而 ``X (Y et al. 2024)`` 这种嵌套写法只有"先引文后括号"才是对的。
    """
    s = str(name).strip()
    s = _CITATION_SUFFIX.sub("", s)
    s = _PAREN_GLOSS.sub(" ", s)
    s = s.replace("’", "'").replace("`", "'")
    s = s.lower()
    s = re.sub(r"[^\w\s\-']", " ", s)          # 去掉标点，保留连字符和下划线
    s = re.sub(r"[\s\-_]+", " ", s).strip()
    s = s.replace("'s", "s").replace("'", "")
    for old, new in _KEY_REWRITES:
        s = s.replace(old, new)
    words = [_singularize_word(w) for w in s.split()]
    return " ".join(words).strip()


def _alias_lookup() -> dict[str, str]:
    table: dict[str, str] = {}
    for canonical, variants in CANONICAL_GROUPS.items():
        table[normalize_key(canonical)] = canonical
        for variant in variants:
            table[normalize_key(variant)] = canonical
    return table


_ALIASES = _alias_lookup()


def protected_names() -> frozenset[str]:
    """
    别名表里出现过的**全部**写法（规范化后）。

    它们代表"我们明确认定这是知识实体"—— 所以任何按名字的删除规则都不许
    删它们。这不是防御性编程，是修一个真实的矛盾：``_ALIASES`` 把
    ``International Energy Agency`` 收成 ``IEA``（政策主体，要保留），
    而"删掉资助机构"这条规则会命中 ``agency`` 一词，把它当噪音删掉。
    两条规则各自都对，撞在一起就成了"静默删掉刻意保留的实体"。
    """
    out: set[str] = set()
    for canonical, variants in CANONICAL_GROUPS.items():
        out.add(normalize_key(canonical))
        for variant in variants:
            out.add(normalize_key(variant))
    return frozenset(out)


_PROTECTED = protected_names()


def canonical_name(name: str) -> str:
    """返回实体名的规范写法（若在别名表里），否则返回原文去空白。"""
    key = normalize_key(name)
    return _ALIASES.get(key, str(name).strip())


def noise_reason(name: str, entity_type: str | None = None) -> str | None:
    """
    判断一个节点是不是"脏数据"。命中返回原因，否则返回 None。

    ``entity_type`` 可选：传入时会额外按类型丢弃（作者、机构、出版物…）。
    """
    raw = str(name).strip()
    if not raw:
        return "empty"

    # 别名表声明过的实体一律不删 —— 它在建图阶段就被认定为"这是知识"，
    # 任何按名字的删除规则与它冲突时，都是规则错了，不是实体错了。
    # 放在类型判断之前：类型是模型标的，别名表是人定的，人定的优先。
    if normalize_key(raw) in _PROTECTED:
        return None

    # 容错：调用方常直接把手里的节点属性字典传进来（graphml 读出来就是
    # {"entity_type": ..., "description": ...}），这里统一取出来，
    # 免得每个调用点都写一遍 .get("entity_type")。
    if isinstance(entity_type, dict):
        entity_type = entity_type.get("entity_type") or ""
    etype = str(entity_type or "").strip().lower()
    if etype in _NOISE_TYPES:
        return f"type:{etype}"

    for reason, pattern in _NOISE_RULES:
        if pattern.search(raw):
            return reason

    if raw.lower().strip(".'\"") in _GEO_POLITICAL:
        return "geo-political"

    if (
        len(raw) <= 2
        and not _HAS_CJK.search(raw)
        and not _FORMULA.match(raw)
    ):
        return "too-short"

    if len(raw.split()) > _MAX_NAME_WORDS:
        return "title-like"

    return None


def noise_rule_names() -> tuple[str, ...]:
    """
    全部**可能的**噪声原因，包括本语料一次都没命中的那些。

    为什么需要它：``build_merge_plan`` 只返回命中项，于是"某条规则从没命中过"
    与"这条规则不存在"在结果里长得一样。报告必须能区分这两件事 ——
    一条零命中的规则不是"已验证有效"，它只是"还没被验证"。
    """
    # 顺序与 noise_reason() 的判定顺序一致：空名 → 类型 → 名字规则 → 地名 →
    # 过短 → 标题式。报告里照这个顺序列，读的人才能顺着代码核对。
    reasons = ["empty"]
    reasons += [f"type:{etype}" for etype in sorted(_NOISE_TYPES)]
    reasons += [reason for reason, _pattern in _NOISE_RULES]
    reasons += ["geo-political", "too-short", "title-like"]
    return tuple(reasons)


def build_merge_plan(
    names,
    types: dict[str, str] | None = None,
) -> dict:
    """
    生成合并计划。

    参数
    ----
    names : 迭代器，图中的全部实体名
    types : 可选的 {name: entity_type} 映射

    返回
    ----
    ``{"merges": [{"target": str, "sources": [str, ...]}], "noise": [...],
       "kept": int}``

    ``sources`` **总是包含 target** —— LightRAG 的 ``amerge_entities`` 接受
    "把一批节点并进 target"，target 已在其中是最稳的用法（否则要依赖它
    在合并过程中新建节点）。

    选 target 的优先级：
    1. 别名表里的规范名（且确实存在于图中）；
    2. 名字最短的那个（``CO2`` 胜过 ``Carbon dioxide (CO2)``）；
    3. 字典序，保证结果可复现（集合迭代顺序在不同 Python 版本/哈希种子下会变）。
    """
    types = types or {}
    noise: list[dict] = []
    groups: dict[str, list[str]] = {}
    preferred: dict[str, str] = {}

    for name in names:
        reason = noise_reason(name, types.get(name))
        if reason:
            noise.append({"name": name, "reason": reason})
            continue
        # 关键：分组键必须**先过别名表**，否则 `CO2` 和 `Carbon Dioxide (CO2)`
        # 的键分别是 "co2" 和 "carbon dioxide"，永远落不进同一组。
        # 第一版就漏了这一步，导致别名表只影响"选哪个名字当 target"，
        # 却没能把变体归到一起 —— 合并组数虚高、净减虚低。
        key = normalize_key(name)
        canonical = _ALIASES.get(key)
        if canonical:
            key = normalize_key(canonical)
            preferred[key] = canonical
        groups.setdefault(key, []).append(name)

    merges: list[dict] = []
    kept = 0
    for key, members in groups.items():
        members = sorted(set(members))
        if len(members) == 1:
            kept += 1
            continue
        target = preferred.get(key)
        if target not in members:
            target = min(members, key=lambda m: (len(m), m))
        merges.append({"target": target, "sources": members})

    merges.sort(key=lambda m: m["target"])
    noise.sort(key=lambda n: n["name"])
    return {"merges": merges, "noise": noise, "kept": kept}


def plan_summary(plan: dict) -> str:
    """人类可读的一行摘要，用于日志和报告。"""
    merges = plan["merges"]
    sources = sum(len(m["sources"]) for m in merges)
    return (
        f"合并组 {len(merges)} 个（覆盖 {sources} 个节点，净减 {sources - len(merges)}），"
        f"脏节点 {len(plan['noise'])} 个，保持不变 {plan['kept']} 个"
    )
