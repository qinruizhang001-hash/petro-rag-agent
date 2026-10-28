"""知识库证据构建器（纯函数，零重依赖）
=====================================
把 LightRAG ``aquery_data()`` 返回的 ``data`` 段，转换成**带出处**的证据卡。

为什么不能复用 ``fragment_builder.build_fragments``
--------------------------------------------------
两个模块的不变量是**相反**的：

* ``build_fragments`` 的任务是"让 bge-reranker 好区分"。它把两端实体的完整
  描述拼进关系片段，产出一串长文本 —— 这一步**主动消灭出处**：拼接之后
  "这句话来自哪篇论文"已经无从得知，因为一个片段里混了三个来源。
  那是对的，reranker 只看语义。
* 本模块的任务是"让人能回查"。一条证据一张卡，**绝不合并**，每条卡都带着
  ``file_path`` 与 ``reference_id``。合并在这里不是优化，是数据损坏。

把两者放在同一个模块里，迟早有人为了"少写几行"在证据路径上复用拼接函数，
而结果不会报错 —— 只是界面上每条引用都指向同一篇文献。所以宁可多一个文件。

``reference_id`` 的作用域（最容易被误用的一点）
----------------------------------------------
``reference_id`` 是 **LightRAG 在一次响应内**给来源编的号（"1"、"2"…），
排序依据是该文件在本次结果里出现的次数。它**不是**全局文档 ID：

* 同一个 ``reference_id`` 在两次查询里指向不同文献；
* 同一篇文献在两次查询里的编号也不同。

所以界面上 ``[3]`` 只能解释为"本次检索的第 3 个来源"，不能存下来当书签。
要长期指向一篇文献，用 ``file_path``（那是入库时写入的血缘，稳定）。

``unknown_source`` 必须被数出来，不能被吞掉
------------------------------------------
入库时没传 ``file_paths`` 的文档，LightRAG 会把血缘记成 ``unknown_source``。
这种证据在界面上**看起来和正常证据一模一样**，只是文件名那一栏是空的。
本模块把它单独计数（``stats.unsourced``），因为"12 条证据里 5 条说不出来源"
与"12 条证据全部可回溯"是两种完全不同的可信度，而界面上少一个文件名
不构成任何提示。

"语义邻近"不等于"能当证据用"
----------------------------
PDF 抽出来的公式块（``𝑡+1 = 𝐶(𝑥𝑡)𝑥𝑡+1 + 𝐷(𝑥𝑡)𝑢𝑡`` 这类）会**排到第一位**：
它满篇都是 "CO₂ storage / pressure / mole fraction"，向量相似度确实高。
但它对"盖层完整性"这个问题提供不了任何机理证据 —— 它证明的是"这篇论文用了
这个离散格式"。

本模块因此给每条切片算一个 ``text_quality``，把"以公式为主"标出来
（``stats.low_quality_chunks`` 单独计数）。**但不重排**：

1. 引用编号 ``[n]`` 必须等于检索名次，否则报告里写的 ``[1]`` 无法回到
   检索日志里核对 —— 那会让"每个数字可回代"这条约束在证据层失效。
2. 重排会让面板与评估口径分家：RAGAS 用的是检索器的原始顺序，
   面板显示另一个顺序，两边的"第一条证据"就不是同一条。

所以这里的处理与 ``indeterminate`` 一致：不改结论，把"这条证据能证明什么"
说清楚，交给读者判断。
"""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional

#: LightRAG 用来拼接多值字段的分隔符（``source_id`` 里是多条 chunk id）。
GRAPH_SEP = "<SEP>"

#: 入库时没写血缘的文档会拿到这个值。它不是"文件名恰好叫这个"，
#: 而是"不知道"—— 见模块文档。
UNKNOWN_SOURCE = "unknown_source"

# ─── 文本可读性 ───
#: 数学字母数字符号（``𝑡`` ``𝐶`` ``𝑥``）。PDF 公式抽出来就是这一片码位，
#: 浏览器缺字形时还会显示成豆腐块 —— 这也是它必须被单独识别的原因之一。
_MATH_ALNUM = re.compile(r"[\U0001D400-\U0001D7FF]")
#: 中日韩统一表意文字 + 拉丁字母。真正的"能读成句子"的部分。
_LETTERS = re.compile(r"[\u4e00-\u9fffA-Za-z]")
_WHITESPACE = re.compile(r"\s")

#: 判定"以公式为主"的两条线。**在实测样本上校准，不是通用常数**：
#: 12 条真实切片里，纯散文的 9 条落在 92–97 % 字母 / 0 % 数学符号；
#: 两条公式块落在 46.5 %/37.5 % 与 72.0 %/21.1 %；
#: 一条"散文夹少量行内公式"落在 84.9 %/7.8 %，**必须不被标出**。
#: 所以线画在 80 % 与 15 %，两侧都有余量。语料换了要重新校准 ——
#: 这两条线描述的是"这份 PDF 语料的抽取质量"，不是物理规律。
MIN_LETTER_RATIO = 0.80
MAX_MATH_RATIO = 0.15

#: 单个描述字段的落盘上限（字符）。超出部分丢弃并在卡上标 ``truncated``。
#:
#: 为什么要截：实体描述是多篇论文合并后的长文本，25 个实体全量下发能到 40 KB。
#: 但**不能静默截** —— 卡上必须带 ``truncated: true``，界面上标"已截断"，
#: 否则读者会以为这就是全部描述，而"这条实体其实还有别的出处"这个信息就丢了。
DESC_LIMIT = 1200

#: chunk 正文上限。入库时 ``chunk_token_size=400``，中文约 600–900 字符，
#: 一般不会触发；留着是防有人改了入库参数之后把整个 PDF 段塞进响应。
CONTENT_LIMIT = 2000


def _split_ids(raw: Any) -> List[str]:
    """``source_id`` → chunk id 列表。

    LightRAG 用 ``<SEP>`` 拼接。这里只做切分与去空，不校验 chunk 是否存在 ——
    校验需要读存储，那会让本模块不再是纯函数。
    """
    if not raw:
        return []
    if isinstance(raw, (list, tuple)):
        items: Iterable[Any] = raw
    else:
        items = str(raw).split(GRAPH_SEP)
    return [str(x).strip() for x in items if str(x).strip()]


def normalize_source(raw: Any) -> str:
    """把 ``file_path`` 归一成"可展示的文件名"，认不出来返回空串。

    返回空串而不是 ``"unknown_source"``：下游要靠"空"来统计无出处证据。
    把哨兵值原样透出去，前端就得自己知道这个魔法字符串 —— 而知道的人不会
    写错，不知道的人会把 ``unknown_source`` 当成一篇文献渲染出来。
    """
    name = str(raw or "").strip()
    if not name or name == UNKNOWN_SOURCE:
        return ""
    return name


def split_sources(raw: Any) -> List[str]:
    """``file_path`` → 文件列表。

    **实体的 ``file_path`` 是多值的**。LightRAG 把多篇论文里同名实体的描述
    合并成一条，血缘也就拼在一起：实测 ``CO2 Leakage`` 这个实体的
    ``file_path`` 是 16 个文件名用 ``<SEP>`` 连成的一个字符串。

    第一版直接把它当单个文件名塞进 ``source`` 字段，界面上那一栏会显示成
    ［ccus_auto_01.pdf<SEP>ccus_auto_02.pdf<SEP>…］—— 一个既不像文件名、
    也不能点开的东西。更糟的是它看起来**像**一个文件名，读者会以为
    "这条实体只来自一篇，只是名字特别长"。

    切片没有这个问题：一个 chunk 只属于一篇文档。
    """
    if not raw:
        return []
    if isinstance(raw, (list, tuple)):
        items = raw
    else:
        items = str(raw).split(GRAPH_SEP)
    out: List[str] = []
    for x in items:
        name = normalize_source(x)
        if name and name not in out:
            out.append(name)
    return out


def _clip(text: str, limit: int) -> tuple[str, bool]:
    text = (text or "").strip()
    if len(text) <= limit:
        return text, False
    return text[:limit].rstrip() + "…", True


def text_quality(text: str) -> Dict[str, Any]:
    """一条切片的"能不能读成句子"。

    返回 ``readable_ratio``（字母占非空白字符的比例）、``math_ratio``
    （数学字母数字符号占比）与 ``low``（是否以公式为主）。

    为什么用**字符占比**而不是"有没有等号"：散文里也常有 ``=``（"注入量 = 速率 × 时间"），
    而公式块的特征是**大量**数学斜体符号，不是某一个符号。实测两条公式块
    的 ``math_ratio`` 是 21 % 与 37 %，纯散文是 0 %，中间没有灰区。
    """
    stripped = _WHITESPACE.sub("", text or "")
    n = len(stripped)
    if n == 0:
        # 空切片不该被当成"低质量"：它是"没内容"，两者在界面上的处置不同。
        return {"readable_ratio": 0.0, "math_ratio": 0.0, "low": False}
    letters = len(_LETTERS.findall(stripped))
    math = len(_MATH_ALNUM.findall(stripped))
    readable = letters / n
    math_ratio = math / n
    return {
        "readable_ratio": round(readable, 4),
        "math_ratio": round(math_ratio, 4),
        "low": math_ratio >= MAX_MATH_RATIO or readable < MIN_LETTER_RATIO,
    }


def build_chunks(data: Dict[str, Any], limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """原文切片卡 —— 知识库里**唯一可引用**的一类证据。

    实体与关系是模型从原文抽出来的中间产物，可以做"图谱命中"展示，
    但读者要核对"这句到底是不是这么写的"时只能看原文。所以引用角标
    ``[n]`` 只挂在切片卡上。
    """
    out: List[Dict[str, Any]] = []
    for chunk in (data or {}).get("chunks") or []:
        content = (chunk.get("content") or "").strip()
        if not content:
            continue
        body, truncated = _clip(content, CONTENT_LIMIT)
        quality = text_quality(content)
        out.append(
            {
                "kind": "chunk",
                "reference_id": str(chunk.get("reference_id") or "").strip(),
                "chunk_id": str(chunk.get("chunk_id") or "").strip(),
                "source": normalize_source(chunk.get("file_path")),
                "content": body,
                "chars": len(content),
                "truncated": truncated,
                # 质量按**完整正文**算，不按截断后的算：截断会把后半段的散文
                # 切掉，让一条正常切片因为前 2000 字里公式多而显得"以公式为主"。
                "text_quality": quality,
            }
        )
        if limit is not None and len(out) >= limit:
            break
    return out


def build_entities(data: Dict[str, Any], limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """图谱实体卡。

    ``sources`` 是这条实体**被抽出来的全部文档**（合并实体必然多来源），
    ``chunk_ids`` 是支撑它的切片。两者数量不对应：一个来源可能贡献多个切片。
    """
    out: List[Dict[str, Any]] = []
    for ent in (data or {}).get("entities") or []:
        name = (ent.get("entity_name") or "").strip()
        if not name:
            continue
        desc, truncated = _clip(ent.get("description") or "", DESC_LIMIT)
        out.append(
            {
                "kind": "entity",
                "name": name,
                "entity_type": (ent.get("entity_type") or "UNKNOWN").strip() or "UNKNOWN",
                "description": desc,
                "sources": split_sources(ent.get("file_path")),
                "chunk_ids": _split_ids(ent.get("source_id")),
                "truncated": truncated,
            }
        )
        if limit is not None and len(out) >= limit:
            break
    return out


def build_relationships(data: Dict[str, Any], limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """图谱关系卡。``keywords`` 在存储里是逗号分隔的字符串，这里切成列表。"""
    out: List[Dict[str, Any]] = []
    for rel in (data or {}).get("relationships") or []:
        src = (rel.get("src_id") or "").strip()
        tgt = (rel.get("tgt_id") or "").strip()
        desc = (rel.get("description") or "").strip()
        if not (src or tgt or desc):
            continue
        body, truncated = _clip(desc, DESC_LIMIT)
        keywords = [
            k.strip()
            for k in str(rel.get("keywords") or "").replace("，", ",").split(",")
            if k.strip()
        ]
        out.append(
            {
                "kind": "relationship",
                "src": src,
                "tgt": tgt,
                "keywords": keywords,
                "description": body,
                "sources": split_sources(rel.get("file_path")),
                "truncated": truncated,
            }
        )
        if limit is not None and len(out) >= limit:
            break
    return out


def build_references(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """来源表：``reference_id`` → 文件名，按首次出现顺序。

    优先用 LightRAG 自己给的 ``data.references``；那个字段缺失时（老版本、
    或空结果）从切片卡重建。**不合并编号**：同一个文件可能占多个
    ``reference_id``（LightRAG 按出现次数排，不去重文件），重建时保持原样，
    因为编号是响应内的锚点，改了就与正文里的角标对不上。
    """
    seen: set[str] = set()
    table: List[Dict[str, Any]] = []
    for card in chunks:
        rid = card.get("reference_id") or ""
        if not rid or rid in seen:
            continue
        seen.add(rid)
        table.append({"reference_id": rid, "source": card.get("source") or ""})
    return table


def build_evidence(data: Dict[str, Any], *, with_graph: bool = True) -> Dict[str, Any]:
    """``data`` 段 → 一次检索的完整证据包。

    ``with_graph=False`` 时只返回原文切片 —— 用于"只要引用、不要图谱噪音"
    的窄面板。默认带上，因为"为什么召回了这条"往往要靠实体/关系解释。
    """
    chunks = build_chunks(data)
    entities = build_entities(data) if with_graph else []
    relationships = build_relationships(data) if with_graph else []

    # 无出处统计只在切片上算：实体/关系的 file_path 是"首次出现的文件"，
    # 拿它算可追溯率会高估 —— 一条实体可能同时来自 5 篇，这里只记了 1 篇。
    unsourced = sum(1 for c in chunks if not c["source"])
    # "以公式为主"的切片数。与 unsourced 并列，因为它回答的是同一个问题：
    # 这条证据**能不能支撑结论**，而不只是"有没有出处"。
    low_quality = sum(1 for c in chunks if c["text_quality"]["low"])

    return {
        "chunks": chunks,
        "entities": entities,
        "relationships": relationships,
        "references": build_references(chunks),
        "stats": {
            "chunks": len(chunks),
            "entities": len(entities),
            "relationships": len(relationships),
            "sources": len({r["source"] for r in build_references(chunks) if r["source"]}),
            # "说不出来源"的条数。界面必须显示，见模块文档。
            "unsourced": unsourced,
            "traceable": len(chunks) - unsourced,
            # "以公式为主、难以作为机理性证据"的条数。同样必须显示。
            "low_quality_chunks": low_quality,
        },
    }


# ══════════════════════════════════════════════════════════
# 反方向：把一份**已经构建好**的证据包按构建规则重新推导一遍
# ══════════════════════════════════════════════════════════

#: ``audit_evidence`` 认定的证据包顶层字段。多一个少一个都要报 ——
#: 与 ``stats`` 同理：字段集合变化是契约变化，不是"实现细节"。
EVIDENCE_KEYS = ("chunks", "entities", "relationships", "references", "stats")

#: ``stats`` 的完整字段集。多出来的键同样算问题：前端不认识的键意味着
#: 有人在别处加了计数而没同步界面，那个数字就永远不会被看见。
STATS_KEYS = (
    "chunks",
    "entities",
    "relationships",
    "sources",
    "unsourced",
    "traceable",
    "low_quality_chunks",
)


def _audit_chunk(index: int, card: Dict[str, Any], seen_ids: set) -> List[str]:
    """单张切片卡的自洽性。返回问题描述列表（空 = 没问题）。"""
    tag = f"chunks[{index}]"
    problems: List[str] = []

    if card.get("kind") != "chunk":
        problems.append(f"{tag}.kind 应为 'chunk'，实际是 {card.get('kind')!r}")

    ref_id = str(card.get("reference_id") or "")
    if not ref_id:
        # 没有编号的切片在界面上仍然会显示，只是正文里那句 [n] 再也回不到它。
        problems.append(f"{tag}.reference_id 为空 —— 正文角标无法回代到这条证据")

    chunk_id = str(card.get("chunk_id") or "")
    if not chunk_id:
        problems.append(f"{tag}.chunk_id 为空 —— 引用跳转找不到落点")
    elif chunk_id in seen_ids:
        problems.append(f"{tag}.chunk_id 重复（{chunk_id}）—— 同一片证据被算了两次")
    else:
        seen_ids.add(chunk_id)

    body = str(card.get("content") or "")
    if not body.strip():
        problems.append(f"{tag}.content 为空 —— build_chunks 本该跳过空切片")

    source = str(card.get("source") or "")
    if GRAPH_SEP in source:
        # 切片只属于一篇文档。出现分隔符说明有人把实体的多值血缘
        # （``split_sources`` 的输入）用在了切片上。
        problems.append(
            f"{tag}.source 里含 {GRAPH_SEP} —— 一个切片只属于一篇文档，"
            "出现分隔符说明多值血缘被当成单值用了"
        )

    chars = card.get("chars")
    truncated = bool(card.get("truncated"))
    if not isinstance(chars, int):
        problems.append(f"{tag}.chars 应是整数，实际是 {type(chars).__name__}")
    elif truncated:
        if chars <= CONTENT_LIMIT:
            problems.append(
                f"{tag} 标了 truncated 但 chars={chars} 没有超过上限 {CONTENT_LIMIT}"
            )
        if not body.endswith("…"):
            # 截断必须看得见。没有省略号时读者会以为这就是全文，
            # 而"这条证据其实还有后半段"这个信息就丢了。
            problems.append(f"{tag} 标了 truncated 但正文没有以省略号结尾")
    else:
        if chars != len(body):
            problems.append(
                f"{tag}.chars={chars} 与正文长度 {len(body)} 不一致"
                "（未截断时两者必须相等）"
            )
        # 质量分只在**未截断**时能重算：它是按完整正文算的，
        # 截断之后后半段散文没了，重算必然对不上。这不是遗漏，是信息不足。
        want_quality = text_quality(body)
        got_quality = card.get("text_quality")
        if got_quality != want_quality:
            problems.append(
                f"{tag}.text_quality 与正文重算结果不一致："
                f"存盘 {got_quality} / 重算 {want_quality}"
            )
    return problems


def _audit_graph_card(
    index: int, card: Dict[str, Any], kind: str, seen_names: set
) -> List[str]:
    """实体卡 / 关系卡的共同部分：``kind``、``sources``、``truncated``。"""
    tag = f"{'entities' if kind == 'entity' else 'relationships'}[{index}]"
    problems: List[str] = []

    if card.get("kind") != kind:
        problems.append(f"{tag}.kind 应为 {kind!r}，实际是 {card.get('kind')!r}")

    if kind == "entity":
        name = str(card.get("name") or "").strip()
        if not name:
            problems.append(f"{tag}.name 为空 —— 界面上会是一张没有标题的卡")
        elif name in seen_names:
            problems.append(
                f"{tag}.name 重复（{name}）—— 列表 key 冲突，界面只会渲染一条"
            )
        else:
            seen_names.add(name)
        if not str(card.get("entity_type") or "").strip():
            problems.append(f"{tag}.entity_type 为空")
        if not isinstance(card.get("chunk_ids"), list):
            problems.append(f"{tag}.chunk_ids 必须是列表（存储里是 <SEP> 拼的字符串）")
    else:
        keywords = card.get("keywords")
        if not isinstance(keywords, list):
            problems.append(f"{tag}.keywords 必须是列表")
        elif any(not str(k).strip() for k in keywords):
            problems.append(f"{tag}.keywords 里有空串 —— 界面上会是一个空标签")

    sources = card.get("sources")
    if not isinstance(sources, list):
        problems.append(
            f"{tag}.sources 必须是列表 —— 实体/关系的 file_path 天然是多值的"
        )
    else:
        if any(GRAPH_SEP in str(s) for s in sources):
            problems.append(f"{tag}.sources 里含 {GRAPH_SEP} —— 多值字段没切开")
        if len(set(map(str, sources))) != len(sources):
            problems.append(f"{tag}.sources 有重复项")

    desc = str(card.get("description") or "")
    if card.get("truncated"):
        if not desc.endswith("…"):
            problems.append(f"{tag} 标了 truncated 但描述没有以省略号结尾")
    elif len(desc) > DESC_LIMIT:
        problems.append(
            f"{tag} 没标 truncated 但描述有 {len(desc)} 字，超过上限 {DESC_LIMIT}"
        )
    return problems


def audit_evidence(evidence: Dict[str, Any]) -> List[str]:
    """把一份**已经构建好**的证据包按构建规则重算一遍，返回所有对不上的地方。

    为什么需要这个函数
    ------------------
    前端 fixture ``web/src/fixtures/search.json`` 必须由真实检索产出，而真实
    检索要 Ollama —— CI 里没有。于是这份 fixture 长期只有"开发机上重抓"这一
    条更新路径，CI 里既不比对也没人看。

    但证据包的一大半是**纯函数**：来源表由切片按首次出现顺序去重得来，
    ``stats`` 是切片/实体/关系的计数，``text_quality`` 由正文算出。
    这些都不需要模型。所以 CI 能查的不是"内容对不对"，而是
    **"这份文件自己跟自己一致吗"**。

    两者都不是小事。手改一个 ``reference_id``、删一条切片而忘了改 ``stats``、
    把 ``low`` 改成 ``false`` —— 界面会照样渲染，只是每一处引用都指错，
    或者那个"以公式为主"的提示凭空消失。这类损坏比"内容过期"更难发现：
    过期的内容至少和当时的后端是一致的，而被手改过的文件只和自己不一致。

    查不到的那一半要说清楚：这份证据**是不是当前后端会返回的那一份**，
    只有重跑检索才知道。调用方必须把这句话打印出来，不能拿本函数的
    "没问题"去顶替一次真正的比对。
    """
    problems: List[str] = []
    if not isinstance(evidence, dict):
        return [f"证据包应是对象，实际是 {type(evidence).__name__}"]

    missing = [k for k in EVIDENCE_KEYS if k not in evidence]
    if missing:
        # 缺字段时后面的重算全部无意义，直接返回。
        return ["evidence 缺少字段：" + "、".join(missing)]
    extra = [k for k in evidence if k not in EVIDENCE_KEYS]
    if extra:
        problems.append("evidence 多出字段：" + "、".join(sorted(extra)))

    for key in EVIDENCE_KEYS:
        if not isinstance(evidence[key], list) and key != "stats":
            return problems + [f"evidence.{key} 应是列表"]
    if not isinstance(evidence["stats"], dict):
        return problems + ["evidence.stats 应是对象"]

    chunks: List[Dict[str, Any]] = [c for c in evidence["chunks"] if isinstance(c, dict)]
    if len(chunks) != len(evidence["chunks"]):
        problems.append("evidence.chunks 里有不是对象的条目")

    seen_ids: set = set()
    for i, card in enumerate(chunks):
        problems.extend(_audit_chunk(i, card, seen_ids))

    seen_names: set = set()
    for i, card in enumerate(evidence["entities"]):
        if isinstance(card, dict):
            problems.extend(_audit_graph_card(i, card, "entity", seen_names))
        else:
            problems.append(f"entities[{i}] 不是对象")
    for i, card in enumerate(evidence["relationships"]):
        if isinstance(card, dict):
            problems.extend(_audit_graph_card(i, card, "relationship", seen_names))
        else:
            problems.append(f"relationships[{i}] 不是对象")

    problems.extend(_audit_references(evidence["references"], chunks))
    problems.extend(_audit_stats(evidence["stats"], chunks, evidence))
    return problems


def _audit_references(
    references: List[Dict[str, Any]], chunks: List[Dict[str, Any]]
) -> List[str]:
    """来源表必须**等于**切片按首次出现顺序去重的结果。

    这张表是 ``build_references`` 的纯函数输出，所以它没有任何"合理但不一致"
    的空间：编号、文件名、顺序三者都必须对得上。顺序也算，因为
    ``[3]`` 的语义是"本次检索的第 3 个来源"，重排会让正文里的角标指错行。
    """
    problems: List[str] = []
    for i, row in enumerate(references):
        if not isinstance(row, dict):
            problems.append(f"references[{i}] 不是对象")
            return problems

    want = build_references(chunks)
    have_ids = [str(r.get("reference_id") or "") for r in references]
    want_ids = [str(r["reference_id"]) for r in want]

    have_map = {str(r.get("reference_id") or ""): str(r.get("source") or "") for r in references}
    want_map = {str(r["reference_id"]): str(r["source"]) for r in want}

    for rid in want_ids:
        if rid not in have_map:
            problems.append(
                f"来源表缺少 reference_id={rid} —— 有切片挂着这个编号，正文里的 [{rid}] 无处可去"
            )
        elif have_map[rid] != want_map[rid]:
            problems.append(
                f"reference_id={rid} 指向 {have_map[rid]!r}，而切片里是 {want_map[rid]!r}"
            )
    used = {str(c.get("reference_id") or "") for c in chunks}
    for rid in have_ids:
        if rid not in used:
            problems.append(
                f"来源表里的 reference_id={rid} 没有任何切片使用 —— 死条目"
            )
    if set(have_ids) == set(want_ids) and have_ids != want_ids:
        problems.append(
            "来源表顺序不是切片首次出现顺序："
            f"存盘 {have_ids} / 应为 {want_ids}"
        )
    return problems


def _audit_stats(
    stats: Dict[str, Any],
    chunks: List[Dict[str, Any]],
    evidence: Dict[str, Any],
) -> List[str]:
    """``stats`` 的每一项都能从卡片数出来，所以逐项重算。

    ``sources`` 用 ``build_references(chunks)`` 重算，**不用**存盘的那张来源表 ——
    否则一处错会同时污染"被检查的对象"和"检查用的尺子"，两个错正好抵消。
    """
    problems: List[str] = []
    unsourced = sum(1 for c in chunks if not str(c.get("source") or ""))
    low = sum(1 for c in chunks if (c.get("text_quality") or {}).get("low") is True)
    want = {
        "chunks": len(chunks),
        "entities": len(evidence["entities"]),
        "relationships": len(evidence["relationships"]),
        "sources": len(
            {r["source"] for r in build_references(chunks) if r["source"]}
        ),
        "unsourced": unsourced,
        "traceable": len(chunks) - unsourced,
        "low_quality_chunks": low,
    }
    for key in STATS_KEYS:
        got = stats.get(key)
        if got != want[key]:
            problems.append(f"stats.{key}={got!r}，按卡片重算是 {want[key]}")
    extra = [k for k in stats if k not in STATS_KEYS]
    if extra:
        problems.append("stats 多出字段：" + "、".join(sorted(extra)))
    return problems
