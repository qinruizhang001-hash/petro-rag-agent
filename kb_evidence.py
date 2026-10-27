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
