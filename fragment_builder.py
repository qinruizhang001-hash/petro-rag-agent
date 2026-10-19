"""
GraphRAG 片段构建器（纯函数，零重依赖）
=====================================
把 LightRAG `aquery_data()` 返回的结构化结果，转换成供 bge-reranker 精排的
候选片段列表。

为什么要单独拆成模块：
- 这段逻辑是"检索质量"的核心（片段长度直接决定 rerank 区分度），
  必须能被离线单元测试覆盖，而不是埋在需要 Ollama 的检索链路里。
- 纯函数、无 IO、无第三方依赖 → CI 里可直接跑断言。

片段构建策略（提升 Rerank 区分度）：
1. 增强关系片段：把两端实体的完整描述拼进关系片段，形成
   "实体A描述 --[关系关键词]--> 实体B描述" 的长语义单元，
   每个片段同时携带实体语义和关系语义（多跳推理上下文完整）。
2. 孤立实体片段：未被任何关系覆盖到的实体单独保留，避免信息丢失。
3. 原始 chunk：文本块原样保留。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from core.temporal import parse_timestamp, stamp_entity_timestamp, ts_sort_key


def build_entity_map(entities: List[Dict[str, Any]]) -> Dict[str, str]:
    """实体索引：entity_name -> 完整描述串。"""
    entity_map: Dict[str, str] = {}
    for ent in entities or []:
        name = (ent.get("entity_name") or "").strip()
        if not name:
            continue
        etype = (ent.get("entity_type") or "UNKNOWN").strip() or "UNKNOWN"
        desc = (ent.get("description") or "").strip()
        entity_map[name] = f"[{etype}] {name}: {desc}" if desc else f"[{etype}] {name}"
    return entity_map


def build_entity_timestamps(entities: List[Dict[str, Any]]) -> Dict[str, str]:
    """
    实体名 → 时间戳（``YYYY`` / ``YYYY-MM`` / ``YYYY-MM-DD``）。

    时间从哪来：优先实体自带的 ``timestamp`` / ``created_at`` 字段，
    没有就从描述文本里认（"2025 年测得…"）。认不出来就不放进字典 ——
    一个猜出来的时间戳比没有时间戳更危险，它会让冲突检测得出错误结论。
    """
    out: Dict[str, str] = {}
    for ent in entities or []:
        name = (ent.get("entity_name") or "").strip()
        if not name:
            continue
        ts = stamp_entity_timestamp(ent)
        if ts:
            out[name] = ts
    return out


def _ts_prefix(ts: Optional[str]) -> str:
    """时间标注。放在类型标签**之后**，保持 ``[关系]`` / ``[实体]`` 前缀不变。"""
    return f"[时间:{ts}] " if ts else ""


def _newer_of(*stamps: Optional[str]) -> Optional[str]:
    """多个候选时间里取最新一个（关系片段两端实体可能来自不同年份）。"""
    valid = [s for s in stamps if s]
    if not valid:
        return None
    return max(valid, key=ts_sort_key)



def build_fragments(data: Dict[str, Any], with_timestamps: bool = True) -> List[str]:
    """
    把 LightRAG 的 data 段（entities / relationships / chunks）构建成片段列表。

    返回按"关系 → 孤立实体 → 原始 chunk"顺序排列、去重保序的字符串列表。

    ``with_timestamps``（默认开）：能认出时间点的片段会在类型标签后加上
    ``[时间:2025-06]``。这一步不改动片段正文，只在最前面加一个短标注 ——
    rerank 的输入因此多出十来个字符，换来的是下游可以按时间排序与判冲突。
    认不出时间的片段保持原样（不加 ``[时间:—]`` 之类的噪音）。
    """
    data = data or {}
    entities = data.get("entities") or []
    relationships = data.get("relationships") or []
    chunks = data.get("chunks") or []

    entity_map = build_entity_map(entities)
    entity_ts = build_entity_timestamps(entities) if with_timestamps else {}

    covered = set()
    docs: List[str] = []

    # ─── 1. 增强关系片段：拼接两端实体描述，扩充文本长度 ──
    for rel in relationships:
        src = (rel.get("src_id") or "").strip()
        tgt = (rel.get("tgt_id") or "").strip()
        desc = (rel.get("description") or "").strip()
        kw = (rel.get("keywords") or "").strip()
        if not (src or tgt or desc):
            continue
        src_part = entity_map.get(src) or src
        tgt_part = entity_map.get(tgt) or tgt
        edge = f"--[{kw}]-->" if kw else "-->"
        piece = f"{src_part} {edge} {tgt_part}"
        if desc:
            piece = f"{piece}: {desc}"
        ts = _newer_of(
            entity_ts.get(src),
            entity_ts.get(tgt),
            parse_timestamp(desc) if with_timestamps else None,
        )
        docs.append(f"[关系]{_ts_prefix(ts)} {piece}")
        covered.update({src, tgt})

    # ─── 2. 孤立实体片段（未被关系覆盖，避免与关系片段重复）──
    for name, full_desc in entity_map.items():
        if name not in covered:
            docs.append(f"[实体]{_ts_prefix(entity_ts.get(name))} {full_desc}")

    # ─── 3. 原始文档 chunk ─────────────────────────────
    for chunk in chunks:
        content = (chunk.get("content") or "").strip()
        if content:
            # chunk 正文必须原样保留（它是"证据原文"），只有 metadata 里
            # 明确带时间时才加标注，绝不从正文猜 —— 猜错等于篡改证据。
            ts = None
            if with_timestamps:
                for key in ("timestamp", "created_at", "time", "date"):
                    raw = chunk.get(key)
                    if isinstance(raw, str) and raw.strip():
                        ts = parse_timestamp(raw)
                        if ts:
                            break
            docs.append(f"[时间:{ts}] {content}" if ts else content)

    # ─── 4. 去重保序 ───────────────────────────────────
    seen = set()
    unique: List[str] = []
    for d in docs:
        if d and d not in seen:
            seen.add(d)
            unique.append(d)

    return unique


def extract_data(result: Dict[str, Any]) -> Dict[str, Any]:
    """从 LightRAG 完整响应里取出 data 段；非 success 时返回空 dict。"""
    if not result or result.get("status") != "success":
        return {}
    return result.get("data") or {}
