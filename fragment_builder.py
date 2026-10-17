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

from typing import Any, Dict, List


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


def build_fragments(data: Dict[str, Any]) -> List[str]:
    """
    把 LightRAG 的 data 段（entities / relationships / chunks）构建成片段列表。

    返回按"关系 → 孤立实体 → 原始 chunk"顺序排列、去重保序的字符串列表。
    """
    data = data or {}
    entities = data.get("entities") or []
    relationships = data.get("relationships") or []
    chunks = data.get("chunks") or []

    entity_map = build_entity_map(entities)

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
        docs.append(f"[关系] {piece}")
        covered.update({src, tgt})

    # ─── 2. 孤立实体片段（未被关系覆盖，避免与关系片段重复）──
    for name, full_desc in entity_map.items():
        if name not in covered:
            docs.append(f"[实体] {full_desc}")

    # ─── 3. 原始文档 chunk ─────────────────────────────
    for chunk in chunks:
        content = (chunk.get("content") or "").strip()
        if content:
            docs.append(content)

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
