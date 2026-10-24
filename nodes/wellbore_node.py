"""
井筒分支：检索（retrieve_wellbore）+ 生成（generate_wellbore），套管腐蚀 / 压力异常
==============================================
并行扇出的右分支。结构与地质分支对称，但**检索的是另一个库**：

  ① ChromaDB 原文切片检索（向量相似度，无需 rerank）
  ② 用云端模型写成 150~250 字的井筒评估片段

为什么这里没有 rerank
--------------------
LightRAG 一次召回 120~150 个片段，必须靠 cross-encoder 才能挑出 Top-2；
而 ChromaDB 直接返回**已经按相似度排好序的 top-k**（这里 k=4），
再叠一层 rerank 只是把同一批结果重排一次，纯属浪费 2 秒 CPU。
少一层不是偷懒，是因为这一层在这个存储上不产生信息。

为什么查询词要拼上域关键词
--------------------------
向量检索对"查询措辞"很敏感。用户可能只写"某 CCUS 项目注入压力 250bar"，
里面一个井筒相关的词都没有 —— 直接拿这段去查原文库，召回的是泛泛的
封存段落。所以查询 = 用户参数 + 井筒域关键词，让检索方向明确落在
套管/水泥环/环空/腐蚀上。这是检索侧的工程手段，不是给模型"喂提示"。

异常策略同 geology：**永不抛出**，失败写进 ``wellbore_status``，
由汇总节点降级处理。
"""
from __future__ import annotations

import time

from chroma_retriever import retrieve_documents
from core.config import get_llm, llm_chat
from core.state import AgentState
from utils.retry import call_with_retry
from utils.tokens import count_tokens, merge_token_usage, record_llm_usage

from utils.logging_setup import get_logger

log = get_logger(__name__)

# 井筒域关键词：把"泛封存"查询拉向"井筒完整性"
WELLBORE_KEYWORDS = "井筒完整性 套管腐蚀 水泥环 环空带压 注入压力异常 热应力 密封失效"

WELLBORE_TOP_K = 4

SEGMENT_PROMPT = """你是一名 CCUS 井筒完整性工程师。下面是从文献库检索到的原文片段。

【已提取的项目参数】：
{params}

【文献片段】：
{evidence}

请写一段 **150~250 字**的井筒风险评估，只覆盖以下三点：
1. 套管与水泥环的腐蚀 / 退化风险（含 CO2 溶于水形成碳酸的影响）
2. 压力异常（环空带压、注入压力超过套管承压、温度引起的热应力）
3. 监测与缓解建议（井下压力计、水泥环完整性测井等）

硬性要求：
- 片段里出现的数值必须原样引用，不要自己换算或补充。
- 片段没提到的结论**不要编造**，直接写"文献片段未覆盖"。
- 只输出这段评估正文，不要标题、不要 JSON。"""


def _build_query(state: AgentState) -> str:
    """用户参数 + 井筒域关键词。参数为空时退回原始输入。"""
    params = (state.get("extracted_params") or "").strip()
    base = params or (state.get("user_input") or "").strip()
    return f"{base}\n{WELLBORE_KEYWORDS}".strip()


def retrieve_wellbore(state: AgentState) -> dict:
    """只召回井筒 Top-K；Grader 判定相关性前不触发结论生成。"""
    t0 = time.perf_counter()
    started = time.time()
    strategy = state.get("retrieval_strategy") or "primary"
    params = (state.get("extracted_params") or "").strip()
    original = (state.get("user_input") or "").strip()
    rewritten = (state.get("wellbore_query") or "").strip()
    if strategy == "expanded":
        query = "\n".join(part for part in (
            original, params, rewritten,
            "well integrity casing corrosion cement sheath annular pressure "
            "CO2 injection thermal stress sustained casing pressure",
            WELLBORE_KEYWORDS,
        ) if part)
        top_k = min(max(WELLBORE_TOP_K * 2, WELLBORE_TOP_K), 8)
        usage_stage = "chroma_retrieve_retry"
    else:
        query = "\n".join(part for part in (
            rewritten or "CO2 storage wellbore integrity casing cement annular pressure",
            params, original, WELLBORE_KEYWORDS,
        ) if part)
        top_k = WELLBORE_TOP_K
        usage_stage = "chroma_retrieve"

    try:
        docs = retrieve_documents(query, k=top_k)
        evidence = "\n\n".join(docs)
        usage = merge_token_usage(
            state,
            usage_stage,
            {
                "query": query,
                "query_tokens": count_tokens(query),
                "candidate_docs": len(docs),
                "candidate_tokens": sum(count_tokens(doc) for doc in docs),
                "injected_context_tokens": count_tokens(evidence),
                "top_k": len(docs),
                "engine": "chromadb",
                "retrieval_strategy": strategy,
            },
        )
        status = {
            "ok": bool(docs), "started_at": round(started, 3),
            "ended_at": round(time.time(), 3),
            "elapsed_s": round(time.perf_counter() - t0, 2),
            "engine": "chromadb", "candidates": len(docs),
            "retrieval_strategy": strategy, "top_k": top_k,
            "error": "" if docs else "ChromaDB 未召回候选片段",
        }
        return {
            "token_usage": usage,
            "wellbore_evidence": evidence,
            "wellbore_candidates": [{"text": doc} for doc in docs],
            "wellbore_status": status,
        }
    except Exception as exc:  # noqa: BLE001 — parallel retrieval must degrade to state
        elapsed = time.perf_counter() - t0
        log.error("[Wellbore Retrieve] %s: %s", type(exc).__name__, exc)
        return {
            "wellbore_evidence": "", "wellbore_candidates": [],
            "wellbore_status": {
                "ok": False, "started_at": round(started, 3),
                "ended_at": round(time.time(), 3), "elapsed_s": round(elapsed, 2),
                "engine": "chromadb", "candidates": 0,
                "retrieval_strategy": strategy, "top_k": top_k,
                "error": f"{type(exc).__name__}: {exc}",
            },
        }


def generate_wellbore(state: AgentState) -> dict:
    """仅使用 Self-RAG 接受的井筒证据生成结论。"""
    t0 = time.perf_counter()
    status = dict(state.get("wellbore_status") or {})
    grade = (state.get("retrieval_grade") or {}).get("wellbore") or {}
    if not grade.get("accepted"):
        error = grade.get("reason") or "Self-RAG 未接受井筒证据"
        status.update({
            "ok": False, "ended_at": round(time.time(), 3),
            "elapsed_s": round(float(status.get("elapsed_s") or 0) +
                                (time.perf_counter() - t0), 2),
            "grading_error": f"井筒证据未通过相关性评估，已阻止无依据生成：{error}",
        })
        if not status.get("error"):
            status["error"] = status["grading_error"]
        return {"wellbore_report": "", "wellbore_status": status}

    evidence = (state.get("wellbore_evidence") or "").strip()
    try:
        mode = state.get("complexity_mode") or "cloud"
        _, model, tier, _ = get_llm("wellbore", mode)
        resp = call_with_retry(
            llm_chat,
            "wellbore",
            messages=[{
                "role": "user",
                "content": SEGMENT_PROMPT.format(
                    params=state.get("extracted_params", "") or state.get("user_input", ""),
                    evidence=evidence or "（无）",
                ),
            }],
            temperature=0.2,
            mode=mode,
            label="wellbore 段生成",
        )
        segment = (resp.choices[0].message.content or "").strip()
        status.update({
            "ok": bool(segment), "ended_at": round(time.time(), 3),
            "elapsed_s": round(float(status.get("elapsed_s") or 0) +
                                (time.perf_counter() - t0), 2),
            "chars": len(segment), "error": "" if segment else "模型返回空内容",
        })
        return {
            **record_llm_usage(state, "wellbore", resp, model, tier),
            "wellbore_report": segment,
            "wellbore_evidence": evidence,
            "wellbore_status": status,
        }
    except Exception as exc:  # noqa: BLE001 — preserve branch degradation contract
        status.update({
            "ok": False, "ended_at": round(time.time(), 3),
            "elapsed_s": round(float(status.get("elapsed_s") or 0) +
                                (time.perf_counter() - t0), 2),
            "chars": 0, "error": f"{type(exc).__name__}: {exc}",
        })
        log.error("[Wellbore Generate] %s: %s", type(exc).__name__, exc)
        return {"wellbore_report": "", "wellbore_evidence": evidence,
                "wellbore_status": status}
