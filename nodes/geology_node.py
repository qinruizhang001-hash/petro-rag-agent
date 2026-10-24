"""
地质分支：检索（retrieve_geology）+ 生成（generate_geology），断层 / 盖层风险
=========================================
并行扇出的左分支。职责单一：**拿到地质证据，并把它变成一段可被汇总的结论**。

  ① LightRAG 图谱检索（实体/关系/chunk）→ bge-reranker 精排 → 上下文压缩
  ② 证据不足时联网补料（原 ``should_search`` 的规则原样搬进来）
  ③ 用云端模型写成 150~250 字的地质评估片段

为什么检索这一层不重写
----------------------
``nodes/retrieve_node.py`` 里的 ``retrieve_knowledge`` 已经是这套检索的
唯一实现（含 rerank 分数驱动的复杂度升级、分块压缩、token 记账）。
这里**调用**它而不是复制一遍 —— 复制出来的第二份会在某次调参后悄悄分叉，
而分叉的表现是"两条分支的检索质量不一样"，没有任何报错。

异常策略：不抛出
----------------
并行分支最怕的是"一条挂了整张图失败"：LangGraph 里节点抛异常会让整个
superstep 失败，另一条分支已经算完的结果也一起丢掉。所以本节点把
**所有**异常收进 ``geology_status``，正常返回 —— 汇总节点据此降级，
在报告中提示该分支未能提供可用证据，并继续使用仍然可信的其他分支。

这不是"吞掉错误"：错误类型、消息、耗时都进了状态和日志，
只是**不再用异常中断主流程**。
"""
from __future__ import annotations

import time

from core.config import get_llm, llm_chat
from core.state import AgentState
from nodes.query_nodes import grade_external_evidence
from nodes.retrieve_node import retrieve_knowledge
from nodes.web_search_node import should_search, web_search_save
from utils.retry import call_with_retry
from utils.tokens import record_llm_usage

from utils.logging_setup import get_logger

log = get_logger(__name__)

SEGMENT_PROMPT = """你是一名 CCUS 地质封存专家。下面是从知识图谱检索到的证据。

【已提取的项目参数】：
{params}

【图谱证据】：
{evidence}

请写一段 **150~250 字**的地质风险评估，只覆盖以下三点：
1. 断层活化风险（注入压力与断层稳定性、是否有微地震监测建议）
2. 盖层完整性（突破压力、泄漏路径）
3. 储层压力与破裂压力的关系

硬性要求：
- 证据里出现的数值必须原样引用（压力、深度、温度、密度），不要自己换算。
- 证据没提到的结论**不要编造**，直接写"图谱证据未覆盖"。
- 只输出这段评估正文，不要标题、不要 JSON、不要分点编号以外的格式。"""


def _segment_messages(state: AgentState, evidence: str) -> list[dict]:
    return [{
        "role": "user",
        "content": SEGMENT_PROMPT.format(
            params=state.get("extracted_params", "") or state.get("user_input", ""),
            evidence=evidence or "（无）",
        ),
    }]


def retrieve_geology(state: AgentState) -> dict:
    """只检索地质 Top-K；等待 grade_node 通过后才允许生成结论。"""
    t0 = time.perf_counter()
    started = time.time()
    strategy = state.get("retrieval_strategy") or "primary"
    params = (state.get("extracted_params") or "").strip()
    original = (state.get("user_input") or "").strip()
    rewritten = (state.get("geology_query") or "").strip()
    if strategy == "expanded":
        query = "\n".join(part for part in (
            rewritten, f"Original question: {original}", f"Project parameters: {params}",
            "CCUS geological carbon storage fault reactivation caprock integrity "
            "injection pressure fracture pressure leakage risk",
        ) if part)
        mode = "mix"  # alternate LightRAG strategy: combine graph and vector retrieval
    else:
        query = "\n".join(part for part in (
            rewritten or "CCUS geological storage fault caprock injection pressure",
            f"Project parameters: {params}", f"Original question: {original}",
        ) if part)
        mode = "hybrid"

    try:
        update = retrieve_knowledge(state, query_override=query, mode=mode)
        candidates = update.get("top_candidates") or []
        evidence = update.get("retrieved_knowledge", "") if candidates else ""
        status = {
            "ok": bool(candidates),
            "started_at": round(started, 3),
            "ended_at": round(time.time(), 3),
            "elapsed_s": round(time.perf_counter() - t0, 2),
            "engine": "lightrag",
            "candidates": len(candidates),
            "top_score": (candidates[0].get("rerank_score") if candidates else None),
            "retrieval_strategy": strategy,
            "retrieval_mode": mode,
            "error": "" if candidates else "LightRAG 未召回候选片段",
        }
        passthrough = {
            key: value for key, value in update.items()
            if key not in {"retrieved_knowledge", "top_candidates", "retrieval_query", "retrieval_mode"}
        }
        return {
            **passthrough,
            "geology_evidence": evidence,
            "geology_candidates": candidates,
            "geology_status": status,
        }
    except Exception as exc:  # noqa: BLE001 — parallel retrieval must degrade to state
        elapsed = time.perf_counter() - t0
        log.error("[Geology Retrieve] %s: %s", type(exc).__name__, exc)
        return {
            "geology_evidence": "",
            "geology_candidates": [],
            "geology_status": {
                "ok": False, "started_at": round(started, 3),
                "ended_at": round(time.time(), 3), "elapsed_s": round(elapsed, 2),
                "engine": "lightrag", "candidates": 0,
                "retrieval_strategy": strategy, "retrieval_mode": mode,
                "error": f"{type(exc).__name__}: {exc}",
            },
        }


def generate_geology(state: AgentState) -> dict:
    """仅使用 Grader 接受的地质证据生成分支结论。"""
    t0 = time.perf_counter()
    status = dict(state.get("geology_status") or {})
    grade = (state.get("retrieval_grade") or {}).get("geology") or {}
    if not grade.get("accepted"):
        error = grade.get("reason") or "Self-RAG 未接受地质证据"
        status.update({
            "ok": False, "ended_at": round(time.time(), 3),
            "elapsed_s": round(float(status.get("elapsed_s") or 0) +
                                (time.perf_counter() - t0), 2),
            "grading_error": f"地质证据未通过相关性评估，已阻止无依据生成：{error}",
        })
        if not status.get("error"):
            status["error"] = status["grading_error"]
        return {"geology_report": "", "geology_status": status}

    evidence = (state.get("geology_evidence") or "").strip()
    update: dict = {}
    try:
        # 联网兜底仍在本地 RAG 证据通过 Grader 后执行。
        local_state = {**state, "retrieved_knowledge": evidence}
        if should_search(local_state) == "web_search":
            web_update = web_search_save(local_state)
            update = {**update, **web_update}
            web_evidence = (web_update.get("web_content") or "").strip()
            if web_evidence:
                web_grade_update = grade_external_evidence(
                    {**state, **web_update}, web_evidence
                )
                web_grade = web_grade_update.get("grade") or {}
                update = {
                    **update,
                    **{k: v for k, v in web_grade_update.items() if k != "grade"},
                }
                status["web_evidence_grade"] = web_grade
                if web_grade.get("accepted"):
                    evidence = "\n\n".join(part for part in (evidence, web_evidence) if part)
                else:
                    log.warning(
                        "[Geology] external search evidence rejected by Self-RAG; "
                        "it will not be passed to generation"
                    )
            else:
                status["web_evidence_grade"] = {
                    "status": "empty", "accepted": False, "score": 0.0,
                    "reason": "No external evidence returned",
                }

        mode = state.get("complexity_mode") or "cloud"
        _, model, tier, _ = get_llm("geology", mode)
        merged = {**state, **update, "geology_evidence": evidence}
        resp = call_with_retry(
            llm_chat, "geology", messages=_segment_messages(state, evidence),
            temperature=0.2, mode=mode, label="geology 段生成",
        )
        segment = (resp.choices[0].message.content or "").strip()
        update = {**update, **record_llm_usage(merged, "geology", resp, model, tier)}
        status.update({
            "ok": bool(segment), "ended_at": round(time.time(), 3),
            "elapsed_s": round(float(status.get("elapsed_s") or 0) +
                                (time.perf_counter() - t0), 2),
            "chars": len(segment), "error": "" if segment else "模型返回空内容",
        })
        return {
            **update, "geology_report": segment, "geology_evidence": evidence,
            "geology_status": status,
        }
    except Exception as exc:  # noqa: BLE001 — preserve branch degradation contract
        status.update({
            "ok": False, "ended_at": round(time.time(), 3),
            "elapsed_s": round(float(status.get("elapsed_s") or 0) +
                                (time.perf_counter() - t0), 2),
            "chars": 0, "error": f"{type(exc).__name__}: {exc}",
        })
        log.error("[Geology Generate] %s: %s", type(exc).__name__, exc)
        return {**update, "geology_report": "", "geology_evidence": evidence,
                "geology_status": status}
