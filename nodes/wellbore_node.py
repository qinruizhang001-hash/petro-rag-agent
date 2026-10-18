"""
井筒分支（wellbore_agent）：套管腐蚀 / 压力异常
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


def wellbore_agent(state: AgentState) -> dict:
    """并行右分支：ChromaDB 检索 + 井筒结论生成。永不抛异常。"""
    t0 = time.perf_counter()
    t_epoch = time.time()   # 墙钟时间戳：跨分支比较用（perf_counter 的起点不可跨节点比）
    log.info("\n[Wellbore] 井筒分支启动：ChromaDB 原文检索（套管 / 压力异常）...")

    evidence = ""
    update: dict = {}
    try:
        # ─── ① 向量检索（Chroma 已按相似度排序，不再 rerank）───
        query = _build_query(state)
        docs = retrieve_documents(query, k=WELLBORE_TOP_K)
        evidence = "\n\n".join(docs)
        log.info(f"  [Wellbore] ChromaDB 返回 {len(docs)} 个切片")

        update = {
            "token_usage": merge_token_usage(
                state,
                "chroma_retrieve",
                {
                    "query_tokens": count_tokens(query),
                    "candidate_docs": len(docs),
                    "candidate_tokens": sum(count_tokens(d) for d in docs),
                    "injected_context_tokens": count_tokens(evidence),
                    "top_k": len(docs),
                    "engine": "chromadb",
                },
            ),
        }

        # ─── ② 生成井筒结论（带指数退避重试）───
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
        update = {
            **update,
            **record_llm_usage({**state, **update}, "wellbore", resp, model, tier),
        }

        elapsed = time.perf_counter() - t0
        log.info(f"  [Wellbore] ✅ 完成（{elapsed:.1f}s，{len(segment)} 字）")
        return {
            **update,
            "wellbore_report": segment,
            "wellbore_evidence": evidence,
            "wellbore_status": {
                "ok": bool(segment),
                "started_at": round(t_epoch, 3),
                "ended_at": round(time.time(), 3),
                "elapsed_s": round(elapsed, 2),
                "chars": len(segment),
                "engine": "chromadb",
                "candidates": len(docs),
                "error": "" if segment else "模型返回空内容",
            },
        }

    except Exception as exc:  # noqa: BLE001 — 分支必须降级而不是中断整张图
        elapsed = time.perf_counter() - t0
        log.error(f"  [Wellbore] ❌ 分支失败（{elapsed:.1f}s）：{type(exc).__name__}: {exc}")
        return {
            **update,
            "wellbore_report": "",
            "wellbore_evidence": evidence,
            "wellbore_status": {
                "ok": False,
                "started_at": round(t_epoch, 3),
                "ended_at": round(time.time(), 3),
                "elapsed_s": round(elapsed, 2),
                "chars": 0,
                "engine": "chromadb",
                "candidates": 0,
                "error": f"{type(exc).__name__}: {exc}",
            },
        }
