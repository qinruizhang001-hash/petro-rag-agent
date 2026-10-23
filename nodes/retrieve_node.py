"""
Agent 2：GraphRAG 检索 + 本地精排 + 上下文压缩
=============================================
三阶段，每一阶段都在解决一个具体的失效模式：

1. **图谱召回**（LightRAG hybrid）—— 实体/关系/chunk 三类片段，数量 35~44 不等。
2. **本地精排**（bge-reranker-v2-m3）—— 只留 Top-K 注入下游。
   这一步同时产出本次检索的**可信度**：无关提问的 Top-1 分数是 0.0001 量级，
   真实 CCUS 提问在 0.27~0.74。这个分数就是"要不要唤醒云端"的判据，
   没有它，"本地模式"分支永远不会被走到。
3. **上下文压缩**（可选）—— 超过阈值时用本地模型把长文本压成要点，
   再送给云端。用几分钱的本地 token 换昂贵的云端输入 token。

⚠️ ``model_routing`` 只在压缩真的发生时回写。LangGraph 对 dict 字段是
**替换**语义而非合并：早先写成 ``"model_routing": prune_update.get(..., {})``，
未压缩时返回空字典，把 extract 节点刚记好的路由整条抹掉了 ——
日志一切正常，只有审计字段丢信息。这类 bug 不会报错，只能靠 E2E 断言兜住。
"""
from __future__ import annotations

import os

from core.config import get_llm, llm_chat
from core.reranker import reranker
from core.state import AgentState
from core.temporal import analyze_temporal
from lightrag_retriever import retrieve_documents
from model_router import RAG_MIN_SCORE, classify_complexity, escalate_complexity
from utils.context_pruner import context_char_count, needs_pruning, prune_context
from utils.tokens import (
    count_tokens,
    merge_token_usage,
    record_usage_payload,
)

from utils.logging_setup import get_logger

log = get_logger(__name__)


def _env_int(name: str, default: int) -> int:
    """读取整数型环境变量；非法值退回默认，不让配置失误炸掉主流程。"""
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        log.info(f"  [Config] {name}={raw!r} 不是合法整数，使用默认值 {default}")
        return default


RAG_TOP_K = _env_int("RAG_TOP_K", 2)


CONTEXT_PRUNE_THRESHOLD = _env_int("CONTEXT_PRUNE_THRESHOLD", 4000)


def retrieve_knowledge(
    state: AgentState, *, query_override: str | None = None, mode: str = "hybrid"
):
    """LightRAG 召回 + 本地精排；mode 可在 Self-RAG 重试时切换。"""
    log.info("\n[Agent 2] 正在通过 LightRAG 知识图谱检索，并执行本地 Rerank 精排...")
    query = (query_override or state.get("geology_query") or
             state.get("extracted_params") or state.get("user_input") or "").strip()
    attempt_stage = (
        "retrieve_retry" if state.get("retrieval_strategy") == "expanded" else "retrieve"
    )

    # ─── 第一阶段：LightRAG 图谱检索（实体 + 关系 + chunk） ──
    raw_docs = retrieve_documents(query, mode=mode)

    if not raw_docs:
        empty_usage = merge_token_usage(
            state,
            attempt_stage,
            {
                "query": query,
                "mode": mode,
                "query_tokens": count_tokens(query),
                "candidate_docs": 0,
                "candidate_tokens": 0,
                "rerank_input_tokens": 0,
                "injected_context_tokens": 0,
                "note": "知识图谱中无相关记录",
                "engine": "lightrag",
            },
        )
        log.info("  -> Token 消耗 [retrieve]: 图谱中无相关记录")
        # 没召回到东西 = RAG 未命中，复杂度判定维持 extract 阶段的结论
        complexity = classify_complexity(
            state["user_input"], rag_triggered=False, top_score=None
        )
        complexity = escalate_complexity(state.get("complexity_info"), complexity)
        log.info(f"  [Router] {complexity['label']} {complexity['reason']}")
        return {
            "retrieved_knowledge": "知识库中无相关记录",
            "token_usage": empty_usage,
            "complexity_mode": complexity["mode"],
            "complexity_info": complexity,
            "temporal_conflicts": [],
            "temporal_context": "",
            "temporal_stats": {"claims": 0, "conflicts": 0, "entities": []},
            "top_candidates": [],
            "retrieval_query": query,
            "retrieval_mode": mode,
        }

    log.info(f"  -> LightRAG 图谱检索返回 {len(raw_docs)} 个候选片段")

    # ─── 第二阶段：本地 Rerank 精排（语义重排打分） ────────
    log.info(f"  -> 正在对 {len(raw_docs)} 个候选片段进行本地 Rerank 精排...")
    pairs = [[query, doc] for doc in raw_docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, raw_docs), key=lambda x: x[0], reverse=True)

    # 只保留语义最相关的 Top-K，降低后续 LLM 上下文 Token
    top_docs = [doc for score, doc in ranked[:RAG_TOP_K]]
    retrieved_text = "\n\n".join(top_docs)
    top_score = float(ranked[0][0]) if ranked else 0.0

    # ─── 时间维度：抽取"井位 + 参数 + 数值 + 时间"并检测冲突 ──
    # 在**压缩之前**做：压缩会把长片段改写成语义摘要，数字往往第一个被丢掉，
    # 那时候再找冲突就什么都找不到了。
    temporal = analyze_temporal(top_docs)
    if temporal["conflicts"]:
        log.warning(
            f"  [Temporal] ⚠️ 检测到 {len(temporal['conflicts'])} 处跨时间冲突数据"
            f"（{temporal['claims']} 条带时间戳的断言）"
        )
        for c in temporal["conflicts"]:
            log.warning(f"     · {c['message']}")
    else:
        log.info(
            f"  [Temporal] 时间轴：{temporal['claims']} 条带时间戳的断言，无冲突"
        )

    # ─── 复杂度升级：RAG 是否真的命中 ──────────────────────
    # 注意"召回了"≠"命中"：图谱对无关提问也会返回片段，但 rerank 分数接近 0。
    # 本项目实测无关提问的 Top-1 是 0.0001、真实 CCUS 提问在 0.27~0.74，
    # 所以用 RAG_MIN_SCORE 这条线把"有内容可用"和"凑数片段"分开，
    # 避免为了一个假命中就把整条链路推到云端。
    rag_triggered = top_score >= RAG_MIN_SCORE
    complexity = classify_complexity(
        state["user_input"], rag_triggered=rag_triggered, top_score=top_score
    )
    complexity = escalate_complexity(state.get("complexity_info"), complexity)
    mode = complexity["mode"]
    log.info(f"  [Router] {complexity['label']} {complexity['reason']}")

    query_tokens = count_tokens(query)
    candidate_tokens = sum(count_tokens(doc) for doc in raw_docs)
    rerank_input_tokens = query_tokens * len(raw_docs) + candidate_tokens

    log.info(f"  -> 精排完成！已从 {len(raw_docs)} 个片段中过滤出最相关的 {len(top_docs)} 个。")
    for i, (score, doc) in enumerate(ranked[:5], start=1):
        log.info(f"     #{i} score={float(score):.4f} tokens={count_tokens(doc)}")

    # ─── 第三阶段：上下文压缩（Context Pruning）────────────
    # 用本地便宜模型把长文本压成要点，再送给云端。
    # 为什么放在 retrieve 而不是 report 里：压缩的对象是"检索结果"，
    # 让它在这一节点就地变短，下游（报告 + 评审）拿到的就是精简版，
    # 两处都省 token，而不是只在报告那一步省。
    base_usage = merge_token_usage(
        state,
        attempt_stage,
        {
            "query": query,
            "mode": mode,
            "query_tokens": query_tokens,
            "candidate_docs": len(raw_docs),
            "candidate_tokens": candidate_tokens,
            "rerank_input_tokens": rerank_input_tokens,
            "injected_context_tokens": count_tokens(retrieved_text),
            "top_k": len(top_docs),
            "top_score": round(top_score, 4),
            "rag_triggered": rag_triggered,
            "engine": "lightrag",
        },
    )

    prune_result = {
        "status": "skipped",
        "before_chars": context_char_count(retrieved_text),
        "after_chars": context_char_count(retrieved_text),
        "saved_chars": 0,
        "saved_ratio": 0.0,
        "reason": f"未达阈值（{CONTEXT_PRUNE_THRESHOLD} 字）",
    }
    prune_update: dict = {}

    if needs_pruning(retrieved_text, CONTEXT_PRUNE_THRESHOLD):
        log.info(f"  [Prune] 检索上下文 {context_char_count(retrieved_text)} 字 "
              f"> {CONTEXT_PRUNE_THRESHOLD} 字阈值 → 调用本地模型压缩")
        # 超长上下文会被切成多块分别压缩（本地模型只有 4096 tokens 上下文），
        # 所以这里必须**累加**每次调用的 token，不能只记最后一次 ——
        # 否则分 3 块压缩时成本统计只算 1/3。
        captured: dict = {
            "prompt_tokens": 0, "completion_tokens": 0, "calls": 0,
            "model": "", "tier": "", "estimated": False,
        }

        def _summarizer(prompt: str) -> str:
            _, p_model, p_tier, _ = get_llm("prune")
            resp = llm_chat(
                "prune",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            usage = getattr(resp, "usage", None)
            captured["prompt_tokens"] += int(getattr(usage, "prompt_tokens", 0) or 0)
            captured["completion_tokens"] += int(
                getattr(usage, "completion_tokens", 0) or 0
            )
            captured["calls"] += 1
            captured["model"] = p_model
            captured["tier"] = p_tier
            if usage is None:
                captured["estimated"] = True
            return resp.choices[0].message.content

        prune_result = prune_context(
            retrieved_text, summarizer=_summarizer, threshold=CONTEXT_PRUNE_THRESHOLD
        )

        if captured["calls"] > 0:
            # 压缩本身也是真实调用，重检轮次单独记账，避免覆盖首轮成本。
            prune_stage = "prune_retry" if attempt_stage == "retrieve_retry" else "prune"
            prune_update = record_usage_payload(
                {**state, "token_usage": base_usage},
                prune_stage,
                captured["prompt_tokens"],
                captured["completion_tokens"],
                captured["model"],
                captured["tier"],
                estimated=captured["estimated"],
            )

        if prune_result["status"] == "pruned":
            retrieved_text = prune_result["compressed"]
            log.info(f"  [Prune] 🗜️ {prune_result['before_chars']} → "
                  f"{prune_result['after_chars']} 字"
                  f"（省 {prune_result['saved_chars']} 字 / "
                  f"{prune_result['saved_ratio'] * 100:.0f}%，"
                  f"{prune_result['chunks']} 块 / {prune_result['calls']} 次调用）")
        else:
            log.warning(f"  [Prune] ⚠️ 压缩未生效，保留原文：{prune_result['reason']}")

    # ─── 时间冲突提示：放在压缩**之后**注入 ─────────────────
    # 顺序不能反。这段文字是给模型的硬指令（"必须用最新数据、禁止取平均"），
    # 一旦被送进压缩器，7B 模型很可能把它概括成"存在历史数据差异"，
    # 指令性就没了 —— 而它恰恰是整条时间维度链路唯一的执行点。
    if temporal["notice"]:
        retrieved_text = f"{temporal['notice']}\n\n{retrieved_text}"
        base_usage["retrieve"]["temporal_conflicts"] = len(temporal["conflicts"])
        log.info(
            f"  [Temporal] 冲突提示已注入上下文（{len(temporal['notice'])} 字，"
            f"位于压缩之后，不会被摘要掉）"
        )

    # 压缩后重新统计实际注入量（这是真正花在云端输入上的那份）
    injected_tokens = count_tokens(retrieved_text)
    if prune_result["status"] == "pruned":
        base_usage["retrieve"]["injected_context_tokens"] = injected_tokens
        base_usage["retrieve"]["prune_saved_chars"] = prune_result["saved_chars"]
    base_usage["retrieve"]["prune_status"] = prune_result["status"]

    log.info(
        f"  -> Token 消耗 [retrieve]: query={query_tokens}, "
        f"candidates={candidate_tokens}, rerank_input≈{rerank_input_tokens}, "
        f"injected_context={injected_tokens}"
    )

    update = {
        "retrieved_knowledge": retrieved_text,
        "top_candidates": [
            {"text": doc, "rerank_score": float(score)}
            for score, doc in ranked[:RAG_TOP_K]
        ],
        "retrieval_query": query,
        "retrieval_mode": mode,
        "token_usage": prune_update.get("token_usage", base_usage),
        "context_pruning": {
            k: prune_result.get(k)
            for k in ("status", "before_chars", "after_chars",
                      "saved_chars", "saved_ratio", "reason",
                      "chunks", "calls")
        },
        "complexity_mode": mode,
        "complexity_info": complexity,
        # ─── 时间维度记忆 ───
        # conflicts 是结构化字段，供 API / 前端直接展示；
        # context 是注入给下游的提示块原文（可审计：报告里那句"存在冲突"
        # 必须有对应的来源，否则无法判断是不是模型编的）。
        "temporal_conflicts": temporal["conflicts"],
        "temporal_context": temporal["notice"],
        "temporal_stats": {
            "claims": temporal["claims"],
            "conflicts": len(temporal["conflicts"]),
            "entities": temporal["entities"],
        },
    }
    # ⚠️ model_routing 只在压缩真的发生时才回写。
    # LangGraph 对 dict 字段是**替换**语义，不是合并：早先这里写成
    # `"model_routing": prune_update.get("model_routing", {})`，未压缩时
    # 返回空字典，把 extract 节点刚记好的路由整条抹掉了 —— 结果是
    # 审计用的 model_routing 里永远看不到 extract，而日志一切正常。
    # 这类"只在审计字段上丢信息"的 bug 不会报错，只能靠 E2E 断言兜住。
    if prune_update.get("model_routing"):
        update["model_routing"] = prune_update["model_routing"]
    return update
