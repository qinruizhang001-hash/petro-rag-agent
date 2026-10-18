"""
地质分支（geology_agent）：断层 / 盖层风险
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
在报告里写明"地质模块检索超时，请人工复核"，并基于井筒数据继续生成。

这不是"吞掉错误"：错误类型、消息、耗时都进了状态和日志，
只是**不再用异常中断主流程**。
"""
from __future__ import annotations

import time

from core.config import get_llm, llm_chat
from core.state import AgentState
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


def geology_agent(state: AgentState) -> dict:
    """并行左分支：LightRAG 检索 + 地质结论生成。永不抛异常。"""
    t0 = time.perf_counter()
    t_epoch = time.time()   # 墙钟时间戳：跨分支比较用（perf_counter 的起点不可跨节点比）
    log.info("\n[Geology] 地质分支启动：LightRAG 图谱检索（断层 / 盖层）...")

    update: dict = {}
    evidence = ""
    try:
        # ─── ① 图谱检索 + 精排 + 压缩（复用唯一实现）───
        update = retrieve_knowledge(state)
        evidence = update.get("retrieved_knowledge", "") or ""

        # ─── ② 证据不足时联网补料 ───
        # 原来的 should_search 是主链路上的条件边；扇出之后没有"主链路"了，
        # 而"证据够不够"这个问题天然属于取证据的这一条分支，所以搬进来。
        merged = {**state, **update}
        if should_search(merged) == "web_search":
            web_update = web_search_save(merged)
            update = {**update, **web_update}
            evidence = evidence + "\n\n" + (web_update.get("web_content") or "")

        # ─── ③ 生成地质结论（带指数退避重试）───
        mode = state.get("complexity_mode") or "cloud"
        _, model, tier, _ = get_llm("geology", mode)
        resp = call_with_retry(
            llm_chat,
            "geology",
            messages=_segment_messages(state, evidence),
            temperature=0.2,
            mode=mode,
            label="geology 段生成",
        )
        segment = (resp.choices[0].message.content or "").strip()
        update = {**update, **record_llm_usage(merged, "geology", resp, model, tier)}

        elapsed = time.perf_counter() - t0
        log.info(f"  [Geology] ✅ 完成（{elapsed:.1f}s，{len(segment)} 字）")
        return {
            **update,
            "geology_report": segment,
            "geology_evidence": evidence,
            "geology_status": {
                "ok": bool(segment),
                "started_at": round(t_epoch, 3),
                "ended_at": round(time.time(), 3),
                "elapsed_s": round(elapsed, 2),
                "chars": len(segment),
                "engine": "lightrag",
                "error": "" if segment else "模型返回空内容",
            },
        }

    except Exception as exc:  # noqa: BLE001 — 分支必须降级而不是中断整张图
        elapsed = time.perf_counter() - t0
        log.error(f"  [Geology] ❌ 分支失败（{elapsed:.1f}s）：{type(exc).__name__}: {exc}")
        return {
            **update,
            "geology_report": "",
            "geology_evidence": evidence,
            "geology_status": {
                "ok": False,
                "started_at": round(t_epoch, 3),
                "ended_at": round(time.time(), 3),
                "elapsed_s": round(elapsed, 2),
                "chars": 0,
                "engine": "lightrag",
                "error": f"{type(exc).__name__}: {exc}",
            },
        }
