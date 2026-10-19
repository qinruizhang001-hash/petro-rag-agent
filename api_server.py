"""
CCUS 智能评估 API 服务
=======================
FastAPI 后端，把 LangGraph 多智能体工作流封装为 HTTP 接口。

启动方式:
    python api_server.py
    # 或
    uvicorn api_server:api --host 127.0.0.1 --port 8000

接口:
    POST /api/analyze       提交项目描述，返回 **SSE 流**（token 级实时推送）
    POST /api/analyze/sync  同一流程的非流式版本（脚本/调试用，一次返回 JSON）
    POST /api/analyze/hitl  同上，但在人工闸门前**暂停**，等 /api/approve
    GET  /api/review/{tid}  查询某条 thread 停在哪一步（进程重启也不丢）
    POST /api/approve       人工审核决定：通过 → 落盘；驳回 → 带意见重写
    GET  /health            健康检查（秒回）
    GET  /charts/{file}     生成的地质图表静态文件
    GET  /docs              Swagger 交互文档

SSE 事件格式（每行 `data: {json}`，空行分隔）:
    {"type":"node",  "node":"extract", "label":"提取 CCUS 关键参数", "status":"start"}
    {"type":"token", "text":"...", "stage":"report"}
    {"type":"chart", "chart":{...,"path":"...","summary":"..."}}
    {"type":"done",  "report":"...", "score":9, "extracted_params":"...",
                     "model_routing":{...}, "chart":{...}, "elapsed_s":78.2,
                     "cost_estimation":{...}, "complexity":{...},
                     "context_pruning":{...}}
    {"type":"error", "message":"..."}

实现上的两个关键点（都是踩过的坑）:

1. **astream_events 会丢弃 LangGraph 的 custom stream writer。** 实测用它替代
   `stream_mode="custom"` 时，`get_stream_writer()` 写出的内容根本不会出现在事件流
   里（只有状态更新的 on_chain_stream）。因此 token 通道自己搭：节点从 config 拿到
   `token_sink` 回调，把 delta 直接塞进 SSE 的队列。astream_events 负责节点级进度。

2. **SqliteSaver 不支持异步。** `astream_events` 走异步路径会抛
   NotImplementedError，所以流式端点用不带 checkpointer 的 `app_stream`。
"""
import asyncio
import json
import os
import time
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

# 导入多智能体工作流（导入时会加载 Rerank 模型，需数秒，属正常现象）
# 图结构在 workflow.py，本文件不再依赖 multi_agent_ccus 这个兼容入口
from workflow import (
    app as agent_app,
    app_stream,
    app_hitl,
    NODE_LABELS,
    compute_cost_estimation,
)
from chart_tools import CHART_DIR

from utils.logging_setup import get_logger

log = get_logger(__name__)

api = FastAPI(
    title="CCUS 智能评估 API",
    description="基于 LangGraph 的多智能体后端服务（SSE 流式）",
)

# 允许前端从其它端口访问（Streamlit 8501 → API 8000）
api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 图表静态目录：让浏览器能直接看历史图片（也方便 curl 检查）
os.makedirs(CHART_DIR, exist_ok=True)
api.mount("/charts", StaticFiles(directory=CHART_DIR), name="charts")


class ProjectRequest(BaseModel):
    user_input: str


class ApprovalRequest(BaseModel):
    thread_id: str
    approved: bool
    reviewer: str = ""
    comment: str = ""


# 人工驳回上限。为什么必须有：驳回会回到 report 重写，而重写要真金白银地
# 调模型。没有上限时"人一直驳回、机器一直重写"就是一个无底洞，
# 且几乎总是审核意见没说清楚导致的。到上限后 API 直接 409，
# 强制审核人要么明确通过，要么放弃这条 thread 重新发起。
_MAX_HUMAN_REJECTIONS = 2


def _physics_of(values: dict) -> dict:
    return values.get("physics_check") or {}


def _human_review_of(values: dict) -> dict:
    """把 state 里的人工决定整理成前端/调用方看得懂的字段。"""
    decision = values.get("human_decision") or {}
    return {
        "required": True,
        "approved": decision.get("approved"),
        # auto=True 表示"这次没有人看过"—— 必须显式告诉调用方，
        # 否则自动放行会被误读成"已人工确认"。
        "auto": bool(decision.get("auto")),
        "reviewer": decision.get("reviewer") or "",
        "comment": decision.get("comment") or "",
        "rejections": int(values.get("human_rejections") or 0),
        "max_rejections": _MAX_HUMAN_REJECTIONS,
    }


def _review_payload(values: dict, thread_id: str, pending: bool) -> dict:
    return {
        "thread_id": thread_id,
        "status": "awaiting_approval" if pending else "success",
        "report": values.get("final_report", ""),
        "extracted_params": values.get("extracted_params", ""),
        "score": values.get("score"),
        "model_routing": values.get("model_routing", {}),
        "history_entry": values.get("history_entry", ""),
        "chart": values.get("chart_info", {}),
        "pinn": values.get("pinn_prediction", {}),
        # ─── 物理一致性校验（规则引擎）───
        "physics_check": _physics_of(values),
        # ─── 人类在环 ───
        "human_review": _human_review_of(values),
        # ─── Token 经济学 ───
        "cost_estimation": compute_cost_estimation(values),
        "complexity": values.get("complexity_info", {}),
        "context_pruning": values.get("context_pruning", {}),
    }


# ─── 同步版本（保留：脚本、curl 调试、回归测试用）─────────
def _run_agent(user_input: str, thread_id: str) -> dict:
    """
    同步执行完整工作流。

    这条路径**不暂停**：``app`` 编译时没有 ``interrupt_before``，
    闸门节点会把"未经人工审核"写成 ``human_decision.auto = True`` 再放行。
    需要人工确认请走 ``/api/analyze/hitl``。
    """
    initial_state = {"user_input": user_input}
    config = {"configurable": {"thread_id": thread_id}}
    return agent_app.invoke(initial_state, config)


@api.post("/api/analyze/sync")
async def analyze_sync(request: ProjectRequest):
    log.info(f"[API] 收到 /sync 请求：{request.user_input[:60]}")
    thread_id = f"api_{int(time.time())}"
    try:
        final_state = await asyncio.to_thread(_run_agent, request.user_input, thread_id)
        return {
            "status": "success",
            "report": final_state["final_report"],
            "extracted_params": final_state.get("extracted_params", ""),
            "score": final_state.get("score", 10),
            "model_routing": final_state.get("model_routing", {}),
            "history_entry": final_state.get("history_entry", ""),
            "chart": final_state.get("chart_info", {}),
            "pinn": final_state.get("pinn_prediction", {}),
            # ─── 物理一致性校验 + 人类在环 ───
            "physics_check": _physics_of(final_state),
            "human_review": _human_review_of(final_state),
            # ─── Token 经济学 ───
            "cost_estimation": compute_cost_estimation(final_state),
            "complexity": final_state.get("complexity_info", {}),
            "context_pruning": final_state.get("context_pruning", {}),
        }
    except Exception as e:
        log.exception("[API] /sync 执行失败")
        return {"status": "error", "message": str(e)}


# ─── 流式版本 ─────────────────────────────────────────────
def _sse(obj: dict) -> str:
    """把一个事件编码成 SSE 帧。ensure_ascii=False 保住中文可读性。"""
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


async def _event_stream(user_input: str, thread_id: str):
    """
    驱动工作流并把事件转成 SSE 帧。

    结构：astream_events（异步）→ 队列 → 本生成器。
    为什么要过一道队列：节点的 token 来自**工作线程**（同步 LLM 客户端），
    而 astream_events 在事件循环里。统一塞进 asyncio.Queue 后，
    两侧的事件才能按时间顺序汇成一条流。
    """
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()
    started = time.time()
    node_outputs: dict = {}
    done_sent = False

    def sink(event: dict):
        """节点（工作线程）调用：线程安全地把事件送进事件循环的队列。"""
        try:
            loop.call_soon_threadsafe(queue.put_nowait, ("event", event))
        except RuntimeError:
            pass  # 事件循环已关闭（客户端断开），忽略

    async def drive():
        try:
            config = {
                "configurable": {
                    "thread_id": thread_id,
                    # 关键：把回调通过 config 传给 generate_report 节点
                    # （节点声明 (state, config) 即可拿到）
                    "token_sink": sink,
                }
            }
            async for ev in app_stream.astream_events(
                {"user_input": user_input}, config, version="v2"
            ):
                kind = ev.get("event")
                name = ev.get("name", "")

                if kind == "on_chain_start" and name in NODE_LABELS:
                    queue.put_nowait(("event", {
                        "type": "node", "node": name,
                        "label": NODE_LABELS[name], "status": "start",
                        # t = 距请求开始的毫秒数。前端的数据看板靠 start/end 两帧
                        # 的 t 相减得到每个节点的真实耗时 —— 不能靠前端自己打点，
                        # 那会把 SSE 排队与网络延迟算进节点耗时里。
                        "t": round((time.time() - started) * 1000),
                    }))
                elif kind == "on_chain_end" and name in NODE_LABELS:
                    out = (ev.get("data") or {}).get("output")
                    if isinstance(out, dict):
                        node_outputs.update(
                            {k: v for k, v in out.items() if v is not None}
                        )
                    queue.put_nowait(("event", {
                        "type": "node", "node": name,
                        "label": NODE_LABELS[name], "status": "end",
                        "t": round((time.time() - started) * 1000),
                    }))
        except Exception as e:  # 任何异常都要变成前端能看懂的 error 事件
            queue.put_nowait(("event", {"type": "error", "message": str(e)}))
        finally:
            queue.put_nowait(("stop", None))

    task = asyncio.create_task(drive())

    try:
        while True:
            try:
                kind, payload = await asyncio.wait_for(queue.get(), timeout=10)
            except asyncio.TimeoutError:
                # 心跳：纯文本注释，前端会跳过，但它能保住连接不被中间层掐断
                yield ": keep-alive\n\n"
                continue

            if kind == "stop":
                break

            if payload.get("type") == "token":
                yield _sse({"type": "token", "text": payload.get("text", ""),
                            "stage": payload.get("stage", "")})
                continue

            if payload.get("type") == "chart":
                yield _sse({"type": "chart", "chart": payload.get("chart", {})})
                continue

            if payload.get("type") == "node":
                yield _sse(payload)
                continue

            if payload.get("type") == "error":
                yield _sse(payload)

        await task

        if not done_sent:
            report = node_outputs.get("final_report", "")
            # 成本必须在所有 token 记完之后算 —— node_outputs 已经攒齐了
            # 每个节点的 token_usage（每个节点返回的都是累计后的全量字典）
            yield _sse({
                "type": "done",
                "report": report,
                "score": node_outputs.get("score"),
                "extracted_params": node_outputs.get("extracted_params", ""),
                "model_routing": node_outputs.get("model_routing", {}),
                "chart": node_outputs.get("chart_info", {}),
                "pinn": node_outputs.get("pinn_prediction", {}),
                "history_entry": node_outputs.get("history_entry", ""),
                # ─── 物理一致性校验 + 人类在环 ───
                "physics_check": _physics_of(node_outputs),
                "human_review": _human_review_of(node_outputs),
                # ─── 并行分支耗时（前端数据看板要用）───
                # aggregate 已经把两条分支的时间戳算成 wall/serial/speedup，
                # 但此前没往 done 里放，前端只能看到总耗时，看不出"并行省了多少"。
                "parallel_timing": node_outputs.get("parallel_timing", {}),
                "geology_status": node_outputs.get("geology_status", {}),
                "wellbore_status": node_outputs.get("wellbore_status", {}),
                "branch_merge": node_outputs.get("branch_merge", {}),
                # ─── Token 经济学 ───
                "cost_estimation": compute_cost_estimation(node_outputs),
                "complexity": node_outputs.get("complexity_info", {}),
                "context_pruning": node_outputs.get("context_pruning", {}),
                "elapsed_s": round(time.time() - started, 1),
            })
            done_sent = True
    finally:
        if not task.done():
            task.cancel()


@api.post("/api/analyze")
async def analyze_ccus_project(request: ProjectRequest):
    """SSE 流式评估。前端用 requests(stream=True) 或 EventSource 消费。"""
    log.info(f"[API] 收到 /stream 请求：{request.user_input[:60]}")
    thread_id = f"api_{int(time.time())}_{uuid.uuid4().hex[:6]}"

    return StreamingResponse(
        _event_stream(request.user_input, thread_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            # 关掉 nginx 一类反向代理的缓冲，否则 token 会被攒着一起吐，
            # 流式的意义就没了
            "X-Accel-Buffering": "no",
        },
    )


# ══════════════════════════════════════════════════════════
# 人类在环（HITL）
# ══════════════════════════════════════════════════════════
# 三个接口构成一次完整的人工审核往返：
#
#   POST /api/analyze/hitl  跑到人工闸门前停住，返回报告 + 物理校验结论
#   GET  /api/review/{tid}  随时查这条 thread 停在哪一步（进程重启也不丢）
#   POST /api/approve       写入人工决定并续跑（通过 → 落盘；驳回 → 重写）
#
# 状态存在 sqlite（SqliteSaver）里，所以"暂停 → 人工几小时后回来点确认"
# 是可行的，不依赖任何内存态。

def _config_of(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def _snapshot(thread_id: str) -> dict:
    snap = app_hitl.get_state(_config_of(thread_id))
    values = dict(snap.values or {})
    waiting = "hitl_gate" in (snap.next or ())
    return {"values": values, "waiting": waiting}


def _start_hitl(user_input: str, thread_id: str) -> dict:
    """同步执行到闸门前（会阻塞几十秒，必须放进线程池）。"""
    app_hitl.invoke({"user_input": user_input}, _config_of(thread_id))
    return _snapshot(thread_id)


def _resume(thread_id: str, approved: bool, reviewer: str, comment: str) -> dict:
    """把人工决定写进 state 并续跑。"""
    snap = _snapshot(thread_id)
    if not snap["values"]:
        raise KeyError(f"未找到 thread_id={thread_id}")
    if not snap["waiting"]:
        raise ValueError(f"thread_id={thread_id} 当前不在等待审核状态")

    rejections = int(snap["values"].get("human_rejections") or 0)
    if not approved and rejections >= _MAX_HUMAN_REJECTIONS:
        raise OverflowError(
            f"已达到人工驳回上限（{_MAX_HUMAN_REJECTIONS} 次）。"
            f"请直接通过，或放弃该 thread 后重新发起评估。"
        )

    update = {
        "human_decision": {
            "approved": bool(approved),
            "reviewer": reviewer or "unknown",
            "comment": comment or "",
            "auto": False,
        }
    }
    if not approved:
        update["human_rejections"] = rejections + 1
        # 驳回必须给出可执行的修改方向，否则下一版只是换一种说法
        update["feedback"] = (
            f"【人工审核驳回 · 第 {rejections + 1} 次】"
            f"{comment or '未填写具体意见，请按原有评审意见修订。'}"
        )

    app_hitl.update_state(_config_of(thread_id), update)
    app_hitl.invoke(None, _config_of(thread_id))
    return _snapshot(thread_id)


@api.post("/api/analyze/hitl")
async def analyze_with_human_review(request: ProjectRequest):
    """
    跑到人工闸门前停下，返回待审报告。

    响应 ``status == "awaiting_approval"``，调用方拿 ``thread_id``
    去 ``POST /api/approve`` 才会真正落盘。
    """
    thread_id = f"hitl_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    log.info(f"[API] 收到 /hitl 请求（{thread_id}）：{request.user_input[:60]}")
    try:
        snap = await asyncio.to_thread(_start_hitl, request.user_input, thread_id)
    except Exception as e:
        log.exception("[API] HITL 执行失败")
        raise HTTPException(status_code=500, detail=str(e))

    log.info(f"[HITL] {thread_id} 已停在人工闸门，等待 /api/approve")
    return _review_payload(snap["values"], thread_id, snap["waiting"])


@api.get("/api/review/{thread_id}")
async def review_status(thread_id: str):
    """查询某条 thread 的当前状态（是否还在等人工确认）。"""
    snap = await asyncio.to_thread(_snapshot, thread_id)
    if not snap["values"]:
        raise HTTPException(status_code=404, detail=f"未找到 thread_id={thread_id}")
    return _review_payload(snap["values"], thread_id, snap["waiting"])


@api.post("/api/approve")
async def approve(request: ApprovalRequest):
    """
    人工审核决定：``approved=true`` 落盘结束；``false`` 带意见打回重写。

    驳回后如果又跑到闸门前，响应会**再次**是 ``awaiting_approval``，
    并带上新的报告与 ``human_review.rejections`` 计数。
    """
    log.info(
        f"[API] 收到 /approve（{request.thread_id}）："
        f"{'通过' if request.approved else '驳回'} / {request.reviewer or '-'}"
    )
    try:
        snap = await asyncio.to_thread(
            _resume, request.thread_id, request.approved,
            request.reviewer, request.comment,
        )
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except OverflowError as e:
        raise HTTPException(status_code=429, detail=str(e))
    except Exception as e:
        log.exception("[API] /approve 执行失败")
        raise HTTPException(status_code=500, detail=str(e))

    payload = _review_payload(snap["values"], request.thread_id, snap["waiting"])
    # 驳回后若评审仍未达标，图会自己再跑一轮并再次停下 —— 如实反映
    if snap["waiting"]:
        payload["status"] = "awaiting_approval"
        payload["note"] = "报告已按审核意见重写，请再次确认"
    return payload


@api.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "CCUS 智能评估 API",
        "streaming": True,
        "human_in_the_loop": True,
    }


if __name__ == "__main__":
    # Docker 里必须监听 0.0.0.0，本地默认仍是回环地址（不主动暴露到局域网）
    uvicorn.run(
        api,
        host=os.getenv("API_HOST", "127.0.0.1"),
        port=int(os.getenv("API_PORT", "8000")),
    )
