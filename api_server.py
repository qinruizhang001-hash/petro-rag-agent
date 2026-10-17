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

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

# 导入多智能体工作流（导入时会加载 Rerank 模型，需数秒，属正常现象）
from multi_agent_ccus import (
    app as agent_app,
    app_stream,
    NODE_LABELS,
    compute_cost_estimation,
)
from chart_tools import CHART_DIR

api = FastAPI(
    title="CCUS 智能评估 API",
    description="基于 LangGraph 的多智能体后端服务（SSE 流式）",
)

# 允许前端从其它端口访问（Streamlit 8505 → API 8000）
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


# ─── 同步版本（保留：脚本、curl 调试、回归测试用）─────────
def _run_agent(user_input: str, thread_id: str) -> dict:
    """
    同步执行完整工作流。
    interrupt_before 已移除，一次 invoke 即可跑完整个流程
    （反思循环由 check_quality 内部的 retry_count 上限控制，最多重写 2 次）。
    """
    initial_state = {"user_input": user_input}
    config = {"configurable": {"thread_id": thread_id}}
    return agent_app.invoke(initial_state, config)


@api.post("/api/analyze/sync")
async def analyze_sync(request: ProjectRequest):
    print(f"\n[API 收到请求 /sync]: {request.user_input}")
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
            # ─── Token 经济学 ───
            "cost_estimation": compute_cost_estimation(final_state),
            "complexity": final_state.get("complexity_info", {}),
            "context_pruning": final_state.get("context_pruning", {}),
        }
    except Exception as e:
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
                "history_entry": node_outputs.get("history_entry", ""),
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
    print(f"\n[API 收到请求 /stream]: {request.user_input}")
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


@api.get("/health")
async def health():
    return {"status": "ok", "service": "CCUS 智能评估 API", "streaming": True}


if __name__ == "__main__":
    uvicorn.run(api, host="127.0.0.1", port=8000)
