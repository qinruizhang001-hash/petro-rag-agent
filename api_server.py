"""
CCUS 智能评估 API 服务
=======================
FastAPI 后端，把 LangGraph 多智能体工作流封装为 HTTP 接口。

启动方式:
    python api_server.py
    # 或
    uvicorn api_server:api --host 127.0.0.1 --port 8000

接口:
    POST /api/analyze   提交 CCUS 项目描述，返回风险评估报告（完整流程，耗时数分钟）
    GET  /health        健康检查（秒回）
    GET  /docs          Swagger 交互文档
"""
import time
import asyncio

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

# 导入多智能体工作流（导入时会加载 Rerank 模型，需数秒，属正常现象）
# 注意：multi_agent_ccus.py 中 app = workflow.compile(checkpointer=memory_saver)
from multi_agent_ccus import app as agent_app

api = FastAPI(
    title="CCUS 智能评估 API",
    description="基于 LangGraph 的多智能体后端服务",
)


class ProjectRequest(BaseModel):
    user_input: str


def _run_agent(user_input: str, thread_id: str) -> dict:
    """
    同步执行完整工作流。
    interrupt_before 已移除，一次 invoke 即可跑完整个流程
    （反思循环由 check_quality 内部的 retry_count 上限控制，最多重写 2 次）。
    """
    initial_state = {"user_input": user_input}
    config = {"configurable": {"thread_id": thread_id}}

    return agent_app.invoke(initial_state, config)


@api.post("/api/analyze")
async def analyze_ccus_project(request: ProjectRequest):
    print(f"\n[API 收到请求]: {request.user_input}")
    thread_id = f"api_{int(time.time())}"
    print(f"[API Thread]: {thread_id}")

    try:
        # invoke 是阻塞调用（完整流程含反思重写可能耗时数分钟），
        # 放到线程池执行，避免卡死 FastAPI 事件循环（/health 等仍可响应）
        final_state = await asyncio.to_thread(_run_agent, request.user_input, thread_id)

        return {
            "status": "success",
            "report": final_state["final_report"],
            "extracted_params": final_state.get("extracted_params", ""),
            "score": final_state.get("score", 10),
            # 多模型路由：{stage: "tier:model"}，如实反映实际使用的模型
            "model_routing": final_state.get("model_routing", {}),
            # 文件记忆：本次追加到 project_history.md 的记录
            "history_entry": final_state.get("history_entry", ""),
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@api.get("/health")
async def health():
    return {"status": "ok", "service": "CCUS 智能评估 API"}


if __name__ == "__main__":
    uvicorn.run(api, host="127.0.0.1", port=8000)
