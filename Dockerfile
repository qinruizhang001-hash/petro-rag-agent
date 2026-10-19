# ─────────────────────────────────────────────────────────────
# CCUS 多智能体风险评估系统 — 生产镜像
#
# 一个镜像，两个角色：`backend`（uvicorn）和 `frontend`（streamlit）用同一份
# 镜像、不同的 command 启动。这样"前端能跑、后端不能跑"这类环境漂移从结构上
# 就不存在 —— 两边的 site-packages 是同一批字节。
#
# 每条系统依赖都对应一个真实踩过的坑：
#
# 1. libgl1 / libglib2.0-0 —— chromadb 依赖的 opencv 需要。缺了会在 import
#    阶段抛 "libGL.so.1: cannot open shared object file"，而这个报错发生在
#    启动时，看起来像"代码坏了"。
# 2. fonts-noto-cjk —— matplotlib 画中文图。缺字体时所有汉字渲染成方块，
#    图能生成、也不报错，只是没法看。
# 3. build-essential —— 少数依赖没有 manylinux wheel，需要现场编译。
#
# 两条不显然但必须遵守的约束：
#
# · torch 先单独装 CPU 版。直接从 PyPI 装会拉 CUDA 版（+2 GB），而本项目
#   的推理走 Ollama 容器，镜像里根本不需要 CUDA。
# · 模型与图谱**不进镜像**：bge-reranker-v2-m3 单份 2.2 GB，LightRAG 存储
#   随语料增长。它们通过 volume 挂载（见 docker-compose.yml）。
#   注意 core/reranker.py 是**在 import 时**加载重排模型的，所以 /app/models
#   为空时后端会在启动阶段直接失败 —— 这是有意让它响亮地失败，而不是等到
#   第一次请求才报一个指向错误方向的 OSError。
# ─────────────────────────────────────────────────────────────
FROM python:3.10-slim

# 离线模式：模型已挂载在 /app/models，不允许 huggingface_hub 在容器里偷偷联网
# 下载。core/__init__.py 里也用 setdefault 设过一次，这里是给 pip 之外的子进程
# （如 uvicorn worker）兜底。
#
# API_HOST / STREAMLIT_SERVER_ADDRESS 都是 0.0.0.0：容器内的回环地址对外不可达。
# 这两个值写在镜像里而不是 compose 里，是为了让 `docker run` 单跑也能用。
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    APP_LOG_FILE=/app/logs/app.log \
    LOG_LEVEL=INFO \
    API_HOST=0.0.0.0 \
    API_PORT=8000 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        libgl1 \
        libglib2.0-0 \
        fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/*

# CPU 版 torch 先行安装，避免 requirements.txt 把 CUDA 版拉进来
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch

# 依赖先于代码复制：只改代码时这一层直接命中缓存，不用重装 2 GB 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 运行期目录（由 compose 挂载；这里先建好，避免首次启动因缺目录失败）
RUN mkdir -p /app/logs /app/charts /app/lightrag_storage /app/chroma_petro

# 8000 = FastAPI（api_server）   8501 = Streamlit（app.py）
EXPOSE 8000 8501

# 健康检查只探 /health，不触发模型加载。
# start-period 给 90 s：容器刚起来时 reranker 还在从 volume 加载（2.2 GB，
# 冷启动 20-60 s），给少了会被健康检查判定失败并反复重启。
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

# 注意是 api_server:api，不是 api_server:app ——
# FastAPI 实例在 api_server.py 里叫 `api`（`app` 这个名字留给 multi_agent_ccus
# 那个兼容入口，uvicorn 指向它会拿到 LangGraph 图而不是 ASGI 应用）。
CMD ["uvicorn", "api_server:api", "--host", "0.0.0.0", "--port", "8000"]
