# ─────────────────────────────────────────────────────────────
# CCUS 多智能体风险评估系统 — 生产镜像
#
# 设计要点（每条都对应一个真实踩过的坑）：
#
# 1. 基础镜像用 slim，但**必须**补两样系统库：
#    · libgl1 / libglib2.0-0 —— chromadb 依赖的 opencv 需要，缺了会在
#      import 阶段抛 "libGL.so.1: cannot open shared object file"，
#      而这个报错发生在启动时，看起来像"代码坏了"。
#    · fonts-noto-cjk —— matplotlib 画中文图，缺字体时所有汉字渲染成方块，
#      图能生成、也不报错，只是没法看。
# 2. torch 先单独装 CPU 版。直接从 PyPI 装会拉 CUDA 版（+2 GB），
#    而本项目跑的是 Ollama 的 CPU 推理，容器里根本不需要 CUDA。
# 3. 模型与图谱**不进镜像**：bge-reranker-v2-m3 单份 2.3 GB，
#    LightRAG 存储随语料增长。它们通过 volume 挂载（见 docker-compose.yml）。
# 4. HF_HUB_OFFLINE=1 在 core/__init__.py 里也会设，这里再设一次是为了
#    在 pip 之外的任何子进程（如 uvicorn reload）也生效。
# ─────────────────────────────────────────────────────────────
FROM python:3.10-slim

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
    STREAMLIT_SERVER_PORT=8505 \
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

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 运行期目录（由 compose 挂载；这里先建好，避免首次启动因权限/缺目录失败）
RUN mkdir -p /app/logs /app/charts /app/lightrag_storage /app/chroma_petro

# 8000 = FastAPI（api_server）  8505 = Streamlit（app.py）
EXPOSE 8000 8505

# 注意：健康检查只探 /health，不触发模型加载 —— 容器刚起来时 reranker
# 还在加载（约 4-40 s），start-period 给足时间，否则会被反复重启。
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

CMD ["uvicorn", "api_server:api", "--host", "0.0.0.0", "--port", "8000"]
