#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# CCUS 多智能体系统 — 云服务器一键部署脚本
#
# 用法（首次在服务器上执行）：
#   chmod +x deploy.sh
#   ./deploy.sh
#
# 也可由 GitHub Actions 通过 SSH 远程调用：
#   ssh user@host "cd /opt/petro-rag-agent && bash deploy.sh"
#
# 前置条件（脚本会逐项检查）：
#   1. Docker Engine + Docker Compose plugin
#   2. Git
#   3. Ollama 在宿主机运行，且 qwen2.5 / nomic-embed-text 已拉取
#   4. models/bge-reranker-v2-m3 已下载（2.2 GB）
#   5. 内存 ≥ 8 GB（推荐 ≥ 16 GB）
# ─────────────────────────────────────────────────────────────
set -euo pipefail

# ─── 可配置项（环境变量覆盖）───────────────────────────────
DEPLOY_DIR="${DEPLOY_DIR:-/opt/petro-rag-agent}"
REPO_URL="${REPO_URL:-https://github.com/qinruizhang001-hash/petro-rag-agent.git}"
BRANCH="${BRANCH:-main}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-180}"
BACKUP_RETAIN="${BACKUP_RETAIN:-5}"

# Ollama 地址：默认连宿主机。如果 Ollama 在另一台机器，改这个。
OLLAMA_URL="${OLLAMA_URL:-http://127.0.0.1:11434}"

# nginx 对外端口：8080 是开发默认；生产改 80 或 443 时把这里改掉。
NGINX_HOST_PORT="${NGINX_HOST_PORT:-8080}"

LOG_FILE="/var/log/ccus-deploy.log"
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')

# ─── 工具函数 ──────────────────────────────────────────────
log()  { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"; }
warn() { log "WARN  $*"; }
fail() { log "ERROR $*"; exit 1; }

cmd_exists() { command -v "$1" >/dev/null 2>&1; }

# ─── 0. 前置检查 ───────────────────────────────────────────
log "=== CCUS 部署开始 ==="

if [[ $EUID -eq 0 ]]; then
    warn "当前以 root 运行。建议用普通用户 + docker 组，避免文件权限问题。"
fi

for need in docker git curl; do
    cmd_exists "$need" || fail "缺少命令: $need"
done

# Docker Compose 插件检测（v2 是 `docker compose`，v1 是 `docker-compose`）
if docker compose version >/dev/null 2>&1; then
    COMPOSE="docker compose"
elif docker-compose version >/dev/null 2>&1; then
    COMPOSE="docker-compose"
else
    fail "未检测到 Docker Compose（需要 v2 插件或独立包）"
fi
log "Docker Compose: $($COMPOSE version --short 2>/dev/null || $COMPOSE version | head -1)"

# ─── 1. 系统资源检查 ───────────────────────────────────────
MEM_KB=$(grep MemTotal /proc/meminfo | awk '{print $2}')
MEM_GB=$((MEM_KB / 1024 / 1024))
log "内存: ${MEM_GB} GB"
if [[ $MEM_GB -lt 8 ]]; then
    fail "内存不足 8 GB（当前 ${MEM_GB} GB）。2 worker × 2.2 GB reranker + Ollama 需要至少 8 GB。"
fi

# ─── 2. Ollama 可用性检查 ──────────────────────────────────
log "检查 Ollama ($OLLAMA_URL)..."
if ! curl -fsS "$OLLAMA_URL/api/tags" >/dev/null 2>&1; then
    fail "Ollama 不可达: $OLLAMA_URL\n" \
         "云服务器部署时，Ollama 必须在宿主机上运行（OLLAMA_HOST=0.0.0.0:11434）。\n" \
         "如果 Ollama 在另一台机器，请设置环境变量 OLLAMA_URL。"
fi

# 检查必需模型
MISSING_MODELS=()
for model in qwen2.5 nomic-embed-text; do
    if ! curl -fsS "$OLLAMA_URL/api/tags" 2>/dev/null | grep -q "$model"; then
        MISSING_MODELS+=("$model")
    fi
done
if [[ ${#MISSING_MODELS[@]} -gt 0 ]]; then
    fail "Ollama 缺少模型: ${MISSING_MODELS[*]}\n" \
         "请在宿主机执行: ollama pull qwen2.5 && ollama pull nomic-embed-text"
fi
log "Ollama 就绪，模型已加载。"

# ─── 3. 代码仓库 ───────────────────────────────────────────
if [[ -d "$DEPLOY_DIR/.git" ]]; then
    log "已有代码目录: $DEPLOY_DIR，执行 git pull..."
    cd "$DEPLOY_DIR"
    # 先暂存本地未提交改动（通常是 .env.docker 的修改）
    git stash push -m "deploy-auto-stash-$TIMESTAMP" || true
    git fetch origin
    git reset --hard "origin/$BRANCH"
    git stash pop || true
else
    log "首次克隆仓库到 $DEPLOY_DIR..."
    git clone --depth 1 --branch "$BRANCH" "$REPO_URL" "$DEPLOY_DIR"
    cd "$DEPLOY_DIR"
fi

CURRENT_COMMIT=$(git rev-parse --short HEAD)
log "当前代码: $CURRENT_COMMIT"

# ─── 4. 数据与模型目录检查 ─────────────────────────────────
for dir in models lightrag_storage chroma_petro charts logs history state; do
    mkdir -p "$dir"
done

if [[ ! -f "models/bge-reranker-v2-m3/model.safetensors" ]]; then
    fail "reranker 模型缺失: models/bge-reranker-v2-m3/\n" \
         "请在宿主机下载：\n" \
         "  pip install huggingface_hub && \n" \
         "  python -c \"from huggingface_hub import snapshot_download; \"\n" \
         "        \"snapshot_download('BAAI/bge-reranker-v2-m3', local_dir='models/bge-reranker-v2-m3')\""
fi
log "reranker 模型已就位。"

# 如果这是首次部署，chroma_petro 和 lightrag_storage 可能为空 —— 这是正常的，
# 系统会在首次查询时自动初始化（或你可以手动执行 ingest_ccus_batch.py）。

# ─── 5. 备份当前运行状态（用于回滚）────────────────────────
BACKUP_DIR="backups/$TIMESTAMP"
mkdir -p "$BACKUP_DIR"

# 备份 compose 当前运行状态（镜像 tag + 容器状态）
$COMPOSE ps > "$BACKUP_DIR/compose_ps.txt" 2>/dev/null || true
$COMPOSE images > "$BACKUP_DIR/compose_images.txt" 2>/dev/null || true

# 备份关键数据（sqlite 是单文件，可以直接 cp）
for f in checkpoint.db project_history.md state/user_profile.md; do
    [[ -f "$f" ]] && cp "$f" "$BACKUP_DIR/" && log "备份: $f"
done

# 保留最近 N 个备份，旧的删掉
if [[ -d backups ]]; then
    ls -1td backups/*/ 2>/dev/null | tail -n +$((BACKUP_RETAIN + 1)) | xargs -r rm -rf
fi

# ─── 6. 环境文件检查 ───────────────────────────────────────
if [[ ! -f ".env.docker" ]]; then
    if [[ -f ".env.docker.example" ]]; then
        cp ".env.docker.example" ".env.docker"
        warn "从 .env.docker.example 创建了 .env.docker，请检查并设置 API_TOKEN。"
    else
        warn "没有 .env.docker，容器将使用默认配置。生产环境务必设置 API_TOKEN。"
    fi
fi

# ─── 7. 构建与启动 ─────────────────────────────────────────
log "停止旧容器..."
$COMPOSE down --remove-orphans || true

log "构建新镜像..."
$COMPOSE build --no-cache

log "启动新容器..."
# 显式传入 nginx 端口，方便同一脚本支持 8080/80/443 多种场景
NGINX_HOST_PORT="$NGINX_HOST_PORT" $COMPOSE up -d

# ─── 8. 健康检查 ───────────────────────────────────────────
log "等待服务就绪（最多 ${HEALTH_TIMEOUT}s）..."
HEALTH_URL="http://127.0.0.1:${NGINX_HOST_PORT}/health"
START_TS=$(date +%s)

while true; do
    ELAPSED=$(($(date +%s) - START_TS))
    if [[ $ELAPSED -ge $HEALTH_TIMEOUT ]]; then
        fail "健康检查超时 (${HEALTH_TIMEOUT}s)。请查看日志:\n" \
             "  $COMPOSE logs --tail 100 backend\n" \
             "  $COMPOSE logs --tail 50 nginx"
    fi

    if curl -fsS "$HEALTH_URL" >/dev/null 2>&1; then
        break
    fi
    sleep 2
done

# 更详细的健康检查：确认依赖全绿
HEALTH_JSON=$(curl -fsS "$HEALTH_URL" 2>/dev/null || echo '{}')
for dep in ollama chroma sqlite reranker; do
    OK=$(echo "$HEALTH_JSON" | python3 -c "import sys,json; d=json.load(sys.stdin).get('checks',{}).get('$dep',{}); print('ok' if d.get('ok') else 'fail')" 2>/dev/null || echo "fail")
    if [[ "$OK" != "ok" ]]; then
        warn "health.$dep 未就绪"
    else
        log "health.$dep OK"
    fi
done

# ─── 9. 清理旧镜像（节省磁盘）──────────────────────────────
log "清理悬空镜像..."
docker image prune -f >/dev/null 2>&1 || true

# ─── 10. 完成 ──────────────────────────────────────────────
EXTERNAL_IP=$(curl -fsS -m 5 http://169.254.169.254/latest/meta-data/public-ipv4 2>/dev/null || \
              curl -fsS -m 5 http://metadata.google.internal/computeMetadata/v1/instance/network-interfaces/0/access-configs/0/external-ip -H "Metadata-Flavor: Google" 2>/dev/null || \
              hostname -I 2>/dev/null | awk '{print $1}' || echo "<your-server-ip>")

log "=== 部署成功 ==="
log "代码: $CURRENT_COMMIT"
log "访问: http://${EXTERNAL_IP}:${NGINX_HOST_PORT}"
log "健康: ${HEALTH_URL}"
log "API 文档: http://${EXTERNAL_IP}:${NGINX_HOST_PORT}/docs"
log "日志: $COMPOSE logs -f backend"
