#!/usr/bin/env bash
#
# 本地把控制台跑起来：Ollama → 后端 → 前端，然后打印地址。
#
# 这个脚本存在的理由很具体：这套界面要能打开，需要**三个**进程同时活着，
# 而它们的启动时间差得很远 —— Ollama 是秒级，后端要十几秒（reranker 3.95 GB
# 先加载），前端是秒级。手敲三遍命令再等，每次都要凭感觉猜"起好了没有"，
# 猜错的后果是界面显示"后端不可达"而人以为是后端挂了。
#
# 所以这里不做别的，只把"起好了没有"变成可检查的事实：
# 每一步都等到**能验证的状态**（/health 返回 ok、4173 返回 200）才往下走。
#
# ```bash
# ./start-local.sh            # 起（已经在跑的就跳过）
# ./start-local.sh --stop     # 停
# ./start-local.sh --status   # 只看状态，什么都不动
# ```
#
# 用 Git Bash / WSL 跑。Windows 双击请用同目录的 `start-local.bat`。

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PY="${PY:-/d/python-learn/.venv/Scripts/python.exe}"
OLLAMA="${OLLAMA:-/c/Users/yeeun/AppData/Local/Programs/Ollama/ollama.exe}"
BACKEND_PORT="${BACKEND_PORT:-8000}"
WEB_PORT="${WEB_PORT:-4173}"
LOG_DIR="$ROOT/logs"
BACKEND_LOG="$LOG_DIR/backend.log"
WEB_LOG="$LOG_DIR/web.log"
PID_DIR="$ROOT/.run"

mkdir -p "$LOG_DIR" "$PID_DIR"

ok()   { printf '  \033[32m✓\033[0m %s\n' "$*"; }
warn() { printf '  \033[33m!\033[0m %s\n' "$*"; }
bad()  { printf '  \033[31m✗\033[0m %s\n' "$*"; }
step() { printf '\n\033[1m%s\033[0m\n' "$*"; }

# 端口上有没有人听。用 curl 而不是 netstat：我们要知道的是"能不能用"，
# 而不是"有没有 socket"。一个占着端口但不响应的进程，在 netstat 里长得
# 和正常服务一模一样。
#
# ## 为什么必须 `--noproxy '*'` 和 `-f`
#
# 这台机器的环境里有 `http_proxy=http://127.0.0.1:56973`。curl 会走它，
# 于是"目标端口没有服务"这件事被代理变成了一次 **502 响应** —— 而 curl
# 对 502 的退出码是 **0**。结果：`if web_ok` 为真，脚本报告"前端已在运行"，
# 然后把地址打印给用户，用户打开是一片空白。
#
# 实测过：把后端和前端都停掉，脚本仍然三行全 ✓ 并直接跳到"可以用了"。
# `--noproxy '*'` 是根本修法（本地探针不该经过任何代理），`-f` 是保险
# （后端真的 500 时，我们也该知道它不正常）。
probe() {
  curl -sf --noproxy '*' -m 3 -o /dev/null "$1" 2>/dev/null
}

# 等到某个条件成立，最多等 $2 秒。等的是状态，不是时长。
wait_for() {
  local label="$1" secs="$2"
  shift 2
  local i=0
  while [ "$i" -lt "$secs" ]; do
    if "$@" >/dev/null 2>&1; then return 0; fi
    i=$((i + 1))
    sleep 1
    # 每 5 秒报一次，免得人以为卡死了。后端冷启动确实要十几秒。
    if [ $((i % 5)) -eq 0 ]; then printf '    … %s（%ds）\n' "$label" "$i"; fi
  done
  return 1
}

# `/health` 会**串行探四个依赖**，其中 Ollama 那一项本身就要 2 秒
# （实测 latency_ms≈2024）。3 秒超时会让一个完全正常的后端被判成"没起"，
# 于是脚本再起一个 uvicorn —— 端口冲突、新进程立刻死掉，但 `wait_for`
# 因为旧进程在响应而成功，最后报一句"已启动"。用户看到的是一切正常，
# 实际多了一个死进程和一条 address already in use。
#
# 所以 health 用 15 秒；web / ollama 那两个探针仍然 3 秒（它们很快）。
health_ok() {
  curl -sf --noproxy '*' -m 15 "http://127.0.0.1:$BACKEND_PORT/health" 2>/dev/null | grep -q '"status":"ok"'
}

health_any() {
  curl -sf --noproxy '*' -m 15 "http://127.0.0.1:$BACKEND_PORT/health" 2>/dev/null | grep -q '"status"'
}

web_ok() {
  probe "http://127.0.0.1:$WEB_PORT/"
}

ollama_ok() {
  probe "http://127.0.0.1:11434/api/tags"
}

status() {
  step "当前状态"
  if ollama_ok; then ok "Ollama  127.0.0.1:11434"; else bad "Ollama  未启动"; fi
  if health_ok; then
    ok "后端    127.0.0.1:$BACKEND_PORT  /health = ok"
  elif health_any; then
    # 起来了但 degraded —— 这是最容易被当成"好了"的状态：进程活着、
    # 界面能打开、告警也能出，只有检索和 embedding 会静默失败。
    warn "后端    127.0.0.1:$BACKEND_PORT  /health = degraded（多半是 Ollama 没起）"
  else
    bad "后端    未启动"
  fi
  if web_ok; then ok "前端    127.0.0.1:$WEB_PORT"; else bad "前端    未启动"; fi
}

stop_all() {
  step "停止"
  for name in web backend ollama; do
    local f="$PID_DIR/$name.pid"
    if [ -f "$f" ]; then
      local pid
      pid="$(cat "$f")"
      # 只杀自己起的那一个。按端口杀会把别人的进程一起带走 ——
      # 而"别人的进程"包括另一个正在调试这个项目的自己。
      if kill "$pid" 2>/dev/null; then ok "已停止 $name（pid $pid）"; else warn "$name 进程已不在（pid $pid）"; fi
      rm -f "$f"
    fi
  done
  # `npx vite` 会再 spawn 一个 node，杀掉 npx 不一定杀得掉它。
  # 不在这里擅自 `pkill -f` —— 那会连别人起的同类进程一起带走。
  # 所以只如实报告端口是否还被占着，让人自己决定。
  sleep 1
  if web_ok; then
    warn "端口 $WEB_PORT 仍被占用（多半是 npx 派生的 node 子进程）"
    warn "手动关：pkill -f 'vite preview'  或从任务管理器结束 node"
  fi
  if health_any; then
    warn "端口 $BACKEND_PORT 仍被占用（后端不是本脚本起的）"
  fi
  warn "只停本脚本起的进程；手动起的那些请自己关"
}

start_ollama() {
  step "1/3 Ollama"
  if ollama_ok; then
    ok "已在运行"
    return 0
  fi
  if [ ! -x "$OLLAMA" ] && ! command -v ollama >/dev/null 2>&1; then
    bad "找不到 ollama。装它：https://ollama.com/download"
    warn "不装也能起界面，但检索与 embedding 会失败（/health 显示 degraded）"
    return 1
  fi
  local bin="$OLLAMA"
  [ -x "$bin" ] || bin="$(command -v ollama)"
  nohup "$bin" serve >>"$LOG_DIR/ollama.log" 2>&1 &
  echo $! >"$PID_DIR/ollama.pid"
  if wait_for "等 Ollama" 30 ollama_ok; then
    ok "已启动（日志 logs/ollama.log）"
  else
    bad "Ollama 30 秒没起来，看 logs/ollama.log"
    return 1
  fi
}

start_backend() {
  step "2/3 后端"
  # 先问"端口上有没有人答话"，而不是"答得好不好"。
  #
  # 只判断 `health_ok` 的话，一个 degraded 的后端（Ollama 没起时就是）
  # 会被当成"没起"，于是再起一个 uvicorn：端口冲突、新进程立刻退出，
  # 而 `wait_for` 因为**旧进程**在响应而成功 —— 最后报"已启动"，
  # 一切看起来正常，只是多了一个死进程。
  if health_any; then
    if health_ok; then
      ok "已在运行 · /health = ok"
    else
      warn "已在运行但 /health = degraded —— 界面能用，检索会失败"
      warn "多半是 Ollama 没起；看 logs/backend.log 里哪一项 ok=false"
    fi
    return 0
  fi
  if [ ! -x "$PY" ]; then
    bad "找不到 Python：$PY"
    warn "用 PY=/path/to/python ./start-local.sh 指定"
    return 1
  fi

  # 阈值文件不进仓库（里面是某个项目的设计限值）。不配的话 18 个测点全部
  # "未配置阈值"，界面显示 18 项不可判定、0 需动作 —— 那不是"系统正常"，
  # 是"没人定过线"。本地跑用 example 是合理的，但底栏会原样显示【替换】。
  local thr="${OPS_THRESHOLDS_PATH:-config/ops_thresholds.example.json}"
  if [ ! -f "$thr" ]; then
    bad "找不到阈值文件：$thr"
    return 1
  fi

  OPS_THRESHOLDS_PATH="$thr" nohup "$PY" -m uvicorn api_server:api \
    --host 127.0.0.1 --port "$BACKEND_PORT" >>"$BACKEND_LOG" 2>&1 &
  echo $! >"$PID_DIR/backend.pid"

  # 后端冷启动要加载 reranker（3.95 GB），实测 15~20 秒。给 90 秒是给
  # 冷文件缓存的第一跑留余量，不是常态。
  if wait_for "等后端 /health 可用（首次要加载 3.95 GB reranker）" 90 health_any; then
    if health_ok; then
      ok "已启动 · /health = ok（日志 logs/backend.log）"
    else
      warn "已启动但 /health = degraded —— 界面能用，检索会失败"
      warn "多半是 Ollama 没起。先看 logs/backend.log 里哪一项 ok=false"
    fi
  else
    bad "后端 90 秒没起来，最后 20 行："
    tail -20 "$BACKEND_LOG" | sed 's/^/      /'
    return 1
  fi
}

start_web() {
  step "3/3 前端"
  if web_ok; then
    ok "已在运行"
    return 0
  fi
  if [ ! -d "$ROOT/web/dist" ]; then
    warn "web/dist 不存在，先构建（约 10 秒）"
    ( cd web && npm run build >>"$WEB_LOG" 2>&1 ) || {
      bad "构建失败，看 logs/web.log"
      return 1
    }
  fi
  ( cd web && nohup npx vite preview --port "$WEB_PORT" --host 127.0.0.1 >>"$WEB_LOG" 2>&1 & echo $! >"$PID_DIR/web.pid" )
  if wait_for "等前端" 40 web_ok; then
    ok "已启动（日志 logs/web.log）"
  else
    bad "前端 40 秒没起来，看 logs/web.log"
    return 1
  fi
}

case "${1:-}" in
  --stop)   stop_all; exit 0 ;;
  --status) status; exit 0 ;;
  "")       ;;
  *)        echo "用法：$0 [--stop|--status]"; exit 2 ;;
esac

start_ollama
ollama_rc=$?
start_backend || exit 1
start_web || exit 1

step "可以用了"
printf '\n  \033[1mhttp://127.0.0.1:%s/?source=live\033[0m\n\n' "$WEB_PORT"
cat <<'HINT'
  ?source=live  打真后端（读数用抓 fixture 时录的那批回放，不是实时时序）
  不加参数      读 src/fixtures 的快照，后端挂了也能打开 —— 演示时用它兜底

  其他入口
    ?mode=engineer        工程师模式（卡片 / 表格切换）
    ?tab=chain            全链视图（捕集 → 运输 → 注入 → MVA）
    ?rail=dialog          右栏对话
    ?kb=search            右栏知识库检索
    ?rail=off             收起右栏（窄屏）

  看状态   ./start-local.sh --status
  停       ./start-local.sh --stop
  体检     cd web && npm run smoke
HINT

if [ "$ollama_rc" -ne 0 ]; then
  warn "Ollama 没起来：界面能打开，但检索和对话会失败"
  exit 1
fi
