# ─────────────────────────────────────────────────────────────
# Gunicorn 配置（生产）
#
#   gunicorn api_server:api --config gunicorn.conf.py
#
# 为什么不是 `uvicorn --workers N`：
#   uvicorn 自带的 --workers 只是起 N 个进程，没有 master 进程管理 ——
#   某个 worker 死掉不会被拉起，也不会优雅重启。gunicorn 的 arbiter 负责
#   这些，这是"生产级"和"能跑"的区别。
#
# 为什么 worker_class 是 uvicorn_worker.UvicornWorker 而不是
# uvicorn.workers.UvicornWorker：
#   后者在 uvicorn 0.54 里已经打 DeprecationWarning，官方让它独立成
#   uvicorn-worker 包。用旧路径等于给下次升级埋雷 —— 某次 uvicorn 小版本
#   一删，容器启动就 ImportError。
# ─────────────────────────────────────────────────────────────
import os
import sys

# ── 监听 ──────────────────────────────────────────────────
bind = "0.0.0.0:8000"

# ── 进程模型 ──────────────────────────────────────────────
# **worker 数被强制限制为 1。** 不是保守，是实测：
#
# checkpoint.db 用 SqliteSaver，它只支持同步接口，跨进程没有互斥。两个 worker
# 各自 connect 同一个库，并发写会失败。实测（Windows，2 进程 × 60 次写 × 10 轮
# = 1,200 次）：默认配置 3 次失败，加 WAL + busy_timeout 仍然 3 次失败。
# 异常是 `attempt to write a readonly database`，抛在 SqliteSaver.put 的
# INSERT 上 —— 也就是**任何一个节点跑完都可能整条任务崩掉**，不只是 HITL。
#
# WAL 治不了它：WAL 解决 SQLITE_BUSY（锁等待），而这里是 Windows VFS 把共享
# 冲突映射成 SQLITE_READONLY，busy handler 不会被调用。所以"多 worker + WAL"
# 是个听起来合理、实测不成立的组合。
#
# 并发不靠多进程：worker_class 是 UvicornWorker，请求在事件循环里并发，
# 重活（LLM、检索、重排）都在 asyncio.to_thread 里跑，一个进程能同时服务多个
# 请求。多开 worker 唯一确定买到的东西是**每个进程各一份 2.2 GB 的 reranker**。
#
# 真要回到多 worker，得先换掉 sqlite checkpointer（Postgres/Redis），
# 然后设 CHECKPOINT_ALLOW_MULTI_WORKER=1 解除这个限制。


def resolve_worker_count(requested: int, allow_multi: bool) -> int:
    """把 WEB_CONCURRENCY 收敛到 checkpoint.db 能承受的值。

    单独抽成函数是为了能被测试直接调用 —— 写死在模块顶层的话，
    "多 worker 被降级"这件事只能靠读日志确认。
    """
    if allow_multi or requested <= 1:
        return max(1, requested)
    print(
        f"[gunicorn] WEB_CONCURRENCY={requested} 被强制降为 1："
        "checkpoint.db（SqliteSaver）跨进程没有互斥，多写者会以 "
        "'attempt to write a readonly database' 失败（实测 3/1200 次写）。"
        "并发由 asyncio + 线程池承担，不需要多进程。"
        "若已换成 Postgres/Redis checkpointer，设 CHECKPOINT_ALLOW_MULTI_WORKER=1。",
        file=sys.stderr,
    )
    return 1


workers = resolve_worker_count(
    int(os.getenv("WEB_CONCURRENCY", "4")),
    os.getenv("CHECKPOINT_ALLOW_MULTI_WORKER") == "1",
)
worker_class = "uvicorn_worker.UvicornWorker"

# 不用 preload_app 来省内存，原因是它会直接破坏功能：
#   · lightrag_retriever 在 import 时启动一个后台线程跑 LightRAG 的 event loop；
#   · 线程不会随 fork 复制，子进程里那个 loop 根本没在跑；
#   · 于是 _submit 把协程投进去以后永远等不到结果 → 检索挂死。
# 换句话说，preload 省下的内存换来的是"看起来起来了但检索不工作"。
# 单 worker 之后内存本来就不紧张（一份 reranker 2.2 GB），更没必要冒这个险。

# ── 超时 ──────────────────────────────────────────────────
# gunicorn 默认 30 s —— 一次评估要 90-150 s，默认值会把正在干活的 worker
# 直接杀掉，表现为"报告生成到一半连接断了"，而日志里只有一行 worker timeout。
timeout = 900
graceful_timeout = 60
# 与 nginx upstream keepalive 配合：空闲连接保留 75 s，避免每次请求重建 TCP
keepalive = 75

# ── 日志 ──────────────────────────────────────────────────
# 输出到 stdout/stderr，由 Docker 收集（`docker compose logs -f backend`）
accesslog = "-"
errorlog = "-"
loglevel = os.getenv("LOG_LEVEL", "info").lower()
access_log_format = '%(h)s "%(r)s" %(s)s %(b)s %(L)ss %({x-forwarded-for}i)s'

# ── 代理头 ────────────────────────────────────────────────
# 只信任来自 nginx 的 X-Forwarded-*。容器网络里只有 nginx 会连后端；
# 若后端端口被直接暴露到公网，应改成具体的网段。
forwarded_allow_ips = "*"

# ── 容器内的心跳文件 ──────────────────────────────────────
# 默认放 /tmp，某些存储驱动上 fsync 慢会让 arbiter 误判 worker 卡死。
# /dev/shm 是内存盘，Docker 默认给 64 MB，一个心跳文件足够。
worker_tmp_dir = "/dev/shm"

# ── 不设置 max_requests ───────────────────────────────────
# 周期性重启 worker 能治内存泄漏，但这里每次重启都要重新加载 2.2 GB 模型。
# 代价远大于收益，所以显式关掉并写明原因，而不是留个默认值让人猜。
max_requests = 0


def _warn_if_memory_short() -> None:
    """启动时算一遍内存账。不阻止启动，但要让"为什么会被 OOM kill"有据可查。

    只在 Linux 上工作（读 /proc/meminfo），其他平台静默跳过。
    """
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            info = {
                line.split(":")[0]: int(line.split()[1])
                for line in f
                if ":" in line
            }
    except OSError:
        return

    available_gb = info.get("MemAvailable", 0) / 1024 / 1024
    # 每个 worker 的常驻内存估计：模型 2.2 GB + torch/onnx 运行时开销
    per_worker_gb = 2.6
    needed_gb = workers * per_worker_gb

    if available_gb and needed_gb > available_gb * 0.9:
        # 注意：worker 数已经被 resolve_worker_count 钳到 1，"调小 WEB_CONCURRENCY"
        # 这条建议在这里已经无效 —— 能调的只剩容器内存。
        print(
            f"[gunicorn] ⚠️ 内存可能不够：{workers} 个 worker × {per_worker_gb:.1f} GB "
            f"≈ {needed_gb:.0f} GB，当前可用 {available_gb:.1f} GB。\n"
            f"[gunicorn]    每个 worker 都会各自加载一份 2.2 GB 的重排模型；"
            f"worker 数已固定为 1，请给容器更多内存。",
            flush=True,
        )


_warn_if_memory_short()
