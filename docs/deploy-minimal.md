# 16 GB 机器上的最小可用配置

这份文件回答一个具体问题：**一台 16 GB 内存、有一块消费级 GPU 的机器，怎么把
整套系统跑起来，以及哪一步会先崩。**

README 的 [Deployment](#deployment-docker) 讲的是架构与取舍，这里只给可照抄的
配置和一张内存账。所有数字都标了来源：`实测` 是本机 `docker stats` 读出来的，
`估算` 是规格推算 —— 两者不能混着用。

## 内存账（决定一切的那张表）

| 进程 | 常驻内存 | 来源 | 说明 |
|---|---:|---|---|
| `backend`（含 reranker） | ~1.6 GB | 实测 | 2.2 GB 的 bge-reranker 权重经 mmap 加载，常驻远小于文件体积；同进程内多请求共享 |
| `frontend` | ~60 MB | 实测 | Streamlit，只有一个页面 |
| `nginx` | ~20 MB | 实测 | 反向代理，不缓存 |
| Ollama `qwen2.5`（7B） | ~4.0 GB | 估算 | 4-bit 量化权重 + KV cache（上下文越长越大） |
| Ollama `nomic-embed-text` | ~0.3 GB | 估算 | 768 维，只在建库/检索时活跃 |
| WSL2 / Docker 自身 | ~1.5–2 GB | 估算 | 内核 + 容器运行时 |
| **合计** | **≈ 7.5–8 GB** | | 16 GB 机器有一倍余量 |

**余量看着很大，但有两件事不能做：**

1. **不要加第二个 worker。** 每个 worker 是一份独立进程，会再加载一份 reranker
   （+2.2 GB 文件、约 +1.5 GB 常驻）与一份模型客户端。真正的原因不是内存而是
   正确性：`SqliteSaver` 跨进程没有写锁，实测 2 进程 × 60 写 × 10 轮出现 3 次
   `attempt to write a readonly database`，加 WAL 与 `busy_timeout=30s` 之后
   **仍然 3 次**。所以 `WEB_CONCURRENCY` 被钉在 1，CI 有断言。
2. **不要把 Ollama 塞回 compose。** 容器里的 Ollama 默认拿不到 GPU，7B 会退回
   CPU，地质分支从 ~70 s 变成几分钟；同时它还多占 4 GB 容器内存。

## 最小配置（五步）

```bash
# 1. 宿主机装 Ollama，并让它监听非回环地址（否则容器连不上）
#    Windows:
setx OLLAMA_HOST 0.0.0.0:11434    # 之后重启 Ollama
#    macOS / Linux:
#    OLLAMA_HOST=0.0.0.0:11434 ollama serve

# 2. 拉两个模型（qwen2.5 生成，nomic-embed-text 做 768 维向量）
ollama pull qwen2.5
ollama pull nomic-embed-text

# 3. 放 reranker 权重（2.2 GB，不在 git 里；缺了后端 import 就崩，不是降级）
python -c "from huggingface_hub import snapshot_download as s; \
  s('BAAI/bge-reranker-v2-m3', local_dir='models/bge-reranker-v2-m3')"

# 4. 建向量库与图谱（首次需要，之后可跳过）
python ingest_ccus_batch.py
python graph_rag_test.py

# 5. 起栈
docker compose up -d --build
```

`python scripts/preflight.py` 会在第 5 步之前把上面五项逐条验一遍，并把
「缺什么 → 实际会看到什么报错 → 怎么修」直接打出来。**先跑它**，因为缺
reranker 的症状是 `OSError: Unrecognized processing class`，指向
sentence-transformers，方向完全错。

## 资源上限（可选但建议）

`docker-compose.yml` 里没有写 `mem_limit`，因为写了就得给一个不随机器变化的数，
而 16 GB 与 64 GB 的机器该给的值不同。如果要限，按上表留 1 GB 余量：

```yaml
services:
  backend:
    mem_limit: 4g          # 1.6 GB 常驻 + 报告生成时的峰值
  frontend:
    mem_limit: 512m
```

限了之后 OOM 会变成"容器被杀"而不是"整机卡死" —— 前者有日志，后者没有。

## 已知会先崩的三处

| 症状 | 真实原因 | 处理 |
|---|---|---|
| 第一次评估就要等 3–5 分钟 | Ollama 冷启动 + 首次加载 7B | 先 `ollama run qwen2.5 "hi"` 预热 |
| 长报告生成中途 `Connection error` | 上下文超 4096，模型侧截断 | 见 README 的 `num_ctx` 说明；不要把整篇证据都塞进一次调用 |
| 容器起来但整站 502 | nginx upstream 缺 `resolver`，重建 backend 后 IP 变了 | `zone + resolve + resolver 127.0.0.11`（README 有完整片段） |

## 不在这份文件里的东西

- **不承诺吞吐。** 单 worker + 本地 7B 的并发能力取决于报告长度，本仓库没有做过
  压测，任何"每秒多少请求"的数字都会是编的。
- **不支持多机。** 没有队列、没有对象存储、没有模型服务分离；`SqliteSaver` 与
  本地文件记忆都是单机设计。横向扩展不是配置问题，是架构问题。
