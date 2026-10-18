# Graphiti 接入评估（时态知识图谱记忆）

> 评估对象：`getzep/graphiti` / `graphiti-core` 0.31.1
> 评估时间：2026-10-18
> 探针脚本：`test_graphiti.py`（需要独立解释器，见文末"复现"）
> 结论：**能力是真的，但这台机器上跑不起来生产级配置；不建议现在接入。**

---

## 1. 一句话结论

Graphiti 的"双时态"不是宣传话术：喂两条带时间戳的记录，第二条写入时它会**自动把第一条标成失效**，
并且能按"某个历史时刻"把图切回当时的样子。实测两个问句都答对了。

代价是：它把复杂度从"存"转移到了"抽"。要抽得准，需要后端服务和强模型；
本机没有 Docker、没有 Java、只有一块 8 GB 显卡和一堆 7B 量化模型，
三者叠加的结果是——**能跑 demo，不能上生产**。

---

## 2. 本机的硬约束（先排除法，再谈可行性）

| 后端 | 状态 | 原因 |
|---|---|---|
| Neo4j 5.26 | ✗ | 要 Docker 或 Java，本机两者都没有 |
| FalkorDB 1.1.2 | ✗ | 同上，需要独立服务 |
| FalkorDBLite | ✗ | `falkordblite` 只有 macOS/Linux wheel，无 Windows 版 |
| **Kuzu 0.11.3** | ✓（但上游已废弃） | 唯一免服务的嵌入式后端，有 `cp313-win_amd64` wheel |

模型侧：

| 用途 | 本机可用 | 说明 |
|---|---|---|
| 抽取/判断（LLM） | `qwen2.5:latest` 7.6B **Q3_K_M** | 4 GB 显存，上下文 4096 |
| 向量化 | `nomic-embed-text` 768 维 | 够用，`EMBEDDING_DIM` 必须改成 768 |

所以这次评估是"在最低配组合下跑"，结论对**低配环境**最有参考价值；
如果换 Neo4j + DeepSeek/GPT 级模型，质量会明显好于下面的数字。

---

## 3. 三个"开箱即坏"的问题（都真实踩到了）

这三个都不是配置问题，是 0.31.1 的包本身有洞。想复现的人一定会撞上。

### 3.1 `import graphiti_core` 直接失败：漏声明 `httpx`

`pip install graphiti-core` 装完之后 `import` 报 `ModuleNotFoundError: No module named 'httpx'`。

原因：graphiti-core 的代码里 `import httpx`（`llm_client/client.py:23`），
但依赖清单里只声明了 `openai>=1.91.0`。而 openai 3.x 已经改用 `httpx2`，
于是 `httpx` 不再被顺带装上——**graphiti 的隐式依赖被上游的版本升级打穿了**。

绕过：`pip install httpx`。

### 3.2 Kuzu 后端第一次写入就崩：`_database` 属性不存在

```
AttributeError: 'KuzuDriver' object has no attribute '_database'
```

`Graphiti._resolve_request_scope()`（`graphiti.py:1325`）里有一句无条件的：

```python
if group_id == self.driver._database:
```

但 `KuzuDriver.__init__` 从来没设置过 `_database`——Neo4j 和 FalkorDB 都设了。
所以 **Kuzu 上只要传 `group_id` 就必崩**。

绕过：构造完 driver 后补一行 `driver._database = GROUP`，让那行提前短路。

### 3.3 FTS 索引从不创建：BM25 检索找不到索引

```
Binder exception: Table RelatesToNode_ doesn't have an index with name edge_name_and_fact
```

- `KuzuDriver.setup_schema()` 只 `CREATE TABLE`，一个全文索引都不建；
- `KuzuDriver.build_indices_and_constraints()` 是 **空实现**，注释写着
  *"Kuzu doesn't support dynamic index creation"*；
- 但检索路径照样 `CALL QUERY_FTS_INDEX('RelatesToNode_', 'edge_name_and_fact', ...)`
  （索引名→表名的映射在 `graph_queries.py:20`）。

结果是第一次 `add_episode` 走到"边去重"就炸。

绕过：手动建 4 个 FTS 索引（脚本里的 `ensure_kuzu_fts_indexes()`）：

```sql
CALL CREATE_FTS_INDEX('Entity',         'node_name_and_summary', ['name', 'summary']);
CALL CREATE_FTS_INDEX('Community',      'community_name',        ['name']);
CALL CREATE_FTS_INDEX('Episodic',       'episode_content',       ['content', 'source_description']);
CALL CREATE_FTS_INDEX('RelatesToNode_', 'edge_name_and_fact',    ['name', 'fact']);
```

> 这三条合起来说明一件事：**Kuzu 后端在 0.31.1 里不是"废弃"，是"坏的"**。
> 上游已经不再维护它，这些问题大概率不会被修。

### 3.4 附带发现：官方 README 的 `json_object` 建议已经过时

官方建议本地模型用 `structured_output_mode="json_object"`（把 schema 塞进 prompt）。
实测这个模式下 7B 会把 **JSON Schema 本身当成内容抄回来**：

```json
{"properties": {"duplicate_facts": {"description": "List of ...", "type": "array"}}, "type": "object"}
```

→ `EdgeDuplicate` 校验失败，`add_episode` 直接中断。

改用默认的 `structured_output_mode="json_schema"` 后问题消失：
Ollama 0.40.3 已支持 `response_format: {"type":"json_schema"}` 的**约束解码**，
语法层面强制模型只能吐这个形状，7B 也照得住。

**这是本次评估最有用的单条经验**：新版 Ollama 上，本地小模型做结构化抽取的可靠性
比"README 说的"要高一个档次。

---

## 4. 实测结果

### 4.1 写入

| | episode 1（2026-10-01） | episode 2（2026-10-17） |
|---|---|---|
| 耗时 | 7.7 s | 13.3 s |
| 抽出节点 | 4 | 4（新增 200bar / 300bar / 断层活化风险） |
| 抽出边 | 3 | 3 条新事实 + 3 条旧事实被置为失效 |

抽出来的图（`A井 -[HAS_PRESSURE]-> 200bar` 这类三元组形式）：

```
A井 --HAS_PRESSURE--> 200bar     valid_at=10-01  invalid_at=10-17   ← 已失效
A井 --HAS_PRESSURE--> 300bar     valid_at=10-17  invalid_at=None    ← 有效
A井 --HAS_PRESSURE--> 从200bar飙升至300bar  valid_at=10-17  None
A井 --HAS_RISK-->     断层活化风险          valid_at=10-17  None
A井 --HAS_STATUS-->   断层/安全状态         valid_at=10-01  invalid_at=10-17  ← 已失效
A井 --HAS_STATUS-->   井口装置/安全状态     valid_at=10-01  invalid_at=None    ← 仍有效
```

**核心能力验证通过**：第二条记录写入时，`200bar` 那条边被自动打上
`invalid_at=2026-10-17T04:00:00`，`断层安全状态` 也一并失效——
后者是模型自己推理出来的（新记录说"断层活化风险"，等于否定了"断层安全"），
不是简单的数值覆盖。

### 4.2 两个问句

| 问句 | 检索方式 | 命中 | 结果 |
|---|---|---|---|
| 10月1日时，A井的状态如何？ | 裸检索 | 6 条 | ✗ **200bar 和 300bar 混在一起**，模型得自己读时间戳 |
| 同上 | as-of 过滤（`valid_at ≤ 10-01` 且当时未失效） | 3 条 | ✓ 只剩 10-01 的三条，无 300bar、无风险 |
| 现在A井有什么风险？ | 裸检索 | 6 条 | ✗ 同样混入已失效的 200bar |
| 同上 | `invalid_at IS NULL` | 4 条 | ✓ 只剩当前有效的四条 |

端到端作答（检索到的事实 → 本地模型组织语言）：

```
问：10月1日时，A井的状态如何？
答：10月1日时，A井的注入压力为200bar，断层处于安全状态，井口装置也处于安全状态。

问：现在A井有什么风险？
答：现在A井存在断层活化风险。
```

两个答案都正确，而且**没有把 300bar 混进 10 月 1 日的回答里**——
这正是普通 RAG 做不到的事。

### 4.3 一个反直觉的点：时态是"数据属性"，不是"默认检索行为"

`graphiti.search()` / `search_()` 默认**不按时间过滤**，已失效的事实照样返回。
时态能力体现在两处：

1. 返回的 `EntityEdge` 带 `valid_at` / `invalid_at` 字段；
2. 你可以用 `SearchFilters(valid_at=..., invalid_at=...)` 在数据库层过滤。

也就是说，**"支持时态查询"不等于"问一句话就能拿到时态答案"**。
要么调用方自己传过滤器，要么把时间戳一起丢给 LLM 让它自己判断（不可靠，见下）。

---

## 5. 查询效果评估

### 做得好的

- **矛盾检测真的会跑**：新旧记录冲突时自动 `invalid_at`，且能推出间接矛盾
  （新记录提"断层风险"→ 旧的"断层安全"失效）。
- **as-of 过滤是数据库层做的**，不是应用层筛，检索 0.1 s，规模上不会退化。
- **抽取质量在这个配置下意外地能看**：7.6B / Q3 量化 / 4096 上下文的组合，
  两条短记录抽出的三元组基本没错。

### 不够好的

- **"没提到"被当成"仍然有效"**：10-17 的记录没提井口装置，
  所以 `井口装置→安全状态` 一直有效。对工程监控这是隐患——
  设备已经拆了，图谱里还是"安全"。
- **会抽冗余事实**：同一件事抽出 `300bar` 和 `从200bar飙升至300bar` 两条边，
  去重没合并掉。
- **实体切分偏碎**：`200bar`、`安全状态` 被当成实体节点。
  这可以用 `entity_types` / `edge_types` 传 pydantic 模型来约束，
  但**约束越强 prompt 越长**，4096 的上下文撑不住真实日报（几百到上千字）。
- **4096 上下文是个静默杀手**：Ollama 超长会**悄悄截断**，不报错，
  只是抽取质量莫名下降。真实数据一上来就会撞到这条线。

### 和现有系统的对比

| | 现在的 LightRAG 图谱 | Graphiti |
|---|---|---|
| 时间维度 | 无（`build_fragments()` 返回的是纯文本行，不带任何时间字段） | 双时态，内建 |
| 冲突处理 | 新旧事实并存，检索时 Top-2 截断，靠运气 | 新事实写入时自动失效旧事实 |
| 写入成本 | 一次性建库 | **每条记录 10 s 级 + 一次 LLM 抽取** |
| 后端 | 本地文件 | 需要 Neo4j/FalkorDB（本机装不了） |

---

## 6. 封装成 MCP Server 的难度

### 结论：工具层很简单，难的全在别处。

**好消息：官方已经写好了，不用自己封装。**

`getzep/graphiti` 仓库里有 `mcp_server/`，暴露 13 个工具：

```
add_memory, search_nodes, search_memory_facts, add_triplet,
get_episodes, get_episode_entities, get_entity_edge,
delete_entity_edge, delete_episode, clear_graph,
summarize_saga, build_communities, get_status
```

而且它的依赖是 **`mcp>=2,<3`**——和我们项目里 `mcp>=2.3.0` 同一代，
不会出现 FastMCP / MCPServer 的版本打架。这一点很关键：
如果它还在 `mcp<2`，整个接入方案就得推倒。

**坏消息：官方 MCP server 只支持 neo4j 和 falkordb，没有 kuzu。**

`mcp_server/src/config/schema.py` 里只有 `Neo4jProviderConfig` 和 `FalkorDBProviderConfig`，
默认 `provider='falkordb'`。所以在这台机器上，官方 server **起不来**——
要么装 Docker，要么自己写一个 Kuzu 版的。

### 分项难度

| 层面 | 难度 | 具体工作量 / 卡点 |
|---|---|---|
| 工具层（暴露两个 MCP 工具） | **低** | 照 `mcp_server_ccus.py` 的写法加 `add_episode` / `query_as_of`，半天量级 |
| 后端层（让它能在本机跑） | **高** | 官方 server 不支持 Kuzu；Kuzu 本身又已废弃且带 3 个 bug |
| 模型层（抽取质量） | **中** | 7B + 4096 ctx 只够 demo；真实日报需要更大上下文或换云端模型 |
| 一致性层（双图谱） | **中高** | LightRAG 和 Graphiti 双写，删除/更新要同步；`evolution_engine` 还在往 LightRAG 写先验 |

### 判断

**不建议现在接入。** 不是因为能力不行，是因为"接入"这件事会同时引入：

1. 一个需要 Docker 的图数据库服务（本机没有）；
2. 一个已被上游废弃、还带三个 bug 的备用后端；
3. 一套和现有 LightRAG 平行的第二图谱，双写一致性要自己维护；
4. 每条记录的抽取成本从"0"变成"一次 LLM 调用"。

**如果确实需要时态能力，更划算的路径是给现有图谱加时间字段**：
`build_fragments()` 现在返回的是 `List[str]`，不带任何元数据；
在带时间信息的文档入库时给 fragment 挂上 `valid_at` / `invalid_at`，
检索时按查询时间过滤。这能覆盖本次评估里 80% 的价值（历史时刻状态、新旧冲突），
成本是改一个 builder + 一个检索条件，不需要新服务、新依赖、新模型。

**什么时候值得重新评估 Graphiti**：如果将来有 Neo4j 或 FalkorDB 可用，
且能用上 DeepSeek/GPT 级别的抽取模型，那时它的"自动矛盾检测"才真正划算——
因为这项能力靠手写规则是补不出来的。

---

## 7. 复现

```bash
# 1. 独立 venv（不要污染项目 .venv）
#    注意：环境里的 http_proxy 会让 pip 卡死在 files.pythonhosted.org，
#    必须绕开代理 + 用国内镜像，否则装 13 分钟一个字节都不动。
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
python -m venv D:/python-learn/graphiti-probe
D:/python-learn/graphiti-probe/Scripts/python.exe -m pip install \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    "graphiti-core[kuzu]" httpx      # httpx 必须手动补，见 3.1

# 2. 跑探针
cd D:/python-learn/petro-rag-agent
D:/python-learn/graphiti-probe/Scripts/python.exe test_graphiti.py
```

预期输出：两条 episode 共约 20 s 抽完，两个问句都答对，as-of 过滤把 6 条切成 3 条。

脚本里已经内置了 3.1–3.4 四个问题的绕过，注释标了出处；
换到别的机器上如果这些 bug 被修了，绕过代码是无害的（最多多建一次已存在的索引）。

`test_graphiti.py` 在主 `.venv` 里被 import 时会抛 `SkipTest` 优雅跳过，
不会影响 `unittest discover` 和 CI。
