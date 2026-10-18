#!/usr/bin/env python
"""Graphiti 时态记忆探针 —— 评估用，不属于 CI。

为什么单独一个解释器：
    本机项目 .venv 里没有 graphiti-core（也不该有，这是评估而不是依赖）。
    用 D:/python-learn/graphiti-probe 这个一次性 venv 跑：
        D:/python-learn/graphiti-probe/Scripts/python.exe test_graphiti.py

它验证什么：
    1. Kuzu 后端（本机唯一不需要 Docker/Java 的后端）能不能起来；
    2. 两条带时间戳的 episode 能不能被抽出实体/关系；
    3. 后一条（300bar）会不会把前一条（200bar）**自动置为失效** ——
       这是 Graphiti 的核心卖点，也是这次评估的真正问题；
    4. "10月1日时状态如何" / "现在有什么风险" 两个问句，能不能被时态过滤分开。

注意：Kuzu 后端已被上游标记 deprecated（graphiti 0.31 会打 DeprecationWarning）。
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------- 运行环境
PROBE_ROOT = Path(__file__).resolve().parent
DB_DIR = PROBE_ROOT / "graphiti_demo.kuzu"

OLLAMA_V1 = "http://localhost:11434/v1"
LLM_MODEL = "qwen2.5:latest"
EMBED_MODEL = "nomic-embed-text:latest"
EMBED_DIM = 768  # nomic-embed-text 是 768；graphiti 默认 1024，必须改

# graphiti 在 import 时就会读这几个变量
os.environ.setdefault("EMBEDDING_DIM", str(EMBED_DIM))
os.environ.setdefault("SEMAPHORE_LIMIT", "1")          # 单卡 8G，别并发打爆
os.environ.setdefault("GRAPHITI_TELEMETRY_ENABLED", "false")
os.environ.setdefault("OPENAI_API_KEY", "ollama")       # 占位，本地不需要真 key

warnings.filterwarnings("ignore", category=DeprecationWarning)

try:  # noqa: E402
    from graphiti_core import Graphiti
    from graphiti_core.cross_encoder.client import CrossEncoderClient
    from graphiti_core.driver.kuzu_driver import KuzuDriver
    from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
    from graphiti_core.llm_client.config import LLMConfig
    from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
    from graphiti_core.nodes import EpisodeType
    from graphiti_core.search.search_config_recipes import COMBINED_HYBRID_SEARCH_RRF
    from graphiti_core.search.search_filters import (
        ComparisonOperator,
        DateFilter,
        SearchFilters,
    )
except ModuleNotFoundError as _exc:
    # 主 .venv 里没有 graphiti（也不该有）。没有这个分支的话，
    # 在仓库根目录跑 `unittest discover` 会收集到这个文件并 ImportError，
    # 把整个测试集带崩。SkipTest 是 unittest 和 pytest 都认的跳过信号。
    if (_exc.name or "").split(".")[0] in {"graphiti_core", "kuzu"}:
        import unittest

        raise unittest.SkipTest(
            f"Graphiti 探针需要独立解释器（D:/python-learn/graphiti-probe）：{_exc}"
        ) from None
    raise

GROUP = "well_a"


class NoopReranker(CrossEncoderClient):
    """Ollama 的 /v1 不支持 logprobs，OpenAIRerankerClient 会拿不到分数。

    搜索配方用 RRF（倒数排名融合）就完全不碰 cross-encoder，
    这里给个哑实现只是为了不让 Graphiti 构造默认的 OpenAI 重排器。
    """

    async def rank(self, query: str, passages: list[str]) -> list[tuple[str, float]]:
        return [(p, 1.0 - i * 0.01) for i, p in enumerate(passages)]


class TolerantGenericClient(OpenAIGenericClient):
    """第三个坑：7B 模型会把 schema 当"内容"再包一层。

    在 json_object 模式下 graphiti 把 pydantic 的 JSON Schema 直接贴进 prompt，
    qwen2.5:7B 有时会照着 schema 的形状回一个信封：
        {"properties": {"duplicate_facts": [...], "contradicted_facts": [...]},
         "type": "object"}
    然后 EdgeDuplicate.model_validate() 报 Field required（缺两个字段）。
    官方的重试只覆盖 JSONDecodeError / RateLimitError，不管 ValidationError，
    所以这里在返回前把信封拆开。拆不出来就原样返回，让它照常报错。
    """

    _ENVELOPE_KEYS = ("properties", "schema", "data", "result", "output", "answer")

    async def generate_response(self, messages, response_model=None, **kwargs):
        data = await super().generate_response(messages, response_model, **kwargs)
        return self._unwrap(data, response_model)

    @classmethod
    def _unwrap(cls, data, response_model):
        if response_model is None or not isinstance(data, dict):
            return data
        fields = set(response_model.model_fields)
        if fields & set(data):
            return data
        for key in cls._ENVELOPE_KEYS:
            inner = data.get(key)
            if isinstance(inner, dict) and fields & set(inner):
                return inner
        return data


def build_graphiti() -> Graphiti:
    llm = TolerantGenericClient(
        config=LLMConfig(
            api_key="ollama",
            model=LLM_MODEL,
            small_model=LLM_MODEL,
            base_url=OLLAMA_V1,
            temperature=0.0,   # 默认是 1，抽取任务需要确定性
            max_tokens=2048,
        ),
        # 官方 README 建议本地模型用 json_object（"schema 塞进 prompt"），
        # 但那是老版 Ollama 的限制。0.40.3 已经支持 json_schema 约束解码，
        # 语法层面强制模型只能吐这个形状 —— 实测换过来之后 7B 不再抄 schema。
        structured_output_mode="json_schema",
        max_tokens=2048,
    )
    embedder = OpenAIEmbedder(
        config=OpenAIEmbedderConfig(
            embedding_model=EMBED_MODEL,
            embedding_dim=EMBED_DIM,
            api_key="ollama",
            base_url=OLLAMA_V1,
        )
    )
    driver = KuzuDriver(db=str(DB_DIR))
    # ── 上游 bug 绕过 ────────────────────────────────────────────────────
    # Graphiti._resolve_request_scope() 里有一句无条件的
    #     if group_id == self.driver._database:
    # 但 KuzuDriver.__init__ 从没设置过 _database（Neo4j/FalkorDB 都设了），
    # 于是 Kuzu 上只要传 group_id 就 AttributeError。把 _database 补成我们的
    # group_id，让这行提前短路返回，就不会再走 driver.clone()。
    driver._database = GROUP
    return Graphiti(llm_client=llm, embedder=embedder, cross_encoder=NoopReranker(), graph_driver=driver)


def hr(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ── 第二个上游坑：Kuzu 后端开箱即坏 ────────────────────────────────────
# KuzuDriver.setup_schema() 只 CREATE TABLE，一个 FTS 索引都不建；
# 而 build_indices_and_constraints() 在 Kuzu 上是 pass（空实现），注释写着
# "Kuzu doesn't support dynamic index creation"。
# 但检索路径照样 CALL QUERY_FTS_INDEX('RelatesToNode_','edge_name_and_fact',...)
# → 第一次 add_episode 走到边去重就炸：
#     Binder exception: Table RelatesToNode_ doesn't have an index ...
# 索引名与表名的映射见 graphiti_core/graph_queries.py:20。
KUZU_FTS_INDEXES = (
    ("Entity", "node_name_and_summary", ["name", "summary"]),
    ("Community", "community_name", ["name"]),
    ("Episodic", "episode_content", ["content", "source_description"]),
    ("RelatesToNode_", "edge_name_and_fact", ["name", "fact"]),
)


def ensure_kuzu_fts_indexes(driver) -> None:
    import kuzu

    conn = kuzu.Connection(driver.db)
    try:
        for table, index, cols in KUZU_FTS_INDEXES:
            try:
                conn.execute(f"CALL CREATE_FTS_INDEX('{table}', '{index}', {cols!r})")
                print(f"  FTS 索引已建：{table}.{index} {cols}")
            except Exception as exc:
                print(f"  FTS 索引跳过 {table}.{index}：{type(exc).__name__}: {exc}")
    finally:
        conn.close()


def show_edge(e) -> None:
    valid = e.valid_at.isoformat() if e.valid_at else "None"
    invalid = e.invalid_at.isoformat() if e.invalid_at else "None  ← 仍然有效"
    print(f"  · {e.fact}")
    print(f"    name={e.name}  valid_at={valid}  invalid_at={invalid}")


async def dump_edges(g: Graphiti) -> list:
    from graphiti_core.edges import EntityEdge

    edges = await EntityEdge.get_by_group_ids(g.driver, [GROUP])
    hr(f"图谱里一共 {len(edges)} 条事实边")
    for e in edges:
        show_edge(e)
    return edges


async def search(g: Graphiti, query: str, label: str, filters: SearchFilters | None = None):
    t0 = time.time()
    try:
        results = await g.search_(
            query,
            config=COMBINED_HYBRID_SEARCH_RRF,
            group_ids=[GROUP],
            search_filter=filters,
        )
    except Exception as exc:  # 过滤器语法在 Kuzu 上可能不被支持，要如实报
        print(f"  ✗ 搜索失败：{type(exc).__name__}: {exc}")
        return []
    edges = results.edges  # search_() 返回 SearchResults，不是 list
    cost = time.time() - t0
    print(f"  [{label}] 命中 {len(edges)} 条（{cost:.1f}s）")
    for e in edges:
        show_edge(e)
    return edges


async def synthesize(question: str, edges: list) -> str:
    """把检索到的事实喂回本地模型，看能不能拼出人话答案。"""
    from openai import AsyncOpenAI

    if not edges:
        return "（无检索结果，无法回答）"
    facts = "\n".join(
        f"- {e.fact}（valid_at={e.valid_at.date() if e.valid_at else '?'}，"
        f"invalid_at={e.invalid_at.date() if e.invalid_at else '至今有效'}）"
        for e in edges
    )
    client = AsyncOpenAI(api_key="ollama", base_url=OLLAMA_V1, timeout=180)
    resp = await client.chat.completions.create(
        model=LLM_MODEL,
        temperature=0,
        max_tokens=400,
        messages=[
            {
                "role": "system",
                "content": "你是 CCUS 现场工程师。只依据给定事实回答，"
                "每条事实后面的 valid_at/invalid_at 是它的生效区间。"
                "如果问题问的是某个历史时刻，只采信那个时刻有效的事实。不要编造。",
            },
            {"role": "user", "content": f"事实：\n{facts}\n\n问题：{question}"},
        ],
    )
    return (resp.choices[0].message.content or "").strip()


async def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    # Kuzu 把整个库落成一个文件（不是目录），两种都要能清掉
    if DB_DIR.is_dir():
        shutil.rmtree(DB_DIR)
    elif DB_DIR.exists():
        DB_DIR.unlink()

    print(f"Python   : {sys.version.split()[0]}")
    print(f"LLM      : {LLM_MODEL} @ {OLLAMA_V1}")
    print(f"Embedder : {EMBED_MODEL} (dim={EMBED_DIM})")
    print(f"Backend  : Kuzu @ {DB_DIR}")

    g = build_graphiti()
    try:
        t0 = time.time()
        await g.build_indices_and_constraints()
        print(f"建索引完成（{time.time() - t0:.1f}s，Kuzu 上其实是空实现）")
        ensure_kuzu_fts_indexes(g.driver)

        # ---------------------------------------------------------- 写入两条 episode
        episodes = [
            (
                "A井 2026-10-01 注入记录",
                "2026年10月1日，A井的注入压力为200bar，井口装置与断层均处于安全状态，无异常。",
                datetime(2026, 10, 1, 4, 0, tzinfo=timezone.utc),
            ),
            (
                "A井 2026-10-17 注入记录",
                "2026年10月17日，A井的注入压力从200bar飙升至300bar，出现断层活化风险，需要立即降低注入速率。",
                datetime(2026, 10, 17, 4, 0, tzinfo=timezone.utc),
            ),
        ]
        for i, (name, body, ref) in enumerate(episodes, 1):
            hr(f"[{i}/2] 写入 episode：{name}（reference_time={ref.isoformat()}）")
            t0 = time.time()
            try:
                result = await asyncio.wait_for(
                    g.add_episode(
                        name=name,
                        episode_body=body,
                        source=EpisodeType.text,
                        source_description="CCUS 现场注入日志",
                        reference_time=ref,
                        group_id=GROUP,
                    ),
                    timeout=1800,
                )
            except asyncio.TimeoutError:
                print("  ✗ add_episode 超过 30 分钟仍未返回，放弃")
                return 2
            except Exception as exc:
                print(f"  ✗ add_episode 失败：{type(exc).__name__}: {exc}")
                return 2
            print(f"  完成（{time.time() - t0:.1f}s）")
            print(f"  抽出节点 {len(result.nodes)} 个、事实边 {len(result.edges)} 条")
            for n in result.nodes:
                print(f"    - 节点：{n.name} ({n.labels})")
            for e in result.edges:
                show_edge(e)

        # ---------------------------------------------------------- 图谱快照
        await dump_edges(g)

        # ---------------------------------------------------------- 时态查询
        hr("Q1  10月1日时，A井的状态如何？  —— 裸检索（不带时间过滤）")
        q1_plain = await search(g, "10月1日 A井 注入压力 状态", "裸检索")

        hr("Q1  同上 —— 带 as-of 时间过滤（valid_at ≤ 10-01 且 当时未失效）")
        as_of = datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc)
        q1_filtered = await search(
            g,
            "A井 注入压力 状态",
            "as-of 过滤",
            SearchFilters(
                valid_at=[[DateFilter(date=as_of, comparison_operator=ComparisonOperator.less_than_equal)]],
                invalid_at=[
                    [DateFilter(comparison_operator=ComparisonOperator.is_null)],
                    [DateFilter(date=as_of, comparison_operator=ComparisonOperator.greater_than)],
                ],
            ),
        )

        hr("Q2  现在A井有什么风险？  —— 裸检索")
        q2_plain = await search(g, "A井 当前 风险 断层活化", "裸检索")

        hr("Q2  同上 —— 只留当前仍有效的事实（invalid_at IS NULL）")
        q2_now = await search(
            g,
            "A井 当前 风险 断层活化",
            "now 过滤",
            SearchFilters(
                invalid_at=[[DateFilter(comparison_operator=ComparisonOperator.is_null)]]
            ),
        )

        # ---------------------------------------------------------- 端到端作答
        hr("端到端作答（检索事实 → 本地模型组织语言）")
        for q, edges in (
            ("10月1日时，A井的状态如何？", q1_filtered or q1_plain),
            ("现在A井有什么风险？", q2_now or q2_plain),
        ):
            print(f"\n问：{q}")
            print(f"答：{await synthesize(q, edges)}")

        return 0
    finally:
        await g.close()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
