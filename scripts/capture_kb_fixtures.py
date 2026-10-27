"""把 `/api/kb/*` 的真实响应抓成前端 fixture。

为什么不用手写 JSON
------------------
与 `capture_ops_fixtures.py` 同一条理由：手写的 fixture 会和契约悄悄分家。
后端多一个字段、少一个枚举值、把 `gaps` 从列表改成字典 —— 这些都不会让
任何东西变红，前端只是继续渲染那份"看起来还对"的旧数据。

为什么不导入 `api_server`
------------------------
它顶部 `from workflow import ...` 会建整张 LangGraph 并加载 2.2 GB 重排模型。
抓知识库 fixture 不需要，也不该需要 —— 这正是 `api_kb` 单独成模块的理由。

**三个端点的可校验性不一样，脚本必须如实区分**
----------------------------------------------
* `overview` / `documents`：纯文件读（`ccus_docs/`、`lightrag_storage/`、
  `chroma.sqlite3`），不碰模型、不碰网络。但"确定"要分两种：
  **与输入无关的确定**（后端怎么序列化）和**与语料相关的确定**（语料里有
  什么）。前者在 CI 里成立，后者不成立 —— `ccus_docs/*.pdf` 与
  `lightrag_storage/` 按设计不入仓库（论文有版权，图谱是 GB 级产物），
  所以 CI 里的语料只有 1 篇。逐字节比对只在有完整语料的开发机上有意义；
  字段集合比对与语料无关，**任何时候都跑**。
* `search`：要跑 LightRAG + Ollama 嵌入 + 图检索，本机约 21 秒，CI 里没有
  Ollama → 只在开发机上抓，**CI 不校验**。

这个差别必须打印出来。一个"fixture 一致"的绿灯如果连 search 也包含了，
而 search 其实从没被比对过，那这个绿灯就是在替没做的检查背书。

**第一版 CI 在这里红过。** 它把 `documents.json` 当无条件逐字节比对，
于是在语料只有 1 篇的 runner 上永远失败 —— 一个永远红的检查比没有检查更糟：
它会训练所有人忽略这个 job，连同里面那些真的能跑的检查。三种情形必须分开
（本机多出文件 = 真漂移，要红；本机缺少文件 = 环境不同，跳过并说明；
完全一致 = 逐字节比对），`tests/test_api_kb.py::FixtureCorpusGateTests` 锁住
这个分法。

用法::

    python scripts/capture_kb_fixtures.py            # 写入 web/src/fixtures/
    python scripts/capture_kb_fixtures.py --check    # 只比对（CI 用，不跑检索）

`--check` 会逐文件打印它走到了哪一层（逐字节 / 字段集合 / 未比对），
不打印一个笼统的"一致"。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT = ROOT / "web" / "src" / "fixtures"

#: 抓检索 fixture 用的固定问题。
#:
#: 选"盖层完整性"而不是某个泛泛的 CCUS 词，是因为它必须命中**多个来源**：
#: 只命中一篇的话，界面上那张"来源表"永远只有一行，引用角标也就退化成
#: 装饰 —— 而多来源正是这个面板要证明的事（实测命中 6 篇、12 个切片）。
SEARCH_QUERY = "盖层完整性对 CO2 泄漏风险的影响"

#: 可确定校验的文件（不依赖模型）。
DETERMINISTIC = ("overview.json", "documents.json")

#: 需要 Ollama 的文件（CI 不校验）。
MODEL_BOUND = "search.json"

#: 内容来自**本机语料**的文件。
#:
#: 这一组比 ``MODEL_BOUND`` 更麻烦，而 CI 第一版正是在这里红的：脚本把
#: ``documents.json`` 当成"确定可比对"，可它的内容是 ``ccus_docs/`` 的目录
#: 列表，而 ``ccus_docs/*.pdf`` 与 ``lightrag_storage/`` 按设计**不入仓库**
#: （论文 PDF 有版权，图谱是 GB 级产物）。于是 CI 里的语料只有 1 篇，
#: 逐字节比对注定失败 —— 一个永远红的检查比没有检查更糟：它会训练所有人
#: 忽略这个 job，连同里面那些真的能跑的检查一起。
#:
#: 所以"确定"要拆成两种：与输入无关的确定（后端怎么序列化），和与语料
#: 相关的确定（语料里有什么）。前者在 CI 里仍然成立，后者不行。
CORPUS_BOUND = ("overview.json", "documents.json")


def _client():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    import api_kb

    app = FastAPI()
    app.include_router(api_kb.router)
    return TestClient(app)


def _ollama_up() -> bool:
    """探一下 Ollama。用标准库而不是 requests：本脚本要能在最小环境里跑。"""
    import urllib.request

    try:
        with urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=2):
            return True
    except Exception:  # noqa: BLE001 — 探不到就是探不到
        return False


def _strip_search(payload: dict) -> dict:
    """去掉每次跑都变的字段。

    ``elapsed_ms`` 是本次检索的真实耗时（本机约 21000），但每次都不同，
    原样落盘会让 `--check` 永远红，然后就没人看了。归零，界面上 0 显示为 "—"。

    **不**去掉 ``reference_id``：它虽然在响应内编号、跨查询不稳定，但同一次
    查询是确定的 —— 它恰恰是"引用角标与来源表对不对得上"这件事的检查对象。
    """
    out = dict(payload)
    out["elapsed_ms"] = 0
    return out


def capture() -> tuple[dict[str, object], str]:
    """返回 ``(payloads, 跳过说明)``。

    跳过说明由 ``main`` 打印，不在 ``capture`` 里 print —— 仓库有一条 AST 守卫
    （``scripts/check_print_usage.py``）规定 ``print`` 只能出现在命令行入口函数里，
    而 ``capture`` 不是入口。它现在不在守卫的扫描列表里，但把它加进去只是时间问题。
    """
    client = _client()
    payloads: dict[str, object] = {}
    note = ""

    overview = client.get("/api/kb/overview")
    overview.raise_for_status()
    payloads["overview.json"] = overview.json()

    documents = client.get("/api/kb/documents")
    documents.raise_for_status()
    payloads["documents.json"] = documents.json()

    if _ollama_up():
        search = client.post(
            "/api/kb/search", json={"query": SEARCH_QUERY, "mode": "hybrid"}
        )
        search.raise_for_status()
        payloads[MODEL_BOUND] = _strip_search(search.json())
    else:
        note = "跳过 search.json：Ollama 未启动（这个 fixture 只能在开发机上抓）"

    return payloads, note


def _dump(payload) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


def _fixture_corpus() -> set[str] | None:
    """从已提交的 ``documents.json`` 读出"这份 fixture 是拿哪些文件抓的"。

    语料清单写在 fixture 里，所以"本机语料与 fixture 是否同一批"这个问题
    不需要任何额外记录就能回答 —— 也不需要把文档数写死在脚本里（写死的话，
    真的入库了一篇新论文时它反而不会红）。
    """
    target = OUT / "documents.json"
    if not target.exists():
        return None
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    rows = data.get("documents") or []
    if not rows:
        return None
    return {str(r.get("name")) for r in rows}


def _corpus_gate(payload: dict) -> tuple[bool, str]:
    """判断本机语料能否与 fixture 逐字节比对。

    三种情形，处置必须不同：

    * **本机多出文件** → 真的漂移（入库了新论文、改了文件名）。这是要红的
      那一类，而且必须红：fixture 里没有它，前端就不会显示它。
    * **本机缺少文件** → 这是"环境不同"，不是漂移。CI 只有 1 篇是因为 PDF
      不入仓库。跳过逐字节比对，但必须把原因说出来。
    * **完全一致** → 逐字节比对。

    把前两种合并成一种"跳过"，等于让新增论文这件事也变成静默的。
    """
    recorded = _fixture_corpus()
    if recorded is None:
        return False, "documents.json 不存在或读不出语料清单，无法判断本机语料是否与它一致"
    live = {str(r.get("name")) for r in payload.get("documents") or []}
    extra = live - recorded
    missing = recorded - live
    if extra:
        raise SystemExit(
            "::error::语料里出现了 fixture 尚未记录的文件：\n  - "
            + "\n  - ".join(sorted(extra))
            + "\n这是真的漂移（新入库 / 改过文件名），请重抓 fixture：\n"
            "  python scripts/capture_kb_fixtures.py\n"
            "重抓后看 git diff —— 如果只是新增一行文档，那是预期内的。"
        )
    if missing:
        head = "、".join(sorted(missing)[:3])
        tail = "…" if len(missing) > 3 else ""
        return False, f"本机语料比 fixture 少 {len(missing)} 篇（{head}{tail}）"
    return True, ""


def _row_shape(payload: dict) -> set[str]:
    """一行文档的字段集合 + 顶层字段集合。

    语料不同时仍能查的东西：字段名。``chunks`` 改名成 ``chunk_count``、
    ``gaps`` 从列表变字典 —— 这些与语料无关，本机只有 1 篇时也查得出来。
    所以"跳过逐字节"不等于"这个文件没人管"。
    """
    keys = set(payload.keys())
    rows = payload.get("documents") or []
    if rows:
        keys |= set(rows[0].keys())
    return keys


def _shape_drift(payload: dict) -> list[str]:
    """对比 fixture 与实况的字段集合，返回差异描述（空列表 = 一致）。"""
    target = OUT / "documents.json"
    try:
        recorded = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    want = _row_shape(recorded)
    got = _row_shape(payload)
    problems: list[str] = []
    if want - got:
        problems.append("缺少字段：" + "、".join(sorted(want - got)))
    if got - want:
        problems.append("多出字段：" + "、".join(sorted(got - want)))
    return problems


def main(argv: list[str]) -> int:
    check_only = "--check" in argv[1:]
    payloads, note = capture()
    if note:
        print(note)

    OUT.mkdir(parents=True, exist_ok=True)
    stale: list[str] = []
    compared: list[str] = []
    skipped: list[str] = []
    corpus_note = ""
    gated = False
    for name, payload in payloads.items():
        text = _dump(payload)
        target = OUT / name
        if check_only:
            if name in CORPUS_BOUND and not gated:
                gated = True
                ok, corpus_note = _corpus_gate(payloads["documents.json"])
                if not ok:
                    skipped.extend(n for n in CORPUS_BOUND if n in payloads)
            if name in skipped:
                continue
            compared.append(name)
            if not target.exists():
                stale.append(f"{name}：文件不存在")
            elif target.read_text(encoding="utf-8") != text:
                stale.append(f"{name}：内容与接口实际输出不一致")
            continue
        target.write_text(text, encoding="utf-8")
        print(f"写入 {target.relative_to(ROOT)}  ({len(text):,} 字节)")

    if check_only:
        if stale:
            raise SystemExit(
                "::error::前端 fixture 与 /api/kb/* 的实际输出不一致：\n  - "
                + "\n  - ".join(stale)
                + "\n修法：python scripts/capture_kb_fixtures.py，然后看 git diff。"
                "\n不要手改 fixture —— 它存在的意义就是不被手改。"
                "\n（如果只是又入库了一篇论文，diff 里应该只有数字变化。）"
            )
        # 报"比对了哪些"，不报"抓到了哪些" —— 后者会让人以为全都校验过。
        if compared:
            print(f"已比对（逐字节）：{', '.join(sorted(compared))}")
        if skipped:
            print(
                f"未比对 {', '.join(sorted(skipped))}：{corpus_note}。\n"
                "  这两个文件的内容来自本机语料（ccus_docs/ 与 lightrag_storage/），"
                "而它们按设计不入仓库 ——\n"
                "  论文 PDF 有版权、图谱是 GB 级产物。逐字节比对只在有完整语料的"
                "开发机上有意义。\n"
                "  接口契约另由 tests.test_api_kb 覆盖，那个 job 不需要语料。"
            )
            # 跳过逐字节 ≠ 这个文件没人管：字段名与语料无关，仍然要查。
            drift = _shape_drift(payloads["documents.json"])
            if drift:
                raise SystemExit(
                    "::error::documents.json 的字段集合变了（这与语料无关，是真漂移）：\n  - "
                    + "\n  - ".join(drift)
                    + "\n修法：python scripts/capture_kb_fixtures.py，然后看 git diff。"
                )
            print("已比对（字段集合）：documents.json —— 语料不同，字段名仍须一致")
            if "overview.json" in skipped:
                # 单独说明：overview 的 graph 段是 `**graph` 展开的，图谱不可用时
                # 字段本来就少几个。硬查字段集合会造出一条假失败，而假失败和
                # 永远红是同一件事 —— 所以这里明说"没查"，不假装查了。
                print(
                    "未比对（字段集合）overview.json：它的 graph 段随图谱是否可用而变，"
                    "字段集合本身不固定；\n"
                    "  该端点的契约由 tests.test_api_kb.TestOverview 逐字段断言。"
                )
        if MODEL_BOUND not in compared:
            # 必须说出来，而且不能只是"未校验"三个字：要给出**为什么**。
            # 一个笼统的绿灯如果让人以为检索也校验了，它就是在替没做的检查背书。
            print(
                f"未比对 {MODEL_BOUND}：本机没有 Ollama，无法重建这份 fixture。"
                "CI 同样不跑它 —— 只有开发机重跑 capture 时才会更新。"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
