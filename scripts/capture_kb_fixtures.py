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
  `chroma.sqlite3`），不碰模型、不碰网络 → 结果确定，**CI 每次都能校验**。
* `search`：要跑 LightRAG + Ollama 嵌入 + 图检索，本机约 21 秒，CI 里没有
  Ollama → 只在开发机上抓，**CI 不校验**。

这个差别必须打印出来。一个"fixture 一致"的绿灯如果连 search 也包含了，
而 search 其实从没被比对过，那这个绿灯就是在替没做的检查背书。

用法::

    python scripts/capture_kb_fixtures.py            # 写入 web/src/fixtures/
    python scripts/capture_kb_fixtures.py --check    # 只比对（CI 用，不跑检索）

`--check` 只比对可确定的两个文件，并在最后明确写出"search.json 未校验"。
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


def main(argv: list[str]) -> int:
    check_only = "--check" in argv[1:]
    payloads, note = capture()
    if note:
        print(note)

    OUT.mkdir(parents=True, exist_ok=True)
    stale: list[str] = []
    compared: list[str] = []
    for name, payload in payloads.items():
        text = _dump(payload)
        target = OUT / name
        if check_only:
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
        print(f"已比对：{', '.join(sorted(compared))}")
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
