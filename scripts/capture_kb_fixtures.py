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
  Ollama → 逐字节比对只在开发机上成立。**但这不等于"CI 不校验"**：证据包的
  一大半是**纯函数**（来源表由切片按首次出现顺序去重得来、`stats` 是卡片计数、
  `text_quality` 由正文算出），所以 CI 里跑第二层 —— **结构与自洽**
  （`kb_evidence.audit_evidence`）。手改一个 `reference_id`、删一条切片而忘了
  改 `stats`、把 `low` 改成 `false`，界面都照样渲染，只是每处引用都指错 ——
  这一层查的就是这个。

**两层都要打印。** 只说"未比对"会抹掉真做过的检查，只说"已比对"会替没做的
检查背书。`--check` 对 `search.json` 会连着输出两行：一行说逐字节没比、
为什么；一行说结构与自洽比了、比了什么、以及**这一层查不到什么**。

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

`--check` 会逐文件打印它走到了哪一层（逐字节 / 字段集合 / 结构与自洽 /
未比对），不打印一个笼统的"一致"。
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

    if check_only:
        for line in _check_payloads(payloads):
            print(line)
        return 0

    # 写之前先审一遍：一份自相矛盾的证据包不该被写进仓库，因为写进去之后
    # 它就成了「前端唯一的数据来源」，而它自己跟自己都对不上。
    problems: list[str] = []
    if MODEL_BOUND in payloads:
        problems = _audit_search(
            payloads[MODEL_BOUND], _corpus_names(payloads.get("documents.json"))
        )
    if problems:
        raise SystemExit(
            "::error::刚抓到的检索证据包自己跟自己不一致（没有落盘）：\n  - "
            + "\n  - ".join(problems)
            + "\n这说明 kb_evidence 的构建规则和审计规则分家了 —— 先看 kb_evidence.py，"
            "\n不要在 capture 脚本里绕过检查。"
        )

    OUT.mkdir(parents=True, exist_ok=True)
    for name, payload in payloads.items():
        text = _dump(payload)
        target = OUT / name
        target.write_text(text, encoding="utf-8")
        print(f"写入 {target.relative_to(ROOT)}  ({len(text):,} 字节)")
    return 0


# ══════════════════════════════════════════════════════════
# search.json 的第二层：结构与自洽（不需要模型）
# ══════════════════════════════════════════════════════════

#: ``/api/kb/search`` 响应的顶层字段。多一个少一个都要报 —— 这是契约，
#: 不是实现细节，前端 `SearchResponse` 直接按它取。
SEARCH_TOP_KEYS = frozenset(
    {"status", "query", "mode", "reason", "evidence", "elapsed_ms"}
)

#: ``status`` 的三个取值。**不是两个**：``empty`` 是关于语料的事实，
#: ``unavailable`` 是系统状态，两者的处置完全相反。
SEARCH_STATUSES = frozenset({"ok", "empty", "unavailable"})


def _corpus_names(documents_payload: dict | None) -> set[str]:
    """从 ``documents`` 响应里取文件名集合。取不到就给空集（跳过跨文件检查）。"""
    if not isinstance(documents_payload, dict):
        return set()
    return {
        str(r.get("name"))
        for r in documents_payload.get("documents") or []
        if isinstance(r, dict) and r.get("name")
    }


def _audit_search(payload: dict, corpus: set[str]) -> list[str]:
    """``search.json`` 的**结构与自洽**检查（不需要 Ollama）。

    分工要说清楚：

    * 这里查的 —— 这份文件**自己跟自己**一致吗（字段集合、问句指纹、来源表
      重算、计数、质量分、切片出处）。全都能从文件本身推出来。
    * 这里查不到的 —— 这份证据**是不是当前后端会返回的那一份**。那需要重跑
      检索，只有开发机做得到。所以调用方必须同时打印"逐字节未比对"。

    把两者混为一谈的代价是不对称的：少说一句"未比对"，绿灯就替没做的检查
    背了书；少说一句"已比对"，真做过的检查就被抹掉了 —— 而后者会让人以为
    这一层没人管，于是下一次改动时它被删掉也没人心疼。
    """
    from kb_evidence import audit_evidence

    problems: list[str] = []
    if not isinstance(payload, dict):
        return [f"顶层应是对象，实际是 {type(payload).__name__}"]

    missing = SEARCH_TOP_KEYS - set(payload)
    extra = set(payload) - SEARCH_TOP_KEYS
    if missing:
        problems.append("顶层缺少字段：" + "、".join(sorted(missing)))
    if extra:
        problems.append("顶层多出字段：" + "、".join(sorted(extra)))

    if payload.get("query") != SEARCH_QUERY:
        # 问句变了，引用角标就指向另一个问题的检索结果 —— 界面照样渲染，
        # 只是"这个数字出自哪一片"从此对不上。
        problems.append(
            f"query 是 {payload.get('query')!r}，不是抓取时用的 {SEARCH_QUERY!r} —— "
            "换了问句，引用角标就指向另一次检索"
        )
    if payload.get("mode") != "hybrid":
        problems.append(f"mode 是 {payload.get('mode')!r}，抓取时用的是 'hybrid'")
    if payload.get("elapsed_ms") != 0:
        # 这个值每次跑都不同，脚本会归零。非零说明这份文件不是 capture 写的。
        problems.append(
            f"elapsed_ms={payload.get('elapsed_ms')!r} 没有归零 —— "
            "说明这份文件不是 capture 脚本产出的（它每次跑都不同，脚本会抹掉）"
        )

    status = payload.get("status")
    reason = str(payload.get("reason") or "")
    if status not in SEARCH_STATUSES:
        problems.append(f"status={status!r} 不在 {sorted(SEARCH_STATUSES)} 里")
    if status == "ok" and reason:
        problems.append("status=ok 却带了 reason —— ok 的定义是「检索到内容且没有异常」")
    if status in ("empty", "unavailable") and not reason:
        problems.append(
            f"status={status} 必须说明原因 —— 空与不可用是两种完全相反的处置"
        )

    evidence = payload.get("evidence")
    if not isinstance(evidence, dict):
        return problems + [f"evidence 应是对象，实际是 {type(evidence).__name__}"]

    has_content = bool(
        evidence.get("chunks")
        or evidence.get("entities")
        or evidence.get("relationships")
    )
    if status == "ok" and not has_content:
        problems.append("status=ok 但一条证据都没有 —— ok 与 empty 的边界就是「有没有内容」")
    if status in ("empty", "unavailable") and has_content:
        problems.append(f"status={status} 却带着证据 —— 没有内容才叫 empty/unavailable")

    problems.extend(audit_evidence(evidence))

    # 跨文件：切片的出处必须在这份 fixture 的语料清单里。
    # 两个都是提交进仓库的文件，所以这一条在 CI 里成立。
    if corpus:
        orphan = sorted(
            {
                str(c.get("source") or "")
                for c in evidence.get("chunks") or []
                if isinstance(c, dict) and c.get("source")
            }
            - corpus
        )
        if orphan:
            problems.append(
                "切片来源不在 documents.json 的语料清单里："
                + "、".join(orphan)
                + " —— 界面上点开这个出处会 404"
            )
    return problems


def _search_layer(byte_compared: list[str]) -> list[str]:
    """``search.json`` 走到了哪一层。两层都要交代，一行都不能省。

    逐字节比对过时**不再重复一遍**：字节相等本身就蕴含了内部自洽，而
    ``_check_payloads`` 已经在上面的「已比对（逐字节）」里点过名了。
    这里要补的是另一条分支 —— 逐字节做不到的时候，必须紧接着说清
    "还查了什么"，否则读起来就是"这个文件没人管"。
    """
    target = OUT / MODEL_BOUND
    if MODEL_BOUND in byte_compared:
        return []

    lines = [
        f"未比对（逐字节）{MODEL_BOUND}：本机没有 Ollama，无法重建这份 fixture。"
        "CI 同样不跑它 —— 只有开发机重跑 capture 时才会更新。"
    ]
    if not target.exists():
        lines.append(f"未比对（结构与自洽）{MODEL_BOUND}：文件不存在，无从检查")
        return lines
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SystemExit(
            f"::error::{MODEL_BOUND} 读不出来：{type(exc).__name__}: {exc}"
        ) from exc

    problems = _audit_search(payload, _fixture_corpus() or set())
    if problems:
        raise SystemExit(
            f"::error::{MODEL_BOUND} 自己跟自己不一致（不需要 Ollama 就能查出来）：\n  - "
            + "\n  - ".join(problems)
            + "\n这一层查的是「文件内部自洽」，不是「内容是否过期」—— 所以能在这里红，"
            "\n说明它被手改过，或者 kb_evidence 的构建规则变了而 fixture 没重抓。"
            "\n修法：在有 Ollama 的机器上 python scripts/capture_kb_fixtures.py。"
        )

    evidence = payload.get("evidence") or {}
    lines.append(
        f"已比对（结构与自洽）{MODEL_BOUND}："
        f"{len(evidence.get('chunks') or [])} 条切片 / "
        f"{len(evidence.get('entities') or [])} 个实体 / "
        f"{len(evidence.get('relationships') or [])} 条关系 / "
        f"{len(evidence.get('references') or [])} 个来源；"
        "查了顶层字段、问句指纹、来源表重算、stats 逐项、质量分、切片出处。"
    )
    lines.append(
        "  这一层查不到的是「内容是不是当前后端会返回的那一份」—— 那需要重跑检索。"
        "所以上面那行「未比对（逐字节）」不能省。"
    )
    return lines


def _check_payloads(payloads: dict[str, object]) -> list[str]:
    """比对已抓到的 payload 与仓库里的 fixture，返回要打印的行。

    为什么返回字符串而不是直接 ``print``
    -----------------------------------
    这里的每一行都是"我到底查了哪一层"的交代，而**交代本身值得被测**：
    CI 跑的是"没有 Ollama"那条分支，如果那段说明被改坏（少一句「未比对逐字节」，
    或者少一句「已比对结构与自洽」），本机开发时一切照旧，只有 reviewer 会看到
    一个替没做的检查背书的绿灯。所以判定逻辑要能脱离 Ollama 被调用 ——
    ``tests/test_api_kb.py`` 直接喂假的 payload 进来验这两句话。
    """
    stale: list[str] = []
    compared: list[str] = []
    skipped: list[str] = []
    corpus_note = ""
    gated = False

    for name, payload in payloads.items():
        text = _dump(payload)
        target = OUT / name
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

    if stale:
        raise SystemExit(
            "::error::前端 fixture 与 /api/kb/* 的实际输出不一致：\n  - "
            + "\n  - ".join(stale)
            + "\n修法：python scripts/capture_kb_fixtures.py，然后看 git diff。"
            "\n不要手改 fixture —— 它存在的意义就是不被手改。"
            "\n（如果只是又入库了一篇论文，diff 里应该只有数字变化。）"
        )

    lines: list[str] = []
    # 报"比对了哪些"，不报"抓到了哪些" —— 后者会让人以为全都校验过。
    if compared:
        lines.append(f"已比对（逐字节）：{', '.join(sorted(compared))}")
    if skipped:
        lines.append(
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
        lines.append("已比对（字段集合）：documents.json —— 语料不同，字段名仍须一致")
        if "overview.json" in skipped:
            # 单独说明：overview 的 graph 段是 `**graph` 展开的，图谱不可用时
            # 字段本来就少几个。硬查字段集合会造出一条假失败，而假失败和
            # 永远红是同一件事 —— 所以这里明说"没查"，不假装查了。
            lines.append(
                "未比对（字段集合）overview.json：它的 graph 段随图谱是否可用而变，"
                "字段集合本身不固定；\n"
                "  该端点的契约由 tests.test_api_kb.TestOverview 逐字段断言。"
            )

    lines.extend(_search_layer(compared))
    return lines


if __name__ == "__main__":
    sys.exit(main(sys.argv))
