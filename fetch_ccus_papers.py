"""
CCUS 文献批量下载（arXiv）
==========================
从 arXiv 公开 API 检索并下载 CCUS / CO2 地质封存 / 断层完整性 / 多相流方向
的论文 PDF，统一重命名为 ``ccus_auto_NN.pdf`` 存入 ``ccus_docs/``。

为什么是 arXiv
--------------
arXiv 有公开 REST API（``export.arxiv.org/api/query``）、PDF 直链稳定、
不设反爬、允许程序化访问（只需遵守 3 秒/请求的礼貌间隔）。
相比抓取出版商页面，这条路不会因为改版或限流而半夜挂掉。

**筛选为什么必须自己做**
arXiv 的 ``all:`` 字段是全文检索，召回噪声极大。实测 ``all:"CO2 sequestration"``
会把"浮游生物固碳""牲畜碳排放""无人机气体探测"一起拉进来 —— 它们确实含
"CO2 sequestration" 这个词组，但和地质封存毫无关系。
所以本脚本做两层过滤：

  1. **地质语境硬门槛**：标题+摘要里必须出现至少一个地质语境词
     （geological / caprock / aquifer / reservoir / fault / saline ...），
     否则直接丢弃。这一条把生物固碳类论文全部挡掉。
  2. **相关性打分**：核心词（CO2 storage、CCUS、断层活化…）权重高于
     方法词（machine learning、surrogate…），按分数排序取前 N 篇。

下载是幂等的：已存在的文件跳过，manifest 记录每篇的来源与元数据，
重复运行不会重复占空间，也不会打乱编号。

用法
----
    python fetch_ccus_papers.py                # 默认下 18 篇
    python fetch_ccus_papers.py --target 20
    python fetch_ccus_papers.py --dry-run      # 只检索排序，不下载
    python fetch_ccus_papers.py --force        # 忽略已有文件重新下载
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

# ── 路径与常量 ────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(PROJECT_ROOT, "ccus_docs")
MANIFEST_PATH = os.path.join(DOCS_DIR, "manifest.json")

ARXIV_API = "http://export.arxiv.org/api/query"
ATOM = {"a": "http://www.w3.org/2005/Atom"}

# arXiv 的 API 使用条款要求可识别的 User-Agent，并限制 ~3 秒/请求。
USER_AGENT = "petro-rag-agent/1.0 (CCUS literature ingestion; contact: local-user)"
REQUEST_INTERVAL = 3.0
MIN_PDF_BYTES = 20_000          # 小于这个基本是错误页而不是论文

# ── 检索式 ────────────────────────────────────────────────
# 每条查询都刻意带上地质/储层语境，避免纯"碳捕集材料"或"生物固碳"混进来。
SEARCH_QUERIES = [
    'all:"CO2 geological storage"',
    'all:"geological carbon storage"',
    'all:"carbon capture and storage" AND all:geological',
    'all:"CO2 sequestration" AND all:geological',
    'all:"CO2 sequestration" AND all:reservoir',
    'all:"caprock integrity"',
    'all:"fault reactivation" AND all:CO2',
    'all:"saline aquifer" AND all:CO2',
    'all:"CO2 plume"',
    'all:"multiphase flow" AND all:"porous media" AND all:CO2',
    'cat:physics.geo-ph AND all:CO2',
    'all:"carbon storage" AND all:leakage',
]

# ── 相关性打分词表 ────────────────────────────────────────
# 地质语境词：命中任一才进入候选池（硬门槛）
GEOLOGICAL_CONTEXT = (
    "geological", "geologic", "caprock", "cap rock", "aquifer", "reservoir",
    "fault", "saline", "subsurface", "basalt", "shale", "sandstone",
    "carbonate", "sedimentary", "geomechanic", "poroelastic", "seal",
    "storage site", "storage complex", "overburden", "wellbore",
)

# 核心词：直接对应本项目关心的问题，权重 3
CORE_TERMS = {
    "co2 storage": 3.0,
    "carbon storage": 3.0,
    "carbon capture and storage": 3.0,
    "ccs": 2.5,
    "ccus": 3.0,
    "geological storage": 3.0,
    "sequestration": 2.5,
    "caprock": 3.0,
    "fault reactivation": 3.0,
    "fault activation": 3.0,
    "co2 plume": 2.5,
    "saline aquifer": 3.0,
    "leakage": 2.0,
    "breakthrough pressure": 3.0,
    "storage capacity": 2.0,
    "pressure management": 2.0,
    "wellbore integrity": 3.0,
}

# 方法/物理词：有加分但不是主题，权重 1
METHOD_TERMS = {
    "multiphase flow": 1.0,
    "porous media": 1.0,
    "permeability": 1.0,
    "fracture": 1.0,
    "two-phase": 1.0,
    "relative permeability": 1.5,
    "numerical simulation": 1.0,
    "reservoir simulation": 1.5,
    "machine learning": 1.0,
    "neural network": 1.0,
    "surrogate": 1.0,
    "bayesian": 1.0,
    "inversion": 1.0,
    "monitoring": 1.0,
    "seismic": 1.0,
    "geomechanics": 1.5,
    "hydraulic fracturing": 1.0,
    "co2 injection": 2.0,
    "injectivity": 1.5,
}

# 明确跑题的信号：出现在标题里直接丢弃（arXiv 的全文检索会带来这些）
OFF_TOPIC_TITLE = (
    "plankton", "livestock", "drone", "cattle", "soil organic",
    "indoor air", "vehicle", "forest fire", "ocean acidification",
    "electrochemical", "membrane separation", "metal-organic framework",
    "photocatalytic", "sorbent", "amine scrubbing",
)


def log(msg: str) -> None:
    print(msg, flush=True)


# ══════════════════════════════════════════════════════════
# 检索
# ══════════════════════════════════════════════════════════
def _fetch(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def search_arxiv(query: str, max_results: int = 25) -> list[dict]:
    """跑一条 arXiv 查询，返回规范化后的条目列表。"""
    url = (
        f"{ARXIV_API}?search_query={urllib.parse.quote(query)}"
        f"&start=0&max_results={max_results}"
        f"&sortBy=relevance&sortOrder=descending"
    )
    try:
        raw = _fetch(url)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        log(f"  [WARN] 查询失败（{type(exc).__name__}: {exc}），跳过：{query}")
        return []

    try:
        root = ET.fromstring(raw)
    except ET.ParseError as exc:
        log(f"  [WARN] 返回内容不是合法 XML（{exc}），跳过：{query}")
        return []

    out = []
    for entry in root.findall("a:entry", ATOM):
        def text(tag: str) -> str:
            node = entry.find(tag, ATOM)
            return " ".join((node.text or "").split()) if node is not None else ""

        entry_id = text("a:id")
        # id 形如 http://arxiv.org/abs/2606.27515v1
        arxiv_id = entry_id.rsplit("/abs/", 1)[-1] if "/abs/" in entry_id else entry_id
        if not arxiv_id:
            continue

        pdf_url = ""
        for link in entry.findall("a:link", ATOM):
            if link.get("title") == "pdf":
                pdf_url = link.get("href") or ""
                break
        if not pdf_url:
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"

        out.append({
            "arxiv_id": arxiv_id,
            # 去掉版本号做去重键：同一篇 v1/v2 不应占两个编号
            "dedupe_key": re.sub(r"v\d+$", "", arxiv_id),
            "title": text("a:title"),
            "abstract": text("a:summary"),
            "published": text("a:published")[:10],
            "updated": text("a:updated")[:10],
            "authors": [
                (a.find("a:name", ATOM).text or "").strip()
                for a in entry.findall("a:author", ATOM)
                if a.find("a:name", ATOM) is not None
            ],
            "abs_url": f"https://arxiv.org/abs/{arxiv_id}",
            "pdf_url": pdf_url,
            "matched_query": query,
        })
    return out


# ══════════════════════════════════════════════════════════
# 相关性筛选
# ══════════════════════════════════════════════════════════
def score_paper(paper: dict) -> tuple[float, list[str]]:
    """
    给一篇论文打相关性分。

    返回 (score, matched_terms)。score 为 0 表示不相关（调用方应丢弃）。
    """
    title = paper["title"].lower()
    blob = f"{title} {paper['abstract'].lower()}"

    # 跑题硬拦：标题里出现明确无关信号
    for bad in OFF_TOPIC_TITLE:
        if bad in title:
            return 0.0, [f"off-topic:{bad}"]

    # 地质语境硬门槛：摘要或标题里必须有一个
    if not any(term in blob for term in GEOLOGICAL_CONTEXT):
        return 0.0, []

    score = 0.0
    matched: list[str] = []
    for term, weight in CORE_TERMS.items():
        if term in blob:
            score += weight
            matched.append(term)
    for term, weight in METHOD_TERMS.items():
        if term in blob:
            score += weight
            matched.append(term)

    # 标题命中核心词额外加权：标题说自己是 CCUS 的，比摘要顺带提一句可信
    for term, weight in CORE_TERMS.items():
        if term in title:
            score += weight
            matched.append(f"title:{term}")

    return round(score, 2), matched


def select_papers(papers: list[dict], target: int) -> list[dict]:
    """去重 → 打分 → 排序 → 取前 target 篇。"""
    seen: dict[str, dict] = {}
    for p in papers:
        key = p["dedupe_key"]
        # 先打分再比较：同一篇论文可能被多条查询召回，而重复项也要参与
        # "保留分数更高那条记录"的判断，所以不能在分支里才打分。
        p["score"], p["matched"] = score_paper(p)
        if key not in seen or p["score"] > seen[key]["score"]:
            seen[key] = p

    relevant = [p for p in seen.values() if p["score"] > 0]
    # 同分时优先新论文：CCUS 领域方法演进快，2020 前的数值结论参考价值低
    relevant.sort(key=lambda p: (p["score"], p["published"]), reverse=True)
    log(f"  -> 去重后 {len(seen)} 篇，其中相关 {len(relevant)} 篇")
    return relevant[:target]


# ══════════════════════════════════════════════════════════
# 下载
# ══════════════════════════════════════════════════════════
def download_pdf(paper: dict, dest: str, retries: int = 3) -> bool:
    """下载单个 PDF，校验文件头与体积。失败重试。"""
    if os.path.exists(dest) and os.path.getsize(dest) >= MIN_PDF_BYTES:
        log(f"  [SKIP] 已存在 {os.path.basename(dest)}")
        return True

    for attempt in range(1, retries + 1):
        try:
            data = _fetch(paper["pdf_url"], timeout=120)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            log(f"  [WARN] 第 {attempt} 次下载失败（{type(exc).__name__}）：{paper['arxiv_id']}")
            time.sleep(REQUEST_INTERVAL * attempt)
            continue

        # 校验：必须是真的 PDF，不能是 HTML 错误页
        if not data.startswith(b"%PDF"):
            log(f"  [WARN] 返回内容不是 PDF（前 8 字节 {data[:8]!r}）：{paper['arxiv_id']}")
            time.sleep(REQUEST_INTERVAL)
            continue
        if len(data) < MIN_PDF_BYTES:
            log(f"  [WARN] 文件过小（{len(data)} B），疑似残缺：{paper['arxiv_id']}")
            time.sleep(REQUEST_INTERVAL)
            continue

        tmp = dest + ".part"
        with open(tmp, "wb") as fh:
            fh.write(data)
        os.replace(tmp, dest)       # 原子替换：中断不会留下半个 PDF
        log(f"  [OK]   {os.path.basename(dest)}  {len(data) / 1024:.0f} KB  {paper['title'][:60]}")
        return True

    return False


def load_manifest() -> dict:
    if os.path.exists(MANIFEST_PATH):
        try:
            with open(MANIFEST_PATH, encoding="utf-8") as fh:
                return json.load(fh)
        except (json.JSONDecodeError, OSError):
            log("[WARN] manifest.json 损坏，将重建。")
    return {"papers": []}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="从 arXiv 批量下载 CCUS 文献")
    ap.add_argument("--target", type=int, default=18, help="目标篇数（默认 18）")
    ap.add_argument("--per-query", type=int, default=25, help="每条查询取多少条候选")
    ap.add_argument("--dry-run", action="store_true", help="只检索排序，不下载")
    ap.add_argument("--force", action="store_true", help="忽略已有文件，重新下载")
    args = ap.parse_args(argv)

    os.makedirs(DOCS_DIR, exist_ok=True)
    log("=" * 62)
    log(f"CCUS 文献批量下载 · 目标 {args.target} 篇 · 源 = arXiv")
    log("=" * 62)

    # ① 检索
    log(f"\n[1/4] 检索 {len(SEARCH_QUERIES)} 条查询...")
    pool: list[dict] = []
    for i, q in enumerate(SEARCH_QUERIES, 1):
        got = search_arxiv(q, max_results=args.per_query)
        log(f"  ({i:2}/{len(SEARCH_QUERIES)}) {len(got):2} 条 ← {q[:64]}")
        pool.extend(got)
        if i < len(SEARCH_QUERIES):
            time.sleep(REQUEST_INTERVAL)

    if not pool:
        log("\n[FAIL] 一条结果都没拿到。检查网络或代理设置（arXiv 需要外网）。")
        return 1

    # ② 筛选
    log(f"\n[2/4] 相关性筛选（{len(pool)} 条候选）...")
    selected = select_papers(pool, args.target)
    if not selected:
        log("\n[FAIL] 没有通过地质语境门槛的论文。")
        return 1

    log("\n  入选论文：")
    for i, p in enumerate(selected, 1):
        log(f"   {i:2}. [{p['score']:5.1f}] {p['published']}  {p['title'][:70]}")

    if args.dry_run:
        log("\n[dry-run] 不下载。")
        return 0

    # ③ 下载
    log(f"\n[3/4] 下载 PDF 到 {os.path.relpath(DOCS_DIR, PROJECT_ROOT)}/ ...")
    manifest = load_manifest()
    existing = {p["dedupe_key"]: p for p in manifest.get("papers", [])}

    ok_count = 0
    records = []
    for i, p in enumerate(selected, 1):
        filename = f"ccus_auto_{i:02d}.pdf"
        dest = os.path.join(DOCS_DIR, filename)

        # 编号复用：同一篇论文重跑时保持同一个文件名，避免重复占用
        prev = existing.get(p["dedupe_key"])
        if prev and not args.force and os.path.exists(os.path.join(DOCS_DIR, prev["file"])):
            filename = prev["file"]
            dest = os.path.join(DOCS_DIR, filename)

        if args.force and os.path.exists(dest):
            os.remove(dest)

        if download_pdf(p, dest):
            ok_count += 1
            records.append({
                "file": filename,
                "arxiv_id": p["arxiv_id"],
                "dedupe_key": p["dedupe_key"],
                "title": p["title"],
                "authors": p["authors"][:6],
                "published": p["published"],
                "abs_url": p["abs_url"],
                "pdf_url": p["pdf_url"],
                "score": p["score"],
                "matched_terms": p["matched"][:12],
                "abstract": p["abstract"],
                "bytes": os.path.getsize(dest) if os.path.exists(dest) else 0,
            })
        time.sleep(REQUEST_INTERVAL)

    # ④ 写 manifest
    log(f"\n[4/4] 写入 manifest.json ...")
    # 合并旧记录：以前下过但本次没选的论文不该被 manifest 抹掉
    merged = {p["dedupe_key"]: p for p in manifest.get("papers", [])}
    for r in records:
        merged[r["dedupe_key"]] = r
    all_papers = sorted(merged.values(), key=lambda r: r["file"])

    with open(MANIFEST_PATH, "w", encoding="utf-8") as fh:
        json.dump({
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source": "arXiv",
            "api": ARXIV_API,
            "total": len(all_papers),
            "papers": all_papers,
        }, fh, ensure_ascii=False, indent=2)

    total_bytes = sum(p.get("bytes", 0) for p in all_papers)
    log(f"\n{'=' * 62}")
    log(f"完成：本次成功 {ok_count} 篇，manifest 共 {len(all_papers)} 篇，"
        f"{total_bytes / 1024 / 1024:.1f} MB")
    log(f"{'=' * 62}")
    return 0 if ok_count else 1


if __name__ == "__main__":
    sys.exit(main())
