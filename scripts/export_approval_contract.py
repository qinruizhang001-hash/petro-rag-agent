"""把「审批状态机」的契约与演示草稿导出成前端 fixture。

为什么要导出，而不是在前端写一张状态表
--------------------------------------
`draft → pending_review → approved / rejected` 这套规则在
`ops/reports.py` 里已经有唯一一份实现。前端如果要决定"哪个按钮能点"，
最省事的写法是在 TS 里再抄一张表 —— 而抄的那一刻起就有了两份真值。
之后后端加一条迁移（比如允许 `approved → draft` 做更正），前端的表不会跟着变，
表现是：**界面上那个按钮点不动，或者点下去 409**。两种情况都不报错，
只在真实审批时被发现。

所以这里的做法是：**在四个状态上逐个动作真跑一遍状态机**，把"从哪个状态、
执行哪个动作、落到哪个状态"记下来。导出的不是"我以为的规则"，是"它实际的行为"。
`--check` 在 CI 里跑，`ops/reports.py` 一改而 fixture 没重新生成就会红。

草稿种子同样不是手写的
----------------------
三份种子都经真实端点生成：`/api/ops/run` → `/api/ops/reports/{id}`。
报告正文里的每个数字都带引用标记，是 `ops/reports.py` 的模板拼出来的；
手写一份"看起来像报告"的正文，等于让界面演示一份并不存在的产物。

第三份种子**故意弄坏**：追加一行没有引用的数字，让 `problems` 非空。
它演示的是这个系统最要紧的一条守卫 —— 引用检查不过就拒绝提交。
这份稿子在 fixture 里带 `seed_note` 说明它是刻意造出来的坏样本，
不会让读者以为真有这么一份稿子。

用法::

    python scripts/export_approval_contract.py            # 写入 web/src/fixtures/
    python scripts/export_approval_contract.py --check    # 只比对不写（给 CI 用）
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

OUT = ROOT / "web" / "src" / "fixtures"

#: 三个动作对应的"合法入参"。探针要的是**能走通**的那一次调用，
#: 所以这里给的是合法值 —— 缺审批人、缺意见这些守卫另由 `guards` 记录。
_ACTIONS: dict[str, dict] = {
    "submit": {"actor": "早班"},
    "approve": {"reviewer": "工艺主管", "note": ""},
    "reject": {"reviewer": "工艺主管", "note": "第三节的处置时限需要补依据"},
    "reopen": {"actor": "早班"},
}

#: 状态 → 造一份处于该状态的草稿所需的动作序列。顺序即真实流转顺序。
_PATH_TO: dict[str, tuple[str, ...]] = {
    "draft": (),
    "pending_review": ("submit",),
    "approved": ("submit", "approve"),
    "rejected": ("submit", "reject"),
}


def _apply(draft, action: str) -> None:
    from ops import reports as ops_reports

    kwargs = _ACTIONS[action]
    getattr(ops_reports, action)(draft, **kwargs)


def _blank_draft(status: str):
    """造一份处于指定状态的草稿。用真实模板，不用假正文。"""
    from ops import reports as ops_reports
    from ops.alarms import evaluate_rows
    from ops.thresholds import load_registry

    readings = _readings()
    registry = load_registry(_thresholds_path())
    report = evaluate_rows(readings, registry)
    start, end = _period(readings)
    draft = ops_reports.build_daily(report, registry, period=(start, end))
    for action in _PATH_TO[status]:
        _apply(draft, action)
    return draft


def _period(readings: list[dict]) -> tuple[datetime, datetime]:
    """读数的时间范围。`build_readings` 给的是 ISO 字符串，报告模板要 datetime。"""
    stamps = [datetime.fromisoformat(str(r["ts"])) for r in readings]
    return min(stamps), max(stamps)


def _readings() -> list[dict]:
    """复用抓 `/api/ops/*` fixture 时那批读数 —— 两份 fixture 必须同源。

    各造一批读数的话，剖面、告警、报告会描述三个不同的现场，
    而界面上没有任何地方能看出它们不是一回事。
    """
    import capture_ops_fixtures as cap

    return cap.build_readings()


def _thresholds_path() -> str:
    """抓 fixture 时用的是临时阈值文件（示例表删去三个盲测点）。

    这里必须造出**同一份**，否则报告正文里的阈值出处与
    `thresholds.json` 里的对不上，而两者都会显示在界面上。
    """
    import capture_ops_fixtures as cap

    if not hasattr(_thresholds_path, "_cache"):
        tmp = Path(tempfile.mkdtemp(prefix="approval-contract-")) / "ops_thresholds.json"
        tmp.write_text(json.dumps(cap.build_config(), ensure_ascii=False), encoding="utf-8")
        os.environ["OPS_THRESHOLDS_PATH"] = str(tmp)
        _thresholds_path._cache = str(tmp)  # type: ignore[attr-defined]
    return _thresholds_path._cache  # type: ignore[attr-defined]


def probe_edges() -> list[dict]:
    """在四个状态上逐个动作真跑，记录走得通的那些。"""
    edges: list[dict] = []
    for status in _PATH_TO:
        for action in _ACTIONS:
            draft = _blank_draft(status)
            try:
                _apply(draft, action)
            except ValueError:
                continue
            edges.append({"from": status, "action": action, "to": draft.status})
    return edges


def probe_guards() -> list[dict]:
    """记录那些**不看状态也要挡**的规则：缺署名、缺意见。

    这些不是状态迁移，是入参校验。前端据此决定"输入框空着时按钮为什么不能点"，
    而真正的拒绝仍由后端做 —— 前端只负责让人看懂原因。
    """
    from ops import reports as ops_reports

    out: list[dict] = []
    # 驳回必须写意见
    d = _blank_draft("pending_review")
    try:
        ops_reports.reject(d, reviewer="工艺主管", note="  ")
    except ValueError as exc:
        out.append({"action": "reject", "field": "note", "message": str(exc)})
    # 驳回同样必须署名。这条容易漏 —— `reject()` 里先查署名再查意见，
    # 只探"空意见"会以为驳回只要意见，于是界面上那个姓名框不出现，
    # 而提交时被后端挡下。
    d = _blank_draft("pending_review")
    try:
        ops_reports.reject(d, reviewer="  ", note="有意见")
    except ValueError as exc:
        out.append({"action": "reject", "field": "reviewer", "message": str(exc)})
    # 批准必须署名
    d = _blank_draft("pending_review")
    try:
        ops_reports.approve(d, reviewer=" ")
    except ValueError as exc:
        out.append({"action": "approve", "field": "reviewer", "message": str(exc)})
    # 引用检查不过不许提交
    d = _blank_draft("draft")
    d.body += "\n<!-- strict -->\n- 环空压力 999.9 bar\n<!-- /strict -->\n"
    d.problems = ops_reports.check_citations(d.body, d.citations)
    try:
        ops_reports.submit(d, actor="早班")
    except ValueError as exc:
        out.append({"action": "submit", "field": "problems", "message": str(exc)})
    return out


#: 抓取时临时目录的绝对路径会渲染进报告正文（`阈值注册表来源：<path>`）。
#: 原样落盘既让 fixture 无法逐字节比对，也会在界面上露出一行
#: `C:\Users\...\Temp\...` —— 读者第一反应是系统出错了。
_ORIGIN = "config/ops_thresholds.example.json（派生：删去 3 个浅层 MVA 测点）"


def _scrub(text: str) -> str:
    path = _thresholds_path()
    return text.replace(path, _ORIGIN)


def _seed_drafts() -> list[dict]:
    """三份种子草稿，全部经真实端点产出。"""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    import api_ops
    from ops import store as ops_store

    app = FastAPI()
    app.include_router(api_ops.router)
    client = TestClient(app)

    readings = _readings()
    start, end = _period(readings)
    period = {"start": start.isoformat(), "end": end.isoformat()}

    specs = [
        ("daily", ""),
        ("regulatory", "SCS-2026-0417"),
    ]
    seeds: list[dict] = []
    for kind, permit in specs:
        r = client.post(
            "/api/ops/run",
            json={
                "readings": readings,
                "period": period,
                "want_report": True,
                "report_kind": kind,
                "permit_id": permit,
            },
        )
        r.raise_for_status()
        generated = r.json()["draft_id"]
        assert generated, f"{kind} 没有产出草稿"

        # 换成确定性草稿号：`new_draft_id` 带 `now()` 的时间戳，
        # 每次跑都不一样，fixture 就没法逐字节比对了。
        draft = ops_store.load(generated)
        assert draft is not None
        stable = f"{draft.kind}-{draft.period[1]:%Y%m%dT%H%M%S}-seed"
        ops_store.save(stable, draft)

        detail = client.get(f"/api/ops/reports/{stable}")
        detail.raise_for_status()
        seeds.append(detail.json())

    # 第三份：刻意弄坏的样本，用来演示「引用检查不过就拒绝提交」。
    bad_id = seeds[0]["draft_id"] + "-badsample"
    draft = ops_store.load(seeds[0]["draft_id"])
    assert draft is not None
    draft.body += (
        "\n<!-- strict -->\n"
        "- 环空压力峰值 999.9 bar\n"
        "<!-- /strict -->\n"
    )
    from ops import reports as ops_reports

    draft.problems = ops_reports.check_citations(draft.body, draft.citations)
    ops_store.save(bad_id, draft)
    bad = client.get(f"/api/ops/reports/{bad_id}")
    bad.raise_for_status()
    payload = bad.json()
    payload["seed_note"] = (
        "这份稿子是刻意造出来的坏样本：正文里追加了一行没有引用标记的数字，"
        "用来演示「引用检查不通过 → 拒绝提交」。它不是任何一次真实运行的产物。"
    )
    seeds.append(payload)

    for s in seeds:
        s["body"] = _scrub(s["body"])
        s["draft_id"] = s["draft_id"]
    return seeds


def build() -> dict[str, object]:
    from ops import store as ops_store

    # 审批库要指到临时目录：默认是 cwd 下的 `ops_reports.db`，
    # 跑一次脚本就往仓库里扔一个 sqlite 文件。
    ops_store.OPS_DB = str(Path(tempfile.mkdtemp(prefix="approval-db-")) / "ops_reports.db")

    from ops import reports as ops_reports

    contract = {
        "generated_by": "scripts/export_approval_contract.py",
        "source": "ops/reports.py（TRANSITIONS / submit / approve / reject / reopen）",
        "statuses": list(ops_reports.STATUS_LABELS),
        "labels": dict(ops_reports.STATUS_LABELS),
        "edges": probe_edges(),
        "guards": probe_guards(),
    }
    return {
        "approval.json": contract,
        "reports.json": {"drafts": _seed_drafts()},
    }


def _dump(payload) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


def _first_diff(old: str, new: str) -> str:
    old_lines = old.split("\n")
    new_lines = new.split("\n")
    for i in range(max(len(old_lines), len(new_lines))):
        a = old_lines[i] if i < len(old_lines) else "<缺行>"
        b = new_lines[i] if i < len(new_lines) else "<缺行>"
        if a != b:
            return f"    第 {i + 1} 行\n    磁盘：{a[:120]}\n    现在：{b[:120]}"
    return "    （内容相同？）"


def main(argv: list[str]) -> int:
    check_only = "--check" in argv[1:]
    payloads = build()

    OUT.mkdir(parents=True, exist_ok=True)
    stale: list[str] = []
    for name, payload in payloads.items():
        text = _dump(payload)
        target = OUT / name
        if check_only:
            if not target.exists():
                stale.append(f"{name}：文件不存在")
            else:
                on_disk = target.read_text(encoding="utf-8")
                if on_disk != text:
                    stale.append(
                        f"{name}：与 ops/reports.py 的实际行为不一致\n"
                        + _first_diff(on_disk, text)
                    )
            continue
        target.write_text(text, encoding="utf-8")
        print(f"写入 {target.relative_to(ROOT)}  ({len(text):,} 字节)")

    if check_only:
        if stale:
            raise SystemExit(
                "::error::审批契约 fixture 与后端实际行为不一致：\n  - "
                + "\n  - ".join(stale)
                + "\n修法：python scripts/export_approval_contract.py，然后看 git diff。"
                "\n这张表是探针跑出来的，不要手改。"
            )
        print(f"审批契约一致：{', '.join(sorted(payloads))}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
