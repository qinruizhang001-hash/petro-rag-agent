"""报告草稿与审批轨迹的持久化。

为什么必须落盘
--------------
``ReportDraft`` 是个 dataclass，活在进程内存里。而审批是个**跨请求、跨班次、
可能跨天**的流程：早班起草 → 提交 → 工艺主管下午审批 → 安环部隔天复核。
如果草稿只在内存里，任何一次进程重启都会让"待审批"的那份稿子消失，
而审批人看到的是 404 —— 他会以为稿子被撤回了。

更要紧的是 ``trail``（审批轨迹）。它是"谁在什么时候基于什么理由批准了什么"
的唯一记录。丢了轨迹的批准，等于没有批准。

为什么用 sqlite 而不是内存字典 + 定期写文件
-------------------------------------------
内存字典在多 worker 下各自一份（虽然本项目强制单 worker，见
``core/checkpoint.py``），而且崩溃就没了。sqlite 自带事务，一次
``INSERT OR REPLACE`` 就是一次原子提交，不需要自己写"先写临时文件再 rename"。

为什么单独一个库文件，不复用 checkpoint.db
------------------------------------------
checkpoint 是**可重建的运行状态**（丢了重跑一次就有），而报告草稿是
**业务记录**（丢了就是丢了）。两者备份策略、保留期限、清理时机完全不同，
混在一个库里会让"清 checkpoint"顺手清掉审计记录。

为什么序列化成 JSON 而不是给每个字段建列
----------------------------------------
``ReportDraft`` 的结构会随业务演进（比如将来加 ``reviewer_dept``）。
JSON 让加字段不需要迁移；而真正需要按列查的只有 ``id`` / ``status`` /
``kind`` / ``period_start`` 四个，它们单独建列并有索引。

诚实的边界：本模块只保证**单进程内的原子性**。多进程并发写同一个库文件时，
sqlite 靠文件锁串行化，写冲突会抛 ``OperationalError: database is locked``
（``timeout`` 内会重试）。本项目的部署形态是单 worker，不构成问题；
若将来改成多 worker，这个模块需要换成真正的数据库，而不是调大 timeout。
"""
from __future__ import annotations

import json
import os
import sqlite3
import uuid
from datetime import datetime
from typing import Any

from ops.contracts import Citation
from ops.reports import CitationProblem, ReportDraft

#: 与 ``CHECKPOINT_DB`` 同样的思路：默认值保持不变，换路径是运维决定。
#: 容器里应指到挂载目录（例如 ``/app/state/ops_reports.db``），
#: 否则容器重建即丢失全部审批记录。
OPS_DB = os.getenv("OPS_DB", "ops_reports.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS ops_reports (
    id            TEXT PRIMARY KEY,
    kind          TEXT NOT NULL,
    title         TEXT NOT NULL,
    status        TEXT NOT NULL,
    period_start  TEXT NOT NULL,
    period_end    TEXT NOT NULL,
    payload       TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_ops_reports_status ON ops_reports(status);
CREATE INDEX IF NOT EXISTS idx_ops_reports_created ON ops_reports(created_at DESC);
"""


def _connect(path: str | None = None) -> sqlite3.Connection:
    """打开库并建表。

    ``check_same_thread=False`` 是必须的：FastAPI 的同步端点跑在线程池里，
    同一个连接会被不同线程用到。
    """
    conn = sqlite3.connect(path or OPS_DB, check_same_thread=False, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    # 审批记录是账本，不是可重建状态 —— 这里和 checkpoint 相反，
    # 用 FULL：丢一条批准记录的代价远大于多等一次刷盘。
    conn.execute("PRAGMA synchronous=FULL")
    conn.executescript(_SCHEMA)
    return conn


def new_draft_id(kind: str) -> str:
    """生成可读的草稿号：``daily-20261027T092014-a3f9``。

    刻意不用纯 uuid：审批人需要在电话里念出这个号，``a3f9`` 念得出来，
    ``550e8400-e29b-41d4-a716-446655440000`` 念不出来。
    """
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    return f"{kind}-{stamp}-{uuid.uuid4().hex[:4]}"


def to_payload(draft: ReportDraft) -> dict[str, Any]:
    """把草稿拍成可 JSON 化的字典。

    ``citations`` 的键是 ``cite.render()``（形如 ``[alarm:A-...]``）而不是
    ``ref``：同一份报告里两处引用同一条告警是正常的，键必须是"渲染后的
    标记"才能区分不同位置，同时保留 ``ref`` 供跳转。
    """
    return {
        "kind": draft.kind,
        "title": draft.title,
        "period": [draft.period[0].isoformat(), draft.period[1].isoformat()],
        "body": draft.body,
        "status": draft.status,
        "permit_id": draft.permit_id,
        "blind_spots": list(draft.blind_spots),
        "trail": [dict(t) for t in draft.trail],
        "citations": {
            key: {"kind": c.kind, "ref": c.ref, "detail": c.detail}
            for key, c in draft.citations.items()
        },
        "problems": [
            {"kind": p.kind, "token": p.token, "line": p.line, "context": p.context}
            for p in draft.problems
        ],
    }


def from_payload(payload: dict[str, Any]) -> ReportDraft:
    """把字典还原成草稿。

    走 ``Citation(...)`` 的构造函数而不是直接塞字典：引用类型的校验
    （``kind`` 三态、``summary`` 必须带算法说明）在 ``__post_init__`` 里，
    绕过它等于让一份从库里读出来的旧数据跳过全部约束。
    """
    start, end = payload["period"]
    draft = ReportDraft(
        kind=payload["kind"],
        title=payload["title"],
        period=(datetime.fromisoformat(start), datetime.fromisoformat(end)),
        body=payload["body"],
        status=payload["status"],
        permit_id=payload.get("permit_id", ""),
        blind_spots=list(payload.get("blind_spots") or []),
        trail=[dict(t) for t in payload.get("trail") or []],
    )
    for key, c in (payload.get("citations") or {}).items():
        draft.citations[key] = Citation(c["kind"], c["ref"], c.get("detail", ""))
    draft.problems = [
        CitationProblem(p["kind"], p["token"], int(p["line"]), p["context"])
        for p in payload.get("problems") or []
    ]
    return draft


def save(draft_id: str, draft: ReportDraft, *, path: str | None = None) -> str:
    """写入或覆盖一份草稿，返回草稿号。"""
    now = datetime.now().isoformat(timespec="seconds")
    payload = json.dumps(to_payload(draft), ensure_ascii=False)
    conn = _connect(path)
    try:
        with conn:
            row = conn.execute(
                "SELECT created_at FROM ops_reports WHERE id = ?", (draft_id,)
            ).fetchone()
            created = row[0] if row else now
            conn.execute(
                "INSERT OR REPLACE INTO ops_reports"
                " (id, kind, title, status, period_start, period_end, payload, created_at, updated_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    draft_id,
                    draft.kind,
                    draft.title,
                    draft.status,
                    draft.period[0].isoformat(),
                    draft.period[1].isoformat(),
                    payload,
                    created,
                    now,
                ),
            )
    finally:
        conn.close()
    return draft_id


def load(draft_id: str, *, path: str | None = None) -> ReportDraft | None:
    """按草稿号读取。找不到返回 ``None``，由调用方决定是 404 还是别的。"""
    conn = _connect(path)
    try:
        row = conn.execute(
            "SELECT payload FROM ops_reports WHERE id = ?", (draft_id,)
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    return from_payload(json.loads(row[0]))


def list_recent(limit: int = 20, *, path: str | None = None) -> list[dict[str, Any]]:
    """最近的草稿摘要（不含正文，正文可能几十 KB）。

    默认按创建时间倒序。``limit`` 在这里夹到 [1, 200]：
    一个运维看板要"全部历史"通常是误操作，而 ``limit=0`` 会返回空列表
    让前端看起来像"没有报告"—— 那是两种完全不同的状态。
    """
    limit = max(1, min(int(limit), 200))
    conn = _connect(path)
    try:
        rows = conn.execute(
            "SELECT id, kind, title, status, period_start, period_end, created_at, updated_at"
            " FROM ops_reports ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    finally:
        conn.close()
    keys = ("id", "kind", "title", "status", "period_start", "period_end",
            "created_at", "updated_at")
    return [dict(zip(keys, r)) for r in rows]
