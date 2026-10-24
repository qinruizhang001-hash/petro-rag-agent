#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
清理「半途中断」的入库残留。

## 为什么需要这个脚本

图谱重建是一个 4 小时级别的任务，中途被打断（合盖、Ctrl-C、会话被杀）是
常态。LightRAG 的中断会留下一种**很难发现**的残留：

- ``kv_store_full_docs`` 里有这篇的正文（写入早，先落盘）
- ``kv_store_doc_status`` 里状态停在 ``processing``
- ``kv_store_text_chunks`` 里**一个 chunk 都没有**（chunk 与实体是最后一起提交的）

这个组合的危险在于：重跑时 LightRAG 的去重判定是

    if mode == "create" and not resume:
        if doc_key not in await full_docs.filter_keys({doc_key}):
            logger.warning("This document is already in the storage.")
            return

也就是说它只看 ``full_docs``。正文在 → 直接 return，**不报错、不入库**。
而入库脚本跳过哪篇的依据是 ``graph_ingest_state.json``，这篇不在里面，
于是脚本认为「我处理过了」，日志打 ✅，实际什么都没进图。

结果：语料静默少一篇，覆盖率看起来正常。这正是本项目最该防的那类错误
——不是崩溃，是「看起来成功了」。

## 判定规则

只清理**确定没有贡献**的文档：``status != processed`` 且该 doc 名下一个
chunk 都不存在。有任何 chunk 落盘的都不动（说明实体可能已经合并进图，
重跑反而安全，实体按名字合并，不会重复）。

## 用法

    python scripts/repair_partial_ingest.py            # 只报告，不改文件
    python scripts/repair_partial_ingest.py --apply    # 实际清理

清理后重跑 ``ingest_ccus_batch.py`` 即可，被清掉的那篇会被当成新文档处理。
"""

from __future__ import annotations

import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

STORAGE_DIR = os.path.join(PROJECT_ROOT, "lightrag_storage")
FULL_DOCS = os.path.join(STORAGE_DIR, "kv_store_full_docs.json")
DOC_STATUS = os.path.join(STORAGE_DIR, "kv_store_doc_status.json")
CHUNKS = os.path.join(STORAGE_DIR, "kv_store_text_chunks.json")


def _load(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _save(path: str, data: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
    os.replace(tmp, path)  # 原子替换：写到一半再被打断也不会留下半个 JSON


def find_partial(storage_dir: str = STORAGE_DIR) -> list[dict]:
    """返回「状态未完成且零 chunk」的文档列表，不做任何修改。"""
    full_docs = _load(os.path.join(storage_dir, "kv_store_full_docs.json"))
    doc_status = _load(os.path.join(storage_dir, "kv_store_doc_status.json"))
    chunks = _load(os.path.join(storage_dir, "kv_store_text_chunks.json"))

    chunk_count: dict[str, int] = {}
    for rec in chunks.values():
        if isinstance(rec, dict):
            doc_id = rec.get("full_doc_id")
            if doc_id:
                chunk_count[doc_id] = chunk_count.get(doc_id, 0) + 1

    out = []
    for doc_id, rec in doc_status.items():
        if not isinstance(rec, dict):
            continue
        status = rec.get("status")
        if status == "processed":
            continue
        n_chunks = chunk_count.get(doc_id, 0)
        if n_chunks:
            # 已经有 chunk 落盘，实体可能已合并进图。不清：重跑时实体按名字
            # 合并，不会产生重复，清掉反而丢内容。
            continue
        out.append(
            {
                "doc_id": doc_id,
                "file_path": rec.get("file_path") or "?",
                "status": status,
                "declared_chunks": len(rec.get("chunks_list") or []),
                "stored_chunks": 0,
                "in_full_docs": doc_id in full_docs,
            }
        )
    return out


def repair(storage_dir: str = STORAGE_DIR, apply: bool = False) -> dict:
    partial = find_partial(storage_dir)
    removed = []
    if apply and partial:
        full_path = os.path.join(storage_dir, "kv_store_full_docs.json")
        status_path = os.path.join(storage_dir, "kv_store_doc_status.json")
        full_docs = _load(full_path)
        doc_status = _load(status_path)
        for item in partial:
            full_docs.pop(item["doc_id"], None)
            doc_status.pop(item["doc_id"], None)
            removed.append(item["file_path"])
        _save(full_path, full_docs)
        _save(status_path, doc_status)
    return {"partial": partial, "removed": removed, "applied": bool(apply and partial)}


def main() -> int:
    ap = argparse.ArgumentParser(description="清理半途中断的入库残留")
    ap.add_argument("--apply", action="store_true", help="实际清理（默认只报告）")
    ap.add_argument("--storage", default=STORAGE_DIR, help="图谱存储目录")
    args = ap.parse_args()

    res = repair(args.storage, apply=args.apply)
    if not res["partial"]:
        print("没有发现半途中断的文档。")
        return 0

    print(f"发现 {len(res['partial'])} 篇半途中断（状态未完成 + 零 chunk）：")
    for it in res["partial"]:
        print(
            f"  {it['file_path']:22} status={it['status']:12} "
            f"声明 {it['declared_chunks']:3} chunk / 实存 0  "
            f"full_docs={'有' if it['in_full_docs'] else '无'}"
        )
    if res["applied"]:
        print(f"\n已清理 {len(res['removed'])} 篇，重跑 ingest 会重新处理它们。")
        print("（只删了 full_docs 与 doc_status 记录，未动图谱与 chunk 存储）")
    else:
        print("\n这是预演。加 --apply 才会真正清理。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
