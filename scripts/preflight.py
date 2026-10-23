#!/usr/bin/env python3
"""
启动前自检（preflight）
=======================
clone 下来跑不起来，最常见的原因不是代码，而是**四个外部依赖**没到位：
reranker 权重（2.2 GB，不在 git 里）、Ollama 及其两个模型、向量库目录、
图谱索引。缺任何一个，症状都不一样，而且大多不是"缺了什么"这种一眼能看懂的错误：

============================  ==========================================
缺什么                        实际看到的报错
============================  ==========================================
models/bge-reranker-v2-m3     ``OSError: Unrecognized processing class``
                              （指向 sentence-transformers，方向完全错）
Ollama 未启动                  ``APIConnectionError: Connection error.``
qwen2.5 未 pull                ``model 'qwen2.5' not found``（HTTP 404）
chroma_petro 空                ``Collection [ccus_docs] does not exist``
lightrag_storage 空           地质分支静默降级，报告里只有井筒结论
============================  ==========================================

所以这个脚本只做一件事：**在启动前把它们逐条验一遍，并把"怎么修"直接打出来**。
它不 import 项目代码（只依赖标准库），因此在依赖装坏的情况下也能跑。

用法::

    python scripts/preflight.py            # 完整检查（起服务前）
    python scripts/preflight.py --gate     # 只检查质量闸门测试所需（最小路径）
    python scripts/preflight.py --json     # 机器可读，供 CI / 脚本消费

退出码：0 = 全部通过（warning 不算失败）；1 = 有 blocking 项。
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# 与 .env.example / docker-compose.yml 保持一致的默认值
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
LOCAL_MODEL = os.getenv("LOCAL_MODEL", "qwen2.5")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
CHROMA_PATH = Path(os.getenv("CCUS_CHROMA_PATH", ROOT / "chroma_petro"))
RERANKER_DIR = ROOT / "models" / "bge-reranker-v2-m3"
LIGHTRAG_DIR = ROOT / "lightrag_storage"

REMEDIATION = {
    "python": "安装 Python 3.10+（本项目在 3.10 / 3.11 / 3.12 上跑过）",
    "reranker": (
        "下载 2.2 GB 权重：\n"
        "    pip install huggingface_hub\n"
        "    python -c \"from huggingface_hub import snapshot_download as s; "
        "s('BAAI/bge-reranker-v2-m3', local_dir='models/bge-reranker-v2-m3')\"\n"
        "  没有它后端在 import 阶段就会崩（不是降级）。"
    ),
    "ollama": (
        "启动 Ollama 并让容器/进程能访问：\n"
        "    OLLAMA_HOST=0.0.0.0:11434 ollama serve\n"
        "  容器里访问宿主机用 http://host.docker.internal:11434/v1"
    ),
    "ollama_model": "ollama pull {model}",
    "chroma": (
        "建库：python ingest_ccus_batch.py\n"
        "  语料在 ccus_docs/（PDF 不入库，需先放到该目录）"
    ),
    "lightrag": (
        "建图谱：python graph_rag_test.py\n"
        "  缺它时地质分支会静默降级 —— 报告仍然生成，但只有井筒结论，"
        "数据完整性提示会写在报告末尾。"
    ),
    "gate_deps": (
        "只跑质量闸门测试所需的最小依赖：\n"
        "    pip install -r requirements-graph.txt"
    ),
}


class Check:
    def __init__(self, name: str, ok: bool, detail: str, *, blocking: bool = True,
                 fix: str = ""):
        self.name = name
        self.ok = ok
        self.detail = detail
        self.blocking = blocking
        self.fix = fix

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "ok": self.ok,
            "blocking": self.blocking,
            "detail": self.detail,
            "fix": self.fix,
        }


def _check_python() -> Check:
    ok = sys.version_info >= (3, 10)
    return Check("python", ok, f"{sys.version.split()[0]}（需要 ≥ 3.10）",
                 fix=REMEDIATION["python"])


def _check_gate_deps() -> Check:
    """最小路径：质量闸门测试只需要 langgraph + openai + dotenv。"""
    missing = []
    for module in ("langgraph", "openai", "dotenv"):
        try:
            __import__(module)
        except Exception:  # noqa: BLE001
            missing.append(module)
    return Check(
        "gate_deps",
        not missing,
        "已安装" if not missing else f"缺少 {', '.join(missing)}",
        fix=REMEDIATION["gate_deps"],
    )


def _check_reranker() -> Check:
    if not RERANKER_DIR.is_dir():
        return Check("reranker", False, f"目录不存在：{RERANKER_DIR}",
                     fix=REMEDIATION["reranker"])
    weights = list(RERANKER_DIR.glob("*.safetensors")) + list(RERANKER_DIR.glob("*.bin"))
    if not weights:
        return Check("reranker", False, f"{RERANKER_DIR} 存在但没有权重文件",
                     fix=REMEDIATION["reranker"])
    size_gb = sum(p.stat().st_size for p in weights) / 1024 ** 3
    return Check("reranker", True, f"{weights[0].name}，{size_gb:.1f} GB")


def _http_json(url: str, timeout: float = 3.0):
    with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310
        return json.loads(resp.read().decode("utf-8"))


def _check_ollama() -> list[Check]:
    try:
        payload = _http_json(f"{OLLAMA_BASE_URL}/api/tags")
    except (urllib.error.URLError, socket.timeout, OSError, ValueError) as exc:
        return [Check("ollama", False, f"{OLLAMA_BASE_URL} 不可达（{exc}）",
                      fix=REMEDIATION["ollama"])]

    names = [m.get("name") or m.get("model") or "" for m in payload.get("models") or []]
    checks = [Check("ollama", True, f"{OLLAMA_BASE_URL} 可达，已加载 {len(names)} 个模型")]
    for model in (LOCAL_MODEL, EMBED_MODEL):
        # Ollama 的 tag 带版本后缀（qwen2.5:7b），所以按前缀匹配
        found = any(n == model or n.startswith(f"{model}:") for n in names)
        checks.append(Check(
            f"ollama:{model}", found,
            "已 pull" if found else f"未找到（现有：{', '.join(names[:5]) or '无'}）",
            fix=REMEDIATION["ollama_model"].format(model=model),
        ))
    return checks


def _check_chroma() -> Check:
    sqlite = CHROMA_PATH / "chroma.sqlite3"
    if not CHROMA_PATH.is_dir():
        return Check("chroma", False, f"目录不存在：{CHROMA_PATH}", fix=REMEDIATION["chroma"])
    if not sqlite.exists() or sqlite.stat().st_size < 4096:
        return Check("chroma", False, f"{CHROMA_PATH} 存在但没有建库（chroma.sqlite3 缺失/过小）",
                     fix=REMEDIATION["chroma"])
    size_mb = sqlite.stat().st_size / 1024 ** 2
    return Check("chroma", True, f"{CHROMA_PATH.name}/chroma.sqlite3，{size_mb:.1f} MB")


def _check_lightrag() -> Check:
    # 非阻断：缺图谱只会让地质分支降级，报告仍会生成并标注数据不完整。
    if not LIGHTRAG_DIR.is_dir():
        return Check("lightrag", False, f"未建图谱（{LIGHTRAG_DIR.name}/ 不存在）",
                     blocking=False, fix=REMEDIATION["lightrag"])
    files = list(LIGHTRAG_DIR.rglob("*"))
    if not files:
        return Check("lightrag", False, f"{LIGHTRAG_DIR.name}/ 为空", blocking=False,
                     fix=REMEDIATION["lightrag"])
    return Check("lightrag", True, f"{LIGHTRAG_DIR.name}/ 共 {len(files)} 个文件")


def _check_env() -> list[Check]:
    token = (os.getenv("API_TOKEN") or "").strip()
    checks = [Check(
        "api_token", bool(token),
        "已设置" if token else "未设置 —— /api/* 鉴权将关闭（本地开发可以，对外不行）",
        blocking=False,
    )]
    if not (os.getenv("CLOUD_API_KEY") or "").strip():
        checks.append(Check(
            "cloud_key", True,
            "未设置 —— 报告阶段降级到本地模型，model_routing 会写明原因（受支持）",
            blocking=False,
        ))
    return checks


def run(mode: str = "full") -> list[Check]:
    checks = [_check_python(), _check_gate_deps()]
    if mode == "gate":
        return checks
    checks.append(_check_reranker())
    checks.extend(_check_ollama())
    checks.append(_check_chroma())
    checks.append(_check_lightrag())
    checks.extend(_check_env())
    return checks


def _print(checks: list[Check]) -> None:
    width = max(len(c.name) for c in checks) + 2
    for c in checks:
        mark = "OK  " if c.ok else ("FAIL" if c.blocking else "WARN")
        print(f"[{mark}] {c.name:<{width}} {c.detail}")
    fixes = [c for c in checks if not c.ok and c.fix]
    if fixes:
        print("\n修复方式：")
        for c in fixes:
            print(f"  · {c.name}: {c.fix}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CCUS 项目启动前自检")
    parser.add_argument("--gate", action="store_true",
                        help="只检查质量闸门测试所需的最小依赖（不需要模型/向量库）")
    parser.add_argument("--json", action="store_true", help="输出 JSON")
    args = parser.parse_args(argv)

    checks = run("gate" if args.gate else "full")
    if args.json:
        print(json.dumps([c.as_dict() for c in checks], ensure_ascii=False, indent=2))
    else:
        _print(checks)

    failed = [c for c in checks if not c.ok and c.blocking]
    if failed:
        print(f"\n{len(failed)} 项阻断性问题，服务起不来。", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
