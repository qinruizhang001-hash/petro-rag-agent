#!/usr/bin/env python
"""下载本地语音识别模型（sherpa-onnx 中文流式 zipformer，14M）。

为什么模型不进仓库
------------------
解包后约 100 MB。放进 git 的后果不是仓库变慢，是**每个 clone 都要为它付费**，
包括那些只想读代码的人。与 ``bge-reranker-v2-m3`` 同样处理：脚本拉、目录被
``.gitignore`` 挡住。

只用标准库
----------
这个脚本要能在**依赖装坏的环境里**跑 —— 模型下不下来，恰恰是装依赖之前
就该知道的事。所以不 import requests，也不 import 项目里任何东西
（``api_asr.MODEL_NAME`` 也不 import：那会把 numpy 一起拉进来）。
代价是模型名在这里再写一遍，两边由 ``tests/test_api_asr.py`` 对齐。

代理
----
默认尊重 ``http_proxy`` / ``https_proxy``。但这台机器上代理曾经出现过
「监听正常、转发失败」的状态，curl 直连能通、走代理报 502 —— 所以留了
``--no-proxy``。报错信息里要能把这条说出来，否则人会以为是地址失效。
"""
from __future__ import annotations

import argparse
import os
import sys
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path

#: 与 api_asr.MODEL_NAME 必须一致（由 test_api_asr 断言）。
MODEL_NAME = "sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23"
URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
    f"{MODEL_NAME}.tar.bz2"
)
EXPECTED_BYTES = 74_004_050
ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "models" / "asr"


def _human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def is_ready(dest: Path) -> bool:
    """和 api_asr.probe_model 同一套判据，但不 import 它（那会把 numpy 拉进来）。

    按**族名**找（``encoder*.onnx``）而不是写死文件名：这个包里是
    ``encoder-epoch-99-avg-1.int8.onnx``，官方文档的示例是 ``encoder.onnx``。
    写死一种，另一个包里文件明明都在却报「模型缺失」，然后人会再下一次。
    """
    d = dest / MODEL_NAME
    if not (d / "tokens.txt").is_file():
        return False
    return all(any(d.glob(f"{stem}*.onnx")) for stem in ("encoder", "decoder", "joiner"))


def download(url: str, out: Path, *, no_proxy: bool) -> None:
    """流式下载，边下边报进度。

    为什么不用 ``urlretrieve``：它不报进度，而 74 MB 在这台机器上要几十分钟。
    没有任何输出的几十分钟与「卡死了」在屏幕上完全一样。
    """
    handlers = []
    if no_proxy:
        handlers.append(urllib.request.ProxyHandler({}))
    opener = urllib.request.build_opener(*handlers)
    req = urllib.request.Request(url, headers={"User-Agent": "petro-rag-agent/fetch-asr"})

    started = time.time()
    with opener.open(req, timeout=60) as resp, out.open("wb") as fh:
        total = int(resp.headers.get("Content-Length") or 0)
        got = 0
        last = 0.0
        while True:
            chunk = resp.read(256 * 1024)
            if not chunk:
                break
            fh.write(chunk)
            got += len(chunk)
            now = time.time()
            if now - last > 2:
                last = now
                rate = got / max(now - started, 1e-6)
                pct = f" {got * 100 / total:5.1f}%" if total else ""
                sys.stdout.write(f"\r  {_human(got)} / {_human(total or 0)}{pct}  {_human(rate)}/s")
                sys.stdout.flush()
    sys.stdout.write("\n")


def extract(tar: Path, dest: Path) -> None:
    """解包。``filter="data"`` 挡掉绝对路径与 ``..`` 之类的成员。"""
    with tarfile.open(tar, "r:bz2") as tf:
        try:
            tf.extractall(dest, filter="data")  # py3.12+
        except TypeError:  # 3.11 及更早没有 filter 参数
            tf.extractall(dest)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="下载本地 ASR 模型")
    ap.add_argument("--check", action="store_true", help="只检查在不在，不下载")
    ap.add_argument("--force", action="store_true", help="已存在也重下")
    ap.add_argument("--no-proxy", action="store_true", help="绕过 http_proxy 直连")
    args = ap.parse_args(argv)

    dest = Path(os.getenv("ASR_MODEL_DIR") or TARGET)
    if args.check:
        ok = is_ready(dest)
        print(("模型就绪: " if ok else "模型缺失: ") + str(dest / MODEL_NAME))
        return 0 if ok else 1

    if is_ready(dest) and not args.force:
        print(f"模型已就绪，跳过下载: {dest / MODEL_NAME}")
        return 0

    dest.mkdir(parents=True, exist_ok=True)
    tar = dest / f"{MODEL_NAME}.tar.bz2"
    print(f"下载 {URL}\n  → {tar}（约 {_human(EXPECTED_BYTES)}）")
    try:
        download(URL, tar, no_proxy=args.no_proxy)
    except urllib.error.HTTPError as exc:
        print(f"HTTP {exc.code}：{exc.reason}", file=sys.stderr)
        if exc.code == 502:
            print(
                "  502 通常不是地址失效，而是 http_proxy 在监听但不转发。\n"
                "  加 --no-proxy 再试一次。",
                file=sys.stderr,
            )
        return 2
    except Exception as exc:  # noqa: BLE001
        print(f"下载失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    size = tar.stat().st_size
    if EXPECTED_BYTES and size != EXPECTED_BYTES:
        # 不静默接受：截断的 tar.bz2 解包会报「不是 bzip2 文件」，
        # 那个报错指向压缩格式，而不是指向「下载没下完」。
        print(
            f"文件大小不对：{size} 字节，应为 {EXPECTED_BYTES}。"
            "可能下载中断，删掉后重试。",
            file=sys.stderr,
        )
        return 3

    print("解包…")
    extract(tar, dest)
    if not is_ready(dest):
        print(f"解包后仍缺文件：{dest / MODEL_NAME}", file=sys.stderr)
        return 4
    tar.unlink()
    print(f"完成: {dest / MODEL_NAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
