"""本地语音识别（sherpa-onnx）—— 让语音输入不再把现场录音送出本机。

为什么单独一个模块，而不是写进 ``api_server.py``
------------------------------------------------
理由与 ``api_ops.py`` 完全相同：``api_server.py`` 顶部会加载 LangGraph 全图与
2.2 GB 重排模型。把语音端点写在那里，意味着**验证「模型没下载时会不会崩」**
这件事也要先把重排模型拉起来 —— 而这恰恰是最需要反复测的一条路径。
本模块只依赖 FastAPI + numpy + （可选的）sherpa-onnx，测试可以直接挂一个裸
``FastAPI()`` 跑。

鉴权同样不在本模块里：``require_token`` 定义在 ``api_server.py``，反向导入会
成环。由 ``api_server.py`` 的 ``include_router`` 统一施加，测试里有一条静态
断言盯着它（同 ``test_api_ops.py``）。

为什么收裸 PCM，不收 webm / mp3
-------------------------------
浏览器 ``MediaRecorder`` 给的是 webm/opus，而 sherpa-onnx 只吃采样点。要在
后端解码就得有 ffmpeg（本机不一定有）或者额外装解码器（镜像变大、许可证要
重新扫一遍）。而前端本来就要做一件事：把麦克风的 Float32 降采样到 16 kHz。
做完之后直接把这串 float32 发过来，后端 ``np.frombuffer`` 就完了 —— 中间
没有任何编解码环节，也就没有任何「格式对不上」的失败模式。

为什么不做 WebSocket 流式
-------------------------
流式会在人说话的过程中不断吐出**半句**，然后自己改自己。这个系统里最贵
的错误是「数值正确、对象错误」：把 ``井口压力`` 先识别成 ``景口压力`` 显示
出来、再说第二遍改成对的 —— 读者已经读过一遍错的了，而第一印象不会撤回。
一次说完、一次返回，结果稳定，人才会认真核那一遍。

代价是没有「边说边出字」的观感。这是有意换来的：录音期间界面显示的是
录音时长与「本地识别中」，而不是猜测中的字。

模型不在仓库里
--------------
``sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23`` 解包后约 100 MB，
与 ``bge-reranker-v2-m3`` 同样处理：不进 git，由
``scripts/fetch_asr_model.py`` 拉。**模型缺失不是 500** —— 这条路径会被
频繁走到（换机器、清缓存、CI），所以它返回 200 + ``available: false`` +
可直接复制执行的下载命令。把「没装模型」渲染成「服务坏了」，会让人去重启
服务，而重启一百次也不会有模型。
"""
from __future__ import annotations

import asyncio
import os
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from utils.logging_setup import get_logger

log = get_logger(__name__)

#: 不带鉴权依赖，由 api_server 在 include 时统一施加（见模块文档）。
router = APIRouter(tags=["asr"])

#: 模型目录名。与 ``scripts/fetch_asr_model.py`` 里的常量必须一致 ——
#: 两处写死同一个字符串是这里唯一可接受的重复，因为脚本要在没有本模块的
#: 情况下也能跑（CI 的零依赖 job 里没有 numpy）。
MODEL_NAME = "sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23"

#: sherpa-onnx 的 zipformer 全部按 16 kHz 训练。改这个数字不会让别的采样率
#: 能识别，只会让识别结果变成乱码 —— 所以客户端必须自己重采样，服务端只做
#: 校验，不做「贴心的自动重采样」。
SAMPLE_RATE = 16000

#: 一段话的上限。60 秒 ≈ 3.8 MB float32，够说很长的问题了；
#: 再长就说明麦克风忘了关 —— 那时候要拒绝，而不是默默吃下 500 MB 内存。
MAX_SECONDS = 60

#: 三个 onnx 的「族名」。真实文件名带训练轮次（见 `_pick_onnx`）。
ONNX_STEMS = ("encoder", "decoder", "joiner")

DOWNLOAD_CMD = "python scripts/fetch_asr_model.py"


# ══════════════════════════════════════════════════════════
# 模型文件探测（纯函数，不碰 sherpa-onnx）
# ══════════════════════════════════════════════════════════


def model_root() -> Path:
    """模型根目录。``ASR_MODEL_DIR`` 可覆盖，便于把大文件放在别的盘。"""
    env = (os.getenv("ASR_MODEL_DIR") or "").strip()
    if env:
        return Path(env).expanduser()
    return Path(__file__).resolve().parent / "models" / "asr" / MODEL_NAME


def _pick_onnx(d: Path, stem: str) -> Path | None:
    """在目录里找 ``<stem>*.onnx``，int8 优先。

    **不能写死文件名。** 这个模型包里是
    ``encoder-epoch-99-avg-1.int8.onnx``（带训练轮次），而 sherpa-onnx 官方
    文档的示例写的是 ``encoder.onnx``。写死其中一种的后果是：换一个模型包，
    文件明明都在，``status`` 却报「模型缺失」—— 然后人跑去重新下载 74 MB，
    下完还是一样。判据只能是「有没有这一族的文件」。
    """
    cands = sorted(p for p in d.glob(f"{stem}*.onnx") if p.is_file())
    if not cands:
        return None
    int8 = [p for p in cands if ".int8." in p.name]
    return (int8 or cands)[0]


def probe_model(d: Path | None = None) -> dict[str, Any]:
    """模型齐不齐。**不 import sherpa-onnx** —— 没装依赖时也要能回答这个问题。

    三种「不齐」要分开说，因为下一步动作不同：
    目录不存在（根本没下载）、没有 ``tokens.txt``（解包没解全）、
    缺某一族的 onnx（包不对）。
    """
    d = d or model_root()
    if not d.is_dir():
        return {
            "ready": False,
            "dir": str(d),
            "quantized": False,
            "files": {},
            "reason": "模型目录不存在",
            "hint": f"跑 `{DOWNLOAD_CMD}` 下载（约 74 MB）。",
        }
    if not (d / "tokens.txt").is_file():
        return {
            "ready": False,
            "dir": str(d),
            "quantized": False,
            "files": {},
            "reason": "模型目录里没有 tokens.txt（解包不完整）",
            "hint": f"删掉 {d} 后重跑 `{DOWNLOAD_CMD}`。",
        }

    picks = {stem: _pick_onnx(d, stem) for stem in ONNX_STEMS}
    missing = [stem for stem, p in picks.items() if p is None]
    if missing:
        return {
            "ready": False,
            "dir": str(d),
            "quantized": False,
            "files": {},
            "reason": f"缺少 {', '.join(missing)} 的 onnx 文件",
            "hint": f"删掉 {d} 后重跑 `{DOWNLOAD_CMD}`。",
        }

    files = {stem: p.name for stem, p in picks.items() if p is not None}
    return {
        "ready": True,
        "dir": str(d),
        "quantized": all(".int8." in name for name in files.values()),
        "files": files,
        "reason": "",
        "hint": "",
    }


def _onnx_paths(d: Path) -> tuple[Path, Path, Path]:
    """挑一组 onnx。int8 优先（体积一半、速度快一倍，中文短句几乎无差别）。"""
    picks = [_pick_onnx(d, stem) for stem in ONNX_STEMS]
    if any(p is None for p in picks):
        raise FileNotFoundError(f"模型目录里缺少 onnx：{d}")
    enc, dec, joi = picks
    assert enc is not None and dec is not None and joi is not None
    return enc, dec, joi


# ══════════════════════════════════════════════════════════
# 识别器（懒加载单例）
# ══════════════════════════════════════════════════════════

#: 14M 模型初始化约 0.3 秒，但**第一次**要读约 100 MB 的 onnx —— 冷盘上
#: 能到几秒。不缓存在模块级的话，每说一句话都要重付一次。
_REC_LOCK = threading.Lock()
_REC: Any = None
_REC_DIR: str | None = None

#: 识别本身串行化。
#:
#: 一个 14M 的模型并发解码能省几十毫秒，代价是「sherpa-onnx 的 C++ 侧对
#: 并发 stream 到底安不安全」要由我们来担保 —— 而那是猜的。这个控制台是
#: 单人本地用的，排队几乎不会被感知；崩一次却要查很久。
_DECODE_LOCK = threading.Lock()


def reset_recognizer() -> None:
    """丢掉缓存的识别器。测试用；也是「换了模型目录」后唯一正确的动作。"""
    global _REC, _REC_DIR
    with _REC_LOCK:
        _REC = None
        _REC_DIR = None


def _build_recognizer(d: Path) -> Any:
    """构造 sherpa-onnx 流式识别器。

    用 ``OnlineRecognizer`` 而不是 ``OfflineRecognizer``，是因为手上的模型是
    流式 zipformer（encoder 带状态输入），离线识别器加载不了它。整段识别
    照样能做：把全部采样一次性喂进一个 stream，然后一直 decode 到不 ready。
    这样「以后要不要改成边说边出字」不需要换模型。
    """
    import sherpa_onnx  # 延迟导入：没装依赖时 status 仍要能回答

    enc, dec, joi = _onnx_paths(d)
    return sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=str(d / "tokens.txt"),
        encoder=str(enc),
        decoder=str(dec),
        joiner=str(joi),
        # 2 线程：本机同时还要跑 Ollama 与重排模型，语音不该抢满核。
        num_threads=2,
        sample_rate=SAMPLE_RATE,
        feature_dim=80,
        decoding_method="greedy_search",
        provider="cpu",
    )


def get_recognizer(d: Path) -> Any:
    global _REC, _REC_DIR
    with _REC_LOCK:
        if _REC is not None and _REC_DIR == str(d):
            return _REC
        _REC = _build_recognizer(d)
        _REC_DIR = str(d)
        return _REC


def _result_text(rec: Any, stream: Any) -> str:
    """取识别结果。三种取法都试，**因为三种都真实存在过**。

    1. ``stream.result.text`` —— 早期版本；
    2. ``recognizer.get_result(stream).text`` —— 1.9 前后；
    3. ``recognizer.get_result(stream)`` 直接返回 ``str`` —— 2.0（本机实测）。

    写死一种的后果不是报错，是**每句话都返回空字符串**。而空字符串在界面上
    读作「没听清」—— 没人会怀疑是版本问题，只会觉得模型不行。这条路径只能
    靠真跑一遍发现：`getattr(res, "text", "")` 对 str 静默给出 ""。
    """
    res = None
    getter = getattr(rec, "get_result", None)
    if callable(getter):
        try:
            res = getter(stream)
        except Exception:  # noqa: BLE001 —— 老版本会抛 AttributeError/TypeError
            res = None
    if res is None:
        res = getattr(stream, "result", None)
    if isinstance(res, str):
        return res.strip()
    text = getattr(res, "text", "") if res is not None else ""
    return (text or "").strip()


def decode_pcm(samples: np.ndarray, d: Path | None = None) -> str:
    """整段 float32 采样 → 文字。同步函数，调用方负责丢进线程。"""
    d = d or model_root()
    with _DECODE_LOCK:
        rec = get_recognizer(d)
        stream = rec.create_stream()
        stream.accept_waveform(SAMPLE_RATE, samples)
        # 不额外补静音：``is_ready`` 内部已经算进了右上下文需要的填充帧，
        # 外面再补一段只会把最后半句的尾部特征改掉。上面这段注释留着，是因为
        # 「补 0.3 秒静音」看起来像个合理的修法，而它会让结尾几个字变差。
        while rec.is_ready(stream):
            rec.decode_stream(stream)
        return _result_text(rec, stream)


# ══════════════════════════════════════════════════════════
# 状态
# ══════════════════════════════════════════════════════════


def status() -> dict[str, Any]:
    """完整可用性：模型在不在 + 依赖装没装。

    两件事必须分开报告。合成一个 ``available`` 会让「模型没下载」和
    「sherpa-onnx 没装」变成同一句话，而这两件事的下一步动作完全不同。
    """
    probe = probe_model()
    dep_ok = True
    dep_reason = ""
    try:
        import sherpa_onnx  # noqa: F401
    except Exception as exc:  # noqa: BLE001 —— ImportError 之外还可能是 DLL 缺失
        dep_ok = False
        dep_reason = f"{type(exc).__name__}: {exc}"

    ready = bool(probe["ready"] and dep_ok)
    reason = probe["reason"] or (f"sherpa-onnx 没装好（{dep_reason}）" if not dep_ok else "")
    hint = probe["hint"] or (
        "跑 `pip install sherpa-onnx`。" if not dep_ok else ""
    )
    return {
        "available": ready,
        "engine": "sherpa-onnx",
        "model": MODEL_NAME,
        "model_ready": bool(probe["ready"]),
        "dependency_ready": dep_ok,
        "quantized": bool(probe["quantized"]),
        "files": probe.get("files", {}),
        "dir": probe["dir"],
        "sample_rate": SAMPLE_RATE,
        "max_seconds": MAX_SECONDS,
        "reason": reason,
        "hint": hint,
        "download": DOWNLOAD_CMD,
    }


# ══════════════════════════════════════════════════════════
# 端点
# ══════════════════════════════════════════════════════════


@router.get("/api/asr/status")
def asr_status() -> dict[str, Any]:
    """能不能用、用的哪个模型、不能用时怎么办。永远 200。

    故意不鉴权之外的任何门槛：这个端点会被前端在**渲染麦克风按钮之前**
    调用，让它失败只会让按钮永远显示「不可用」，而不是「不可用的原因」。
    """
    return status()


@router.post("/api/asr")
async def asr_transcribe(request: Request):
    """裸 PCM（little-endian float32，16 kHz 单声道）→ 文字。

    请求体是二进制，不走 multipart：multipart 会多一层 boundary 解析，
    而这里只有一段音频，没有文件名、没有字段名，包一层只是多一个能出错的地方。
    """
    st = status()
    if not st["available"]:
        # 503 而不是 500：这是「依赖还没就绪」，不是「代码有 bug」。
        # 消息里带上下载命令，调用方不需要去翻文档。
        return JSONResponse(
            status_code=503,
            content={
                "detail": st["reason"] or "本地识别不可用",
                "hint": st["hint"],
                "download": st["download"],
                "model_ready": st["model_ready"],
                "dependency_ready": st["dependency_ready"],
            },
        )

    rate_hdr = request.headers.get("x-sample-rate")
    if rate_hdr:
        try:
            rate = int(rate_hdr)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"X-Sample-Rate 不是整数：{rate_hdr!r}")
        if rate != SAMPLE_RATE:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"只接受 {SAMPLE_RATE} Hz。收到 {rate} Hz —— 请在前端重采样，"
                    "服务端不猜采样率：猜错不会报错，只会识别成乱码。"
                ),
            )

    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="请求体是空的：没有音频可识别。")
    if len(body) % 4 != 0:
        raise HTTPException(
            status_code=400,
            detail=f"请求体长度 {len(body)} 不是 4 的倍数，不是 float32 序列。",
        )

    n = len(body) // 4
    if n > SAMPLE_RATE * MAX_SECONDS:
        raise HTTPException(
            status_code=413,
            detail=(
                f"音频 {n / SAMPLE_RATE:.1f} 秒，超过上限 {MAX_SECONDS} 秒。"
                "麦克风可能忘了关 —— 一次说一句。"
            ),
        )

    samples = np.frombuffer(body, dtype="<f4").astype(np.float32)
    if not np.isfinite(samples).all():
        raise HTTPException(status_code=400, detail="音频里有 NaN / Inf，无法识别。")
    peak = float(np.max(np.abs(samples))) if n else 0.0
    if peak < 1e-4:
        # 全零不是错误，是麦克风静音或被系统静音。返回 200 + 空文本会让
        # 界面说「没听清」，而真正的原因是「一个采样都没收到」。
        return JSONResponse(
            status_code=200,
            content={
                "text": "",
                "engine": "sherpa-onnx",
                "model": MODEL_NAME,
                "duration_s": round(n / SAMPLE_RATE, 2),
                "silent": True,
                "elapsed_ms": 0,
            },
        )

    t0 = time.perf_counter()
    try:
        # 识别是纯 CPU 同步计算，几秒钟。直接在事件循环里跑会把整个后端
        # （包括 /health）一起卡住 —— 那时候前端会以为后端挂了。
        text = await asyncio.to_thread(decode_pcm, samples)
    except Exception as exc:  # noqa: BLE001
        log.exception("[ASR] 识别失败")
        raise HTTPException(status_code=500, detail=f"识别失败：{type(exc).__name__}: {exc}")

    elapsed = int((time.perf_counter() - t0) * 1000)
    log.info("[ASR] %.2f 秒音频 → %d 字，用时 %d ms", n / SAMPLE_RATE, len(text), elapsed)
    return {
        "text": text,
        "engine": "sherpa-onnx",
        "model": MODEL_NAME,
        "duration_s": round(n / SAMPLE_RATE, 2),
        "silent": False,
        "elapsed_ms": elapsed,
    }
