"""
本地语音识别接口（api_asr）的测试
==================================
这一组守住的东西，按重要性排：

1. **模型缺失不是 500**。这条路径换机器、清缓存、跑 CI 都会走到。返回 500
   会让人去重启服务，而重启一百次也不会有模型 —— 所以它必须是 200 + 明确的
   原因 + 可以直接复制执行的命令。
2. **两件事分开报告**。``model_ready`` 与 ``dependency_ready`` 合成一个
   ``available`` 之后，「模型没下载」和「sherpa-onnx 没装」变成同一句话，
   而下一步动作完全不同（下模型 vs 装包）。
3. **采样率不猜**。猜错不会报错，只会把识别结果变成乱码 —— 那是最难归因的
   一类故障，所以非 16 kHz 直接 400。
4. **鉴权没漏**。``api_asr.router`` 自己不带鉴权（不能反向导入
   ``require_token``，否则成环），靠 ``api_server`` 的 ``include_router``
   施加。漏写不会报错，只会让录音接口裸奔 —— 所以有一条静态断言读源码。

测试不导入 ``api_server``（会拉起 2.2 GB 重排模型），也不需要真的装
``sherpa-onnx`` 或真的下载模型：需要「已就绪」的用例会造一个空壳模型目录
并往 ``sys.modules`` 里塞一个假的 ``sherpa_onnx``。
"""
from __future__ import annotations

import ast
import os
import re
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from fastapi.testclient import TestClient

import api_asr

ROOT = Path(__file__).resolve().parent.parent


def _app() -> FastAPI:
    app = FastAPI()
    app.include_router(api_asr.router)
    return app


def _fake_model_dir(
    root: Path, *, quantized: bool = False, style: str = "epoch"
) -> Path:
    """造一个「文件都在、内容是空」的模型目录。

    ``style`` 默认是 ``"epoch"`` —— **照真实模型包的文件名来**：
    ``encoder-epoch-99-avg-1.int8.onnx``。第一版 fixture 用的是官方文档里的
    ``encoder.onnx``，于是测试全绿，而真模型装上之后 ``status`` 报「缺少
    onnx 文件」，文件却都在。fixture 比真实产物漂亮，就是在测一个不存在的
    世界。
    """
    d = root / api_asr.MODEL_NAME
    d.mkdir(parents=True, exist_ok=True)
    (d / "tokens.txt").write_text("<unk> 0\n", encoding="utf-8")
    suffix = ".int8.onnx" if quantized else ".onnx"
    for n in ("encoder", "decoder", "joiner"):
        name = f"{n}-epoch-99-avg-1{suffix}" if style == "epoch" else f"{n}{suffix}"
        (d / name).write_bytes(b"")
    return d


def _fake_sherpa() -> types.ModuleType:
    mod = types.ModuleType("sherpa_onnx")
    mod.OnlineRecognizer = object  # 只要能被 import 到就够
    return mod


def _pcm(seconds: float = 1.0, *, amplitude: float = 0.2) -> bytes:
    n = int(api_asr.SAMPLE_RATE * seconds)
    t = np.arange(n, dtype=np.float32) / api_asr.SAMPLE_RATE
    return (amplitude * np.sin(2 * np.pi * 220 * t)).astype("<f4").tobytes()


class TestProbe(unittest.TestCase):
    """模型文件探测。**不 import sherpa-onnx**，所以没装依赖时也要能跑。"""

    def test_missing_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            r = api_asr.probe_model(Path(tmp) / "nope")
        self.assertFalse(r["ready"])
        self.assertIn("不存在", r["reason"])
        self.assertIn("fetch_asr_model", r["hint"])

    def test_dir_without_tokens(self) -> None:
        # 解包不完整与「根本没下载」是两件事，原因文案要能分开。
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp) / api_asr.MODEL_NAME
            d.mkdir()
            (d / "encoder.onnx").write_bytes(b"")
            r = api_asr.probe_model(d)
        self.assertFalse(r["ready"])
        self.assertIn("tokens.txt", r["reason"])

    def test_dir_without_onnx(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp) / api_asr.MODEL_NAME
            d.mkdir()
            (d / "tokens.txt").write_text("x", encoding="utf-8")
            r = api_asr.probe_model(d)
        self.assertFalse(r["ready"])
        self.assertIn("onnx", r["reason"])

    def test_float_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            d = _fake_model_dir(Path(tmp))
            r = api_asr.probe_model(d)
        self.assertTrue(r["ready"])
        self.assertFalse(r["quantized"])

    def test_quantized_preferred(self) -> None:
        # int8 体积一半、速度快一倍、中文短句上几乎无差别 —— 有就优先用。
        with tempfile.TemporaryDirectory() as tmp:
            d = _fake_model_dir(Path(tmp), quantized=True)
            r = api_asr.probe_model(d)
            enc, _, _ = api_asr._onnx_paths(d)
        self.assertTrue(r["ready"])
        self.assertTrue(r["quantized"])
        self.assertIn(".int8.", enc.name)

    def test_real_model_file_names(self) -> None:
        # 真实模型包里带训练轮次。写死 `encoder.onnx` 会在这里红。
        with tempfile.TemporaryDirectory() as tmp:
            d = _fake_model_dir(Path(tmp), quantized=True, style="epoch")
            r = api_asr.probe_model(d)
        self.assertTrue(r["ready"])
        self.assertEqual(r["files"]["encoder"], "encoder-epoch-99-avg-1.int8.onnx")

    def test_plain_model_file_names(self) -> None:
        # 官方文档里的写法也要认 —— 别的模型包就是这个名字。
        with tempfile.TemporaryDirectory() as tmp:
            d = _fake_model_dir(Path(tmp), style="plain")
            r = api_asr.probe_model(d)
        self.assertTrue(r["ready"])
        self.assertEqual(r["files"]["encoder"], "encoder.onnx")

    def test_mixed_prefers_int8(self) -> None:
        # 一个包里同时有 fp32 与 int8 时挑 int8，且整组要一致 ——
        # 混着用（int8 encoder + fp32 joiner）能跑，但没人验证过。
        with tempfile.TemporaryDirectory() as tmp:
            d = _fake_model_dir(Path(tmp), quantized=True)
            (d / "encoder.onnx").write_bytes(b"")
            r = api_asr.probe_model(d)
            enc, dec, joi = api_asr._onnx_paths(d)
        self.assertTrue(r["ready"])
        self.assertTrue(r["quantized"])
        for p in (enc, dec, joi):
            self.assertIn(".int8.", p.name)


class TestStatus(unittest.TestCase):
    def test_status_never_raises_without_dependency(self) -> None:
        # 没装 sherpa-onnx、也没有模型目录：仍然要能回答「为什么不能用」。
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(
            os.environ, {"ASR_MODEL_DIR": str(Path(tmp) / "nope")}
        ):
            r = TestClient(_app()).get("/api/asr/status")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertFalse(body["available"])
        self.assertFalse(body["model_ready"])
        self.assertTrue(body["reason"])
        self.assertIn("fetch_asr_model", body["download"])

    def test_flags_are_separate(self) -> None:
        # 模型齐了但依赖没装：model_ready=True / dependency_ready=False。
        # 合成一个 available 的话，「下模型」和「装包」会变成同一句话。
        with tempfile.TemporaryDirectory() as tmp:
            d = _fake_model_dir(Path(tmp))
            with mock.patch.dict(os.environ, {"ASR_MODEL_DIR": str(d)}), mock.patch.dict(
                sys.modules, {"sherpa_onnx": None}
            ):
                # sys.modules 里是 None 时 `import sherpa_onnx` 抛 ImportError，
                # 正好模拟「没装」。
                body = api_asr.status()
        self.assertTrue(body["model_ready"])
        self.assertFalse(body["dependency_ready"])
        self.assertFalse(body["available"])
        self.assertIn("pip install sherpa-onnx", body["hint"])

    def test_available_when_both_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            d = _fake_model_dir(Path(tmp), quantized=True)
            with mock.patch.dict(os.environ, {"ASR_MODEL_DIR": str(d)}), mock.patch.dict(
                sys.modules, {"sherpa_onnx": _fake_sherpa()}
            ):
                body = api_asr.status()
        self.assertTrue(body["available"])
        self.assertEqual(body["engine"], "sherpa-onnx")
        self.assertEqual(body["sample_rate"], api_asr.SAMPLE_RATE)


class TestTranscribeEndpoint(unittest.TestCase):
    def _ready(self, tmp: str):
        d = _fake_model_dir(Path(tmp), quantized=True)
        return mock.patch.dict(os.environ, {"ASR_MODEL_DIR": str(d)}), mock.patch.dict(
            sys.modules, {"sherpa_onnx": _fake_sherpa()}
        )

    def test_503_when_model_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(
            os.environ, {"ASR_MODEL_DIR": str(Path(tmp) / "nope")}
        ):
            r = TestClient(_app()).post(
                "/api/asr", content=_pcm(), headers={"X-Sample-Rate": "16000"}
            )
        self.assertEqual(r.status_code, 503)
        body = r.json()
        # 消息里要有「怎么修」，而不是只有「失败」。
        self.assertIn("fetch_asr_model", body["download"])
        self.assertFalse(body["model_ready"])

    def test_happy_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env, dep = self._ready(tmp)
            with env, dep, mock.patch.object(
                api_asr, "decode_pcm", return_value="井口压力多少"
            ):
                r = TestClient(_app()).post(
                    "/api/asr", content=_pcm(1.0), headers={"X-Sample-Rate": "16000"}
                )
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["text"], "井口压力多少")
        self.assertEqual(body["engine"], "sherpa-onnx")
        self.assertAlmostEqual(body["duration_s"], 1.0, places=1)

    def test_rejects_wrong_sample_rate(self) -> None:
        # 不猜采样率：猜错不报错，只把结果变成乱码。
        with tempfile.TemporaryDirectory() as tmp:
            env, dep = self._ready(tmp)
            with env, dep:
                r = TestClient(_app()).post(
                    "/api/asr", content=_pcm(), headers={"X-Sample-Rate": "48000"}
                )
        self.assertEqual(r.status_code, 400)
        self.assertIn("16000", r.json()["detail"])

    def test_rejects_empty_body(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env, dep = self._ready(tmp)
            with env, dep:
                r = TestClient(_app()).post("/api/asr", content=b"")
        self.assertEqual(r.status_code, 400)

    def test_rejects_partial_float(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env, dep = self._ready(tmp)
            with env, dep:
                r = TestClient(_app()).post("/api/asr", content=b"\x00\x00\x00")
        self.assertEqual(r.status_code, 400)
        self.assertIn("4 的倍数", r.json()["detail"])

    def test_rejects_too_long(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env, dep = self._ready(tmp)
            with env, dep:
                r = TestClient(_app()).post(
                    "/api/asr", content=_pcm(api_asr.MAX_SECONDS + 2)
                )
        self.assertEqual(r.status_code, 413)
        self.assertIn("上限", r.json()["detail"])

    def test_rejects_nan(self) -> None:
        bad = np.array([0.1, np.nan, 0.2, 0.3], dtype="<f4").tobytes()
        with tempfile.TemporaryDirectory() as tmp:
            env, dep = self._ready(tmp)
            with env, dep:
                r = TestClient(_app()).post("/api/asr", content=bad)
        self.assertEqual(r.status_code, 400)
        self.assertIn("NaN", r.json()["detail"])

    def test_silence_is_200_not_error(self) -> None:
        # 全零不是「识别失败」，是「一个采样都没收到」。200 + silent 让界面
        # 能说「麦克风静音」，而不是说「没听清」。
        with tempfile.TemporaryDirectory() as tmp:
            env, dep = self._ready(tmp)
            with env, dep, mock.patch.object(api_asr, "decode_pcm") as dec:
                r = TestClient(_app()).post("/api/asr", content=_pcm(1.0, amplitude=0.0))
                called = dec.called
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json()["silent"])
        self.assertEqual(r.json()["text"], "")
        self.assertFalse(called)

    def test_recognition_failure_is_500_with_detail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env, dep = self._ready(tmp)
            with env, dep, mock.patch.object(
                api_asr, "decode_pcm", side_effect=RuntimeError("onnx 炸了")
            ):
                r = TestClient(_app()).post("/api/asr", content=_pcm())
        self.assertEqual(r.status_code, 500)
        self.assertIn("onnx 炸了", r.json()["detail"])


class TestResultText(unittest.TestCase):
    """新旧 sherpa-onnx 取结果的写法不同，两种都要认。"""

    def test_new_api(self) -> None:
        rec = types.SimpleNamespace(get_result=lambda s: types.SimpleNamespace(text=" 你好 "))
        self.assertEqual(api_asr._result_text(rec, object()), "你好")

    def test_old_api(self) -> None:
        rec = types.SimpleNamespace()
        stream = types.SimpleNamespace(result=types.SimpleNamespace(text="老接口"))
        self.assertEqual(api_asr._result_text(rec, stream), "老接口")

    def test_empty(self) -> None:
        rec = types.SimpleNamespace(get_result=lambda s: types.SimpleNamespace(text=""))
        self.assertEqual(api_asr._result_text(rec, object()), "")

    def test_v2_returns_plain_str(self) -> None:
        # sherpa-onnx 2.0 的 get_result 直接返回 str。`getattr(res, "text", "")`
        # 对 str 静默给出 ""，于是每句话都变成「没听清」—— 只有真跑一遍才会
        # 发现，单元测试必须钉住它。
        rec = types.SimpleNamespace(get_result=lambda s: " 你好世界 ")
        self.assertEqual(api_asr._result_text(rec, object()), "你好世界")


@unittest.skipUnless(
    api_asr.probe_model()["ready"],
    "本机没有模型（CI 就是这样），跳过真实识别",
)
class TestRealModel(unittest.TestCase):
    """真的把音频喂进 sherpa-onnx。

    这条断言的价值在**本地**：`sherpa-onnx` 2.0 的 ``get_result`` 直接返回
    ``str``，而 ``getattr(res, "text", "")`` 对 str 静默给出 ""。结果是
    「依赖在、模型在、每句话都返回空」，界面上读作「没听清」—— 单元测试
    全绿，功能全废。只有真跑一遍音频才发现。

    用模型包里自带的中文测试音频（``test_wavs/0.wav``，16 kHz 单声道），
    所以这条用例不需要麦克风、不需要人说话。
    """

    def test_decodes_bundled_wav(self) -> None:
        import wave

        wav = api_asr.model_root() / "test_wavs" / "0.wav"
        if not wav.is_file():
            self.skipTest("模型包里没有 test_wavs/0.wav")
        with wave.open(str(wav)) as w:
            self.assertEqual(w.getframerate(), api_asr.SAMPLE_RATE, "测试音频不是 16 kHz")
            self.assertEqual(w.getnchannels(), 1)
            raw = w.readframes(w.getnframes())
        samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        text = api_asr.decode_pcm(samples)
        # 不断言具体字句（模型换了就变），只断言「有中文字出来了」。
        self.assertTrue(text, "识别结果是空的 —— 多半是取结果的接口又变了")
        self.assertTrue(
            any("\u4e00" <= ch <= "\u9fff" for ch in text),
            f"识别结果里没有中文：{text!r}",
        )


class TestStatic(unittest.TestCase):
    def test_router_is_mounted_with_auth(self) -> None:
        # 漏写 dependencies 不会报错，只会让录音接口裸奔。
        src = (ROOT / "api_server.py").read_text(encoding="utf-8")
        self.assertRegex(
            src,
            re.compile(r"include_router\(\s*asr_router\s*,\s*dependencies=\[Depends\(require_token\)\]\s*\)"),
        )

    def test_model_name_matches_fetch_script(self) -> None:
        # 两处各写一遍模型名是刻意的（脚本不能 import numpy），
        # 所以这里把它们钉在一起。
        src = (ROOT / "scripts" / "fetch_asr_model.py").read_text(encoding="utf-8")
        m = re.search(r'^MODEL_NAME = "([^"]+)"', src, re.M)
        self.assertIsNotNone(m)
        self.assertEqual(m.group(1), api_asr.MODEL_NAME)

    def test_fetch_script_imports_stdlib_only(self) -> None:
        # 它要能在「依赖装坏」的环境里跑 —— 那是装依赖之前就该知道的事。
        #
        # 用 AST 而不是子串搜索：这个脚本的文档里写着「不 import requests」，
        # 子串匹配会把这句话当成真的 import，然后报一个「代码里明明没写」
        # 的失败。源码里的词与源码的行为是两件事。
        tree = ast.parse((ROOT / "scripts" / "fetch_asr_model.py").read_text(encoding="utf-8"))
        got: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                got.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                got.add(node.module.split(".")[0])
        allowed = {"__future__", "argparse", "os", "sys", "tarfile", "time", "urllib", "pathlib"}
        self.assertEqual(got - allowed, set())

    def test_router_has_no_auth_dependency_itself(self) -> None:
        # 反向导入 require_token 会成环。这条断言盯着「有人图省事直接加进去」。
        self.assertEqual(api_asr.router.dependencies, [])


if __name__ == "__main__":
    unittest.main()
