"""
本地 OpenAI 兼容网关（CCUS 项目的 CLOUD_BASE_URL 落点）
========================================================
把 ``provider/model`` 这样的模型名路由到真正的上游，对外只暴露一套
OpenAI wire format。监听 ``127.0.0.1:3425``，与 README 里写的
``CLOUD_BASE_URL=http://127.0.0.1:3425/v1`` 一致。

为什么仓库里要带一个网关
------------------------
README 一直说"任何 OpenAI 兼容聚合器都可以"（magpie / one-api / LiteLLM）。
那句话是对的，但它有个后果：**CI 和别人的机器上没有那个聚合器，网关这一跳
就永远没被真正跑通过**。实测过一次 —— 端口没人监听时，报告阶段直接
`Connection error`，看起来像应用坏了，其实是网关没起。

这个文件把那一跳变成仓库里可复现的东西：不装任何第三方软件，一条命令
就能让 ``CLOUD_BASE_URL`` 真的可达，于是"网关 → 上游"的路由、鉴权头透传、
流式 SSE 转发都能被真实测到。想换回 magpie 只要关掉它、启动 magpie ——
端口和协议都一样，应用侧零改动。

路由规则
--------
模型名必须带 ``provider/`` 前缀（与 magpie 的约定一致）：

    ollama/qwen2.5:latest   → OLLAMA_BASE_URL   （默认 http://localhost:11434/v1）
    deepseek/deepseek-chat  → DEEPSEEK_BASE_URL （需要 DEEPSEEK_API_KEY）
    openai/gpt-4o-mini      → OPENAI_BASE_URL   （需要 OPENAI_API_KEY）

没有配置 key 的 provider 不会出现在 ``/v1/models`` 里，调用它会返回 404 ——
让"这个上游没配"在体检阶段就暴露，而不是等到报告生成时才 401。

安全边界
--------
默认只绑 ``127.0.0.1``：本机任何进程都能连，外网不能。这是本项目对"本地
网关"的定义 —— 不做多租户，不做计费，只做路由。若确实要绑到非回环地址，
必须设置 ``GATEWAY_API_KEY``，否则拒绝启动（见 ``main``）。

用法:
    python gateway.py                     # 前台运行，Ctrl+C 退出
    python gateway.py --port 3425 --verbose
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 上游定义：provider 前缀 → (base_url 环境变量, base_url 默认值, key 环境变量)
PROVIDERS: dict[str, tuple[str, str, str]] = {
    "ollama": ("OLLAMA_BASE_URL", "http://localhost:11434/v1", "OLLAMA_API_KEY"),
    "deepseek": ("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1", "DEEPSEEK_API_KEY"),
    "openai": ("OPENAI_BASE_URL", "https://api.openai.com/v1", "OPENAI_API_KEY"),
}

# 本地上游不需要真 key，占位即可（Ollama 不校验）
_PLACEHOLDER_KEY = "gateway"

_VERBOSE = False


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} | {msg}", flush=True)


def vlog(msg: str) -> None:
    if _VERBOSE:
        log(msg)


def provider_base(provider: str) -> tuple[str, str] | None:
    """返回 (base_url, api_key)；provider 未知或未配置 key 时返回 None。"""
    spec = PROVIDERS.get(provider)
    if not spec:
        return None
    url_env, url_default, key_env = spec
    base = (os.getenv(url_env) or url_default).rstrip("/")
    key = (os.getenv(key_env) or "").strip()
    if provider == "ollama":
        return base, key or _PLACEHOLDER_KEY
    # 真云端必须配 key，否则视为不可用 —— 宁可 /models 里不出现，
    # 也不要让调用方以为它能用
    return (base, key) if key else None


def split_model(model: str) -> tuple[str, str]:
    """
    把 ``provider/model`` 拆开。

    注意模型名自己可能带冒号和斜杠（``qwen2.5:latest``、``org/model``），
    所以只按**第一个** ``/`` 切一次，其余原样保留给上游。
    """
    if "/" not in model:
        return "", model
    provider, rest = model.split("/", 1)
    return provider.strip().lower(), rest


def upstream_request(url: str, payload: dict | None, key: str, stream: bool,
                     timeout: int = 300):
    """构造发往上游的请求。stream 只影响读法，请求头一致。"""
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method="POST" if data else "GET")
    req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "text/event-stream" if stream else "application/json")
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    return req


class Handler(BaseHTTPRequestHandler):
    server_version = "CCUSGateway/1.0"
    # HTTP/1.0：流式响应不需要 Content-Length，靠连接关闭表示结束。
    # 用 1.1 就必须手写 chunked 编码，这里没必要引入那个出错面。
    protocol_version = "HTTP/1.0"

    # ─── 工具 ────────────────────────────────────────────
    def _send_json(self, status: int, body: dict) -> None:
        raw = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _error(self, status: int, message: str, err_type: str = "invalid_request_error",
               code: str = "") -> None:
        self._send_json(status, {"error": {"message": message, "type": err_type,
                                           "param": None, "code": code or None}})

    def _auth_ok(self) -> bool:
        expected = (os.getenv("GATEWAY_API_KEY") or "").strip()
        if not expected:
            return True          # 回环地址 + 未设 key = 不校验（与 magpie 一致）
        got = (self.headers.get("Authorization") or "").removeprefix("Bearer ").strip()
        return got == expected

    def log_message(self, fmt, *args):   # 默认实现会把每个请求打到 stderr，太吵
        vlog(f"{self.address_string()} {fmt % args}")

    # ─── GET /v1/models ──────────────────────────────────
    def do_GET(self):
        path = self.path.split("?", 1)[0].rstrip("/")
        if path in ("/health", "/healthz"):
            return self._send_json(200, {"status": "ok", "providers": self._available()})
        if path.endswith("/models"):
            return self._list_models()
        self._error(404, f"Unknown path: {path}", code="not_found")

    def _available(self) -> dict[str, str]:
        out = {}
        for provider in PROVIDERS:
            got = provider_base(provider)
            if got:
                out[provider] = got[0]
        return out

    def _list_models(self) -> None:
        """
        聚合各上游的模型列表，统一加 ``provider/`` 前缀。

        本地 Ollama 的列表是实时探测的（它知道装了哪些模型）；云端上游用
        ``/models`` 端点，失败就跳过 —— 网关的体检价值在于"能列出来"，
        某个上游临时不可用不应该让整个列表 500。
        """
        data: list[dict] = []
        for provider in PROVIDERS:
            got = provider_base(provider)
            if not got:
                continue
            base, key = got
            try:
                req = upstream_request(f"{base}/models", None, key, stream=False)
                with urllib.request.urlopen(req, timeout=10) as resp:
                    body = json.loads(resp.read().decode("utf-8", "replace"))
            except Exception as exc:
                vlog(f"  {provider} /models 不可用：{type(exc).__name__}: {exc}")
                continue
            items = body.get("data") or body.get("models") or []
            for it in items:
                mid = it.get("id") or it.get("name") if isinstance(it, dict) else None
                if mid:
                    data.append({"id": f"{provider}/{mid}", "object": "model",
                                 "owned_by": provider})
        self._send_json(200, {"object": "list", "data": data})

    # ─── POST /v1/chat/completions ───────────────────────
    def do_POST(self):
        path = self.path.split("?", 1)[0].rstrip("/")
        if not self._auth_ok():
            return self._error(401, "Invalid API key", "authentication_error")
        try:
            length = int(self.headers.get("Content-Length") or 0)
            payload = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
        except (ValueError, json.JSONDecodeError) as exc:
            return self._error(400, f"Malformed JSON body: {exc}")

        if path.endswith("/chat/completions"):
            return self._chat(payload)
        if path.endswith("/embeddings"):
            return self._passthrough(payload, "embeddings")
        self._error(404, f"Unknown path: {path}", code="not_found")

    def _resolve(self, payload: dict) -> tuple[str, str, str, str] | None:
        """解析出 (provider, upstream_model, base, key)；失败时已回写错误。"""
        model = (payload.get("model") or "").strip()
        if not model:
            self._error(400, "Missing 'model' field")
            return None
        provider, upstream_model = split_model(model)
        if not provider:
            self._error(
                400,
                f"Model '{model}' must be prefixed with a provider, e.g. "
                f"'ollama/qwen2.5:latest'. Known providers: "
                f"{', '.join(sorted(PROVIDERS))}.",
                code="model_not_found",
            )
            return None
        got = provider_base(provider)
        if not got:
            known = ", ".join(sorted(PROVIDERS))
            self._error(
                404,
                f"Model '{model}' not found: provider '{provider}' is unknown or has no "
                f"API key configured. Configured providers: "
                f"{', '.join(self._available()) or 'none'} (known: {known}).",
                code="model_not_found",
            )
            return None
        base, key = got
        return provider, upstream_model, base, key

    def _chat(self, payload: dict) -> None:
        resolved = self._resolve(payload)
        if not resolved:
            return
        provider, upstream_model, base, key = resolved
        payload = {**payload, "model": upstream_model}
        stream = bool(payload.get("stream"))

        # 少数本地上游不认 stream_options（OpenAI 的用法扩展），
        # 直接透传会让整个请求 400。应用侧本来就有"去掉重试"的兜底，
        # 但网关比应用更清楚上游是谁，所以在这里先摘掉更省一次往返。
        if stream and "stream_options" in payload and provider == "ollama":
            payload.pop("stream_options")

        url = f"{base}/chat/completions"
        t0 = time.time()
        req = upstream_request(url, payload, key, stream=stream)
        try:
            resp = urllib.request.urlopen(req, timeout=300)
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", "replace")
            log(f"  ✗ {provider}/{upstream_model} → HTTP {exc.code} ({time.time()-t0:.1f}s)")
            try:
                return self._send_json(exc.code, json.loads(raw))
            except json.JSONDecodeError:
                return self._error(exc.code, raw[:500], "upstream_error")
        except Exception as exc:
            log(f"  ✗ {provider}/{upstream_model} → 上游不可达：{type(exc).__name__}")
            return self._error(
                502,
                f"Upstream '{provider}' unreachable at {base}: {type(exc).__name__}: {exc}. "
                f"Is the service running?",
                "upstream_error",
            )

        with resp:
            if not stream:
                raw = resp.read()
                log(f"  ✓ {provider}/{upstream_model} → 200 ({time.time()-t0:.1f}s, "
                    f"{len(raw)}B)")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
                return

            # ── 流式：逐块转发，不攒 ──
            # 攒完再发就失去了流式的意义：前端要的是第一个 token 的到达时间。
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()
            chunks = 0
            try:
                while True:
                    block = resp.read(1024)
                    if not block:
                        break
                    self.wfile.write(block)
                    self.wfile.flush()
                    chunks += 1
            except (BrokenPipeError, ConnectionResetError):
                log(f"  ⚠ {provider}/{upstream_model} 客户端提前断开（{chunks} 块后）")
                return
            log(f"  ✓ {provider}/{upstream_model} → 200 流式 "
                f"({time.time()-t0:.1f}s, {chunks} 块)")

    def _passthrough(self, payload: dict, endpoint: str) -> None:
        """embeddings 等其它端点：解析 provider 后原样转发。"""
        resolved = self._resolve(payload)
        if not resolved:
            return
        provider, upstream_model, base, key = resolved
        payload = {**payload, "model": upstream_model}
        req = upstream_request(f"{base}/{endpoint}", payload, key, stream=False)
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as exc:
            return self._send_json(exc.code, json.loads(exc.read() or b"{}"))
        except Exception as exc:
            return self._error(502, f"Upstream '{provider}' unreachable: {exc}",
                               "upstream_error")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def main(argv: list[str] | None = None) -> int:
    global _VERBOSE
    ap = argparse.ArgumentParser(description="本地 OpenAI 兼容网关")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=3425)
    ap.add_argument("--verbose", action="store_true", help="打印每个请求")
    args = ap.parse_args(argv)
    _VERBOSE = args.verbose

    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(PROJECT_ROOT, ".env"), override=False)
    except Exception:
        pass

    # 回环 + 无 key = 本机开放；绑到别处必须设 key，否则等于把 LLM 额度
    # 匿名挂在网络上
    loopback = args.host in ("127.0.0.1", "localhost", "::1")
    if not loopback and not (os.getenv("GATEWAY_API_KEY") or "").strip():
        print("❌ 拒绝启动：绑定到非回环地址时必须设置 GATEWAY_API_KEY", file=sys.stderr)
        return 2

    available = {p: provider_base(p) for p in PROVIDERS}
    print("=" * 62)
    print(f"CCUS 本地网关  http://{args.host}:{args.port}/v1")
    print("=" * 62)
    for provider, got in available.items():
        print(f"  {provider:<9} → {got[0] if got else '（未配置 key，不可用）'}")
    print(f"  鉴权      : {'GATEWAY_API_KEY 已启用' if os.getenv('GATEWAY_API_KEY') else '不校验（仅回环）'}")
    print("=" * 62, flush=True)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n收到 Ctrl+C，关闭网关。")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
