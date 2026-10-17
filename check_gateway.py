"""
LLM 网关体检（Gateway Doctor）
==============================
检查 CLOUD_BASE_URL 指向的 OpenAI 兼容端点是否可用，并核对路由里配置的模型
是否真的存在于该端点。

为什么需要这个脚本
------------------
把 CLOUD_BASE_URL 指向本地聚合网关（magpie / one-api / LiteLLM 等）后，
出问题时有三类完全不同的原因，而它们的报错长得很像：

  1. 网关没启动        → Connection refused
  2. 代理劫持了回环请求 → **502 Bad Gateway**（最容易被误判成"网关挂了"）
  3. 模型名对不上      → 404 / model_not_found（网关统一用 provider/model 命名）
  4. Key 不对          → 401

本脚本逐项分辨这四种情况，并在失败时给出可直接照做的修复命令。

用法:
    python check_gateway.py                # 检查 CLOUD_BASE_URL
    python check_gateway.py --local        # 检查本地 Ollama
    python check_gateway.py --send         # 额外发一次极小的真实请求（花 token）

退出码: 0 全部通过 / 1 有失败项（可接入 CI）
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def bypass_loopback_proxy() -> list[str]:
    """
    让回环地址绕过系统代理，并返回被清掉的代理变量（供报告展示）。

    若不清，http_proxy 会把 127.0.0.1:3425 的请求转给本地代理；目标端口没人
    监听时代理返回 502，看起来像网关自己坏了 —— 这个坑本项目踩过两次。
    """
    removed = []
    for var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY",
                "all_proxy", "ALL_PROXY"):
        if os.environ.pop(var, None) is not None:
            removed.append(var)
    loopback = ["localhost", "127.0.0.1", "::1"]
    for var in ("NO_PROXY", "no_proxy"):
        existing = [p.strip() for p in (os.getenv(var) or "").split(",") if p.strip()]
        os.environ[var] = ",".join(existing + [h for h in loopback if h not in existing])
    return removed


PASS, FAIL, WARN = [], [], []


def check(name: str, cond: bool, detail: str = "", hint: str = "") -> bool:
    if cond:
        print(f"  [PASS] {name}" + (f"  ({detail})" if detail else ""))
        PASS.append(name)
    else:
        print(f"  [FAIL] {name}" + (f"  ({detail})" if detail else ""))
        if hint:
            for line in hint.strip().splitlines():
                print(f"         → {line}")
        FAIL.append(name)
    return cond


def warn(name: str, detail: str = "") -> None:
    print(f"  [WARN] {name}" + (f"  ({detail})" if detail else ""))
    WARN.append(name)


def http_json(url: str, payload: dict | None = None, api_key: str = "",
              timeout: int = 20) -> tuple[int, object]:
    """发一个 JSON 请求，返回 (status, body)。不抛异常，交由调用方判定。"""
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method="POST" if data else "GET")
    req.add_header("Content-Type", "application/json")
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode("utf-8", "replace")
            try:
                return r.status, json.loads(raw)
            except json.JSONDecodeError:
                return r.status, raw
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", "replace")
        try:
            return e.code, json.loads(raw)
        except json.JSONDecodeError:
            return e.code, raw
    except urllib.error.URLError as e:
        return 0, str(e.reason)
    except Exception as e:  # 超时等
        return 0, str(e)


def main() -> int:
    ap = argparse.ArgumentParser(description="LLM 网关体检")
    ap.add_argument("--local", action="store_true", help="改查本地 Ollama 端点")
    ap.add_argument("--send", action="store_true", help="额外发一次极小的真实请求")
    args = ap.parse_args()

    # 读 .env（与主程序一致），但不覆盖已存在的真实环境变量
    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(PROJECT_ROOT, ".env"), override=False)
    except Exception:
        pass

    removed = bypass_loopback_proxy()

    if args.local:
        base = (os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434/v1").rstrip("/")
        key = os.getenv("LOCAL_API_KEY", "ollama")
        label = "本地 Ollama"
    else:
        base = (os.getenv("CLOUD_BASE_URL") or "https://api.deepseek.com/v1").rstrip("/")
        key = (os.getenv("CLOUD_API_KEY") or "").strip()
        label = "云端 / 网关"

    print("=" * 62)
    print(f"LLM 网关体检 — {label}")
    print("=" * 62)
    print(f"  Base URL : {base}")

    is_loopback = any(h in base for h in ("127.0.0.1", "localhost", "::1"))
    if is_loopback:
        print(f"  回环地址 : 是（已清除代理变量: {', '.join(removed) if removed else '无'}）")
    elif removed:
        print(f"  代理变量 : 保留 {', '.join(removed)}（非回环地址，需要走代理）")

    print()
    print("[1] 端点可达性")
    status, body = http_json(f"{base}/models", api_key=key)

    if status == 0:
        reason = str(body)
        is_service_absent = "拒绝" in reason or "refused" in reason.lower() or "10061" in reason
        if is_loopback:
            check("端点可达", False, reason,
                  hint="网关没在跑。启动方式取决于你用的聚合器：\n"
                       "  magpie:  magpie.exe（启动 App 后网关自动监听 127.0.0.1:3425）\n"
                       "  或把 CLOUD_BASE_URL 改回云厂商直连地址")
        else:
            check("端点可达", False, reason,
                  hint="检查网络/防火墙；若必须走代理，确认 https_proxy 已设置")
        return summarize()
    if status == 502:
        check("端点可达", False, f"HTTP 502（这是**代理**返回的，不是网关）",
              hint="仍有代理在拦回环请求。清掉环境变量后重试：\n"
                   "  Windows: set http_proxy= && set https_proxy=\n"
                   "  bash   : unset http_proxy https_proxy ALL_PROXY")
        return summarize()
    if status in (401, 403):
        check("端点可达", True, f"HTTP {status}（服务在跑，只是鉴权没过）")
        check("鉴权通过", False, f"HTTP {status}",
              hint="本地网关通常接受任意 key（填 magpie 即可）；\n"
                   "云厂商直连则需要真实 CLOUD_API_KEY")
        return summarize()

    check("端点可达", status == 200, f"HTTP {status}")

    print()
    print("[2] 模型列表")
    model_ids: list[str] = []
    if isinstance(body, dict):
        items = body.get("data") or body.get("models") or []
        for it in items:
            if isinstance(it, dict):
                mid = it.get("id") or it.get("name") or it.get("model")
                if mid:
                    model_ids.append(str(mid))
    check("拿到了模型列表", bool(model_ids), f"{len(model_ids)} 个模型")

    if model_ids:
        preview = model_ids[:12]
        print(f"         可用模型（前 {len(preview)} 个）: {', '.join(preview)}")
        if len(model_ids) > len(preview):
            print(f"         …还有 {len(model_ids) - len(preview)} 个")

    print()
    print("[3] 路由表与模型名是否对得上")
    routes = {}
    for stage, default in (
        ("extract", "local"), ("report", "cloud"), ("evaluate", "local"),
        ("history", "local"), ("summary", "local"),
    ):
        routes[stage] = (os.getenv(f"ROUTE_{stage.upper()}") or "").strip() or default

    default_model = (os.getenv("LOCAL_MODEL", "qwen2.5") if args.local
                     else os.getenv("CLOUD_MODEL", "deepseek-chat"))

    for stage, spec in routes.items():
        tier = spec.split(":", 1)[0].strip().lower()
        wanted = spec.split(":", 1)[1].strip() if ":" in spec else default_model
        applies = (tier == "local") if args.local else (tier == "cloud")
        if not applies:
            print(f"         {stage:<9} → {spec}  （不在本次检查的档位，跳过）")
            continue
        if not model_ids:
            warn(f"{stage} 模型 {wanted}", "无模型列表可比对")
            continue
        hit = wanted in model_ids
        detail = f"{wanted}" + ("" if hit else "  不在列表中")
        # 模型名对不上不算硬失败：部分网关的 /models 是静态白名单，
        # 但真实调用仍可能成功。标 WARN，并在 --send 时用真实请求确认。
        if hit:
            check(f"{stage} 模型存在", True, detail)
        else:
            warn(f"{stage} 模型 {wanted}", "不在 /models 列表；若真实调用成功则忽略")

    if args.send:
        print()
        print("[4] 真实请求（极小）")
        target = None
        for stage, spec in routes.items():
            tier = spec.split(":", 1)[0].strip().lower()
            if (tier == "local") == args.local:
                target = spec.split(":", 1)[1].strip() if ":" in spec else default_model
                break
        target = target or default_model
        print(f"         使用模型: {target}")
        st, bd = http_json(
            f"{base}/chat/completions",
            payload={"model": target,
                     "messages": [{"role": "user", "content": "回复 OK 两个字符"}],
                     "max_tokens": 8},
            api_key=key, timeout=90,
        )
        ok = st == 200 and isinstance(bd, dict) and bd.get("choices")
        check("真实调用成功", bool(ok),
              f"HTTP {st}" if ok else f"HTTP {st}: {str(bd)[:180]}",
              hint="模型名不存在时，用 [2] 里的名字替换 CLOUD_MODEL / ROUTE_*")
        if ok:
            txt = (bd["choices"][0].get("message") or {}).get("content", "")
            print(f"         返回: {txt.strip()[:60]!r}")

    return summarize()


def summarize() -> int:
    print()
    print("=" * 62)
    print(f"结果：{len(PASS)} PASS / {len(WARN)} WARN / {len(FAIL)} FAIL")
    if FAIL:
        print("失败项：" + "；".join(FAIL))
        print("=" * 62)
        return 1
    print("✅ 网关可用")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    sys.exit(main())
