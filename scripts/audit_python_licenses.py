#!/usr/bin/env python
"""Python 依赖的许可证审查。

与 `web/scripts/audit-licenses.mjs` 同一套判定规则，只是数据源不同：
Node 读 `node_modules/**/package.json`，Python 读包的协议元数据。

## 两种数据源，用途不同

| 模式 | 数据来源 | 用在哪 |
|---|---|---|
| 默认 | **已安装包**的 `importlib.metadata` | 本地。装上来的是哪个版本、它自己声明了什么，这是实际分发的东西 |
| `--from-pypi` | PyPI 的 JSON API | CI。`requirements.txt` 里有 torch 系依赖，装一遍要几个 GB；而 CI 需要的是"这个依赖树里有没有 AGPL"，不是"能不能跑起来" |

`--from-pypi` 取的是**最新版本**的元数据。这与 `requirements.txt` 里清一色
`>=` 的实际解析结果一致 —— 但也意味着它不会发现"某个旧版本是 AGPL、
新版本改成 MIT"这类历史问题。所以本地仍然应该跑默认模式。

## 为什么需要"例外登记"

`pymupdf` 是 AGPL-3.0 或 Artifex 商业授权（见 `docs/licenses.md` §3）。
直接阻断会让 CI 长期红着 —— 而一个长期红的检查很快就会被 `--no-verify`
绕过，那时它连"提醒"都不算了。

所以采用**显式例外**：把已接受的风险写进 `docs/license-exceptions.json`，
带理由、决定日期和**复审期限**。CI 的判定变成：

| 情况 | 结果 |
|---|---|
| 命中阻断，且已登记例外，未过期 | 通过（打印警告） |
| 命中阻断，未登记 | **失败** |
| 例外已过期 | **失败**（逼一次真实复审） |
| 登记的例外已经不再是阻断协议 | **失败**（过期条目要删掉，否则表会腐烂） |

这样"我知道这件事，我决定暂时接受，我在某个日期前会重新看"是一个
可以被机器检查的陈述，而不是一句口头承诺。

用法：

    python scripts/audit_python_licenses.py              # 本地：读已安装包
    python scripts/audit_python_licenses.py --from-pypi  # CI：读 PyPI 元数据
    python scripts/audit_python_licenses.py --json       # 机器可读

退出码：有未处置的问题 → 1；读取失败（网络/文件）→ 2。
"""

from __future__ import annotations

import argparse
import importlib.metadata as md
import json
import re
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REQUIREMENTS = ROOT / "requirements.txt"
EXCEPTIONS = ROOT / "docs" / "license-exceptions.json"

#: 阻断协议的特征词。**顺序有意义**：先排掉 AFFERO 与 LESSER，否则它们
#: 会被后面的 "GENERAL PUBLIC LICENSE" 吞掉，判定成同一个东西。
#:
#: 为什么不用 `^AGPL` 这种锚定写法：真实世界里的 license 字段不是 SPDX 标识符。
#: `pymupdf` 写的是 "Dual Licensed - GNU AFFERO GPL 3.0 or Artifex Commercial
#: License"，锚定 `^AGPL` 完全匹配不到 —— 于是这个审查脚本自己漏掉了
#: 唯一一个真正阻断的依赖，还反过来报"例外已失效"。锚定只对规范字段成立，
#: 而审查的对象恰恰是不规范的字段。
DENY_MARKERS: tuple[tuple[str, str], ...] = (
    ("AGPL", "AGPL"),
    ("AFFERO", "AGPL"),
    ("LGPL", "LGPL"),
    ("LESSER GENERAL PUBLIC", "LGPL"),
    ("SSPL", "SSPL"),
    ("SERVER SIDE PUBLIC", "SSPL"),
    ("EUPL", "EUPL"),
    ("EUROPEAN UNION PUBLIC", "EUPL"),
    ("OPEN SOFTWARE LICENSE", "OSL"),
    ("CPAL", "CPAL"),
    ("NONCOMMERCIAL", "非商用"),
    ("NON-COMMERCIAL", "非商用"),
    ("NON COMMERCIAL", "非商用"),
    ("GENERAL PUBLIC LICENSE", "GPL"),
    ("GPL", "GPL"),
    ("OSL", "OSL"),
    ("RPL", "RPL"),
)

ATTRIBUTION_MARKERS = ("CC-BY", "CREATIVE COMMONS ATTRIBUTION")


def classify(license_text: str) -> str:
    """返回 `deny` / `attribution` / `unknown` / `ok`。"""
    s = " ".join(license_text.upper().split())
    if not s or s in {"UNKNOWN", "NOT INSTALLED", "UNREADABLE"}:
        return "unknown"
    for marker, _kind in DENY_MARKERS:
        # 用词边界而不是 `in`：`GPL` 会命中 "GPLv3" 这类正常写法，
        # 但也会命中某些包名/描述里的子串。词边界把误伤面收窄。
        if re.search(rf"\b{re.escape(marker)}\b", s) or marker in s:
            return "deny"
    # CC-BY-NC：`NC` 这两个字母单独做词边界太容易误伤（"NC" 也可能是州名缩写），
    # 所以限定在 CC-BY 语境里判。
    if "CC-BY" in s and re.search(r"-NC\b", s):
        return "deny"
    if any(m in s for m in ATTRIBUTION_MARKERS):
        return "attribution"
    return "ok"


#: 没有 license 字段、但只在 classifier 里写协议的包。classifier 的写法是
#: `License :: OSI Approved :: MIT License`，取最后一段。
CLASSIFIER_PREFIX = "License ::"


def requirement_names() -> list[str]:
    """从 requirements.txt 取包名（去掉版本约束、extras、注释）。"""
    names: list[str] = []
    for raw in REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        name = re.split(r"[<>=!~\[;]", line, maxsplit=1)[0].strip()
        if name:
            names.append(name)
    return names


def license_of(meta: md.PackageMetadata) -> str:
    """取协议标识。优先 License-Expression（PEP 639 的规范字段）。"""
    expr = meta.get("License-Expression")
    if expr:
        return " ".join(expr.split())
    raw = meta.get("License")
    if raw:
        # 有些包的 License 字段是整篇协议正文。只取第一行，并截断 ——
        # 把正文塞进判定会让正则匹配到"本协议不适用于..."这种句子。
        first = " ".join(raw.split())[:120]
        return first
    for cls in meta.get_all("Classifier") or []:
        if cls.startswith(CLASSIFIER_PREFIX):
            return cls.rsplit("::", 1)[-1].strip()
    return "UNKNOWN"


def load_exceptions() -> dict[str, dict]:
    if not EXCEPTIONS.exists():
        return {}
    data = json.loads(EXCEPTIONS.read_text(encoding="utf-8"))
    return {e["package"].lower(): e for e in data.get("exceptions", [])}


def metadata_from_pypi(name: str) -> dict | None:
    """从 PyPI JSON API 取协议元数据。不需要安装。

    只取判定需要的三个字段：版本、license 表达式、classifier 里的协议行。
    """
    import urllib.error
    import urllib.request

    url = f"https://pypi.org/pypi/{name}/json"
    try:
        with urllib.request.urlopen(url, timeout=20) as resp:
            info = json.loads(resp.read().decode("utf-8"))["info"]
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise
    expr = info.get("license_expression") or info.get("license") or ""
    for cls in info.get("classifiers") or []:
        if cls.startswith(CLASSIFIER_PREFIX):
            # classifier 通常比 license 字段更规范，两者都看：取更"像协议名"的那个
            cls_value = cls.rsplit("::", 1)[-1].strip()
            if not expr or len(expr) > 80:
                expr = cls_value
            break
    return {"version": info.get("version"), "license": " ".join(str(expr).split())[:120] or "UNKNOWN"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    ap.add_argument(
        "--from-pypi",
        action="store_true",
        help="读 PyPI 元数据而不是已安装包（CI 用：不需要装 torch 系依赖）",
    )
    args = ap.parse_args()

    exceptions = load_exceptions()
    rows: list[dict] = []
    for name in requirement_names():
        if args.from_pypi:
            try:
                info = metadata_from_pypi(name)
            except Exception as e:  # 网络问题不该被当成"协议没问题"
                print(f"  ✗ 读取 {name} 的 PyPI 元数据失败：{e}", file=sys.stderr)
                return 2
            if info is None:
                rows.append({"package": name, "version": None, "license": "NOT INSTALLED"})
            else:
                rows.append({"package": name, **info})
            continue
        try:
            meta = md.metadata(name)
        except md.PackageNotFoundError:
            rows.append({"package": name, "version": None, "license": "NOT INSTALLED"})
            continue
        rows.append(
            {
                "package": name,
                "version": meta.get("Version"),
                "license": license_of(meta),
            }
        )

    for r in rows:
        r["verdict"] = classify(r["license"])
    denied = [r for r in rows if r["verdict"] == "deny"]
    attribution = [r for r in rows if r["verdict"] == "attribution"]
    unknown = [r for r in rows if r["verdict"] == "unknown" and r["license"] != "NOT INSTALLED"]
    missing = [r for r in rows if r["license"] == "NOT INSTALLED"]

    problems: list[str] = []
    accepted: list[dict] = []
    for r in denied:
        exc = exceptions.get(r["package"].lower())
        if not exc:
            problems.append(f"{r['package']} {r['version']} 是阻断协议 {r['license']!r}，且未登记例外")
            continue
        review_by = exc.get("review_by")
        if review_by:
            try:
                if date.fromisoformat(review_by) < date.today():
                    problems.append(f"{r['package']} 的例外已于 {review_by} 过期，需要重新复审")
                    continue
            except ValueError:
                problems.append(f"{r['package']} 的例外里 review_by 不是 ISO 日期：{review_by!r}")
                continue
        accepted.append(r)

    # 例外表腐烂检查：登记了但现在已经不再是阻断协议的条目，必须删掉。
    denied_names = {r["package"].lower() for r in denied}
    for name in exceptions:
        if name not in denied_names:
            problems.append(f"例外 {name!r} 已失效（它现在不是阻断协议，或已不在依赖里），请删除该条目")

    if args.json:
        print(
            json.dumps(
                {
                    "total": len(rows),
                    "denied": [f"{r['package']} {r['version']} → {r['license']}" for r in denied],
                    "accepted": [f"{r['package']} {r['version']} → {r['license']}" for r in accepted],
                    "attribution": [f"{r['package']} {r['version']} → {r['license']}" for r in attribution],
                    "unknown": [r["package"] for r in unknown],
                    "not_installed": [r["package"] for r in missing],
                    "problems": problems,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 1 if problems else 0

    print(f"\n  审查 {len(rows)} 个 Python 依赖\n")
    for r in rows:
        flag = ""
        if r["verdict"] == "deny":
            flag = "  ⚠ 已登记例外" if r in accepted else "  ✗ 阻断"
        elif r["verdict"] == "attribution":
            flag = "  · 需署名"
        print(f"  {r['package']:26} {(r['version'] or '-'):12} {r['license'][:46]:46}{flag}")

    if attribution:
        print("\n  需要保留署名：")
        for r in attribution:
            print(f"    · {r['package']} {r['version']} → {r['license']}")

    if unknown:
        print("\n  ⚠ 协议未知（元数据里没有 license 字段）：")
        for r in unknown:
            print(f"    · {r['package']} {r['version']}")

    if missing:
        print(f"\n  · {len(missing)} 个包未安装，已跳过：{', '.join(r['package'] for r in missing)}")
        print("    （在 CI 里这一步跑在装好 requirements.txt 的 job 中，不应该有跳过项）")

    if accepted:
        print("\n  已登记的例外：")
        for r in accepted:
            exc = exceptions[r["package"].lower()]
            print(f"    · {r['package']} {r['version']} → {r['license']}")
            print(f"      范围：{exc.get('scope', '（未写）')}")
            print(f"      复审期限：{exc.get('review_by', '（未写）')}")

    if problems:
        print("\n  ✗ 存在未处置的问题：")
        for p in problems:
            print(f"    · {p}")
        print("")
        return 1

    print(f"\n  ✓ {len(rows)} 个依赖，{len(accepted)} 个已登记例外，无未处置的阻断协议\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
