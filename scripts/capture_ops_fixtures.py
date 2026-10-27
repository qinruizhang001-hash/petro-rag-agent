"""把 `/api/ops/*` 的真实响应抓成前端 fixture。

为什么不用手写 JSON
------------------
手写的 fixture 会和契约悄悄分家。后端字段改名、枚举多一个值、错误形状从
`{"detail": "..."}` 变成 `{"detail": {"message": ...}}` —— 这些都不会让任何
东西变红，前端只是继续渲染那份"看起来还对"的旧数据。等到真接上后端才发现，
而那时改动已经散在十几个组件里。

所以这里起一个**只挂 ops 路由的裸 FastAPI**，用 `TestClient` 真发请求，
把响应原样落盘。fixture 因此永远等于接口的实际输出；接口一变，重跑一次
脚本就能看到 diff。

为什么不导入 `api_server`
------------------------
它顶部 `from workflow import ...` 会建整张 LangGraph 并加载 2.2 GB 重排模型。
抓 fixture 不需要那些，也不该需要 —— 这正是 `api_ops` 单独成模块的理由。

场景是**构造的，阈值不是**
------------------------
读数是为演示造的（要让六个状态都出现），但阈值直接来自
`config/ops_thresholds.example.json`，一个数字都没有改动，`source` 里
也保留着 `【替换】` 前缀 —— 界面上会照原样显示"这还不是你项目的真实阈值"。
给示例阈值换一个像样的出处，才是这个项目最不该做的事。

用法::

    python scripts/capture_ops_fixtures.py            # 写入 web/src/fixtures/
    python scripts/capture_ops_fixtures.py --check    # 只比对不写（给 CI 用）
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT = ROOT / "web" / "src" / "fixtures"

#: 抓取基准时刻。固定而不是 `now()`：fixture 要能逐字节比对，
#: 每次跑出不同时间戳的 fixture 只能靠人眼确认，等于没有检查。
T0 = datetime(2026, 10, 27, 6, 0)

#: 故意留三个 MVA 浅层测点不配阈值。
#:
#: 四个浅层 MVA 测点里三个盲，是这个界面最该演示的状态：它读起来像
#: "监测正常"，实际是"三个测点根本没在判定"。只挑一个盲测点演示不出这个效果 ——
#: 一个孤零零的破折号会被当成偶发缺数，三个连成一片才会让人去问为什么。
BLIND = ("soil_co2_flux", "gw_conductivity", "surface_heave")

#: 每个测点的取值曲线。键是测点名，值是一串 25 个点（4 小时 × 10 分钟）。
#: 只写会动的和需要出现的状态，其余测点用 `_flat` 补一条平稳序列。
SERIES: dict[str, list[float]] = {
    # 环空压力：20.2 → 41.6 bar。越过 critical(40)，同时命中 R_PERSIST
    # （连续 3 点在 alert 以上）与 R_RAPID（1 小时内跨级）—— 三条规则各留一行判据。
    "annulus_p": [
        20.2, 20.4, 20.6, 21.0, 21.6, 22.4, 23.5, 24.8,
        26.3, 28.0, 29.8, 31.5, 33.0, 34.2, 35.2, 36.0,
        36.8, 37.5, 38.2, 38.8, 39.4, 40.0, 40.6, 41.2, 41.6,
    ],
    # 管道压降：全程 12.4 → 12.7 bar，**不越 warn(15)**。
    #
    # 这一条刻意不动，是为了让"管道阻力上升"那条归因被排除掉：
    # 出口压力在涨、压降却是平的，说明阻力不在管段内部，而是下游背压。
    # 界面上必须以删除线保留这条被排除的原因 —— 读者要看到"我们考虑过
    # 管道结垢、因为压降没变而排除了它"，否则会怀疑是漏了。
    "pipeline_dp": [
        12.40, 12.41, 12.42, 12.44, 12.45, 12.46, 12.48, 12.50,
        12.51, 12.52, 12.54, 12.55, 12.56, 12.58, 12.60, 12.61,
        12.62, 12.64, 12.65, 12.66, 12.68, 12.70, 12.71, 12.72, 12.74,
    ],
    # 管道出口压力：78.5 → 115.4 bar，越过 alert(110)。
    # 单独看这是"管输压力异常"，但压降没动 —— 见上一条。
    "pipeline_outlet_p": [
        78.5, 79.1, 79.8, 80.6, 81.5, 82.4, 83.4, 84.5,
        85.6, 86.8, 88.0, 89.3, 90.6, 91.9, 93.2, 94.6,
        96.0, 97.4, 98.8, 100.2, 101.6, 103.0, 104.5, 110.0, 115.4,
    ],
    # 捕集率（越低越坏）：93.2 → 86.8 %，越过 warn(88) 但没到 alert(82)。
    "capture_rate": [
        93.2, 93.1, 93.0, 92.9, 92.7, 92.5, 92.3, 92.0,
        91.7, 91.4, 91.1, 90.7, 90.3, 89.9, 89.6, 89.3,
        89.0, 88.7, 88.4, 88.2, 87.9, 87.6, 87.3, 87.0, 86.8,
    ],
    # 水露点：全程 4.6 ℃ 平稳，**只在最后一个点**抬到 5.4 —— 越过 warn(5)
    # 一个点，不满足 R_PERSIST 的连续 3 点，因此停在 warn 而不被升级。
    # "刚越线一次"与"已经在线上待了三刻钟"是两件不同的事，
    # 界面上必须能从判据里读出来，而不是只看颜色。
    "h2o_dewpoint": [4.6] * 24 + [5.4],
    # 储层压力：268 → 296.3 bar。warn=300、alert=340 → 提示带宽 4，
    # 提示线 296。压到 296.3 就是"还有 3.7 bar 就要报警"，
    # 这一级不该叫醒任何人，但也不该等到越线才第一次出现在看板上。
    "reservoir_p": [round(268 + i * 1.18, 2) for i in range(25)],
    # 井底压力 = 储层压力 + 注入压差。注入压差由注入指数反算：
    # 236 t/d ÷ 5.9 t/(d·bar) ≈ 40 bar。所以 bhp 跟着 reservoir_p 一起抬，
    # 308 → 336 bar。**不独立编一条曲线**：三个数字互相矛盾的话，
    # 界面上"注入指数正常"与"井底压力异常"会同时成立，而没人能看出哪条错。
    "bhp": [round(268 + i * 1.18 + 40.0, 2) for i in range(25)],
    # 破裂压力只有一个点：它来自 LOT/XLOT，是**周期性作业结果**，
    # 不是逐时读数。给它编 25 个点会让"破裂压力每 10 分钟测一次"这种
    # 不可能的事变成 fixture 的既成事实，而下游没人会去质疑。
    "frac_p": [452.0],
}

#: 平稳序列的基准值。这些测点全程不动，因此全部落在 normal ——
#: 一张只有告警的看板看不出"大部分是好的"，而没有这一半，
#: 读者无法判断系统是在正常工作还是在集体失联。
FLAT: dict[str, float] = {
    "regenerator_duty": 2.62,
    "solvent_loss": 0.94,
    "pipeline_outlet_p": 78.5,
    "co2_purity": 98.6,
    "h2o_dewpoint": -12.4,
    "wellhead_p": 74.2,
    "injection_rate": 236.0,
    "injectivity_index": 5.9,
    "reservoir_p": 268.0,
    "soil_co2_flux": 1.35,
    "gw_ph": 7.12,
    "gw_conductivity": 412.0,
    "surface_heave": 1.8,
}

POINTS = 25
STEP = timedelta(minutes=10)


def _flat(value: float, n: int = POINTS) -> list[float]:
    """平稳序列加一点点抖动。

    不加抖动会让"没有变化"和"传感器卡死在同一个读数"看起来完全一样，
    而后者是真实故障。抖动只影响可读性，不影响判定（幅度远小于任何阈值间隙）。
    """
    return [round(value + ((i % 5) - 2) * 0.004 * abs(value), 4) for i in range(n)]


def build_readings() -> list[dict]:
    """16 个测点各 25 个点。

    遍历的是 ``FLAT ∪ SERIES`` 而不是 ``FLAT`` —— 第一版只遍历了 FLAT，
    于是三条"会动"的测点（环空压力、管道压降、捕集率）**根本没进 fixture**，
    而它们正是唯一会越限的三条。界面上于是只剩一片 normal 加三个盲区，
    看起来像是"系统工作正常"，实际是抓数据的时候漏了。
    """
    rows: list[dict] = []
    for sensor in dict.fromkeys([*FLAT, *SERIES]):
        series = SERIES.get(sensor) or _flat(FLAT[sensor])
        for i, v in enumerate(series):
            rows.append(
                {
                    "sensor": sensor,
                    "value": v,
                    "ts": (T0 + i * STEP).isoformat(),
                    "quality": "good",
                }
            )
    return rows


def build_config() -> dict:
    """示例阈值表，删掉三个浅层 MVA 测点。"""
    src = ROOT / "config" / "ops_thresholds.example.json"
    cfg = json.loads(src.read_text(encoding="utf-8"))
    for key in BLIND:
        cfg.pop(key, None)
    cfg["_说明"] = (
        "由 scripts/capture_ops_fixtures.py 从 ops_thresholds.example.json 派生，"
        "删掉了三个浅层 MVA 测点以演示盲区状态。数字仍是示例值，未改。"
    )
    return cfg


def _client():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    import api_ops

    app = FastAPI()
    app.include_router(api_ops.router)
    return TestClient(app)


#: 抓取时用的是临时配置文件，路径每次都不一样。原样落盘会让 fixture 无法逐字节
#: 比对（`--check` 永远报"有变化"，然后就没人看了），更要紧的是这个路径会
#: **直接渲染到界面上** —— 一行 `C:\Users\...\Temp\ops-fixture-xxxx\...` 出现在
#: 阈值来源栏里，读者第一反应是系统出错了。
#:
#: 换成一句如实的说明：数字来自示例表，且少了三个测点。它比路径更有用。
_ORIGIN_REWRITES = {
    "origin": "config/ops_thresholds.example.json（派生：删去 3 个浅层 MVA 测点）",
    "threshold_origin": "config/ops_thresholds.example.json（派生：删去 3 个浅层 MVA 测点）",
}


def _strip_volatile(payload):
    """去掉每次跑都会变的字段，让 fixture 可逐字节比对。"""
    if isinstance(payload, dict):
        return {
            k: (
                _ORIGIN_REWRITES[k]
                if k in _ORIGIN_REWRITES
                else _strip_volatile(v)
            )
            for k, v in payload.items()
            if k != "generated_at"
        }
    if isinstance(payload, list):
        return [_strip_volatile(v) for v in payload]
    return payload


def capture() -> dict[str, object]:
    readings = build_readings()
    period = {"start": T0.isoformat(), "end": (T0 + (POINTS - 1) * STEP).isoformat()}

    tmp = Path(tempfile.mkdtemp(prefix="ops-fixture-")) / "ops_thresholds.json"
    tmp.write_text(json.dumps(build_config(), ensure_ascii=False), encoding="utf-8")

    old = os.environ.get("OPS_THRESHOLDS_PATH")
    os.environ["OPS_THRESHOLDS_PATH"] = str(tmp)
    try:
        client = _client()

        thresholds = client.get("/api/ops/thresholds")
        thresholds.raise_for_status()

        alarms = client.post(
            "/api/ops/alarms", json={"readings": readings, "period": period}
        )
        alarms.raise_for_status()

        # 带一次工况问答：它的 `unsourced_numbers` 是"守卫是否触发"的
        # 唯一外部证据，前端要能显示这个字段，所以 fixture 里得有。
        condition = client.post(
            "/api/ops/condition",
            json={
                "question": "过去 4 小时井筒和管道有什么异常",
                "readings": readings,
                "period": period,
                "now": (T0 + (POINTS - 1) * STEP).isoformat(),
            },
        )
        condition.raise_for_status()

        return {
            "thresholds.json": _strip_volatile(thresholds.json()),
            "alarms.json": _strip_volatile(alarms.json()),
            "condition.json": _strip_volatile(condition.json()),
        }
    finally:
        if old is None:
            os.environ.pop("OPS_THRESHOLDS_PATH", None)
        else:
            os.environ["OPS_THRESHOLDS_PATH"] = old


def _dump(payload) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


def _first_diff(expected: str, actual: str, limit: int = 14) -> str:
    """定位第一处差异，带上下文。

    只报"内容不一致"等于让下一个人重跑一遍去猜。开发机与 CI 的差别可能只是
    时区、Python 版本、或某个浮点末位 —— 把行号与两边的值打出来，几秒钟就能
    判断这是"真的漂移"还是"环境不同"，而这两者的处置完全相反。
    """
    import difflib

    out: list[str] = []
    for line in difflib.unified_diff(
        expected.splitlines(),
        actual.splitlines(),
        fromfile="已提交的 fixture",
        tofile="接口实际输出",
        n=2,
        lineterm="",
    ):
        out.append(line)
        if len(out) >= limit:
            out.append("…（只显示前几处差异）")
            break
    return "\n".join(out)


def main(argv: list[str]) -> int:
    check_only = "--check" in argv[1:]
    payloads = capture()

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
                        f"{name}：内容与接口实际输出不一致\n"
                        + _first_diff(on_disk, text)
                    )
            continue
        target.write_text(text, encoding="utf-8")
        print(f"写入 {target.relative_to(ROOT)}  ({len(text):,} 字节)")

    if check_only:
        if stale:
            raise SystemExit(
                "::error::前端 fixture 与 /api/ops/* 的实际输出不一致：\n  - "
                + "\n  - ".join(stale)
                + "\n修法：python scripts/capture_ops_fixtures.py，然后看 git diff。"
                "\n不要手改 fixture —— 它存在的意义就是不被手改。"
            )
        print(f"fixture 一致：{', '.join(sorted(payloads))}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
