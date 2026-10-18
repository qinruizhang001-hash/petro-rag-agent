"""
并行扇出 / 扇入的实测基准
=========================
跑**一次**完整工作流，把并行收益打出来：

    python bench_parallel.py
    python bench_parallel.py --input "某CCUS项目注入压力300bar，温度85℃，存在断层。"

为什么要单独一个脚本，而不是只看 E2E 的总耗时
--------------------------------------------
E2E 的总耗时里混着三次检索、两次工作流、API 服务启停、模型冷加载……
任何一项波动都能盖过并行带来的那几秒，用它判断"扇出到底有没有生效"等于
用体温计量身高。这里只做一件事：跑一次工作流，直接读状态里的
``parallel_timing``，那是两条分支自己打的点。

注意：本脚本会真实写 ``project_history.md``（工作流的最后一环）。
反复跑之前建议 ``git checkout -- project_history.md``。
"""
from __future__ import annotations

import argparse
import time

import core  # noqa: F401  必须最先：环境预置（HF_HUB_OFFLINE / NO_PROXY 等）

DEFAULT_INPUT = "某CCUS项目注入压力250bar，温度80℃，存在隐伏碳盐酸岩储层断层，请评估风险。"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=DEFAULT_INPUT)
    ap.add_argument("--thread", default="")
    args = ap.parse_args()

    from workflow import app  # 延迟导入：上面先跑 core 的环境预置

    thread_id = args.thread or f"bench_{int(time.time())}"
    config = {"configurable": {"thread_id": thread_id}}

    print(f"\n输入：{args.input}")
    print(f"thread：{thread_id}\n")

    t0 = time.perf_counter()
    state = app.invoke({"user_input": args.input}, config)
    wall = time.perf_counter() - t0

    timing = state.get("parallel_timing") or {}
    geo = state.get("geology_status") or {}
    well = state.get("wellbore_status") or {}
    merge = state.get("branch_merge") or {}

    print("=" * 66)
    print("分支执行情况")
    print("=" * 66)
    print(f"  geology   ok={geo.get('ok')}  {geo.get('elapsed_s')}s  "
          f"{geo.get('chars')} 字  engine={geo.get('engine')}"
          + (f"  error={geo.get('error')}" if geo.get("error") else ""))
    print(f"  wellbore  ok={well.get('ok')}  {well.get('elapsed_s')}s  "
          f"{well.get('chars')} 字  engine={well.get('engine')}"
          + (f"  error={well.get('error')}" if well.get("error") else ""))

    print("\n" + "=" * 66)
    print("并行收益（两条分支自打时间戳，不是按节点名推算）")
    print("=" * 66)
    print(f"  地质分支      {timing.get('geology_s', 0):>7.2f} s")
    print(f"  井筒分支      {timing.get('wellbore_s', 0):>7.2f} s")
    print(f"  串行需        {timing.get('serial_s', 0):>7.2f} s   （两者相加）")
    print(f"  并行实际      {timing.get('wall_s', 0):>7.2f} s   （扇出墙钟）")
    print(f"  重叠          {timing.get('overlap_s', 0):>7.2f} s")
    print(f"  加速比        {timing.get('speedup', 1.0):>7.2f} ×")
    print(f"  汇总节点      {timing.get('aggregate_s', 0):>7.2f} s")
    print(f"  扇出+汇总     {timing.get('total_s', 0):>7.2f} s")
    print(f"  整条工作流    {wall:>7.2f} s   （含记忆/提取/评审/闸门/落盘）")

    print("\n" + "=" * 66)
    print("降级情况")
    print("=" * 66)
    print(f"  degraded={merge.get('degraded')}  missing={merge.get('missing')}  "
          f"reasons={merge.get('reasons')}")
    if merge.get("degraded"):
        print("  ⚠️ 报告末尾应含「数据完整性提示」："
              f"{'已包含' if '数据完整性提示' in (state.get('final_report') or '') else '缺失！'}")

    print("\n" + "=" * 66)
    print(f"报告 {len(state.get('final_report') or '')} 字 / 评审 {state.get('score')} 分")
    print("=" * 66)
    print((state.get("final_report") or "")[:400])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
