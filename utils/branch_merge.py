"""
并行分支的合并 / 降级 / 计时（纯函数）
=====================================
扇入节点（``nodes/aggregate_node.py``）里最容易出错、也最需要被测的三件事，
全部放在这里：

  1. **哪条分支缺了** —— ``compose_branch_inputs``
  2. **缺了以后报告里写什么** —— ``degradation_block``
  3. **并行到底省了多少** —— ``parallel_timing_of``

为什么抽出来
------------
``nodes/aggregate_node.py`` 一 import 就会拉起 matplotlib、openai、reranker，
而 CI 的 ``unit-core`` 一个三方包都不装。纯逻辑留在节点里 = 这四条
"缺地质 / 缺井筒 / 全缺 / 全齐"的组合永远没有自动化测试，只能靠人工跑
一次 E2E 看它有没有崩。抽到这里之后，``tests/test_branch_merge.py``
可以在零依赖环境下把它们穷举完。

另一个理由：这三个函数**不应该**碰模型。降级文案一旦交给 LLM 生成，
就变成了"希望它照做"；放在这里意味着它是代码，不是祈祷。

不抛异常
--------
本模块所有函数对任何输入都要能返回结果（``state`` 里字段缺失、类型不对、
值为 None 都算合法输入）—— 因为它们正是"出问题时才被调用"的那段代码。
"""
from __future__ import annotations

# 降级文案：必须与需求逐字一致（测试会断言），改之前先想清楚
GEOLOGY_MISSING_TEXT = "地质模块未能提供可用证据，请人工复核"
WELLBORE_MISSING_TEXT = "井筒模块未能提供可用证据，请人工复核"


def _text_of(state: dict, key: str) -> str:
    value = (state or {}).get(key)
    return value.strip() if isinstance(value, str) else ""


def _status_of(state: dict, key: str) -> dict:
    value = (state or {}).get(key)
    return value if isinstance(value, dict) else {}


def compose_branch_inputs(state: dict) -> dict:
    """整理两条分支的产物，并判定是否降级。

    返回::

        {
          "geology": 文本（缺失时是降级文案，**绝不是空串**）,
          "wellbore": 文本,
          "geology_ok": bool,
          "wellbore_ok": bool,
          "missing": ["geology"],                        # 缺失分支名
          "degraded": bool,
          "reasons": {"geology": "TimeoutError: ..."},    # 来自 *_status.error
        }

    注意 ``geology`` 字段的语义是"可以放进 Prompt 的文本"，不是"分支是否成功"。
    缺了就填降级文案 —— 这样下游无论怎么用都不会拿到空字符串，
    也就不存在"空字段被拼进 Prompt 后模型以为没这条要求"的漏洞。
    """
    geo = _text_of(state, "geology_report")
    well = _text_of(state, "wellbore_report")

    missing: list[str] = []
    reasons: dict[str, str] = {}
    if not geo:
        missing.append("geology")
        reasons["geology"] = str(
            _status_of(state, "geology_status").get("error") or "分支未产出结论"
        )
    if not well:
        missing.append("wellbore")
        reasons["wellbore"] = str(
            _status_of(state, "wellbore_status").get("error") or "分支未产出结论"
        )

    return {
        "geology": geo or GEOLOGY_MISSING_TEXT,
        "wellbore": well or WELLBORE_MISSING_TEXT,
        "geology_ok": bool(geo),
        "wellbore_ok": bool(well),
        "missing": missing,
        "degraded": bool(missing),
        "reasons": reasons,
    }


def degradation_block(info: dict) -> str:
    """报告末尾的「数据完整性提示」。全部成功时返回空串（不污染正常报告）。

    用引用块 + "不代表该维度无风险"这种明确措辞：这段是给人看的，
    而且它的存在本身就是在防止一种误读 —— 读者把"没提"当成"没问题"。
    """
    if not (info or {}).get("degraded"):
        return ""

    reasons = info.get("reasons") or {}
    lines = [
        "> ⚠️ **数据完整性提示**：本次报告未能获得全部分支的证据，",
        "> 以下模块失败，其结论**不代表该维度无风险**，请人工复核：",
    ]
    if "geology" in (info.get("missing") or []):
        lines.append(f"> - {GEOLOGY_MISSING_TEXT}（原因：{reasons.get('geology', '未知')}）")
    if "wellbore" in (info.get("missing") or []):
        lines.append(f"> - {WELLBORE_MISSING_TEXT}（原因：{reasons.get('wellbore', '未知')}）")
    lines += [
        ">",
        "> 其余结论仅基于可用分支的数据生成，请结合现场资料判断。",
    ]
    return "\n".join(lines)


def parallel_timing_of(state: dict) -> dict:
    """由两条分支的时间戳算并行收益。

    墙钟 = max(结束) − min(开始)；串行 = 两条分支耗时之和。

    两个值都是**实测**的，不是按节点名推算的。所以如果哪天有人把扇出改回
    串行（比如为了省显存），``speedup`` 会立刻掉到 1.0 附近 —— 这个字段
    就是那个回归的探针。缺时间戳时退回 ``max(elapsed)``，
    保证任何状态下都能返回一个说得通的数，而不是崩掉。
    """
    g = _status_of(state, "geology_status")
    w = _status_of(state, "wellbore_status")

    def _f(d: dict, key: str) -> float:
        try:
            return float(d.get(key) or 0.0)
        except (TypeError, ValueError):
            return 0.0

    g_s, w_s = _f(g, "elapsed_s"), _f(w, "elapsed_s")
    serial = g_s + w_s

    starts = [v for v in (_f(g, "started_at"), _f(w, "started_at")) if v > 0]
    ends = [v for v in (_f(g, "ended_at"), _f(w, "ended_at")) if v > 0]
    wall = (max(ends) - min(starts)) if starts and ends else max(g_s, w_s)
    # 时间戳异常（时钟回拨、跨进程续跑）时别算出负墙钟
    wall = max(wall, 0.0)

    return {
        "geology_s": round(g_s, 2),
        "wellbore_s": round(w_s, 2),
        "wall_s": round(wall, 2),
        "serial_s": round(serial, 2),
        "speedup": round(serial / wall, 2) if wall > 0 else 1.0,
        "overlap_s": round(max(0.0, serial - wall), 2),
    }
