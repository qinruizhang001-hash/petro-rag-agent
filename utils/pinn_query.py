"""
从自然语言里抽 PINN 的调用参数（纯函数，零依赖）
================================================
``predict_pressure(x, t)`` 需要两个数值，而用户是用中文问的：
"预测距离井口 50 米处，注入 10 天后的压力"。

为什么不用语言模型去抽
----------------------
工具规划那一步已经由模型产出 JSON 了，但它**不保证**每次都规划出这个工具：
本地 7B 模型在上下文变长后经常"觉得自己能答"，直接跳过工具调用，于是压力值
变成模型口算 —— 那正是本项目一直想消灭的幻觉。所以这里做确定性兜底：
只要问题里同时出现了"井口 + 距离"和"注入 + 天数"，就一定要调用模型算，
不管模型有没有规划。

为什么单独成模块
----------------
``nodes/aggregate_node.py`` 一 import 就拉起 torch / openai / matplotlib，
把正则放进去等于这堆规则永远没有自动化测试。而正则恰恰是**最容易静默失败**
的一环：抽错了不会报错，只会算出一个看起来合理的错值。放在这里之后，
``tests/test_pinn_query.py`` 可以在零依赖环境下把各种说法穷举完。

返回 ``None`` 而不是抛异常：调用方是"有就用、没有就跳过"的兜底逻辑。

只支持中文问法
--------------
"50 meters from the well after 10 days" 这种英文语序**不会**命中。要支持它
就得再加一套模式，而英文里的 "10 days" 在历史记录、文献摘录里到处都是，
宽松匹配会让兜底逻辑误触发 —— 误触发比不触发更糟：它会往报告里塞一个
用户没问过的压力值。英文提问仍可走"模型规划 → 工具调用"那条路。
"""
from __future__ import annotations

import re

# 距离：井口/井 后面跟一个数 + 长度单位。
# 允许 "距离井口 50 米" / "距井 50m" / "井口50米" / "井口距离 50 米"。
_X_RE = re.compile(
    r"(?:井口|井筒|井)\s*(?:距离|处)?\s*[:：]?\s*"
    r"(\d+(?:\.\d+)?)\s*(?:米|m(?![a-z])|meters?)",
    re.IGNORECASE,
)

# 时间：注入 + 数 + 天（也接受 "10 天后" / "10 days"）。
_T_RE = re.compile(
    r"(?:注入|注入时长|注入时间|注入周期)\s*[:：]?\s*"
    r"(\d+(?:\.\d+)?)\s*(?:天|日|days?)",
    re.IGNORECASE,
)
# 兜底：没有"注入"二字，但明确写了"X 天后"。
_T_LOOSE_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:天|日)\s*(?:后|之后|以后)", re.IGNORECASE
)

# 小时也要能认（"注入 240 小时后"），换算成天。
_T_HOUR_RE = re.compile(
    r"(?:注入|注入时长|注入时间)\s*[:：]?\s*"
    r"(\d+(?:\.\d+)?)\s*(?:小时|hours?|h(?![a-z]))",
    re.IGNORECASE,
)

# 合理区间：超出这个范围的数字几乎一定是抽错了（比如把 250 bar 当成距离）。
_X_RANGE = (0.0, 100.0)      # 计算域长度 L = 100 m
_T_RANGE = (0.0, 30.0)       # 训练覆盖 30 天


def extract_pinn_query(text: str) -> tuple[float, float] | None:
    """从文本里抽出 ``(x 米, t 天)``；缺任一项或明显不合理时返回 ``None``。

    例::

        >>> extract_pinn_query("预测距离井口 50 米处，注入 10 天后的压力")
        (50.0, 10.0)
        >>> extract_pinn_query("注入压力 250 bar，温度 80℃") is None
        True
    """
    if not isinstance(text, str) or not text.strip():
        return None

    m_x = _X_RE.search(text)
    if not m_x:
        return None
    x = float(m_x.group(1))

    m_t = _T_RE.search(text) or _T_LOOSE_RE.search(text)
    if m_t:
        t = float(m_t.group(1))
    else:
        m_h = _T_HOUR_RE.search(text)
        if not m_h:
            return None
        t = float(m_h.group(1)) / 24.0

    if not (_X_RANGE[0] <= x <= _X_RANGE[1]):
        return None
    if not (_T_RANGE[0] <= t <= _T_RANGE[1]):
        return None
    return x, t
