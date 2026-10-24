"""
概率溯源守卫（纯规则，不调模型）
================================
报告里出现的每个百分比，要么能在证据 / 工具结果里找到同一个数字，要么就是
编的。RAGAS 实测：一份报告写了 5 个"XX 风险概率 12%/8%/5%/15%/10%"，而整条
流程只算过密度和压力曲线 —— 这 5 个数字一个都无处可查，忠实度掉到 0.095。

这个模块只做两件事，都是确定性字符串规则：

* ``unsupported_probabilities`` —— 找出无法溯源的断言
* ``strip_unsupported_probabilities`` —— 把断言所在的**整句**删掉

为什么独立成 ``utils/``：它是纯函数，不该被绑在评审节点的依赖链上
（``nodes/evaluate_node`` → ``utils/tokens`` → ``core/reranker`` → torch）。
放在这里之后，``nodes/aggregate_finalize.py`` 也能直接用它做收尾，
而 CI 的零依赖单测可以覆盖它。
"""
from __future__ import annotations

import re

# 「概率 … 12%」——只盯概率断言，不去管"含水饱和度 12%"这类普通数值。
_PROBABILITY_CLAIM = re.compile(r"(?:概率|比例|占比)[^。；\n]{0,14}?(\d+(?:\.\d+)?)\s*%")
_ANY_PERCENT = re.compile(r"(\d+(?:\.\d+)?)\s*%")

# 编造概率的扣分上限。定得比"重写阈值"低，是为了让这类报告一定被打回，
# 而不是靠提示词求模型自觉 —— 实测 7B 会照着示例编 5 个百分数。
UNSUPPORTED_PROBABILITY_SCORE_CAP = 4.0
_SENTENCE_END = re.compile(r"(?<=[。！？!?])")


def unsupported_probabilities(report: str, evidence: str) -> list[str]:
    """报告里出现、但证据/工具结果里没有的概率/比例断言。

    RAGAS 的 Faithfulness 正是被这类断言拉低的：报告写「断层活化风险概率
    12%（100 次蒙特卡洛模拟）」，而全流程只算过密度和压力曲线，12% 无处可查。
    这里用确定性规则先拦一遍，比让模型自查可靠。

    「比例」也在拦截范围内：只匹配「概率」时，报告改写成「压力累积达到破裂
    压力的比例为 95%」就绕过了守卫，而 95% 同样是凭空来的（实测忠实度 0.545）。
    """
    allowed = set(_ANY_PERCENT.findall(evidence or ""))
    claims: list[str] = []
    for match in _PROBABILITY_CLAIM.finditer(report or ""):
        if match.group(1) not in allowed:
            claims.append(match.group(0).strip())
    return claims


def strip_unsupported_probabilities(report: str, evidence: str) -> tuple[str, list[str]]:
    """把无法溯源的百分比连同它所在的整句一起删掉。

    只扣分等于承认"这段文字不该发出去"，却照样把它发给用户：实测某次重写
    预算耗尽后，一份带 5 个编造概率的报告被强制放行，RAGAS 忠实度 0.000。
    扣分能推动重写，但推动失败时必须有人动手删，那个人只能是代码。

    删**整句**而不是把数字替换成「该风险未做概率量化」，是因为替换文本本身
    就是一句证据支持不了的新陈述：实测一份报告被替换出 11 处「（该风险未做
    概率量化）」，Faithfulness 里这些句子全部判为无依据 —— 修幻觉的代码
    反而在制造无法溯源的句子。整句删掉不留残句。
    """
    claims = unsupported_probabilities(report, evidence)
    if not claims:
        return report, []
    unsupported = set(claims)
    kept: list[str] = []
    for sentence in _SENTENCE_END.split(report):
        if not sentence.strip():
            continue
        offending = [
            match.group(0).strip()
            for match in _PROBABILITY_CLAIM.finditer(sentence)
            if match.group(0).strip() in unsupported
        ]
        if offending:
            continue
        kept.append(sentence)
    cleaned = "".join(kept).strip()
    return cleaned, claims
