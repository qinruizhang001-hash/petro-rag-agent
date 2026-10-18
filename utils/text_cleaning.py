"""
PDF 文本清洗与切片（纯函数，零依赖）
====================================
入库流水线的"原料处理"环节。刻意不 import pypdf / chromadb，这样它能进
CI 的离线单测（``tests.test_text_cleaning``），不需要装任何三方包。

为什么清洗值得单独一个模块
--------------------------
pypdf 抽出来的不是"文本"，而是"版面被压平之后的字符流"，带着一堆噪声：

  - arXiv 每页左侧的水印戳 ``arXiv:2606.27515v1 [physics.geo-ph] 25 Jun 2026``
  - 独立成行的页码
  - 行尾断词：``permea-\\nability``（不修就会变成 "permea ability"，
    嵌入模型拿到的不是同一个词，检索直接失效）
  - 参考文献列表（几十条文献占掉大量 token，且对实体抽取几乎无贡献）

这些噪声不会被下游报错，只会**静默拉低召回质量**。所以清洗必须在切片之前，
而且必须有测试盯着 —— 尤其是断词修复，它坏掉的时候看起来一切正常。
"""
from __future__ import annotations

import re

# arXiv 水印戳：arXiv:2606.27515v1 [physics.geo-ph] 25 Jun 2026
_ARXIV_STAMP = re.compile(
    r"arXiv:\s*\d{4}\.\d{4,5}v?\d*\s*(?:\[[^\]]{0,60}\])?\s*\d{0,2}\s*[A-Za-z]{3,9}\s*\d{4}",
    re.IGNORECASE,
)
# 独立成行的页码（1~4 位数字，允许两侧空白）
_PAGE_NUMBER = re.compile(r"^[ \t]*\d{1,4}[ \t]*$", re.MULTILINE)
# 行尾断词：字母 - 换行 字母
_HYPHEN_BREAK = re.compile(r"([A-Za-z])-\n([a-z])")
# 连续空行 / 行内多空格
_MULTI_BLANK = re.compile(r"\n{3,}")
_MULTI_SPACE = re.compile(r"[ \t]{2,}")
# 常见页眉页脚残留（预印本标记、版权行）
_PREPRINT_MARK = re.compile(
    r"^\s*(?:Preprint\.?\s+Under\s+Review\.?|Under review.*|Submitted to.*|"
    r"Published as a conference paper.*|Proceedings of.*)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
# 参考文献标题
_REF_HEADING = re.compile(
    r"\n[ \t]*(?:References|REFERENCES|Bibliography|BIBLIOGRAPHY|"
    r"Reference list|参考文献)[ \t]*\n"
)

# 作者单位 / 通讯信息。PDF 首页这块会被实体抽取当成"实体"，实测在 765 个
# 节点里贡献了 43 个（38 个大学名 + 5 个年份），占 5.6%。它们不是错误，
# 但对"断层活化—盖层完整性"这类推理毫无用处，只会稀释图谱。
_AFFILIATION = re.compile(
    r"(universit|institute|college|department|dept\.?\s|laborator|"
    r"school of|faculty of|academy of|research cent|"
    r"corresponding author|e-?mail|orcid|@\w+\.\w+)",
    re.IGNORECASE,
)
# 邮编 / 街道地址： "10100 Burnet Rd., Bldg. 130, Austin, TX 78758, USA"
_POSTAL = re.compile(
    r"(\b\d{2,6}\s+[A-Z][\w.\-]*(?:\s+[A-Z][\w.\-]*)*\s+"
    r"(?:Rd|Road|St|Street|Ave|Avenue|Blvd|Bldg|Building|Dr|Drive)\b"
    r"|\b[A-Z]{2}\s+\d{5}(?:-\d{4})?\b)",
)


def strip_front_matter(text: str, window: int = 1800) -> str:
    """
    删掉论文开头的作者 / 单位 / 地址块。

    只在前 ``window`` 个字符里动手：这个模式只出现在首页页眉，正文里出现
    "University of Alberta" 往往是有意义的（比如研究由该校团队完成），
    不该被删。
    """
    if not text:
        return text
    head, tail = text[:window], text[window:]
    kept = [
        line for line in head.split("\n")
        if not (_AFFILIATION.search(line) or _POSTAL.search(line))
    ]
    return "\n".join(kept) + ("\n" + tail if tail else "")


def strip_references(text: str) -> str:
    """
    砍掉参考文献列表。

    只砍**位于文本后半段**的最后一个匹配：正文里也会出现 "references" 这个词
    （比如 "cross-references"），而真正的文献列表一定在末尾。用"最后一个 +
    位置在后半段"两个条件，比单纯找第一个匹配稳得多。
    """
    matches = list(_REF_HEADING.finditer(text))
    if not matches:
        return text
    cut = matches[-1]
    if cut.start() < len(text) * 0.5:
        return text
    return text[: cut.start()]


def clean_pdf_text(raw: str, drop_references: bool = True) -> str:
    """把 pypdf 的原始输出整理成可入库的连续文本。"""
    if not raw:
        return ""

    text = raw

    # ① 断词必须在压平换行之前修 —— 顺序反了就没有 "\n" 可依据了
    text = _HYPHEN_BREAK.sub(r"\1\2", text)

    # ② 去掉水印戳与页眉页脚
    text = _ARXIV_STAMP.sub(" ", text)
    text = _PREPRINT_MARK.sub("", text)

    # ③ 去掉独立成行的页码
    text = _PAGE_NUMBER.sub("", text)

    # ④ 压平多余空白
    text = _MULTI_SPACE.sub(" ", text)
    text = _MULTI_BLANK.sub("\n\n", text)

    # ⑤ 参考文献放最后砍，否则前面删页码可能把标题行打散
    if drop_references:
        text = strip_references(text)

    return text.strip()


def chunk_text(text: str, size: int = 1200, overlap: int = 150) -> list[str]:
    """
    段落感知切片。

    按段落累积到 ``size`` 字符，而不是机械地每 ``size`` 个字符切一刀 ——
    句子被腰斩会让实体抽取拿到半句话，抽出"断层活化导致"这种残缺实体。

    单段超长时硬切（有些 PDF 抽出来整页就是一个"段落"），此时保证不丢字符。
    ``overlap`` 让相邻块共享结尾，避免刚好跨块的论述两边都读不到。
    """
    if not text:
        return []
    if size <= 0:
        raise ValueError("size 必须为正数")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap 必须满足 0 <= overlap < size")

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    buf = ""

    def flush() -> None:
        nonlocal buf
        if buf.strip():
            chunks.append(buf.strip())
        buf = ""

    for para in paragraphs:
        # 超长段落：先flush 已积累的，再硬切成多块
        if len(para) > size:
            flush()
            start = 0
            while start < len(para):
                chunks.append(para[start:start + size].strip())
                if start + size >= len(para):
                    break
                start += size - overlap
            buf = ""
            continue

        if len(buf) + len(para) + 2 <= size:
            buf = f"{buf}\n\n{para}" if buf else para
        else:
            flush()
            # 用上一块结尾做重叠
            tail = chunks[-1][-overlap:] if chunks and overlap else ""
            buf = f"{tail}\n\n{para}" if tail else para

    flush()
    return [c for c in chunks if c]


def select_graph_text(text: str, budget: int) -> str:
    """
    从全文里截出"喂给知识图谱"的那一段。

    为什么不是全文：LightRAG 每处理一个 chunk 就要跑一次 LLM 实体抽取，
    实测在 CPU 上约 31 秒/千字符。20 篇论文全文（约 100 万字符）意味着
    8 小时以上的推理 —— 不现实。取开头 ``budget`` 字符（标题 / 摘要 /
    引言），这一段决定了论文"讲什么问题、涉及哪些地质概念"，正是图谱
    最需要的部分；结果与讨论里的大量数值对实体关系贡献很小。

    截断点会向前找一个句末标点，避免留下半句话。
    """
    if budget <= 0:
        return ""
    if len(text) <= budget:
        return text

    window = text[:budget]
    # ① 优先在句末收尾。
    #
    # 取窗口内**最后**一个句末标点。下界放在 1/4 处而不是 1/2 处：真实摘要
    # 里句号很密，rfind 自然落在靠后的位置，不会浪费预算；但窗口尾部如果
    # 恰好是一长串没有标点的公式或数字，下界太高就会直接退化成硬截，
    # 把 "fracture pres|sure" 切成两半，图谱里就出现 "fracture pres" 这种
    # 残缺实体。宁可少喂几十个字符，也不要半个词。
    floor = budget // 4
    best = -1
    for punct in (". ", ".\n", "? ", "! ", "。"):
        pos = window.rfind(punct, floor)
        if pos > best:
            best = pos
    if best != -1:
        return window[: best + 1].strip()

    # ② 找不到句末就退到**词边界**，绝不切在词中间。
    #    这一步比硬截只多花一行代码，但避免了残缺实体进入图谱。
    space = window.rfind(" ", floor)
    if space != -1:
        return window[:space].strip()

    # ③ 连空格都没有（极长的无分隔串）才硬截
    return window.strip()
