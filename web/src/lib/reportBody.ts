/**
 * 报告正文的轻量渲染。
 *
 * ## 为什么自己解析而不是引一个 markdown 库
 *
 * 正文是 `ops/reports.py` 的模板拼出来的，用的是**固定的一小撮**语法：
 * `#`/`##`/`###` 标题、`- ` 列表（含两级缩进）、`> ` 原文引用、`**粗体**`，
 * 以及引用标记 `[alarm:A-...]`。没有表格、没有链接、没有 HTML。
 *
 * 为这六种形状引一个 30 KB 的解析器，代价不只是体积：markdown 库会
 * **忠实地**渲染任意 HTML 与链接，而这份正文最终会被截图放进监管材料。
 * 一个从模型输出里漏进来的 `<script>` 或外链，在监管文件里是另一类问题。
 * 白名单解析器在这里比通用解析器更安全，也更短。
 *
 * ## 必须吃掉的两样东西
 *
 * `<!-- strict -->` / `<!-- /strict -->` 是**给检查器看的**区块标记，
 * 标记内部每个数字都必须带引用。它们对读者没有意义，原样渲染会变成
 * 一行看不懂的注释。而它们出现在每一段带数字的正文里。
 *
 * ## 这个函数是纯的
 *
 * 不碰 DOM、不读时间。所以"渲染错了"可以在 `node --test` 里断言，
 * 而不是靠人看截图 —— 截图看不出 `>` 少了一层缩进，也看不出
 * 引用标记被当成普通文字。
 */

export type BlockKind = "h1" | "h2" | "h3" | "li" | "li2" | "quote" | "p";

export type Segment =
  | { kind: "text"; text: string }
  | { kind: "bold"; text: string }
  | { kind: "cite"; token: string; refKind: string; ref: string };

export interface Block {
  kind: BlockKind;
  segments: Segment[];
}

const STRICT_OPEN = "<!-- strict -->";
const STRICT_CLOSE = "<!-- /strict -->";

/** 引用标记：`[alarm:A-20261027T1000-capture_rate]`。 */
const CITE = /\[(alarm|reading|summary):([^\]]+)\]/g;

/**
 * 把一行文字切成文本 / 粗体 / 引用标记三种片段。
 *
 * 粗体与引用标记可能嵌套（`**捕集率**：86.8 % [alarm:...]`），所以先按
 * 引用标记切，再在每段里切粗体 —— 顺序反过来会让 `**` 把引用标记包进去，
 * 于是角标变成粗体文字而不是一个可悬停的徽标。
 */
export function segment(text: string): Segment[] {
  const out: Segment[] = [];
  const push = (kind: "text" | "bold", value: string) => {
    if (!value) return;
    out.push({ kind, text: value });
  };

  let cursor = 0;
  CITE.lastIndex = 0;
  for (let m = CITE.exec(text); m !== null; m = CITE.exec(text)) {
    emitText(text.slice(cursor, m.index), push);
    out.push({ kind: "cite", token: m[0], refKind: m[1]!, ref: m[2]! });
    cursor = m.index + m[0].length;
  }
  emitText(text.slice(cursor), push);
  return out;
}

function emitText(
  raw: string,
  push: (kind: "text" | "bold", value: string) => void,
): void {
  const parts = raw.split(/\*\*(.+?)\*\*/g);
  parts.forEach((part, i) => {
    if (!part) return;
    // split 带捕获组时，奇数下标是被 `**` 包住的内容
    push(i % 2 === 1 ? "bold" : "text", part);
  });
}

/**
 * 正文 → 块序列。
 *
 * 空行不产生块（渲染时由 CSS 的行距负责），`---` 之类的分隔线也不支持 ——
 * 模板里没有，加了反而会让"正文里出现了一条线"变成一个没人能解释的现象。
 */
export function parseBody(body: string): Block[] {
  const blocks: Block[] = [];
  for (const raw of body.split("\n")) {
    const line = raw.replace(/\s+$/, "");
    const trimmed = line.trim();

    // 区块标记吃掉，不渲染
    if (trimmed === STRICT_OPEN || trimmed === STRICT_CLOSE) continue;
    if (!trimmed) continue;

    // 模板里 `#` 只出现在标题，正文里没有注释语法
    if (trimmed.startsWith("### ")) {
      blocks.push({ kind: "h3", segments: segment(trimmed.slice(4)) });
    } else if (trimmed.startsWith("## ")) {
      blocks.push({ kind: "h2", segments: segment(trimmed.slice(3)) });
    } else if (trimmed.startsWith("# ")) {
      blocks.push({ kind: "h1", segments: segment(trimmed.slice(2)) });
    } else if (trimmed.startsWith("> ")) {
      blocks.push({ kind: "quote", segments: segment(trimmed.slice(2)) });
    } else if (line.startsWith("  - ")) {
      // 二级列表：归因与处置挂在告警下面，缩进就是它的语义
      blocks.push({ kind: "li2", segments: segment(line.slice(4)) });
    } else if (trimmed.startsWith("- ")) {
      blocks.push({ kind: "li", segments: segment(trimmed.slice(2)) });
    } else if (trimmed.startsWith("~~")) {
      // 已排除的归因用删除线，模板里是 `~~原因~~：理由`
      blocks.push({ kind: "li", segments: segment(trimmed) });
    } else {
      blocks.push({ kind: "p", segments: segment(trimmed) });
    }
  }
  return blocks;
}

/**
 * 正文里出现过的引用标记（去重、保序）。
 *
 * 用来在正文末尾列"本稿引用了哪些来源"。直接扫原文而不是从 `citations`
 * 字典取，是因为字典里可能有正文中已删除段落留下的孤儿条目 ——
 * 列出"一个正文里找不到的引用"会让读者去正文里找，然后找不到。
 */
export function citedTokens(body: string): string[] {
  const seen: string[] = [];
  CITE.lastIndex = 0;
  for (let m = CITE.exec(body); m !== null; m = CITE.exec(body)) {
    if (!seen.includes(m[0])) seen.push(m[0]);
  }
  return seen;
}
