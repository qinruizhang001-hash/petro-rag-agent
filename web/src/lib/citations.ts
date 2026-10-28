/**
 * 正文里的引用角标解析。
 *
 * ## 为什么这是一个独立文件
 *
 * 它是唯一能"把 `[2]` 变成可点角标"的逻辑，而它只有一条正确行为：
 * **认不出来的标记一律原样留着**。这条规则写错的表现是静默的 ——
 * 段落照常显示，只是角标不能点，或者反过来把数学区间 `[0,1]` 变成
 * 一个"未解析引用"的假告警。两种错都不会抛异常，所以只能靠单测锁。
 *
 * ## 为什么只认 1–2 位数字
 *
 * `[2026]` 更可能是年份，`[123]` 更可能是编号而不是引用号。
 * 本项目的 `reference_id` 是一次检索内的序号（"1"…"12"），
 * 所以 1–2 位足够；放宽到任意位数会让正文里的年份变成角标。
 *
 * ## 为什么解析不到的标记不做任何标注
 *
 * `[1,2]` 既可能是两个引用，也可能是"区间 1–2"。在正文里把解析不到的
 * 标记染成告警色，会在一片数学符号上造出一串假警报，而假警报的代价是
 * 真的警报被忽略。**证据清单本身永远是完整的**（每条证据都带角标），
 * 所以读者核对出处不依赖正文里那几个标记。
 */

export type CitationSegment =
  | { kind: "text"; text: string }
  | { kind: "cite"; ids: string[]; raw: string };

/**
 * 角标形态：`[1]`、`[1,2]`、`[1、2]`、`[1，2]`。
 *
 * `(?!\()` 用来跳过 markdown 链接语法 `[1](http://…)` —— 那种方括号是
 * 链接文字，不是引用。全角括号 `（` 不在此列，它更可能是"见（1）"。
 */
const MARKER = /\[(\d{1,2}(?:\s*[,，、]\s*\d{1,2})*)\](?!\()/g;

/**
 * 把一段正文切成"文字"与"角标"两种片段。
 *
 * 编号为 `0` 或以 `0` 开头的整段标记会被当成普通文字：`[0,1]` 是区间，
 * 而引用号从 1 开始。宁可少认一个角标，也不要把区间画成引用。
 */
export function splitCitations(text: string): CitationSegment[] {
  const out: CitationSegment[] = [];
  let last = 0;

  for (const match of text.matchAll(MARKER)) {
    const raw = match[0];
    const ids = (match[1] ?? "")
      .split(/[,，、]/)
      .map((s) => s.trim())
      .filter(Boolean);

    // 编号非法（0 / 前导零 / 空）→ 整段当文字，不拆
    const usable = ids.length > 0 && ids.every((id) => /^[1-9]\d?$/.test(id));
    if (!usable) continue;

    const start = match.index ?? 0;
    if (start > last) out.push({ kind: "text", text: text.slice(last, start) });
    out.push({ kind: "cite", ids, raw });
    last = start + raw.length;
  }

  if (last < text.length) out.push({ kind: "text", text: text.slice(last) });
  return out;
}

/**
 * 正文里出现过、但证据里找不到的角标编号。
 *
 * 只用于**提示**，不用于改写正文 —— 理由见文件头：认错一个区间
 * 的代价比漏报一个坏引用高。返回去重后的编号，保持首次出现顺序。
 */
export function unresolvedIds(
  segments: CitationSegment[],
  known: ReadonlySet<string>,
): string[] {
  const seen = new Set<string>();
  for (const seg of segments) {
    if (seg.kind !== "cite") continue;
    for (const id of seg.ids) {
      if (!known.has(id)) seen.add(id);
    }
  }
  return [...seen];
}

/** 分组只需要这几个字段，所以不 import `chat.ts` 的 `Citation` —— 那会把整条数据链带进纯逻辑模块。 */
export interface CitationLike {
  reference_id: string;
  source: string;
  traceable: boolean;
  formulaHeavy: boolean;
}

export interface CitationGroup<T extends CitationLike> {
  reference_id: string;
  source: string;
  traceable: boolean;
  /** 该来源的切片是否**全部**以公式为主 */
  formulaHeavy: boolean;
  chunks: T[];
}

/**
 * 按来源合并引用。
 *
 * ## 为什么必须合并
 *
 * `reference_id` 是**来源**的编号，不是切片的编号：一次检索 12 片原文来自
 * 6 篇文献，于是 12 片里只有 6 个编号、每个重复两次。逐片列出来，右栏会
 * 看到「[2] ccus_auto_04.pdf」出现三遍 —— 读者第一反应是列表坏了。
 *
 * 合并之后每篇一行、带「N 片」，点开看第一篇；完整 12 片在知识库 → 检索里
 * （那里一片一张卡、不合并 —— 两个界面的粒度不同是有意的：对话回答
 * "这条结论有哪些出处"，知识库回答"这一片原文到底怎么写的"）。
 */
export function groupCitations<T extends CitationLike>(citations: T[]): CitationGroup<T>[] {
  const order: string[] = [];
  const byId = new Map<string, T[]>();

  for (const c of citations) {
    const list = byId.get(c.reference_id);
    if (list) list.push(c);
    else {
      byId.set(c.reference_id, [c]);
      order.push(c.reference_id);
    }
  }

  return order.map((id) => {
    const chunks = byId.get(id) ?? [];
    return {
      reference_id: id,
      source: chunks[0]?.source ?? "",
      traceable: chunks.length > 0 && chunks.every((c) => c.traceable),
      formulaHeavy: chunks.length > 0 && chunks.every((c) => c.formulaHeavy),
      chunks,
    };
  });
}
