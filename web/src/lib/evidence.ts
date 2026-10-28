/**
 * 证据强度：这条证据能信到什么程度。
 *
 * ## 三件事必须分开说
 *
 * | 维度 | 事实 | 读者该做什么 |
 * |---|---|---|
 * | 有 `file_path` | 能点回具体文献与切片 | 直接核对原文 |
 * | 只有向量召回 | 入库时没写来源，**无法**回溯 | 只能当线索，不能当引用 |
 * | 公式为主 | 能证明"论文用了这个格式"，证明不了机理 | 找别的证据 |
 *
 * 三者互不替代：一条切片可以既无血缘又以公式为主。所以这里不用一个
 * "0–3 分"的强度分把它们压成一个数 —— 压完之后，界面上只剩一个颜色，
 * 而读者无从知道缺的是哪一项（与 `AttributionItem` 不用星级同理）。
 *
 * ## 为什么"无血缘"要单独成一个等级而不是"缺省"
 *
 * `source` 是空串时，最省事的写法是渲染成 `—` 或直接不显示。那样一来，
 * 一条不可溯源的证据与一条**本来就没有出处的中间产物**（实体、关系）
 * 在视觉上完全一样，而后者本来就不该被引用。等级必须显式。
 */

import type { KbEvidence } from "./kb-types";

export type Strength = "traceable" | "unattributed";

export interface StrengthMeta {
  label: string;
  /** 悬浮说明。要写"为什么"，不只是"是什么" */
  title: string;
  /** 字面量类名：Tailwind 扫不到模板字符串，见 `severity.ts` 的同类注释 */
  cls: string;
}

export const STRENGTH_META: Record<Strength, StrengthMeta> = {
  traceable: {
    label: "可溯源",
    title:
      "这条切片入库时写了 file_path，可以点回具体文献的某一片原文 —— 引用角标只应该挂在这类证据上。",
    cls: "border-sev-normal/50 text-sev-normal",
  },
  unattributed: {
    label: "无文件血缘",
    title:
      "这条证据没有来源文件名（Chroma 向量库那 1,982 条的 metadata 只有 source=legacy_v2）。" +
      "它能作为检索线索，但无法回溯到具体文献，不能当作可引用证据。",
    cls: "border-sev-notice/60 text-sev-notice",
  },
};

/** 有来源文件名 → 可溯源；没有 → 无血缘。**空串与 undefined 同等对待**。 */
export function strengthOf(source: string | null | undefined): Strength {
  return (source ?? "").trim() ? "traceable" : "unattributed";
}

export interface EvidenceSummary {
  chunks: number;
  /** 带 file_path 的切片数 */
  traceable: number;
  /** 没有来源文件名的切片数 */
  unattributed: number;
  /** 以公式为主、不宜作为机理证据的切片数 */
  formulaHeavy: number;
  /** 涉及文献篇数 */
  sources: number;
  /** 图谱实体 / 关系。**不可引用**，只作检索线索 */
  entities: number;
  relationships: number;
}

/**
 * 从后端的 `stats` 重建摘要，但两个会改变结论的数字**本地重算**。
 *
 * `stats.traceable` / `stats.unsourced` 是后端算的，直接用当然更快。
 * 但这两个数决定界面上写"全部可回溯"还是"有 3 片无法回溯"，属于
 * 会改变结论的读数 —— 后端某个字段改名时，`stats.traceable` 变成
 * `undefined`，`undefined > 0` 为假，界面会安静地显示"全部可回溯"。
 * 所以逐片重算；片数为 0 时重算结果也是 0，与事实一致。
 */
export function summarize(evidence: KbEvidence | null | undefined): EvidenceSummary {
  const chunks = evidence?.chunks ?? [];

  let traceable = 0;
  let formulaHeavy = 0;
  for (const c of chunks) {
    if (strengthOf(c.source) === "traceable") traceable += 1;
    if (c.text_quality?.low) formulaHeavy += 1;
  }

  return {
    chunks: chunks.length,
    traceable,
    unattributed: chunks.length - traceable,
    formulaHeavy,
    // 后三个描述的是"这次检索"而不是"这一屏渲染了什么"，取后端统计。
    // 不用列表长度兜底：兜底会让"列表被截断"与"检索只有这么几条"
    // 显示成同一个数，而前者是界面问题、后者是事实。
    sources: evidence?.stats?.sources ?? 0,
    entities: evidence?.stats?.entities ?? 0,
    relationships: evidence?.stats?.relationships ?? 0,
  };
}
