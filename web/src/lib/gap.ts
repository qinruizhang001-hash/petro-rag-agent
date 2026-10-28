import { kbDataSource } from "./kb";
import type { KbGapKind, KbGapsPayload } from "./kb-types";

/**
 * 缺口反馈的**判定**，与它的渲染分开。
 *
 * 为什么单开一个文件：这三件事全是"判错了也不会报错"的类型 ——
 *
 * * 演示模式下让按钮可点，用户打完字才被告知"其实没发出去"；
 * * 队列读不出来时显示"0 条"，读者据此认为没人反馈过；
 * * 把 `unavailable` 也当成"可以登记缺口"，一次服务故障就被记成语料缺口。
 *
 * 判定放在纯函数里，`scripts/test-gap.mjs` 才能不启浏览器地断言它们。
 * 组件里只剩渲染 —— 而"没接上"和"接上了但判错"在截图上长得一样。
 */

/**
 * 种类的界面措辞。**与后端枚举一一对应**（`api_kb.GAP_KINDS`），
 * 加一种要同时改两边；`test-gap.mjs` 会比对这两份清单的长度与键。
 */
export const GAP_KIND_LABEL: Record<KbGapKind, string> = {
  missing_document: "缺一篇文献",
  not_in_graph: "这篇没进图谱",
  unanswered: "检索没答到",
  wrong_evidence: "引的原文不对",
  other: "其他",
};

/** 可以登记缺口的检索状态。`unavailable` **不在里面**。 */
export const REPORTABLE_STATUSES = ["ok", "empty"] as const;

export function canReportFromStatus(status: string): boolean {
  return (REPORTABLE_STATUSES as readonly string[]).includes(status);
}

/**
 * 表单能不能提交。返回 `null` 表示可以，否则是要显示给用户的原因。
 *
 * fixture 模式下**在用户动手之前**就拦住：缺口反馈的全部价值在于送到
 * 维护语料的人手上，存在浏览器里等于没写。让他打完一百字再告诉他，
 * 是同一类欺骗的另一种形式。
 *
 * 注意这与审批轨迹（`draftStore.ts`）的处理相反 —— 那个存本机是对的，
 * 因为它是**你自己的**演练，刷新不丢就有意义。
 */
export function gapFormBlocked(): string | null {
  if (kbDataSource === "fixture") {
    return "演示模式不落盘 —— 缺口反馈的价值在于送到维护语料的人手上，存在浏览器里等于没写。接上后端（VITE_KB_SOURCE=live）后可用。";
  }
  return null;
}

export interface QueueSummary {
  headline: string;
  detail: string;
}

/**
 * 队列的第一行摘要。
 *
 * **读不到必须说读不到。** 队列文件损坏、权限不对、或者这台机器上
 * 根本没有那个文件时，`total` 也是 0 —— 而"0 条"读起来是"没人反馈过"，
 * 那是一个关于人的结论，不是关于文件系统的结论。
 */
export function queueSummary(data: KbGapsPayload): QueueSummary {
  if (!data.available) {
    return {
      headline: "队列读不到",
      detail: `${data.reason}——这不等于没人反馈过。`,
    };
  }
  if (data.total === 0) {
    return { headline: "还没有人登记过缺口。", detail: "" };
  }
  const broken =
    data.unreadable_lines > 0 ? ` · ${data.unreadable_lines} 行读不出来` : "";
  const kinds = Object.entries(data.by_kind)
    .map(([k, n]) => `${GAP_KIND_LABEL[k as KbGapKind] ?? k} ${n}`)
    .join(" · ");
  return { headline: `已登记 ${data.total} 条${broken}`, detail: kinds };
}

/** 一条记录的单行摘要（队列列表用）。 */
export function recordLine(record: {
  kind: string;
  target?: string;
  recorded_at: string;
}): string {
  const label = GAP_KIND_LABEL[record.kind as KbGapKind] ?? record.kind;
  const target = record.target ? `：${record.target}` : "";
  // 2026-10-28T09:43:50+00:00 → 2026-10-28 09:43
  const when = record.recorded_at.slice(0, 16).replace("T", " ");
  return `${label}${target} · ${when}`;
}
