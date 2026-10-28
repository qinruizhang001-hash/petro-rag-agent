/**
 * 「让 agent 现场生成一份报告」这一动作的判定。
 *
 * ## 为什么这一层单独存在
 *
 * 这个按钮是界面上唯一**会新建一条记录**的动作。它的失效方式全是安静的：
 *
 * - 演示模式下也能点，点完弹一句「生成成功」，而磁盘上什么都没有；
 * - 后端跑了两个 agent、其中一个失败，界面只显示「已生成」，读者以为
 *   报告完整 —— 实际上缺的那一节没人补；
 * - 存在阻断性冲突时照样说「3 项偏离」，而其中一条的意思是「这份稿子
 *   不能提交」，两者都被压成同一个数字；
 * - 读数来自**回放样本**，报告却写得像当期实测。
 *
 * 每一条都对应下面一个纯函数，`scripts/test-run.mjs` 直接断言。
 *
 * ## 为什么不做成「一键出报告」
 *
 * `ops/run` 是**确定性**的（不调模型），所以它很快；但快不等于简单：
 * 它依次跑「告警判定 → 报告撰写」，再跨 agent 做冲突检测。界面上必须
 * 让人看见这两步分别有没有成，而不是只看见一个转圈。
 */

export type ReportKind = "daily" | "weekly" | "regulatory";

export const RUN_KIND_LABEL: Record<ReportKind, string> = {
  daily: "日报",
  weekly: "周报",
  regulatory: "监管材料",
};

/** 主管 agent 派出去的一次子任务。 */
export interface AgentRun {
  agent: string;
  /** 为什么派它 —— 界面上直接显示，不翻译成"步骤 1" */
  reason: string;
  ok: boolean;
  elapsed_s: number;
  error: string;
}

/** 跨 agent 的冲突。`blocking` 为真时这份稿子不能提交。 */
export interface Conflict {
  kind: string;
  summary: string;
  detail: string[];
  blocking: boolean;
  next_step: string;
}

export interface OpsRunResult {
  /** 一句话总结，后端已排版好（含分级计数与上限校核） */
  summary: string;
  can_submit: boolean;
  runs: AgentRun[];
  conflicts: Conflict[];
  /** Markdown 简报：冲突的处置建议 */
  brief: string;
  period: [string, string];
  /** 空串 = 这一趟没产出报告（只跑了告警判定） */
  draft_id: string;
}

/**
 * 生成动作被挡住的原因。`null` = 可以点。
 *
 * 演示模式下必须挡住，而不是"点了之后再说没后端"：这个动作的价值全在
 * 「新记录真的落进了审批队列」，而演示模式没有队列 —— 让它看起来成功，
 * 等于让人以为自己的操作被记下了。
 */
export function generateBlocked(demo: boolean): string | null {
  if (demo) {
    return (
      "演示模式（fixture）没有后端：报告要由 ops 主管 agent 现场跑「告警判定 → " +
      "报告撰写 → 冲突检测」生成，没有可复用的快照。接上后端后这个按钮才可用。"
    );
  }
  return null;
}

/** 主管跑了几个 agent、成了几个。 */
export function runHeadline(r: OpsRunResult): string {
  const total = r.runs.length;
  const failed = r.runs.filter((x) => !x.ok).length;
  const blocking = r.conflicts.filter((c) => c.blocking).length;
  const parts = [`${total} 个 agent 执行完毕`];
  if (failed > 0) parts[0] = `${total} 个 agent 中 ${failed} 个失败`;
  if (r.conflicts.length > 0) {
    parts.push(
      blocking > 0
        ? `${r.conflicts.length} 项冲突（${blocking} 项阻断）`
        : `${r.conflicts.length} 项偏离（均非阻断）`,
    );
  }
  return parts.join(" · ");
}

/**
 * 失败的 agent 逐条说明。
 *
 * 不合成一句「生成失败」：缺的是哪一节、为什么缺，决定了这份稿子能不能用。
 * 一个失败但仍有 `draft_id` 的返回是**最危险**的情形 —— 稿子看起来完整，
 * 只是少了一节。
 */
export function failedAgents(r: OpsRunResult): string[] {
  return r.runs
    .filter((x) => !x.ok)
    .map((x) => `${x.agent} 未完成：${x.error || "未给出原因"}`);
}

/** 冲突逐条，阻断的排前面。 */
export function conflictLines(r: OpsRunResult): string[] {
  const ordered = [...r.conflicts].sort(
    (a, b) => Number(b.blocking) - Number(a.blocking),
  );
  return ordered.map(
    (c) => `${c.blocking ? "阻断" : "提示"}｜${c.summary}`,
  );
}

/**
 * 报告用的是哪批读数。
 *
 * `/api/ops/*` 是无状态的，读数由调用方送进去；界面送的是抓 fixture 时
 * 录下的那批（`readings.json`）。所以生成出来的报告**不是当期实测** ——
 * 这句话必须跟着报告一起出现，否则一份写着"2026-10-27 06:00–10:00"的
 * 日报会被当成今天的日报读。
 */
export function replayNote(period: [string, string] | null): string {
  if (!period || period.length !== 2) return "";
  const fmt = (s: string) => s.replace("T", " ").slice(0, 16);
  return `读数取自回放样本（${fmt(period[0])} – ${fmt(period[1])}），不是当期实测。`;
}

/** 生成成功后的一句话。带 draft_id，方便对着日志核。 */
export function runSuccessNote(r: OpsRunResult): string {
  if (!r.draft_id) {
    return "跑完了，但没有产出报告草稿 —— 检查 want_report 与报告期的读数是否为空。";
  }
  return `已生成 ${r.draft_id}，已进入下方草稿列表（状态：草稿）。`;
}
