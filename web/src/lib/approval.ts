/**
 * 报告审批的**状态机**与**守卫**。
 *
 * ## 为什么这张表不是手写的
 *
 * `draft → pending_review → approved / rejected` 这套规则在
 * `ops/reports.py` 里已经有唯一一份实现。前端要在界面上决定"哪个按钮能点"，
 * 最省事的写法是在 TS 里再抄一张表 —— 而抄的那一刻起就有了两份真值。
 *
 * 之后后端加一条迁移（比如允许 `approved → draft` 做更正），前端的表不会
 * 跟着变。表现是：**界面上那个按钮点不动，或者点下去 409**。两种情况都不
 * 报错，只在真实审批时被发现 —— 而那时已经有人在电话里说"我批了"。
 *
 * 所以 `fixtures/approval.json` 是 `scripts/export_approval_contract.py`
 * **在四个状态上逐个动作真跑一遍**导出来的，不是人写的声明。CI 里
 * `--check` 会比对：`ops/reports.py` 一改而这份 fixture 没重新生成就红。
 *
 * ## 为什么这里仍然要有 `applyAction`
 *
 * live 模式下迁移由后端做，前端只负责显示。但 fixture 模式（默认，离线演示）
 * 没有后端可写，而"点了提交没反应"比"不能提交"更糟。所以本机演练路径
 * 需要同一套规则，只是执行者换成这个纯函数。
 *
 * 两条路径共用**同一份** `approval.json`，所以演练看到的行为与真实后端一致。
 * 差异只在持久化：本机存在浏览器里，live 写进后端 sqlite。
 */
import contract from "../fixtures/approval.json";

export type DraftStatus = "draft" | "pending_review" | "approved" | "rejected";
export type ApprovalAction = "submit" | "approve" | "reject" | "reopen";

export interface TrailEntry {
  at: string;
  from: DraftStatus;
  to: DraftStatus;
  actor: string;
  note: string;
}

export interface ReportCitation {
  kind: string;
  ref: string;
  detail: string;
}

/** `/api/ops/reports/{id}` 的响应形状。 */
export interface ReportDraft {
  draft_id: string;
  kind: "daily" | "weekly" | "regulatory";
  title: string;
  /** ISO 时间对 `[起, 止]` */
  period: [string, string];
  body: string;
  status: DraftStatus;
  status_label: string;
  trail: TrailEntry[];
  /** 引用检查问题，**提交前必须为空** */
  problems: string[];
  permit_id: string;
  blind_spots: string[];
  citation_count: number;
  /** 正文里每个 `[alarm:...]` 标记指向什么 */
  citations: Record<string, ReportCitation>;
  /** 后端下发的可执行动作。**不要在前端另算** */
  allowed: ApprovalAction[];
  /** 仅 fixture 种子有：说明这份稿子是刻意造出来的样本 */
  seed_note?: string;
}

/** 列表项。live 的 `/api/ops/reports` 只给摘要（正文可能几十 KB）。 */
export interface ReportSummary {
  draft_id: string;
  kind: ReportDraft["kind"];
  title: string;
  status: DraftStatus;
  status_label: string;
  period: [string, string];
  citation_count: number;
  problem_count: number;
  /** 本机演练时有；后端列表不带正文，因此没有这个字段 */
  seed_note?: string;
}

export interface ActionInput {
  actor?: string;
  reviewer?: string;
  note?: string;
}

// ─── 契约（来自生成的 fixture）─────────────────────────────
interface Edge {
  from: DraftStatus;
  action: ApprovalAction;
  to: DraftStatus;
}

interface Guard {
  action: ApprovalAction;
  field: "reviewer" | "note" | "problems";
  message: string;
}

const EDGES = contract.edges as Edge[];
const GUARDS = contract.guards as Guard[];

export const STATUS_LABEL = contract.labels as Record<DraftStatus, string>;

/**
 * 动作的中文名。
 *
 * 这一份是**界面文案**，不是规则 —— 规则（谁能迁移到哪）全在 `EDGES` 里。
 * 两者分开是必要的：文案会改（"提交审批"也许要改成"报审"），
 * 而规则改了必须走重新生成 fixture 那条路。
 *
 * 漏写一个动作不会报错，只会渲染出一个空按钮。所以 `test-approval.mjs`
 * 里有一条"契约里出现的每个动作都有中文名"。
 */
export const ACTION_LABEL: Record<ApprovalAction, string> = {
  submit: "提交审批",
  approve: "批准",
  reject: "驳回",
  reopen: "重新打开",
};

/** 动作的一句话解释，放在按钮 title 里。 */
export const ACTION_HINT: Record<ApprovalAction, string> = {
  submit: "提交后进入待审批。引用检查不通过时不允许提交",
  approve: "批准必须署名 —— 无署名的批准无法追溯，等于没人负责",
  reject: "驳回必须写意见，否则下一稿会原样再来一遍",
  reopen: "把已驳回的稿子改回草稿，保留完整审批轨迹",
};

/** 该状态下允许的动作，顺序与后端 `allowed_actions()` 一致。 */
export function actionsFrom(status: DraftStatus): ApprovalAction[] {
  return EDGES.filter((e) => e.from === status).map((e) => e.action);
}

/** 该动作在该状态下的落点；不合法返回 `null`。 */
export function targetOf(
  status: DraftStatus,
  action: ApprovalAction,
): DraftStatus | null {
  return EDGES.find((e) => e.from === status && e.action === action)?.to ?? null;
}

/** 终态：没有任何可执行动作。 */
export function isTerminal(status: DraftStatus): boolean {
  return actionsFrom(status).length === 0;
}

/**
 * 该动作需要哪些输入框。
 *
 * 从 `GUARDS` 反推，而不是再写一张"approve 要 reviewer"的表 ——
 * 后者会和后端悄悄分家，表现是"界面不让你填，但后端要求填"。
 * `problems` 不是输入框，是稿子自身的状态，故排除。
 *
 * 顺序固定成「先署名、后意见」：`GUARDS` 的顺序取决于探针的执行顺序，
 * 而表单里姓名在意见上面是常识。让探针顺序决定表单布局，改一次探针
 * 就会把两个框换个位置 —— 那不是产品行为。
 */
const FIELD_ORDER: Record<"reviewer" | "note", number> = { reviewer: 0, note: 1 };

export function requiredFields(action: ApprovalAction): Array<"reviewer" | "note"> {
  return GUARDS.filter(
    (g) => g.action === action && g.field !== "problems",
  )
    .map((g) => g.field as "reviewer" | "note")
    .sort((a, b) => FIELD_ORDER[a] - FIELD_ORDER[b]);
}

/** 该动作有哪些非输入类前置条件（目前只有"引用检查必须干净"）。 */
export function requiresCleanCitations(action: ApprovalAction): boolean {
  return GUARDS.some((g) => g.action === action && g.field === "problems");
}

function guardMessage(
  action: ApprovalAction,
  field: Guard["field"],
): string | null {
  return GUARDS.find((g) => g.action === action && g.field === field)?.message ?? null;
}

/**
 * 为什么这个动作现在不能做。返回 `null` 表示可以做。
 *
 * 界面上**必须把这个原因显示出来**，而不是只把按钮置灰。一个灰按钮
 * 能表达的只有"不行"，而读者需要知道"差什么" —— 是没署名、没写意见，
 * 还是稿子里还有三个数字没出处。
 */
export function blockedReason(
  draft: Pick<ReportDraft, "status" | "problems">,
  action: ApprovalAction,
  input: ActionInput,
): string | null {
  if (targetOf(draft.status, action) === null) {
    return `当前状态为「${STATUS_LABEL[draft.status]}」，不能执行「${ACTION_LABEL[action]}」`;
  }
  if (requiresCleanCitations(action) && draft.problems.length > 0) {
    const raw = guardMessage(action, "problems");
    if (!raw) return "引用检查未通过，不允许执行该动作";
    // 后端那段话是多行的：第一行是结论，后面逐条列"第几行的哪个数字没出处"。
    // 按钮旁边只放结论 —— 明细由界面单独列出（带行号、可逐条读）。
    // 冒号要去掉：结论后面接不上明细时，"…拒绝提交审批：" 以一个悬空的
    // 冒号结尾，读起来像内容没加载出来。
    const head = raw.split("\n")[0]!.replace(/[：:]\s*$/, "");
    return head;
  }
  for (const field of requiredFields(action)) {
    if (!(input[field] ?? "").trim()) {
      // 后端原话最准确（它知道为什么要这一项），直接引用
      return guardMessage(action, field) ?? `「${ACTION_LABEL[action]}」必须填写${field}`;
    }
  }
  return null;
}

export type ApplyResult =
  | { ok: true; draft: ReportDraft }
  | { ok: false; message: string };

/**
 * 在本机执行一次状态迁移（fixture 模式用；live 模式由后端执行）。
 *
 * 纯函数：不读时钟、不碰存储。`now` 由调用方传进来，否则测试里断言不了
 * 轨迹时间，而"轨迹对不对"正是这个功能唯一值得测的东西。
 */
export function applyAction(
  draft: ReportDraft,
  action: ApprovalAction,
  input: ActionInput,
  now: Date,
): ApplyResult {
  const reason = blockedReason(draft, action, input);
  if (reason) return { ok: false, message: reason };

  const to = targetOf(draft.status, action);
  if (to === null) return { ok: false, message: "状态迁移表里没有这条边" };

  const actor =
    (action === "approve" || action === "reject"
      ? input.reviewer
      : input.actor) ?? "本机操作人";

  const next: ReportDraft = {
    ...draft,
    status: to,
    status_label: STATUS_LABEL[to],
    allowed: actionsFrom(to),
    trail: [
      ...draft.trail,
      {
        at: formatStamp(now),
        from: draft.status,
        to,
        actor: actor.trim() || "本机操作人",
        note: (input.note ?? "").trim(),
      },
    ],
  };
  return { ok: true, draft: next };
}

// ─── 显示辅助 ─────────────────────────────────────────────
/** 本地时间 `2026-10-28 17:03`。轨迹要能被人读，不要 ISO 的 `T` 与毫秒。 */
export function formatStamp(d: Date): string {
  const p = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(
    d.getHours(),
  )}:${p(d.getMinutes())}`;
}

/** 报告期。年份与起始日同年时省掉重复的年份。 */
export function formatPeriod(period: [string, string]): string {
  const fmt = (iso: string) => {
    const d = new Date(iso);
    const p = (n: number) => String(n).padStart(2, "0");
    return { y: d.getFullYear(), s: `${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}` };
  };
  const a = fmt(period[0]);
  const b = fmt(period[1]);
  return a.y === b.y ? `${a.y}-${a.s} → ${b.s}` : `${a.y}-${a.s} → ${b.y}-${b.s}`;
}

export const KIND_LABEL: Record<ReportDraft["kind"], string> = {
  daily: "日报",
  weekly: "周报",
  regulatory: "监管材料",
};

export function toSummary(draft: ReportDraft): ReportSummary {
  return {
    draft_id: draft.draft_id,
    kind: draft.kind,
    title: draft.title,
    status: draft.status,
    status_label: draft.status_label,
    period: draft.period,
    citation_count: draft.citation_count,
    problem_count: draft.problems.length,
    seed_note: draft.seed_note,
  };
}
