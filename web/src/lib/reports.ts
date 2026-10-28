/**
 * 报告草稿的读取与审批动作。
 *
 * ## 两种数据源，一个界面
 *
 * - `live`：`/api/ops/reports` 一套。审批写进后端 sqlite，跨请求、跨班次可见。
 * - `fixture`（默认）：离线演示。草稿来自 `fixtures/reports.json`
 *   （`scripts/export_approval_contract.py` 用真实端点产出），迁移在本机执行，
 *   存在浏览器里。
 *
 * 两条路径共用 `lib/approval.ts` 的同一份契约与同一套守卫，所以**演练看到的
 * 行为与真实后端一致**。差异只有一处，而且必须显示在界面上：
 * 本机轨迹不是审计记录。
 *
 * ## 为什么本机也要持久化
 *
 * 上一版的告警处置标记只活在 `useState` 里，刷新即丢 —— 而"刷新一下看看
 * 状态对不对"是每个人都会做的动作。审批更严重：一条"待审批"的记录消失，
 * 审批人会以为稿子被撤回了。
 *
 * 所以本机演练也落 `localStorage`，键独立于告警标记（`ccus-ops.drafts.v1`）：
 * 两种记录的保留期限与清理时机不同，混在一个键里，"清掉告警标记"会顺手
 * 清掉审批轨迹。
 *
 * ## 本机存储的一个陷阱：种子变了
 *
 * `fixtures/reports.json` 会被重新生成（报告模板改了、读数变了）。此时
 * 浏览器里存着的旧草稿会把新种子**盖住**，而界面上看不出任何异常 ——
 * 它只是继续显示上一版的正文。所以存一份种子指纹，对不上就整体丢弃。
 * 丢弃是安全的：本机演练数据不是业务记录，丢掉只是回到初始状态。
 */
import {
  applyAction,
  formatStamp,
  toSummary,
} from "./approval";
import type {
  ActionInput,
  ApprovalAction,
  DraftStatus,
  ReportDraft,
  ReportSummary,
} from "./approval";
import { dataSource } from "./api";
import { DRAFT_STORAGE_KEY, decodeDrafts, encodeDrafts, seedFingerprint } from "./draftStore";
import type { DropReason } from "./draftStore";
import seeds from "../fixtures/reports.json";

export { DRAFT_STORAGE_KEY, decodeDrafts, encodeDrafts, seedFingerprint };

const BASE = import.meta.env.VITE_OPS_BASE ?? "/api";

/** 本机演练是否在用（`fixture` 模式）。界面据此决定怎么写那句免责说明。 */
export const rehearsal = dataSource !== "live";

const SEED_FINGERPRINT = seedFingerprint(seeds.drafts as unknown as ReportDraft[]);

// ─── 本机存储 ─────────────────────────────────────────────
function storageOrNull(): Storage | null {
  if (typeof localStorage === "undefined") return null;
  try {
    const probe = `${DRAFT_STORAGE_KEY}.probe`;
    localStorage.setItem(probe, "1");
    localStorage.removeItem(probe);
    return localStorage;
  } catch {
    // Safari 无痕、企业策略禁用本地存储时 `setItem` 会抛。
    // 只有真写一次才知道 —— `typeof localStorage` 在这些场景下是 "object"。
    return null;
  }
}

export function draftStorageAvailable(): boolean {
  return storageOrNull() !== null;
}

// ─── 本机源 ───────────────────────────────────────────────
type Listener = () => void;

let drafts: ReportDraft[] | null = null;
let summaries: ReportSummary[] = [];
let dropReason: DropReason | null = null;
const listeners = new Set<Listener>();

function recompute(): void {
  summaries = (drafts ?? []).map(toSummary);
}

function notify(): void {
  recompute();
  for (const fn of listeners) fn();
}

function loadLocal(): ReportDraft[] {
  if (drafts) return drafts;
  const store = storageOrNull();
  const outcome = decodeDrafts(store?.getItem(DRAFT_STORAGE_KEY) ?? null, SEED_FINGERPRINT);
  if (outcome.ok) {
    drafts = outcome.drafts;
    dropReason = null;
  } else {
    drafts = seeds.drafts as unknown as ReportDraft[];
    dropReason = outcome.reason;
  }
  recompute();
  return drafts;
}

function persistLocal(): void {
  const store = storageOrNull();
  if (!store || !drafts) return;
  try {
    store.setItem(DRAFT_STORAGE_KEY, encodeDrafts(drafts, SEED_FINGERPRINT));
  } catch {
    // 配额满或被策略禁用。界面已经在别处说明"存不住"，
    // 这里不抛 —— 抛出去会让一次成功的状态迁移看起来失败了。
  }
}

/** 本机存不住时的原因，交给界面照实说。 */
export function localDropReason(): DropReason | null {
  loadLocal();
  return dropReason;
}

export function resetLocalDrafts(): void {
  drafts = seeds.drafts as unknown as ReportDraft[];
  dropReason = null;
  const store = storageOrNull();
  if (store) {
    try {
      store.removeItem(DRAFT_STORAGE_KEY);
    } catch {
      /* 同上 */
    }
  }
  notify();
}

// ─── 后端源 ───────────────────────────────────────────────
async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) throw new Error(`${res.status} ${path}`);
  return (await res.json()) as T;
}

/**
 * 后端错误里带 `detail`。409 的 detail 是**对象**（含 problems 列表），
 * 直接当字符串读会得到 `[object Object]` —— 一个让人以为是前端 bug 的报错。
 */
async function postJson<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const raw = await res.text();
  let payload: unknown = raw;
  try {
    payload = raw ? JSON.parse(raw) : null;
  } catch {
    /* 保持原文 */
  }
  if (!res.ok) {
    const detail = (payload as { detail?: unknown } | null)?.detail;
    const msg =
      typeof detail === "string"
        ? detail
        : detail && typeof detail === "object" && "message" in detail
          ? String((detail as { message: unknown }).message)
          : `HTTP ${res.status}`;
    throw new Error(msg);
  }
  return payload as T;
}

interface BackendSummary {
  id: string;
  kind: ReportDraft["kind"];
  title: string;
  status: DraftStatus;
  period_start: string;
  period_end: string;
  created_at: string;
  updated_at: string;
}

// ─── 统一接口 ─────────────────────────────────────────────
export interface DraftSource {
  mode: "local" | "backend";
  list(): Promise<ReportSummary[]>;
  get(id: string): Promise<ReportDraft>;
  act(id: string, action: ApprovalAction, input: ActionInput): Promise<ReportDraft>;
  subscribe(fn: Listener): () => void;
}

const localSource: DraftSource = {
  mode: "local",
  async list() {
    loadLocal();
    return summaries;
  },
  async get(id) {
    const found = loadLocal().find((d) => d.draft_id === id);
    if (!found) throw new Error(`未找到报告草稿 ${id}`);
    return found;
  },
  async act(id, action, input) {
    const all = loadLocal();
    const idx = all.findIndex((d) => d.draft_id === id);
    if (idx < 0) throw new Error(`未找到报告草稿 ${id}`);
    const result = applyAction(all[idx]!, action, input, new Date());
    if (!result.ok) throw new Error(result.message);
    drafts = all.map((d, i) => (i === idx ? result.draft : d));
    persistLocal();
    notify();
    return result.draft;
  },
  subscribe(fn) {
    listeners.add(fn);
    return () => listeners.delete(fn);
  },
};

const backendSource: DraftSource = {
  mode: "backend",
  async list() {
    const payload = await getJson<{ reports: BackendSummary[]; count: number }>(
      "/ops/reports?limit=50",
    );
    return payload.reports.map((r) => ({
      draft_id: r.id,
      kind: r.kind,
      title: r.title,
      status: r.status,
      status_label: r.status,
      period: [r.period_start, r.period_end],
      // 列表接口不含正文，因此没有引用数与问题数。
      // 这里写 -1 而不是 0：0 会被读成"没有问题"，而真实情况是"没取到"。
      citation_count: -1,
      problem_count: -1,
    }));
  },
  async get(id) {
    return getJson<ReportDraft>(`/ops/reports/${encodeURIComponent(id)}`);
  },
  async act(id, action, input) {
    const body =
      action === "approve" || action === "reject"
        ? { reviewer: input.reviewer ?? "", note: input.note ?? "" }
        : { actor: input.actor ?? "起草方" };
    return postJson<ReportDraft>(`/ops/reports/${encodeURIComponent(id)}/${action}`, body);
  },
  subscribe() {
    // 后端源没有跨组件推送：动作的返回值就是最新状态，
    // 组件自己 setState 即可。返回一个空订阅让接口统一。
    return () => {};
  },
};

export const draftSource: DraftSource = rehearsal ? localSource : backendSource;

export { formatStamp };
