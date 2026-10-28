/**
 * 本机草稿存储的编解码。
 *
 * 这个文件**只有纯函数**，不 import `api.ts`、不读 `localStorage`。
 * 分开的理由很具体：`api.ts` 在模块顶层读 `import.meta.env`，那是 Vite
 * 提供的；在 `node --test` 里它是 `undefined`，一个 `import` 就会抛
 * `Cannot read properties of undefined`。把编解码留在同一个文件里，
 * 这些函数就永远测不了 —— 而它们恰好是"刷新之后审批轨迹还在不在"的
 * 全部逻辑。
 */
import type { ReportDraft } from "./approval";

export const DRAFT_STORAGE_KEY = "ccus-ops.drafts.v1";

export type DropReason = "empty" | "corrupt" | "stale";

/**
 * 种子内容的指纹（FNV-1a，32 位）。
 *
 * 只取"结构"不取正文：正文一改指纹就变，会让每次重新生成 fixture 都
 * 丢掉用户在本机做的演练 —— 而那恰好是演示前最不该发生的事。
 * 结构（草稿号 / 状态 / 轨迹长度 / 问题数）变了才说明种子换了。
 */
export function seedFingerprint(drafts: ReportDraft[]): string {
  const key = drafts
    .map((d) => `${d.draft_id}:${d.status}:${d.trail.length}:${d.problems.length}`)
    .join("|");
  let h = 0x811c9dc5;
  for (let i = 0; i < key.length; i += 1) {
    h ^= key.charCodeAt(i);
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h.toString(16).padStart(8, "0");
}

interface Envelope {
  v: 1;
  seed: string;
  drafts: ReportDraft[];
}

export type DecodeOutcome =
  | { ok: true; drafts: ReportDraft[] }
  | { ok: false; reason: DropReason };

/**
 * 解出本机草稿。
 *
 * 坏数据**整体丢弃**，不做逐条抢救：一份只恢复了一半的审批表，会让某几条
 * 显示成「已批准」而实际没人批过 —— 那比空表危险得多。空表至少会让人
 * 重新走一遍流程。
 *
 * 种子换了也整体丢弃：留着旧草稿会让界面继续显示上一版的正文，
 * 而看不出任何异常 —— 它只是"看起来还对"。
 */
export function decodeDrafts(raw: string | null, seed: string): DecodeOutcome {
  if (!raw) return { ok: false, reason: "empty" };
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch {
    return { ok: false, reason: "corrupt" };
  }
  if (!parsed || typeof parsed !== "object") return { ok: false, reason: "corrupt" };
  const env = parsed as Partial<Envelope>;
  if (env.v !== 1) return { ok: false, reason: "corrupt" };
  if (!Array.isArray(env.drafts)) return { ok: false, reason: "corrupt" };
  if (env.seed !== seed) return { ok: false, reason: "stale" };

  for (const d of env.drafts) {
    if (!d || typeof d !== "object") return { ok: false, reason: "corrupt" };
    if (typeof d.draft_id !== "string" || !d.draft_id) {
      return { ok: false, reason: "corrupt" };
    }
    if (typeof d.status !== "string") return { ok: false, reason: "corrupt" };
    if (!Array.isArray(d.trail) || !Array.isArray(d.problems)) {
      return { ok: false, reason: "corrupt" };
    }
  }
  return { ok: true, drafts: env.drafts as ReportDraft[] };
}

export function encodeDrafts(drafts: ReportDraft[], seed: string): string {
  const env: Envelope = { v: 1, seed, drafts };
  return JSON.stringify(env);
}
