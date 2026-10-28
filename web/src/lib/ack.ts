/**
 * 告警处置状态（**存在本机，不写后端**）。
 *
 * ## 它是什么、不是什么
 *
 * 真正的处置是跨班次的流程：谁确认的、什么时候、依据什么、要不要留备注，
 * 这套东西必须有审计（后端 `ops/` 侧已有，前端还没有提交通道）。
 * 这里做的是"本机处置标记"：它能在刷新之后还在，**但不能替代那个流程**。
 *
 * ## 为什么必须显式说清存到哪了
 *
 * 一个写着"已确认"的徽标，看起来和真正提交过的处置一模一样。如果界面上
 * 不说清"只在这台机器上、没写进后端"，就会有人在下班交接时指着它说
 * "这个我确认过了" —— 而系统里什么都没有。这条与"阈值没配要报不可判定
 * 而不是正常"是同一条原则：**不要把不知道渲染成已完成**。
 *
 * 所以这里不只存状态，还要把"存成功了没有"暴露给界面：浏览器禁用存储
 * （无痕模式、配额满）时 `setAck` 会失败，此时界面必须说"刷新仍会丢"，
 * 而不是继续显示"已保存"。
 *
 * ## 为什么不需要按报告快照做作用域
 *
 * `alarm_id` 自带时间戳（`A-20261027T1000-capture_rate`），换一个窗口就是
 * 另一批 id。所以"上一次的标记跑到这一次的告警上"在结构上不会发生 ——
 * 不需要额外记 period，也就不会出现"period 对不上于是全丢"这种静默清空。
 *
 * ## 为什么用模块级 store + `useSyncExternalStore`
 *
 * 状态要跨 `AlarmCard` / `MetricCard` / 左栏标题栏三处读，而它们不在
 * 同一条 props 链上（左栏标题栏是 `Panel` 的 aside）。挂一个模块级 store，
 * 三处各自订阅，谁也不用给谁传回调。
 */

import { useSyncExternalStore } from "react";

export type AckState = "acknowledged" | "ignored";

export interface AckEntry {
  state: AckState;
  /** 标记时刻（本地 HH:mm），给人读 */
  at: string;
  /** 标记时刻（epoch ms），给过期清理用 —— `at` 只有时分，判断不了新旧 */
  ts: number;
}

export const ACK_LABEL: Record<AckState, string> = {
  acknowledged: "已确认",
  ignored: "已忽略",
};

/** 徽标配色。字面量，理由同 `severity.ts`。 */
export const ACK_CLASS: Record<AckState, string> = {
  acknowledged: "border-sev-normal/60 text-sev-normal",
  ignored: "border-line-strong text-ink-2",
};

/**
 * 只有文字色。
 *
 * 从 `ACK_CLASS` 里 `split(" ")[1]` 取也行，但那是在解析自己写的类名字符串 ——
 * 哪天有人把两个类换个顺序，取到的就是边框色，而界面上"已确认"会变成
 * 一个没有颜色的词，没有任何报错。两张表各写一次，代价是一个字面量。
 */
export const ACK_TEXT_CLASS: Record<AckState, string> = {
  acknowledged: "text-sev-normal",
  ignored: "text-ink-2",
};

/* ────────────────────────── 序列化（纯函数，可测） ────────────────────────── */

/** 存储键带版本号：将来改结构时不会把旧数据读成新结构。 */
export const ACK_STORAGE_KEY = "ccus-ops.acks.v1";

/** 标记保留 7 天。告警窗口是按天的，一周前的标记没有交接价值，只会撑大存储。 */
export const ACK_TTL_MS = 7 * 24 * 60 * 60 * 1000;

interface Envelope {
  v: 1;
  entries: Record<string, AckEntry>;
}

/**
 * 编码。**只写这三个字段**：调用方塞进来的额外键一律不落盘，
 * 免得将来有人往条目上挂一个对象，把存储撑大还读不回来。
 */
export function encodeAcks(entries: Record<string, AckEntry>): string {
  const clean: Record<string, AckEntry> = {};
  for (const [id, e] of Object.entries(entries)) {
    clean[id] = { state: e.state, at: e.at, ts: e.ts };
  }
  const body: Envelope = { v: 1, entries: clean };
  return JSON.stringify(body);
}

/**
 * 解码。
 *
 * 任何一处不对就**整体丢弃并返回空表**，而不是"尽量恢复"：
 * 一个半恢复的处置表比空表更危险 —— 它会让某几条告警显示成"已处理"，
 * 而没有任何人处理过它们。同时把丢弃原因返回给调用方，好让界面能说清楚
 * 为什么标记没了，而不是让用户以为是"自己记错了"。
 */
export function decodeAcks(
  raw: string | null | undefined,
  now: number,
  ttl = ACK_TTL_MS,
): { entries: Record<string, AckEntry>; dropped: "empty" | "corrupt" | null } {
  if (!raw) return { entries: {}, dropped: null };
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch {
    return { entries: {}, dropped: "corrupt" };
  }
  if (!parsed || typeof parsed !== "object") return { entries: {}, dropped: "corrupt" };
  const env = parsed as Partial<Envelope>;
  if (env.v !== 1 || !env.entries || typeof env.entries !== "object") {
    return { entries: {}, dropped: "corrupt" };
  }

  const entries: Record<string, AckEntry> = {};
  for (const [id, value] of Object.entries(env.entries)) {
    if (!id || !value || typeof value !== "object") continue;
    const { state, at, ts } = value as Partial<AckEntry>;
    if (state !== "acknowledged" && state !== "ignored") continue;
    if (typeof at !== "string" || !at) continue;
    if (typeof ts !== "number" || !Number.isFinite(ts)) continue;
    // 未来时间戳（系统时钟被改过）当过期处理：它无法参与"多久以前"的判断。
    if (now - ts > ttl || ts > now) continue;
    entries[id] = { state, at, ts };
  }
  return { entries, dropped: Object.keys(entries).length === 0 ? "empty" : null };
}

/* ────────────────────────── 运行时 store ────────────────────────── */

const store = new Map<string, AckEntry>();
const listeners = new Set<() => void>();

interface Snapshot {
  entries: Record<string, AckEntry>;
  /** 本机存储是否可用。`false` = 刷新仍会丢，界面必须照实说 */
  storage: boolean;
}

let snapshot: Snapshot = { entries: {}, storage: true };
let hydrated = false;
let lastDrop: "empty" | "corrupt" | null = null;

function storageOrNull(): Storage | null {
  try {
    const s = globalThis.localStorage;
    if (!s) return null;
    // 真写一次：Safari 无痕模式里 localStorage 存在但 setItem 抛错。
    const probe = "__ccus_probe__";
    s.setItem(probe, "1");
    s.removeItem(probe);
    return s;
  } catch {
    return null;
  }
}

function persist() {
  const s = storageOrNull();
  if (!s) {
    if (snapshot.storage) {
      snapshot = { ...snapshot, storage: false };
      for (const l of listeners) l();
    }
    return;
  }
  try {
    s.setItem(ACK_STORAGE_KEY, encodeAcks(Object.fromEntries(store)));
    if (!snapshot.storage) {
      snapshot = { ...snapshot, storage: true };
      for (const l of listeners) l();
    }
  } catch {
    if (snapshot.storage) {
      snapshot = { ...snapshot, storage: false };
      for (const l of listeners) l();
    }
  }
}

function publish() {
  // 每次变更换一个新对象：`useSyncExternalStore` 用 `Object.is` 比较快照，
  // 原地改会让订阅者永远收不到更新（React 认为没变）。
  snapshot = { entries: Object.fromEntries(store), storage: snapshot.storage };
  persist();
  for (const l of listeners) l();
}

/**
 * 首次读盘。幂等，可以放心在每次渲染前调。
 *
 * 不做成模块顶层副作用：模块顶层跑 IO 会让测试与 SSR 都拿到"已经脏了"的状态，
 * 而且 `localStorage` 在 Node 里根本不存在。
 */
export function hydrateAcks(now = Date.now()): void {
  if (hydrated) return;
  hydrated = true;
  const s = storageOrNull();
  if (!s) {
    snapshot = { entries: {}, storage: false };
    return;
  }
  let raw: string | null = null;
  try {
    raw = s.getItem(ACK_STORAGE_KEY);
  } catch {
    snapshot = { entries: {}, storage: false };
    return;
  }
  const { entries, dropped } = decodeAcks(raw, now);
  lastDrop = dropped;
  store.clear();
  for (const [id, entry] of Object.entries(entries)) store.set(id, entry);
  snapshot = { entries: Object.fromEntries(store), storage: true };
}

/** 上一次读盘丢了数据的原因；`null` = 没丢。给界面写原因用。 */
export function ackDropReason(): "empty" | "corrupt" | null {
  return lastDrop;
}

function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

function clockLabel(d: Date): string {
  return `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`;
}

/** 设为某个状态；传 `null` 清除。 */
export function setAck(alarmId: string, state: AckState | null): void {
  if (!alarmId) return;
  hydrateAcks();
  if (state === null) {
    if (!store.delete(alarmId)) return;
  } else {
    const prev = store.get(alarmId);
    if (prev?.state === state) return;
    const now = new Date();
    store.set(alarmId, { state, at: clockLabel(now), ts: now.getTime() });
  }
  publish();
}

/** 清空全部标记（切换数据源、重新载入报告时用）。 */
export function clearAcks(): void {
  hydrateAcks();
  if (store.size === 0) return;
  store.clear();
  publish();
}

/** 非 React 环境读单个状态（测试与脚本用）。 */
export function ackOf(alarmId: string): AckEntry | undefined {
  return store.get(alarmId);
}

/** 订阅整张表。返回值在变更时换新引用，可直接进 `useMemo` 依赖。 */
export function useAcks(): Record<string, AckEntry> {
  hydrateAcks();
  return useSyncExternalStore(
    subscribe,
    () => snapshot.entries,
    () => snapshot.entries,
  );
}

/** 本机存储是否可用。`false` 时界面必须说"刷新仍会丢"。 */
export function useAckStorage(): boolean {
  hydrateAcks();
  return useSyncExternalStore(
    subscribe,
    () => snapshot.storage,
    () => snapshot.storage,
  );
}
