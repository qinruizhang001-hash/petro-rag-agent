/**
 * 告警处置状态（**仅本页**）。
 *
 * ## 为什么它不写后端
 *
 * 真正的处置是跨班次的流程：谁确认的、什么时候、依据什么、要不要留备注，
 * 这套东西必须有审计（后端 `ops/` 侧已有，前端还没有提交通道）。
 * 在这里先做"本页标记"是为了演示时能区分"已经有人看过了"和"还没人看"，
 * 不是为了替代那个流程。
 *
 * ## 所以它必须**显式**说自己是本页的
 *
 * 一个写着"已确认"的徽标，看起来和真正提交过的处置一模一样。
 * 如果界面上不说清"刷新即失效、没有写进后端"，就会有人在下班交接时
 * 指着它说"这个我确认过了" —— 而系统里什么都没有。这条与"阈值没配要报
 * 不可判定而不是正常"是同一条原则：**不要把不知道渲染成已完成**。
 *
 * ## 为什么不用 useState 放在 LeftRail 里
 *
 * 状态要跨 `AlarmCard` / `MetricCard` / 左栏标题栏三处读，而它们不在
 * 同一条 props 链上（左栏标题栏是 `Panel` 的 aside）。用 `useSyncExternalStore`
 * 挂一个模块级 store，三处各自订阅，谁也不用给谁传回调。
 */

import { useSyncExternalStore } from "react";

export type AckState = "acknowledged" | "ignored";

export interface AckEntry {
  state: AckState;
  /** 标记时刻（本地 HH:mm） */
  at: string;
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

const store = new Map<string, AckEntry>();
const listeners = new Set<() => void>();
let snapshot: Record<string, AckEntry> = {};

function publish() {
  // 每次变更换一个新对象：`useSyncExternalStore` 用 `Object.is` 比较快照，
  // 原地改会让订阅者永远收不到更新（React 认为没变）。
  snapshot = Object.fromEntries(store);
  for (const l of listeners) l();
}

function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

function clockLabel(): string {
  const d = new Date();
  return `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`;
}

/** 设为某个状态；传 `null` 清除。 */
export function setAck(alarmId: string, state: AckState | null): void {
  if (!alarmId) return;
  if (state === null) {
    if (!store.delete(alarmId)) return;
  } else {
    const prev = store.get(alarmId);
    if (prev?.state === state) return;
    store.set(alarmId, { state, at: clockLabel() });
  }
  publish();
}

/** 清空全部标记（切换数据源、重新载入报告时用）。 */
export function clearAcks(): void {
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
  return useSyncExternalStore(subscribe, () => snapshot, () => snapshot);
}
