import type { AlarmReport, ConditionAnswer, ThresholdStatus } from "./types";

import alarmsFixture from "../fixtures/alarms.json";
import conditionFixture from "../fixtures/condition.json";
import thresholdsFixture from "../fixtures/thresholds.json";

/**
 * `/api/ops/*` 客户端。
 *
 * 两种数据源：
 * - `fixture`（默认）：直接读 `src/fixtures/*.json`。那是
 *   `scripts/capture_ops_fixtures.py` 从**真实接口**抓下来的响应，
 *   不是手写的。所以"离线跑起来"看到的就是后端实际会给出的结构。
 * - `live`：打到 `VITE_OPS_BASE`（默认同源 `/api`）。
 *
 * 为什么默认是 fixture：这套界面在头两周是给人看设计的，
 * 而一个"必须先起后端 + 起 Ollama + 建图谱"才能打开的界面，
 * 会让每次评审都卡在环境上。等第 3 周接对话时再默认切 live。
 */

const BASE = import.meta.env.VITE_OPS_BASE ?? "/api";
const SOURCE = import.meta.env.VITE_OPS_SOURCE ?? "fixture";

export const dataSource = SOURCE;

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    // 中文注释：`409` 的 detail 是**对象**（含 problems 列表），
    // 直接当字符串读会得到 "[object Object]" —— 一个让人以为是前端 bug 的报错。
    const raw = await res.text();
    let detail: unknown = raw;
    try {
      detail = JSON.parse(raw);
    } catch {
      /* 保持原文 */
    }
    throw new Error(`${res.status} ${path}: ${JSON.stringify(detail)}`);
  }
  return (await res.json()) as T;
}

export async function fetchThresholds(): Promise<ThresholdStatus> {
  if (SOURCE === "fixture") return thresholdsFixture as ThresholdStatus;
  const res = await fetch(`${BASE}/ops/thresholds`);
  if (!res.ok) throw new Error(`${res.status} /ops/thresholds`);
  return (await res.json()) as ThresholdStatus;
}

export interface AlarmsQuery {
  readings: Array<{
    sensor: string;
    value: number;
    ts: string;
    quality?: string;
  }>;
  period?: { start: string; end: string };
}

/**
 * 取告警。
 *
 * fixture 模式下忽略入参 —— 因为 fixture 本身就是一次真实调用的结果。
 * 这个不对称是刻意的：假装能"用 fixture 重算"只会让人以为改个入参
 * 就能看到不同结果，而它不会。
 */
export async function fetchAlarms(query?: AlarmsQuery): Promise<AlarmReport> {
  if (SOURCE === "fixture") return alarmsFixture as unknown as AlarmReport;
  if (!query) throw new Error("live 模式下必须提供 readings");
  return post<AlarmReport>("/ops/alarms", query);
}

export async function fetchCondition(question: string): Promise<ConditionAnswer> {
  if (SOURCE === "fixture") return conditionFixture as unknown as ConditionAnswer;
  return post<ConditionAnswer>("/ops/condition", { question, readings: [] });
}
