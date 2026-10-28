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
 * 读数从哪来。
 *
 * `/api/ops/*` 是**无状态**的：它不自己去找时序数据，判定所需的 `readings`
 * 必须由调用方送进去。这个设计是对的（后端不必持有历史库），但它把一个
 * 问题留给了界面 —— live 模式有接口地址，却没有读数可送。
 *
 * 曾经的样子是 `fetchAlarms()` 直接 `throw new Error("live 模式下必须提供
 * readings")`，而 `App` 的 `Promise.all` 一挂，整个控制台只剩一行
 * 「数据加载失败」：左栏 0 张告警、剖面 0 个 canvas、右栏 0 个面板。
 * 也就是说 `VITE_OPS_SOURCE=live` 这个被 README 写进用法文档的模式，
 * **一次都没启动成功过**，而 fixture 模式下一切正常，谁也不会发现。
 *
 * 现在的做法：把抓 fixture 时录下的那批读数（`readings.json`，由
 * `scripts/capture_ops_fixtures.py` 生成，不是手写）当作**输入样本**送过去。
 * 这样 live 分支跑的是真接口、真计算，只是输入是回放的。界面上的角标会
 * 如实写成「读数采样回放」，不假装它是实时数据 —— 输入是录的、结果是算的，
 * 这两件事必须分开说。
 */
async function sampledReadings(): Promise<AlarmsQuery["readings"]> {
  // 动态 import：48 KB 只在 live 模式下才拉。fixture 模式下这段代码不会执行，
  // 那个 chunk 也就不会被请求（首屏体积守卫不用为它让路）。
  const mod = (await import("../fixtures/readings.json")) as unknown as {
    default: AlarmsQuery["readings"];
  };
  return mod.default;
}

/**
 * 取告警。
 *
 * fixture 模式下忽略入参 —— 因为 fixture 本身就是一次真实调用的结果。
 * 这个不对称是刻意的：假装能"用 fixture 重算"只会让人以为改个入参
 * 就能看到不同结果，而它不会。
 *
 * live 模式下**仍然允许**调用方自带 `readings`（推送通道或联调时用真实时序），
 * 没带才回落到采样。回落是有日志价值的默认值，不是"猜一个能跑通的输入"。
 */
export async function fetchAlarms(query?: AlarmsQuery): Promise<AlarmReport> {
  if (SOURCE === "fixture") return alarmsFixture as unknown as AlarmReport;
  const body: AlarmsQuery = query ?? { readings: await sampledReadings() };
  return post<AlarmReport>("/ops/alarms", body);
}

/**
 * 工况问答。
 *
 * 送 `readings: []` 在结构上能过（`/ops/condition` 不推时间窗口），但
 * `ops_condition.answer` 拿到空时序只会回一句"没有读数" —— 一个**看起来
 * 成功**的空回答。live 模式下同样要送采样，否则这个面板永远是空的。
 */
export async function fetchCondition(question: string): Promise<ConditionAnswer> {
  if (SOURCE === "fixture") return conditionFixture as unknown as ConditionAnswer;
  return post<ConditionAnswer>("/ops/condition", {
    question,
    readings: await sampledReadings(),
  });
}
