import type { Alarm, Direction, Severity, ThresholdStatus } from "./types";

/**
 * 测点全量表的**判定**，与它的渲染分开。
 *
 * ## 这一层为什么单独存在
 *
 * 表格里每一格都是"算错了也不会报错"的类型：
 *
 * * 余量的正负号搞反（`high` 与 `low` 两个方向的减法不同）→ 最危险的测点
 *   排在最后，而表格看起来完全正常；
 * * 没有阈值的测点被算成余量 0 → 它排在"贴着线"的位置，看起来像最危险，
 *   也像最安全，反正不像"不知道"；
 * * 窗口化的边界算错 → 滚到底时最后几行永远不渲染，而那几行可能正是
 *   critical 沉底之后的"已处理"项。
 *
 * 所以它们是纯函数，`scripts/test-sensors.mjs` 直接断言。
 */

export interface SensorRow {
  key: string;
  alarmId: string;
  label: string;
  stageLabel: string;
  unit: string;
  direction: Direction;
  /** `null` 表示没有可信读数 —— 此时不能显示 0 */
  value: number | null;
  /** 测点目录里的 warn 线。`null` = 没配阈值（不是"阈值为 0"） */
  warn: number | null;
  /**
   * 到 warn 线的余量，**正值表示还没到线**。
   *
   * 两个方向的减法不同：`high` 是 `warn − value`（超上限为负），
   * `low` 是 `value − warn`（低于下限为负）。统一成"负 = 已越线"之后，
   * 表格里一列就能横向比较，不必按方向分两张表。
   */
  margin: number | null;
  /**
   * 余量占「warn→alert 间距」的比例。`null` = 算不出来。
   *
   * 基准必须是**一把随项目缩放的尺子**，不能是阈值的百分比：`co2_purity`
   * 的 warn=95、上限 100，按 10% 算出来的基准比量程还高，于是每个合法读数
   * 都会被标成"接近上限"。warn→alert 是项目自己定的一个带宽，天然有量纲。
   */
  marginRatio: number | null;
  severity: Severity;
  /**
   * 判定不了。与"正常"必须分开：没有阈值的测点界面上如果显示成正常，
   * 读者会以为"这个指标没问题"，而真相是"没人说过多少算有问题"。
   */
  undecidable: boolean;
  /** 判定不了的原因，直接显示给读者 */
  undecidableReason: string;
}

/** 一行表格的固定高度（像素）。窗口化要求它是个常量。 */
export const ROW_HEIGHT = 22;

/** 窗口上下各多渲染几行，滚动时不留白。 */
export const OVERSCAN = 4;

function marginOf(
  value: number | null,
  warn: number | null,
  direction: Direction,
): number | null {
  if (value === null || warn === null) return null;
  return direction === "high" ? warn - value : value - warn;
}

/**
 * `warn → alert` 的带宽，用作余量的比较基准。
 *
 * 两个方向的减法同样不同：`high` 时 alert 更大（20 → 30），
 * `low` 时 alert 更小（88 → 82）。带宽 ≤ 0 说明配置本身有问题，
 * 这时**不给比例**，而不是给一个负数或 0 —— 0 会被读成"贴着线"。
 */
function bandOf(
  warn: number | null,
  alert: number | null,
  direction: Direction,
): number | null {
  if (warn === null || alert === null) return null;
  const band = direction === "high" ? alert - warn : warn - alert;
  return band > 0 ? band : null;
}

export function buildSensorRows(
  alarms: Alarm[],
  thresholds: ThresholdStatus,
): SensorRow[] {
  const byKey = new Map(thresholds.sensors.map((s) => [s.key, s]));
  const rows: SensorRow[] = [];

  for (const alarm of alarms) {
    const spec = byKey.get(alarm.sensor);
    // 告警里出现、目录里没有：不能丢掉这一行（那等于把一个真实告警藏起来），
    // 也不能编一个阈值。照实显示"目录里没有这个测点"。
    const warn = spec?.threshold?.warn ?? null;
    const alert = spec?.threshold?.alert ?? null;
    const direction: Direction = spec?.direction ?? alarm.direction;

    const margin = marginOf(alarm.value, warn, direction);
    const band = bandOf(warn, alert, direction);

    let reason = "";
    if (!spec) reason = "测点目录里没有它 —— 阈值无从谈起";
    else if (warn === null) reason = "未配置阈值";
    else if (alarm.value === null) reason = "没有可信读数";

    /*
     * 后端也会标 `indeterminate`。它通常与上面三条中的一条重合，但**不能靠
     * 这个重合**：后端若新增一条判定不可判定的理由（比如质量分为 bad），
     * 这里就会显示"状态：不可判定"而底部那句「不可判定的 N 项」数不到它 ——
     * 两个地方各自都说得通，合起来自相矛盾。
     */
    const backendUndecidable = alarm.severity === "indeterminate";
    const undecidable = reason !== "" || backendUndecidable;
    if (reason === "" && backendUndecidable) {
      reason = "后端判定为不可判定（原因未随测点下发）";
    }

    rows.push({
      key: alarm.sensor,
      alarmId: alarm.alarm_id,
      label: alarm.label,
      stageLabel: spec?.stage_label ?? "",
      unit: spec?.unit ?? "",
      direction,
      value: alarm.value,
      warn,
      margin,
      marginRatio: margin !== null && band !== null ? margin / band : null,
      severity: alarm.severity,
      undecidable,
      undecidableReason: reason,
    });
  }
  return sortRows(rows);
}

/**
 * 排序：余量最小的在前（离 warn 线最近的先看到），判定不了的**沉底**。
 *
 * 沉底而不是按 0 参与排序：把它们排在"贴着线"的位置会让最危险的那几条
 * 被三条"不知道"顶下去；按 0 排又会让它们看起来像"正好在线上"。
 * 两段稳定分区，同段内保持后端给的顺序。
 */
export function sortRows(rows: SensorRow[]): SensorRow[] {
  const known: SensorRow[] = [];
  const unknown: SensorRow[] = [];
  for (const r of rows) (r.margin === null ? unknown : known).push(r);
  // `sort` 在 V8 上是稳定的，同余量时保留原序
  known.sort((a, b) => (a.margin as number) - (b.margin as number));
  return [...known, ...unknown];
}

export interface Range {
  start: number;
  /** 开区间上界，`rows.slice(start, end)` 就是该渲染的那些 */
  end: number;
}

/**
 * 窗口化：滚动位置 → 该渲染的行区间。
 *
 * ## 18 行为什么也要窗口化
 *
 * 不是为了省渲染（18 行怎么渲染都不慢），而是因为**行数是数据属性**：
 * 目录来自 `config/ops_thresholds.json`，换一个测点更密的场地（管线每
 * 10 km 一个测点、多井场）就是几百行，而这块屏的用途是"扫一眼"。
 * 更实际的理由是：窗口化的边界（第一行、最后一行、滚过头）比"能不能
 * 显示"更容易写错，所以它必须是纯函数并且有测试。
 *
 * 越界一律夹住，不抛错也不返回负数区间：滚动容器的 `scrollTop` 在
 * 惯性滚动时会短暂超过内容高度（移动端回弹），那时渲染出空区间会让
 * 表格闪一下空白。
 */
export function visibleRange(
  scrollTop: number,
  viewportHeight: number,
  total: number,
  rowHeight: number = ROW_HEIGHT,
  overscan: number = OVERSCAN,
): Range {
  if (total <= 0 || rowHeight <= 0) return { start: 0, end: 0 };

  const top = Math.max(0, scrollTop);
  const first = Math.floor(top / rowHeight);
  const count = Math.ceil(Math.max(0, viewportHeight) / rowHeight) + 1;

  const start = Math.max(0, Math.min(first - overscan, total - 1));
  const end = Math.max(start, Math.min(total, first + count + overscan));
  return { start, end };
}

/** 表格正文的总高度（像素），用于撑出滚动条。 */
export function bodyHeight(total: number, rowHeight: number = ROW_HEIGHT): number {
  return Math.max(0, total) * Math.max(0, rowHeight);
}

/** 余量显示成给人看的一句话。 */
export function formatMargin(row: SensorRow): string {
  if (row.margin === null) return "—";
  const abs = Math.abs(row.margin);
  const digits = abs >= 100 ? 0 : abs >= 10 ? 1 : 2;
  /*
   * 只有带小数点时才裁尾零。
   *
   * 裸的 `toFixed(d).replace(/\.?0+$/, "")` 会把 `100` 变成 `1` —— 正则里的
   * `\.?` 是可选的，`0+$` 于是能直接吃掉整数末尾的零。而 100 这个量级恰好
   * 是最需要看清的那一档（余量 100 bar 与余量 1 bar 是两种工况）。
   */
  const raw = abs.toFixed(digits);
  const text = raw.includes(".") ? raw.replace(/\.?0+$/, "") : raw;
  // 余量正好为 0 是"就在线上"，既不是余也不是超
  if (abs === 0) return "0（在线上）";
  return row.margin > 0 ? `余 ${text}` : `超 ${text}`;
}
