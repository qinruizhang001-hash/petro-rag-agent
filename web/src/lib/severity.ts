import { INDETERMINATE } from "./types";
import type { Alarm, Direction, GradedSeverity, Severity } from "./types";

/**
 * 六级状态的唯一真相。
 *
 * 这里放三件东西：等级的排序、给人看的标签、以及**不可判定为什么特殊**。
 * 三件事分散在组件里会导致同一个状态在两个地方长得不一样，
 * 而深色界面上"看起来差不多"正是最难发现的那种 bug。
 */

export interface SeverityMeta {
  /** 中文标签。与后端 `SEVERITY_LABELS` 保持一致 */
  label: string;
  /** Tailwind 类名后缀，用于 `text-sev-*` / `border-sev-*` */
  token: string;
  /** 排序用。不可判定不参与排序，这里是 `null` */
  rank: number | null;
  /**
   * 是否需要人工动作。`notice` 只是"下个班次复核"，不是告警 ——
   * 把它算进行动级会让左栏顶部常年堆满不该叫人的条目。
   */
  actionable: boolean;
}

export const SEVERITY_META: Record<Severity, SeverityMeta> = {
  normal: { label: "正常", token: "normal", rank: 0, actionable: false },
  notice: { label: "关注", token: "notice", rank: 1, actionable: false },
  warn: { label: "警告", token: "warn", rank: 2, actionable: true },
  alert: { label: "告警", token: "alert", rank: 3, actionable: true },
  critical: { label: "严重", token: "critical", rank: 4, actionable: true },
  /**
   * 第六态。颜色是灰的、边框是虚线、数值显示破折号 —— 三重信号缺一不可。
   * 少任何一重，它都会被读成"正常"，而"把不知道渲染成没事"是这套界面
   * 唯一不会报错的失败。
   */
  indeterminate: {
    label: "不可判定",
    token: "unknown",
    rank: null,
    actionable: false,
  },
};

export function isIndeterminate(severity: Severity): boolean {
  return severity === INDETERMINATE;
}

/**
 * 六级状态的 Tailwind 类名，**写成字面量**。
 *
 * 为什么不能拼：`text-sev-${token}` 在运行时是对的，但 Tailwind 是
 * **编译期扫描源码文本**生成 CSS 的 —— 它扫不到一个模板字符串，
 * 于是那批类名根本不会出现在产物里。症状是颜色全部失效、文字继承父级，
 * 而且没有任何报错：控制台干净、DOM 上有 class、样式就是不生效。
 *
 * 这是这套界面里最容易犯且最难发现的一类错，所以宁可写一张重复的表。
 */
export const SEVERITY_CLASS: Record<
  Severity,
  { text: string; border: string; bg: string; softBg: string; dot: string }
> = {
  normal: {
    text: "text-sev-normal",
    border: "border-sev-normal",
    bg: "bg-sev-normal",
    softBg: "bg-sev-normal/10",
    dot: "bg-sev-normal",
  },
  notice: {
    text: "text-sev-notice",
    border: "border-sev-notice",
    bg: "bg-sev-notice",
    softBg: "bg-sev-notice/10",
    dot: "bg-sev-notice",
  },
  warn: {
    text: "text-sev-warn",
    border: "border-sev-warn",
    bg: "bg-sev-warn",
    softBg: "bg-sev-warn/10",
    dot: "bg-sev-warn",
  },
  alert: {
    text: "text-sev-alert",
    border: "border-sev-alert",
    bg: "bg-sev-alert",
    softBg: "bg-sev-alert/10",
    dot: "bg-sev-alert",
  },
  critical: {
    text: "text-sev-critical",
    border: "border-sev-critical",
    bg: "bg-sev-critical",
    softBg: "bg-sev-critical/10",
    dot: "bg-sev-critical",
  },
  indeterminate: {
    text: "text-sev-unknown",
    border: "border-sev-unknown/60",
    bg: "bg-sev-unknown",
    softBg: "bg-sev-unknown/10",
    dot: "bg-sev-unknown",
  },
};

/** 后端已经算好了最高等级，但前端仍需要一个本地比较（用于分组、排序）。 */
export function worstOf(severities: Severity[]): Severity {
  let best: GradedSeverity | null = null;
  for (const s of severities) {
    const rank = SEVERITY_META[s].rank;
    if (rank === null) continue;
    if (best === null || rank > (SEVERITY_META[best].rank ?? -1)) {
      best = s as GradedSeverity;
    }
  }
  // 一个可判定的都没有：这**不是** normal，是全部不可判定
  return best ?? INDETERMINATE;
}

/** 排序键：不可判定排在所有可判定等级之后（它是"没结论"，不是"结论是好的"）。 */
export function sortKey(severity: Severity): number {
  const rank = SEVERITY_META[severity].rank;
  return rank === null ? -1 : rank;
}

/**
 * 显示用的小数位数 —— **由读数的量级决定，读数与余量共用这一把尺子**。
 *
 * 为什么必须共用：卡片上并排放着三个数
 *
 * ```
 * 储层监测压力   296bar
 * 阈值 300 · 余量 3.68
 * ```
 *
 * 读者会（也应该）心算 `296 + 3.68 = 299.68 ≠ 300`。这两个数都没错，但
 * **放在一起就是错的**：一个按 0 位小数显示，另一个按 2 位。审计的人看到
 * 的第一件事就是"这三个数对不上"，然后开始怀疑别的数字。
 *
 * 所以余量按读数的小数位取整。代价是 3.68 显示成 4（在 300 的尺度上是
 * 0.1%），换来的是屏幕上那行算术真的成立。
 */
export function digitsForValue(value: number | null): number {
  if (value === null || !Number.isFinite(value) || value === 0) return 2;
  const abs = Math.abs(value);
  if (abs >= 100) return 0;
  if (abs >= 10) return 1;
  // 小量级要留到第一位有效数字，否则 0.004 会显示成 0
  if (abs < 0.01) return Math.min(6, -Math.floor(Math.log10(abs)) + 1);
  return 2;
}

/** `toFixed` + 裁尾零。`296.30` → `296.3`，`40.00` → `40`。 */
function fixed(value: number, digits: number): string {
  const text = value.toFixed(digits);
  if (!text.includes(".")) return text;
  return text.replace(/\.?0+$/, "");
}

/**
 * 数值显示。
 *
 * `null` 一律渲染成破折号。**绝不能落成 `0`** ——
 * 一个 0 会被读成"读数是零"，而实际是"没有可信读数"。
 *
 * `digits` 是**小量级的**小数位（默认 2）；大数按量级自动减少，
 * 见 `digitsForValue`。
 */
export function formatValue(value: number | null, digits = 2): string {
  if (value === null || !Number.isFinite(value)) return "—";
  if (value === 0) return "0";
  const abs = Math.abs(value);
  // 大数不保留两位小数（380.00 bar 的 .00 是噪音）；小数至少留到第一位有效数字
  const d = abs >= 100 ? 0 : abs >= 10 ? 1 : abs < 0.01 ? digitsForValue(value) : digits;
  return fixed(value, d);
}

export interface MarginInfo {
  /** 余量文本，例如 "余量 2.4" / "超 1.2" */
  text: string;
  /** 距阈值的余量；`null` 表示算不出来（没有阈值 / 没有读数） */
  value: number | null;
  /** 是否已经越限（用于把余量文字染色） */
  breached: boolean;
  /** 已越过的等级名；空串表示未越限 */
  level: string;
  /** 这次对比的是哪条线；`null` = 没有可对比的线 */
  referenceValue: number | null;
  /** 那条线的标签："阈值" / "警告线" / "告警线" / "严重线" */
  referenceLabel: string;
}

//: 加"线"字是必要的：「严重 40.0 bar」会被读成"严重度是 40 bar"，
//: 而「严重线 40.0 bar」只可能是"严重那条线在 40.0"。
const LEVEL_LABEL: Record<string, string> = {
  warn: "警告线",
  alert: "告警线",
  critical: "严重线",
};

/**
 * 指标卡的"阈值余量"。
 *
 * 只显示数值是不够的：`92.4 %` 单独看无法判断好坏，`余量 2.4` 才能。
 *
 * ## 两个阈值不是同一个东西
 *
 * `alarm.threshold_value` 的含义是**"命中了哪一级的那条线"**，所以：
 * - 越限时它是被越过的那条线（critical 40.0）；
 * - `normal` 时它是 `null` —— 一条都没命中，自然没有"命中的线"。
 *
 * 界面上要显示"这个测点的阈值是多少"，那是**测点目录**里的 `warn`。
 * 第一版把 `alarm.threshold_value` 当阈值用，结果所有 normal 测点都显示
 * 「阈值 — · 无阈值」—— 一个配了阈值、离限值还很远的测点，看起来像没配置。
 * 这与"盲区"在视觉上无法区分，而盲区是这套界面最需要说清的东西。
 *
 * ## 对比基准必须跟着越限等级走
 *
 * 第二版显示「阈值 20.0 bar · 超 critical 1.6 bar」—— 一句话里两个数互相
 * 矛盾：阈值若是 20，41.6 就不是"超 1.6"。所以越限之后，基准线换成被越过
 * 的那条（`严重 40.0 bar · 超 1.6 bar`），读者才知道 1.6 是相对什么算的。
 *
 * ## 越限之后看哪个数
 *
 * 越限之后显示"超出被越过那一级多少"，而不是"距 warn 多少"：
 * 一个 critical 的告警上写"超 warn 21.6"没有意义 —— 读者已经知道它越了 warn。
 */
export function describeMargin(
  value: number | null,
  /** 测点目录里的 warn 阈值；`null` = 未配置 */
  warnThreshold: number | null,
  /** `alarm.threshold_level`："" / "notice" / "warn" / "alert" / "critical" */
  crossedLevel: string,
  /** `alarm.threshold_value`：被越过那条线的位置 */
  crossedValue: number | null,
  direction: Direction,
): MarginInfo {
  const none: MarginInfo = {
    text: "无读数",
    value: null,
    breached: false,
    level: "",
    referenceValue: null,
    referenceLabel: "阈值",
  };

  if (value === null) return none;

  /*
   * 余量的小数位跟**读数**走，不跟余量自己的量级走。
   *
   * 余量是"两个显示出来的数之差"，所以它必须和那两个数在同一把尺子上：
   * 读数 296（0 位）、阈值 300（0 位），余量就该是 4 —— 而不是 3.68。
   * 反过来让余量自己决定精度（3.68），屏幕上那行算术就永远对不上。
   */
  const d = digitsForValue(value);

  // 已越限：基准换成被越过的那条线。notice 不算"越过"，它只是进了提示带。
  if (crossedLevel && crossedLevel !== "notice" && crossedValue !== null) {
    const past =
      direction === "high" ? value - crossedValue : crossedValue - value;
    return {
      text: `超 ${fixed(Math.abs(past), d)}`,
      value: past,
      breached: true,
      level: crossedLevel,
      referenceValue: crossedValue,
      referenceLabel: LEVEL_LABEL[crossedLevel] ?? crossedLevel,
    };
  }

  if (warnThreshold === null) {
    return { ...none, text: "未配置阈值" };
  }

  // 未越限：基准是 warn 线。notice 也走这条 —— 它还在 warn 之内。
  const gap = direction === "high" ? warnThreshold - value : value - warnThreshold;
  return {
    text: gap >= 0 ? `余量 ${fixed(gap, d)}` : `超 ${fixed(-gap, d)}`,
    value: gap,
    breached: gap < 0,
    level: gap < 0 ? "warn" : "",
    referenceValue: warnThreshold,
    referenceLabel: "阈值",
  };
}

/**
 * 看板上的排列顺序：严重 → 告警 → 警告 → **不可判定** → 关注 → 正常。
 *
 * 不可判定夹在中间，两个方向都不能挪：
 * - 排到最前，它会把真正的严重告警挤下去 —— 而"没结论"不该优先于"结论很糟"；
 * - 排到最后，它等于被归类成"最不重要"，可它其实是"这里有一块我们没看见"。
 *
 * 显式写出每一级而不是拿 `rank` 去加减，是因为这个顺序**不是**严重度的线性
 * 映射（不可判定压根不在那个序列里），用偏移量凑出来的数会在下一次加等级时错位。
 */
export function rankForDisplay(severity: Severity): number {
  switch (severity) {
    case "critical":
      return 5;
    case "alert":
      return 4;
    case "warn":
      return 3;
    case INDETERMINATE:
      return 2.5;
    case "notice":
      return 2;
    case "normal":
      return 1;
  }
}

export function sortAlarms(alarms: Alarm[]): Alarm[] {
  return [...alarms].sort((a, b) => {
    const d = rankForDisplay(b.severity) - rankForDisplay(a.severity);
    if (d !== 0) return d;
    return a.sensor.localeCompare(b.sensor);
  });
}
