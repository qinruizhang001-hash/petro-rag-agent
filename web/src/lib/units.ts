/**
 * 单位显示。
 *
 * ## 为什么需要一个函数而不是直接拼
 *
 * 测点目录里无量纲的测点（`gw_ph`）`unit` 是 `"-"`。直接拼在数字后面会
 * 得到「7.18-」—— 读起来像一个负数，或者一个被截断的区间。这类字符串
 * 不会报错，只会让人在评审时问一句"这个减号是什么意思"。
 *
 * ## 为什么 "-" 也要认
 *
 * 后端把"没有单位"写成 `"-"` 而不是空串，是因为空串在表格里会塌成 0 宽，
 * 一眼看不出是"没填"还是"填了空的"。界面这一侧知道两者的意思相同，
 * 所以统一收口在这里，而不是每个组件各写一遍 `unit !== "-"`。
 */

/** 需要当成"没有单位"的写法。`1` 是量纲分析里的无量纲记法。 */
const BLANK_UNITS = new Set(["", "-", "—", "–", "1", "无"]);

export function unitLabel(unit: string | null | undefined): string {
  const text = (unit ?? "").trim();
  return BLANK_UNITS.has(text) ? "" : text;
}

/**
 * 数值 + 单位。单位为空时不留尾随空格。
 *
 * `"7.18"` 而不是 `"7.18 "` —— 后者在 `inline-block` 里会多出一格宽度，
 * 表格的对齐会跟着抖。
 */
export function withUnit(value: string, unit: string | null | undefined): string {
  const u = unitLabel(unit);
  return u ? `${value} ${u}` : value;
}
