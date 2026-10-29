/**
 * 「这些线是谁定的？」
 *
 * 界面上每个读数旁边都有一句「余量 4」或「低 1.2」，读者会默认那条线是
 * **这个项目的设计限值**。本地跑的时候它多半不是 —— `config/ops_thresholds.json`
 * 不进仓库（里面是某个项目的真实设计值），仓库里只有
 * `config/ops_thresholds.example.json`，每个 `source` 里都写着「【替换】」。
 *
 * 于是出现一种最难发现的错：18 个测点全部"有阈值"、分级看着合理、
 * 余量算得对，但那些线是别人填的占位值。**算得对，判据是假的。**
 *
 * 这与"未配置阈值"不是同一件事 —— 那个界面已经会显示「18 项不可判定」，
 * 很吵，不会被忽略。占位值相反：它安静得像一切正常。
 *
 * ## 判据看内容，不看文件名
 *
 * `origin` 里的文件名只说明**用了哪个文件**：有人会把 example 复制成
 * `ops_thresholds.json` 再填真值，那份文件名就叫 ops_thresholds.json。
 * 反过来，一份真实文件里也可能残留「【替换】」。
 *
 * 所以两件事分开报：
 *
 * - `placeholder`：任一 `source` 里出现「【替换】」→ **确定**是占位值。
 * - `template`：文件名含 `example` → 用了模板，可能已填、也可能没填。
 *
 * 只有 `placeholder` 才值得用虚线边框警告；`template` 只写清"来自模板"，
 * 因为把一份填好的模板喊成"假的"，下次真出问题时没人会看。
 */

import type { ThresholdStatus } from "./types";

export interface ThresholdOrigin {
  /** 是否需要提示。`false` 时界面上什么都不渲染。 */
  needsAttention: boolean;
  /** 底栏那一格的字。 */
  label: string;
  /** 悬停说明：谁定的、为什么可疑、怎么办。 */
  hint: string;
  /** 含「【替换】」的测点数。 */
  placeholderCount: number;
  /** 声明的测点总数。 */
  total: number;
  /** 阈值文件路径（原样，不裁剪 —— 路径本身就是证据）。 */
  origin: string;
}

/** 占位标记。后端与示例文件都用这两个字符，改文案要同步改后端。 */
const PLACEHOLDER_MARK = "【替换】";

function isTemplateFile(origin: string): boolean {
  // 只看路径最后一段的文件名，避免目录名里带 example 造成误判。
  const name = origin.split(/[\\/]/).pop() ?? origin;
  return /example/i.test(name);
}

export function describeThresholds(
  thresholds: ThresholdStatus | null,
): ThresholdOrigin {
  const origin = thresholds?.origin ?? "";
  const sensors = thresholds?.sensors ?? [];
  const total = sensors.length;
  const placeholderCount = sensors.filter(
    (s) => s.threshold?.source?.includes(PLACEHOLDER_MARK) ?? false,
  ).length;

  if (!thresholds || total === 0) {
    return {
      needsAttention: false,
      label: "",
      hint: "",
      placeholderCount: 0,
      total: 0,
      origin,
    };
  }

  if (placeholderCount > 0) {
    return {
      needsAttention: true,
      label: `阈值 ${placeholderCount}/${total} 项为占位值`,
      hint:
        `这些线来自 ${origin}，其中 ${placeholderCount} 个测点的出处写着「${PLACEHOLDER_MARK}」—— ` +
        `那是模板里的占位值，不是本项目的设计限值。\n` +
        `读数、余量、分级都按这些线算，算得对，但判据是假的。\n` +
        `换成真实限值：把 config/ops_thresholds.example.json 复制成 config/ops_thresholds.json 并逐项填写。`,
      placeholderCount,
      total,
      origin,
    };
  }

  if (isTemplateFile(origin)) {
    return {
      needsAttention: false,
      label: `阈值：来自示例模板`,
      hint:
        `${origin} 是仓库里的模板文件。它的 source 字段已不再含「${PLACEHOLDER_MARK}」，` +
        `说明有人填过 —— 但文件名仍叫 example。\n` +
        `如果这些确实是本项目的设计限值，改名为 config/ops_thresholds.json 以免下次被当成模板覆盖。`,
      placeholderCount: 0,
      total,
      origin,
    };
  }

  return {
    needsAttention: false,
    label: "",
    hint: "",
    placeholderCount: 0,
    total,
    origin,
  };
}
