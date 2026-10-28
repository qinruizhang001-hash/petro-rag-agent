import { test, before } from "node:test";
import assert from "node:assert/strict";
import { mkdirSync } from "node:fs";
import { join } from "node:path";
import { pathToFileURL } from "node:url";

import { build } from "esbuild";

/**
 * 数值显示的一致性。
 *
 * ## 这里守的是什么
 *
 * 卡片上并排放着三个数：
 *
 * ```
 * 储层监测压力   296bar
 * 阈值 300 · 余量 3.68
 * ```
 *
 * 两个数都没错，但**放在一起就是错的** —— 读者会心算 `296 + 3.68 = 299.68`，
 * 得到 300 以外的答案，然后开始怀疑别的数字。审计的人第一眼就抓这个。
 *
 * 根因是两者用了不同的精度：读数按量级取 0 位，余量按自己的量级取 2 位。
 * 修法是让余量跟**读数**用同一把尺子（`digitsForValue`），代价是 3.68 显示
 * 成 4 —— 在 300 的尺度上是 0.1%，换来屏幕上那行算术真的成立。
 *
 * 下面用一条**性质断言**覆盖全部 fixture 测点：显示出来的读数 ± 显示出来的
 * 余量，必须落在显示出来的那条线上（误差不超过半个末位）。
 */

const ENTRY = `
import * as sev from "./src/lib/severity";
import * as units from "./src/lib/units";
import alarms from "./src/fixtures/alarms.json";
import thresholds from "./src/fixtures/thresholds.json";
export { sev, units, alarms, thresholds };
`;

let M;
/**
 * 在 `before` 里取 —— 模块顶层取会拿到 undefined（包还没 import）。
 * 这个坑在 `test-sensors.mjs` 里已经踩过一次。
 */
let formatValue;
let digitsForValue;
let describeMargin;

before(async () => {
  const dir = join(process.cwd(), "node_modules", ".cache", "ccus-numbers");
  mkdirSync(dir, { recursive: true });
  const outfile = join(dir, "bundle.mjs");
  await build({
    stdin: { contents: ENTRY, resolveDir: process.cwd(), loader: "tsx" },
    outfile,
    bundle: true,
    format: "esm",
    platform: "node",
    logLevel: "error",
  });
  M = await import(pathToFileURL(outfile).href);
  ({ formatValue, digitsForValue, describeMargin } = M.sev);
});

// ══════════════════════════════════════════════════════════
// 一、精度由读数的量级决定
// ══════════════════════════════════════════════════════════

test("三位数不显示 .00，两位数保留一位，小数保留两位", () => {
  assert.equal(formatValue(380), "380");
  assert.equal(formatValue(296.32), "296");
  assert.equal(formatValue(12.74), "12.7");
  assert.equal(formatValue(7.177), "7.18");
});

test("小量级要留到第一位有效数字，不能显示成 0", () => {
  // `toFixed(2).replace(/\.?0+$/, "")` 会把 0.004 变成 "0" ——
  // 一个真实的读数被显示成零，是这张卡最不能犯的错。
  assert.equal(formatValue(0.004), "0.004");
  assert.equal(formatValue(0.0004), "0.0004");
  assert.equal(formatValue(0), "0");
});

test("没有读数显示破折号，不显示 0", () => {
  assert.equal(formatValue(null), "—");
  assert.equal(formatValue(NaN), "—");
});

test("digitsForValue 跟着量级走", () => {
  assert.equal(digitsForValue(380), 0);
  assert.equal(digitsForValue(12.74), 1);
  assert.equal(digitsForValue(7.177), 2);
  assert.equal(digitsForValue(0.004), 4);
  assert.equal(digitsForValue(null), 2);
});

// ══════════════════════════════════════════════════════════
// 二、读数 ± 余量 = 那条线（性质断言）
// ══════════════════════════════════════════════════════════

/** 把 "余量 3.6" / "超 1.2" 解析回**绝对值**。方向由 `breached` 与 `direction` 决定。 */
function parseMagnitude(text) {
  const m = text.match(/^(余量|超) ([\d.]+)$/);
  assert.ok(m, `余量文案的格式变了：${text}`);
  return Number(m[2]);
}

test("屏幕上那行算术必须成立：读数 ± 余量 = 阈值", () => {
  const byKey = new Map(M.thresholds.sensors.map((s) => [s.key, s]));
  let checked = 0;

  for (const alarm of M.alarms.alarms) {
    const spec = byKey.get(alarm.sensor);
    const warn = spec?.threshold?.warn ?? null;
    if (alarm.value === null || warn === null) continue;

    const margin = describeMargin(
      alarm.value,
      warn,
      alarm.threshold_level,
      alarm.threshold_value,
      spec.direction,
    );
    if (margin.value === null) continue;

    const shownValue = Number(formatValue(alarm.value));
    const shownMargin = parseMagnitude(margin.text);
    const shownLine = Number(formatValue(margin.referenceValue));

    /*
     * 读数与那条线的相对位置：high 方向读数是"线 + 余量"，low 方向是"线 − 余量"；
     * 越限之后基准换成被越过的那条线，读数的位置又反过来。四种组合都要对上。
     */
    const dirSign = spec.direction === "high" ? 1 : -1;
    const step = dirSign * (margin.breached ? -1 : 1) * shownMargin;
    const recomputed = shownValue + step;

    const d = digitsForValue(alarm.value);
    const tolerance = 0.5 * 10 ** -d + 1e-9;

    assert.ok(
      Math.abs(recomputed - shownLine) <= tolerance,
      `${alarm.label}：${formatValue(alarm.value)} ${margin.text} 对不上 ` +
        `${margin.referenceLabel} ${formatValue(margin.referenceValue)} —— ` +
        `差 ${Math.abs(recomputed - shownLine)}，容差 ${tolerance}`,
    );
    checked += 1;
  }

  assert.ok(checked >= 10, `只核对了 ${checked} 条，fixture 变了？`);
});

test("余量的小数位跟读数走，不跟自己的量级走", () => {
  // 读数 296.32 按 0 位显示，余量就必须是 4（不是 3.68）
  const m = describeMargin(296.32, 300, "", null, "high");
  assert.equal(m.text, "余量 4");
  assert.equal(Number(formatValue(296.32)) + 4, 300);
});

test("越限时基准换成被越过的那条线，算术同样成立", () => {
  const m = describeMargin(41.6, 40, "critical", 40, "high");
  assert.equal(m.text, "超 1.6");
  assert.equal(m.referenceLabel, "严重线");
  assert.equal(Number(formatValue(41.6)) - 1.6, 40);
});

test("low 方向的余量方向反过来，算术仍成立", () => {
  const m = describeMargin(85.2, 88, "", null, "low");
  assert.equal(m.text, "超 2.8");
  assert.equal(Number(formatValue(85.2)) + 2.8, 88);
});

// ══════════════════════════════════════════════════════════
// 三、单位：无量纲不能渲染成减号
// ══════════════════════════════════════════════════════════

test("无量纲测点不拼单位", () => {
  // pH 的 unit 是 "-"，直接拼会得到「7.18-」—— 像一个负数或被截断的区间
  assert.equal(M.units.unitLabel("-"), "");
  assert.equal(M.units.unitLabel(""), "");
  assert.equal(M.units.unitLabel(null), "");
  assert.equal(M.units.withUnit("7.18", "-"), "7.18");
});

test("有单位时用一个空格分开，且不留尾随空格", () => {
  assert.equal(M.units.withUnit("415", "uS/cm"), "415 uS/cm");
  assert.equal(M.units.unitLabel(" g/(m²·d) "), "g/(m²·d)");
  assert.equal(M.units.withUnit("1.36", "  "), "1.36");
});
