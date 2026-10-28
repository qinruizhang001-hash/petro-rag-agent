import { test, before } from "node:test";
import assert from "node:assert/strict";
import { mkdirSync } from "node:fs";
import { join } from "node:path";
import { pathToFileURL } from "node:url";

import { build } from "esbuild";

/**
 * 测点全量表（`lib/sensorTable.ts`）的判定测试。
 *
 * ## 为什么这些断言不能靠看截图
 *
 * 表格里每一格都是"算错了也不会报错"的类型：
 *
 * - 余量符号搞反 → 最危险的测点排到最后，而表格看起来完全正常；
 * - 没有阈值被算成余量 0 → 它落在"贴着线"的位置，既像最危险也像最安全，
 *   就是不像"不知道"；
 * - 窗口化边界算错 → 滚到底时最后几行永远不渲染；
 * - `formatMargin` 的尾零裁剪把 `100` 裁成 `1` → 余量 100 bar 与余量 1 bar
 *   是两种工况，而屏幕上只差两个字符。
 *
 * 这些函数是纯的，所以可以直接断言，不需要起浏览器。
 *
 * ## 为什么单独打一个包
 *
 * `sensorTable.ts` 只 import 类型（`import type`），运行时零依赖 —— 但它是
 * `.ts`，Node 直接 import 不了。用 esbuild 打一个 ESM 包再 import，和
 * `test-gap.mjs` / `test-approval.mjs` 同一套做法。
 */

const ENTRY = `
import * as t from "./src/lib/sensorTable";
export { t };
`;

async function bundle() {
  const dir = join(process.cwd(), "node_modules", ".cache", "ccus-sensors");
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
  return import(pathToFileURL(outfile).href);
}

let T;
/** 行高常量。在 `before` 里取 —— 模块顶层取会在包加载前就跑，拿到 undefined。 */
let RH;

before(async () => {
  ({ t: T } = await bundle());
  RH = T.ROW_HEIGHT;
});

// ══════════════════════════════════════════════════════════
// 构造器：只填断言用得到的字段
// ══════════════════════════════════════════════════════════

function alarm(over = {}) {
  const { sensor = "annulus_pressure", value = 18, direction = "high" } = over;
  return {
    alarm_id: `a-${sensor}`,
    sensor,
    label: over.label ?? sensor,
    stage: "injection",
    severity: over.severity ?? "normal",
    severity_label: "正常",
    direction,
    value,
    ts: "2026-10-28T09:00:00+08:00",
    rationale: [],
    attribution: [],
    actions: [],
    threshold_level: "normal",
    threshold_value: null,
    provenance: "sensor",
    quality: "good",
  };
}

function spec(over = {}) {
  const { key = "annulus_pressure", direction = "high" } = over;
  return {
    key,
    label: over.label ?? key,
    unit: over.unit ?? "bar",
    stage: "injection",
    stage_label: over.stage_label ?? "注入",
    direction,
    basis: "",
    threshold:
      over.threshold === undefined
        ? { warn: 20, alert: 30, critical: 40, source: "config/ops_thresholds.json" }
        : over.threshold,
  };
}

function catalog(...sensors) {
  return {
    origin: "config/ops_thresholds.json",
    configured: sensors.filter((s) => s.threshold).map((s) => s.key),
    missing: sensors.filter((s) => !s.threshold).map((s) => s.key),
    sensors,
  };
}

/** 一行表格，省掉每处都写 `rows[0]` */
function one(alarms, thresholds) {
  const rows = T.buildSensorRows(alarms, thresholds);
  assert.equal(rows.length, 1, "这一个用例应当只产生一行");
  return rows[0];
}

// ══════════════════════════════════════════════════════════
// 一、余量的符号：两个方向统一成「负 = 已越线」
// ══════════════════════════════════════════════════════════

test("high 方向：低于 warn 是正余量，高于 warn 是负余量", () => {
  const safe = one([alarm({ value: 18 })], catalog(spec()));
  assert.equal(safe.margin, 2, "warn=20、value=18 应当余 2");

  const over = one([alarm({ value: 23 })], catalog(spec()));
  assert.equal(over.margin, -3, "warn=20、value=23 应当超 3");
});

test("low 方向：高于 warn 是正余量，低于 warn 是负余量", () => {
  const s = spec({ key: "co2_purity", direction: "low", threshold: { warn: 88, alert: 82, critical: 76, source: "x" } });

  const safe = one([alarm({ sensor: "co2_purity", value: 92, direction: "low" })], catalog(s));
  assert.equal(safe.margin, 4, "low 方向 warn=88、value=92 应当余 4");

  const over = one([alarm({ sensor: "co2_purity", value: 85, direction: "low" })], catalog(s));
  assert.equal(over.margin, -3, "low 方向 warn=88、value=85 应当超 3");
});

test("两个方向的正负号含义一致 —— 一列就能横向比较", () => {
  const high = one([alarm({ value: 23 })], catalog(spec()));
  const low = one(
    [alarm({ sensor: "co2_purity", value: 85, direction: "low" })],
    catalog(spec({ key: "co2_purity", direction: "low", threshold: { warn: 88, alert: 82, critical: 76, source: "x" } })),
  );
  assert.ok(high.margin < 0 && low.margin < 0, "两条都越了线，符号必须都是负的");
});

// ══════════════════════════════════════════════════════════
// 二、比例的基准是 warn→alert 间距，不是阈值的百分比
// ══════════════════════════════════════════════════════════

test("比例按 warn→alert 带宽算（high）", () => {
  const r = one([alarm({ value: 25 })], catalog(spec()));
  // warn=20 alert=30 → 带宽 10；余量 20−25 = −5 → −0.5
  assert.equal(r.marginRatio, -0.5);
});

test("比例按 warn→alert 带宽算（low，带宽方向相反）", () => {
  const r = one(
    [alarm({ sensor: "co2_purity", value: 85, direction: "low" })],
    catalog(spec({ key: "co2_purity", direction: "low", threshold: { warn: 88, alert: 82, critical: 76, source: "x" } })),
  );
  // warn=88 alert=82 → 带宽 6；余量 85−88 = −3 → −0.5
  assert.equal(r.marginRatio, -0.5);
});

test("比例的尺子是 warn→alert 带宽，不是阈值的百分比", () => {
  /*
   * co2_purity：warn=95、alert=97、量程上限 100。读数 96 越了 warn 线 1 个
   * 单位，占带宽（2）的一半 → −0.5。
   *
   * 若基准取"warn 的 10%"（9.5），同一个读数只有 −0.105 —— 换一把尺子，
   * 这一列跨测点就不可比了：同样"离 warn 线一格"的两个测点，在 co2_purity
   * 上看起来毫无风险，在 annulus_pressure（warn=20、alert=30）上看起来
   * 逼近极限。而这两条读数对现场的意思是一样的：都刚过线。
   */
  const r = one(
    [alarm({ sensor: "co2_purity", value: 96, direction: "high" })],
    catalog(spec({ key: "co2_purity", threshold: { warn: 95, alert: 97, critical: 100, source: "x" } })),
  );
  assert.equal(r.marginRatio, -0.5, "基准必须是一把随项目缩放的尺子");
});

test("带宽 ≤ 0 时不给比例，而不是给 0", () => {
  const r = one(
    [alarm({ value: 18 })],
    catalog(spec({ threshold: { warn: 20, alert: 20, critical: 20, source: "x" } })),
  );
  assert.equal(r.margin, 2, "余量本身还能算");
  assert.equal(r.marginRatio, null, "0 会被读成「贴着线」，必须给 null");
});

test("没配 alert 也算不出比例，但余量照给", () => {
  const r = one([alarm({ value: 18 })], catalog(spec({ threshold: { warn: 20, alert: null, critical: null, source: "x" } })));
  assert.equal(r.margin, 2);
  assert.equal(r.marginRatio, null);
});

// ══════════════════════════════════════════════════════════
// 三、三种「不知道」互不冒充，也都不冒充「正常」
// ══════════════════════════════════════════════════════════

test("没配阈值的测点：余量是 null，不是 0", () => {
  const r = one([alarm({ value: 18 })], catalog(spec({ threshold: null })));
  assert.equal(r.margin, null, "0 会让它排在「贴着线」的位置");
  assert.equal(r.warn, null);
  assert.equal(r.undecidable, true);
  assert.match(r.undecidableReason, /未配置阈值/);
});

test("目录里没有的测点：既不丢，也不编阈值", () => {
  const rows = T.buildSensorRows([alarm({ sensor: "ghost_sensor", value: 7 })], catalog());
  assert.equal(rows.length, 1, "丢掉这一行等于把一个真实告警藏起来");
  assert.equal(rows[0].margin, null);
  assert.equal(rows[0].warn, null);
  assert.match(rows[0].undecidableReason, /目录/);
});

test("有阈值但没有读数：是「没有可信读数」，不是「未配置阈值」", () => {
  const r = one([alarm({ value: null })], catalog(spec()));
  assert.equal(r.value, null);
  assert.equal(r.margin, null);
  assert.equal(r.undecidable, true);
  assert.match(r.undecidableReason, /读数/);
});

test("正常读数不会被误标成不可判定", () => {
  const r = one([alarm({ value: 18 })], catalog(spec()));
  assert.equal(r.undecidable, false);
  assert.equal(r.undecidableReason, "");
});

test("后端标的 indeterminate 也算不可判定，即使本地算得出余量", () => {
  /*
   * 目前后端只在"没阈值 / 没读数"时标 indeterminate，与本地三条理由重合。
   * 但那个重合不是契约 —— 后端多一条理由（比如质量为 bad）时，表格会显示
   * "状态：不可判定"而底部那句「不可判定的 N 项」数不到它。
   */
  const r = one([alarm({ value: 18, severity: "indeterminate" })], catalog(spec()));
  assert.equal(r.undecidable, true);
  assert.ok(r.undecidableReason.length > 0, "不可判定必须说得出原因");
});

// ══════════════════════════════════════════════════════════
// 四、排序：余量小的在前，判定不了的沉底
// ══════════════════════════════════════════════════════════

test("离 warn 线最近的排在最前", () => {
  const rows = T.buildSensorRows(
    [alarm({ sensor: "a", value: 5 }), alarm({ sensor: "b", value: 19 }), alarm({ sensor: "c", value: 12 })],
    catalog(spec({ key: "a" }), spec({ key: "b" }), spec({ key: "c" })),
  );
  assert.deepEqual(rows.map((r) => r.key), ["b", "c", "a"]);
});

test("不可判定的沉底，不被当成「正好在线上」", () => {
  const rows = T.buildSensorRows(
    [
      alarm({ sensor: "unknown", value: 7 }),
      alarm({ sensor: "close", value: 19 }),
    ],
    catalog(spec({ key: "close" })),
  );
  assert.deepEqual(rows.map((r) => r.key), ["close", "unknown"]);
  assert.equal(rows[1].undecidable, true);
});

test("余量相同时保持后端给的顺序", () => {
  const rows = T.buildSensorRows(
    [alarm({ sensor: "first", value: 19 }), alarm({ sensor: "second", value: 19 })],
    catalog(spec({ key: "first" }), spec({ key: "second" })),
  );
  assert.deepEqual(rows.map((r) => r.key), ["first", "second"]);
});

test("一行都不丢：行数等于告警数", () => {
  const alarms = Array.from({ length: 18 }, (_, i) => alarm({ sensor: `s${i}`, value: 10 + i }));
  const rows = T.buildSensorRows(alarms, catalog());
  assert.equal(rows.length, 18);
});

// ══════════════════════════════════════════════════════════
// 五、窗口化：边界比"能不能显示"更容易写错
// ══════════════════════════════════════════════════════════

test("滚动位置为 0 时从第一行开始", () => {
  assert.equal(T.visibleRange(0, RH * 12, 18).start, 0);
});

test("可视区间覆盖到视口末尾，且带越界余量", () => {
  const { start, end } = T.visibleRange(0, RH * 12, 18);
  assert.ok(end >= 12, `视口 12 行，只渲染到 ${end} 行会留白`);
  assert.ok(end <= 18);
  assert.equal(start, 0);
});

test("滚到底时最后一行在区间里", () => {
  const total = 18;
  const scrollTop = T.bodyHeight(total) - RH * 12;
  const { end } = T.visibleRange(scrollTop, RH * 12, total);
  assert.equal(end, total, "最后一行不渲染时，滚到底会看到一片空白");
});

test("scrollTop 超过内容高度（移动端回弹）被夹住，不返回空区间", () => {
  const { start, end } = T.visibleRange(T.bodyHeight(18) + 500, RH * 12, 18);
  assert.ok(start < 18, "start 超出总行数会让 slice 返回空数组");
  assert.ok(end <= 18);
  assert.ok(end > start);
});

test("负数 scrollTop 被夹成 0", () => {
  assert.deepEqual(T.visibleRange(-40, RH * 12, 18), { start: 0, end: T.visibleRange(0, RH * 12, 18).end });
});

test("零行 / 零行高都返回空区间，不抛错", () => {
  assert.deepEqual(T.visibleRange(0, RH * 12, 0), { start: 0, end: 0 });
  assert.deepEqual(T.visibleRange(0, RH * 12, 18, 0), { start: 0, end: 0 });
});

test("渲染区间永远不会超过总行数", () => {
  for (const top of [0, 50, 300, 1000, 100000]) {
    const { start, end } = T.visibleRange(top, RH * 12, 18);
    assert.ok(start >= 0 && end <= 18 && start <= end, `top=${top} 给出了非法区间`);
  }
});

test("bodyHeight 撑出滚动条", () => {
  assert.equal(T.bodyHeight(18), 18 * RH);
  assert.equal(T.bodyHeight(0), 0);
  assert.equal(T.bodyHeight(-3), 0, "负数行数不能算出负高度");
});

// ══════════════════════════════════════════════════════════
// 六、余量的措辞
// ══════════════════════════════════════════════════════════

test("三位数的余量不会被裁成一位", () => {
  /*
   * 裸的 `toFixed(0).replace(/\.?0+$/, "")` 会把 "100" 变成 "1" ——
   * 正则里 `\.?` 是可选的，`0+$` 于是能直接吃掉整数末尾的零。
   */
  const r = one([alarm({ value: -80 })], catalog(spec({ threshold: { warn: 20, alert: 30, critical: 40, source: "x" } })));
  assert.equal(T.formatMargin(r), "余 100");
});

test("越线说「超」，未越线说「余」", () => {
  assert.match(T.formatMargin(one([alarm({ value: 23 })], catalog(spec()))), /^超 3$/);
  assert.match(T.formatMargin(one([alarm({ value: 18 })], catalog(spec()))), /^余 2$/);
});

test("余量正好为 0 既不是余也不是超", () => {
  const text = T.formatMargin(one([alarm({ value: 20 })], catalog(spec())));
  assert.match(text, /在线上/);
  assert.doesNotMatch(text, /余 0|超 0/);
});

test("算不出来的余量显示破折号，不显示 0", () => {
  assert.equal(T.formatMargin(one([alarm({ value: 18 })], catalog(spec({ threshold: null })))), "—");
});

test("小余量跟着读数的分辨率走，但绝不显示成 0", () => {
  /*
   * 两条判据合起来才对：
   *
   * 1. 读数显示 19.9、阈值 20，余量就该是 0.1 —— 表格里三个数并排，
   *    读者会心算，余量按自己的量级取两位（0.06）会让那行算术对不上。
   * 2. 但读数 19.999 时余量是 0.001，按读数取 1 位得到 `0.0`，而 `0.0`
   *    在这套界面里读作"就在线上"。一个非零的余量显示成零，是这张表最
   *    不能犯的错，所以一路加位到能看见为止。
   */
  assert.equal(T.formatMargin(one([alarm({ value: 19.94 })], catalog(spec()))), "余 0.1");
  assert.equal(T.formatMargin(one([alarm({ value: 19.999 })], catalog(spec()))), "余 0.001");
});

test("越线的动词跟方向走：high 说「超」，low 说「低」", () => {
  // 捕集率读数 86.8 对警告线 88 —— 写「超 1.2」会让人以为两个数里有一个错了
  const low = one(
    [alarm({ value: 86.8, direction: "low" })],
    catalog(
      spec({ direction: "low", threshold: { warn: 88, alert: 80, critical: 75, source: "x" } }),
    ),
  );
  assert.equal(T.formatMargin(low), "低 1.2");
  const high = one([alarm({ value: 23 })], catalog(spec()));
  assert.equal(T.formatMargin(high), "超 3");
});
