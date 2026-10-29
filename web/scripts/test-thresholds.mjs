import { test, before } from "node:test";
import assert from "node:assert/strict";
import { mkdirSync } from "node:fs";
import { join } from "node:path";
import { pathToFileURL } from "node:url";

import { build } from "esbuild";

/**
 * 「这些线是谁定的？」
 *
 * ## 这里守的是什么
 *
 * 界面上每个读数旁边都有一句「余量 4」「低 1.2」。读者会默认那条线是这个
 * 项目的设计限值。本地跑的时候它不是 —— 仓库里只有
 * `config/ops_thresholds.example.json`，每个 source 都写着「【替换】」。
 *
 * 这是一种最难发现的错：18 个测点全部"有阈值"、分级看着合理、余量算得对，
 * 但那些线是占位值。**算得对，判据是假的。**
 *
 * 它和"未配置阈值"不是一回事：后者界面显示「18 项不可判定」，很吵，不会
 * 被忽略。占位值相反 —— 它安静得像一切正常。
 *
 * ## 判据看内容，不看文件名
 *
 * 有人会把 example 复制成 `ops_thresholds.json` 再填真值（文件名就叫
 * ops_thresholds.json）；反过来，一份真实文件里也可能残留「【替换】」。
 * 所以两件事分开报，只有"内容里有【替换】"才描虚线警告。
 */

const ENTRY = `
import * as T from "./src/lib/thresholds";
import real from "./src/fixtures/thresholds.json";
export { T, real };
`;

let describeThresholds;
let real;

before(async () => {
  const dir = join(process.cwd(), "node_modules", ".cache", "ccus-thresholds");
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
  const mod = await import(pathToFileURL(outfile).href);
  describeThresholds = mod.T.describeThresholds;
  real = mod.real;
});

/** 造一份阈值目录。`srcs` 决定每个测点的 source 文案。 */
function catalog(srcs, origin = "config/ops_thresholds.json") {
  return {
    origin,
    configured: srcs.filter((s) => s !== null).map((_, i) => `s${i}`),
    missing: srcs.filter((s) => s === null).map((_, i) => `m${i}`),
    sensors: srcs.map((src, i) => ({
      key: `s${i}`,
      label: `测点${i}`,
      unit: "bar",
      stage: "capture",
      stage_label: "捕集",
      direction: "high",
      basis: "",
      threshold:
        src === null
          ? null
          : { warn: 10, alert: 20, critical: 30, source: src },
    })),
  };
}

const REAL_SRC = "项目可研报告 · 捕集单元性能保证值";
const FAKE_SRC = "【替换】项目可研报告 · 捕集单元性能保证值";

test("全占位：必须报出，并给出去哪改", () => {
  const d = describeThresholds(catalog([FAKE_SRC, FAKE_SRC]));
  assert.equal(d.needsAttention, true);
  assert.match(d.label, /2\/2 项为占位值/);
  // 悬停说明要同时说清"哪来的""为什么可疑""怎么改"
  assert.match(d.hint, /ops_thresholds\.example\.json/);
  assert.match(d.hint, /算得对，但判据是假的/);
  assert.match(d.hint, /config\/ops_thresholds\.json/);
});

test("部分占位只报那几项，不把全部喊成假的", () => {
  const d = describeThresholds(
    catalog([FAKE_SRC, REAL_SRC, REAL_SRC, FAKE_SRC, REAL_SRC]),
  );
  assert.equal(d.placeholderCount, 2);
  assert.equal(d.total, 5);
  assert.match(d.label, /2\/5/);
});

test("内容干净、文件名仍是 example：不警告，但要说清来自模板", () => {
  const d = describeThresholds(
    catalog([REAL_SRC], "config/ops_thresholds.example.json"),
  );
  // 把一份填好的模板喊成"假的"，下次真出问题时没人会看
  assert.equal(d.needsAttention, false);
  assert.match(d.label, /示例模板/);
  assert.match(d.hint, /改名为 config\/ops_thresholds\.json/);
});

test("内容干净、文件名也干净：一个字都不显示", () => {
  const d = describeThresholds(catalog([REAL_SRC]));
  assert.equal(d.needsAttention, false);
  assert.equal(d.label, "");
  assert.equal(d.hint, "");
});

test("目录名里带 example 不算模板文件", () => {
  // 只看最后一段文件名。`example_dir/ops_thresholds.json` 是真实文件。
  const d = describeThresholds(catalog([REAL_SRC], "example_dir/ops_thresholds.json"));
  assert.equal(d.label, "");
});

test("没配阈值的测点不参与占位计数，但算进总数", () => {
  // null = 已声明没配线。它由「不可判定」那一路负责，不该被算成占位值。
  const d = describeThresholds(catalog([FAKE_SRC, null, null]));
  assert.equal(d.placeholderCount, 1);
  assert.equal(d.total, 3);
  assert.match(d.label, /1\/3/);
});

test("全部没配阈值：不报占位（那是另一条路的事）", () => {
  const d = describeThresholds(catalog([null, null]));
  assert.equal(d.placeholderCount, 0);
  assert.equal(d.label, "");
});

test("thresholds 为 null（还没加载）不抛错也不显示", () => {
  const d = describeThresholds(null);
  assert.equal(d.needsAttention, false);
  assert.equal(d.label, "");
  assert.equal(d.total, 0);
});

test("fixture 里那份真实响应确实会被判成占位值", () => {
  // 这条不是重复：它守的是"判据对**实际数据**成立"，而不只是对我造的样本
  // 成立。fixture 换一份、source 文案一改，上面八条可能全绿而这条红。
  const d = describeThresholds(real);
  assert.ok(d.total > 0, "fixture 里应该有测点");
  assert.equal(d.needsAttention, true, "仓库里的示例阈值必须被判成占位值");
  assert.match(d.origin, /example/);

  /*
   * 占位数 = 总数 − 没配线的数量，而不是"等于总数"。
   *
   * 这份 fixture 的 origin 里写着「派生：删去 3 个浅层 MVA 测点」—— 那 3 个
   * （soil_co2_flux / gw_conductivity / surface_heave）的 threshold 是 null。
   * 它们的判据是"没有线"，由「不可判定」那一路负责；把它们算成"占位值"
   * 会让底栏报 18/18，而实际只有 15 项是可核对的假线。
   *
   * 写成关系而不是 15 这个数字：fixture 增删测点时断言仍然成立，而"把
   * 没配线的算进占位"这个错误仍然会被抓到。
   */
  const unconfigured = real.sensors.filter((s) => !s.threshold).length;
  assert.ok(unconfigured > 0, "这份 fixture 里确实有没配阈值的测点");
  assert.equal(d.placeholderCount, d.total - unconfigured);
  assert.ok(d.placeholderCount < d.total);
});
