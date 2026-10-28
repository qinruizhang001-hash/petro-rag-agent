import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { build } from "esbuild";

/**
 * 标签避让的单元测试。
 *
 * `placeLabels` 不碰 DOM、不读时间，所以可以直接断言坐标。这很重要：
 * 控制器"没接上"和"接上了但算错"在截图上长得一样 —— 都是标签重叠。
 *
 * 编译产物落在 `node_modules/.cache/` 而不是 `%TEMP%`：Node 按**导入方
 * 所在目录**向上找 `node_modules`，落临时目录会报 `Cannot find package`。
 */

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const cacheDir = join(root, "node_modules", ".cache", "ccus-labels");
mkdirSync(cacheDir, { recursive: true });
const outfile = join(cacheDir, "labels.mjs");

const result = await build({
  entryPoints: [join(root, "src/lib/labels.ts")],
  bundle: true,
  format: "esm",
  outfile,
  write: false,
});
writeFileSync(outfile, result.outputFiles[0].text);
const { placeLabels } = await import(pathToFileURL(outfile).href);

/** 造一个标签：默认 80×24，从锚点向右展开（left 对齐） */
const box = (over = {}) => ({
  dx: 0,
  dy: 0,
  align: "left",
  w: 80,
  h: 24,
  movable: true,
  ...over,
});
const anchor = (id, x, y) => ({ id, x, y, visible: true });
const mapOf = (pairs) => new Map(pairs);

test("不重叠时原样返回，一个像素都不动", () => {
  const anchors = [anchor("a", 0, 0), anchor("b", 0, 100)];
  const boxes = mapOf([
    ["a", box()],
    ["b", box()],
  ]);
  const out = placeLabels(anchors, boxes);
  assert.deepEqual(
    out.map((o) => o.y),
    [0, 100],
  );
});

test("垂直重叠的标签被推开，且推完之后不再相交", () => {
  const anchors = [anchor("a", 0, 0), anchor("b", 0, 10)];
  const boxes = mapOf([
    ["a", box()],
    ["b", box()],
  ]);
  const out = placeLabels(anchors, boxes);
  const a = out.find((o) => o.id === "a");
  const b = out.find((o) => o.id === "b");
  // 高度 24、间距 2 → 第二个的中心至少要落到 12 + 2 + 12 = 26
  assert.ok(b.y >= 26, `b.y=${b.y}，应当被推到 26 以下`);
  assert.ok(Math.abs(b.y - a.y) >= 24, "两个标签的垂直中心仍然太近");
});

test("水平不重叠就不推：深度尺标签不该被中间的测点挤走", () => {
  const anchors = [anchor("m", 0, 0), anchor("ruler", 600, 4)];
  const boxes = mapOf([
    ["m", box()],
    ["ruler", box({ align: "right", dx: -8, movable: false })],
  ]);
  const out = placeLabels(anchors, boxes);
  assert.equal(out.find((o) => o.id === "ruler").y, 4, "深度尺被推动了");
});

test("不可移动的标签只当障碍物，自己不挪", () => {
  const anchors = [anchor("stratum", 0, 0), anchor("m", 0, 6)];
  const boxes = mapOf([
    ["stratum", box({ movable: false })],
    ["m", box()],
  ]);
  const out = placeLabels(anchors, boxes);
  assert.equal(out.find((o) => o.id === "stratum").y, 0, "层名被推离了它标注的边界线");
  assert.ok(out.find((o) => o.id === "m").y > 6, "测点没有被推开");
});

test("推得太远就放弃：宁可重叠，也不让标签飞出画布", () => {
  const anchors = [anchor("a", 0, 0), anchor("b", 0, 2)];
  const boxes = mapOf([
    ["a", box({ h: 200 })],
    ["b", box()],
  ]);
  const out = placeLabels(anchors, boxes, { maxShift: 8 });
  assert.equal(out.find((o) => o.id === "b").y, 2, "超出 maxShift 仍然移动了");
});

test("不改入参", () => {
  const anchors = [anchor("a", 0, 0), anchor("b", 0, 4)];
  const boxes = mapOf([
    ["a", box()],
    ["b", box()],
  ]);
  const snapshot = JSON.stringify(anchors);
  placeLabels(anchors, boxes);
  assert.equal(JSON.stringify(anchors), snapshot, "入参被就地修改了");
});

test("不可见的标签不参与排布，也不会被当成障碍", () => {
  const anchors = [anchor("a", 0, 0), { id: "b", x: 0, y: 0, visible: false }];
  const boxes = mapOf([
    ["a", box()],
    ["b", box()],
  ]);
  const out = placeLabels(anchors, boxes);
  assert.equal(out.find((o) => o.id === "a").y, 0);
  assert.equal(out.find((o) => o.id === "b").y, 0);
});

test("没有尺寸的标签被跳过，不会误推别人", () => {
  const anchors = [anchor("a", 0, 0), anchor("b", 0, 4)];
  const boxes = mapOf([["a", box()]]);
  const out = placeLabels(anchors, boxes);
  assert.equal(out.find((o) => o.id === "b").y, 4);
});

test("固定的标签排在后面也必须被避开（排序要按 movable 优先）", () => {
  // ruler 的 y 比测点大：如果按纯 y 排序，测点先占位、ruler 后处理，
  // 而 ruler 不会动 —— 重叠就留在原地了
  const anchors = [anchor("m", 0, 0), anchor("ruler", 0, 6)];
  const boxes = mapOf([
    ["m", box()],
    ["ruler", box({ movable: false })],
  ]);
  const out = placeLabels(anchors, boxes);
  assert.notEqual(
    out.find((o) => o.id === "m").y,
    0,
    "测点没有让开一个排在它后面的固定标签",
  );
  assert.equal(out.find((o) => o.id === "ruler").y, 6, "固定标签自己动了");
});
