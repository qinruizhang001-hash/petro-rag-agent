#!/usr/bin/env node
/**
 * 降档判定的单元测试。
 *
 * ## 为什么要为一个"六行的判定函数"写测试
 *
 * 因为它是**唯一**能保证 60 FPS 的那段逻辑。它写错了不会报错、不会崩、
 * 界面上也看不出区别 —— 只会在某台集显机器上一直掉帧，而没人知道
 * 那个本该发生的降级从来没发生。这类"测不到的控制器"和"没接上的控制器"
 * 在代码审查里长得一模一样。
 *
 * ## 为什么用 node 内置的 test runner
 *
 * 这个仓库没有前端测试框架（没有 vitest / jest），为六个断言引入一整套
 * 测试基建不划算。Node 22 自带 `node:test` + `node:assert`，并且能直接
 * 执行 `.ts`（类型剥离，Node 22.18+ 默认开启）。
 *
 * 代价是：只能测**纯函数**。`quality.ts` 没有任何运行时 import，
 * 正好满足这个约束 —— 这不是巧合，把判定抽成纯函数就是为了让它可测。
 *
 * ```bash
 * node --test scripts/test-quality.mjs
 * ```
 */

import assert from "node:assert/strict";
import { test } from "node:test";

import {
  DEGRADE_ORDER,
  QUALITY_PARTICLES,
  QUALITY_PIXEL_RATIO_CAP,
  QUALITY_VOLUME,
  degrade,
  isQuality,
  nextQuality,
} from "../src/three/quality.ts";

/** 真实调用点用的阈值（`FRAME_BUDGET_MS * 1.15`），在这里写死以免漂移。 */
const THRESHOLD = (1000 / 60) * 1.15;

test("超阈值时降一档", () => {
  assert.equal(nextQuality("high", 25, THRESHOLD), "medium");
  assert.equal(nextQuality("medium", 25, THRESHOLD), "low");
});

test("已经在最低档时返回 null，不会无限降下去", () => {
  // 这条是关键：没有终点的话，"一直降"是一个逻辑上永不结束的过程，
  // 而每一档都有副作用（改 pixelRatio、关图元）。
  assert.equal(nextQuality("low", 25, THRESHOLD), null);
  assert.equal(nextQuality("low", 9999, THRESHOLD), null);
});

test("达标时不动", () => {
  assert.equal(nextQuality("high", 10, THRESHOLD), null);
  assert.equal(nextQuality("high", 16.7, THRESHOLD), null);
});

test("正好等于阈值不降级", () => {
  // 用 `>` 而不是 `>=`：卡在阈值上的机器（例如 vsync 锁 60 Hz 时
  // p50 恰好 16.67）不该被判为慢，否则每一台正常机器都会降到 low。
  assert.equal(nextQuality("high", THRESHOLD, THRESHOLD), null);
});

test("NaN / 负数不会触发降级", () => {
  // 采样窗口里出现过 0 或负数（时钟回拨、标签页恢复）时，
  // 判定必须是"不动"，而不是"降级" —— 否则一次异常会让整块屏永久降档。
  assert.equal(nextQuality("high", Number.NaN, THRESHOLD), null);
  assert.equal(nextQuality("high", -5, THRESHOLD), null);
});

test("degrade 链与 DEGRADE_ORDER 一致", () => {
  assert.equal(degrade("high"), "medium");
  assert.equal(degrade("medium"), "low");
  assert.equal(degrade("low"), null);
  assert.deepEqual([...DEGRADE_ORDER], ["high", "medium", "low"]);
});

test("档位越低，pixelRatio 上限单调不增", () => {
  const caps = DEGRADE_ORDER.map((q) => QUALITY_PIXEL_RATIO_CAP[q]);
  for (let i = 1; i < caps.length; i++) {
    assert.ok(
      caps[i] <= caps[i - 1],
      `${DEGRADE_ORDER[i]} 的 pixelRatio 上限 ${caps[i]} 高于上一档 ${caps[i - 1]}`,
    );
  }
});

test("最低档关掉的是装饰，不是数据", () => {
  // 这条是产品红线，不是性能参数：low 档允许关掉粒子和 3D 包络体，
  // 但剖面轮廓、等压环、测点球必须都在（它们不在这些表里，所以只要
  // 这两张表为 false 就说明降级范围没有越界）。
  assert.equal(QUALITY_PARTICLES.low, false);
  assert.equal(QUALITY_VOLUME.low, false);
  // 高/中档两件装饰都保留，否则"medium"看起来会和"low"一样
  assert.equal(QUALITY_PARTICLES.medium, true);
  assert.equal(QUALITY_VOLUME.medium, true);
});

test("三张配置表的键完全一致", () => {
  // 新增一个档位时最容易漏配其中一张表，而漏配的结果是 `undefined` ——
  // `visible = undefined` 在 three 里等于 false，图元静默消失。
  const keys = [...DEGRADE_ORDER].sort();
  assert.deepEqual(Object.keys(QUALITY_PIXEL_RATIO_CAP).sort(), keys);
  assert.deepEqual(Object.keys(QUALITY_PARTICLES).sort(), keys);
  assert.deepEqual(Object.keys(QUALITY_VOLUME).sort(), keys);
});

test("isQuality 只认三档", () => {
  assert.equal(isQuality("high"), true);
  assert.equal(isQuality("low"), true);
  assert.equal(isQuality("ultra"), false);
  assert.equal(isQuality(null), false);
  assert.equal(isQuality(""), false);
});
