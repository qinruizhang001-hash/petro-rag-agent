#!/usr/bin/env node
/**
 * 文本清洗的单元测试。
 *
 * `readable()` 只有一行，但它的**范围**很容易被后人扩大成整串 NFKC ——
 * 那会把 `m³/d` 变成 `m3/d`，而这套界面上单位写错比豆腐块更糟：
 * 豆腐块一眼看出是显示问题，`m3/d` 看起来像一个正常的单位。
 *
 * 所以这里同时锁两件事：该修的修掉，**不该动的不要动**。
 */

import assert from "node:assert/strict";
import { test } from "node:test";

import { readable } from "../src/lib/text.ts";

test("数学斜体还原成 ASCII", () => {
  // 𝐶 = U+1D436，𝑥 = U+1D465，𝑡 = U+1D461
  assert.equal(readable("𝑡𝑡+1 = 𝐶𝐶(𝑥𝑥𝑡𝑡)"), "tt+1 = CC(xxtt)");
});

test("数学粗体 / 双线体 / 数字也还原", () => {
  assert.equal(readable("𝐀𝐁"), "AB");
  assert.equal(readable("𝔸𝔹"), "AB");
  assert.equal(readable("𝟏𝟐𝟑"), "123");
});

test("单位的上标不能被吃掉", () => {
  // 这条是核心：整串 NFKC 会把 m³ 变成 m3，而那是另一个写法。
  assert.equal(readable("m³/d"), "m³/d");
  assert.equal(readable("25 ℃"), "25 ℃");
  assert.equal(readable("CO₂"), "CO₂");
});

test("解码失败的占位符被移除", () => {
  assert.equal(readable("�abc"), "abc");
});

test("普通中英文原样返回", () => {
  const s = "盖层完整性：实测破裂压力 41.6 bar，超过阈值 1.6 bar。";
  assert.equal(readable(s), s);
});
