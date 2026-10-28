import { test } from "node:test";
import assert from "node:assert/strict";

import {
  ACK_TTL_MS,
  decodeAcks,
  encodeAcks,
} from "../src/lib/ack.ts";

/**
 * 处置标记的**持久化格式**。
 *
 * 这里测的是纯函数，不是 store：store 要 `localStorage`，而 `node:test` 里
 * 没有浏览器环境。真正会咬人的地方恰好也在纯函数这一侧 —— 读到一份坏数据时
 * "尽量恢复"会让某几条告警显示成"已处理"，而没有任何人处理过它们。
 * 所以规则是**一处不对就整体丢弃**，并且把原因交回去让界面能解释。
 */

const NOW = Date.UTC(2026, 9, 28, 8, 0, 0);
const entry = (over = {}) => ({
  state: "acknowledged",
  at: "16:00",
  ts: NOW - 60_000,
  ...over,
});

test("编码后能原样解回来", () => {
  const src = {
    "A-20261027T1000-capture_rate": entry(),
    "A-20261027T1000-pipeline_dp": entry({ state: "ignored", at: "16:05" }),
  };
  const { entries, dropped } = decodeAcks(encodeAcks(src), NOW);
  assert.equal(dropped, null);
  assert.deepEqual(entries, src);
});

test("没有存储过时不算丢数据", () => {
  for (const raw of [null, undefined, ""]) {
    const { entries, dropped } = decodeAcks(raw, NOW);
    assert.deepEqual(entries, {});
    assert.equal(dropped, null, "第一次打开页面不该被报成「标记丢了」");
  }
});

test("坏 JSON 整体丢弃，并报 corrupt", () => {
  const { entries, dropped } = decodeAcks("{不是 json", NOW);
  assert.deepEqual(entries, {});
  assert.equal(dropped, "corrupt");
});

test("版本号不对就丢弃 —— 否则旧结构会被读成新结构", () => {
  const raw = JSON.stringify({ v: 2, entries: { x: entry() } });
  const { entries, dropped } = decodeAcks(raw, NOW);
  assert.deepEqual(entries, {});
  assert.equal(dropped, "corrupt");
});

test("过期的标记被清掉（7 天前的交接没有价值）", () => {
  const old = entry({ ts: NOW - ACK_TTL_MS - 1 });
  const fresh = entry();
  const raw = encodeAcks({ old: old, fresh: fresh });
  const { entries } = decodeAcks(raw, NOW);
  assert.equal(entries.old, undefined);
  assert.ok(entries.fresh, "没过期的那条不该跟着一起没");
});

test("未来时间戳当过期处理 —— 系统时钟被改过时它无法参与新旧判断", () => {
  const raw = encodeAcks({ x: entry({ ts: NOW + 60_000 }) });
  assert.deepEqual(decodeAcks(raw, NOW).entries, {});
});

test("字段不合法的条目单独剔除，不影响其他条目", () => {
  const raw = JSON.stringify({
    v: 1,
    entries: {
      badState: entry({ state: "done" }),
      badAt: entry({ at: "" }),
      badTs: entry({ ts: "昨天" }),
      noTs: { state: "ignored", at: "16:00" },
      good: entry(),
    },
  });
  const { entries, dropped } = decodeAcks(raw, NOW);
  assert.deepEqual(Object.keys(entries), ["good"]);
  assert.equal(dropped, null, "还剩一条可用的，就不该说「全丢了」");
});

test("全部条目都不可用时报 empty，界面才能说清为什么是空的", () => {
  const raw = JSON.stringify({ v: 1, entries: { x: { state: "nope" } } });
  const { entries, dropped } = decodeAcks(raw, NOW);
  assert.deepEqual(entries, {});
  assert.equal(dropped, "empty");
});

test("编码结果带版本号，且只含 state/at/ts", () => {
  const raw = encodeAcks({ x: entry({ extra: "不该被写进去" }) });
  const parsed = JSON.parse(raw);
  assert.equal(parsed.v, 1);
  assert.deepEqual(Object.keys(parsed.entries.x).sort(), ["at", "state", "ts"]);
});
