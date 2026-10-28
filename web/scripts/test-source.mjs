/**
 * 数据源选择的判定。
 *
 * 这里测的是"一份产物能不能同时服务两种场景"，以及**覆盖到底生效没有**。
 *
 * 为什么值得单独一组测试：这件事的全部失败方式都不报错。
 * `?source=lvie` 拼错 → 界面照旧显示 fixture 快照，看着一切正常，
 * 而讲的人以为自己在演示"接了后端"。`?source=LIVE` 同理。
 * 所以判据必须是"拼错的值不许被猜成 live"，而不是"能切过去就行"。
 */

import assert from "node:assert/strict";
import { test } from "node:test";

import {
  buildSource,
  choiceFromLocation,
  resolveSource,
  sourceHint,
  sourceLabel,
} from "../src/lib/source.ts";

// ── 构建默认值 ──────────────────────────────────────────────────────

test("构建时没设 VITE_OPS_SOURCE 就是 fixture", () => {
  assert.equal(buildSource(undefined), "fixture");
  assert.equal(buildSource(""), "fixture");
});

test("构建时设了 live 才是 live", () => {
  assert.equal(buildSource("live"), "live");
});

test("构建值拼错或大小写不符一律当 fixture，不猜", () => {
  assert.equal(buildSource("LIVE"), "fixture");
  assert.equal(buildSource("Live"), "fixture");
  assert.equal(buildSource("lvie"), "fixture");
  assert.equal(buildSource("true"), "fixture");
});

// ── URL 覆盖 ────────────────────────────────────────────────────────

test("没有 ?source 时用构建值，且不算覆盖", () => {
  const c = resolveSource("", "fixture");
  assert.deepEqual(c, { source: "fixture", overridden: false });

  const d = resolveSource("?mode=deck&perf=1", "live");
  assert.deepEqual(d, { source: "live", overridden: false });
});

test("?source=live 在 fixture 产物上真的切到 live，并标为覆盖", () => {
  assert.deepEqual(resolveSource("?source=live", "fixture"), {
    source: "live",
    overridden: true,
  });
});

test("?source=fixture 在 live 产物上也能切回来", () => {
  assert.deepEqual(resolveSource("?source=fixture", "live"), {
    source: "fixture",
    overridden: true,
  });
});

test("覆盖值等于构建值时不算覆盖（否则底栏会显示一句没有信息的话）", () => {
  assert.deepEqual(resolveSource("?source=live", "live"), {
    source: "live",
    overridden: false,
  });
  assert.deepEqual(resolveSource("?source=fixture", "fixture"), {
    source: "fixture",
    overridden: false,
  });
});

test("拼错的覆盖值回落到构建值，且不许静默升级成 live", () => {
  // 这是本文件最重要的一条：猜错的代价是一份演示快照去打后端。
  for (const bad of ["lvie", "LIVE", "Live", "1", "true", ""]) {
    assert.deepEqual(
      resolveSource(`?source=${bad}`, "fixture"),
      { source: "fixture", overridden: false },
      `?source=${bad} 不应生效`,
    );
  }
});

test("覆盖值与别的参数并存时仍然生效", () => {
  assert.deepEqual(resolveSource("?mode=deck&source=live&perf=1", "fixture"), {
    source: "live",
    overridden: true,
  });
});

test("?source 出现两次时以第一个为准（URLSearchParams 的既定语义）", () => {
  assert.deepEqual(resolveSource("?source=live&source=fixture", "fixture"), {
    source: "live",
    overridden: true,
  });
});

// ── 底栏那一格 ──────────────────────────────────────────────────────

test("覆盖生效时，底栏文案要说出来", () => {
  const label = sourceLabel({ source: "live", overridden: true });
  assert.match(label, /live/);
  assert.match(label, /\?source/);
});

test("没覆盖时不提 ?source，免得每次截图都带一句无用的话", () => {
  const label = sourceLabel({ source: "fixture", overridden: false });
  assert.doesNotMatch(label, /\?source/);
  assert.match(label, /fixture/);
});

test("悬停说明同时交代「这次是什么」和「这份产物默认是什么」", () => {
  const hint = sourceHint({ source: "live", overridden: true }, "fixture");
  assert.match(hint, /覆盖/);
  assert.match(hint, /默认是 fixture/);
  // 两句实质说明都还在：这是唯一解释"读数为什么是回放"的地方。
  assert.match(hint, /回放/);
});

test("没覆盖时悬停说明给出可用的切换方式", () => {
  const hint = sourceHint({ source: "fixture", overridden: false }, "fixture");
  assert.match(hint, /\?source=/);
  assert.match(hint, /真实响应/);
});

// ── 从地址读取（两个模块共用的入口） ────────────────────────────────

test("没有 window 时（Node / SSR）直接用构建值，不抛错", () => {
  assert.deepEqual(choiceFromLocation("live"), {
    source: "live",
    overridden: false,
  });
});

test("有 window 时读地址栏", () => {
  // node:test 每个文件一个进程，这里改完就删，不会漏到别的文件
  globalThis.window = { location: { search: "?mode=deck&source=live" } };
  try {
    assert.deepEqual(choiceFromLocation("fixture"), {
      source: "live",
      overridden: true,
    });
  } finally {
    delete globalThis.window;
  }
});
