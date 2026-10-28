import { test, before } from "node:test";
import assert from "node:assert/strict";
import { mkdirSync } from "node:fs";
import { join } from "node:path";
import { pathToFileURL } from "node:url";

import { build } from "esbuild";

/**
 * 缺口反馈（只读知识库的写入入口）的测试。
 *
 * ## 这些断言值钱在哪
 *
 * 这个功能的所有失效方式都不会报错：
 *
 * - 演示模式下按钮可点，用户打完一百字才被告知"其实没发出去"；
 * - 队列读不出来时显示「0 条」，读者据此认为没人反馈过 —— 一个关于人的
 *   结论，实际是关于文件系统的；
 * - 检索服务挂了（`unavailable`）也允许登记"没答到"，一次故障被记成语料缺口；
 * - 前端自己写一句"稍后自动处理"，而入库要占满一张显卡、建图几十分钟。
 *
 * 每一条都对应下面一个用例。
 *
 * ## 为什么要打两个包
 *
 * `kb.ts` 在**模块顶层**读 `import.meta.env.VITE_KB_SOURCE` 决定走 fixture
 * 还是 live。Node 里 `import.meta.env` 不存在，所以用 esbuild 的 `define`
 * 分别造出两种形态：一个"没有后端"（默认，与浏览器里没配变量时一致），
 * 一个 `live`（并用 `fetch` 桩验证它真的发 POST、真的把后端文案透出来）。
 *
 * 只测 fixture 那一半是不够的：真实部署走的是 live，而 live 分支在
 * 演示环境里从来没被跑过。
 */

const ENTRY = `
import * as gap from "./src/lib/gap";
import * as kb from "./src/lib/kb";
export { gap, kb };
`;

const ENV_KEYS = [
  "VITE_OPS_BASE",
  "VITE_OPS_SOURCE",
  "VITE_KB_BASE",
  "VITE_KB_SOURCE",
  "VITE_HEALTH_URL",
];

async function bundle(name, define) {
  // 产物落在项目里：Node 按导入方所在目录向上找 node_modules
  const dir = join(process.cwd(), "node_modules", ".cache", `ccus-gap-${name}`);
  mkdirSync(dir, { recursive: true });
  const outfile = join(dir, "bundle.mjs");
  await build({
    stdin: { contents: ENTRY, resolveDir: process.cwd(), loader: "tsx" },
    outfile,
    bundle: true,
    format: "esm",
    platform: "node",
    logLevel: "error",
    define,
  });
  return import(pathToFileURL(outfile).href);
}

/** 与 `api_kb.GAP_KINDS` 同一份清单。两边各有一条断言锁着它。 */
const KINDS = [
  "missing_document",
  "not_in_graph",
  "unanswered",
  "wrong_evidence",
  "other",
];

let fixture;
let live;

before(async () => {
  const off = Object.fromEntries(
    ENV_KEYS.map((k) => [`import.meta.env.${k}`, "undefined"]),
  );
  fixture = await bundle("fixture", off);
  live = await bundle("live", {
    ...off,
    "import.meta.env.VITE_KB_SOURCE": '"live"',
    "import.meta.env.VITE_KB_BASE": '"/api"',
  });
});

// ══════════════════════════════════════════════════════════
// 一、种类清单与后端对齐
// ══════════════════════════════════════════════════════════

test("缺口种类的界面措辞覆盖后端枚举的每一个值", () => {
  assert.deepEqual(
    Object.keys(fixture.gap.GAP_KIND_LABEL).sort(),
    [...KINDS].sort(),
    "界面措辞与 api_kb.GAP_KINDS 分家了 —— 加一种而不加措辞，" +
      "队列里会出现一个英文枚举名，读者不知道那是什么意思",
  );
  for (const k of KINDS) {
    const label = fixture.gap.GAP_KIND_LABEL[k];
    // 只断言"有措辞、且没露出英文枚举名"。**不断言措辞本身** ——
    // 断言用了哪个词会让改一句文案就红，而红的原因不是逻辑坏了。
    assert.ok(label && label.trim().length >= 2, `${k} 的措辞为空`);
    assert.doesNotMatch(label, /[a-z_]{4,}/, `${k} 的措辞里露出了英文枚举名`);
  }
});

// ══════════════════════════════════════════════════════════
// 二、演示模式：动手之前就拦住
// ══════════════════════════════════════════════════════════

test("演示模式下表单在用户动手之前就是禁用的，并且说清为什么", () => {
  const blocked = fixture.gap.gapFormBlocked();
  assert.ok(blocked, "fixture 模式必须拦住 —— 存在浏览器里等于没写");
  assert.match(blocked, /演示模式/);
  assert.match(blocked, /等于没写/, "要说清后果，不能只说「不可用」");
  assert.match(blocked, /VITE_KB_SOURCE=live/, "要给出接上后端的办法");
});

test("接上后端后不再拦截", () => {
  assert.equal(live.gap.gapFormBlocked(), null);
});

test("演示模式下真的提交会抛错，而不是安静成功", async () => {
  await assert.rejects(
    () => fixture.kb.submitKbGap({ kind: "other", note: "x" }),
    /演示模式不落盘/,
  );
});

test("演示模式下队列显示「读不到」，不是「0 条」", async () => {
  const data = await fixture.kb.fetchKbGaps();
  assert.equal(data.available, false, "读不到必须是 available:false");
  const summary = fixture.gap.queueSummary(data);
  assert.match(summary.headline, /读不到/);
  assert.match(summary.detail, /不等于没人反馈过/);
  assert.doesNotMatch(summary.headline, /0 条/);
});

// ══════════════════════════════════════════════════════════
// 三、队列摘要：三种情形长得不一样
// ══════════════════════════════════════════════════════════

test("空队列与读不到是两句话", () => {
  const empty = fixture.gap.queueSummary({
    available: true,
    reason: "",
    total: 0,
    unreadable_lines: 0,
    by_kind: {},
    records: [],
  });
  assert.match(empty.headline, /还没有人登记/);
  assert.equal(empty.detail, "");
});

test("有积压时按种类计数，坏行要数出来", () => {
  const s = fixture.gap.queueSummary({
    available: true,
    reason: "",
    total: 3,
    unreadable_lines: 2,
    by_kind: { not_in_graph: 2, unanswered: 1 },
    records: [],
  });
  assert.match(s.headline, /已登记 3 条/);
  assert.match(s.headline, /2 行读不出来/, "坏行不数出来，全坏时看起来像空队列");
  assert.match(s.detail, /这篇没进图谱 2/);
  assert.match(s.detail, /检索没答到 1/);
});

test("队列列表的一行带对象与时间", () => {
  const line = fixture.gap.recordLine({
    kind: "not_in_graph",
    target: "ccus4.pdf",
    recorded_at: "2026-10-28T09:43:50+00:00",
  });
  assert.equal(line, "这篇没进图谱：ccus4.pdf · 2026-10-28 09:43");
});

test("未知种类不会渲染成 undefined", () => {
  const line = fixture.gap.recordLine({
    kind: "brand_new_kind",
    recorded_at: "2026-10-28T09:43:50+00:00",
  });
  assert.doesNotMatch(line, /undefined/);
  assert.match(line, /brand_new_kind/);
});

// ══════════════════════════════════════════════════════════
// 四、哪些检索状态可以登记
// ══════════════════════════════════════════════════════════

test("unavailable 不给登记入口", () => {
  assert.equal(
    fixture.gap.canReportFromStatus("unavailable"),
    false,
    "检索根本没跑起来时登记「没答到」，会把一次服务故障记成语料缺口 —— " +
      "队列里那个数字随后会被当成「该补文献了」",
  );
});

test("ok 与 empty 可以登记", () => {
  assert.equal(fixture.gap.canReportFromStatus("empty"), true);
  assert.equal(fixture.gap.canReportFromStatus("ok"), true);
});

// ══════════════════════════════════════════════════════════
// 五、live 分支：真的发 POST，真的把后端文案透出来
// ══════════════════════════════════════════════════════════

function stubFetch(handler) {
  const calls = [];
  globalThis.fetch = async (url, init) => {
    calls.push({ url: String(url), init });
    return handler(String(url), init);
  };
  return calls;
}

test("live 提交走 POST /api/kb/gaps，并把记录回显", async () => {
  const calls = stubFetch(async () =>
    new Response(
      JSON.stringify({
        id: "gap-1",
        recorded_at: "2026-10-28T09:43:50+00:00",
        kind: "not_in_graph",
        target: "ccus4.pdf",
        next_step: "进的是人工队列，不会自动重建索引。",
      }),
      { status: 200, headers: { "Content-Type": "application/json" } },
    ),
  );

  const rec = await live.kb.submitKbGap({
    kind: "not_in_graph",
    target: "ccus4.pdf",
  });

  assert.equal(calls.length, 1);
  assert.equal(calls[0].url, "/api/kb/gaps");
  assert.equal(calls[0].init.method, "POST");
  assert.equal(rec.id, "gap-1");
  assert.match(rec.next_step, /不会自动重建索引/);
});

test("后端的 400 文案原样显示，不换成「提交失败」", async () => {
  stubFetch(
    async () =>
      new Response(
        JSON.stringify({ detail: "登记「某篇没进图谱」必须说是哪一篇" }),
        { status: 400, headers: { "Content-Type": "application/json" } },
      ),
  );
  await assert.rejects(
    () => live.kb.submitKbGap({ kind: "not_in_graph", note: "忘了说是哪篇" }),
    /必须说是哪一篇/,
    "换成「提交失败」用户只能重试，而他需要的是改",
  );
});

test("响应体不是 JSON 时退回状态码，不吞掉失败", async () => {
  stubFetch(async () => new Response("<html>502</html>", { status: 502 }));
  await assert.rejects(
    () => live.kb.submitKbGap({ kind: "other", note: "x" }),
    /502/,
  );
});

test("live 读队列带 limit 参数", async () => {
  const calls = stubFetch(async () =>
    new Response(
      JSON.stringify({
        available: true,
        reason: "",
        total: 1,
        unreadable_lines: 0,
        by_kind: { other: 1 },
        records: [],
      }),
      { status: 200, headers: { "Content-Type": "application/json" } },
    ),
  );
  const data = await live.kb.fetchKbGaps(5);
  assert.equal(calls[0].url, "/api/kb/gaps?limit=5");
  assert.equal(data.total, 1);
});
