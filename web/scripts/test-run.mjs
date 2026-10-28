import { test, before } from "node:test";
import assert from "node:assert/strict";
import { mkdirSync } from "node:fs";
import { join } from "node:path";
import { pathToFileURL } from "node:url";

import { build } from "esbuild";

/**
 * 「让 agent 现场生成报告」的判定测试。
 *
 * ## 这些断言值钱在哪
 *
 * 这是界面上唯一会新建记录的按钮，而它的失效方式全是安静的：
 *
 * - 演示模式下也能点，弹一句「已生成」，磁盘上什么都没有；
 * - 两个 agent 里一个失败，界面只显示「已生成」，读者以为报告完整；
 * - 存在阻断性冲突时被压成「3 项偏离」，而其中一条的意思是「不能提交」；
 * - 读数来自回放样本，报告却写得像当期实测。
 *
 * ## 为什么打两个包
 *
 * `api.ts` 在**模块顶层**读 `import.meta.env.VITE_OPS_SOURCE`。Node 里
 * 没有 `import.meta.env`，所以用 esbuild 的 `define` 分别造出 fixture 与
 * live 两种形态，live 那份用 `fetch` 桩验证它真的发 POST、真的把后端 400
 * 的文案透出来 —— 只测纯函数会漏掉"按钮接错了端点"。
 */

const ENTRY = `
import * as run from "./src/lib/run";
import * as api from "./src/lib/api";
export { run, api };
`;

const ENV_KEYS = [
  "VITE_OPS_BASE",
  "VITE_OPS_SOURCE",
  "VITE_KB_BASE",
  "VITE_KB_SOURCE",
  "VITE_HEALTH_URL",
];

async function bundle(name, define) {
  const dir = join(process.cwd(), "node_modules", ".cache", `ccus-run-${name}`);
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

let fixture;
let live;

before(async () => {
  const off = Object.fromEntries(
    ENV_KEYS.map((k) => [`import.meta.env.${k}`, "undefined"]),
  );
  fixture = await bundle("fixture", off);
  live = await bundle("live", {
    ...off,
    "import.meta.env.VITE_OPS_SOURCE": '"live"',
    "import.meta.env.VITE_OPS_BASE": '"/api"',
  });
});

/** 一次正常的返回：两个 agent 都成，三条非阻断冲突。 */
function result(over = {}) {
  return {
    summary: "最高等级：严重｜…",
    can_submit: true,
    runs: [
      { agent: "alarms", reason: "判定基础", ok: true, elapsed_s: 0.001, error: "" },
      { agent: "reports", reason: "生成草稿", ok: true, elapsed_s: 0.002, error: "" },
    ],
    conflicts: [
      { kind: "a", summary: "捕集率 偏离", detail: [], blocking: false, next_step: "" },
      { kind: "b", summary: "管段压差 偏离", detail: [], blocking: false, next_step: "" },
      { kind: "c", summary: "出口压力 偏离", detail: [], blocking: false, next_step: "" },
    ],
    brief: "",
    period: ["2026-10-27T06:00:00", "2026-10-27T10:00:00"],
    draft_id: "daily-20261028T183831-27ea",
    ...over,
  };
}

// ══════════════════════════════════════════════════════════
// 一、演示模式下动手之前就拦住
// ══════════════════════════════════════════════════════════

test("演示模式下生成按钮被挡住，并说清为什么", () => {
  const why = fixture.run.generateBlocked(true);
  assert.ok(why, "演示模式必须挡住");
  assert.match(why, /演示模式/);
  assert.match(why, /没有可复用的快照/, "要说清后果，不能只说「不可用」");
});

test("接上后端后不再拦截", () => {
  assert.equal(live.run.generateBlocked(false), null);
});

test("演示模式下真的调用会抛错，不会编一份草稿", async () => {
  await assert.rejects(
    () => fixture.api.runOpsReport("daily"),
    /不会编一份给你/,
    "编一份出来等于让人以为操作被记下了",
  );
});

// ══════════════════════════════════════════════════════════
// 二、跑成了什么样：成功、失败、阻断必须长得不一样
// ══════════════════════════════════════════════════════════

test("全部成功且无阻断冲突时，说「均非阻断」", () => {
  const line = fixture.run.runHeadline(result());
  assert.match(line, /2 个 agent 执行完毕/);
  assert.match(line, /3 项偏离（均非阻断）/);
});

test("有 agent 失败时，标题不再说「执行完毕」", () => {
  const r = result({
    runs: [
      { agent: "alarms", reason: "", ok: true, elapsed_s: 0.001, error: "" },
      { agent: "reports", reason: "", ok: false, elapsed_s: 0.002, error: "引用检查未过" },
    ],
  });
  const line = fixture.run.runHeadline(r);
  assert.match(line, /1 个失败/, "失败必须进标题，不能只在明细里");
  assert.doesNotMatch(line, /执行完毕/);
});

test("失败的 agent 逐条说明，不合成一句「生成失败」", () => {
  const r = result({
    runs: [{ agent: "reports", reason: "", ok: false, elapsed_s: 0, error: "报告期为 0 个采样点" }],
  });
  const lines = fixture.run.failedAgents(r);
  assert.equal(lines.length, 1);
  assert.match(lines[0], /reports/);
  assert.match(lines[0], /0 个采样点/, "缺的是哪一节、为什么缺，决定这份稿子能不能用");
});

test("失败但没给原因时，照实说没给原因", () => {
  const r = result({
    runs: [{ agent: "reports", reason: "", ok: false, elapsed_s: 0, error: "" }],
  });
  assert.match(fixture.run.failedAgents(r)[0], /未给出原因/);
});

test("阻断性冲突排在前面，并且带「阻断」两个字", () => {
  const r = result({
    conflicts: [
      { kind: "a", summary: "非阻断的那条", detail: [], blocking: false, next_step: "" },
      { kind: "b", summary: "阻断的那条", detail: [], blocking: true, next_step: "" },
    ],
  });
  const lines = fixture.run.conflictLines(r);
  assert.equal(lines.length, 2);
  assert.match(lines[0], /^阻断｜阻断的那条$/);
  assert.match(lines[1], /^提示｜非阻断的那条$/);
});

test("有阻断冲突时标题里数得出来", () => {
  const r = result({
    conflicts: [
      { kind: "a", summary: "x", detail: [], blocking: true, next_step: "" },
      { kind: "b", summary: "y", detail: [], blocking: false, next_step: "" },
    ],
  });
  assert.match(fixture.run.runHeadline(r), /2 项冲突（1 项阻断）/);
});

// ══════════════════════════════════════════════════════════
// 三、报告用的是哪批读数
// ══════════════════════════════════════════════════════════

test("回放说明带两个时间戳，并明说不是当期实测", () => {
  const note = fixture.run.replayNote(["2026-10-27T06:00:00", "2026-10-27T10:00:00"]);
  assert.match(note, /回放样本/);
  assert.match(note, /2026-10-27 06:00/);
  assert.match(note, /2026-10-27 10:00/);
  assert.match(note, /不是当期实测/);
});

test("没有报告期时不给半句话", () => {
  assert.equal(fixture.run.replayNote(null), "");
});

test("没产出草稿时说清是没产出，不说「已生成」", () => {
  const note = fixture.run.runSuccessNote(result({ draft_id: "" }));
  assert.match(note, /没有产出报告草稿/);
  assert.doesNotMatch(note, /已生成/);
});

test("产出草稿时带上 draft_id", () => {
  const note = fixture.run.runSuccessNote(result());
  assert.match(note, /daily-20261028T183831-27ea/);
  assert.match(note, /草稿/);
});

// ══════════════════════════════════════════════════════════
// 四、live 分支：真的发 POST，真的把后端文案透出来
// ══════════════════════════════════════════════════════════

function stubFetch(handler) {
  const calls = [];
  globalThis.fetch = async (url, init) => {
    calls.push({ url: String(url), init });
    return handler(String(url), init);
  };
  return calls;
}

test("live 生成走 POST /api/ops/run，并带上回放读数", async () => {
  const calls = stubFetch(
    async () =>
      new Response(JSON.stringify(result()), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
  );

  const r = await live.api.runOpsReport("daily");

  assert.equal(calls.length, 1);
  assert.equal(calls[0].url, "/api/ops/run");
  assert.equal(calls[0].init.method, "POST");

  const body = JSON.parse(calls[0].init.body);
  assert.equal(body.report_kind, "daily");
  assert.equal(body.want_report, true);
  assert.ok(Array.isArray(body.readings) && body.readings.length > 100, "读数必须真的送进去");
  assert.equal(
    body.want_condition,
    false,
    "工况问答要一个问句；没人提问时跑它只会得到一句「没有读数」的空回答",
  );
  assert.equal(r.draft_id, "daily-20261028T183831-27ea");
});

test("监管材料把许可证号送出去", async () => {
  const calls = stubFetch(
    async () =>
      new Response(JSON.stringify(result()), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
  );
  await live.api.runOpsReport("regulatory", "EPA-VI-0001");
  const body = JSON.parse(calls[0].init.body);
  assert.equal(body.report_kind, "regulatory");
  assert.equal(body.permit_id, "EPA-VI-0001");
});

test("后端 400 的文案原样显示，不换成「生成失败」", async () => {
  stubFetch(
    async () =>
      new Response(
        JSON.stringify({ detail: "监管材料必须提供 permit_id：没有许可证号的监管材料无法归档" }),
        { status: 400, headers: { "Content-Type": "application/json" } },
      ),
  );
  await assert.rejects(
    () => live.api.runOpsReport("regulatory", ""),
    /必须提供 permit_id/,
    "换成「生成失败」用户只能重试，而他需要的是补信息",
  );
});
