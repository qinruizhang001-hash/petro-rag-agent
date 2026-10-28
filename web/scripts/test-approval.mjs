import { test, before } from "node:test";
import assert from "node:assert/strict";
import { mkdirSync } from "node:fs";
import { join } from "node:path";
import { pathToFileURL } from "node:url";

import { build } from "esbuild";

/**
 * 审批状态机的测试。
 *
 * ## 为什么这些断言值钱
 *
 * 审批功能的失效方式**全都不会报错**：
 *
 * - 草稿被直接批准 → 审批环节变成形式，而界面看起来完全正常；
 * - 驳回没写意见就过了 → 下一稿原样再来，没人知道要改什么；
 * - 前端抄了一张和后端不一样的状态表 → 按钮点不动，或者点下去 409；
 * - 本机存的审批表坏了但"抢救"回一半 → 某几条显示「已批准」而实际没人批过。
 *
 * 每一条都对应下面一个用例。
 *
 * ## 为什么要 esbuild 打一次包
 *
 * `lib/approval.ts` 静态 import 了 `fixtures/approval.json`。Node 的 ESM
 * 要求 JSON 带 `with { type: "json" }`，而 Vite 不需要 —— 为了测试去改
 * 产品代码的 import 写法是本末倒置。esbuild 会照 Vite 的语义解析它，
 * 所以测试跑的和浏览器里跑的是同一份代码。
 */

const ENTRY = `
import * as approval from "./src/lib/approval";
import * as reportBody from "./src/lib/reportBody";
import * as draftStore from "./src/lib/draftStore";
import drafts from "./src/fixtures/reports.json";
import contractJson from "./src/fixtures/approval.json";

export { approval, reportBody, draftStore };
export const seedDrafts = drafts.drafts;
export const contract = contractJson;
`;

let mod;

before(async () => {
  // 产物落在项目里：Node 按导入方所在目录向上找 node_modules
  const dir = join(process.cwd(), "node_modules", ".cache", "ccus-approval");
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
  mod = await import(pathToFileURL(outfile).href);
});

// ─── 契约 ─────────────────────────────────────────────────
test("契约里出现的每个动作都有中文名与提示", () => {
  const { approval, contract } = mod;
  const actions = new Set(contract.edges.map((e) => e.action));
  assert.ok(actions.size >= 4, `契约里只有 ${actions.size} 个动作，应当至少 4 个`);
  for (const a of actions) {
    assert.ok(
      approval.ACTION_LABEL[a],
      `动作 ${a} 没有中文名 —— 界面上会渲染出一个空按钮`,
    );
    assert.ok(approval.ACTION_HINT[a], `动作 ${a} 没有提示文案`);
  }
});

test("四个状态都有标签，且没有多余状态", () => {
  const { approval, contract } = mod;
  assert.deepEqual(contract.statuses.slice().sort(), [
    "approved",
    "draft",
    "pending_review",
    "rejected",
  ]);
  for (const s of contract.statuses) {
    assert.ok(approval.STATUS_LABEL[s], `状态 ${s} 没有中文标签`);
  }
});

test("每个状态的可执行动作与契约里的边一一对应", () => {
  const { approval, contract } = mod;
  for (const status of contract.statuses) {
    const fromEdges = contract.edges
      .filter((e) => e.from === status)
      .map((e) => e.action);
    assert.deepEqual(
      approval.actionsFrom(status),
      fromEdges,
      `${status} 的可执行动作与契约不一致`,
    );
  }
});

test("草稿只能提交，待审批能批准与驳回，已批准是终态", () => {
  const { approval } = mod;
  assert.deepEqual(approval.actionsFrom("draft"), ["submit"]);
  assert.deepEqual(approval.actionsFrom("pending_review"), ["approve", "reject"]);
  assert.deepEqual(approval.actionsFrom("rejected"), ["reopen"]);
  assert.deepEqual(approval.actionsFrom("approved"), []);
  assert.equal(approval.isTerminal("approved"), true);
  assert.equal(approval.isTerminal("draft"), false);
});

test("未知状态一个动作都不给", () => {
  const { approval } = mod;
  // 库里读到一条状态字段被改坏的旧记录时，宁可一个按钮都不给。
  // 猜错的后果是批准一份不知道处于什么状态的稿子。
  assert.deepEqual(approval.actionsFrom("已批准?"), []);
  assert.equal(approval.targetOf("已批准?", "approve"), null);
});

// ─── 守卫 ─────────────────────────────────────────────────
test("草稿不能直接批准，且理由说清了当前状态", () => {
  const { approval } = mod;
  const reason = approval.blockedReason({ status: "draft", problems: [] }, "approve", {
    reviewer: "张工",
  });
  assert.ok(reason, "草稿直接批准必须被挡 —— 那等于没有审批环节");
  assert.match(reason, /草稿/, "理由里要写明当前状态，否则读者不知道差什么");
  assert.match(reason, /批准/);
});

test("批准缺署名时，理由用的是后端原话", () => {
  const { approval, contract } = mod;
  const backend = contract.guards.find((g) => g.action === "approve" && g.field === "reviewer");
  assert.ok(backend, "契约里没有「批准必须署名」这条守卫");
  const reason = approval.blockedReason(
    { status: "pending_review", problems: [] },
    "approve",
    { reviewer: "   " },
  );
  assert.equal(reason, backend.message, "理由应当逐字等于后端给出的那句话");
  assert.match(reason, /追溯/, "这句话要说明为什么必须署名，而不只是「必填」");
});

test("驳回缺意见时被挡，且理由说清了后果", () => {
  const { approval } = mod;
  const reason = approval.blockedReason(
    { status: "pending_review", problems: [] },
    "reject",
    { reviewer: "张工", note: "" },
  );
  assert.ok(reason);
  assert.match(reason, /驳回必须写意见/);
  assert.match(reason, /原样再来/, "要说明后果，否则「必填」看起来只是形式");
});

test("引用检查未过时拒绝提交，且理由只取结论那一行", () => {
  const { approval } = mod;
  const reason = approval.blockedReason(
    { status: "draft", problems: ["第 195 行数字 '999.9' 没有引用标记"] },
    "submit",
    {},
  );
  assert.ok(reason);
  assert.match(reason, /引用检查未通过/);
  // 逐条明细由界面单独列出（带行号），不该塞进按钮旁边的理由里 ——
  // 那一段有换行，塞进 title 会变成一个不可读的长条。
  assert.doesNotMatch(reason, /\n/, "理由里不该带换行");
  assert.doesNotMatch(reason, /[：:]$/, "理由不该以悬空的冒号结尾 —— 明细不跟在后面");
});

test("守卫的输入需求是从契约反推的，不是另写一张表", () => {
  const { approval } = mod;
  assert.deepEqual(approval.requiredFields("approve"), ["reviewer"]);
  assert.deepEqual(approval.requiredFields("reject"), ["reviewer", "note"]);
  assert.deepEqual(approval.requiredFields("submit"), []);
  assert.equal(approval.requiresCleanCitations("submit"), true);
  assert.equal(approval.requiresCleanCitations("approve"), false);
});

// ─── 迁移 ─────────────────────────────────────────────────
function seed() {
  const d = mod.seedDrafts.find((x) => x.status === "draft");
  assert.ok(d, "fixture 里应当有一份草稿状态的种子");
  return d;
}

test("提交：状态与轨迹同步变化，可执行动作跟着换", () => {
  const { approval } = mod;
  const before = seed();
  const r = approval.applyAction(before, "submit", { actor: "早班" }, new Date(2026, 9, 28, 17, 3));
  assert.equal(r.ok, true);
  assert.equal(r.draft.status, "pending_review");
  assert.equal(r.draft.status_label, "待审批");
  assert.deepEqual(r.draft.allowed, ["approve", "reject"]);
  assert.equal(r.draft.trail.length, before.trail.length + 1);
  const last = r.draft.trail.at(-1);
  assert.equal(last.from, "draft");
  assert.equal(last.to, "pending_review");
  assert.equal(last.actor, "早班");
  assert.equal(last.at, "2026-10-28 17:03", "轨迹时间要用传进来的 now，不要读时钟");
});

test("applyAction 是纯的：失败时原对象一点不动", () => {
  const { approval } = mod;
  const before = seed();
  const snapshot = JSON.stringify(before);
  const r = approval.applyAction(before, "approve", { reviewer: "张工" }, new Date());
  assert.equal(r.ok, false);
  assert.equal(JSON.stringify(before), snapshot, "被拒的动作不该改动入参");
});

test("批准记的是署名，不是「起草方」", () => {
  const { approval } = mod;
  let d = seed();
  d = approval.applyAction(d, "submit", { actor: "早班" }, new Date()).draft;
  const r = approval.applyAction(d, "approve", { reviewer: "张工", note: "已核对" }, new Date());
  assert.equal(r.ok, true);
  assert.equal(r.draft.status, "approved");
  assert.equal(r.draft.trail.at(-1).actor, "张工");
  assert.equal(r.draft.trail.at(-1).note, "已核对");
  assert.deepEqual(r.draft.allowed, []);
});

test("驳回 → 重新打开：轨迹三步全留着", () => {
  const { approval } = mod;
  let d = seed();
  d = approval.applyAction(d, "submit", { actor: "早班" }, new Date()).draft;
  d = approval.applyAction(
    d,
    "reject",
    { reviewer: "张工", note: "第三节归因缺少排除理由" },
    new Date(),
  ).draft;
  assert.equal(d.status, "rejected");
  d = approval.applyAction(d, "reopen", { actor: "早班" }, new Date()).draft;
  assert.equal(d.status, "draft");
  assert.equal(d.trail.length, 3);
  assert.match(
    d.trail[1].note,
    /第三节/,
    "驳回意见必须留在轨迹里 —— 它是「为什么被打回」的唯一记录",
  );
});

test("引用检查不过的种子，连提交都过不去", () => {
  const { approval } = mod;
  const bad = mod.seedDrafts.find((d) => d.problems.length > 0);
  assert.ok(bad, "fixture 里应当有一份刻意弄坏的样本，用来演示这条守卫");
  const r = approval.applyAction(bad, "submit", { actor: "早班" }, new Date());
  assert.equal(r.ok, false);
  assert.match(r.message, /引用检查未通过/);
});

// ─── 本机存储 ─────────────────────────────────────────────
test("本机草稿能原样存回来", () => {
  const { draftStore, approval } = mod;
  const d = approval.applyAction(seed(), "submit", { actor: "早班" }, new Date()).draft;
  const raw = draftStore.encodeDrafts([d], "abcd1234");
  const back = draftStore.decodeDrafts(raw, "abcd1234");
  assert.equal(back.ok, true);
  assert.equal(back.drafts[0].status, "pending_review");
  assert.equal(back.drafts[0].trail.length, 1);
});

test("没存过 → empty，坏 JSON → corrupt，版本号不对 → corrupt", () => {
  const { draftStore } = mod;
  assert.deepEqual(draftStore.decodeDrafts(null, "x"), { ok: false, reason: "empty" });
  assert.deepEqual(draftStore.decodeDrafts("{", "x"), { ok: false, reason: "corrupt" });
  assert.deepEqual(
    draftStore.decodeDrafts(JSON.stringify({ v: 2, seed: "x", drafts: [] }), "x"),
    { ok: false, reason: "corrupt" },
  );
});

test("坏数据整体丢弃，不逐条抢救", () => {
  const { draftStore, approval } = mod;
  const good = approval.applyAction(seed(), "submit", { actor: "早班" }, new Date()).draft;
  // 一份好的 + 一份缺 trail 的：整体丢弃。只恢复一半的审批表会让某几条
  // 显示成「已批准」而实际没人批过 —— 那比空表危险得多。
  const raw = JSON.stringify({
    v: 1,
    seed: "x",
    drafts: [good, { draft_id: "bad", status: "approved" }],
  });
  assert.deepEqual(draftStore.decodeDrafts(raw, "x"), { ok: false, reason: "corrupt" });
});

test("种子换了就整体丢弃（stale）", () => {
  const { draftStore, approval } = mod;
  const d = approval.applyAction(seed(), "submit", { actor: "早班" }, new Date()).draft;
  const raw = draftStore.encodeDrafts([d], "old-seed");
  assert.deepEqual(draftStore.decodeDrafts(raw, "new-seed"), {
    ok: false,
    reason: "stale",
  });
});

test("指纹对结构敏感，对正文不敏感", () => {
  const { draftStore } = mod;
  const base = mod.seedDrafts;
  const f = draftStore.seedFingerprint(base);
  const bodyChanged = base.map((d) => ({ ...d, body: d.body + "追加一行" }));
  assert.equal(
    draftStore.seedFingerprint(bodyChanged),
    f,
    "正文改了不该丢本机演练数据 —— 重新生成 fixture 是常事",
  );
  const statusChanged = base.map((d, i) => (i === 0 ? { ...d, status: "approved" } : d));
  assert.notEqual(
    draftStore.seedFingerprint(statusChanged),
    f,
    "状态变了说明种子换了，必须丢弃本机数据",
  );
});

// ─── 正文渲染 ─────────────────────────────────────────────
test("区块标记被吃掉，不会渲染成一行看不懂的注释", () => {
  const { reportBody } = mod;
  const blocks = reportBody.parseBody(
    "## 一、总体结论\n\n<!-- strict -->\n- 需人工介入的告警 4 条\n<!-- /strict -->\n",
  );
  const all = JSON.stringify(blocks);
  assert.doesNotMatch(all, /strict/, "strict 标记漏进正文了");
  assert.equal(blocks.length, 2);
  assert.equal(blocks[0].kind, "h2");
  assert.equal(blocks[1].kind, "li");
});

test("标题、列表、原文引用各自分类，二级列表靠缩进区分", () => {
  const { reportBody } = mod;
  const kinds = reportBody
    .parseBody(
      [
        "# 标题",
        "## 一节",
        "### 捕集",
        "- 一级",
        "  - 二级",
        "> 原文",
        "普通段落",
      ].join("\n"),
    )
    .map((b) => b.kind);
  assert.deepEqual(kinds, ["h1", "h2", "h3", "li", "li2", "quote", "p"]);
});

test("引用标记单独成段，不会被粗体包进去", () => {
  const { reportBody } = mod;
  const segs = reportBody.segment("**捕集率**：86.8 % [alarm:A-1-capture_rate]");
  assert.deepEqual(
    segs.map((s) => s.kind),
    ["bold", "text", "cite"],
  );
  const cite = segs.find((s) => s.kind === "cite");
  assert.equal(cite.refKind, "alarm");
  assert.equal(cite.ref, "A-1-capture_rate");
});

test("citedTokens 去重且保序", () => {
  const { reportBody } = mod;
  const tokens = reportBody.citedTokens(
    "[alarm:A] [alarm:B] [alarm:A] [summary:统计]",
  );
  assert.deepEqual(tokens, ["[alarm:A]", "[alarm:B]", "[summary:统计]"]);
});

test("报告期同年时省掉重复年份", () => {
  const { approval } = mod;
  const same = approval.formatPeriod(["2026-10-27T06:00:00", "2026-10-27T10:00:00"]);
  assert.equal(same, "2026-10-27 06:00 → 10-27 10:00");
  const cross = approval.formatPeriod(["2026-10-01T00:00:00", "2027-01-01T00:00:00"]);
  assert.match(cross, /2026-10-01 00:00 → 2027-01-01 00:00/);
});
