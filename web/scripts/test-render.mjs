#!/usr/bin/env node
/**
 * 渲染冒烟测试：把关键组件在 Node 里真的渲染一遍。
 *
 * ## 为什么不用浏览器截图
 *
 * 截图能证明"看起来对"，但它需要一台能起窗口的机器；CI 与部分开发环境
 * 起不了 GUI 浏览器（沙箱 / 无桌面）。那种环境里截图脚本会静默挂住，
 * 最后谁都不跑它 —— 于是"界面能渲染"这件事没有任何检查。
 *
 * 所以这里退一步：用 `react-dom/server` 渲染成 HTML，断言**该出现的文字
 * 出现了**。它不检查样式，但能抓住三类真实故障：
 *
 * 1. 组件在初始渲染时抛异常（`undefined.stats` 这类）—— 白屏；
 * 2. 某个分支永远不渲染（判空写反）—— 面板空白；
 * 3. 关键措辞消失（"可溯源""检索来源"）—— 结论没显示出来。
 *
 * ## 为什么走 esbuild 打包
 *
 * Node 22 的类型剥离不处理 JSX，也不处理 `.json` 的裸 import。
 * esbuild 已经在 `node_modules` 里（vite 的依赖），用它把入口打成一个
 * 临时 ESM 文件再 import —— 不引入新依赖，也不需要构建整个应用。
 *
 * ## 这里**不**渲染的东西
 *
 * `GeoSection` / `GeoSectionCanvas` 在渲染期就碰 `document` 与 WebGL，
 * 那是浏览器专有的。它们的数字来源（`lib/section.ts`）是纯函数，
 * 由 `test-quality.mjs` 与类型检查覆盖，不在这里重复。
 */

import assert from "node:assert/strict";
import { mkdirSync } from "node:fs";
import { join } from "node:path";
import test, { before } from "node:test";
import { pathToFileURL } from "node:url";

import { build } from "esbuild";

/**
 * 入口源码。
 *
 * `import.meta.env` 在 Node 里不存在，而 `api.ts` / `kb.ts` / `health.ts`
 * 在模块顶层就读它。用 esbuild 的 `define` 把它们逐个替换成 `undefined`，
 * 于是 `?? "fixture"` 这类默认值生效 —— 与浏览器里没配环境变量时一致。
 *
 * 渲染封装必须写在入口里（而不是在测试里 `Component(props)`）：组件里有
 * hooks，直接调用等于在 React 之外调 `useState`，会抛
 * "Invalid hook call" —— 那是一个与本次改动无关的假失败。
 */
const ENTRY = `
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { AlarmCard } from "./src/components/AlarmCard";
import { DialogPanel } from "./src/components/dialog/DialogPanel";
import { KbResultView } from "./src/components/kb/KbResultView";
import { LeftRail } from "./src/components/LeftRail";
import { StatusBar } from "./src/components/StatusBar";
import { clearAcks, setAck } from "./src/lib/ack";
import { askStream } from "./src/lib/chat";
import { ApprovalPanel } from "./src/components/reports/ApprovalPanel";
import { ReportBody } from "./src/components/reports/ReportBody";
import { FIXTURE_QUERY } from "./src/lib/kb";
import { HEALTH_INITIAL } from "./src/lib/health";
import alarms from "./src/fixtures/alarms.json";
import search from "./src/fixtures/search.json";
import thresholds from "./src/fixtures/thresholds.json";
import reportSeeds from "./src/fixtures/reports.json";

const html = (el) => renderToStaticMarkup(el);

export const fixtures = {
  alarms,
  search,
  thresholds,
  health: HEALTH_INITIAL,
  reportSeeds: reportSeeds.drafts,
};
export { setAck, clearAcks, askStream };
export const fixtureQuery = FIXTURE_QUERY;

export const renderKb = (result) =>
  html(createElement(KbResultView, { result, onOpenDoc: () => {} }));

export const renderStatus = (props) => html(createElement(StatusBar, props));

export const renderAlarm = (props) => html(createElement(AlarmCard, props));

export const renderLeftRail = () =>
  html(createElement(LeftRail, { report: alarms, thresholds }));

export const renderDialog = () => html(createElement(DialogPanel));

export const renderApproval = () => html(createElement(ApprovalPanel));

export const renderReportBody = (draft) =>
  html(createElement(ReportBody, { body: draft.body, citations: draft.citations }));
`;

const ENV_KEYS = [
  "VITE_OPS_BASE",
  "VITE_OPS_SOURCE",
  "VITE_KB_BASE",
  "VITE_KB_SOURCE",
  "VITE_HEALTH_URL",
  "VITE_OPS_WS",
];

/** 去标签后比较：断言"说清了什么"，不断言"用了哪个 class"。 */
function text(html) {
  return html
    .replace(/<[^>]+>/g, " ")
    .replace(/&nbsp;/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

let mod;

before(async () => {
  /*
   * 产物必须落在项目里，不能落在系统临时目录：`react` 是外置依赖，
   * 而 Node 是按**导入方所在目录**向上找 `node_modules` 的。
   * 放在 %TEMP% 下会得到 `Cannot find package 'react'` —— 一个看起来
   * 像依赖没装、其实是路径问题的报错。
   */
  const dir = join(process.cwd(), "node_modules", ".cache", "ccus-render");
  mkdirSync(dir, { recursive: true });
  const outfile = join(dir, "bundle.mjs");

  await build({
    stdin: { contents: ENTRY, resolveDir: process.cwd(), loader: "tsx" },
    outfile,
    bundle: true,
    format: "esm",
    platform: "node",
    jsx: "automatic",
    logLevel: "error",
    // react 必须外置：打进包里会得到两份 React，`useSyncExternalStore` 直接抛
    external: ["react", "react-dom", "react-dom/server", "react/jsx-runtime"],
    define: Object.fromEntries(
      ENV_KEYS.map((k) => [`import.meta.env.${k}`, "undefined"]),
    ),
  });

  mod = await import(pathToFileURL(outfile).href);
});

test("KbResultView 渲染出证据强度、检索来源与每条证据的血缘徽标", () => {
  const plain = text(mod.renderKb(mod.fixtures.search));

  assert.match(plain, /证据强度/, "证据强度那一块没渲染出来");
  assert.match(plain, /可溯源/, "可溯源的计数/徽标没出现");
  assert.match(
    plain,
    /无文件血缘/,
    "无血缘这一档没出现 —— 不可溯源的证据会看起来和可溯源的一样",
  );
  assert.match(plain, /公式为主/, "公式为主的提示没出现");
  assert.match(plain, /检索来源/, "fixture/live 没标注");
  assert.match(plain, /fixture（真实响应快照）/, "fixture 来源没写清");

  // search.json 的 12 片全部带 file_path：这一行必须数出 12，而不是显示成 0
  assert.match(plain, /可溯源\s*12\s*片带 file_path/, "可溯源计数不是逐片算出来的 12");
  assert.match(plain, /无文件血缘\s*0/, "无血缘计数应为 0");
});

test("KbResultView 在 unavailable 时不渲染成「没有内容」", () => {
  const plain = text(
    mod.renderKb({ ...mod.fixtures.search, status: "unavailable", reason: "Ollama 未启动" }),
  );
  assert.match(plain, /检索未连接/);
  assert.match(plain, /不等于库里没有内容/);
  assert.doesNotMatch(plain, /证据强度/);
});

test("StatusBar 把数据来源与后端可达性分开显示", () => {
  const plain = text(
    mod.renderStatus({
      counts: { normal: 10, notice: 1, warn: 1, alert: 2, critical: 1, indeterminate: 3 },
      source: "fixture",
      period: ["2026-10-27T06:00:00", "2026-10-27T10:00:00"],
      health: mod.fixtures.health,
    }),
  );
  assert.match(plain, /fixture（真实响应快照）/);
  // 还没探过时既不能写"可达"，也不能写"不可达"
  assert.match(plain, /后端未探测/);
  assert.doesNotMatch(plain, /后端可达/);
});

test("AlarmCard 的处置标记常驻，且自称「本机」而不是「已提交」", () => {
  const alarm = mod.fixtures.alarms.alarms.find(
    (a) => a.severity === "alert" || a.severity === "critical",
  );
  assert.ok(alarm, "fixture 里应当至少有一条 alert/critical 告警");
  const props = { alarm, unit: "bar", warnThreshold: 10, excluded: [] };

  const before = text(mod.renderAlarm(props));
  assert.match(before, /未标记/, "默认状态要写明「未标记」，而不是留空");
  assert.match(before, /确认/);
  assert.match(before, /忽略/);
  // 折叠态也要能读到读数与等级 —— 处置标记不该把这些挤掉
  assert.match(before, new RegExp(alarm.label));

  mod.setAck(alarm.alarm_id, "acknowledged");
  try {
    const after = text(mod.renderAlarm(props));
    assert.match(after, /已确认/, "标记后卡片上要出现状态");
    // 断言的是"说清了存在哪"，不是"用了哪个词"：换成「本机」之外的说法可以，
    // 但不能再出现「仅本页 / 刷新即失效」这种已经不符合事实的描述。
    assert.match(after, /本机/, "标记必须自称存在本机，不能暗示已提交后端");
    assert.doesNotMatch(after, /刷新即失效/, "现在刷新不会丢了，这句话已经不成立");
    // 已处置不等于问题消失：读数与等级仍然显示
    assert.match(after, new RegExp(alarm.label));
    assert.match(after, /警告|告警|严重/);

    const rail = text(mod.renderLeftRail());
    assert.match(rail, /条已标记处置/, "左栏要说清有几条被标记");
    assert.match(rail, /没有写入后端审计/, "左栏要说明它没有落库");
  } finally {
    mod.clearAcks();
  }

  const restored = text(mod.renderLeftRail());
  assert.doesNotMatch(restored, /条已标记处置/, "清空标记后提示必须消失");
});

test("LeftRail 在没有任何标记时不说「已标记」", () => {
  const plain = text(mod.renderLeftRail());
  assert.match(plain, /报警与关键指标/);
  assert.doesNotMatch(plain, /条已标记处置/);
});

test("DialogPanel 空态给出可点的示例问题，而不是一片空白", () => {
  const html = mod.renderDialog();
  const plain = text(html);
  assert.match(plain, /可以这样问/);
  assert.match(plain, /盖层完整性对 CO2 泄漏风险的影响/);
  // 输入框必须一直在（它不该跟着消息列表一起滚走）
  assert.match(html, /<textarea/);
  // 演示模式的说明不能丢：不说清楚的话，用户会以为系统回答了他的问题
  assert.match(plain, /清空/);
});

test("问答是分段的：数值回答先渲染，引用后到", async () => {
  const seen = [];
  const outcome = await mod.askStream(mod.fixtureQuery, (e) => seen.push(e));

  // 顺序即契约。如果有人把两件事重新 await 在一起（回到"一次性返回"），
  // 这里会立刻红 —— 那正是这次改动要防的回归。
  assert.deepEqual(
    seen.map((e) => e.type),
    ["stage", "answer", "stage", "evidence", "done"],
    "事件顺序必须是 取数开始 → 回答 → 检索开始 → 出处 → 结束",
  );

  const answer = seen.find((e) => e.type === "answer");
  assert.ok(answer.answer.text.length > 0, "answer 事件必须带正文，否则界面没东西可渲染");
  assert.ok(
    Array.isArray(answer.answer.facts) && answer.answer.facts.length > 0,
    "数字依据要跟正文一起到，不能等检索",
  );

  const evidence = seen.find((e) => e.type === "evidence");
  assert.ok(evidence.citations.length > 0, "问句对得上快照时应当有引用");
  assert.equal(
    answer.answer.text,
    outcome.answer.text,
    "分段渲染拼出来的必须和一次性返回的结果一致，否则是两条不同的路径",
  );
});

test("问句对不上快照时，evidence 事件明确说「不挂引用」而不是静默为空", async () => {
  const seen = [];
  await mod.askStream("完全无关的问题", (e) => seen.push(e));
  const evidence = seen.find((e) => e.type === "evidence");
  assert.deepEqual(evidence.citations, []);
  assert.match(evidence.retrievalNote, /不是同一次检索/);
});

test("审批面板首屏就把「本机演练 / 不是审计记录」写在脸上", () => {
  // 这句话藏起来的后果很具体：一次演练看起来像一次真实审批。
  // 而审批的全部价值就在于它能被追溯 —— 存在浏览器里的轨迹不能。
  const plain = text(mod.renderApproval());
  assert.match(plain, /本机演练/, "fixture 模式下没写明这是演练");
  assert.match(plain, /不是审计记录/, "没写清本机轨迹不是审计记录");
  assert.match(plain, /不跨班次/, "没写清它不跨班次");
  // 渲染环境里没有 localStorage，必须照实说"刷新就会丢"，
  // 而不是显示"已保存"。
  assert.match(plain, /刷新就会丢/, "存不住时界面必须照实说");
  assert.doesNotMatch(plain, /已保存/, "存不住还说已保存，是这一页最不该有的谎");
});

test("报告正文渲染：区块标记被吃掉，引用标记变成徽标", () => {
  const seed = mod.fixtures.reportSeeds.find((d) => d.problems.length === 0);
  assert.ok(seed, "fixture 里应当有一份干净的草稿");
  const html = mod.renderReportBody(seed);
  const plain = text(html);
  assert.doesNotMatch(plain, /strict/, "区块标记漏进正文了");
  assert.doesNotMatch(plain, /<!--/, "HTML 注释漏进正文了");
  // 引用标记要渲染成徽标，而不是原样吐出 `[alarm:...]` 这一串
  assert.doesNotMatch(plain, /\[alarm:/, "引用标记被当成普通文字渲染了");
  assert.match(plain, /总体结论/, "正文标题没渲染出来");
});
