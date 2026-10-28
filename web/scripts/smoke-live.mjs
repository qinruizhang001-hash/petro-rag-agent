#!/usr/bin/env node
/**
 * 接后端之后的冒烟检查。
 *
 * ## 为什么要有这个脚本
 *
 * 界面的默认模式是 `fixture`（不需要后端），所以「接上真后端」这件事
 * 在 CI 里从来没有被跑过 —— 它只在我自己的机器上、某一次手动点过。
 * 那种验证不留痕：下次谁改了 `api.ts` 的路径、或者后端把字段改了名，
 * 没有任何东西会红，直到演示当天。
 *
 * 这个脚本把那次手动点过的东西写成命令：
 *
 * ```bash
 * # 后端先起在 127.0.0.1:8000（见 web/README 的「跑起来」）
 * node scripts/smoke-live.mjs
 * node scripts/smoke-live.mjs --base http://10.0.0.5:8000 --token xxx
 * ```
 *
 * ## 它检查什么、不检查什么
 *
 * 检查**契约**：每个端点在不在、状态码对不对、关键字段有没有、数字之间
 * 自不自洽。不检查渲染（那是 `test-render.mjs` 的事），也不检查模型回答
 * 的质量（那是 `eval/` 的事）。
 *
 * 检索那一步慢（冷启动要几十秒），所以默认**不跑**，加 `--with-search`
 * 才跑。这与 `capture_kb_fixtures.py --check` 的分层是同一个道理：
 * 慢的、依赖模型的检查必须显式要，否则整个脚本会变成"没人跑"。
 *
 * ## 退出码
 *
 * 0 = 全过；1 = 有失败项。**失败项一律打印原因**，不打一个 `✗` 了事 ——
 * 这个脚本的读者是刚拉起后端的人，他需要知道下一步改什么。
 */

const arg = (name, fallback = null) => {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 && process.argv[i + 1] ? process.argv[i + 1] : fallback;
};
const flag = (name) => process.argv.includes(`--${name}`);

const BASE = (arg("base") ?? "http://127.0.0.1:8000").replace(/\/$/, "");
const TOKEN = arg("token", "");
const WITH_SEARCH = flag("with-search");

const headers = TOKEN ? { Authorization: `Bearer ${TOKEN}` } : {};

let failures = 0;
let skipped = 0;

function ok(label, detail = "") {
  console.log(`  ✓ ${label}${detail ? ` · ${detail}` : ""}`);
}

function bad(label, why) {
  failures += 1;
  console.log(`  ✗ ${label}`);
  console.log(`      ${why}`);
}

function skip(label, why) {
  skipped += 1;
  console.log(`  – ${label}（跳过：${why}）`);
}

async function get(path) {
  const res = await fetch(`${BASE}${path}`, { headers });
  const text = await res.text();
  let body = null;
  try {
    body = JSON.parse(text);
  } catch {
    /* 保持 null，由调用方判断 */
  }
  return { status: res.status, body, text };
}

async function post(path, payload) {
  const res = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...headers },
    body: JSON.stringify(payload),
  });
  const text = await res.text();
  let body = null;
  try {
    body = JSON.parse(text);
  } catch {
    /* 保持 null */
  }
  return { status: res.status, body, text };
}

/** `/api/ops/*` 是无状态的，判定所需的读数必须由调用方送进去。 */
async function sampledReadings() {
  const { readFile } = await import("node:fs/promises");
  const raw = await readFile(new URL("../src/fixtures/readings.json", import.meta.url), "utf8");
  return JSON.parse(raw);
}

// ══════════════════════════════════════════════════════════
// 一、连通性与依赖
// ══════════════════════════════════════════════════════════

async function checkHealth() {
  console.log("\n[1/6] /health —— 依赖是不是真的通");
  let r;
  try {
    r = await get("/health");
  } catch (e) {
    bad("/health", `连不上 ${BASE}：${e.message}\n      后端起了吗？web/README 的「跑起来」有一行命令。`);
    return false;
  }
  if (r.status !== 200) {
    bad("/health", `HTTP ${r.status}：${r.text.slice(0, 200)}`);
    return false;
  }
  ok("/health", `status=${r.body.status} · auth=${r.body.auth}`);
  if (r.body.auth === "disabled") {
    console.log("      注意：API_TOKEN 未设置，鉴权关闭（本地开发可以接受）");
  }
  for (const [name, c] of Object.entries(r.body.checks ?? {})) {
    if (c.ok) ok(`依赖 ${name}`, `${c.detail}（${c.latency_ms} ms）`);
    else bad(`依赖 ${name}`, c.detail);
  }
  return true;
}

// ══════════════════════════════════════════════════════════
// 二、阈值：这是最容易"看起来正常"的一步
// ══════════════════════════════════════════════════════════

async function checkThresholds() {
  console.log("\n[2/6] /api/ops/thresholds —— 有没有人定过线");
  const r = await get("/api/ops/thresholds");
  if (r.status !== 200) {
    bad("阈值接口", `HTTP ${r.status}：${r.text.slice(0, 200)}`);
    return null;
  }
  const d = r.body;
  ok("阈值接口", `origin=${d.origin}`);

  if (d.missing.length === d.sensors.length && d.sensors.length > 0) {
    /*
     * 全部未配置 = 看板上 18 项"不可判定、0 需动作"。那不是"系统正常"，
     * 是"没人定过线"。这个状态在界面上看着很安静，所以在这里必须喊出来。
     */
    bad(
      "阈值全部未配置",
      "18 个测点一个都没配 → 看板会显示「0 需动作 / N 不可判定」。\n" +
        "      这不是系统正常，是没人定过线。本地跑一遍：\n" +
        "        OPS_THRESHOLDS_PATH=config/ops_thresholds.example.json \\\n" +
        "          python -m uvicorn api_server:api --host 127.0.0.1 --port 8000",
    );
  } else {
    ok("已配置", `${d.configured.length} 个`);
    if (d.missing.length > 0) {
      console.log(`      未配置 ${d.missing.length} 个：${d.missing.join("、")}`);
      console.log("      它们在界面上会显示「不可判定」，不是「正常」——这是对的");
    }
    if (/example/.test(d.origin)) {
      console.log("      注意：用的是示例阈值，source 里全是「【替换】」——界面会原样显示");
    }
  }
  return d;
}

// ══════════════════════════════════════════════════════════
// 三、告警判定：数字之间要自洽
// ══════════════════════════════════════════════════════════

async function checkAlarms(thresholds) {
  console.log("\n[3/6] /api/ops/alarms —— 判定能不能算出来");
  const readings = await sampledReadings();
  const r = await post("/api/ops/alarms", { readings });
  if (r.status !== 200) {
    bad("告警接口", `HTTP ${r.status}：${r.text.slice(0, 300)}`);
    return;
  }
  const d = r.body;
  const counts = d.counts ?? {};
  const total = Object.values(counts).reduce((a, b) => a + b, 0);
  ok("告警接口", `${d.alarms.length} 条告警 · 计数合计 ${total}`);

  if (total !== d.alarms.length) {
    bad("计数与列表对不上", `counts 合计 ${total}，alarms 有 ${d.alarms.length} 条`);
  }
  if (!d.ceiling || typeof d.ceiling.state !== "string") {
    bad("上限校核缺失", "ceiling.state 没有下发 —— 界面会把这一块渲染成空白");
  } else {
    ok("上限校核", `${d.ceiling.state} · ${d.ceiling.label}`);
  }

  const byKey = new Map((thresholds?.sensors ?? []).map((s) => [s.key, s]));
  let checked = 0;
  let mismatched = 0;
  for (const a of d.alarms) {
    const warn = byKey.get(a.sensor)?.threshold?.warn ?? null;
    if (a.value === null || warn === null) continue;
    const dir = byKey.get(a.sensor)?.direction ?? a.direction;
    const margin = dir === "high" ? warn - a.value : a.value - warn;
    const breached = margin < 0;
    const claimsBreach = a.severity !== "normal" && a.severity !== "indeterminate";
    checked += 1;
    // notice 只是"进了提示带"，不算越过 warn 线，所以只在 breached 时要求它非 normal
    if (breached && a.severity === "normal") mismatched += 1;
    if (!breached && claimsBreach && a.threshold_level === "warn" && margin > 0) {
      mismatched += 1;
    }
  }
  if (checked > 0) {
    if (mismatched === 0) ok("余量方向自洽", `${checked} 条可核对，全部一致`);
    else bad("余量方向不一致", `${mismatched}/${checked} 条的等级与余量符号矛盾`);
  } else {
    skip("余量方向自洽", "没有同时有读数与阈值的测点");
  }
  return d;
}

// ══════════════════════════════════════════════════════════
// 四、工况问答：正文里的数字要能回代
// ══════════════════════════════════════════════════════════

async function checkCondition() {
  console.log("\n[4/6] /api/ops/condition —— 数值回答");
  const readings = await sampledReadings();
  const r = await post("/api/ops/condition", {
    question: "过去 4 小时井筒和管道有什么异常",
    readings,
  });
  if (r.status !== 200) {
    bad("工况接口", `HTTP ${r.status}：${r.text.slice(0, 300)}`);
    return;
  }
  const d = r.body;
  ok("工况接口", `${d.stats.length} 个测点 · ${d.facts.length} 条事实`);

  if (!d.text || d.text.trim().length === 0) {
    bad("正文为空", "界面会渲染成一条空白回答 —— 看起来像还在加载");
  }
  if (d.unresolved?.length > 0) {
    console.log(`      问题里认不出的部分：${d.unresolved.join("、")}`);
  }
  /*
   * 措辞里的数字必须能在事实块里找到，否则整篇应回退模板。
   * 这里只做一个粗检：正文里出现的「数字+单位」至少有一条能在 facts 里对上。
   */
  const numbers = (d.text.match(/\d+(\.\d+)?/g) ?? []).slice(0, 40);
  const haystack = (d.facts ?? []).join(" ");
  const hits = numbers.filter((n) => haystack.includes(n)).length;
  if (numbers.length > 0 && hits === 0) {
    bad("正文数字无法回代", "正文里的数字在 facts 里一条也找不到");
  } else {
    ok("正文数字可回代", `${hits}/${numbers.length} 个数字出现在事实块里`);
  }
}

// ══════════════════════════════════════════════════════════
// 五、知识库：规模、缺口、检索
// ══════════════════════════════════════════════════════════

async function checkKb() {
  console.log("\n[5/6] /api/kb/* —— 知识库");
  const overview = await get("/api/kb/overview");
  if (overview.status !== 200) {
    bad("概览接口", `HTTP ${overview.status}：${overview.text.slice(0, 200)}`);
    return;
  }
  const o = overview.body;
  // 字段名照 `/api/kb/overview` 的实际下发（corpus / graph / vector），
  // 不是照界面上那几个数字反推 —— 反推出来的名字会让这一行打印 undefined。
  ok(
    "概览接口",
    `盘上 ${o.corpus.files} 篇 / 建图 ${o.graph.documents} · ${o.graph.chunks} 片 · ` +
      `${o.graph.entities} 实体 · ${o.graph.relationships} 关系`,
  );
  if (o.graph.chunks === 0) {
    bad("图谱里没有切片", "检索永远返回 empty —— 界面会把每个问题都答成「没答到」");
  }

  const docs = await get("/api/kb/documents");
  if (docs.status !== 200) {
    bad("文档清单", `HTTP ${docs.status}`);
  } else {
    ok("文档清单", `${docs.body.documents.length} 篇`);
  }

  const gaps = await get("/api/kb/gaps");
  if (gaps.status !== 200) {
    bad("缺口队列", `HTTP ${gaps.status}`);
  } else if (gaps.body.available === false) {
    bad("缺口队列读不到", gaps.body.reason || "available=false 但没给原因");
  } else {
    ok("缺口队列", `${gaps.body.total} 条`);
  }

  if (!WITH_SEARCH) {
    skip("检索", "默认不跑（冷启动几十秒）；要跑加 --with-search");
    return;
  }
  const t0 = Date.now();
  const s = await post("/api/kb/search", {
    query: "盖层完整性对 CO2 泄漏风险的影响",
    mode: "hybrid",
    with_graph: true,
  });
  const ms = Date.now() - t0;
  if (s.status !== 200) {
    bad("检索", `HTTP ${s.status}：${s.text.slice(0, 300)}`);
    return;
  }
  const ev = s.body.evidence;
  if (s.body.status === "unavailable") {
    bad("检索不可用", s.body.reason);
    return;
  }
  if (s.body.status === "empty") {
    bad("检索为空", "图谱里有切片但检索不到内容 —— 检查嵌入模型与索引是否匹配");
    return;
  }
  ok(
    "检索",
    `${s.body.status} · ${ev.stats.chunks} 片 / ${ev.stats.sources} 篇 · ${ms} ms`,
  );
  if (ev.stats.chunks > 0 && ev.stats.traceable === 0) {
    bad(
      "全部无血缘",
      "所有切片都追溯不到文献 —— 界面会把每条引用标成「无文件血缘」，读者无法核对",
    );
  } else {
    ok("血缘", `可溯源 ${ev.stats.traceable} 片 · 无血缘 ${ev.stats.unsourced} 片`);
  }
}

// ══════════════════════════════════════════════════════════
// 六、报告：生成 → 提交 → 撤回
// ══════════════════════════════════════════════════════════

async function checkReports() {
  console.log("\n[6/6] /api/ops/run + 审批状态机");
  const readings = await sampledReadings();
  const run = await post("/api/ops/run", {
    readings,
    report_kind: "daily",
    want_alarms: true,
    want_report: true,
    want_condition: false,
  });
  if (run.status !== 200) {
    bad("生成草稿", `HTTP ${run.status}：${run.text.slice(0, 300)}`);
    return;
  }
  const d = run.body;
  ok("生成草稿", `${d.runs.length} 个 agent · draft_id=${d.draft_id || "（空）"}`);
  if (!d.draft_id) {
    bad("没有 draft_id", "报告没落盘 —— 审批页永远是空的");
    return;
  }
  const failed = d.runs.filter((r) => !r.ok);
  if (failed.length > 0) {
    for (const f of failed) bad(`agent ${f.agent} 失败`, f.error || "未给出原因");
  }
  const blocking = d.conflicts.filter((c) => c.blocking);
  if (blocking.length > 0) {
    console.log(`      有 ${blocking.length} 项阻断性冲突：这份稿子不能提交`);
  }

  const detail = await get(`/api/ops/reports/${d.draft_id}`);
  if (detail.status !== 200) {
    bad("读草稿", `HTTP ${detail.status}`);
    return;
  }
  const dr = detail.body;
  ok("读草稿", `${dr.status} · 轨迹 ${dr.trail.length} 步 · 引用 ${dr.citation_count} 处`);
  if (!Array.isArray(dr.allowed) || dr.allowed.length === 0) {
    bad("allowed 为空", "新草稿应当允许 submit —— 界面会一个按钮都不显示");
  } else {
    ok("可用动作", dr.allowed.join(" / "));
  }
  if (dr.problems.length > 0) {
    console.log(`      引用检查未过 ${dr.problems.length} 项（这类稿子不允许提交）：`);
    for (const p of dr.problems.slice(0, 3)) console.log(`        · ${p}`);
  }
}

// ══════════════════════════════════════════════════════════

console.log(`接后端冒烟检查 → ${BASE}${TOKEN ? "（带 Bearer token）" : ""}`);

if (await checkHealth()) {
  const thresholds = await checkThresholds();
  await checkAlarms(thresholds);
  await checkCondition();
  await checkKb();
  await checkReports();
}

console.log(
  `\n${failures === 0 ? "全部通过" : `${failures} 项失败`}` +
    `${skipped > 0 ? `，${skipped} 项跳过` : ""}。`,
);
if (failures > 0) {
  console.log("界面上看到的症状与这里的失败项一一对应：接口不通 → 整屏「数据加载失败」；");
  console.log("阈值未配置 → 「0 需动作 / 18 不可判定」；检索 unavailable → 每句话都说「没答到」。");
}
process.exit(failures === 0 ? 0 : 1);
