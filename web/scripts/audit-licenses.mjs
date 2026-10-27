#!/usr/bin/env node
/**
 * 依赖许可证审计。
 *
 * ## 为什么必须扫传递依赖
 *
 * 只看 `package.json` 里那 13 个直接依赖是不够的：GPL 传染性判断看的是
 * **整个依赖树**。一个 MIT 的库完全可以在自己的依赖里带一个 GPL 的解析器，
 * 而那一层不会出现在任何人的 `package.json` 里。实测这个项目 144 个包，
 * 直接依赖只有 13 个 —— 漏掉的是 131 个。
 *
 * ## 判定规则
 *
 * | 类别 | 协议 | 处置 |
 * |---|---|---|
 * | 放行 | MIT / ISC / Apache-2.0 / BSD-2/3-Clause / 0BSD / CC0 / Unlicense | — |
 * | 需署名 | CC-BY-3.0/4.0 | 允许商用，但必须保留署名 |
 * | **阻断** | GPL-* / AGPL-* / LGPL-* / SSPL / OSL / EUPL / CC-BY-NC-* | 退出码 1 |
 *
 * LGPL 也阻断：动态链接的 LGPL 通常可以闭源分发，但这个项目的前端是
 * **打包**（bundle）的，代码被内联进一个文件，"动态链接"这个前提不成立。
 * 与其在法务上争这条线，不如直接不用。
 *
 * CC-BY-NC 阻断：非商用条款与"以后卖钱"直接冲突。
 *
 * ## 为什么不用 license-checker 之类的现成工具
 *
 * 它们的输出是一张几百行的表，而这里需要的是一个**能进 CI 的判定**。
 * 四十行代码换掉一个依赖，并且规则写在代码里、可以被审查。
 *
 * ```bash
 * node scripts/audit-licenses.mjs           # 人类可读
 * node scripts/audit-licenses.mjs --json    # 机器可读
 * ```
 */

import { readdirSync, readFileSync, existsSync } from "node:fs";
import { join } from "node:path";

const ROOT = new URL("../node_modules/", import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, "$1");

/**
 * 阻断协议的特征词。**顺序有意义**：先排掉 AFFERO 与 LESSER，否则它们会被
 * 后面的 "GENERAL PUBLIC LICENSE" 吞掉。
 *
 * 为什么不用 `^AGPL` 这种锚定写法：真实世界里的 `license` 字段不一定是
 * SPDX 标识符。Python 侧就踩过这个坑 —— `pymupdf` 写的是
 * "Dual Licensed - GNU AFFERO GPL 3.0 or Artifex Commercial License"，
 * 锚定 `^AGPL` 完全匹配不到，于是审查脚本自己漏掉了唯一一个真正阻断的依赖。
 * 锚定只对规范字段成立，而审查的对象恰恰是不规范的字段。
 */
const DENY_MARKERS = [
  "AGPL",
  "AFFERO",
  "LGPL",
  "LESSER GENERAL PUBLIC",
  "SSPL",
  "SERVER SIDE PUBLIC",
  "EUPL",
  "EUROPEAN UNION PUBLIC",
  "OPEN SOFTWARE LICENSE",
  "CPAL",
  "NONCOMMERCIAL",
  "NON-COMMERCIAL",
  "NON COMMERCIAL",
  "GENERAL PUBLIC LICENSE",
  "GPL",
  "OSL",
  "RPL",
];

/** 需要保留署名的协议。允许商用，但产物里要能说明用了谁的东西。 */
const ATTRIBUTION_MARKERS = ["CC-BY", "CREATIVE COMMONS ATTRIBUTION"];

/**
 * 判定。返回 `deny` / `attribution` / `unknown` / `ok`。
 *
 * 与 `scripts/audit_python_licenses.py` 的 `classify()` 保持同一套规则 ——
 * 两边不一致时，前端说"干净"、后端说"有 AGPL"，而没人知道该信哪个。
 */
function classify(license) {
  const s = String(license).toUpperCase().split(/\s+/).join(" ");
  if (!s || s === "UNKNOWN" || s === "UNREADABLE") return "unknown";
  if (DENY_MARKERS.some((m) => s.includes(m))) return "deny";
  // CC-BY-NC：`NC` 两个字母单独匹配太容易误伤，限定在 CC-BY 语境里判
  if (s.includes("CC-BY") && /-NC\b/.test(s)) return "deny";
  if (ATTRIBUTION_MARKERS.some((m) => s.includes(m))) return "attribution";
  return "ok";
}

function licenseOf(pkg) {
  const raw = pkg.license ?? pkg.licenses;
  if (typeof raw === "string") return raw.trim();
  if (Array.isArray(raw)) {
    return raw
      .map((x) => (typeof x === "string" ? x : x?.type))
      .filter(Boolean)
      .join(" OR ");
  }
  if (raw && typeof raw === "object" && raw.type) return String(raw.type);
  // 有些包只在 README / LICENSE 文件里写协议，package.json 里没有字段。
  // 不猜 —— 标成 UNKNOWN，让人去看。
  return "UNKNOWN";
}

/**
 * 守卫的守卫。
 *
 * 一张正则表最容易出的问题是"它其实什么也没匹配到"：写错一个字，
 * 所有包都判为放行，然后 CI 亮着绿灯说"无 GPL"—— 一个替没做的检查
 * 背书的绿灯，比没有这道检查更危险。
 *
 * 所以这里对 DENY 表本身做断言：必须命中四种典型 copyleft，
 * 必须**不**误伤四种常见宽松协议。
 */
const SELF_TEST = {
  mustMatch: ["GPL-2.0", "GPL-3.0-only", "AGPL-3.0", "LGPL-2.1", "SSPL-1.0", "CC-BY-NC-4.0"],
  mustNotMatch: ["MIT", "ISC", "Apache-2.0", "BSD-3-Clause", "CC-BY-4.0", "0BSD", "Unlicense"],
};

function selfTest() {
  const bad = [];
  for (const s of SELF_TEST.mustMatch) {
    if (classify(s) !== "deny") bad.push(`应阻断却判成 ${classify(s)}：${s}`);
  }
  for (const s of SELF_TEST.mustNotMatch) {
    const v = classify(s);
    if (v === "deny") bad.push(`应放行却阻断：${s}`);
  }
  // CC-BY 必须落在 attribution 而不是 ok —— 否则"需要署名"这件事会静默消失
  if (classify("CC-BY-4.0") !== "attribution") bad.push("CC-BY-4.0 应判为需署名");
  if (bad.length) {
    console.error("\n  ✗ 阻断名单自检失败：");
    for (const b of bad) console.error(`    · ${b}`);
    console.error("");
    process.exit(2);
  }
  console.log(
    `  ✓ 阻断名单自检通过（命中 ${SELF_TEST.mustMatch.length} 个 copyleft，放行 ${SELF_TEST.mustNotMatch.length} 个宽松协议）`,
  );
}

if (process.argv.includes("--self-test")) {
  selfTest();
  process.exit(0);
}


/** 递归遍历 node_modules，包含嵌套的 node_modules 与 @scope 目录。 */
function walk(dir, out, depth = 0) {
  if (depth > 6 || !existsSync(dir)) return;
  let entries;
  try {
    entries = readdirSync(dir, { withFileTypes: true });
  } catch {
    return;
  }
  for (const e of entries) {
    if (!e.isDirectory()) continue;
    if (e.name === ".bin" || e.name === ".cache") continue;
    const full = join(dir, e.name);
    if (e.name.startsWith("@")) {
      walk(full, out, depth); // scope 目录本身不是包
      continue;
    }
    const pj = join(full, "package.json");
    if (existsSync(pj)) {
      try {
        const pkg = JSON.parse(readFileSync(pj, "utf8"));
        out.push({ name: pkg.name ?? e.name, version: pkg.version ?? "?", license: licenseOf(pkg) });
      } catch {
        out.push({ name: e.name, version: "?", license: "UNREADABLE" });
      }
    }
    walk(join(full, "node_modules"), out, depth + 1);
  }
}

const packages = [];
walk(ROOT, packages);

// 去重：同一个包可能在多个位置出现（npm 的嵌套安装）
const seen = new Map();
for (const p of packages) {
  const key = `${p.name}@${p.version}`;
  if (!seen.has(key)) seen.set(key, p);
}
const all = [...seen.values()].sort((a, b) => a.name.localeCompare(b.name));

for (const p of all) p.verdict = classify(p.license);
const denied = all.filter((p) => p.verdict === "deny");
const attribution = all.filter((p) => p.verdict === "attribution");
const unknown = all.filter((p) => p.verdict === "unknown");

const byLicense = new Map();
for (const p of all) byLicense.set(p.license, (byLicense.get(p.license) ?? 0) + 1);
const dist = [...byLicense.entries()].sort((a, b) => b[1] - a[1]);

if (process.argv.includes("--json")) {
  console.log(
    JSON.stringify(
      {
        total: all.length,
        distribution: Object.fromEntries(dist),
        denied: denied.map((p) => `${p.name}@${p.version} → ${p.license}`),
        attribution: attribution.map((p) => `${p.name}@${p.version} → ${p.license}`),
        unknown: unknown.map((p) => `${p.name}@${p.version}`),
      },
      null,
      2,
    ),
  );
  process.exit(denied.length ? 1 : 0);
}

console.log(`\n  扫描 ${all.length} 个包（含传递依赖）\n`);
for (const [lic, n] of dist) {
  const v = classify(lic);
  const flag = v === "deny" ? "  ✗ 阻断" : v === "attribution" ? "  · 需署名" : "";
  console.log(`  ${String(n).padStart(4)}  ${lic}${flag}`);
}

if (attribution.length) {
  console.log("\n  需要保留署名的包：");
  for (const p of attribution) console.log(`    · ${p.name}@${p.version} → ${p.license}`);
}

if (unknown.length) {
  // 不判失败，但必须让人看见：一个没有 license 字段的包，
  // 在法律上等于"未授予任何权利"，比 GPL 更危险。
  console.log("\n  ⚠ 协议未知（package.json 里没有 license 字段）：");
  for (const p of unknown) console.log(`    · ${p.name}@${p.version}`);
}

if (denied.length) {
  console.log("\n  ✗ 发现阻断协议：");
  for (const p of denied) console.log(`    · ${p.name}@${p.version} → ${p.license}`);
  console.log("");
  process.exit(1);
}

// 守卫的守卫：如果 ROOT 解析错了（例如在错误的目录下运行），
// 上面会扫到 0 个包，然后"零阻断"这个绿灯就替一次没做的检查背书。
if (all.length < 10) {
  console.error(`\n  ✗ 只扫到 ${all.length} 个包，路径大概不对：${ROOT}\n`);
  process.exit(2);
}

console.log(`\n  ✓ ${all.length} 个包，无 GPL / AGPL / LGPL / SSPL\n`);
