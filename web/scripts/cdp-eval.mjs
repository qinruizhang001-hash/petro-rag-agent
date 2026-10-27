#!/usr/bin/env node
/**
 * 在真实浏览器里执行一段 JS 并打印结果。
 *
 * ## 为什么需要它
 *
 * 调试"这个数字为什么是这样"时，最快的路径永远是直接在页面的上下文里
 * 求值，而不是改代码、重新构建、再截图。这个脚本把那条路径缩短成一行命令：
 *
 * ```bash
 * node scripts/cdp-eval.mjs --expr "document.title"
 * node scripts/cdp-eval.mjs --url "http://127.0.0.1:4173/?mode=deck" \
 *   --expr "JSON.stringify(window.__perf && 'ok')"
 *
 * # CPU 降速 20 倍，看自适应降档会不会触发（不需要真找一台慢机器）
 * node scripts/cdp-eval.mjs --throttle 20 --ready "typeof window.__perf === 'function'" \
 *   --expr "new Promise(r => setTimeout(async () => r(JSON.stringify((await window.__perf()).scene)), 8000))"
 * ```
 *
 * `--throttle` 存在的理由：自适应降级只在**慢机器**上触发，而开发机不慢。
 * 一个只在别人机器上生效的控制器，等于没有测过 —— CPU 降速是唯一能在本机
 * 复现"慢"的手段，而且它走的是同一条真实代码路径（rAF 变慢 → 中位数超阈值）。
 */

import { openPage } from "./lib/cdp.mjs";

function arg(name, fallback) {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 && process.argv[i + 1] ? process.argv[i + 1] : fallback;
}

const url = arg("url", "http://127.0.0.1:4173/");
const expr = arg("expr", "document.title");
const ready = arg("ready", "document.readyState === 'complete'");
const throttle = Number(arg("throttle", "1"));

const page = await openPage(url, { readyExpr: ready });
try {
  if (throttle > 1) {
    await page.send("Emulation.setCPUThrottlingRate", { rate: throttle });
    console.log(`（CPU 降速 ${throttle}×）`);
  }
  const value = await page.evaluate(expr);
  console.log(typeof value === "string" ? value : JSON.stringify(value, null, 2));
} finally {
  await page.close();
}
