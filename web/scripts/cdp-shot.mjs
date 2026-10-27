#!/usr/bin/env node
/**
 * 给页面拍一张确定性截图。
 *
 * ## 为什么要"确定性"
 *
 * 文档里的截图要能反复重拍并且逐像素对得上，否则"这张图是不是旧版本"
 * 永远说不清。所以这里做三件事：
 *
 * 1. **固定视口**（默认 1680×1000）。断点决定了三栏是否出现，
 *    而 1120 宽时右栏会被 `xl` 隐藏 —— 截出来是一张不同的界面。
 * 2. **等 `readyExpr`**，而不是 `setTimeout`。等固定时长在慢机器上会截到
 *    "正在构建剖面…"，而那正是最需要证明它已经建好的时候。
 * 3. **等两帧再拍**。`ResizeObserver` 触发的首次渲染发生在布局之后，
 *    早一拍会拍到空白画布。
 *
 * ```bash
 * node scripts/cdp-shot.mjs --out docs/ui/web/deck.png --url "http://127.0.0.1:4173/?mode=deck"
 * ```
 */

import { writeFileSync } from "node:fs";
import { openPage, sleep } from "./lib/cdp.mjs";

function arg(name, fallback) {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 && process.argv[i + 1] ? process.argv[i + 1] : fallback;
}

const url = arg("url", "http://127.0.0.1:4173/?mode=deck");
const out = arg("out", "shot.png");
const width = Number(arg("width", "1680"));
const height = Number(arg("height", "1000"));
// 默认等画布建好：`ready` 标志由 GeoSectionCanvas 在首帧后置位。
// 不在剖面 Tab 上的页面没有这个标志，所以默认条件放宽成"DOM 就绪"。
const ready = arg(
  "ready",
  "document.readyState === 'complete' && !document.body.innerText.includes('正在构建剖面')",
);

const page = await openPage(url, { readyExpr: ready });
try {
  await page.send("Emulation.setDeviceMetricsOverride", {
    width,
    height,
    deviceScaleFactor: 1,
    mobile: false,
  });
  // 视口变了要重新布局，ResizeObserver → resize → 渲染一帧。等两拍。
  await page.evaluate(
    "new Promise(r => requestAnimationFrame(() => requestAnimationFrame(() => setTimeout(r, 250))))",
  );
  await sleep(150);

  const { data } = await page.send("Page.captureScreenshot", { format: "png" });
  writeFileSync(out, Buffer.from(data, "base64"));
  console.log(`已保存 ${out}（${width}×${height}）`);
} finally {
  await page.close();
}
