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
 *
 * ## 需要先操作才能到达的状态，写进参数，不要写进人脑
 *
 * 「知识库检索结果」那一屏只有点一次检索才会出现，而检索要十几秒。
 * 手点 + 手截的后果是这张图没法重拍 —— 换个查询词就是另一批文献，
 * 图里的 `12 片 / 6 篇` 与任何一次新的检索都对不上。
 *
 * 所以 `--kb-search "<问句>"` 把这一步也交给脚本：填词、点检索、
 * **等按钮从「检索中」回到「检索」**（等状态而不是等时长），再拍。
 * 命令里带着问句，重拍就是同一次检索。
 *
 * `--ask "<问句>"` 同理，拍的是对话面板的回答。它比检索多一步：正文
 * 由模板生成（几秒），引用来自检索（十几秒），两者不是同一个完成时刻。
 * 只等按钮回到「发送」会拍到「数字依据 / 原文证据」还空着的中间态 ——
 * 那张图看起来像"系统答了但没有依据"。所以还要等「数字依据」出现。
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
const kbSearch = arg("kb-search", "");
const ask = arg("ask", "");

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

  if (kbSearch) {
    /*
     * 用 native setter 设值，不是 `el.value = q`。
     *
     * React 受控组件靠自己的 value tracker 判断"变了没有"：直接赋 value
     * 会让 tracker 记下同一个值，随后的 input 事件被当成"没变"而丢弃，
     * 于是 query 一直是空的、按钮一直是 disabled、点击什么都不发生 ——
     * 而截图会拍成"检索面板本来就这样"。设完还要派发 input 事件。
     */
    const res = await page.evaluate(`(async () => {
      const sleep = (ms) => new Promise((s) => setTimeout(s, ms));
      const el = document.querySelector('[aria-label="知识库检索"]');
      if (!el) return { ok: false, why: "找不到检索输入框（URL 里缺 ?kb=search？）" };
      const desc = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value");
      desc.set.call(el, ${JSON.stringify(kbSearch)});
      el.dispatchEvent(new Event("input", { bubbles: true }));
      await sleep(400);
      // 提交按钮是输入框的**兄弟**。按文字找会先命中「概览 / 检索」那个页签，
      // 点了只是切页签，不发请求 —— 而界面上看起来一模一样。
      const btn = el.parentElement.querySelector("button");
      if (!btn) return { ok: false, why: "找不到提交按钮" };
      if (btn.disabled) return { ok: false, why: "提交按钮是 disabled（问句没进 React 状态）" };
      btn.click();
      // 等状态，不等时长：检索本身 4~15 秒，取决于 LightRAG 是否预热。
      for (let i = 0; i < 150; i++) {
        await sleep(1000);
        if (btn.innerText.trim() !== "检索中") break;
      }
      await sleep(600);
      const text = document.body.innerText;
      return {
        ok: text.includes("证据强度"),
        why: text.includes("证据强度") ? "" : "检索结束了但没有结果区",
        head: (text.match(/\\d+ 片 \\/ \\d+ 篇/) || [])[0] || null,
      };
    })()`);
    if (!res || !res.ok) {
      throw new Error(`检索没能完成：${res ? res.why : "脚本返回空"}`);
    }
    console.log(`检索完成：${res.head ?? "(未取到计数)"}`);
  }

  if (ask) {
    const res = await page.evaluate(`(async () => {
      const sleep = (ms) => new Promise((s) => setTimeout(s, ms));
      const ta = document.querySelector('[aria-label="向系统提问"]');
      if (!ta) return { ok: false, why: "找不到提问框（URL 里缺 ?rail=dialog？）" };
      const desc = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value");
      desc.set.call(ta, ${JSON.stringify(ask)});
      ta.dispatchEvent(new Event("input", { bubbles: true }));
      await sleep(400);
      const btn = ta.parentElement.querySelector("button");
      if (!btn) return { ok: false, why: "找不到发送按钮" };
      if (btn.disabled) return { ok: false, why: "发送按钮是 disabled（问句没进 React 状态）" };
      btn.click();
      // 第一段：等正文。模板生成，几秒。
      for (let i = 0; i < 180; i++) {
        await sleep(1000);
        if (btn.innerText.trim() !== "处理中") break;
      }
      // 第二段：等引用。检索侧十几秒，与正文不是同一个完成时刻。
      let cited = false;
      for (let i = 0; i < 120; i++) {
        await sleep(1000);
        if (document.body.innerText.includes("数字依据")) { cited = true; break; }
      }
      await sleep(700);
      const text = document.body.innerText;
      return {
        ok: cited,
        why: cited ? "" : "正文回来了但「数字依据」一直没出现（引用侧没完成）",
        nums: (text.match(/数字依据（\\d+ 条/) || [])[0] || null,
        evidence: (text.match(/\\d+ 片 \\/ \\d+ 篇/) || [])[0] || null,
      };
    })()`);
    if (!res || !res.ok) {
      throw new Error(`对话没能完成：${res ? res.why : "脚本返回空"}`);
    }
    console.log(`对话完成：${res.nums ?? "-"} · 证据 ${res.evidence ?? "-"}`);
  }

  const { data } = await page.send("Page.captureScreenshot", { format: "png" });
  writeFileSync(out, Buffer.from(data, "base64"));
  console.log(`已保存 ${out}（${width}×${height}）`);
} finally {
  await page.close();
}
