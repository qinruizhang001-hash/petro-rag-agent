#!/usr/bin/env node
/**
 * 实测渲染帧率（CDP）。
 *
 * ## 为什么需要这个脚本
 *
 * "在集显上能跑 60 FPS"是一句必须能被证伪的话。人手看一眼说"挺流畅"，
 * 在 120 Hz 屏上掉到 40 帧也感觉不出来，而在一台开了省电模式的办公本上
 * 掉到 20 帧会立刻被投诉 —— 两种结论都拿不出证据。
 *
 * 这个脚本做一件事：打开页面，读 `window.__perf()`，打印可贴进文档的数字。
 *
 * ## 三个刻意的选择
 *
 * 1. **有头浏览器，不用 headless。**
 *    headless Chrome 默认走 SwiftShader（纯 CPU 光栅化），帧率比真实 GPU
 *    低一个数量级 —— 拿它当"集显 60 FPS"的证据是反向作弊。而且后台标签页
 *    的 `requestAnimationFrame` 会被节流，headless 里窗口不可见时更容易触发。
 *
 * 2. **采样两轮，只信第二轮。**
 *    第一轮包含着色器编译、纹理上传、模块加载。把它们算进 p50，
 *    得到的是"启动耗时"而不是"稳态帧率"。
 *
 * 3. **先量帧率，后查显卡。**
 *    查显卡要新建一个 WebGL 上下文，那个上下文本身会占显存、触发一次
 *    GPU 唤醒。顺序反过来会污染测量。
 *
 * ## 用法
 *
 * ```bash
 * # 另开一个终端：npm run preview -- --host 127.0.0.1 --port 4173
 * node scripts/measure-fps.mjs
 * node scripts/measure-fps.mjs --url "http://127.0.0.1:4173/?mode=deck&perf=1&kb=search"
 * ```
 *
 * 退出码：p50 超过一帧预算，或超预算帧数占比 > 5% → 1。
 */

import { spawn } from "node:child_process";
import { existsSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const DEFAULT_URL = "http://127.0.0.1:4173/?mode=deck&perf=1";
const BROWSERS = [
  "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe",
  "C:\\Program Files\\Microsoft\\Edge\\Application\\msedge.exe",
  "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
  "/usr/bin/google-chrome",
  "/usr/bin/chromium",
];

function arg(name, fallback) {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 && process.argv[i + 1] ? process.argv[i + 1] : fallback;
}

const url = arg("url", DEFAULT_URL);
const port = Number(arg("port", "9222"));
const budget = 1000 / 60;

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function httpJson(path) {
  const res = await fetch(`http://127.0.0.1:${port}${path}`);
  if (!res.ok) throw new Error(`${path} → HTTP ${res.status}`);
  return res.json();
}

/** 极简 CDP 客户端：一条 WebSocket，请求按 id 配对。 */
function connect(wsUrl) {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(wsUrl);
    const pending = new Map();
    let seq = 0;

    ws.addEventListener("open", () =>
      resolve({
        send(method, params = {}) {
          const id = ++seq;
          return new Promise((res, rej) => {
            pending.set(id, { res, rej });
            ws.send(JSON.stringify({ id, method, params }));
          });
        },
        close: () => ws.close(),
      }),
    );
    ws.addEventListener("error", () => reject(new Error(`CDP 连接失败：${wsUrl}`)));
    ws.addEventListener("message", (ev) => {
      const msg = JSON.parse(ev.data);
      const p = pending.get(msg.id);
      if (!p) return;
      pending.delete(msg.id);
      if (msg.error) p.rej(new Error(msg.error.message));
      else p.res(msg.result);
    });
  });
}

async function evaluate(cdp, expression) {
  const { result, exceptionDetails } = await cdp.send("Runtime.evaluate", {
    expression,
    awaitPromise: true,
    returnByValue: true,
  });
  if (exceptionDetails) {
    throw new Error(exceptionDetails.exception?.description ?? exceptionDetails.text);
  }
  return result.value;
}

async function main() {
  const exe = BROWSERS.find((p) => existsSync(p));
  if (!exe) {
    console.error("找不到 Chrome / Edge。请手动指定 BROWSERS 里的路径。");
    process.exit(2);
  }

  // 先确认页面在服务上。一个"浏览器打开了 404"会表现为 __perf 永远不存在，
  // 最后报"超时"—— 那个错误指向的是脚本，而真实原因是服务器没起。
  try {
    const res = await fetch(url, { method: "GET" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
  } catch (e) {
    console.error(`打不开 ${url}\n  ${e.message}\n先跑：npm run preview -- --host 127.0.0.1 --port 4173`);
    process.exit(2);
  }

  const profile = mkdtempSync(join(tmpdir(), "ccus-perf-"));
  const child = spawn(
    exe,
    [
      `--remote-debugging-port=${port}`,
      `--user-data-dir=${profile}`,
      "--no-first-run",
      "--no-default-browser-check",
      "--disable-extensions",
      "--no-proxy-server",
      "--window-size=1680,1000",
      url,
    ],
    { stdio: "ignore", detached: false },
  );

  let cdp;
  try {
    // 等调试端口起来
    let version = null;
    for (let i = 0; i < 60 && !version; i++) {
      try {
        version = await httpJson("/json/version");
      } catch {
        await sleep(250);
      }
    }
    if (!version) throw new Error("调试端口 60 次轮询都没起来");

    // 找到页面 target（新标签页 / about:blank 也可能在列表里）
    let page = null;
    for (let i = 0; i < 40 && !page; i++) {
      const list = await httpJson("/json/list");
      page = list.find((t) => t.type === "page" && t.url.startsWith("http"));
      if (!page) await sleep(250);
    }
    if (!page) throw new Error("没有找到页面 target");

    cdp = await connect(page.webSocketDebuggerUrl);
    await cdp.send("Runtime.enable");

    // 等 `window.__perf` 就位（React 挂载 + 动态 import three 之后才有）
    let ready = false;
    for (let i = 0; i < 80 && !ready; i++) {
      ready = await evaluate(cdp, "typeof window.__perf === 'function'");
      if (!ready) await sleep(250);
    }
    if (!ready) {
      throw new Error("window.__perf 一直没出现 —— 页面是不是没带 ?perf=1，或者渲染报错了");
    }

    // 第一轮丢掉：含着色器编译与纹理上传
    await evaluate(cdp, "window.__perf()");
    await sleep(600);

    const report = await evaluate(cdp, "window.__perf()");

    // GPU 名字取自 `scene.gpu`，也就是页面**已经在用的那个**上下文。
    // 不在这里另开 canvas 查：新建上下文会占显存、触发一次 GPU 唤醒，
    // 而这个调用紧跟在测量之后，很容易把最后几帧的数据污染掉。
    print(report, url);
    process.exitCode = verdict(report) ? 0 : 1;
  } finally {
    cdp?.close();
    child.kill();
    await sleep(300);
    try {
      rmSync(profile, { recursive: true, force: true });
    } catch {
      /* 浏览器可能还握着文件句柄，留给系统清理 */
    }
  }
}

function print(r, target) {
  const f = r.frames;
  const s = r.scene;
  console.log("");
  console.log(`  目标        ${target}`);
  console.log(`  显卡        ${s ? s.gpu : "（场景未挂载，读不到）"}`);
  console.log(`  采样         ${f.frames} 帧，预算 ${r.budgetMs} ms`);
  console.log("");
  console.log(`  FPS (p50)   ${f.fps}`);
  console.log(`  p50 / p95   ${f.p50} / ${f.p95} ms`);
  console.log(`  p99 / max   ${f.p99} / ${f.max} ms`);
  console.log(`  超预算帧     ${f.overBudget} / ${f.frames}`);
  if (s) {
    console.log("");
    console.log(`  draw calls  ${s.drawCalls}`);
    console.log(`  triangles   ${s.triangles.toLocaleString()}`);
    console.log(`  points      ${s.points.toLocaleString()}`);
    console.log(`  geometries  ${s.geometries}   textures ${s.textures}   programs ${s.programs}`);
    console.log(`  画布        ${s.width}×${s.height} @${s.pixelRatio}x`);
    console.log(`  渲染循环     ${s.animating ? "运行中" : "已停止（工程师模式）"}`);
  } else {
    console.log("");
    console.log("  场景        未挂载（中央区不在剖面 Tab，或 WebGL 不可用）");
  }
  /*
   * FPS 高于 120 只可能意味着 rAF 没锁 vsync（远程桌面、虚拟显示器、
   * 某些驱动的省电策略）。这时 `fps` 读作"每帧还剩多少余量"，
   * 不读作"用户看到的流畅度" —— 不加这句，303 会被当成"这台机器能跑 303 帧"。
   */
  if (f.fps > 120) {
    console.log("");
    console.log(`  注：rAF 未锁 vsync（${f.fps} > 120），FPS 读作余量而非观感。`);
    console.log(`      判断依据是 p50 ${f.p50} ms —— 它才是这一帧的真实成本。`);
  }
  console.log("");
}

function verdict(r) {
  const f = r.frames;
  const reasons = [];
  // p50 卡在 16.7 说明已经顶到 60 Hz 上限，这时 p50 > budget 是浮点噪声
  if (f.p50 > budget * 1.05) reasons.push(`p50 ${f.p50} ms 超过预算 ${budget.toFixed(1)} ms`);
  if (f.overBudget > f.frames * 0.05) {
    reasons.push(`超预算帧 ${f.overBudget}/${f.frames} 超过 5%`);
  }
  if (!r.scene) reasons.push("没有读到场景（无法证明是 3D 在跑）");
  else if (!r.scene.animating) reasons.push("渲染循环没在跑，测到的是空转的 rAF");

  if (reasons.length === 0) {
    console.log(`  ✓ 通过：p50 ${f.p50} ms，超预算 ${f.overBudget}/${f.frames}`);
    return true;
  }
  console.log("  ✗ 未通过：");
  for (const x of reasons) console.log(`    · ${x}`);
  return false;
}

main().catch((e) => {
  console.error(`测量失败：${e.message}`);
  process.exit(2);
});
