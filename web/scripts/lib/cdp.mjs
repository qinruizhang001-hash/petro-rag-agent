/**
 * 极简 CDP 客户端（无第三方依赖）。
 *
 * ## 为什么不用 puppeteer / playwright
 *
 * 这里只需要三件事：开一个浏览器、连上一个页面、执行一段 JS 拿回结果。
 * puppeteer 会带来一个自带 Chromium 的 300 MB 依赖，而"用用户机器上真实
 * 存在的那块 GPU"恰恰是这套测量成立的前提 —— 自带一个浏览器反而让
 * "在集显上量"这件事变得更不可控。
 *
 * ## 为什么必须有头
 *
 * headless Chrome 默认走 SwiftShader（纯 CPU 光栅化），帧率比真实 GPU
 * 低一个数量级；后台标签页的 `requestAnimationFrame` 还会被节流。
 * 拿 headless 的数字当"集显 60 FPS"的证据是反向作弊。
 */

import { spawn } from "node:child_process";
import { existsSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const BROWSERS = [
  "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe",
  "C:\\Program Files\\Microsoft\\Edge\\Application\\msedge.exe",
  "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
  "/usr/bin/google-chrome",
  "/usr/bin/chromium",
];

export const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function httpJson(port, path) {
  const res = await fetch(`http://127.0.0.1:${port}${path}`);
  if (!res.ok) throw new Error(`${path} → HTTP ${res.status}`);
  return res.json();
}

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

/**
 * 打开一个页面，等 `readyExpr` 成立，返回 `{ evaluate, close }`。
 *
 * 等不到就抛错 —— 一个"浏览器打开了 404"会表现为永远等不到就绪，
 * 最后报超时，而那个错误指向脚本，真实原因却是服务器没起。
 */
export async function openPage(url, { port = 9222, readyExpr = "true", timeoutMs = 25000 } = {}) {
  const exe = BROWSERS.find((p) => existsSync(p));
  if (!exe) {
    throw new Error("找不到 Chrome / Edge。请手动改 scripts/lib/cdp.mjs 里的 BROWSERS。");
  }

  try {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
  } catch (e) {
    throw new Error(
      `打不开 ${url}\n  ${e.message}\n先跑：npm run preview -- --host 127.0.0.1 --port 4173`,
    );
  }

  const profile = mkdtempSync(join(tmpdir(), "ccus-cdp-"));
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

  let cdp = null;
  let closed = false;
  const close = async () => {
    if (closed) return;
    closed = true;
    cdp?.close();
    child.kill();
    await sleep(300);
    try {
      rmSync(profile, { recursive: true, force: true });
    } catch {
      /* 浏览器可能还握着文件句柄，留给系统清理 */
    }
  };

  try {
    let version = null;
    for (let i = 0; i < 60 && !version; i++) {
      try {
        version = await httpJson(port, "/json/version");
      } catch {
        await sleep(250);
      }
    }
    if (!version) throw new Error("调试端口 60 次轮询都没起来");

    let page = null;
    for (let i = 0; i < 40 && !page; i++) {
      const list = await httpJson(port, "/json/list");
      page = list.find((t) => t.type === "page" && t.url.startsWith("http"));
      if (!page) await sleep(250);
    }
    if (!page) throw new Error("没有找到页面 target");

    cdp = await connect(page.webSocketDebuggerUrl);
    await cdp.send("Runtime.enable");

    const evaluate = async (expression) => {
      const { result, exceptionDetails } = await cdp.send("Runtime.evaluate", {
        expression,
        awaitPromise: true,
        returnByValue: true,
      });
      if (exceptionDetails) {
        throw new Error(exceptionDetails.exception?.description ?? exceptionDetails.text);
      }
      return result.value;
    };

    const deadline = Date.now() + timeoutMs;
    while (Date.now() < deadline) {
      if (await evaluate(readyExpr)) return { evaluate, send: cdp.send, close };
      await sleep(250);
    }
    throw new Error(`等不到就绪条件：${readyExpr}`);
  } catch (e) {
    await close();
    throw e;
  }
}
