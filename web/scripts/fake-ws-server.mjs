#!/usr/bin/env node
/**
 * 假的推送服务端（零依赖）。
 *
 * ## 为什么需要它
 *
 * `src/lib/push.ts` 是一条**真实存在但默认不启用**的通路。一条没有任何
 * 调用者的通路，和"写了没接上"在代码审查里长得一模一样 —— 而后者会在
 * 真正接入的那天才暴露：消息类型对不上、状态角标不更新、重连把组件搞崩。
 *
 * 所以这里起一个真的 WebSocket 服务端，让那条通路被真的走过一遍：
 *
 * ```bash
 * node scripts/fake-ws-server.mjs --port 8787 &
 * node scripts/cdp-eval.mjs --url "http://127.0.0.1:4173/?ws=ws://127.0.0.1:8787/" \
 *   --expr "new Promise(r=>setTimeout(()=>r(document.body.innerText.includes('推送已连接')),2500))"
 * ```
 *
 * ## 为什么不装 ws 包
 *
 * 只需要"握手 + 发文本帧"两件事，二十行。装一个包换来的是一份需要审计
 * 的许可证和一份需要维护的版本号（见 `docs/licenses.md`）。
 *
 * 它只发不收（除了关闭帧），这是刻意的：假服务端越简单，出问题时越容易
 * 判断是客户端的问题还是这个脚本的问题。
 */

import { readFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { createServer } from "node:http";

const MAGIC = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";

function arg(name, fallback) {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 && process.argv[i + 1] ? process.argv[i + 1] : fallback;
}

const port = Number(arg("port", "8787"));
/** 注入的严重告警数。默认 99 —— 一个 fixture 里绝不会出现的值，一眼能认出。 */
const critical = Number(arg("critical", "99"));

function frame(text) {
  const payload = Buffer.from(text, "utf8");
  const len = payload.length;
  let header;
  if (len < 126) {
    header = Buffer.from([0x81, len]);
  } else if (len < 65536) {
    header = Buffer.alloc(4);
    header[0] = 0x81;
    header[1] = 126;
    header.writeUInt16BE(len, 2);
  } else {
    header = Buffer.alloc(10);
    header[0] = 0x81;
    header[1] = 127;
    header.writeBigUInt64BE(BigInt(len), 2);
  }
  return Buffer.concat([header, payload]);
}

function load(name) {
  return JSON.parse(readFileSync(new URL(`../src/fixtures/${name}.json`, import.meta.url), "utf8"));
}

const server = createServer((req, res) => {
  res.writeHead(426, { "content-type": "text/plain" });
  res.end("This is a WebSocket endpoint. Connect with ws://\n");
});

server.on("upgrade", (req, socket) => {
  const key = req.headers["sec-websocket-key"];
  if (!key) {
    socket.destroy();
    return;
  }
  const accept = createHash("sha1").update(key + MAGIC).digest("base64");
  socket.write(
    "HTTP/1.1 101 Switching Protocols\r\n" +
      "Upgrade: websocket\r\n" +
      "Connection: Upgrade\r\n" +
      `Sec-WebSocket-Accept: ${accept}\r\n\r\n`,
  );

  const send = (obj) => {
    if (!socket.writable) return;
    socket.write(frame(JSON.stringify(obj)));
  };

  // 先推阈值，再推一份"严重告警数被改过"的报告。
  // 顺序是有意的：界面上要先看到阈值就位，再看到数字变 —— 两个都能观察，
  // 才说明 `subscribe` 的分发是真的按 `type` 走的，而不是"随便来一条就刷新"。
  send({ type: "thresholds", payload: load("thresholds") });

  const report = load("alarms");
  report.counts.critical = critical;
  send({ type: "alarm", payload: report });

  // 心跳：真实服务端也要发，否则中间的代理会在空闲 60 s 后掐掉连接。
  const timer = setInterval(() => send({ type: "ping" }), 1000);

  const stop = () => {
    clearInterval(timer);
    socket.destroy();
  };
  socket.on("close", stop);
  socket.on("error", stop);
  // 浏览器关闭标签页时会发一个关闭帧（opcode 8）。不处理它，
  // 这个 socket 会一直挂到进程退出 —— 而"服务端不回收连接"这件事
  // 在自测时表现为端口占用，很难联想到是这里。
  socket.on("data", (buf) => {
    if ((buf[0] & 0x0f) === 0x8) stop();
  });
});

server.listen(port, "127.0.0.1", () => {
  console.log(`假推送服务已启动：ws://127.0.0.1:${port}/  （critical=${critical}）`);
  console.log("Ctrl+C 退出");
});
