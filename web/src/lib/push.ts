/**
 * 实时推送通道（WebSocket）。
 *
 * ## 为什么现在就要有
 *
 * 现在这套界面是**拉取**的：打开页面拉一次告警，然后就不动了。真实运维
 * 场景不是这样 —— 告警是推过来的，操作员不该靠刷新页面才知道井口压力
 * 超限。
 *
 * 但真接口还不存在。所以这里只做**通道**，不做业务：一个带重连的
 * WebSocket 客户端、一个消息类型联合、一个订阅函数。组件拿到的是
 * `AlarmReport`，它不知道这份数据来自 fixture、HTTP 还是 WebSocket。
 *
 * ## 为什么不装 socket.io
 *
 * 浏览器的 `WebSocket` 已经够用，而 socket.io 需要在后端也装一套
 * （协议不兼容裸 WebSocket），等于给未来的后端选型提前上锁。这个项目
 * 的后端是 FastAPI，`websockets` 一行就能起，没有理由引一个客户端协议。
 *
 * ## 未配置时是什么行为
 *
 * `pushUrl()` 返回 `null` → 不建立任何连接、不重试、不打日志。
 * **不是"假装连上"**：一个静默重试的假连接会让人以为推送通道是通的，
 * 而真实原因只是没配 URL。
 */

import type { AlarmReport, ThresholdStatus } from "./types";

/** 服务端 → 客户端。`type` 是判别式，新增消息类型时联合类型会强制处理。 */
export type PushMessage =
  | { type: "alarm"; payload: AlarmReport }
  | { type: "thresholds"; payload: ThresholdStatus }
  /** 心跳。服务端定期发，用来让中间的代理不要掐掉空闲连接。 */
  | { type: "ping" };

export type PushStatus = "idle" | "connecting" | "open" | "closed";

export interface PushClient {
  /** 当前连接状态。用于界面上的一个角标 —— 推送断了必须看得见。 */
  readonly status: PushStatus;
  /** 订阅消息。返回退订函数。 */
  subscribe(onMessage: (m: PushMessage) => void): () => void;
  /** 订阅状态变化。 */
  onStatus(fn: (s: PushStatus) => void): () => void;
  close(): void;
}

/**
 * 推送地址。未配置时返回 `null`。
 *
 * 单独一个函数而不是模块级常量：`import.meta.env` 在测试环境里可能不存在，
 * 模块顶层求值会在 import 阶段就抛错，而那时错误信息指向的是 import 语句。
 */
export function pushUrl(): string | null {
  const raw = import.meta.env.VITE_OPS_WS;
  return typeof raw === "string" && raw.trim() ? raw.trim() : null;
}

/** 重连退避：1s → 2s → 4s → 8s → 封顶 15s，带 ±20% 抖动。 */
function backoff(attempt: number): number {
  const base = Math.min(15000, 1000 * 2 ** Math.min(attempt, 4));
  // 抖动是必须的：一块大屏和十台工控机同时断线、同时重连，
  // 会在服务端恢复的瞬间形成一次自制的 DDoS。
  return Math.round(base * (0.8 + Math.random() * 0.4));
}

/**
 * 建立推送通道。
 *
 * 返回的对象在 `close()` 之前会**一直尝试重连** —— 这块屏挂在墙上，
 * 网络闪断一次就永久失联是不可接受的。但每次重连都会把状态推到界面，
 * 所以"一直在重连"是看得见的，不是静默的。
 */
export function createPushClient(url: string): PushClient {
  let socket: WebSocket | null = null;
  let attempt = 0;
  let closed = false;
  let timer: ReturnType<typeof setTimeout> | null = null;

  const handlers = new Set<(m: PushMessage) => void>();
  const statusHandlers = new Set<(s: PushStatus) => void>();
  let status: PushStatus = "idle";

  const setStatus = (s: PushStatus) => {
    if (status === s) return;
    status = s;
    for (const fn of statusHandlers) fn(s);
  };

  const connect = () => {
    if (closed) return;
    setStatus("connecting");

    let ws: WebSocket;
    try {
      ws = new WebSocket(url);
    } catch {
      // 地址不合法（例如 `http://` 而不是 `ws://`）。重试没有意义，
      // 但也不能静默 —— 停在 closed 让界面显示出来。
      setStatus("closed");
      return;
    }
    socket = ws;

    ws.addEventListener("open", () => {
      attempt = 0;
      setStatus("open");
    });

    ws.addEventListener("message", (ev) => {
      // 一条坏消息不能让整个通道失效：解析失败就丢掉这一条。
      // 把 JSON.parse 放进 try 里，是因为服务端升级协议时会先发出
      // 一条人读的字符串（"connected"），而那条不是 JSON。
      let msg: PushMessage;
      try {
        msg = JSON.parse(String(ev.data)) as PushMessage;
      } catch {
        return;
      }
      if (!msg || typeof msg.type !== "string") return;
      for (const fn of handlers) fn(msg);
    });

    ws.addEventListener("close", () => {
      if (closed) return;
      setStatus("closed");
      const delay = backoff(attempt++);
      timer = setTimeout(connect, delay);
    });

    // `error` 之后一定会跟一个 `close`，所以这里不重复安排重连 ——
    // 两处都排会得到两条并行重连链，每断一次翻一倍。
    ws.addEventListener("error", () => {
      /* 交给 close 处理 */
    });
  };

  connect();

  return {
    get status() {
      return status;
    },
    subscribe(onMessage) {
      handlers.add(onMessage);
      return () => handlers.delete(onMessage);
    },
    onStatus(fn) {
      statusHandlers.add(fn);
      fn(status);
      return () => statusHandlers.delete(fn);
    },
    close() {
      closed = true;
      if (timer) clearTimeout(timer);
      timer = null;
      // 先摘监听再关：否则 `close` 事件会把状态改成 closed 并安排重连，
      // 而 `closed` 标志此时已经是 true —— 顺序写反就会出现一条
      // 永不退出的定时器链，页面切走后还在跑。
      socket?.close();
      socket = null;
      setStatus("idle");
    },
  };
}
