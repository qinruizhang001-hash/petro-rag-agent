import type {
  KbDocDetail,
  KbDocuments,
  KbGapInput,
  KbGapRecord,
  KbGapsPayload,
  KbOverview,
  KbSearchResult,
} from "./kb-types";

import documentsFixture from "../fixtures/documents.json";
import overviewFixture from "../fixtures/overview.json";

/**
 * `/api/kb/*` 客户端。
 *
 * 两种数据源与 `api.ts` 一致（`VITE_KB_SOURCE` / `VITE_OPS_SOURCE`，
 * 默认 `fixture`）：fixture 是 `scripts/capture_kb_fixtures.py` 从**真实接口**
 * 抓下来的，不是手写的。
 *
 * 为什么检索 fixture 要动态 import
 * --------------------------------
 * `search.json` 有 97 KB（12 个切片 + 27 个实体 + 25 条关系的原文）。
 * 静态 import 会把它塞进首屏 chunk，而首屏 chunk 有一条 CI 守卫
 * （`ci.yml`：main < 320000 字节，防 three 泄漏进来）。更实际的理由是：
 * 用户打开控制台时并不会立刻检索，为一个还没用到的面板付 97 KB 不划算。
 *
 * 动态 import 后它变成一个独立 chunk，点"检索"时才拉。
 */

const BASE = import.meta.env.VITE_KB_BASE ?? "/api";
const SOURCE = import.meta.env.VITE_KB_SOURCE ?? import.meta.env.VITE_OPS_SOURCE ?? "fixture";

export const kbDataSource = SOURCE;

/** fixture 模式下检索结果对应的固定问题（见 capture 脚本的 SEARCH_QUERY）。 */
export const FIXTURE_QUERY = "盖层完整性对 CO2 泄漏风险的影响";

async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) {
    const raw = await res.text();
    let detail: unknown = raw;
    try {
      detail = JSON.parse(raw);
    } catch {
      /* 保持原文 */
    }
    throw new Error(`${res.status} ${path}: ${JSON.stringify(detail)}`);
  }
  return (await res.json()) as T;
}

export async function fetchKbOverview(): Promise<KbOverview> {
  if (SOURCE === "fixture") return overviewFixture as unknown as KbOverview;
  return getJson<KbOverview>("/kb/overview");
}

export async function fetchKbDocuments(): Promise<KbDocuments> {
  if (SOURCE === "fixture") return documentsFixture as unknown as KbDocuments;
  return getJson<KbDocuments>("/kb/documents");
}

/**
 * 单篇文档的切片原文。
 *
 * fixture 模式下**没有**这个端点 —— 12 篇论文的全文切片加起来是 MB 级，
 * 不值得为了演示存一份。所以这里直接抛一个说得清的错误，让界面显示
 * "演示数据不含原文，接上后端后可查"，而不是假装成功再渲染一片空白。
 */
export async function fetchKbDocument(
  name: string,
  focus?: string,
): Promise<KbDocDetail> {
  if (SOURCE === "fixture") {
    throw new Error(
      "演示数据不含切片原文（体积太大）。用 VITE_KB_SOURCE=live 接后端后可用。",
    );
  }
  const q = focus ? `?focus=${encodeURIComponent(focus)}` : "";
  return getJson<KbDocDetail>(`/kb/documents/${encodeURIComponent(name)}${q}`);
}

/**
 * 检索知识库。
 *
 * fixture 模式下**忽略入参**：结果固定是抓取时那个问题的真实响应。
 * 这个不对称是刻意的 —— 假装能用 fixture 重算只会让人以为改个词
 * 就能看到不同结果，而它不会。界面上会显示一条提示说明这件事。
 */
export async function searchKb(
  query: string,
  mode: "hybrid" | "local" | "global" | "naive" | "mix" = "hybrid",
): Promise<KbSearchResult> {
  if (SOURCE === "fixture") {
    const mod = (await import("../fixtures/search.json")) as unknown as {
      default: KbSearchResult;
    };
    return mod.default;
  }
  const res = await fetch(`${BASE}/kb/search`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, mode }),
  });
  if (!res.ok) {
    // 400（空 query）与 422（模式名错）是真错误；`unavailable` 不是 ——
    // 后端对它返回 200，好让界面能显示"检索没跑起来"而不是一个通用报错。
    throw new Error(`${res.status} /kb/search`);
  }
  return (await res.json()) as KbSearchResult;
}

/**
 * 缺口反馈：**唯一会写盘的知识库端点**，也是唯一"只读 → 可记录"的入口。
 *
 * ## fixture 模式下为什么直接拒绝，而不是存在 localStorage
 *
 * 审批轨迹（`draftStore.ts`）存本机是对的：那是**你自己的**演练，刷新不丢
 * 就有意义。缺口反馈不一样 —— 它的全部价值在于**送到维护语料的人手上**。
 * 存在浏览器里等于没写，而且提交者会以为已经提过了，于是不再提。
 *
 * 所以这里抛错，界面把这句话显示出来，并且**在用户动手之前**就把按钮
 * 标成不可用（见 `KbGapForm`）。让他打完一百字再告诉他"这个不会发出去"
 * 是同一类欺骗的另一种形式。
 */
export async function submitKbGap(input: KbGapInput): Promise<KbGapRecord> {
  if (SOURCE === "fixture") {
    throw new Error(
      "演示模式不落盘：缺口反馈要送到维护语料的人手上，存在浏览器里等于没写。",
    );
  }
  const res = await fetch(`${BASE}/kb/gaps`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input),
  });
  if (!res.ok) {
    // 后端的 400 文案是**能直接给人看**的（"必须说是哪一篇"），
    // 不要换成"提交失败"。用户能据此改，而"提交失败"只能让他重试。
    let detail = `${res.status}`;
    try {
      const body = (await res.json()) as { detail?: unknown };
      if (typeof body.detail === "string") detail = body.detail;
    } catch {
      /* 保留状态码 */
    }
    throw new Error(detail);
  }
  return (await res.json()) as KbGapRecord;
}

/**
 * 缺口队列（最近若干条，新的在前）。
 *
 * fixture 模式下返回 `available: false` 而不是"0 条" —— 队列在后端，
 * 演示模式根本读不到它。显示成"0 条"会让读者以为没人反馈过。
 */
export async function fetchKbGaps(limit = 20): Promise<KbGapsPayload> {
  if (SOURCE === "fixture") {
    return {
      available: false,
      reason: "演示模式读不到队列：它写在维护者那台机器的 history/ 下。",
      total: 0,
      unreadable_lines: 0,
      by_kind: {},
      records: [],
    };
  }
  return getJson<KbGapsPayload>(`/kb/gaps?limit=${limit}`);
}
