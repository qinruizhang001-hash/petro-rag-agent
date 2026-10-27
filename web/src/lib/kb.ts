import type {
  KbDocDetail,
  KbDocuments,
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
