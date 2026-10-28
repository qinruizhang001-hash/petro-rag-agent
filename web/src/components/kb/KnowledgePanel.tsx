import { useEffect, useState } from "react";
import { fetchKbDocuments, fetchKbOverview } from "../../lib/kb";
import type {
  FocusDoc,
  KbDocuments,
  KbOverview as KbOverviewData,
} from "../../lib/kb-types";
import { cn } from "../../lib/utils";
import { KbOverview } from "./KbOverview";
import { KbSearch } from "./KbSearch";

/**
 * 知识库面板 —— 外壳与数据加载。
 *
 * ## 为什么内部还分两页
 *
 * 右栏只有 320 px。概览（数字 + 缺口 + 24 行文档清单）和检索（输入 + 证据卡）
 * 叠在一页会有三屏高，而滚动条一出现，缺口那段就看不见了 ——
 * 恰好把最该被看见的东西推出了视口。
 *
 * ## 组件边界
 *
 * | 文件 | 职责 | 状态 |
 * |---|---|---|
 * | `KnowledgePanel.tsx` | 页签、概览数据加载 | `tab` / `overview` / `docs` |
 * | `KbOverview.tsx` | 规模、缺口、向量库、文档清单 | 无（纯展示） |
 * | `KbSearch.tsx` | 输入、提交、详情路由 | `query` / `result` / `detail` |
 * | `KbResultView.tsx` | 三种检索状态 | 无（纯展示） |
 * | `KbChunkCard.tsx` | 一条原文证据 | `expanded` |
 * | `KbDocDetail.tsx` | 引用跳转落点 | 无（纯展示） |
 * | `KbGapForm.tsx` | 缺口登记表单（`React.lazy`，点了才拉） | `note` / 提交态 |
 * | `KbGapQueue.tsx` | 已登记的缺口（证明它落地了） | `data` |
 *
 * 概览数据在**外壳**加载而不是 `KbOverview` 内部：它是"要不要显示缺口提示"
 * 的依据，而未来若要在页签上标一个缺口角标，那个数字必须在外壳手里。
 */
type Tab = "overview" | "search";

export function KnowledgePanel({ focusDoc }: { focusDoc?: FocusDoc | null }) {
  // `?kb=search` 供截图与评审直接定位到检索页。
  // 一个只能靠点击才能到达的状态，评审时总会有人截错图 —— 这条约定
  // 在 `?mode=` / `?tab=chain` / `?rail=dialog` 上已经用了四次。
  const [tab, setTab] = useState<Tab>(() => {
    if (typeof window === "undefined") return "overview";
    return new URLSearchParams(window.location.search).get("kb") === "search"
      ? "search"
      : "overview";
  });
  const [overview, setOverview] = useState<KbOverviewData | null>(null);
  const [docs, setDocs] = useState<KbDocuments | null>(null);
  const [error, setError] = useState<string | null>(null);

  /*
   * 从对话里的引用角标跳进来时，直接停在检索页。
   *
   * 停在概览会让这次跳转看起来"没反应"——读者点了一个角标，看到的是
   * 24 行文档清单，而他期待的是那片原文。
   */
  useEffect(() => {
    if (focusDoc?.name) setTab("search");
  }, [focusDoc]);

  useEffect(() => {
    let alive = true;
    Promise.all([fetchKbOverview(), fetchKbDocuments()])
      .then(([o, d]) => {
        if (!alive) return;
        setOverview(o);
        setDocs(d);
      })
      .catch((e: unknown) => {
        if (alive) setError(e instanceof Error ? e.message : String(e));
      });
    return () => {
      alive = false;
    };
  }, []);

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <TabBar tab={tab} onChange={setTab} />

      <div className="mt-2 min-h-0 flex-1 overflow-auto pr-1">
        {tab === "search" ? (
          <KbSearch focusDoc={focusDoc} />
        ) : error ? (
          <div className="p-3 text-2xs leading-relaxed text-sev-alert">
            知识库状态读取失败：{error}
          </div>
        ) : !overview || !docs ? (
          <div className="p-3 text-2xs text-ink-2">载入知识库状态…</div>
        ) : (
          <KbOverview overview={overview} docs={docs} />
        )}
      </div>
    </div>
  );
}

function TabBar({ tab, onChange }: { tab: Tab; onChange: (t: Tab) => void }) {
  return (
    <div className="flex rounded-panel border border-line p-0.5" role="tablist">
      {(
        [
          ["overview", "概览"],
          ["search", "检索"],
        ] as const
      ).map(([value, label]) => (
        <button
          key={value}
          type="button"
          role="tab"
          aria-selected={tab === value}
          onClick={() => onChange(value)}
          className={cn(
            "flex-1 rounded-panel px-1.5 py-0.5 text-2xs",
            tab === value ? "bg-acc/15 text-acc" : "text-ink-2 hover:text-ink-1",
          )}
        >
          {label}
        </button>
      ))}
    </div>
  );
}
