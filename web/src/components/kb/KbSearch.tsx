import { useEffect, useState } from "react";
import { FIXTURE_QUERY, fetchKbDocument, kbDataSource, searchKb } from "../../lib/kb";
import type { FocusDoc, KbDocDetail, KbSearchResult } from "../../lib/kb-types";
import { KbDocDetail as DocDetailView } from "./KbDocDetail";
import { KbResultView } from "./KbResultView";

/**
 * 检索页：输入 + 结果 + 原文详情。
 *
 * 三个状态都在这一层，因为它们是同一条交互链：检索 → 点引用 → 看原文 →
 * 返回。拆开之后"返回"要跨组件传一个回调，而那个回调除了重置两个
 * `useState` 之外什么也不做。
 *
 * ## `focusDoc` 是外部推进来的跳转请求
 *
 * 对话里的引用角标点开时，`RightRail` 把"要看哪篇的哪一片"塞进来，
 * 这一页负责真正去取。取不到全文时**不报错页**，而是用调用方带过来的
 * 那一片原文落脚（见 `FocusDoc.fallback`）—— 跳转这件事本身应当成立，
 * 缺的只是上下文。
 */
export function KbSearch({ focusDoc }: { focusDoc?: FocusDoc | null }) {
  const [query, setQuery] = useState(() => {
    if (typeof window === "undefined") return "";
    return new URLSearchParams(window.location.search).get("q") ?? "";
  });
  const [result, setResult] = useState<KbSearchResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [detail, setDetail] = useState<KbDocDetail | null>(null);
  const [detailError, setDetailError] = useState<string | null>(null);
  const [partialNote, setPartialNote] = useState<string | null>(null);

  const openDoc = (name: string, focus?: string, fallback?: FocusDoc["fallback"]) => {
    setDetailError(null);
    setDetail(null);
    setPartialNote(null);
    fetchKbDocument(name, focus)
      .then(setDetail)
      .catch((e: unknown) => {
        const msg = e instanceof Error ? e.message : String(e);
        if (fallback) {
          /*
           * 全文取不到，但这一片是真的。`total_chunks` 填 0 只是为了让类型
           * 成立 —— 界面上不会显示它（有 `partialNote` 时那个分支不走）。
           * 绝不能把它渲染成"共 1 片"。
           */
          setDetail({
            name,
            title: "",
            url: "",
            status: "partial",
            total_chunks: 0,
            offset: 0,
            focused: fallback.chunk_id,
            has_more: false,
            chunks: [
              {
                chunk_id: fallback.chunk_id,
                index: 0,
                chars: fallback.chars,
                content: fallback.content,
              },
            ],
          });
          setPartialNote(
            "演示数据不含整篇原文，这里只有检索命中的那一片。完整切片需要 VITE_KB_SOURCE=live 接后端。",
          );
        } else {
          setDetailError(msg);
        }
      });
  };

  /*
   * 外部跳转请求。依赖整个对象而不是 `.name`：同一篇文献的两个不同切片
   * 是两次不同的跳转，只看名字会第二次点不动。
   */
  useEffect(() => {
    if (!focusDoc?.name) return;
    openDoc(focusDoc.name, focusDoc.focus, focusDoc.fallback);
    // `openDoc` 每次渲染都是新函数，进依赖会变成死循环。
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [focusDoc]);

  const submit = async () => {
    const q = query.trim();
    if (!q || busy) return;
    setBusy(true);
    setError(null);
    try {
      setResult(await searchKb(q));
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  /*
   * `?q=` 自动提交，**只在 fixture 模式下生效**。
   *
   * 这个参数的用途是截图与评审（配合 `?kb=search`），不是功能。live 模式下
   * 打开页面就自动跑一次 21 秒的检索，会让"我刚点进来怎么卡住了"变成
   * 一个查不出原因的问题 —— 而它其实是我自己写进 URL 的。
   */
  useEffect(() => {
    if (kbDataSource !== "fixture") return;
    if (new URLSearchParams(window.location.search).get("q")) void submit();
    // 只在挂载时跑一次：submit 每次渲染都是新函数，进依赖会变成死循环。
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  if (detail || detailError) {
    return (
      <DocDetailView
        name={detail?.name ?? focusDoc?.name ?? ""}
        detail={detail}
        error={detailError}
        partialNote={partialNote}
        onBack={() => {
          setDetail(null);
          setDetailError(null);
          setPartialNote(null);
        }}
      />
    );
  }

  return (
    <div className="space-y-2 text-2xs">
      <div className="flex gap-1">
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") void submit();
          }}
          placeholder="问知识库：盖层完整性、井筒腐蚀…"
          aria-label="知识库检索"
          className="min-w-0 flex-1 rounded-panel border border-line bg-surface-2 px-1.5 py-1 text-ink-0 outline-none placeholder:text-ink-2 focus:border-acc/60"
        />
        <button
          type="button"
          onClick={() => void submit()}
          disabled={busy || !query.trim()}
          className="shrink-0 rounded-panel border border-acc/40 px-2 py-1 text-acc disabled:border-line disabled:text-ink-2"
        >
          {busy ? "检索中" : "检索"}
        </button>
      </div>

      {/*
        fixture 模式下的检索结果是固定问题的真实响应，不随输入变化。
        不说清楚的话，用户会以为"我搜井筒腐蚀怎么出来盖层的"是 bug。
      */}
      {kbDataSource === "fixture" && (
        <p className="text-ink-2">
          演示数据：结果固定为「{FIXTURE_QUERY}」的真实检索，不随输入变化。
        </p>
      )}

      {error && <p className="text-sev-alert">请求失败：{error}</p>}

      {result && !busy && <KbResultView result={result} onOpenDoc={openDoc} />}
    </div>
  );
}
