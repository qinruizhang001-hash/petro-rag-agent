import { useMemo } from "react";
import type { KbDocDetail } from "../../lib/kb-types";
import { readable } from "../../lib/text";
import { cn } from "../../lib/utils";

/**
 * 文档详情 —— 引用跳转的落点。
 *
 * ## 为什么要把命中的那一片高亮
 *
 * 点一个引用角标，期望是"看到这句原文的上下文"，而不是"打开这篇文献的
 * 第一页"。`focus` 参数由后端换算成 `chunks[].chunk_id`，这里只负责
 * 找出下标并加边框。找不到时（切片被重新切过、offset 变了）不高亮，
 * 也不报错 —— 展示全文仍然是正确行为。
 */
export function KbDocDetail({
  name,
  detail,
  error,
  partialNote,
  onBack,
}: {
  name: string;
  detail: KbDocDetail | null;
  error: string | null;
  /**
   * "只含这一片"的说明。
   *
   * 有它的时候**不显示"共 N 片"** —— 演示数据里根本没取到那个数字，
   * 而一个显示成"共 1 片"的详情页，读起来像"这篇文献只有一片"。
   * 缺数据和数据是 1 是两件事，不能靠把 0 填进去来省一个分支。
   */
  partialNote?: string | null;
  onBack: () => void;
}) {
  const focusedIndex = useMemo(
    () => detail?.chunks.findIndex((c) => c.chunk_id === detail.focused) ?? -1,
    [detail],
  );

  return (
    <div className="space-y-2 text-2xs">
      <div className="flex items-baseline gap-1">
        <button type="button" onClick={onBack} className="text-acc hover:underline">
          ← 返回
        </button>
        <span className="min-w-0 flex-1 truncate text-ink-0" title={name}>
          {name}
        </span>
      </div>

      {error && (
        <p className="rounded-panel border border-line px-2 py-1.5 text-ink-1">{error}</p>
      )}

      {detail && (
        <>
          {detail.title && (
            <p className="text-ink-1" title={detail.title}>
              {detail.title}
            </p>
          )}
          {partialNote ? (
            <p className="rounded-panel border border-sev-notice/50 bg-sev-notice/5 px-2 py-1 leading-relaxed text-sev-notice">
              {partialNote}
            </p>
          ) : (
            <p className="text-ink-2">
              共 <span className="font-num text-ink-1">{detail.total_chunks}</span> 片
              {detail.has_more &&
                `，显示 ${detail.offset + 1}–${detail.offset + detail.chunks.length}`}
            </p>
          )}
          <div className="space-y-1">
            {detail.chunks.map((c, i) => (
              <div
                key={c.chunk_id}
                className={cn(
                  "rounded-panel border px-1.5 py-1",
                  i === focusedIndex ? "border-acc/60 bg-acc/5" : "border-line",
                )}
              >
                <div className="flex items-baseline justify-between text-ink-2">
                  <span className="font-num">#{c.index}</span>
                  <span className="font-num">{c.chars} 字</span>
                </div>
                <p className="mt-0.5 whitespace-pre-wrap break-words text-ink-1">
                  {readable(c.content)}
                </p>
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  );
}
