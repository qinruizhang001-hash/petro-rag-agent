import { useState } from "react";
import { submitKbGap } from "../../lib/kb";
import { GAP_KIND_LABEL, gapFormBlocked } from "../../lib/gap";
import type { KbGapKind, KbGapRecord } from "../../lib/kb-types";
import { cn } from "../../lib/utils";

/**
 * 缺口反馈：把"这个知识库答不了我"变成一条**有对象的**记录。
 *
 * ## 这个组件要防的三件事
 *
 * 1. **假装提交成功**。fixture 模式没有后端，按钮在用户动手之前就是禁用的，
 *    并且旁边写着为什么。判定在 `lib/gap.ts`（纯函数，被 `test-gap.mjs` 锁着）。
 * 2. **自己编一句"稍后自动处理"**。`next_step` 由后端给
 *    （`api_kb.GAP_NEXT_STEP`），因为那是一个承诺：入库要占满一张显卡、
 *    建图要几十分钟，不能由一次点击触发。承诺分散在几个组件里时，
 *    总有一份会说成"将自动重建索引"。
 * 3. **报"0 条"代替"读不到"**。见 `queueSummary`。
 */

export function KbGapForm({
  kind,
  target,
  query,
  onClose,
}: {
  kind: KbGapKind;
  target?: string;
  query?: string;
  onClose?: () => void;
}) {
  const [note, setNote] = useState("");
  const [busy, setBusy] = useState(false);
  const [saved, setSaved] = useState<KbGapRecord | null>(null);
  const [error, setError] = useState<string | null>(null);
  const blocked = gapFormBlocked();

  async function submit() {
    if (blocked || busy) return;
    setBusy(true);
    setError(null);
    try {
      const record = await submitKbGap({ kind, target, query, note });
      setSaved(record);
      setNote("");
    } catch (e: unknown) {
      // 后端的 400 文案能直接给人看（"必须说是哪一篇"），原样显示 ——
      // 换成"提交失败"用户只能重试，而他需要的是改。
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="mt-1.5 rounded-panel border border-line bg-surface-2 px-2 py-1.5">
      <div className="flex items-baseline justify-between gap-2">
        <span className="text-ink-1">登记：{GAP_KIND_LABEL[kind]}</span>
        {onClose && (
          <button
            type="button"
            onClick={onClose}
            className="text-ink-2 hover:text-ink-0"
          >
            收起
          </button>
        )}
      </div>

      {/* 要说清"这一条是关于什么的"：用户填的是补充说明，对象由入口带进来 */}
      <dl className="mt-1 space-y-px text-ink-2">
        {target && (
          <div className="flex gap-1">
            <dt className="shrink-0">对象</dt>
            <dd className="truncate font-num" title={target}>
              {target}
            </dd>
          </div>
        )}
        {query && (
          <div className="flex gap-1">
            <dt className="shrink-0">问题</dt>
            <dd className="truncate" title={query}>
              {query}
            </dd>
          </div>
        )}
      </dl>

      <textarea
        value={note}
        onChange={(e) => setNote(e.target.value)}
        maxLength={2000}
        rows={2}
        placeholder="补充说明（可选）—— 写清你期待它回答什么，比只写「不对」有用"
        disabled={!!blocked || busy}
        className={cn(
          "mt-1 w-full resize-y rounded border border-line bg-surface-1 px-1.5 py-1",
          "text-ink-0 placeholder:text-ink-3 focus:border-ink-2 focus:outline-none",
          "disabled:opacity-60",
        )}
      />

      <div className="mt-1 flex items-center gap-2">
        <button
          type="button"
          onClick={submit}
          disabled={!!blocked || busy}
          className={cn(
            "rounded border border-line px-2 py-0.5",
            blocked
              ? "cursor-not-allowed text-ink-3"
              : "text-ink-0 hover:border-ink-2",
          )}
        >
          {busy ? "提交中…" : "登记这条缺口"}
        </button>
        {saved && (
          <span className="text-ink-2">
            已记录 <span className="font-num">{saved.id}</span>
          </span>
        )}
      </div>

      {blocked && <p className="mt-1 text-sev-notice">{blocked}</p>}
      {error && <p className="mt-1 text-sev-critical">{error}</p>}
      {saved?.next_step && (
        <p className="mt-1 border-t border-line pt-1 text-ink-2">
          {saved.next_step}
        </p>
      )}
    </div>
  );
}

/**
 * 已登记的缺口队列（最近若干条）。
 *
 * 它存在的理由只有一个：**让提交者看见它落地了**。一个只显示"提交成功"
 * 的表单，用户没法区分"记下了"和"按钮点了但没反应"。
 *
 * 队列放在 `history/kb_gaps.jsonl`（不进仓库），所以演示模式读不到 ——
 * 那时显示原因，不显示"0 条"。
 */
