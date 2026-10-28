import { useEffect, useState } from "react";
import { fetchKbGaps } from "../../lib/kb";
import { queueSummary, recordLine } from "../../lib/gap";
import type { KbGapsPayload } from "../../lib/kb-types";

/**
 * 已登记的缺口队列（最近若干条）。
 *
 * ## 为什么和表单分两个文件
 *
 * 表单是 `React.lazy` 的（只在点了「登记」之后才需要），而队列是常驻显示。
 * 放在同一个文件里，那个静态 import 会把表单一起拉回首屏 ——
 * `vite build` 会明确警告"dynamically imported but also statically imported"，
 * 但**产物照样生成**，首屏一个字节都没少。拆开才是真的按需。
 *
 * ## 它存在的理由
 *
 * 只有一个：**让提交者看见它落地了**。一个只显示"提交成功"的表单，
 * 用户没法区分"记下了"和"按钮点了但没反应"。
 *
 * 队列放在 `history/kb_gaps.jsonl`（不进仓库），所以演示模式读不到 ——
 * 那时显示原因，不显示"0 条"（判定在 `lib/gap.ts::queueSummary`）。
 */
export function KbGapQueue({ refreshKey = 0 }: { refreshKey?: number }) {
  const [data, setData] = useState<KbGapsPayload | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    fetchKbGaps(20)
      .then((d) => {
        if (alive) setData(d);
      })
      .catch((e: unknown) => {
        if (alive) setError(e instanceof Error ? e.message : String(e));
      });
    return () => {
      alive = false;
    };
  }, [refreshKey]);

  if (error) return <p className="mt-1 text-ink-2">队列读取失败：{error}</p>;
  if (!data) return null;

  const summary = queueSummary(data);
  return (
    <div className="mt-1">
      <div className="text-ink-1">{summary.headline}</div>
      {summary.detail && <p className="text-ink-2">{summary.detail}</p>}
      {data.records.length > 0 && (
        <ul className="mt-0.5 space-y-px text-ink-2">
          {data.records.slice(0, 3).map((r) => (
            <li key={r.id} className="truncate" title={r.note || r.query || r.target}>
              · {recordLine(r)}
            </li>
          ))}
        </ul>
      )}
      {data.next_step && <p className="mt-0.5 text-ink-2">{data.next_step}</p>}
    </div>
  );
}
