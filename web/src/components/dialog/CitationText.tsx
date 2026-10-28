import { Fragment } from "react";
import type { Citation } from "../../lib/chat";
import { splitCitations } from "../../lib/citations";

/**
 * 正文 + 可点角标。
 *
 * ## 只有"整段都能解析"才变成角标
 *
 * `[1,2]` 既可能是两个引用，也可能是"区间 1–2"。所以一段标记里的
 * 编号**全部**能在证据里找到时才渲染成角标；否则原样输出。
 * 反过来（解析不到就染红）会在数学符号上造出一串假警报，
 * 而假警报的代价是真警报被忽略。
 *
 * ## 两种角标必须**形状**不同，不能只靠颜色
 *
 * 可点的：实线边框 + 末尾一个 `↗`（表示"会跳走"）+ 悬浮时背景变亮。
 * 无血缘的：虚线边框 + 没有箭头 + `cursor-not-allowed`。
 *
 * 只改颜色是不够的 —— 色觉障碍、强光下的投影、以及灰度打印都会让
 * "实线蓝"和"灰蓝"变成同一个东西。形状是唯一不会丢的那一维。
 * 而"点了没反应"比"不能点"更像故障，所以不可点的那一个必须**看起来**
 * 就不可点。
 *
 * ## 角标是可点按钮，不是 `<sup>`
 *
 * 现场有人用触摸屏、也有人戴手套。一个 9 px 的上标点不中，
 * 所以这里做成带边框的小按钮（至少 16 px 高），并给出 `title`。
 */
export function CitationText({
  text,
  citations,
  onPick,
  hint = "点开看这一片原文",
}: {
  text: string;
  citations: Citation[];
  onPick: (c: Citation) => void;
  /** 按钮的 tooltip 后半句。正文角标跳详情，列表按钮就地展开，两者要说清 */
  hint?: string;
}) {
  const byId = new Map(citations.map((c) => [c.reference_id, c]));
  const segments = splitCitations(text);

  return (
    <>
      {segments.map((seg, i) => {
        if (seg.kind === "text") return <Fragment key={i}>{seg.text}</Fragment>;

        const resolved = seg.ids.map((id) => byId.get(id));
        if (resolved.some((c) => c === undefined)) {
          // 有编号对不上证据：原样输出，不改写正文（见文件头）
          return <Fragment key={i}>{seg.raw}</Fragment>;
        }

        return (
          <Fragment key={i}>
            {resolved.map((c) => {
              if (!c!.source) {
                return (
                  <span
                    key={`${i}-${c!.reference_id}`}
                    role="img"
                    aria-label={`引用 ${c!.reference_id} 没有文件级血缘，无法打开`}
                    title="该证据没有文件级血缘，无法定位到具体文献（Chroma 向量召回不带出处），所以点不开"
                    className="mx-0.5 inline-flex translate-y-[-1px] cursor-not-allowed items-center rounded-panel border border-dashed border-line-strong px-1 font-num text-ink-2"
                  >
                    {c!.reference_id}
                  </span>
                );
              }
              return (
                <button
                  key={`${i}-${c!.reference_id}`}
                  type="button"
                  onClick={() => onPick(c!)}
                  title={`${c!.source} · ${hint}`}
                  aria-label={`引用 ${c!.reference_id}：${c!.source}，${hint}`}
                  className="mx-0.5 inline-flex translate-y-[-1px] items-center gap-0.5 rounded-panel border border-acc/60 px-1 font-num text-acc hover:bg-acc/10"
                >
                  {c!.reference_id}
                  {/* `↗` 是"会跳走"的唯一非颜色信号 —— 见文件头 */}
                  <span aria-hidden className="text-[0.9em] leading-none">
                    ↗
                  </span>
                </button>
              );
            })}
          </Fragment>
        );
      })}
    </>
  );
}
