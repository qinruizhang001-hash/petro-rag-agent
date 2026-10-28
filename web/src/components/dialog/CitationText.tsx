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
 * ## 角标是可点按钮，不是 `<sup>`
 *
 * 现场有人用触摸屏、也有人戴手套。一个 9 px 的上标点不中，
 * 所以这里做成带边框的小按钮（至少 16 px 高），并给出 `title`。
 */
export function CitationText({
  text,
  citations,
  onPick,
}: {
  text: string;
  citations: Citation[];
  onPick: (c: Citation) => void;
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
            {resolved.map((c) => (
              <button
                key={`${i}-${c!.reference_id}`}
                type="button"
                onClick={() => onPick(c!)}
                title={`${c!.source || "无出处"} · 点开看这一片原文`}
                className="mx-0.5 inline-flex translate-y-[-1px] items-center rounded-panel border border-acc/50 px-1 font-num text-acc hover:bg-acc/10"
              >
                {c!.reference_id}
              </button>
            ))}
          </Fragment>
        );
      })}
    </>
  );
}
