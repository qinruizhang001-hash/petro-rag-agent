import { parseBody } from "../../lib/reportBody";
import type { Block, Segment } from "../../lib/reportBody";
import type { ReportCitation } from "../../lib/approval";
import { cn } from "../../lib/utils";

/**
 * 报告正文。
 *
 * 渲染规则在 `lib/reportBody.ts`（纯函数，可测）；这里只管排版。
 *
 * ## 引用标记为什么不做成可点
 *
 * 正文里的 `[alarm:A-20261027T1000-capture_rate]` 指向一条**告警判定**，
 * 不是一篇文献切片。界面上没有"告警详情页"这个落点，做成可点会让读者
 * 点一下什么也不发生 —— 那比不能点更像故障。
 *
 * 所以它是一个悬停出处的徽标：虚线边框、`cursor-help`、`title` 里写明
 * "这条引用的出处是哪条记录"。形状与知识库里那些可点的实线角标不同，
 * 一眼能分出"能跳"和"只能看"。
 */
export function ReportBody({
  body,
  citations,
}: {
  body: string;
  citations: Record<string, ReportCitation>;
}) {
  const blocks = parseBody(body);
  return (
    <div className="space-y-1">
      {blocks.map((b, i) => (
        <BlockView key={i} block={b} citations={citations} />
      ))}
    </div>
  );
}

const HEADING: Record<"h1" | "h2" | "h3", string> = {
  h1: "mt-2 text-xs font-medium text-ink-0 first:mt-0",
  h2: "mt-2 text-2xs font-medium text-acc first:mt-0",
  h3: "mt-1 text-2xs text-ink-1",
};

function BlockView({
  block,
  citations,
}: {
  block: Block;
  citations: Record<string, ReportCitation>;
}) {
  const inner = <Segments segments={block.segments} citations={citations} />;
  switch (block.kind) {
    case "h1":
    case "h2":
    case "h3":
      return <p className={HEADING[block.kind]}>{inner}</p>;
    case "quote":
      return (
        <p className="border-l border-line pl-2 text-2xs leading-relaxed text-ink-2">
          {inner}
        </p>
      );
    case "li":
      return (
        <p className="flex gap-1 text-2xs leading-relaxed text-ink-1">
          <span aria-hidden className="shrink-0 text-ink-2">
            ·
          </span>
          <span className="min-w-0">{inner}</span>
        </p>
      );
    case "li2":
      return (
        <p className="flex gap-1 pl-3 text-2xs leading-relaxed text-ink-2">
          <span aria-hidden className="shrink-0">
            –
          </span>
          <span className="min-w-0">{inner}</span>
        </p>
      );
    default:
      return <p className="text-2xs leading-relaxed text-ink-1">{inner}</p>;
  }
}

function Segments({
  segments,
  citations,
}: {
  segments: Segment[];
  citations: Record<string, ReportCitation>;
}) {
  return (
    <>
      {segments.map((s, i) => {
        if (s.kind === "bold") {
          return (
            <strong key={i} className="font-medium text-ink-0">
              {s.text}
            </strong>
          );
        }
        if (s.kind === "cite") {
          const c = citations[s.token];
          return (
            <span
              key={i}
              title={
                c
                  ? `${s.token}\n${c.kind} · ${c.ref}\n${c.detail}`
                  : `${s.token}\n（后端没有为这个标记提供明细）`
              }
              className={cn(
                "mx-0.5 inline-block cursor-help rounded-panel border border-dashed px-1",
                "border-acc/40 align-baseline text-[0.85em] text-acc/90",
              )}
            >
              {s.refKind === "summary" ? "汇总" : s.ref}
            </span>
          );
        }
        return <span key={i}>{s.text}</span>;
      })}
    </>
  );
}
