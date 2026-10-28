import type { Citation } from "../../lib/chat";
import { readable } from "../../lib/text";
import { StrengthChip } from "../kb/StrengthChip";

/**
 * 引用落点：一片原文。
 *
 * ## 为什么是"就地展开"而不是跳到知识库页
 *
 * 点角标的期望是"看到这句话的出处"，不是"离开当前对话去另一个页面"。
 * 右栏只有一页宽，跳走之后返回要重新定位到刚才那条消息 —— 那条路径
 * 每多一步，读者就会少核对一次。
 *
 * ## 为什么这里不调 `/kb/documents/{name}`
 *
 * 那片原文已经在检索响应里了（`citations[].content`），直接显示即可。
 * 好处有两个：fixture 模式下也能用（快照不含全文切片，但含这 12 片）；
 * 而且不依赖后端在线 —— 一条已经拿到的证据不该因为后端掉线而打不开。
 *
 * "打开整篇原文"仍然是可选动作，它需要后端，所以单独一个按钮，
 * 失败时把原因写在按钮下面，而不是让整块面板变成错误页。
 */
export function ChunkPeek({
  citation,
  siblings = 1,
  onBack,
  onOpenDoc,
}: {
  citation: Citation;
  /** 同一来源的切片数。对话里的证据列表按来源合并，所以这里要说明"这是第几片" */
  siblings?: number;
  onBack: () => void;
  onOpenDoc: (name: string, focus?: string) => void;
}) {
  return (
    <div className="space-y-2 text-2xs">
      <div className="flex items-baseline gap-1.5">
        <button type="button" onClick={onBack} className="text-acc hover:underline">
          ← 返回对话
        </button>
        <span className="ml-auto font-num text-ink-2">[{citation.reference_id}]</span>
      </div>

      <div className="rounded-panel border border-line px-2 py-1.5">
        <div className="flex items-baseline gap-1.5">
          <span className="min-w-0 flex-1 truncate text-ink-0" title={citation.source || "无出处"}>
            {citation.source || "无出处"}
          </span>
          <StrengthChip source={citation.source} />
        </div>
        <p className="mt-0.5 font-num text-ink-2">
          {citation.chars} 字
          {citation.formulaHeavy && (
            <span className="ml-2 text-sev-notice">公式为主，不宜作为机理证据</span>
          )}
        </p>
        <p className="mt-1 whitespace-pre-wrap break-words leading-relaxed text-ink-1">
          {readable(citation.content)}
        </p>
      </div>

      {/*
        角标是**响应内编号**：换一个问题就重新排。所以长期指向只能靠
        文件名 + chunk_id。这句话必须出现在这里 —— 角标长得就像一个
        稳定的引用编号，而它连下一次检索都对不上。
      */}
      <p className="text-ink-2">
        [{citation.reference_id}] 是本次检索内的来源序号，不是文献编号。
        {siblings > 1 && `该来源共 ${siblings} 片，这里显示第 1 片；完整列表在知识库 → 检索里。`}
      </p>

      {citation.traceable ? (
        <button
          type="button"
          onClick={() => onOpenDoc(citation.source, citation.chunk_id)}
          className="rounded-panel border border-acc/40 px-2 py-1 text-acc hover:bg-acc/10"
        >
          在知识库中打开整篇（定位到这一片）
        </button>
      ) : (
        <p className="text-sev-notice">
          这条证据没有来源文件名，打不开原文 —— 它能作为线索，不能当引用。
        </p>
      )}
    </div>
  );
}
