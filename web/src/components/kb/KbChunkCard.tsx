import { useState } from "react";
import type { KbChunkCard } from "../../lib/kb-types";
import { readable } from "../../lib/text";
import { cn } from "../../lib/utils";
import { RefBadge } from "./RefBadge";
import { StrengthChip } from "./StrengthChip";

/**
 * 一条原文证据。
 *
 * ## 一条证据一张卡，绝不合并
 *
 * 后端 `build_fragments` 会把两端实体的描述拼成一个字符串喂给 reranker ——
 * 那是对的，reranker 只看语义。但拼接之后"这句来自哪一篇"已经无从得知，
 * 所以证据路径上**不复用**那个函数（见 `kb_evidence.py` 的不变量注释）。
 * 如果这里改成合并展示，每条引用都会指向同一篇文献，而且不会报错。
 *
 * ## 为什么默认折叠
 *
 * 右栏 320 px，一片原文常常 1500 字。全展开的话，一次检索会把后续的
 * 图谱命中推到几屏之外 —— 而"哪些实体被命中了"往往是判断检索是否
 * 跑偏的第一眼依据。
 */
export function KbChunkCard({
  card,
  onOpen,
}: {
  card: KbChunkCard;
  onOpen: (name: string, focus?: string) => void;
}) {
  const [expanded, setExpanded] = useState(false);
  const long = card.chars > 220;
  const formulaHeavy = card.text_quality?.low ?? false;

  return (
    <div className="rounded-panel border border-line px-1.5 py-1">
      <div className="flex items-baseline gap-1">
        <RefBadge id={card.reference_id} />
        <button
          type="button"
          onClick={() => card.source && onOpen(card.source, card.chunk_id)}
          disabled={!card.source}
          className="min-w-0 flex-1 truncate text-left text-acc hover:underline disabled:text-ink-2"
          title="打开原文（定位到这一片）"
        >
          {card.source || "无出处"}
        </button>
        {/*
          血缘徽标：每条证据都要自己说清"能不能点回原文"。
          只在汇总里数一句是不够的 —— 读者核对的是**这一条**，
          而一条没有文件名的切片长得很像有文件名的（都有角标、都有正文）。
        */}
        <StrengthChip source={card.source} />
        {formulaHeavy && (
          <span
            className="shrink-0 rounded-panel border border-sev-notice/60 px-1 text-2xs text-sev-notice"
            title={
              "这条切片以公式为主（数学符号占 " +
              (card.text_quality.math_ratio * 100).toFixed(0) +
              "%），能证明「论文用了这个格式」，但很难证明任何机理结论。" +
              "检索到它是因为语义邻近，不是因为它在回答这个问题。"
            }
          >
            公式为主
          </span>
        )}
      </div>
      <p
        className={cn(
          "mt-1 whitespace-pre-wrap break-words text-ink-1",
          !expanded && "line-clamp-4",
        )}
      >
        {readable(card.content)}
      </p>
      {formulaHeavy && (
        // 这句话必须出现在正文里，而不是只藏在 title 里：
        // 排在第一位的那条切片如果以公式为主，读者会以为系统找到了支持。
        <p className="mt-1 text-2xs leading-relaxed text-ink-2">
          语义邻近 ≠ 可用证据：本片以公式为主，不宜作为机理结论的支撑。
        </p>
      )}
      {long && (
        <button
          type="button"
          onClick={() => setExpanded((v) => !v)}
          className="mt-0.5 text-acc hover:underline"
        >
          {expanded ? "收起" : `展开全文（${card.chars} 字）`}
        </button>
      )}
    </div>
  );
}
