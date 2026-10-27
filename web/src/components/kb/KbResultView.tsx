import type { KbSearchResult } from "../../lib/kb-types";
import { readable } from "../../lib/text";
import { KbChunkCard } from "./KbChunkCard";
import { RefBadge } from "./RefBadge";

/**
 * 检索结果的三种状态。
 *
 * ## 三种状态不能合并
 *
 * | status | 含义 | 用户该做什么 |
 * |---|---|---|
 * | `ok` | 有结果 | 看证据 |
 * | `empty` | **库里确实没有**相关内容 | 补文献 |
 * | `unavailable` | **检索没跑起来**（Ollama / 图谱未就绪） | 修服务 |
 *
 * `empty` 与 `unavailable` 如果都渲染成"没有结果"，操作员会去补文献，
 * 而真实原因只是 Ollama 没启动。把"我不知道"渲染成"没有"，
 * 是唯一一种不会报错的失败。
 */
export function KbResultView({
  result,
  onOpenDoc,
}: {
  result: KbSearchResult;
  onOpenDoc: (name: string, focus?: string) => void;
}) {
  if (result.status === "unavailable") {
    return (
      <section className="rounded-panel border border-sev-notice/50 bg-sev-notice/5 px-2 py-1.5">
        <h3 className="text-sev-notice">检索未连接</h3>
        <p className="mt-0.5 text-ink-1">{result.reason}</p>
        <p className="mt-1 text-ink-2">
          这不等于库里没有内容 —— 是检索服务没跑起来。先确认 Ollama 与图谱已就绪，
          再重试。
        </p>
      </section>
    );
  }

  if (result.status === "empty") {
    return (
      <section className="rounded-panel border border-line px-2 py-1.5">
        <h3 className="text-ink-1">图谱里没有相关内容</h3>
        <p className="mt-0.5 text-ink-2">{result.reason}</p>
      </section>
    );
  }

  return (
    <div className="space-y-2">
      <References result={result} onOpenDoc={onOpenDoc} />
      <Chunks result={result} onOpenDoc={onOpenDoc} />
      <GraphHits result={result} />
      <p className="text-ink-2">
        耗时 <span className="font-num">{result.elapsed_ms > 0 ? result.elapsed_ms : "—"}</span>
        {result.elapsed_ms > 0 ? " ms" : ""} · 模式 {result.mode}
      </p>
    </div>
  );
}

/** 来源表：一次检索用到了哪几篇文献。 */
function References({
  result,
  onOpenDoc,
}: {
  result: KbSearchResult;
  onOpenDoc: (name: string, focus?: string) => void;
}) {
  const { evidence } = result;
  return (
    <section>
      <h3 className="flex items-baseline justify-between text-ink-1">
        <span>来源</span>
        <span className="font-num text-ink-2">
          {evidence.stats.sources} 篇 / {evidence.stats.chunks} 片
        </span>
      </h3>
      <ul className="mt-0.5 space-y-px">
        {evidence.references.map((r) => (
          <li key={r.reference_id} className="flex items-baseline gap-1.5">
            <RefBadge id={r.reference_id} />
            <button
              type="button"
              onClick={() => r.source && onOpenDoc(r.source)}
              disabled={!r.source}
              className="min-w-0 flex-1 truncate text-left text-acc hover:underline disabled:text-ink-2 disabled:no-underline"
              title={r.source || "无出处"}
            >
              {r.source || "无出处"}
            </button>
          </li>
        ))}
      </ul>
      {/*
        无出处条数。为 0 时也要说一句"全部可回溯" —— 一个可以显示为 0 的
        数字比"没有提示"强：读者能确认这件事被检查过。
      */}
      <p className="mt-1 text-ink-2">
        {evidence.stats.unsourced > 0 ? (
          <span className="text-sev-notice">
            {evidence.stats.unsourced} 片无法回溯来源（入库时未写血缘）
          </span>
        ) : (
          <>全部 {evidence.stats.traceable} 片可回溯到原文</>
        )}
      </p>
      {/*
        公式为主的切片数。与"无出处"并列，因为它回答的是同一层问题：
        这条证据**能不能支撑结论**。检索器会把公式块排到前面（满篇都是
        CO2 / pressure，向量相似度确实高），而它对"盖层完整性"提供不了
        任何机理证据 —— 这件事必须被数出来，否则面板看起来像找到了支持。
      */}
      <p className="mt-0.5 text-ink-2">
        {evidence.stats.low_quality_chunks > 0 ? (
          <span className="text-sev-notice">
            {evidence.stats.low_quality_chunks} 片以公式为主，不宜作为机理证据
          </span>
        ) : (
          <>全部切片均以文字叙述为主</>
        )}
      </p>
    </section>
  );
}

/** 原文证据：一条一张卡。 */
function Chunks({
  result,
  onOpenDoc,
}: {
  result: KbSearchResult;
  onOpenDoc: (name: string, focus?: string) => void;
}) {
  return (
    <section className="space-y-1">
      <h3 className="text-ink-1">原文证据</h3>
      {result.evidence.chunks.map((c) => (
        <KbChunkCard
          key={c.chunk_id || c.reference_id + c.content.slice(0, 8)}
          card={c}
          onOpen={onOpenDoc}
        />
      ))}
    </section>
  );
}

/** 图谱命中：实体与关系。 */
function GraphHits({ result }: { result: KbSearchResult }) {
  const { evidence } = result;
  if (evidence.entities.length === 0 && evidence.relationships.length === 0) return null;

  return (
    <section className="space-y-1">
      <h3 className="flex items-baseline justify-between text-ink-1">
        <span>图谱命中</span>
        <span className="font-num text-ink-2">
          {evidence.stats.entities} 实体 / {evidence.stats.relationships} 关系
        </span>
      </h3>
      {evidence.entities.slice(0, 6).map((e) => (
        <div key={e.name} className="rounded-panel border border-line px-1.5 py-1">
          <div className="flex items-baseline gap-1">
            <span className="min-w-0 flex-1 truncate text-ink-0" title={e.name}>
              {e.name}
            </span>
            <span className="shrink-0 text-ink-2">{e.entity_type}</span>
          </div>
          <p className="mt-0.5 line-clamp-2 text-ink-2">{readable(e.description)}</p>
          {/* 实体是多来源的：显示篇数而不是一个糊在一起的长文件名 */}
          <p className="mt-0.5 text-ink-2">
            {e.sources.length} 篇 · {e.chunk_ids.length} 片支撑
          </p>
        </div>
      ))}
      {evidence.relationships.slice(0, 4).map((r) => (
        <div key={`${r.src}->${r.tgt}`} className="rounded-panel border border-line px-1.5 py-1">
          <div className="flex flex-wrap items-baseline gap-1 text-ink-1">
            <span className="truncate">{r.src}</span>
            <span className="text-acc-2">→</span>
            <span className="truncate">{r.tgt}</span>
          </div>
          <p className="mt-0.5 line-clamp-2 text-ink-2">{readable(r.description)}</p>
        </div>
      ))}
    </section>
  );
}
