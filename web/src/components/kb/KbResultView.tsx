import { Suspense, lazy, useState } from "react";
import { summarize } from "../../lib/evidence";
import { kbDataSource } from "../../lib/kb";
import type { KbSearchResult } from "../../lib/kb-types";
import { readable } from "../../lib/text";
import { KbChunkCard } from "./KbChunkCard";
// 与 `KbOverview` 同一个理由：表单只在点击后出现，不该占首屏体积。
const KbGapForm = lazy(() =>
  import("./KbGapForm").then((m) => ({ default: m.KbGapForm })),
);
import { RefBadge } from "./RefBadge";
import { StrengthChip } from "./StrengthChip";

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
 *
 * ## 三种状态各自有一个"登记"入口，但只有两种该给
 *
 * * `empty` → 登记「检索没答到」。这是关于**语料**的事实，反馈有意义。
 * * `ok`    → 登记「引的原文不对」。检索跑对了但证据不相关，反馈有意义。
 * * `unavailable` → **不给**。这时检索根本没跑，登记一条"没答到"会把
 *   一次服务故障记成一次语料缺口，而队列里那个数字会被当成"该补文献了"。
 */
export function KbResultView({
  result,
  onOpenDoc,
}: {
  result: KbSearchResult;
  onOpenDoc: (name: string, focus?: string) => void;
}) {
  /*
   * 表单开在这里而不是 `KbSearch`：一次检索的结果只属于这一个视图，
   * 而"哪一次检索要反馈"就是 `result` 本身。带着 `result.query` 下去，
   * 后端就能复现这次检索 —— `wrong_evidence` 强制要求这个字段。
   */
  const [form, setForm] = useState<"unanswered" | "wrong_evidence" | null>(null);

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
        {form === "unanswered" ? (
          <Suspense fallback={null}>
            <KbGapForm
              kind="unanswered"
              query={result.query}
              onClose={() => setForm(null)}
            />
          </Suspense>
        ) : (
          <button
            type="button"
            onClick={() => setForm("unanswered")}
            className="mt-1 rounded border border-line px-2 py-0.5 text-ink-1 hover:border-ink-2 hover:text-ink-0"
          >
            登记「这个问题该有答案」
          </button>
        )}
      </section>
    );
  }

  return (
    <div className="space-y-2">
      <EvidenceStrength result={result} />
      <References result={result} onOpenDoc={onOpenDoc} />
      <Chunks result={result} onOpenDoc={onOpenDoc} />
      <GraphHits result={result} />
      {form === "wrong_evidence" ? (
        <Suspense fallback={null}>
          <KbGapForm
            kind="wrong_evidence"
            query={result.query}
            onClose={() => setForm(null)}
          />
        </Suspense>
      ) : (
        <button
          type="button"
          onClick={() => setForm("wrong_evidence")}
          className="rounded border border-line px-2 py-0.5 text-ink-2 hover:border-ink-2 hover:text-ink-0"
        >
          登记「引的原文不对」
        </button>
      )}
      <p className="text-ink-2">
        耗时 <span className="font-num">{result.elapsed_ms > 0 ? result.elapsed_ms : "—"}</span>
        {result.elapsed_ms > 0 ? " ms" : ""} · 模式 {result.mode}
      </p>
    </div>
  );
}

/**
 * 证据强度。
 *
 * ## 为什么这一块排在证据卡前面
 *
 * "这次检索找到了 12 片"是一个会让人直接往下读的数字，而"其中 3 片
 * 无法回溯到任何文献"会改变他怎么读那 12 片。把后者放在最后，
 * 读者已经看完证据了；放在最前，他带着这个前提往下看。
 *
 * ## 三个数字分开数，不合成一个"强度分"
 *
 * 可溯源、无血缘、公式为主是**三个不同的缺陷**，一条切片可以同时犯后两个。
 * 合成一个分数之后，界面上只剩一个颜色，而读者无从知道缺的是哪一项。
 */
function EvidenceStrength({ result }: { result: KbSearchResult }) {
  const s = summarize(result.evidence);
  const fixture = kbDataSource === "fixture";

  return (
    <section className="rounded-panel border border-line px-2 py-1.5">
      <h3 className="flex items-baseline justify-between text-ink-1">
        <span>证据强度</span>
        <span className="font-num text-ink-2">
          {s.chunks} 片 / {s.sources} 篇
        </span>
      </h3>

      <ul className="mt-1 space-y-0.5">
        <li className="flex items-baseline gap-1.5">
          <span className="w-14 shrink-0 text-ink-2">可溯源</span>
          <span className="font-num text-ink-0">{s.traceable}</span>
          <span className="text-ink-2">片带 file_path，能点回原文</span>
        </li>
        <li className="flex items-baseline gap-1.5">
          <span className="w-14 shrink-0 text-ink-2">无文件血缘</span>
          <span className={s.unattributed > 0 ? "font-num text-sev-notice" : "font-num text-ink-0"}>
            {s.unattributed}
          </span>
          <span className="text-ink-2">
            {s.unattributed > 0
              ? "片只有向量召回，无法回溯到具体文献"
              : "片 —— 这一批没有不可回溯的证据"}
          </span>
        </li>
        <li className="flex items-baseline gap-1.5">
          <span className="w-14 shrink-0 text-ink-2">公式为主</span>
          <span className={s.formulaHeavy > 0 ? "font-num text-sev-notice" : "font-num text-ink-0"}>
            {s.formulaHeavy}
          </span>
          <span className="text-ink-2">
            {s.formulaHeavy > 0
              ? "片 —— 语义邻近，但不支撑机理结论"
              : "片 —— 全部以文字叙述为主"}
          </span>
        </li>
      </ul>

      <p className="mt-1 border-t border-line pt-1 text-ink-2">
        检索来源：
        <span className={fixture ? "text-ink-1" : "text-acc"}>
          {fixture ? "fixture（真实响应快照）" : "live（后端实时检索）"}
        </span>
        {fixture && " · 结果是抓取时那个固定问题的，不随输入变化"}
      </p>
    </section>
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
        <span className="font-num text-ink-2">{evidence.stats.sources} 篇</span>
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
            <StrengthChip source={r.source} />
          </li>
        ))}
      </ul>
      {/*
        计数只在上面的「证据强度」里出现一次。
        同一件事在两处各算一遍，迟早会有一处忘了改，而界面上两个数字
        并排显示时，读者无从判断哪个是对的。
      */}
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
      {/*
        实体与关系是模型抽出来的中间产物，**不能引用**。
        它们带 `sources`（多来源），但那不是"这句话出自哪一片" ——
        一个合并实体来自 16 篇论文，点它不知道会落到哪一篇。
        所以这里没有角标，只有"多少篇支撑"。
      */}
      <p className="text-ink-2">实体与关系是检索线索，不是可引用证据 —— 引用角标只挂在原文切片上。</p>
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
