import { Suspense, lazy, useState } from "react";
import type {
  KbDocuments,
  KbGapKind,
  KbOverview as KbOverviewData,
} from "../../lib/kb-types";
import { cn } from "../../lib/utils";
import { formatHours } from "./format";
import { KbGapQueue } from "./KbGapQueue";

/*
 * 登记表单按需加载。理由与 `ApprovalPanel` / `geoscene` 相同：它只在
 * **点了「登记」之后**才可能出现，而首屏 chunk 有一条 320 KB 的守卫线
 * （防 three 泄漏）。表单加进来时首屏从 313 KB 涨到 318.6 KB，余量只剩
 * 1.3 KB —— 下一个功能就会撞线，而撞线之后最省事的做法是"把数字改大"。
 * 拆出去就没有这个问题，代价是一次点击时多拉 3 KB。
 */
const KbGapForm = lazy(() =>
  import("./KbGapForm").then((m) => ({ default: m.KbGapForm })),
);

/**
 * 概览页。
 *
 * ## 这一页要回答的问题不是"库里有多少篇"
 *
 * 是**"这个知识库能信到什么程度"**。所以三件事按这个顺序排：
 *
 * 1. **规模**（六个数字，让人知道体量）
 * 2. **缺口**（读完"24 篇"立刻看到"其中 2 篇不在库里"，才不会把 24
 *    当成一个完整的数字。一个"共 22 篇"的读数看起来是完整的，
 *    而它其实漏了 2 篇，且没有任何地方会提示）
 * 3. **可引用性**（只有图谱切片带 `file_path` 血缘；向量库那 1,982 条
 *    没有文件级血缘。这件事必须写在界面上，否则用户会以为每条结论都能溯源）
 *
 * ## 缺口可以"登记"，但不能"修复"
 *
 * 每条缺口旁边有一个「登记」按钮，把"这篇没进图谱"变成一条**有对象**的
 * 记录写进队列。它**不触发入库** —— 入库要占满一张显卡、建图要几十分钟，
 * 不能由一次点击触发。界面上写清这一点（文案由后端给），因为一个看起来
 * 能一键修复的按钮，点了没反应比没有按钮更糟。
 */
export function KbOverview({
  overview,
  docs,
}: {
  overview: KbOverviewData;
  docs: KbDocuments;
}) {
  const { corpus, graph, vector, gaps } = overview;
  const ingest = formatHours(graph.ingest_seconds);
  const gapCount =
    gaps.not_ingested.length +
    gaps.unknown_source.length +
    gaps.orphaned.length +
    gaps.unfinished.length;

  return (
    <div className="space-y-3 text-2xs">
      <div className="grid grid-cols-2 gap-x-2 gap-y-1.5">
        <Stat label="语料文件" value={corpus.files} unit="篇" />
        <Stat label="已建图" value={graph.documents} unit="篇" />
        <Stat label="原文切片" value={graph.chunks} unit="片" />
        <Stat label="图谱实体" value={graph.entities} unit="个" />
        <Stat label="图谱关系" value={graph.relationships} unit="条" />
        <Stat label="建图耗时" value={ingest.value} unit={ingest.unit} />
      </div>

      {gapCount > 0 && <Gaps gaps={gaps} count={gapCount} />}

      <VectorCard vector={vector} graphChunks={graph.chunks} />

      <DocList docs={docs} missingTitles={overview.titles.missing.length} notIngested={gaps.not_ingested.length} />
    </div>
  );
}

function Stat({
  label,
  value,
  unit,
}: {
  label: string;
  value: number | string;
  unit: string;
}) {
  return (
    <div>
      <div className="text-ink-2">{label}</div>
      <div className="flex items-baseline gap-0.5">
        <span className="font-num text-sm text-ink-0">
          {typeof value === "number" ? value.toLocaleString() : value}
        </span>
        {unit && <span className="text-ink-2">{unit}</span>}
      </div>
    </div>
  );
}

function Gaps({ gaps, count }: { gaps: KbOverviewData["gaps"]; count: number }) {
  /*
   * 登记表单挂在**这一层**而不是 `GapList` 里：一次只开一个表单，
   * 而"开的是哪一条"只有父级知道。挂在列表项里会让每行各有一份状态，
   * 点开两行时两个表单同时存在 —— 而它们写的是同一个队列。
   */
  const [form, setForm] = useState<{ kind: KbGapKind; target: string } | null>(
    null,
  );
  const [refreshKey, setRefreshKey] = useState(0);

  return (
    <section className="rounded-panel border border-sev-notice/40 bg-sev-notice/5 px-2 py-1.5">
      <h3 className="text-sev-notice">有 {count} 处缺口 · 检索覆盖不到</h3>
      {gaps.not_ingested.length > 0 && (
        <GapList
          label="盘上有、图谱里没有"
          items={gaps.not_ingested}
          note="这些文件没有切片，检索不到它们的内容。"
          advice="先看该文档的入库状态：卡在 processing 会阻塞后续插入，用 scripts/repair_partial_ingest.py 清掉再重跑入库。"
          onReport={(name) => setForm({ kind: "not_in_graph", target: name })}
        />
      )}
      {form && (
        <Suspense fallback={null}>
          <KbGapForm
            kind={form.kind}
            target={form.target}
            onClose={() => {
              setForm(null);
              setRefreshKey((k) => k + 1);
            }}
          />
        </Suspense>
      )}
      {gaps.unknown_source.length > 0 && (
        <GapList
          label="来源无法识别"
          items={gaps.unknown_source}
          note="切片的 file_path 认不出对应哪个文件。"
          advice="多半是文件被改名或换过目录：改回原名，或补 manifest.json 的题录后重新入库。"
        />
      )}
      {gaps.orphaned.length > 0 && (
        <GapList
          label="图谱里有、盘上已删"
          items={gaps.orphaned}
          note="检索可能命中一篇打不开的文献。"
          advice="要么把原文件放回 ccus_docs/，要么从图谱里删掉该文档 —— 留着比缺更糟：它会让引用指向一个打不开的出处。"
        />
      )}
      {gaps.unfinished.length > 0 && (
        <GapList
          label="建图未完成"
          items={gaps.unfinished}
          note="processing 状态会阻塞后续所有插入，必须删掉重来。"
          advice="先跑 repair_partial_ingest.py，确认状态清空后再重新入库。"
        />
      )}

      {/*
        "缺了 2 篇"只说了问题，没说下一步。给建议而不是给链接：
        这个仓库没有稳定的文献获取入口，硬凑一个 URL 会变成一条迟早失效的死链。
        分类说清"从哪类来源补"，读者自己知道该去哪儿。
      */}
      <details className="mt-1.5 border-t border-line pt-1.5">
        <summary className="cursor-pointer text-ink-1 hover:text-ink-0">
          建议补充（从哪类来源）
        </summary>
        <ol className="mt-1 list-inside list-decimal space-y-0.5 text-ink-2">
          <li>
            <span className="text-ink-1">目标场地的公开项目报告</span>
            —— IEAGHG、GCCSI 的场地级报告，讲的是同一类储层与盖层，比通用综述更贴
          </li>
          <li>
            <span className="text-ink-1">同行评议期刊</span>
            —— IJGGC、SPE Journal、Fuel：机理与监测方法的主要来源，适合补
            "盖层完整性 / 井筒腐蚀"这类问题的证据
          </li>
          <li>
            <span className="text-ink-1">监管与标准文件</span>
            —— EPA Class VI、EU CCS Directive：阈值与上限的唯一法定出处，
            比论文更适合回答"这条线是多少"
          </li>
        </ol>
        <p className="mt-1 text-ink-2">
          补进来之后先补题录（DOI / arXiv ID），否则文档清单里只有文件名，
          读者无法判断它是不是同一篇。
        </p>
      </details>

      {/*
        队列要显示，因为"提交成功"三个字没法区分"记下了"和"按钮点了没反应"。
        演示模式下它显示"读不到"而不是"0 条" —— 队列在维护者的机器上。
      */}
      <KbGapQueue refreshKey={refreshKey} />
    </section>
  );
}

function GapList({
  label,
  items,
  note,
  advice,
  onReport,
}: {
  label: string;
  items: string[];
  note: string;
  /** 这一类的处置建议。缺口列表只报问题的话，读者下一步只能靠猜 */
  advice?: string;
  /**
   * 逐条登记。**只有"盘上有、图谱里没有"这一类给**：另外三类
   * （来源无法识别 / 图谱里有盘上已删 / 建图未完成）都能在本机直接处置，
   * 登记进队列反而绕远 —— 反馈是给"这台机器上解决不了的事"用的。
   */
  onReport?: (item: string) => void;
}) {
  return (
    <div className="mt-1">
      <div className="text-ink-1">
        {label}（{items.length}）
      </div>
      <ul className="mt-0.5 space-y-px text-ink-2">
        {items.map((n) => (
          <li key={n} className="flex items-baseline gap-1">
            <span className="truncate" title={n}>
              · {n}
            </span>
            {onReport && (
              <button
                type="button"
                onClick={() => onReport(n)}
                className="shrink-0 rounded border border-line px-1 text-ink-2 hover:border-ink-2 hover:text-ink-0"
                title="登记这条缺口 —— 它会进人工队列，不会自动重新入库"
              >
                登记
              </button>
            )}
          </li>
        ))}
      </ul>
      <p className="mt-0.5 text-ink-2">{note}</p>
      {advice && <p className="mt-0.5 text-ink-1">建议：{advice}</p>}
    </div>
  );
}

function VectorCard({
  vector,
  graphChunks,
}: {
  vector: KbOverviewData["vector"];
  graphChunks: number;
}) {
  return (
    <section className="rounded-panel border border-line px-2 py-1.5">
      <div className="flex items-baseline justify-between">
        <span className="text-ink-1">向量库</span>
        <span className="font-num text-ink-0">{vector.vectors.toLocaleString()}</span>
      </div>
      <p className="mt-0.5 text-ink-2">
        {vector.active_collection || "未配置"} ·{" "}
        {vector.available ? `${vector.collections.length} 个集合` : "库不可读"}
      </p>
      {/*
        没有血缘这件事必须写出来。它是当前的事实，不是待办 ——
        藏起来的结果是用户以为向量分支的结论也能点回原文。
      */}
      {!vector.lineage && (
        <p className="mt-1 border-t border-line pt-1 text-sev-notice">
          无文件级血缘：向量分支的召回无法回溯到具体文献。 只有图谱切片（{graphChunks}{" "}
          片）能给出引用。
        </p>
      )}
    </section>
  );
}

function DocList({
  docs,
  missingTitles,
  notIngested,
}: {
  docs: KbDocuments;
  missingTitles: number;
  notIngested: number;
}) {
  return (
    <section>
      <h3 className="mb-1 flex items-baseline justify-between text-ink-1">
        <span>文档清单</span>
        <span className="font-num text-ink-2">
          {docs.total - notIngested}/{docs.total}
        </span>
      </h3>
      <ul className="space-y-px">
        {docs.documents.map((d) => (
          <li
            key={d.name}
            className={cn(
              "flex items-baseline gap-1.5 border-l-2 pl-1.5 py-0.5",
              d.ingested ? "border-transparent" : "border-sev-notice/60",
            )}
          >
            <span
              className={cn(
                "min-w-0 flex-1 truncate",
                d.ingested ? "text-ink-1" : "text-ink-2",
              )}
              title={d.title || d.name}
            >
              {d.name}
            </span>
            {d.ingested ? (
              <span className="shrink-0 font-num text-ink-2">{d.chunks}</span>
            ) : (
              <span className="shrink-0 text-sev-notice">未入库</span>
            )}
          </li>
        ))}
      </ul>
      {missingTitles > 0 && (
        <p className="mt-1 text-ink-2">
          {missingTitles} 篇没有题录（文件名即全部信息）。
          <span className="text-ink-1">
            建议：按 DOI / arXiv ID 补进 manifest.json
          </span>
          —— 没有题录时，同一篇论文的两次抓取会被当成两篇。
        </p>
      )}
    </section>
  );
}
