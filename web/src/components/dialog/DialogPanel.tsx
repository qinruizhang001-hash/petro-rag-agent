import { useEffect, useRef, useState } from "react";
import { dataSource } from "../../lib/api";
import {
  STAGE_LABEL,
  SUGGESTIONS,
  askStream,
  clockNow,
  demoTranscript,
  nextId,
} from "../../lib/chat";
import type { AskStage, ChatMessage, Citation } from "../../lib/chat";
import { groupCitations } from "../../lib/citations";
import { fetchKbDocument } from "../../lib/kb";
import { cn } from "../../lib/utils";
import { VOICE_PRIVACY_NOTE, mergeTranscript } from "../../lib/voice";
import { StrengthChip } from "../kb/StrengthChip";
import { ChunkPeek } from "./ChunkPeek";
import { CitationText } from "./CitationText";
import { VoiceButton } from "./VoiceButton";

/**
 * 对话（基础版）。
 *
 * ## 它做什么、不做什么
 *
 * 做：显示对话历史、发问、展示回答、把回答的**数值依据**（事实块）与
 * **原文证据**（可点角标）分开摆出来。
 *
 * 不做：审批状态机。审批是跨班次流程（draft → submitted → approved →
 * rejected → reopened），它需要落盘、权限、审计三样东西同时在位。
 * 在对话里塞一个只在内存里改状态的"批准"按钮，会比没有更危险 ——
 * 它看起来像批准过了。
 *
 * ## 两种"点引用"是两件事
 *
 * | 点哪里 | 去哪 | 为什么 |
 * |---|---|---|
 * | 正文里的角标 `[2]` | 知识库详情页（`KbDocDetail`） | 读者想知道"这句话出自哪"，他要的是上下文，不是这一片 |
 * | 「原文证据」列表里的 `[2]` | 就地展开这一片 | 他已经在看清单了，点开是为了核对这一段，不该被带离当前对话 |
 *
 * 两个入口都做成"跳走"会让核对变成一次往返；都做成"就地展开"则
 * 永远看不到上下文。分开是对的，但**必须让两者的样子不同**，
 * 否则同一个人在同一个页面上点两个长得一样的按钮，得到两种结果。
 */
export function DialogPanel({
  onOpenCitation,
}: {
  /** 正文角标的落点。不传时退回"就地展开"，让这个面板能单独用 */
  onOpenCitation?: (c: Citation) => void;
}) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  /** 语音识别中"还没定下来"的那半句。只显示，不进 input。 */
  const [interim, setInterim] = useState("");
  const [busy, setBusy] = useState(false);
  /** 当前真的在跑的那一步。`null` = 没在跑。文案直接取 `STAGE_LABEL`，不另编 */
  const [stage, setStage] = useState<AskStage | null>(null);
  const [peek, setPeek] = useState<{ citation: Citation; siblings: number } | null>(null);
  const [docError, setDocError] = useState<string | null>(null);
  const listRef = useRef<HTMLDivElement>(null);

  /** 打开一片原文。`siblings` 是同一来源的片数（合并列表里要告诉读者"这是第 1 片"）。 */
  const openPeek = (citation: Citation, siblings = 1) => {
    setDocError(null);
    setPeek({ citation, siblings });
  };

  // 演示模式的初始两段对话：来自真实抓取的快照，各自内部一致。
  useEffect(() => {
    if (dataSource !== "fixture") return;
    let alive = true;
    demoTranscript()
      .then((seeded) => {
        if (alive && seeded.length > 0) setMessages(seeded);
      })
      .catch(() => {
        /* 快照读不出来就保持空对话，不拿编的文案填 */
      });
    return () => {
      alive = false;
    };
  }, []);

  // 新消息出现后滚到底。`messages.length` 变化即触发 —— 内容替换
  // （pending → 正式消息）长度不变，但那时用户已经在看那一条了。
  useEffect(() => {
    const el = listRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages.length, busy]);

  const send = async (raw: string) => {
    const q = raw.trim();
    if (!q || busy) return;
    setBusy(true);
    setInput("");
    setPeek(null);
    setStage("condition");

    const userId = nextId("u");
    const pendingId = nextId("p");
    setMessages((prev) => [
      ...prev,
      { id: userId, role: "user", kind: "answer", text: q, at: clockNow() },
      {
        id: pendingId,
        role: "assistant",
        kind: "answer",
        text: "",
        at: clockNow(),
        pending: true,
      },
    ]);

    try {
      /*
       * 分段渲染，不是逐字打印。
       *
       * `answer` 一到就先把正文和数字依据渲染出来（这一步是模板计算，毫秒级），
       * 出处还在检索（实测 ~21 秒）。原先两件事 await 在一起，于是数值回答
       * 明明早就好了，界面还要陪着检索一起等 —— 那才是"只有一句正在取数与检索"。
       */
      await askStream(q, (event) => {
        if (event.type === "stage") {
          setStage(event.stage);
          return;
        }
        if (event.type === "answer") {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === pendingId
                ? {
                    id: pendingId,
                    role: "assistant",
                    kind: "answer",
                    text: event.answer.text,
                    at: clockNow(),
                    demo: event.demo,
                    facts: event.answer.facts,
                    unresolved: event.answer.unresolved,
                    unsourced: event.answer.unsourced_numbers,
                    windowLabel: event.answer.window?.label,
                    notes: event.notes,
                    citations: [],
                    retrievalNote: "",
                    retrieving: true,
                  }
                : m,
            ),
          );
          return;
        }
        if (event.type === "evidence") {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === pendingId
                ? {
                    ...m,
                    citations: event.citations,
                    retrievalNote: event.retrievalNote,
                    retrieving: false,
                  }
                : m,
            ),
          );
        }
      });
    } catch (e: unknown) {
      const text = e instanceof Error ? e.message : String(e);
      setMessages((prev) =>
        prev.map((m) =>
          m.id === pendingId
            ? {
                id: pendingId,
                role: "assistant",
                kind: "error",
                text: `请求失败：${text}`,
                at: clockNow(),
              }
            : m,
        ),
      );
    } finally {
      setBusy(false);
      setStage(null);
    }
  };

  const openDoc = (name: string, focus?: string) => {
    setDocError(null);
    fetchKbDocument(name, focus).catch((e: unknown) => {
      setDocError(e instanceof Error ? e.message : String(e));
    });
  };

  if (peek) {
    return (
      <div className="flex min-h-0 flex-1 flex-col overflow-auto pr-1">
        <ChunkPeek
          citation={peek.citation}
          siblings={peek.siblings}
          onBack={() => {
            setPeek(null);
            setDocError(null);
          }}
          onOpenDoc={openDoc}
        />
        {docError && (
          <p className="mt-2 rounded-panel border border-line px-2 py-1.5 text-ink-1">
            {docError}
          </p>
        )}
      </div>
    );
  }

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-1.5">
      <div className="flex items-baseline gap-1.5 text-2xs text-ink-2">
        <span
          className={cn(
            "rounded-panel border px-1",
            dataSource === "fixture"
              ? "border-line-strong text-ink-2"
              : "border-acc/40 text-acc",
          )}
          title={
            dataSource === "fixture"
              ? "回答来自 scripts/capture_ops_fixtures.py 抓下来的真实响应快照，不随输入变化"
              : "回答来自运行中的 /api/ops/condition"
          }
        >
          {dataSource === "fixture" ? "演示快照" : "live"}
        </span>
        <span className="font-num">{messages.length} 条</span>
        <button
          type="button"
          onClick={() => {
            setMessages([]);
            setPeek(null);
            setDocError(null);
          }}
          className="ml-auto text-ink-2 hover:text-ink-1"
        >
          清空
        </button>
      </div>

      <div ref={listRef} className="min-h-0 flex-1 space-y-2 overflow-auto pr-1">
        {messages.length === 0 && <Empty onPick={(q) => void send(q)} />}
        {messages.map((m) => (
          <Bubble
            key={m.id}
            message={m}
            stageLabel={m.pending && stage ? `${STAGE_LABEL[stage]}…` : undefined}
            onPeek={openPeek}
            onJump={onOpenCitation ?? openPeek}
          />
        ))}
        <VoiceBoundary />
        <NotDone />
      </div>

      <div className="flex gap-1">
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            // Enter 发送、Shift+Enter 换行。现场单手操作的场景比
            // "我想在问题里写一段话"更常见。
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              void send(input);
            }
          }}
          rows={2}
          placeholder="问工况：过去 4 小时有什么异常？"
          aria-label="向系统提问"
          className="min-w-0 flex-1 resize-none rounded-panel border border-line bg-surface-2 px-1.5 py-1 text-2xs text-ink-0 outline-none placeholder:text-ink-2 focus:border-acc/60"
        />
        <button
          type="button"
          onClick={() => void send(input)}
          disabled={busy || !input.trim()}
          className="shrink-0 self-stretch rounded-panel border border-acc/40 px-2 text-2xs text-acc disabled:border-line disabled:text-ink-2"
        >
          {busy ? "处理中" : "发送"}
        </button>
      </div>

      {/*
        语音。**它只往输入框里塞字** —— 没有发送能力，也不需要。
        理由在 `lib/voice.ts` 顶部：识别错一个测点名，系统会给出一个数值
        正确、对象错误的回答，而提问的人已经走了。
      */}
      <div className="mt-1 flex items-start justify-end gap-2">
        {interim && (
          <p className="min-w-0 flex-1 text-right text-2xs leading-tight text-ink-2">
            <span className="italic">「{interim}」</span>
            <span className="ml-1">识别中，未确认</span>
            {/*
              说的时候才显示"音频去哪了"。常驻会占掉一行、被读成噪音；
              而只在文档里写，按按钮的人不会翻文档。
            */}
            <span
              className="ml-1 border-b border-dashed border-line-strong"
              title={VOICE_PRIVACY_NOTE}
            >
              · 云端识别
            </span>
          </p>
        )}
        <div className={cn("ml-auto", interim ? "" : "flex-1")}>
          <VoiceButton
            onCommit={(text) => setInput((prev) => mergeTranscript(prev, text))}
            onInterim={(t) => setInterim(t ?? "")}
            disabled={busy}
          />
        </div>
      </div>
    </div>
  );
}

function Empty({ onPick }: { onPick: (q: string) => void }) {
  return (
    <div className="space-y-2 text-2xs">
      <p className="text-ink-2">
        对话会同时给出两样东西：<span className="text-ink-1">数字依据</span>
        （每条都能回代）和<span className="text-ink-1">原文证据</span>
        （点角标看切片）。两者都不成立时，它会说不知道。
      </p>
      <div className="space-y-1">
        <p className="text-ink-2">可以这样问：</p>
        {SUGGESTIONS.map((q) => (
          <button
            key={q}
            type="button"
            onClick={() => onPick(q)}
            className="block w-full rounded-panel border border-line px-2 py-1 text-left text-ink-1 hover:border-acc/40 hover:text-acc"
          >
            {q}
          </button>
        ))}
      </div>
    </div>
  );
}

function Bubble({
  message,
  stageLabel,
  onPeek,
  onJump,
}: {
  message: ChatMessage;
  /** 正在跑的那一步的人话说明。只有 `pending` 的气泡会用到 */
  stageLabel?: string;
  /** 证据列表里的按钮：就地展开这一片 */
  onPeek: (c: Citation, siblings?: number) => void;
  /** 正文里的角标：跳到知识库详情 */
  onJump: (c: Citation) => void;
}) {
  if (message.role === "user") {
    return (
      <div className="flex flex-col items-end gap-0.5">
        <div className="max-w-[92%] rounded-panel border border-acc/30 bg-acc/10 px-2 py-1 text-2xs text-ink-0">
          {message.text}
        </div>
        <span className="text-2xs text-ink-2">
          {message.at}
          {message.demo && " · 演示"}
        </span>
      </div>
    );
  }

  if (message.kind === "error") {
    return (
      <div className="rounded-panel border border-sev-alert/50 bg-sev-alert/10 px-2 py-1.5 text-2xs text-sev-alert">
        {message.text}
      </div>
    );
  }

  const citations = message.citations ?? [];
  // 按来源合并：`reference_id` 是来源编号，12 片原文来自 6 篇时
  // 逐片列出来会看到同一个角标重复三遍，读起来像列表坏了。
  const groups = groupCitations(citations);
  const retrieval = message.kind === "retrieval";

  return (
    <article className="rounded-panel border border-line bg-surface-2 px-2 py-1.5 text-2xs">
      <header className="flex items-baseline gap-1.5 text-ink-2">
        <span className="text-ink-1">{retrieval ? "检索" : "系统"}</span>
        {message.windowLabel && <span>· {message.windowLabel}</span>}
        {message.demo && (
          <span className="rounded-panel border border-line-strong px-1" title="来自固定快照，不是本次会话产生的">
            演示
          </span>
        )}
        <span className="ml-auto font-num">{message.at}</span>
      </header>

      {message.pending ? (
        <p className="mt-1 text-ink-2">{stageLabel ?? "正在处理…"}</p>
      ) : (
        <p className="mt-1 whitespace-pre-wrap break-words leading-relaxed text-ink-0">
          <CitationText
            text={message.text}
            citations={citations}
            onPick={onJump}
            hint="跳到知识库看这篇的原文"
          />
        </p>
      )}

      {/*
        正文已经能读了、出处还在路上。必须与 `pending` 分开写：
        "整条还没好"和"只有引用还没好"会让读者做两件不同的事 ——
        后者现在就可以读，前者只能等。
      */}
      {message.retrieving && (
        <p className="mt-1 text-ink-2">{STAGE_LABEL.retrieval}…（这一步最慢，正文已可读）</p>
      )}

      {/*
        无血缘角标为什么点不开，必须**写在正文下面**，不能只放在 tooltip 里。
        tooltip 要悬浮才出现，而"点了没反应"才是读者实际遇到的现象 ——
        那时他会先怀疑界面坏了。
      */}
      {(() => {
        const orphan = citations.find((c) => !c.source);
        if (!orphan) return null;
        return (
          <p className="mt-1 text-ink-2">
            正文里虚线边框的角标（如 {orphan.reference_id}）没有文件级血缘，点不开 ——
            它来自向量召回，只能说明库里有这段，说明不了出自哪一篇。
          </p>
        );
      })()}

      {/*
        守卫触发必须显示，不能藏。`unsourced_numbers` 非空意味着模型措辞里
        出现了事实块中找不到的数字、正文已整篇回退到模板 —— 那是**一次真实
        的拦截**，读者有权知道这段文字被替换过。
      */}
      {message.unresolved && message.unresolved.length > 0 && (
        <p className="mt-1 rounded-panel border border-sev-notice/50 bg-sev-notice/5 px-1.5 py-1 text-sev-notice">
          问题里有 {message.unresolved.length} 处没认出来：
          {message.unresolved.join("、")}。这一段没有假装回答它。
        </p>
      )}
      {message.unsourced && message.unsourced.length > 0 && (
        <p className="mt-1 rounded-panel border border-sev-alert/50 bg-sev-alert/5 px-1.5 py-1 text-sev-alert">
          措辞里有 {message.unsourced.length} 个数字无法在事实块中找到（
          {message.unsourced.join("、")}），正文已整篇回退到模板。
        </p>
      )}

      {message.facts && message.facts.length > 0 && (
        <details className="mt-1">
          <summary className="cursor-pointer text-ink-2 hover:text-ink-1">
            数字依据（{message.facts.length} 条，逐条可回代）
          </summary>
          <ul className="mt-1 space-y-0.5 border-l border-line pl-2 text-ink-2">
            {message.facts.map((f) => (
              <li key={f}>{f}</li>
            ))}
          </ul>
        </details>
      )}

      {citations.length > 0 && (
        <section className="mt-1.5 border-t border-line pt-1.5">
          <h4 className="flex items-baseline justify-between text-ink-1">
            <span>原文证据</span>
            <span className="font-num text-ink-2">
              {citations.length} 片 / {groups.length} 篇
            </span>
          </h4>
          <ul className="mt-0.5 space-y-px">
            {groups.map((g) => (
              <li key={g.reference_id} className="flex items-baseline gap-1">
                {/*
                  编号 = 出处，点了去详情页；文件名 = 这一篇的正文，点了就地展开。
                  两个动作不同，所以样子不同（带框的按钮 / 像链接的文字）。
                  没有文件血缘时编号不可点 —— 那个落点会是一页空白，
                  而"点了没反应"比"不能点"更像故障。
                */}
                <button
                  type="button"
                  onClick={() => onJump(g.chunks[0]!)}
                  disabled={!g.source}
                  title={
                    g.source
                      ? `${g.source} · 跳到知识库看这篇的原文`
                      : "该证据没有文件级血缘，无法定位到具体文献"
                  }
                  className="shrink-0 rounded-panel border border-acc/40 px-1 font-num text-acc hover:bg-acc/10 disabled:border-line disabled:text-ink-2 disabled:hover:bg-transparent"
                >
                  {g.reference_id}
                </button>
                <button
                  type="button"
                  onClick={() => onPeek(g.chunks[0]!, g.chunks.length)}
                  title="就地展开这一篇的第一片原文（不离开对话）"
                  className="min-w-0 flex-1 truncate text-left text-ink-1 hover:text-acc"
                >
                  {g.source || "无出处"}
                </button>
                <span className="shrink-0 font-num text-ink-2">{g.chunks.length} 片</span>
                <StrengthChip source={g.source} />
              </li>
            ))}
          </ul>
        </section>
      )}

      {message.retrievalNote && (
        <p className="mt-1 leading-relaxed text-ink-2">{message.retrievalNote}</p>
      )}

      {message.notes?.map((n) => (
        <p key={n} className="mt-1 text-sev-notice">
          {n}
        </p>
      ))}
    </article>
  );
}

/**
 * 还没做的部分。
 *
 * 写在界面上而不是只写在 README 里：一个能问能答的对话，最容易让人
 * 顺口问一句"那能不能直接批准"。答案是"这里没有批准这个动作"，
 * 而这句话必须出现在被问到的地方。
 */
function NotDone() {
  return (
    <details className="rounded-panel border border-line px-2 py-1.5 text-2xs">
      <summary className="cursor-pointer text-ink-2 hover:text-ink-1">还没做</summary>
      <ul className="mt-1 space-y-0.5 text-ink-2">
        <li>
          <span className="text-ink-1">审批状态机</span>
          （draft → submitted → approved / rejected → reopened）。它要落盘、要权限、
          要审计三样同时在位，只在内存里改状态的"批准"按钮比没有更危险。
        </li>
        <li>
          <span className="text-ink-1">流式回答</span>：现在是一次性返回，
          长回答要等几秒，期间只有一句"正在取数与检索"。
        </li>
        <li>
          <span className="text-ink-1">流式语音</span>：本地识别是一次说完才出字
          （模型本身是流式的，接口没做增量），录音期间只显示时长。不用它换
          「边说边出半句」是因为：`井口压力` 先显示成 `景口压力` 再改对，
          读者已经读过错的那一遍了。
        </li>
      </ul>
    </details>
  );
}

/**
 * 语音权限边界。
 *
 * 这段表格是整套界面里唯一"做错了会伤到人"的地方，而且它不依赖后端。
 * 放在对话页而不是写进文档，因为下一个接手的人会先看界面。
 * 默认折叠：它是约束，不是每天要读的内容。
 */
function VoiceBoundary() {
  return (
    <details className="rounded-panel border border-line px-2 py-1.5 text-2xs">
      <summary className="cursor-pointer text-ink-1">
        语音权限边界（安全约束，先于功能）
      </summary>
      <table className="mt-1.5 w-full border-collapse text-left">
        <thead>
          <tr className="text-ink-2">
            <th className="w-1/2 border-b border-line pb-1 font-normal">语音可以做</th>
            <th className="w-1/2 border-b border-line pb-1 pl-2 font-normal text-sev-alert">
              语音不可以做
            </th>
          </tr>
        </thead>
        <tbody className="align-top">
          <tr>
            <td className="py-1 pr-2 text-ink-1">查询工况、打开某个测点</td>
            <td className="py-1 pl-2 text-ink-1">直接开关阀门</td>
          </tr>
          <tr>
            <td className="py-1 pr-2 text-ink-1">请求生成报告草稿</td>
            <td className="py-1 pl-2 text-ink-1">直接调泵速</td>
          </tr>
          <tr>
            <td className="py-1 pr-2 text-ink-1">发起"建议降速"的讨论</td>
            <td className="py-1 pl-2 text-ink-1">任何改变物理状态的操作</td>
          </tr>
        </tbody>
      </table>
      <p className="mt-1.5 text-ink-2">
        语音最多做到"打开确认面板"。落笔必须是人手点击 + 签名 ——
        确认弹窗里的签名框为空时，提交按钮不放行。
      </p>
      <p className="mt-1 text-ink-2">
        确认弹窗必须写清三件事：动作的具体参数（42 t/h → 35 t/h，不是"降速"）、
        系统已核验/未核验的边界、操作人签名。第 2 条最容易漏 ——
        只写"确认执行吗"等于把责任推给人。
      </p>
    </details>
  );
}
