/**
 * 对话层：把"数值回答"与"原文证据"拼成一条可核对的消息。
 *
 * ## 为什么是两次调用，而不是一个新接口
 *
 * `/ops/condition` 回答的是**数字**（窗口、统计量、越限点数），
 * `/kb/search` 回答的是**出处**（哪一篇的哪一片）。两边的守卫完全不同：
 * 前者有"措辞里的数字必须能在事实块里找到"的校验，后者有
 * `reference_id` → `file_path` 的血缘。合成一个接口就要在一个响应里
 * 同时表达两种可信度，而界面上恰恰需要把它们分开显示。
 *
 * ## fixture 模式下不能"顺手挂上"引用
 *
 * 检索快照固定是「盖层完整性对 CO₂ 泄漏风险的影响」那一次的真实响应。
 * 如果用户问的是别的句子，把这份证据挂到回答下面，会得到一条
 * **看起来有出处、实际出处是别的问题**的回答 —— 这比没有引用糟得多，
 * 因为它把"编造"从模型搬到了界面上，而且没有任何地方会报错。
 * 所以：问句对不上快照时，引用一律为空，并把原因写出来。
 */

import { dataSource, fetchCondition } from "./api";
import { strengthOf } from "./evidence";
import { FIXTURE_QUERY, kbDataSource, searchKb } from "./kb";
import type { KbChunkCard, KbSearchStatus } from "./kb-types";
import type { ConditionAnswer } from "./types";

/** 条件快照对应的固定问题（见 `scripts/capture_ops_fixtures.py`）。 */
export const FIXTURE_CONDITION_QUESTION = "过去 4 小时井筒和管道有什么异常";

export interface Citation {
  reference_id: string;
  /** 文献文件名。空串 = 无文件级血缘，不能当引用 */
  source: string;
  chunk_id: string;
  content: string;
  chars: number;
  /** 以公式为主：能证明"论文用了这个格式"，证明不了机理 */
  formulaHeavy: boolean;
  traceable: boolean;
}

export type RetrievalState = "ok" | KbSearchStatus | "failed" | "skipped";

export interface AskOutcome {
  answer: ConditionAnswer;
  citations: Citation[];
  retrieval: RetrievalState;
  /** 检索侧的一句话结论；空串表示没跑检索 */
  retrievalNote: string;
  /** 关于"这条回答是什么、不是什么"的说明，逐条显示在正文上方 */
  notes: string[];
  /** 回答来自固定快照 */
  demo: boolean;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  /**
   * `answer` = 数值问答；`retrieval` = 纯检索结果（演示用）；
   * `error` = 请求失败。三种在界面上长得不一样，因为它们的可信度不一样。
   */
  kind: "answer" | "retrieval" | "error";
  text: string;
  at: string;
  /** 来自固定快照，不是本次会话产生的 */
  demo?: boolean;
  pending?: boolean;
  facts?: string[];
  unresolved?: string[];
  unsourced?: string[];
  windowLabel?: string;
  citations?: Citation[];
  notes?: string[];
  retrievalNote?: string;
}

export const SUGGESTIONS = [
  "过去 4 小时井筒和管道有什么异常",
  "盖层完整性对 CO2 泄漏风险的影响",
  "注入压力离破裂压力还有多少余量",
] as const;

let seq = 0;

export function nextId(prefix = "m"): string {
  seq += 1;
  return `${prefix}-${Date.now().toString(36)}-${seq}`;
}

export function clockNow(): string {
  const d = new Date();
  return `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`;
}

/** 去掉空白与中英文标点后比较 —— 「盖层完整性对 CO₂ 泄漏风险的影响」与用户手打版本要能对上。 */
export function normalizeQuestion(text: string): string {
  return text
    .replace(/[\s\u3000]/g, "")
    .replace(/[，。？！、；：,.?!;:""''（）()【】[\]]/g, "")
    .replace(/CO₂/g, "CO2")
    .toLowerCase();
}

function toCitation(card: KbChunkCard): Citation {
  return {
    reference_id: card.reference_id,
    source: card.source ?? "",
    chunk_id: card.chunk_id ?? "",
    content: card.content ?? "",
    chars: card.chars ?? (card.content ?? "").length,
    formulaHeavy: card.text_quality?.low ?? false,
    traceable: strengthOf(card.source) === "traceable",
  };
}

/**
 * 问一句。
 *
 * 只有"数值回答"失败时抛异常 —— 检索失败不抛，因为一条没有引用的
 * 数值回答仍然是可用的（它的每个数字都能在事实块里回代）；
 * 把整条消息标成失败，会让人以为数字也不可信。
 */
export async function ask(question: string): Promise<AskOutcome> {
  const notes: string[] = [];
  const demo = dataSource === "fixture";

  if (demo) {
    notes.push(
      `演示模式：回答来自固定快照（问题「${FIXTURE_CONDITION_QUESTION}」），不随输入变化。`,
    );
    if (normalizeQuestion(question) !== normalizeQuestion(FIXTURE_CONDITION_QUESTION)) {
      notes.push("你问的是另一句话，所以这段回答并不是对它的回答。");
    }
  }

  const answer = await fetchCondition(question);

  let retrieval: RetrievalState = "skipped";
  let retrievalNote = "";
  let citations: Citation[] = [];

  try {
    const result = await searchKb(question);
    if (result.status !== "ok") {
      retrieval = result.status;
      retrievalNote =
        result.status === "unavailable"
          ? `检索没跑起来（${result.reason || "未知原因"}），本条回答没有引用。这不等于库里没有内容。`
          : `图谱里没有相关内容（${result.reason || "无命中"}），本条回答没有引用。`;
    } else if (
      kbDataSource === "fixture" &&
      normalizeQuestion(question) !== normalizeQuestion(FIXTURE_QUERY)
    ) {
      retrieval = "skipped";
      retrievalNote =
        `演示模式：检索快照固定为「${FIXTURE_QUERY}」的响应，与你问的这句话不是同一次检索 —— ` +
        "挂上去等于伪造出处，所以这里不挂引用。";
    } else {
      retrieval = "ok";
      citations = result.evidence.chunks.map(toCitation);
      retrievalNote =
        `引用来自同一次检索：${result.evidence.stats.chunks} 片原文 / ` +
        `${result.evidence.stats.sources} 篇文献。点角标看原文切片。`;
    }
  } catch (e: unknown) {
    retrieval = "failed";
    retrievalNote = `知识库检索失败（${e instanceof Error ? e.message : String(e)}），本条回答没有引用。`;
  }

  return { answer, citations, retrieval, retrievalNote, notes, demo };
}

/** 把一次 `AskOutcome` 折成一条助手消息。 */
export function toMessage(outcome: AskOutcome): ChatMessage {
  return {
    id: nextId(),
    role: "assistant",
    kind: "answer",
    text: outcome.answer.text,
    at: clockNow(),
    demo: outcome.demo,
    facts: outcome.answer.facts,
    unresolved: outcome.answer.unresolved,
    unsourced: outcome.answer.unsourced_numbers,
    windowLabel: outcome.answer.window?.label,
    citations: outcome.citations,
    notes: outcome.notes,
    retrievalNote: outcome.retrievalNote,
  };
}

/**
 * 演示模式的初始对话。
 *
 * 两段都是**真实抓下来的快照**，不是编的示例文案：
 * 1. 数值问答 —— 与条件快照的问句一致；
 * 2. 检索 —— 与检索快照的问句一致，引用角标可点，落点是那 12 片原文。
 *
 * 它们各自内部一致（问句对得上回答），但**彼此不是同一次会话**，
 * 所以两条都打"演示"标。把两段混成一段（拿检索证据去支撑数值回答）
 * 会造出一条在任何后端都不存在的对话。
 */
export async function demoTranscript(): Promise<ChatMessage[]> {
  const out: ChatMessage[] = [];

  try {
    const answer = await fetchCondition(FIXTURE_CONDITION_QUESTION);
    out.push({
      id: nextId("demo-q"),
      role: "user",
      kind: "answer",
      text: FIXTURE_CONDITION_QUESTION,
      at: clockNow(),
      demo: true,
    });
    out.push({
      id: nextId("demo-a"),
      role: "assistant",
      kind: "answer",
      text: answer.text,
      at: clockNow(),
      demo: true,
      facts: answer.facts,
      unresolved: answer.unresolved,
      unsourced: answer.unsourced_numbers,
      windowLabel: answer.window?.label,
      citations: [],
      notes: ["演示快照：这条回答只含数值，不含检索引用。"],
      retrievalNote:
        "条件快照本身不带出处 —— 数字的依据在上面「数字依据」里，每一条都能回代。",
    });
  } catch {
    /* 快照读不出来就不放这一段；不要用编的文案填上 */
  }

  try {
    const result = await searchKb(FIXTURE_QUERY);
    if (result.status === "ok") {
      const citations = result.evidence.chunks.map(toCitation);
      const traceable = citations.filter((c) => c.traceable).length;
      const formula = citations.filter((c) => c.formulaHeavy).length;
      out.push({
        id: nextId("demo-q"),
        role: "user",
        kind: "retrieval",
        text: FIXTURE_QUERY,
        at: clockNow(),
        demo: true,
      });
      out.push({
        id: nextId("demo-r"),
        role: "assistant",
        kind: "retrieval",
        text:
          `检索返回 ${citations.length} 片原文，来自 ${result.evidence.stats.sources} 篇文献；` +
          `其中 ${traceable} 片可溯源，${citations.length - traceable} 片无文件血缘，` +
          `${formula} 片以公式为主。`,
        at: clockNow(),
        demo: true,
        citations,
        retrievalNote: "这是一次检索结果，不是一段生成的结论 —— 点角标看每片原文。",
      });
    }
  } catch {
    /* 同上：检索快照缺失时不放这一段 */
  }

  return out;
}
