#!/usr/bin/env node
/**
 * 引用角标解析与证据强度的单元测试。
 *
 * ## 为什么这两块要测
 *
 * `splitCitations` 的失败是**静默**的：正文照常显示，只是角标不能点，
 * 或者反过来把数学区间 `[0,1]` 变成一个"未解析引用"的假告警。
 * 两种都不会抛异常，也都不会出现在控制台里 —— 只有测试能发现。
 *
 * `summarize` 的两个数字决定界面上写"全部可回溯"还是"有 3 片无法回溯"。
 * 它数错时，界面上那句结论读起来同样斩钉截铁。
 */

import assert from "node:assert/strict";
import { test } from "node:test";

import { groupCitations, splitCitations, unresolvedIds } from "../src/lib/citations.ts";
import { strengthOf, summarize } from "../src/lib/evidence.ts";

const texts = (segments) => segments.filter((s) => s.kind === "text").map((s) => s.text);
const cites = (segments) =>
  segments.filter((s) => s.kind === "cite").map((s) => s.ids);

test("单个角标被拆出来，前后的文字都保留", () => {
  const segs = splitCitations("盖层完整性是关键[1]。");
  assert.deepEqual(cites(segs), [["1"]]);
  assert.deepEqual(texts(segs), ["盖层完整性是关键", "。"]);
});

test("多个编号一起的角标拆成一个多 id 片段", () => {
  const segs = splitCitations("见文献[1,2]与[3]");
  assert.deepEqual(cites(segs), [["1", "2"], ["3"]]);
});

test("全角逗号与顿号也算分隔符", () => {
  assert.deepEqual(cites(splitCitations("见[1，2、3]")), [["1", "2", "3"]]);
});

test("四位数不当角标：它更可能是年份", () => {
  const segs = splitCitations("2026 年的报告[2026]");
  assert.deepEqual(cites(segs), []);
  assert.deepEqual(texts(segs), ["2026 年的报告[2026]"]);
});

test("0 开头的编号当区间，不当角标", () => {
  // 数学区间 [0,1] 与引用 [0,1] 长得一样；引用号从 1 开始，所以判给区间
  assert.deepEqual(cites(splitCitations("在区间[0,1]上连续")), []);
  assert.deepEqual(cites(splitCitations("见[0]")), []);
});

test("markdown 链接的方括号不是角标", () => {
  assert.deepEqual(cites(splitCitations("见[1](https://example.org)")), []);
});

test("非数字的方括号原样保留", () => {
  assert.deepEqual(cites(splitCitations("见[注]与[a,b]")), []);
});

test("没有标记时只有一段文字，且内容一字不改", () => {
  const src = "压力 115.4 bar，超告警线 1.6 bar。";
  const segs = splitCitations(src);
  assert.deepEqual(cites(segs), []);
  assert.equal(texts(segs).join(""), src);
});

test("空串不产生空片段", () => {
  assert.deepEqual(splitCitations(""), []);
});

test("unresolvedIds 只报证据里没有的编号，并去重保序", () => {
  const segs = splitCitations("见[1][9]与[2,9]");
  assert.deepEqual(unresolvedIds(segs, new Set(["1", "2"])), ["9"]);
  assert.deepEqual(unresolvedIds(segs, new Set(["1", "2", "9"])), []);
});

test("有文件名的切片是可溯源，空串与 undefined 都是无血缘", () => {
  assert.equal(strengthOf("ccus_auto_02.pdf"), "traceable");
  assert.equal(strengthOf("  ccus3.pdf  "), "traceable");
  assert.equal(strengthOf(""), "unattributed");
  assert.equal(strengthOf("   "), "unattributed");
  assert.equal(strengthOf(undefined), "unattributed");
  assert.equal(strengthOf(null), "unattributed");
});

test("summarize 逐片重算，不信后端的 stats", () => {
  const evidence = {
    chunks: [
      { reference_id: "1", chunk_id: "a", source: "p1.pdf", chars: 10, content: "", truncated: false, text_quality: { readable_ratio: 1, math_ratio: 0, low: false } },
      { reference_id: "2", chunk_id: "b", source: "", chars: 10, content: "", truncated: false, text_quality: { readable_ratio: 0.4, math_ratio: 0.5, low: true } },
      { reference_id: "3", chunk_id: "c", source: "p2.pdf", chars: 10, content: "", truncated: false, text_quality: { readable_ratio: 1, math_ratio: 0, low: false } },
    ],
    entities: [],
    relationships: [],
    references: [],
    // 后端说 3 片全部可溯源、0 片无出处 —— 与逐片事实不符
    stats: { chunks: 3, entities: 27, relationships: 25, sources: 2, unsourced: 0, traceable: 3, low_quality_chunks: 0 },
  };
  const s = summarize(evidence);
  assert.equal(s.chunks, 3);
  assert.equal(s.traceable, 2);
  assert.equal(s.unattributed, 1);
  assert.equal(s.formulaHeavy, 1);
  assert.equal(s.sources, 2);
});

test("summarize 对缺失字段不炸，也不把空证据说成有证据", () => {
  const s = summarize({ chunks: [], entities: [], relationships: [], references: [], stats: {} });
  assert.equal(s.chunks, 0);
  assert.equal(s.traceable, 0);
  assert.equal(s.unattributed, 0);
  assert.equal(s.formulaHeavy, 0);
  assert.equal(s.entities, 0);
  assert.equal(summarize(null).chunks, 0);
});

test("groupCitations 按 reference_id 合并，保留首次出现顺序", () => {
  const mk = (id, source, formulaHeavy = false) => ({
    reference_id: id,
    source,
    traceable: Boolean(source),
    formulaHeavy,
  });
  const groups = groupCitations([
    mk("1", "a.pdf"),
    mk("3", "b.pdf"),
    mk("2", "c.pdf"),
    mk("1", "a.pdf", true),
    mk("3", "b.pdf"),
  ]);

  assert.deepEqual(
    groups.map((g) => [g.reference_id, g.chunks.length]),
    [["1", 2], ["3", 2], ["2", 1]],
  );
  // 只有该来源的**全部**切片都以公式为主，才把整组标成公式为主
  assert.equal(groups[0].formulaHeavy, false);
  assert.equal(groups[1].formulaHeavy, false);
  assert.equal(groups[2].formulaHeavy, false);
  assert.equal(groupCitations([mk("1", "", true), mk("1", "", true)])[0].formulaHeavy, true);
  // 无文件名的来源整组都是不可溯源
  assert.equal(groupCitations([mk("9", "")])[0].traceable, false);
  assert.deepEqual(groupCitations([]), []);
});
