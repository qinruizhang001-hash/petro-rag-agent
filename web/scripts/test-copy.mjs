import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

/**
 * 文案守卫。
 *
 * 这里断言的是**源码文本**，不是渲染结果 —— 故意的。这些句子的作用
 * 是"不要让读者把示意值当成实测"，而它们被删掉的方式从来不是逻辑变化，
 * 是一次"这句太长了，简化一下"的重构。逻辑测试抓不到，文本断言可以。
 *
 * 断言"说清了什么"，不断言"用了哪个词"：`不能作为井位设计` 里的
 * "不能"与"井位设计"分开查，改语序不会红，删掉限制会红。
 */

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const read = (p) => readFileSync(join(root, p), "utf8");

test("剖面图例：三种渲染状态都有标签", () => {
  const geo = read("src/components/GeoSection.tsx");
  assert.ok(geo.includes("视图 3D"), "3D 状态没有标签，读者不知道自己在看哪一张");
  assert.ok(geo.includes("视图 SVG 降级"), "降级状态没有标签，降级会静默发生");
});

test("剖面图例：羽流与分层都标注为示意，且引用同一份来源常量", () => {
  const geo = read("src/components/GeoSection.tsx");
  // 必须是引用常量，不能各写一句 —— 两处各写一份，迟早在某次改动后不一致
  assert.ok(geo.includes("PLUME_SOURCE"), "羽流没有引用 PLUME_SOURCE");
  assert.ok(geo.includes("STRATA_SOURCE"), "分层没有引用 STRATA_SOURCE");

  const section = read("src/lib/section.ts");
  assert.match(section, /PLUME_SOURCE = "示意/, "羽流来源不再声明为示意值");
  assert.match(section, /STRATA_SOURCE = "示意/, "分层来源不再声明为示意值");
});

test("剖面图例：限制说明写清了不能拿来做什么", () => {
  const geo = read("src/components/GeoSection.tsx");
  for (const phrase of ["井位设计", "封存容量", "压力预测"]) {
    assert.ok(
      geo.includes(phrase),
      `限制说明缺少「${phrase}」—— 少了它，这张图会被截图放进汇报材料`,
    );
  }
});

test("引用角标：无文件血缘时不渲染成可点按钮", () => {
  const cit = read("src/components/dialog/CitationText.tsx");
  assert.ok(
    cit.includes("if (!c!.source)"),
    "无血缘的引用又变成可点按钮了 —— 点了没反应比不能点更像故障",
  );
});
