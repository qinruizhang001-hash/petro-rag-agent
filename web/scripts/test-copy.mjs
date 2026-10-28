import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

// 唯一一处运行期 import：`GEOMETRY_PARAMS` 的 `origin` 是数据而不是源码片段，
// 只有真读出来才知道它是不是空串（文本断言会漏掉 `origin: ""` 之外的所有空法，
// 比如 `origin: STRATA_SOURCE` 而那个常量被改成空）。
import { GEOMETRY_PARAMS } from "../src/lib/section";

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

test("剖面几何：每个参数都写得出出处，且没有重复 key", () => {
  // 这条对应 `section.ts` 里那句「空字符串是不允许的（有测试锁着）」。
  // 少了它，加一个新参数时最容易的写法就是 `origin: ""` —— 编译通过、
  // 界面照常渲染，只是那个数悄悄变成「无需解释」。
  assert.ok(GEOMETRY_PARAMS.length > 0, "几何参数表空了，图例会显示「0 项」");
  const seen = new Set();
  for (const p of GEOMETRY_PARAMS) {
    assert.ok(p.key, "参数没有 key");
    assert.ok(!seen.has(p.key), `参数 key 重复：${p.key}`);
    seen.add(p.key);
    assert.ok(p.label, `${p.key} 没有 label`);
    assert.ok(p.value, `${p.key} 没有 value`);
    assert.ok(
      typeof p.origin === "string" && p.origin.trim().length >= 4,
      `${p.key} 的 origin 是空的或太短 —— 说不清出处的参数不该出现在图上`,
    );
  }
  // 至少要有一个参数承认自己是占位值。全部声称有出处，比全部空着更危险：
  // 读者会照着去引用。
  assert.ok(
    GEOMETRY_PARAMS.some((p) => p.origin.includes("占位") || p.origin.includes("示意")),
    "没有任何参数标注为占位/示意 —— 这五个数目前全部不是实测值",
  );

  // 数据表存在 ≠ 界面用上了。删掉渲染那一段，上面所有断言依然全绿，
  // 而图例退回一句笼统的「羽流 示意」，读者又开始猜哪个数是实测的。
  const geo = read("src/components/GeoSection.tsx");
  assert.ok(
    geo.includes("GEOMETRY_PARAMS"),
    "逐参数来源列表没有渲染 —— 图例只剩一句笼统的「示意」",
  );
});

test("引用角标：无文件血缘时不渲染成可点按钮", () => {
  const cit = read("src/components/dialog/CitationText.tsx");
  assert.ok(
    cit.includes("if (!c!.source)"),
    "无血缘的引用又变成可点按钮了 —— 点了没反应比不能点更像故障",
  );
});

test("live 模式：告警请求必须自带读数，不能抛错把整屏变成一行报错", () => {
  const api = read("src/lib/api.ts");
  // 这条守卫对应一次真实的白屏：`fetchAlarms()` 曾在 live 下直接
  // `throw new Error("live 模式下必须提供 readings")`，而 App 的
  // `Promise.all` 一挂，左栏 / 剖面 / 右栏全部消失，只剩一行
  // 「数据加载失败」。fixture 模式下永远复现不了。
  assert.ok(
    api.includes("sampledReadings"),
    "live 分支没有读数来源了 —— VITE_OPS_SOURCE=live 会重新变成白屏",
  );
  assert.ok(
    !api.includes("必须提供 readings"),
    "又回到了「live 模式下必须提供 readings」：这句话没有任何调用方能满足",
  );
});

test("live 模式：状态栏必须说清输入是回放而不是实时", () => {
  const bar = read("src/components/StatusBar.tsx");
  assert.ok(
    bar.includes("读数采样回放"),
    "live 角标只写「live」会让读者以为读数是实时采集的",
  );
});

test("live 模式：读数采样文件存在且非空", () => {
  // 删掉它不会让 `tsc` 红（动态 import 只在运行期解析），
  // 只会让 live 模式在打开时炸在一个 `Cannot find module` 上。
  const raw = JSON.parse(read("src/fixtures/readings.json"));
  assert.ok(Array.isArray(raw), "readings.json 必须是读数数组");
  assert.ok(raw.length > 0, "readings.json 是空的：后端会回 400，界面还是白屏");
  for (const key of ["sensor", "value", "ts"]) {
    assert.ok(key in raw[0], `读数缺少 ${key} 字段`);
  }
});
