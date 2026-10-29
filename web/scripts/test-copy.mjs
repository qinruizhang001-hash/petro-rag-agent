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

test("测点表的视图由 mode 收口，不是只把按钮藏起来", () => {
  /*
   * 只藏按钮会留下一个**回不去**的状态：工程师模式切到表格 → 切到指挥舱
   * → 表格还在，而切回去的按钮已经不在了。这不是渲染错，是状态没收口，
   * 静态渲染测试抓不到（它不点按钮），所以守源码。
   */
  const rail = read("src/components/LeftRail.tsx");
  assert.match(
    rail,
    /const shownView = mode === "engineer" \? view : "cards"/,
    "少了这行收口，指挥舱会停在表格上",
  );
  assert.match(rail, /shownView === "table" \?/, "渲染分支必须看 shownView");
  assert.doesNotMatch(
    rail,
    /^\s*\{view === "table" \?/m,
    "渲染分支直接看 view 时，mode 只影响按钮可见性，不影响显示什么",
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

test("?source 必须同时管住 ops 与知识库", () => {
  // 只改一边的后果：底栏写 live、知识库还是快照 —— 同一屏两个数据源，
  // 而两边各自都"看起来正常"。所以两个模块都必须走同一个入口。
  for (const f of ["src/lib/api.ts", "src/lib/kb.ts"]) {
    const src = read(f);
    assert.ok(src.includes("choiceFromLocation("), `${f} 必须走 lib/source.ts 的同一套判定`);
    assert.ok(
      !src.includes('typeof window === "undefined"'),
      `${f} 不许自己判断 window（那份判断只该有一处）`,
    );
  }
});

test("面板标题行能换行，标题不被压断", () => {
  /*
   * 默认的 flex 让标题和 aside 一起缩，两个都会缩到内容宽度以下 ——
   * 工程师模式给标题行多塞了「卡片 / 表格」之后，「报警与关键指标」被压成
   * 「报警与关键指 / 标」，旁边的计数被压成竖排单字。两个元素都在、都读不
   * 出来，而控制台里没有任何报错。
   *
   * 静态渲染测试看不出这件事（它只比较文本），所以用源码守卫锁形状。
   */
  const panel = read("src/components/Panel.tsx");
  assert.match(panel, /header className="[^"]*flex-wrap/, "标题行缺 flex-wrap");
  assert.match(panel, /<h2[^>]*whitespace-nowrap/, "标题缺 whitespace-nowrap，会被压断");

  // 计数是这一栏的结论，不许被按钮挤掉；挤不下时换行的是按钮。
  const rail = read("src/components/LeftRail.tsx");
  assert.match(rail, /tnum whitespace-nowrap/, "左栏计数缺 whitespace-nowrap");
  assert.match(rail, /flex flex-wrap items-center justify-end/, "左栏 aside 缺 flex-wrap");
});

test("live 模式：状态栏必须说清输入是回放而不是实时", () => {
  // 这句文案现在住在 `lib/source.ts`（数据源判定与措辞的唯一出处）。
  const src = read("src/lib/source.ts");
  assert.ok(
    src.includes("读数采样回放"),
    "live 角标只写「live」会让读者以为读数是实时采集的",
  );
  // 而且 StatusBar 不许自己再拼一份：两处各写一句，改一处就会露出另一处。
  const bar = read("src/components/StatusBar.tsx");
  assert.ok(bar.includes("sourceLabel("), "底栏必须用 sourceLabel() 出这行字");
  assert.ok(!bar.includes("读数采样回放"), "底栏不许内联这句文案（会与 source.ts 分家）");
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

test("语音：状态环仍然不可交互", () => {
  /*
   * `VoiceRing` 的注释写死了"环上没有任何可点的元素"，理由是：把状态环
   * 做成按钮，会有人对着它说话然后以为说完了就执行了。
   *
   * 接了语音输入之后，这条最容易被破坏 —— 麦克风图标画在环上、或者给环
   * 挂一个 onClick，都是"顺手"就能做出来的事，而且**渲染出来更好看**。
   * 所以用源码守卫把它钉住：可以点的按钮在 `VoiceButton.tsx` 里，
   * 那个环永远只是显示。
   */
  const ring = read("src/components/VoiceRing.tsx");
  assert.ok(!/onClick|onKeyDown|<button|role="button"|tabIndex/.test(ring), "VoiceRing 不许可交互");
  // 麦克风按钮必须是一个独立的组件文件，不是一个挂在环上的回调
  assert.ok(read("src/components/dialog/VoiceButton.tsx").includes("<button"));
});

test("语音：识别结果不许直接发送", () => {
  /*
   * `lib/voice.ts` 是纯文本处理，它不该知道"发送"这件事存在。
   * 一旦它 import 了 `askStream` 或拿到 `send`，就出现一条"说一句话就提交
   * 一个问题"的路径 —— 而识别错一个测点名，系统会给出一个数值正确、
   * 对象错误的回答。
   */
  const voice = read("src/lib/voice.ts");
  assert.ok(!/askStream|send\(/.test(voice), "语音模块不许碰发送");

  // 接入点必须走 mergeTranscript → setInput，而不是 send(...)
  const panel = read("src/components/dialog/DialogPanel.tsx");
  assert.match(
    panel,
    /onCommit=\{\(text\) => setInput\(\(prev\) => mergeTranscript\(prev, text\)\)\}/,
    "识别结果必须并进输入框（不覆盖），不能直接发出去",
  );
});

test("语音：隐私说明写清了音频会离开本机", () => {
  /*
   * 这个项目自称 local-first，而浏览器自带的语音识别是云端识别 ——
   * 它是整个系统里唯一一个把现场录音送出本机的功能。
   *
   * 这条断言守的是"说了"，不是"说了哪几个字"：换措辞不会红，
   * 把这段说明删掉会红。
   */
  const voice = read("src/lib/voice.ts");
  assert.match(voice, /云端识别/);
  assert.match(voice, /唯一例外/);
  assert.match(voice, /涉密/);
});

test("语音：隐私说明跟着引擎走，不能写死一条", () => {
  /*
   * 加了本地识别之后，「音频去哪了」有两个答案。把其中一条写死在
   * 按钮的 title 里，另一条路就会显示错的那一句 —— 而这一句恰好是
   * 用户唯一能看到的告知。
   *
   * 所以守的是结构：title 里必须出现 `route.note`，而 `route.note` 由
   * `pickVoiceRoute` 按引擎给出（那部分由 test-voice.mjs 覆盖）。
   */
  const btn = read("src/components/dialog/VoiceButton.tsx");
  assert.match(btn, /route\.note/, "隐私说明必须来自路由，不能写死");
  assert.match(btn, /route\.degraded/, "回落原因必须显示，不能悄悄回落");
  // 角标用两种形状区分，不只靠颜色
  assert.match(btn, /border-dashed/, "云端角标要与本地角标形状不同");
});

test("语音：本地识别模块同样不许碰发送", () => {
  const asr = read("src/lib/asr.ts");
  assert.ok(!/askStream|send\(/.test(asr), "asr.ts 只负责录音与上传");
});

test("语音：录音格式的契约两端都在", () => {
  /*
   * 服务端只认 16 kHz（`np.frombuffer(dtype="<f4")`），前端必须自报
   * 采样率，服务端必须校验。任何一端单独改都会让识别结果变成乱码 ——
   * 而乱码不报错。
   */
  const asr = read("src/lib/asr.ts");
  assert.match(asr, /X-Sample-Rate/);
  assert.match(asr, /TARGET_RATE = 16000/);
});

test("语音：asr 模块不许进首屏", () => {
  /*
   * `lib/asr.ts` 一被首屏静态引用，录音管线（含 AudioContext 相关代码）
   * 就会进主分块。它只在对话面板里用，而对话面板是 lazy 的。
   *
   * 只查 dialog 目录之外的引用 —— 那是唯一应该 import 它的地方。
   */
  const files = [
    "src/App.tsx",
    "src/components/LeftRail.tsx",
    "src/components/StatusBar.tsx",
    "src/components/RightRail.tsx",
  ];
  for (const f of files) {
    let src;
    try {
      src = read(f);
    } catch {
      continue;
    }
    assert.ok(!/lib\/asr/.test(src), `${f} 不该静态引用 lib/asr`);
  }
});
