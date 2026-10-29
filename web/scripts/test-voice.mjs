import { test, before } from "node:test";
import assert from "node:assert/strict";
import { mkdirSync } from "node:fs";
import { join } from "node:path";
import { pathToFileURL } from "node:url";

import { build } from "esbuild";

/**
 * 语音输入 —— 守的是"它不会替人说话"。
 *
 * ## 这里守的是什么
 *
 * 语音识别的产物必须是**待确认的文字**，不是"一个问题"。识别错一个测点名
 * （`井口压力` → `景口压力`）之后，系统会认真地给出一个数值正确、对象错误
 * 的回答，而提问的人已经走了。
 *
 * 所以下面三组断言分别对应三条边界：
 *
 * 1. `mergeTranscript` **只追加、不覆盖** —— 用户可能已经在输入框里打了一半字。
 * 2. 不可用时**必须说清原因** —— 一个灰着又不说话的按钮，读者只能猜是
 *    "还没做"还是"坏了"。
 * 3. `no-speech` / `aborted` **不是错误** —— 把它画成红的，会让人以为
 *    麦克风坏了，而它只是这次没说话。
 */

const ENTRY = `
import * as V from "./src/lib/voice";
export { V };
`;

let V;

before(async () => {
  const dir = join(process.cwd(), "node_modules", ".cache", "ccus-voice");
  mkdirSync(dir, { recursive: true });
  const outfile = join(dir, "bundle.mjs");
  await build({
    stdin: { contents: ENTRY, resolveDir: process.cwd(), loader: "tsx" },
    outfile,
    bundle: true,
    format: "esm",
    platform: "node",
    logLevel: "error",
  });
  const mod = await import(pathToFileURL(outfile).href);
  V = mod.V;
});

/* ---------------------------------------------------------------- 支持判定 */

test("没有 window（Node / SSR）不抛错，判成不支持", () => {
  assert.equal(V.detectVoiceSupport(undefined), "unsupported");
});

test("浏览器没有语音接口：判成不支持，并说清换哪个浏览器", () => {
  assert.equal(V.detectVoiceSupport({}), "unsupported");
  const why = V.voiceBlockedReason("idle", "unsupported");
  assert.ok(why);
  assert.match(why, /Chrome|Edge/);
  // 不能只说"不支持" —— 读者要知道下一步
  assert.match(why, /键盘/);
});

test("webkitSpeechRecognition 也算支持（Chrome 只给带前缀的那个）", () => {
  const support = V.detectVoiceSupport({
    webkitSpeechRecognition: function () {},
    isSecureContext: true,
  });
  assert.equal(support, "available");
});

test("不在安全上下文里：判成 insecure，不是 unsupported", () => {
  // 两者的处理完全不同：unsupported 要换浏览器，insecure 要换地址。
  // 混成一句"语音不可用"，用户会去重装浏览器。
  const support = V.detectVoiceSupport({
    webkitSpeechRecognition: function () {},
    isSecureContext: false,
  });
  assert.equal(support, "insecure");
  const why = V.voiceBlockedReason("idle", "insecure");
  assert.match(why, /https|127\.0\.0\.1/);
  // 关键：浏览器不会弹授权框，所以不能让人去找权限开关
  assert.match(why, /不会弹授权/);
});

test("http + 127.0.0.1 视为安全上下文", () => {
  const support = V.detectVoiceSupport({
    webkitSpeechRecognition: function () {},
    location: { protocol: "http:", hostname: "127.0.0.1" },
  });
  assert.equal(support, "available");
});

test("http + 局域网 IP 不是安全上下文", () => {
  const support = V.detectVoiceSupport({
    webkitSpeechRecognition: function () {},
    location: { protocol: "http:", hostname: "192.168.1.20" },
  });
  assert.equal(support, "insecure");
});

test("可用时 blockedReason 为 null（按钮可点）", () => {
  assert.equal(V.voiceBlockedReason("idle", "available"), null);
});

test("授权等待中给出原因，而不是让按钮空着灰", () => {
  const why = V.voiceBlockedReason("requesting", "available");
  assert.ok(why);
  assert.match(why, /授权/);
});

/* ------------------------------------------------------------ 文本合并 */

test("空输入框：直接放进去", () => {
  assert.equal(V.mergeTranscript("", "井口压力多少"), "井口压力多少");
});

test("中文之间不加空格", () => {
  assert.equal(V.mergeTranscript("问一下", "井口压力"), "问一下井口压力");
});

test("英文/数字接缝加一个空格", () => {
  // `pressure42` 会被当成一个不存在的测点名
  assert.equal(V.mergeTranscript("pressure", "42"), "pressure 42");
});

test("已有文字末尾的空白先去掉再接", () => {
  assert.equal(V.mergeTranscript("井口压力   ", "多少"), "井口压力多少");
});

test("识别结果为空时原样返回已有文字，不清空", () => {
  // 这条是"不覆盖"的底线：一次没听到不该把用户打了一半的字抹掉
  assert.equal(V.mergeTranscript("已经打了一半", ""), "已经打了一半");
  assert.equal(V.mergeTranscript("已经打了一半", "   "), "已经打了一半");
});

test("只追加不覆盖：接完必须包含原来的内容", () => {
  const base = "过去 4 小时";
  const out = V.mergeTranscript(base, "井筒有什么异常");
  assert.ok(out.startsWith(base), "原有文字不能被替换掉");
  assert.ok(out.length > base.length);
});

/* ------------------------------------------------------------ 错误翻译 */

test("权限被拒：指向地址栏的锁，而不是再点一次", () => {
  const p = V.describeVoiceError("not-allowed");
  assert.equal(p.severe, true);
  assert.match(p.hint, /锁图标/);
  // 浏览器不会问第二次，所以说"再试一次"是错的
  assert.match(p.hint, /不会再问第二次/);
});

test("没听到声音不是错误", () => {
  const p = V.describeVoiceError("no-speech");
  assert.equal(p.severe, false);
  assert.ok(p.label);
  assert.ok(p.hint);
});

test("用户自己按停：一个字都不显示", () => {
  const p = V.describeVoiceError("aborted");
  assert.equal(p.severe, false);
  assert.equal(p.label, "");
});

test("连不上识别服务：提示改用键盘，别反复重试", () => {
  const p = V.describeVoiceError("network");
  assert.equal(p.severe, true);
  assert.match(p.hint, /键盘/);
});

test("未知错误也要把原始 code 带出来", () => {
  // 吞掉 code 的后果是现场无法复现：用户只记得"弹了个红字"
  const p = V.describeVoiceError("some-new-code");
  assert.equal(p.severe, true);
  assert.match(p.hint, /some-new-code/);
});

test("空 code 不崩，也不显示 undefined", () => {
  const p = V.describeVoiceError("");
  assert.ok(p.hint);
  assert.doesNotMatch(p.hint, /undefined/);
});

/* ------------------------------------------------------------ 按钮文案 */

test("按钮文案跟着状态走，四种都能分开读", () => {
  assert.equal(V.voiceButtonLabel("idle", "available"), "语音");
  assert.equal(V.voiceButtonLabel("listening", "available"), "停止");
  assert.equal(V.voiceButtonLabel("requesting", "available"), "授权中");
  assert.equal(V.voiceButtonLabel("idle", "unsupported"), "不支持语音");
  assert.equal(V.voiceButtonLabel("idle", "insecure"), "需要 https");
});

/* ------------------------------------------------------------ 文案底线 */

test("隐私说明必须说清音频会离开本机", () => {
  // 这个项目自称 local-first，语音是唯一例外。不说清就是骗人。
  assert.match(V.VOICE_PRIVACY_NOTE, /云端/);
  assert.match(V.VOICE_PRIVACY_NOTE, /送到厂商服务器/);
  assert.match(V.VOICE_PRIVACY_NOTE, /唯一例外/);
});

test("范围说明必须说清不发送", () => {
  assert.match(V.VOICE_SCOPE_NOTE, /不发送/);
  assert.match(V.VOICE_SCOPE_NOTE, /按 Enter/);
});

/* ══════════════════════════════════════════════════════════════
 * 本地识别这条路
 * ══════════════════════════════════════════════════════════════ */

/* -------------------------------------------------- 麦克风是另一个能力 */

test("麦克风能力与云端识别能力必须分开判定", () => {
  const secure = { isSecureContext: true, location: { protocol: "https:", hostname: "x.test" } };
  // Firefox：有 getUserMedia、没有 SpeechRecognition。合成一个判定的话，
  // 界面会说「不支持语音」，而本地识别其实能跑。
  const firefox = {
    ...secure,
    navigator: { mediaDevices: { getUserMedia: () => {} } },
  };
  assert.equal(V.detectMicSupport(firefox), "available");
  assert.equal(V.detectVoiceSupport(firefox), "unsupported");

  assert.equal(V.detectMicSupport(undefined), "unsupported");
  assert.equal(
    V.detectMicSupport({ isSecureContext: true, navigator: { mediaDevices: {} } }),
    "unsupported",
  );
});

test("非安全上下文下麦克风不可用，且优先报这个", () => {
  const insecure = {
    isSecureContext: false,
    location: { protocol: "http:", hostname: "10.0.0.5" },
    navigator: { mediaDevices: { getUserMedia: () => {} } },
  };
  assert.equal(V.detectMicSupport(insecure), "insecure");
  // localhost / 127.0.0.1 是安全上下文
  assert.equal(
    V.detectMicSupport({
      location: { protocol: "http:", hostname: "127.0.0.1" },
      navigator: { mediaDevices: { getUserMedia: () => {} } },
    }),
    "available",
  );
});

/* -------------------------------------------------- 走哪条路 */

const LOCAL_OK = {
  available: true,
  model_ready: true,
  dependency_ready: true,
  reason: "",
  hint: "",
  download: "python scripts/fetch_asr_model.py",
  model: "m",
  max_seconds: 60,
};

const NO_MODEL = {
  ...LOCAL_OK,
  available: false,
  model_ready: false,
  reason: "模型目录不存在",
  hint: "跑 fetch_asr_model.py",
};

test("本地就绪时走本地，且隐私说明换成不出本机", () => {
  const r = V.pickVoiceRoute(LOCAL_OK, "available", "available");
  assert.equal(r.engine, "local");
  assert.equal(r.degraded, "");
  assert.match(r.note, /不出本机/);
  // 这一条是整件事的重点：走本地时不该再出现「送到厂商服务器」。
  assert.doesNotMatch(r.note, /送到厂商服务器/);
  assert.match(r.badge, /本地/);
});

test("本地没就绪时回落云端，但必须写明回落了", () => {
  const r = V.pickVoiceRoute(NO_MODEL, "available", "available");
  assert.equal(r.engine, "browser");
  assert.match(r.degraded, /模型目录不存在/);
  // 隐私说明必须跟着换成云端那一版 —— 悄悄回落等于让人以为录音没出网。
  assert.match(r.note, /送到厂商服务器/);
  assert.match(r.badge, /云端/);
});

test("后端连不上时回落云端，原因要指向后端而不是模型", () => {
  const r = V.pickVoiceRoute(null, "available", "available");
  assert.equal(r.engine, "browser");
  assert.match(r.degraded, /后端/);
});

test("两条路都不通时必须同时说清两件事", () => {
  // Firefox + 后端没起来：只说「本地模型没下载」会把人引到一个修不好
  // 问题的方向 —— 他下完模型，浏览器照样没有云端识别。
  const r = V.pickVoiceRoute(NO_MODEL, "available", "unsupported");
  assert.equal(r.engine, "none");
  assert.match(r.blocked, /模型目录不存在/);
  assert.match(r.blocked, /云端识别/);
});

test("非安全上下文优先于本地模型缺失", () => {
  // 模型齐全也不能用，因为 getUserMedia 会被直接拒。先说 https，
  // 否则人会去下模型，然后麦克风照样不弹授权框。
  const r = V.pickVoiceRoute(LOCAL_OK, "insecure", "available");
  assert.equal(r.engine, "none");
  assert.match(r.blocked, /https/);
  assert.doesNotMatch(r.blocked, /模型/);
});

test("没有录音接口时两条路都不通", () => {
  const r = V.pickVoiceRoute(LOCAL_OK, "unsupported", "unsupported");
  assert.equal(r.engine, "none");
  assert.match(r.blocked, /getUserMedia/);
});

/* -------------------------------------------------- 按钮文案与可点性 */

test("本地路径多一个识别中的状态", () => {
  const local = V.pickVoiceRoute(LOCAL_OK, "available", "available");
  assert.equal(V.voiceRouteLabel("idle", local), "语音");
  assert.equal(V.voiceRouteLabel("listening", local), "停止");
  assert.equal(V.voiceRouteLabel("requesting", local), "授权中");
  assert.equal(V.voiceRouteLabel("transcribing", local), "识别中");
});

test("不可点时按钮上放短词，长原因放外面", () => {
  const r = V.pickVoiceRoute(LOCAL_OK, "insecure", "available");
  const label = V.voiceRouteLabel("idle", r);
  assert.equal(label, "需要 https");
  assert.ok(V.voiceRouteBlockedReason("idle", r).length > 20);
  // 可以点时返回 null，不是空字符串 —— 空字符串是真值，会走进「不可点」分支。
  const ok = V.pickVoiceRoute(LOCAL_OK, "available", "available");
  assert.equal(V.voiceRouteBlockedReason("idle", ok), null);
});

/* -------------------------------------------------- 本地路径的错误 */

test("模型缺失要给出可以直接跑的命令", () => {
  const p = V.describeLocalError("model-missing", { download: "python scripts/fetch_asr_model.py" });
  assert.match(p.label, /模型/);
  assert.match(p.hint, /fetch_asr_model\.py/);
  // 不是「错误」：这不是故障，是还没做那一步。
  assert.equal(p.severe, false);
});

test("依赖没装与模型没下是两句话", () => {
  const dep = V.describeLocalError("dependency-missing");
  const mod = V.describeLocalError("model-missing");
  assert.match(dep.hint, /pip install sherpa-onnx/);
  assert.notEqual(dep.label, mod.label);
});

test("后端连不上要指向后端，不是麦克风", () => {
  const p = V.describeLocalError("no-backend");
  assert.equal(p.severe, true);
  assert.match(p.hint, /后端/);
  assert.doesNotMatch(p.hint, /麦克风/);
});

test("七个本地错误各有各的说法，没有兜底的重复", () => {
  const kinds = [
    "no-backend",
    "model-missing",
    "dependency-missing",
    "mic-denied",
    "decode-failed",
    "too-long",
    "server-error",
  ];
  const labels = kinds.map((k) => V.describeLocalError(k).label);
  assert.equal(new Set(labels).size, kinds.length);
  for (const k of kinds) {
    assert.ok(V.describeLocalError(k).hint.length > 0, k + " 没有 hint");
  }
});

/* -------------------------------------------------- 计时 */

test("计时显示", () => {
  assert.equal(V.formatClock(0), "0:00");
  assert.equal(V.formatClock(7_400), "0:07");
  assert.equal(V.formatClock(73_400), "1:13");
  assert.equal(V.formatClock(-5), "0:00");
});

/* -------------------------------------------------- 文案底线（本地） */

test("本地说明必须说清不出本机", () => {
  assert.match(V.VOICE_LOCAL_NOTE, /不出本机/);
  assert.match(V.VOICE_LOCAL_NOTE, /127\.0\.0\.1/);
  assert.match(V.VOICE_LOCAL_NOTE, /sherpa-onnx/);
});

test("云端说明仍然要说清音频出网", () => {
  // 加了本地这条路之后，这句话不能因为「现在有本地了」就被删掉 ——
  // 回落时它就是唯一的告知。
  assert.match(V.VOICE_PRIVACY_NOTE, /送到厂商服务器/);
});
