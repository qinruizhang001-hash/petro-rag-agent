/**
 * 语音输入 —— 只做一件事：把说的话变成**输入框里的字**。
 *
 * ## 它为什么不自动发送
 *
 * 识别会错，而且错得很有特点：`井口压力` 听成 `景口压力`、`捕集率` 听成
 * `补给率`、`42` 听成 `4.2`。这个系统里"问错一个测点"不是小事 —— 它会
 * 认真地给出一个**数值正确、对象错误**的回答，而提问的人已经走了。
 *
 * 所以语音的产物永远是"待确认的文字"，不是"一个问题"。落笔必须是人手
 * 按 Enter 或点发送。这与 `VoiceBoundary` 里那条约束同源：语音最多做到
 * 打开确认面板，落笔要人。
 *
 * ## 它为什么不碰 `VoiceRing`
 *
 * `VoiceRing` 的注释写死了"环上没有任何可点的元素"，理由也写在那儿：把
 * 状态环做成按钮，会有人对着它说话然后以为说完了就执行了。麦克风按钮
 * 因此是**输入框旁边的一个独立按钮**，与那个环没有关系。
 *
 * ## 音频去哪了
 *
 * 有两条路，界面上必须写清当前走的是哪一条：
 *
 * - **本地**（`/api/asr` → sherpa-onnx）：录音在 127.0.0.1 上转成文字，不出本机。
 * - **云端**（浏览器 `SpeechRecognition`）：Chrome / Edge 会把它送到厂商服务器。
 *
 * 本地这条路依赖两样东西：后端装了 `sherpa-onnx`、模型下载到了
 * `models/asr/`。缺任何一样就**回落到云端**，并且按钮旁边要明说回落了 ——
 * 悄悄回落等于让人以为自己在用本地识别，而实际录音已经出网了。
 *
 * 本地这条路还有个副产品：它只要 `getUserMedia`，不要
 * `SpeechRecognition`。所以 Firefox 上也能用语音（走本地），
 * 「换 Chrome」这句话只在云端那条路上才成立。
 */

export type VoiceSupport =
  /** 浏览器没有 SpeechRecognition（Firefox 全系、部分 Safari）。 */
  | "unsupported"
  /** 有接口，但页面不在安全上下文里 —— 麦克风会被直接拒绝且没有提示。 */
  | "insecure"
  | "available";

export type VoicePhase =
  | "idle"
  | "requesting"
  | "listening"
  /** 本地路径专有：已经录完，等后端把采样点转成文字。 */
  | "transcribing"
  | "error";

export type VoiceErrorKind =
  | "not-allowed"
  | "service-not-allowed"
  | "no-speech"
  | "audio-capture"
  | "network"
  | "aborted"
  | "unknown";

export interface VoiceProblem {
  kind: VoiceErrorKind;
  /** 一句话说清发生了什么。 */
  label: string;
  /** 下一步做什么。空字符串 = 不需要动作（用户自己停的、或者只是没说话）。 */
  hint: string;
  /** 是否该画成错误。`no-speech` / `aborted` 不是错误。 */
  severe: boolean;
}

/**
 * 云端识别（浏览器自带）的隐私说明。
 *
 * 只在**真的走到这条路**时显示。无条件显示会变成一句没人再读的免责声明，
 * 而它的作用是让人在按下按钮之前知道自己放弃了什么。
 */
export const VOICE_PRIVACY_NOTE =
  "当前走的是浏览器自带的云端识别：音频片段会送到厂商服务器，" +
  "不是在这台机器上转成文字的。这个系统的其它部分都在本机跑，" +
  "语音是唯一例外。涉密现场不要用；把本地识别跑起来就不用看这条了。";

/** 本地识别（sherpa-onnx）的隐私说明。 */
export const VOICE_LOCAL_NOTE =
  "当前走的是本地识别：录音只发到 127.0.0.1 的后端，由 sherpa-onnx 转成文字，" +
  "不出本机、不联网。";

/** 语音能做/不能做的边界，与 `VoiceBoundary` 表格同一套说法。 */
export const VOICE_SCOPE_NOTE =
  "语音只把话变成输入框里的字，不发送、不执行。发问要你自己按 Enter —— " +
  "识别错一个测点名，系统会给出一个数值正确、对象错误的回答。";

/**
 * 这个浏览器能不能用。
 *
 * 传入 `window` 而不是直接读全局：Node 里跑测试时没有 window，而这个函数
 * 是唯一需要环境的地方，注入进来才测得动。
 */
export function detectVoiceSupport(
  win:
    | {
        SpeechRecognition?: unknown;
        webkitSpeechRecognition?: unknown;
        isSecureContext?: boolean;
        location?: { protocol?: string; hostname?: string };
      }
    | undefined,
): VoiceSupport {
  if (!win) return "unsupported";
  const Ctor = win.SpeechRecognition ?? win.webkitSpeechRecognition;
  if (typeof Ctor !== "function") return "unsupported";

  /*
   * 安全上下文检查必须在**按按钮之前**。
   *
   * 非 https 且非 localhost 时，`getUserMedia` 与语音识别都会被浏览器
   * 直接拒绝，而且 `SpeechRecognition` 报的是一个笼统的 `not-allowed` ——
   * 于是界面会说"你拒绝了麦克风权限"，而用户从没被问过。把"环境不允许"
   * 说成"你拒绝了"，是让人去翻浏览器设置找一个根本不存在的开关。
   *
   * `isSecureContext` 优先，它才是浏览器真正的判据；拿不到时按 protocol 猜。
   */
  if (win.isSecureContext === false) return "insecure";
  const proto = win.location?.protocol;
  const host = win.location?.hostname;
  if (proto && proto !== "https:" && host !== "localhost" && host !== "127.0.0.1") {
    return "insecure";
  }
  return "available";
}

/** 把识别结果并进已有文字，**绝不覆盖**。 */
export function mergeTranscript(base: string, spoken: string): string {
  const add = spoken.trim();
  if (!add) return base;
  const head = base.replace(/\s+$/, "");
  if (!head) return add;

  /*
   * 中文之间不加空格，英文/数字之间加一个。
   *
   * 判据只看"接缝两侧是不是 ASCII 字母数字"：`井口压力` + `多少` 直接接；
   * `pressure` + `42` 需要空格，否则变成 `pressure42`。用码位判断而不是
   * 语言判断 —— 后者在混合句子里必错，而且错得很随机。
   */
  const tailIsAscii = /[A-Za-z0-9]$/.test(head);
  const headIsAscii = /^[A-Za-z0-9]/.test(add);
  const sep = tailIsAscii && headIsAscii ? " " : "";
  return `${head}${sep}${add}`;
}

/**
 * 把 `SpeechRecognitionErrorEvent.error` 翻成人话。
 *
 * 六个分支里只有三个该画成红的。`no-speech`（没听到）和 `aborted`
 * （自己按停）是正常操作，画成错误会让人以为麦克风坏了 —— 而它只是
 * 这次没说话。
 */
export function describeVoiceError(code: string): VoiceProblem {
  switch (code) {
    case "not-allowed":
    case "service-not-allowed":
      return {
        kind: code,
        label: "麦克风权限被拒绝",
        hint:
          "点地址栏左边的锁图标 → 网站设置 → 麦克风改成「允许」，然后刷新页面。" +
          "浏览器不会再问第二次，所以只能从那儿改。",
        severe: true,
      };
    case "no-speech":
      return {
        kind: "no-speech",
        label: "没听到声音",
        hint: "再说一次，或者检查麦克风有没有被别的程序占着。",
        severe: false,
      };
    case "audio-capture":
      return {
        kind: "audio-capture",
        label: "找不到麦克风",
        hint: "系统里没有可用的录音设备，或者它被别的程序独占了。",
        severe: true,
      };
    case "network":
      return {
        kind: "network",
        label: "语音识别服务连不上",
        hint:
          "浏览器自带的识别要走厂商的服务器。离线或代理拦截时会走到这里 —— " +
          "这种情况下用键盘输入，别反复重试。",
        severe: true,
      };
    case "aborted":
      return { kind: "aborted", label: "", hint: "", severe: false };
    default:
      return {
        kind: "unknown",
        label: "语音识别出错",
        hint: `浏览器报的原因：${code || "(空)"}。可以先改用键盘输入。`,
        severe: true,
      };
  }
}

/** 按钮上写什么。 */
export function voiceButtonLabel(
  phase: VoicePhase,
  support: VoiceSupport,
): string {
  if (support === "unsupported") return "不支持语音";
  if (support === "insecure") return "需要 https";
  if (phase === "listening") return "停止";
  if (phase === "requesting") return "授权中";
  return "语音";
}

/**
 * 按钮可不可点，以及不可点时写什么。
 *
 * 返回 `null` 表示可以点。不可点时**必须给出原因** —— 一个灰着又不说话的
 * 按钮，读者只能猜是"还没做"还是"坏了"。
 */
export function voiceBlockedReason(
  phase: VoicePhase,
  support: VoiceSupport,
): string | null {
  if (support === "unsupported") {
    return (
      "这个浏览器没有语音识别接口（Firefox 全系、部分 Safari 都没有）。" +
      "用 Chrome 或 Edge 打开，或者直接用键盘输入。"
    );
  }
  if (support === "insecure") {
    return (
      "麦克风只在 https 或 localhost 下可用。当前页面不在安全上下文里，" +
      "浏览器会直接拒绝且不会弹授权框 —— 换成 https 或用 127.0.0.1 打开。"
    );
  }
  if (phase === "requesting") return "正在等浏览器的麦克风授权…";
  return null;
}

/* ══════════════════════════════════════════════════════════
 * 本地识别（/api/asr → sherpa-onnx）
 * ══════════════════════════════════════════════════════════ */

/** `/api/asr/status` 的形状。字段名与 `api_asr.status()` 一一对应。 */
export interface AsrStatus {
  available: boolean;
  model_ready: boolean;
  dependency_ready: boolean;
  reason: string;
  hint: string;
  download: string;
  model: string;
  max_seconds: number;
}

export type VoiceEngine = "local" | "browser" | "none";

export interface VoiceRoute {
  engine: VoiceEngine;
  /** 一句话说清音频去哪了。界面上必须显示，不能只写进 title。 */
  note: string;
  /** 引擎角标，一行小字。 */
  badge: string;
  /** 为什么没走本地。空字符串 = 就是本地。 */
  degraded: string;
  /** 彻底不能用时的原因。空字符串 = 能用。 */
  blocked: string;
  /** 不能用时按钮上写什么（长原因放 `blocked`，按钮放不下）。 */
  blockedShort: string;
}

const HTTPS_BLOCK =
  "麦克风只在 https 或 localhost 下可用。当前页面不在安全上下文里，" +
  "浏览器会直接拒绝且不会弹授权框 —— 换成 https 或用 127.0.0.1 打开。";

const NO_MIC_API =
  "这个浏览器不提供网页录音接口（navigator.mediaDevices.getUserMedia）。" +
  "本地识别和云端识别都要它，所以两条路都不通 —— 直接用键盘输入。";

/**
 * 麦克风能不能用。与 `detectVoiceSupport` 分开，因为它们是**两个**能力。
 *
 * 本地识别只要 `getUserMedia`，不要 `SpeechRecognition`。合成一个判定的
 * 后果是 Firefox 上明明能跑本地识别，界面却说「不支持语音」。
 */
export function detectMicSupport(
  win:
    | {
        isSecureContext?: boolean;
        location?: { protocol?: string; hostname?: string };
        navigator?: { mediaDevices?: { getUserMedia?: unknown } };
      }
    | undefined,
): VoiceSupport {
  if (!win) return "unsupported";
  if (typeof win.navigator?.mediaDevices?.getUserMedia !== "function") {
    return "unsupported";
  }
  if (win.isSecureContext === false) return "insecure";
  const proto = win.location?.protocol;
  const host = win.location?.hostname;
  if (proto && proto !== "https:" && host !== "localhost" && host !== "127.0.0.1") {
    return "insecure";
  }
  return "available";
}

/**
 * 走哪条路。顺序是**先判环境、再判本地、最后回落云端**。
 *
 * 先判环境的原因：非安全上下文下两条路都走不了，此时说「本地模型没下载」
 * 会把人引到一个修不好问题的方向（他下完模型，麦克风照样被拒）。
 *
 * 回落必须带 `degraded`，界面上要显示。悄悄回落是这条链上最坏的一种
 * 失败：用户以为自己用的是本地识别，而录音已经出网了。
 */
export function pickVoiceRoute(
  status: AsrStatus | null,
  mic: VoiceSupport,
  speech: VoiceSupport,
): VoiceRoute {
  if (mic === "insecure") {
    return { engine: "none", note: "", badge: "", degraded: "", blocked: HTTPS_BLOCK, blockedShort: "需要 https" };
  }
  if (mic === "unsupported") {
    return { engine: "none", note: "", badge: "", degraded: "", blocked: NO_MIC_API, blockedShort: "不支持语音" };
  }
  if (status?.available) {
    return {
      engine: "local",
      note: VOICE_LOCAL_NOTE,
      badge: "本地识别 · 不出本机",
      degraded: "",
      blocked: "",
      blockedShort: "",
    };
  }

  // 本地不可用。原因必须具体到下一步能做什么。
  const why = status
    ? status.reason || "本地识别未就绪"
    : "没连上后端（/api/asr/status 无响应）";

  if (speech === "available") {
    return {
      engine: "browser",
      note: VOICE_PRIVACY_NOTE,
      badge: "云端识别 · 音频出网",
      degraded: `本地识别没启用：${why}`,
      blocked: "",
      blockedShort: "",
    };
  }
  return {
    engine: "none",
    note: "",
    badge: "",
    degraded: "",
    blocked:
      `${why}；而这个浏览器也没有云端识别（Firefox 全系没有），两条路都不通。` +
      "把本地模型下下来，或者直接用键盘输入。",
    blockedShort: "语音不可用",
  };
}

/** 按钮上写什么（按路由而不是按 support）。 */
export function voiceRouteLabel(phase: VoicePhase, route: VoiceRoute): string {
  if (route.engine === "none") return route.blockedShort || "语音不可用";
  if (phase === "listening") return "停止";
  if (phase === "requesting") return "授权中";
  if (phase === "transcribing") return "识别中";
  return "语音";
}

/** 按钮可不可点；不可点时写什么。`null` = 可以点。 */
export function voiceRouteBlockedReason(
  phase: VoicePhase,
  route: VoiceRoute,
): string | null {
  if (route.engine === "none") return route.blocked;
  if (phase === "requesting") return "正在等浏览器的麦克风授权…";
  if (phase === "transcribing") return "正在本地识别，不用重复点。";
  return null;
}

/* ══════════════════════════════════════════════════════════
 * 本地路径的错误
 * ══════════════════════════════════════════════════════════ */

export type LocalErrorKind =
  | "no-backend"
  | "model-missing"
  | "dependency-missing"
  | "mic-denied"
  | "decode-failed"
  | "too-long"
  | "server-error";

/**
 * 本地路径的错误翻译。**与云端那套分开**，因为下一步动作完全不同：
 * 云端错要去看浏览器设置，本地错要去看后端和模型。
 */
export function describeLocalError(
  kind: LocalErrorKind,
  opts: { download?: string; detail?: string } = {},
): VoiceProblem {
  const dl = opts.download || "python scripts/fetch_asr_model.py";
  switch (kind) {
    case "model-missing":
      return {
        kind: "unknown",
        label: "本地模型还没下载",
        hint: `在项目根目录跑 \`${dl}\`（约 74 MB），下完刷新页面就能用本地识别。`,
        severe: false,
      };
    case "dependency-missing":
      return {
        kind: "unknown",
        label: "后端没装 sherpa-onnx",
        hint: "在跑后端的那个 Python 环境里 `pip install sherpa-onnx`，然后重启后端。",
        severe: false,
      };
    case "no-backend":
      return {
        kind: "network",
        label: "连不上本地识别后端",
        hint:
          "后端的 /api/asr 没有响应。先用 start-local.sh（或 .bat）把后端起来 —— " +
          "语音和检索用的是同一个后端，检索能用的话这里不该失败。",
        severe: true,
      };
    case "mic-denied":
      return {
        kind: "not-allowed",
        label: "麦克风权限被拒绝",
        hint: "点地址栏左边的锁图标 → 网站设置 → 麦克风改成「允许」，然后刷新页面。",
        severe: true,
      };
    case "decode-failed":
      return {
        kind: "unknown",
        label: "这段录音解不开",
        hint: "浏览器没能把刚录的音频解成采样点。再录一次；连续失败就改用键盘输入。",
        severe: true,
      };
    case "too-long":
      return {
        kind: "unknown",
        label: "说得太长了",
        hint: "一次说一句，超过 60 秒会被拒绝 —— 那时候通常是麦克风忘了关。",
        severe: false,
      };
    case "server-error":
      return {
        kind: "unknown",
        label: "识别失败",
        hint: `${opts.detail || "后端报错"}。后端的 logs/app.log 里有完整堆栈。`,
        severe: true,
      };
  }
}

/** 录音计时用。`73_400` → `1:13`。 */
export function formatClock(ms: number): string {
  const total = Math.max(0, Math.floor(ms / 1000));
  const m = Math.floor(total / 60);
  const s = total % 60;
  return `${m}:${String(s).padStart(2, "0")}`;
}
