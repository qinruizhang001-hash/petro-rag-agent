import { useCallback, useEffect, useRef, useState } from "react";

import {
  AsrError,
  MAX_SECONDS,
  fetchAsrStatus,
  startRecording,
  transcribe,
} from "../../lib/asr";
import type { RecordingHandle } from "../../lib/asr";
import {
  VOICE_SCOPE_NOTE,
  describeLocalError,
  describeVoiceError,
  detectMicSupport,
  detectVoiceSupport,
  formatClock,
  pickVoiceRoute,
  voiceRouteBlockedReason,
  voiceRouteLabel,
} from "../../lib/voice";
import type { AsrStatus, VoicePhase, VoiceProblem, VoiceRoute, VoiceSupport } from "../../lib/voice";
import { cn } from "../../lib/utils";

/**
 * 浏览器 `SpeechRecognition` 的最小接口。
 *
 * 不引 `@types/dom-speech-recognition`：这个项目里只有这一个文件用它，
 * 为四个回调引一个包不值得，而 `any` 会让 `onresult` 的形状完全靠猜。
 */
interface RecognitionLike {
  lang: string;
  continuous: boolean;
  interimResults: boolean;
  maxAlternatives: number;
  start: () => void;
  stop: () => void;
  abort: () => void;
  onstart: (() => void) | null;
  onend: (() => void) | null;
  onerror: ((e: { error?: string }) => void) | null;
  onresult:
    | ((e: {
        resultIndex: number;
        results: {
          length: number;
          [i: number]: { isFinal: boolean; 0: { transcript: string } };
        };
      }) => void)
    | null;
}

type Ctor = new () => RecognitionLike;

function ctorOf(): Ctor | null {
  if (typeof window === "undefined") return null;
  const w = window as unknown as {
    SpeechRecognition?: Ctor;
    webkitSpeechRecognition?: Ctor;
  };
  return w.SpeechRecognition ?? w.webkitSpeechRecognition ?? null;
}

/**
 * 语音输入按钮。
 *
 * ## 它只产出一段文字，然后什么都不做
 *
 * 识别结果通过 `onCommit` 交给父组件，父组件**并进输入框**。这里没有
 * 发送能力，也不持有 `send` —— 一个能直接发问的语音按钮，等于让识别错误
 * 直接变成一个已提交的问题。
 *
 * ## 两条路，界面必须说清走的是哪条
 *
 * 本地（`/api/asr` → sherpa-onnx）录音不出本机；云端（浏览器自带）音频
 * 会送到厂商服务器。缺模型或缺依赖时回落到云端，但**回落要写在界面上** ——
 * 悄悄回落会让用户以为自己用的是本地识别，而录音已经出网了。
 *
 * 角标是短的那一句（`本地识别 · 不出本机`），长说明在 title 里。只写进
 * title 不行：决定"要不要按"的人不会去悬停。
 */
export function VoiceButton({
  onCommit,
  onInterim,
  disabled,
}: {
  /** 识别出一段最终文本时调用。父组件负责 merge 进输入框。 */
  onCommit: (text: string) => void;
  /** 边说边出的临时文本；`null` 表示清空。 */
  onInterim: (text: string | null) => void;
  /** 回答正在生成时禁用 —— 那时候发送按钮也禁用，不该再往里塞字。 */
  disabled?: boolean;
}) {
  const [mic, setMic] = useState<VoiceSupport>("unsupported");
  const [speech, setSpeech] = useState<VoiceSupport>("unsupported");
  const [status, setStatus] = useState<AsrStatus | null>(null);
  const [phase, setPhase] = useState<VoicePhase>("idle");
  const [problem, setProblem] = useState<VoiceProblem | null>(null);
  const [elapsed, setElapsed] = useState(0);

  const recRef = useRef<RecognitionLike | null>(null);
  const recHandleRef = useRef<RecordingHandle | null>(null);
  /** 已经提交过的片段数，避免同一条结果被 commit 两次。 */
  const doneRef = useRef(0);
  /** 组件卸载后不要再 setState。 */
  const aliveRef = useRef(true);

  // 环境判定放在 effect 里：SSR / Node 下没有 window，渲染期读会抛。
  useEffect(() => {
    setMic(detectMicSupport(window as unknown as Parameters<typeof detectMicSupport>[0]));
    setSpeech(detectVoiceSupport(window as unknown as Parameters<typeof detectVoiceSupport>[0]));
  }, []);

  /*
   * 探测后端。**不受 `?source=fixture` 影响** —— 数据源决定的是告警与检索
   * 从哪来，而语音这一格决定的是录音去哪。后端在跑就该用本地，哪怕这一屏
   * 看的是快照。后端不在时 `fetchAsrStatus` 返回 null，路由自动回落。
   */
  useEffect(() => {
    const ac = new AbortController();
    fetchAsrStatus(ac.signal).then((s) => {
      if (aliveRef.current) setStatus(s);
    });
    return () => ac.abort();
  }, []);

  const route: VoiceRoute = pickVoiceRoute(status, mic, speech);

  const cleanup = useCallback(() => {
    recRef.current?.abort();
    recRef.current = null;
    recHandleRef.current?.cancel();
    recHandleRef.current = null;
  }, []);

  // 卸载时一定要收掉麦克风，否则指示灯会一直亮着 —— 切页签、关面板都会走到。
  useEffect(() => {
    aliveRef.current = true;
    return () => {
      aliveRef.current = false;
      cleanup();
    };
  }, [cleanup]);

  /* ── 云端：浏览器自带的 SpeechRecognition ─────────────── */

  const startBrowser = useCallback(() => {
    const Ctor = ctorOf();
    if (!Ctor) {
      setProblem(describeVoiceError("unsupported"));
      return;
    }
    setProblem(null);
    setPhase("requesting");
    onInterim(null);
    doneRef.current = 0;

    const rec = new Ctor();
    /*
     * `continuous = false`：一句话就停。
     *
     * 连续识别在中文里会把停顿当成句子边界，然后一路听下去 —— 现场
     * 工程师说完"过去四小时有什么异常"之后在跟同事说话，后半句会被
     * 当成问题的一部分接上去。一次一句，人要停下来看一眼再发。
     */
    rec.lang = "zh-CN";
    rec.continuous = false;
    rec.interimResults = true;
    rec.maxAlternatives = 1;

    rec.onstart = () => setPhase("listening");

    rec.onresult = (e) => {
      let interim = "";
      for (let i = e.resultIndex; i < e.results.length; i += 1) {
        const r = e.results[i];
        if (!r) continue;
        const text = r[0]?.transcript ?? "";
        if (r.isFinal) {
          // 用 doneRef 而不是清空 results：同一个 final 结果在 Safari 上
          // 会重复投递一次，重复 commit 会让输入框出现两遍同样的话。
          if (i >= doneRef.current) {
            doneRef.current = i + 1;
            onCommit(text);
          }
        } else {
          interim += text;
        }
      }
      onInterim(interim.trim() ? interim : null);
    };

    rec.onerror = (e) => {
      const p = describeVoiceError(e.error ?? "");
      // `aborted` 是我们自己 stop 触发的，不是错误，也不该清掉已有的字。
      if (p.kind === "aborted") return;
      setProblem(p);
      setPhase(p.severe ? "error" : "idle");
      onInterim(null);
    };

    rec.onend = () => {
      onInterim(null);
      setPhase((prev) => (prev === "error" ? "error" : "idle"));
      recRef.current = null;
    };

    recRef.current = rec;
    try {
      rec.start();
    } catch {
      // `start()` 在已经在跑的时候会抛 InvalidStateError。这不是故障，
      // 是连点两下；把状态摆正就行，不要弹一个红框。
      setPhase("listening");
    }
  }, [onCommit, onInterim]);

  /* ── 本地：录音 → 采样点 → /api/asr ───────────────────── */

  const finishLocal = useCallback(async () => {
    const handle = recHandleRef.current;
    if (!handle) return;
    recHandleRef.current = null;
    setPhase("transcribing");
    try {
      const rec = await handle.stop();
      if (!aliveRef.current) return;
      if (rec.durationMs > MAX_SECONDS * 1000) {
        setProblem(describeLocalError("too-long"));
        setPhase("idle");
        return;
      }
      const out = await transcribe(rec.samples);
      if (!aliveRef.current) return;
      if (!out.text) {
        // 空文本不是错误，但也不能安静地什么都不发生 —— 那样人会以为
        // 按钮没生效，然后连按三次。
        setProblem({
          kind: "no-speech",
          label: out.silent ? "没有收到声音" : "没听清",
          hint: out.silent
            ? "麦克风是静音状态，或者被系统静音了 —— 一个采样都没收到。"
            : "再说一次，离麦克风近一点。",
          severe: false,
        });
      } else {
        setProblem(null);
        onCommit(out.text);
      }
      setPhase("idle");
    } catch (e) {
      if (!aliveRef.current) return;
      if (e instanceof DOMException && e.name === "AbortError") {
        setPhase("idle");
        return;
      }
      const kind = e instanceof AsrError ? e.kind : "server-error";
      setProblem(describeLocalError(kind, { detail: e instanceof Error ? e.message : String(e) }));
      setPhase("idle");
    }
  }, [onCommit]);

  const startLocal = useCallback(async () => {
    setProblem(null);
    onInterim(null);
    setPhase("requesting");
    try {
      const handle = await startRecording();
      if (!aliveRef.current) {
        handle.cancel();
        return;
      }
      recHandleRef.current = handle;
      setElapsed(0);
      setPhase("listening");
    } catch (e) {
      if (!aliveRef.current) return;
      const kind = e instanceof AsrError ? e.kind : "mic-denied";
      setProblem(describeLocalError(kind, { detail: e instanceof Error ? e.message : String(e) }));
      setPhase("idle");
    }
  }, [onInterim]);

  // 录音计时 + 到点自动停。上限与后端 413 的那条线是同一个数字，
  // 但这里先停一次：让人看着表被截断，比录满 60 秒再收到 413 好。
  useEffect(() => {
    if (phase !== "listening" || route.engine !== "local") return;
    const t = window.setInterval(() => {
      const ms = recHandleRef.current?.elapsedMs() ?? 0;
      setElapsed(ms);
      if (ms >= MAX_SECONDS * 1000) void finishLocal();
    }, 250);
    return () => window.clearInterval(t);
  }, [phase, route.engine, finishLocal]);

  const stop = useCallback(() => {
    if (route.engine === "local") {
      void finishLocal();
    } else {
      recRef.current?.stop();
    }
  }, [route.engine, finishLocal]);

  const start = useCallback(() => {
    if (route.engine === "local") void startLocal();
    else startBrowser();
  }, [route.engine, startLocal, startBrowser]);

  const blocked = voiceRouteBlockedReason(phase, route);
  const canClick = !disabled && blocked === null;
  const listening = phase === "listening";

  return (
    <div className="flex flex-col items-end gap-1">
      <button
        type="button"
        onClick={listening ? stop : start}
        disabled={!canClick}
        aria-pressed={listening}
        title={[VOICE_SCOPE_NOTE, route.note, route.degraded, blocked ?? ""]
          .filter(Boolean)
          .join("\n\n")}
        className={cn(
          "shrink-0 self-stretch rounded-panel border px-2 text-2xs",
          listening
            ? "border-sev-alert/60 text-sev-alert"
            : "border-acc/40 text-acc",
          !canClick && "border-line text-ink-2",
        )}
      >
        {voiceRouteLabel(phase, route)}
      </button>

      {/*
        引擎角标。形状上必须能分开读：本地是实线，云端是虚线 ——
        只靠颜色（绿/橙）在灰度打印和色觉障碍下会变成同一个东西。
      */}
      {route.engine !== "none" && (
        <span
          className={cn(
            "rounded-panel border px-1 text-2xs leading-tight",
            route.engine === "local"
              ? "border-line-strong text-ink-2"
              : "border-sev-warn/60 text-sev-warn border-dashed",
          )}
        >
          {route.badge}
        </span>
      )}

      {listening && route.engine === "local" && (
        <p className="text-right text-2xs leading-tight text-sev-alert">
          录音中 {formatClock(elapsed)} / {formatClock(MAX_SECONDS * 1000)}
        </p>
      )}

      {/* 回落原因必须显示。悄悄回落等于让人以为录音没出网。 */}
      {route.degraded && !disabled && (
        <p className="max-w-56 text-right text-2xs leading-tight text-ink-2">
          {route.degraded}
        </p>
      )}

      {/*
        不可点的原因写在外面，不只在 title 里。
        一个灰着又不说话的按钮，读者只能猜"还没做"还是"坏了" ——
        而这台机器上两种都可能。
      */}
      {blocked && !disabled && (
        <p className="max-w-40 text-right text-2xs leading-tight text-ink-2">{blocked}</p>
      )}

      {problem && problem.severe && (
        <p className="max-w-56 text-right text-2xs leading-tight text-sev-alert">
          {problem.label}
          {problem.hint && <span className="text-ink-2"> · {problem.hint}</span>}
        </p>
      )}
      {problem && !problem.severe && problem.label && (
        <p className="max-w-56 text-right text-2xs leading-tight text-ink-2">
          {problem.label}
          {problem.hint && <span> · {problem.hint}</span>}
        </p>
      )}
    </div>
  );
}
