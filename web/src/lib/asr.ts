/**
 * 本地语音识别的前端一侧：录音 → 16 kHz 单声道采样点 → `POST /api/asr`。
 *
 * ## 为什么不是 `MediaRecorder` 直传 webm
 *
 * 后端用 sherpa-onnx，它只吃采样点。webm/opus 要解码器：ffmpeg 本机不一定
 * 有，装个 Python 解码库又要把镜像撑大、许可证重扫一遍。而浏览器**本来
 * 就能**把音频变成采样点 —— `decodeAudioData` 解出 `AudioBuffer`，再用一个
 * 16 kHz 的 `OfflineAudioContext` 播一遍，出来的就是 16 kHz 单声道
 * float32。中间没有编解码，也就没有「格式对不上」这一类失败。
 *
 * 用 `OfflineAudioContext` 而不是自己写线性插值降采样，是因为它有**低通
 * 滤波**。48 kHz → 16 kHz 直接抽点会把 8 kHz 以上的能量折回到语音频段里，
 * 听起来只是「有点闷」，识别结果却是随机的错字 —— 这种错误最难归因。
 *
 * ## 为什么录音不用 `AudioWorklet`
 *
 * Worklet 给的是实时 PCM，适合边说边传。这里一次说一句、说完才发，用不上
 * 实时性，却要多一个独立模块文件与一条 `addModule` 失败分支。
 *
 * ## 这个文件只在浏览器里跑
 *
 * 纯判定逻辑（走哪条路、错怎么翻译）都在 `voice.ts`，能在 Node 里测。
 * 这里只留「非浏览器不可」的那部分，所以它没有任何单元测试 —— 它由
 * `npm run smoke` 与真机点击覆盖。
 */
import type { AsrStatus, LocalErrorKind } from "./voice";

/** sherpa-onnx 的模型全部按 16 kHz 训练。改这个数不会让别的采样率能识别。 */
const TARGET_RATE = 16000;

/** 与 `api_asr.MAX_SECONDS` 对齐。前端先拦一次，免得白等一次上传。 */
export const MAX_SECONDS = 60;

export class AsrError extends Error {
  constructor(
    public kind: LocalErrorKind,
    public detail?: string,
  ) {
    super(`${kind}${detail ? `: ${detail}` : ""}`);
    this.name = "AsrError";
  }
}

/**
 * 后端能不能本地识别。**任何失败都返回 `null`，不抛。**
 *
 * 这个函数在渲染麦克风按钮之前被调用。让它抛，按钮就永远显示「不可用」，
 * 而不是「不可用的原因」—— 而这两种状态在界面上必须能分开读。
 */
export async function fetchAsrStatus(signal?: AbortSignal): Promise<AsrStatus | null> {
  try {
    const res = await fetch("/api/asr/status", { signal });
    if (!res.ok) return null;
    const data = (await res.json()) as AsrStatus;
    // 形状校验：后端换了字段名时，`undefined` 会被当成 `false`，
    // 于是界面安静地显示「本地不可用」—— 看起来像没下载模型。
    if (typeof data?.available !== "boolean") return null;
    return data;
  } catch {
    return null;
  }
}

export interface Recording {
  /** 16 kHz 单声道 float32。 */
  samples: Float32Array;
  /** 从按下到松开的墙钟时长，只用来显示。 */
  durationMs: number;
}

export interface RecordingHandle {
  /** 结束录音并解码。失败抛 `AsrError`。 */
  stop: () => Promise<Recording>;
  /** 丢掉这次录音，不解码、不上传。组件卸载时用。 */
  cancel: () => void;
  elapsedMs: () => number;
}

function pickMime(): string | undefined {
  if (typeof MediaRecorder === "undefined") return undefined;
  // Chrome/Edge 给 webm/opus，Safari 只给 mp4/aac。两个 `decodeAudioData`
  // 都能解，所以这里只是挑一个**能确定支持的**，不是挑最好的。
  const candidates = [
    "audio/webm;codecs=opus",
    "audio/webm",
    "audio/mp4",
    "audio/ogg;codecs=opus",
  ];
  for (const m of candidates) {
    if (typeof MediaRecorder.isTypeSupported !== "function") return undefined;
    if (MediaRecorder.isTypeSupported(m)) return m;
  }
  return undefined;
}

/**
 * 解成 16 kHz 单声道。
 *
 * 两步：先按原生采样率解出 `AudioBuffer`，再让一个 16 kHz 的离线上下文
 * 播它 —— 重采样交给浏览器（有滤波），不自己插值。
 */
async function decodeTo16k(buf: ArrayBuffer): Promise<Float32Array> {
  const Ctx =
    window.AudioContext ??
    (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
  if (!Ctx) throw new AsrError("decode-failed", "这个浏览器没有 AudioContext");

  const tmp = new Ctx();
  let decoded: AudioBuffer;
  try {
    decoded = await tmp.decodeAudioData(buf);
  } catch (e) {
    throw new AsrError("decode-failed", e instanceof Error ? e.message : String(e));
  } finally {
    void tmp.close();
  }

  const frames = Math.max(1, Math.ceil(decoded.duration * TARGET_RATE));
  const off = new OfflineAudioContext(1, frames, TARGET_RATE);
  const src = off.createBufferSource();
  src.buffer = decoded;
  src.connect(off.destination);
  src.start();
  const rendered = await off.startRendering();
  return new Float32Array(rendered.getChannelData(0));
}

/** 开录。权限被拒时抛 `AsrError("mic-denied")`。 */
export async function startRecording(): Promise<RecordingHandle> {
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new AsrError("mic-denied", "navigator.mediaDevices.getUserMedia 不存在");
  }

  let stream: MediaStream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        // 这两个开关对识别是净收益：现场有泵和风机的底噪，
        // 而回声消除能挡掉「扬声器里正在播的告警音」被录进去。
        echoCancellation: true,
        noiseSuppression: true,
      },
    });
  } catch (e) {
    const name = e instanceof DOMException ? e.name : "";
    // `NotFoundError` 是「没有麦克风」，`NotAllowedError` 是「用户拒绝」。
    // 合成一句「权限被拒绝」会让人去翻一个已经开着、但没插设备的设置页。
    const kind: LocalErrorKind = name === "NotFoundError" ? "decode-failed" : "mic-denied";
    throw new AsrError(kind, e instanceof Error ? e.message : String(e));
  }

  const mime = pickMime();
  let rec: MediaRecorder;
  try {
    rec = new MediaRecorder(stream, mime ? { mimeType: mime } : undefined);
  } catch (e) {
    stream.getTracks().forEach((t) => t.stop());
    throw new AsrError("decode-failed", e instanceof Error ? e.message : String(e));
  }

  const chunks: BlobPart[] = [];
  rec.ondataavailable = (e) => {
    if (e.data && e.data.size > 0) chunks.push(e.data);
  };
  const startedAt = Date.now();
  const releaseTracks = () => stream.getTracks().forEach((t) => t.stop());

  rec.start();

  return {
    elapsedMs: () => Date.now() - startedAt,

    cancel: () => {
      try {
        if (rec.state !== "inactive") rec.stop();
      } catch {
        /* 已经停了 */
      }
      releaseTracks();
    },

    stop: () =>
      new Promise<Recording>((resolve, reject) => {
        rec.onstop = () => {
          releaseTracks();
          const durationMs = Date.now() - startedAt;
          if (chunks.length === 0) {
            reject(new AsrError("decode-failed", "MediaRecorder 没有产出任何数据（录得太短？）"));
            return;
          }
          const blob = new Blob(chunks, { type: mime || "audio/webm" });
          blob
            .arrayBuffer()
            .then(decodeTo16k)
            .then((samples) => resolve({ samples, durationMs }))
            .catch((e) => reject(e instanceof AsrError ? e : new AsrError("decode-failed", String(e))));
        };
        if (rec.state === "inactive") {
          // 极短的一次按下：recorder 还没真正启动就停了。不能指望 onstop。
          releaseTracks();
          reject(new AsrError("decode-failed", "录音还没开始就结束了"));
          return;
        }
        rec.stop();
      }),
  };
}

/**
 * 把采样点送去识别。
 *
 * `body` 直接是 float32 的字节：服务端 `np.frombuffer(dtype="<f4")`。
 * 这条约定是**唯一**的格式契约，两边各写一句注释盯着它。
 */
export async function transcribe(
  samples: Float32Array,
  signal?: AbortSignal,
): Promise<{ text: string; elapsedMs: number; silent: boolean }> {
  if (samples.length > TARGET_RATE * MAX_SECONDS) {
    throw new AsrError("too-long", `${(samples.length / TARGET_RATE).toFixed(1)} 秒`);
  }
  // `slice()` 一定产出新的 ArrayBuffer（不会是 SharedArrayBuffer），
  // 所以这里的 cast 是安全的；它顺带把「视图可能不覆盖整个 buffer」也解决了
  // —— 一次最多 3.8 MB 的复制，比留一条「只在某些情况下正确」的分支便宜。
  const body = samples.slice().buffer as ArrayBuffer;

  let res: Response;
  try {
    res = await fetch("/api/asr", {
      method: "POST",
      headers: {
        "Content-Type": "application/octet-stream",
        "X-Sample-Rate": String(TARGET_RATE),
      },
      body,
      signal,
    });
  } catch (e) {
    if (e instanceof DOMException && e.name === "AbortError") throw e;
    throw new AsrError("no-backend", e instanceof Error ? e.message : String(e));
  }

  if (!res.ok) {
    let payload: { detail?: string; model_ready?: boolean; dependency_ready?: boolean } | null = null;
    try {
      payload = await res.json();
    } catch {
      payload = null;
    }
    const detail = payload?.detail ?? `HTTP ${res.status}`;
    if (res.status === 503) {
      throw new AsrError(payload?.model_ready === false ? "model-missing" : "dependency-missing", detail);
    }
    if (res.status === 413) throw new AsrError("too-long", detail);
    throw new AsrError("server-error", detail);
  }

  const data = (await res.json()) as { text?: string; elapsed_ms?: number; silent?: boolean };
  return {
    text: (data.text ?? "").trim(),
    elapsedMs: data.elapsed_ms ?? 0,
    silent: Boolean(data.silent),
  };
}
