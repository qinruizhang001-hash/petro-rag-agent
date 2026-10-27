import { useEffect, useRef, useState } from "react";
import { FRAME_BUDGET_MS, measureFrames } from "../lib/perf";
import type { FrameSample } from "../lib/perf";
import type { SceneStats } from "../three/geoscene";
import { activeScene } from "../three/sceneRegistry";
import { cn } from "../lib/utils";

/**
 * 渲染开销面板（`?perf=1`）。
 *
 * ## 为什么这东西要进产品代码
 *
 * "在集显上能跑 60 FPS"是一句必须能被证伪的话。没有测量，它就只是一句
 * 写在文档里的承诺，而下一次有人把 3000 个粒子改成 30 万个、或者把
 * `pixelRatio` 上限从 2 改成 `devicePixelRatio` 时，没有任何东西会红。
 *
 * ## 两个入口，一个开关
 *
 * - 人看：右上角这块面板，每两秒刷新一次。
 * - 机器读：`window.__perf()`，供 `web/scripts/measure-fps.mjs` 通过 CDP
 *   调用，产出可贴进文档的数字（而不是"我感觉挺流畅"）。
 *
 * 两者都用同一个 `?perf=1` 打开，因为它们的用途是同一件事：取证。
 *
 * ## 面板必须能关掉
 *
 * 它不是常驻 UI —— 常驻的性能数字会变成背景噪声，最后谁都不看。
 * 默认关闭，只在 URL 上显式要求时出现，这样截图与演示不会被它污染。
 */

/** `window.__perf` 的返回形状。CDP 脚本按这个结构断言。 */
export interface PerfReport {
  /** 采样时刻（ISO 8601），用于把数字与某次构建对应起来 */
  at: string;
  /** 帧耗时分布。`fps` 由 p50 折算 */
  frames: FrameSample;
  /** 一帧的渲染开销。场景不存在时为 null（例如停在全链 Tab） */
  scene: SceneStats | null;
  /** 一帧预算（毫秒），即 1000/60 */
  budgetMs: number;
}

declare global {
  interface Window {
    __perf?: () => Promise<PerfReport>;
  }
}

function readSceneStats(): SceneStats | null {
  const scene = activeScene();
  if (!scene) return null;
  try {
    return scene.stats();
  } catch {
    // 上下文已丢失（GPU 重置、标签页被回收）时 `renderer.info` 可能抛。
    // 探针不该因此让整个界面白屏 —— 它只是诊断工具。
    return null;
  }
}

export function PerfOverlay() {
  const [sample, setSample] = useState<FrameSample | null>(null);
  const [stats, setStats] = useState<SceneStats | null>(null);
  const [collapsed, setCollapsed] = useState(false);
  const running = useRef(false);

  useEffect(() => {
    window.__perf = async (): Promise<PerfReport> => ({
      at: new Date().toISOString(),
      frames: await measureFrames(180),
      scene: readSceneStats(),
      budgetMs: Math.round(FRAME_BUDGET_MS * 10) / 10,
    });

    // 面板自身的采样：120 帧（约 2 秒）一轮，跑完再排下一轮。
    // 用 `running` 而不是 setInterval —— 采样是异步的，定时器会在
    // 上一轮还没结束时再起一轮，两轮争同一块画布，测出来的是互相干扰。
    let cancelled = false;
    const loop = async () => {
      if (cancelled || running.current) return;
      running.current = true;
      try {
        const next = await measureFrames(120);
        if (cancelled) return;
        setSample(next);
        setStats(readSceneStats());
      } finally {
        running.current = false;
      }
      if (!cancelled) setTimeout(loop, 400);
    };
    void loop();

    return () => {
      cancelled = true;
      delete window.__perf;
    };
  }, []);

  // p50 超过预算才算"掉帧"。单看 fps 不够：60 Hz 上限会把 16.7 ms
  // 的样本压成 60，而 16.6 与 16.9 在这块屏上是同一件事。
  const slow = sample ? sample.p50 > FRAME_BUDGET_MS : false;
  const jitter = sample ? sample.overBudget > sample.frames * 0.05 : false;

  return (
    <div className="pointer-events-auto absolute right-2 top-2 z-50 w-44 rounded-panel border border-line bg-surface-0/92 px-2 py-1.5 font-num text-2xs backdrop-blur-sm">
      <button
        type="button"
        onClick={() => setCollapsed((v) => !v)}
        className="flex w-full items-baseline justify-between text-ink-2"
      >
        <span className="font-sans">渲染开销</span>
        <span>{collapsed ? "▸" : "▾"}</span>
      </button>

      {!collapsed && (
        <div className="mt-1 space-y-0.5">
          <Row
            label="FPS (p50)"
            value={sample ? String(sample.fps) : "—"}
            tone={slow ? "bad" : "ok"}
          />
          <Row label="p95" value={sample ? `${sample.p95} ms` : "—"} />
          <Row
            label="超预算帧"
            value={sample ? `${sample.overBudget}/${sample.frames}` : "—"}
            tone={jitter ? "bad" : "ok"}
          />
          <div className="my-1 border-t border-line" />
          <Row label="draw calls" value={stats ? String(stats.drawCalls) : "—"} />
          <Row label="triangles" value={stats ? stats.triangles.toLocaleString() : "—"} />
          <Row label="points" value={stats ? stats.points.toLocaleString() : "—"} />
          <Row label="geometries" value={stats ? String(stats.geometries) : "—"} />
          <Row label="textures" value={stats ? String(stats.textures) : "—"} />
          <Row
            label="画布"
            value={stats ? `${stats.width}×${stats.height}@${stats.pixelRatio}` : "—"}
          />
          <Row label="GPU" value={stats ? shortGpu(stats.gpu) : "—"} />
          <Row
            label="档位"
            value={stats ? `${stats.quality}${stats.qualityPinned ? " (钉死)" : ""}` : "—"}
            tone={stats && stats.quality !== "high" ? "bad" : undefined}
          />
          <Row label="循环" value={stats ? (stats.animating ? "跑" : "停") : "—"} />
          {sample && sample.fps > 120 && (
            /*
             * rAF 没有锁 vsync 时（远程桌面、虚拟显示器、部分驱动配置），
             * FPS 会远高于屏幕刷新率。这时它读作"每帧还剩多少余量"，
             * 不读作"用户看到的流畅度" —— 说清楚，否则 303 这个数会被
             * 当成"这台机器能跑 303 帧"写进汇报。
             */
            <p className="pt-1 font-sans leading-tight text-ink-2">
              rAF 未锁 vsync，FPS 读作余量
            </p>
          )}
        </div>
      )}
    </div>
  );
}

function Row({
  label,
  value,
  tone,
}: {
  label: string;
  value: string;
  tone?: "ok" | "bad";
}) {
  return (
    <div className="flex items-baseline justify-between gap-1">
      <span className="truncate font-sans text-ink-2">{label}</span>
      <span
        className={cn(
          "shrink-0",
          tone === "bad" ? "text-sev-alert" : tone === "ok" ? "text-ink-0" : "text-ink-1",
        )}
      >
        {value}
      </span>
    </div>
  );
}

/**
 * 把 `WEBGL_debug_renderer_info` 的长串压成一行。
 *
 * 原始值是 `ANGLE (NVIDIA, NVIDIA GeForce RTX 5060 Laptop GPU (0x00002D19)
 * Direct3D11 vs_5_0 ps_5_0, D3D11)` —— 一百多个字符，在 176 px 宽的栏里
 * 只会显示成 `ANGLE (NVIDIA, NVID…`，恰好把型号切掉，留下的是没信息量的部分。
 */
function shortGpu(raw: string): string {
  const m = raw.match(/(NVIDIA|Intel|AMD|Apple|Adreno|Mali|Microsoft)[^,)]*/);
  return (m ? m[0] : raw).trim();
}
