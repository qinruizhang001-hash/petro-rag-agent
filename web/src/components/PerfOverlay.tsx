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
 *
 * ## 结论先行
 *
 * 面板原先第一行就是 `FPS (p50)`。对不搞渲染的人，那个数字既判断不了好坏
 * （60 算好吗？30 算坏吗？），也说明不了该做什么。而这块面板真正要回答的
 * 只有两个问题："现在看着卡不卡"、"是不是已经自己降过档了"。
 * 所以顶部先给一句人话结论，技术数字收进「技术细节」——
 * 它们仍然在，只是不再挡在最前面。
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
  const downgraded = !!stats && stats.quality !== "high";

  /*
   * 结论的顺序按"用户会先注意到什么"排：
   * 卡了 → 先说卡；不卡但降过档 → 说降档；否则说流畅。
   * 反过来（先报降档）会让一台其实很顺的机器看起来有问题。
   */
  const verdict = !sample
    ? { label: "测量中", cls: "text-ink-2" }
    : slow || jitter
      ? { label: "画面吃力", cls: "text-sev-alert" }
      : downgraded
        ? { label: "已自动降档", cls: "text-sev-warn" }
        : { label: "画面流畅", cls: "text-sev-normal" };

  return (
    <div className="pointer-events-auto absolute right-2 top-2 z-50 w-56 rounded-panel border border-line bg-surface-0/92 px-2 py-1.5 text-2xs backdrop-blur-sm">
      <button
        type="button"
        onClick={() => setCollapsed((v) => !v)}
        className="flex w-full items-baseline justify-between gap-1"
      >
        <span className="flex items-baseline gap-1">
          <span className="font-sans text-ink-2">渲染</span>
          <span className={cn("font-sans font-medium", verdict.cls)}>{verdict.label}</span>
        </span>
        <span className="text-ink-2">{collapsed ? "▸" : "▾"}</span>
      </button>

      {!collapsed && (
        <div className="mt-1 space-y-0.5">
          {/*
            给人读的结论。技术数字在这下面，展开才有 —— 顺序反了的话，
            非技术人员读到的第一个词就是 draw calls。
          */}
          <p className="font-sans leading-snug text-ink-2">
            {!sample
              ? "正在采样，约 2 秒一轮。"
              : slow || jitter
                ? "每帧经常来不及画完，滚动和转视角时会感到卡。"
                : "帧耗时稳定在 60 Hz 的预算里，看着是顺的。"}
            {downgraded && (
              <>
                {" "}
                这台机器跑不动最高档，程序自己降到了
                <span className="text-ink-1"> {stats?.quality} </span>
                档：只降填充率（分辨率与粒子数），数据图元一个没少。
              </>
            )}
          </p>

          <div className="my-1 border-t border-line" />
          <Row
            label="流畅度（帧/秒）"
            value={sample ? String(sample.fps) : "—"}
            tone={slow ? "bad" : "ok"}
            hint="每秒画多少帧，按中位数帧耗时折算，不是平均值 —— 平均值会被一次垃圾回收的尖峰拉低。"
          />
          <Row
            label="一帧要多久"
            value={sample ? `${sample.p50} ms` : "—"}
            hint="中位数帧耗时。16.7 ms 是 60 Hz 的一帧预算，超过就来不及画完。"
          />
          <Row
            label="最慢的 5%"
            value={sample ? `${sample.p95} ms` : "—"}
            hint="最慢那 5% 的帧耗时。人感到的卡顿来自这里 —— 平均值看不出来。"
          />
          <Row
            label="超时的帧"
            value={sample ? `${sample.overBudget}/${sample.frames}` : "—"}
            tone={jitter ? "bad" : "ok"}
            hint="耗时超过 16.7 ms 的帧数 / 总采样帧数。占比 >5% 记为抖动。"
          />

          <details className="pt-1">
            <summary className="cursor-pointer font-sans text-ink-2 hover:text-ink-1">
              技术细节（排查用）
            </summary>
            <div className="mt-1 space-y-0.5 font-num">
              <Row
                label="draw calls"
                value={stats ? String(stats.drawCalls) : "—"}
                hint="每帧提交的绘制批次。透明网格开 DoubleSide 会让它凭空翻倍。"
              />
              <Row
                label="triangles"
                value={stats ? stats.triangles.toLocaleString() : "—"}
                hint="每帧三角形数。与 draw calls 一起看：三角形多而批次少，说明合并得好。"
              />
              <Row
                label="points"
                value={stats ? stats.points.toLocaleString() : "—"}
                hint="每帧点精灵数（羽流粒子）。降档时先动这一项，不动任何数据图元。"
              />
              <Row
                label="geometries"
                value={stats ? String(stats.geometries) : "—"}
                hint="常驻 GPU 的几何体数量。只增不减 = 泄漏。"
              />
              <Row
                label="textures"
                value={stats ? String(stats.textures) : "—"}
                hint="常驻 GPU 的纹理数量。同样只增不减就是泄漏。"
              />
              <Row
                label="画布"
                value={stats ? `${stats.width}×${stats.height}@${stats.pixelRatio}` : "—"}
                hint="实际渲染分辨率 @ 像素比。降档后这里会变小 —— 这是填充率在降。"
              />
              <Row
                label="GPU"
                value={stats ? shortGpu(stats.gpu) : "—"}
                hint="WebGL 报告的适配器名。"
              />
              <Row
                label="档位"
                value={stats ? `${stats.quality}${stats.qualityPinned ? " (钉死)" : ""}` : "—"}
                tone={downgraded ? "bad" : undefined}
                hint="自适应画质档位。只降不升，最多降两档，判据是帧耗时的中位数。"
              />
              <Row
                label="循环"
                value={stats ? (stats.animating ? "跑" : "停") : "—"}
                hint="渲染循环是否在跑。停在后台标签页时会自动停。"
              />
            </div>
          </details>

          {sample && sample.fps > 120 && (
            /*
             * rAF 没有锁 vsync 时（远程桌面、虚拟显示器、部分驱动配置），
             * FPS 会远高于屏幕刷新率。这时它读作"每帧还剩多少余量"，
             * 不读作"用户看到的流畅度" —— 说清楚，否则 303 这个数会被
             * 当成"这台机器能跑 303 帧"写进汇报。
             */
            <p className="pt-1 font-sans leading-tight text-ink-2">
              屏幕没锁刷新率，帧/秒读作余量，不代表观感
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
  hint,
}: {
  label: string;
  value: string;
  tone?: "ok" | "bad";
  /** 一句话说明这个指标是什么。非前端的人看不懂缩写，而看不懂的下一步是不看 */
  hint?: string;
}) {
  return (
    <div className="flex items-baseline justify-between gap-1" title={hint}>
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
 * Direct3D11 vs_5_0 ps_5_0, D3D11)` —— 一百多个字符，在 224 px 宽的栏里
 * 只会显示成 `ANGLE (NVIDIA, NVID…`，恰好把型号切掉，留下的是没信息量的部分。
 */
function shortGpu(raw: string): string {
  const m = raw.match(/(NVIDIA|Intel|AMD|Apple|Adreno|Mali|Microsoft)[^,)]*/);
  return (m ? m[0] : raw).trim();
}
