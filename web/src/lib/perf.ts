/**
 * 渲染开销探针。
 *
 * ## 为什么产品代码里要有这个东西
 *
 * "在集显上能跑 60 FPS"是一句必须能被证伪的话。没有测量，它就只是一句
 * 写在文档里的承诺，而下一次有人把 3000 个粒子改成 30 万个、或者把
 * `pixelRatio` 上限从 2 改成 `devicePixelRatio` 时，没有任何东西会红。
 *
 * 所以这里留两个入口：
 * - `?perf=1` → 界面上浮一个小面板，显示实测 FPS 与每帧的 draw call / 三角数；
 * - `window.__perf` → 给 `web/scripts/measure-fps.mjs` 用 CDP 读取，
 *   产出可贴进文档的数字（而不是"我感觉挺流畅"）。
 *
 * ## 不引入 stats.js
 *
 * `stats.js` 是 MIT，可以用。但它把 FPS 面板画在另一个 `<canvas>` 上、
 * 有自己的 DOM 与样式体系，为了三行数字多一个依赖不划算 ——
 * 而这里已经有 Tailwind 令牌了。
 */

export interface FrameSample {
  /** 采样帧数 */
  frames: number;
  /** 每帧耗时（毫秒）的分位数 */
  p50: number;
  p95: number;
  p99: number;
  max: number;
  /** 由 p50 折算的 FPS。用中位数而不是平均值：一次 GC 尖峰会把平均值拉垮 */
  fps: number;
  /** 超过 16.7 ms 的帧数 —— 真正决定"卡不卡"的是这个，不是平均值 */
  overBudget: number;
}

/** 一帧的预算（60 FPS）。 */
export const FRAME_BUDGET_MS = 1000 / 60;

/**
 * 采样若干帧，返回耗时分布。
 *
 * 用 `requestAnimationFrame` 的相邻时间戳之差：它包含浏览器在这一帧里
 * 做的一切（JS、样式、布局、GPU 提交），比 `performance.now()` 包住
 * `renderer.render()` 更接近用户感知到的卡顿。
 *
 * 注意它测的是**显示帧**，所以结果会被 `requestAnimationFrame` 的节流上限
 * （通常是 60 或 120 Hz）截断：如果 p50 正好是 16.7，说明已经顶到上限，
 * 真实余量比数字看起来更大。
 */
export function measureFrames(frames = 180): Promise<FrameSample> {
  return new Promise((resolve) => {
    const stamps: number[] = [];
    let last = 0;

    const step = (t: number) => {
      if (last) stamps.push(t - last);
      last = t;
      if (stamps.length >= frames) {
        resolve(summarize(stamps));
        return;
      }
      requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  });
}

function quantile(sorted: number[], q: number): number {
  if (!sorted.length) return 0;
  // `noUncheckedIndexedAccess` 下索引访问是 `number | undefined`。
  // 上面的空数组检查已经保证了下标合法，但编译器看不出来 —— 写 `?? 0`
  // 而不是 `!`，因为 `!` 会在将来某次改成"传未排序数组"时把 NaN 放进来。
  const i = Math.min(sorted.length - 1, Math.floor(q * sorted.length));
  return sorted[i] ?? 0;
}

export function summarize(deltas: number[]): FrameSample {
  // 第一帧的 delta 不可靠（上一帧可能是几十秒前），丢掉。
  const xs = deltas.slice(1).filter((d) => d > 0 && d < 5000);
  const sorted = [...xs].sort((a, b) => a - b);
  const p50 = quantile(sorted, 0.5);
  return {
    frames: xs.length,
    p50: round(p50),
    p95: round(quantile(sorted, 0.95)),
    p99: round(quantile(sorted, 0.99)),
    max: round(sorted[sorted.length - 1] ?? 0),
    fps: p50 > 0 ? Math.round(1000 / p50) : 0,
    overBudget: xs.filter((d) => d > FRAME_BUDGET_MS).length,
  };
}

function round(x: number): number {
  return Math.round(x * 10) / 10;
}
