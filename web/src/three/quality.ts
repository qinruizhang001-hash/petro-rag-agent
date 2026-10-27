/**
 * 渲染质量分级。
 *
 * ## 为什么需要"分级"而不是"调参"
 *
 * 这条红线是硬的：「必须保证集显电脑 60 FPS」。但集显是个很宽的区间 ——
 * Intel UHD 620 与 Iris Xe 差三倍，而同一块 iGPU 在插电与省电模式下
 * 又差一倍。任何一组写死的参数，在某个端点上一定是错的：
 * 为 UHD 620 调，Iris Xe 上白亏画质；为 Iris Xe 调，UHD 620 上掉帧。
 *
 * 所以不能"调到一个值"，只能"给一个能自己往下走的台阶"。
 *
 * ## 三档，每档动的是**填充率**，不是几何量
 *
 * 集显的瓶颈几乎总是像素填充，而不是顶点/三角形 —— 33 个 draw call、
 * 8,508 个三角形对任何一块 2015 年之后的 GPU 都是噪声。真正吃时间的是
 * 叠加混合的半透明面积（羽流包络体 + 3000 个粒子，每个都要 blend 一次）。
 *
 * | 档 | pixelRatio 上限 | 粒子 | 3D 包络体 | 数据图元 |
 * |---|---|---|---|---|
 * | high | 2 | 有 | 有 | 全部 |
 * | medium | 1.25 | 有 | 有 | 全部 |
 * | low | 1 | **无** | **无** | **全部** |
 *
 * 注意 low 档**不动**任何数据图元：剖面轮廓、等压环、测点球、地层板全在。
 * 降级掉的只有"让人看出它是个体"和"流动感"这两件装饰。
 * 判断标准是这条红线：**这个图元消失后，有没有一条结论跟着消失。**
 * 没有 —— CO₂ 铺到哪一层、离监测井多远、压力影响范围多大，全都还读得出来。
 */

export type Quality = "high" | "medium" | "low";

/** 降级顺序。只降不升 —— 见 `geoscene.ts` 里 `sampleFrame` 的注释。 */
export const DEGRADE_ORDER: readonly Quality[] = ["high", "medium", "low"];

/** `pixelRatio` 上限。这是最有效的一根杠杆：像素数按平方缩放。 */
export const QUALITY_PIXEL_RATIO_CAP: Record<Quality, number> = {
  high: 2,
  medium: 1.25,
  low: 1,
};

/** 粒子（装饰）是否启用。 */
export const QUALITY_PARTICLES: Record<Quality, boolean> = {
  high: true,
  medium: true,
  low: false,
};

/** 3D 包络体（装饰）是否启用。 */
export const QUALITY_VOLUME: Record<Quality, boolean> = {
  high: true,
  medium: true,
  low: false,
};

/** 下一档。已经是最后一档时返回 `null`。 */
export function degrade(q: Quality): Quality | null {
  const i = DEGRADE_ORDER.indexOf(q);
  const next = i >= 0 ? DEGRADE_ORDER[i + 1] : undefined;
  return next ?? null;
}

/**
 * 判定：这一窗该不该降档。
 *
 * 做成纯函数而不是写在 `sampleFrame` 里，是因为**判定必须能被测**。
 * 一个"测不到的控制器"和"没接上的控制器"在代码审查里长得一模一样 ——
 * 而后者会安静地什么都不做，界面上看不出任何区别。
 *
 * @param current   当前档位
 * @param medianMs  这一窗帧耗时的**中位数**（不是平均：一次 GC 尖峰会把
 *                  平均值拉过阈值，于是在一台本来及格的机器上触发降级）
 * @param thresholdMs 触发线。调用方传 `FRAME_BUDGET_MS * 1.15`，
 *                  留一点余量避免在 16.7 / 16.8 之间反复判定
 * @returns 要降到的新档位；`null` 表示维持不变
 */
export function nextQuality(
  current: Quality,
  medianMs: number,
  thresholdMs: number,
): Quality | null {
  if (!(medianMs > thresholdMs)) return null;
  return degrade(current);
}

export function isQuality(v: string | null): v is Quality {
  return v === "high" || v === "medium" || v === "low";
}
