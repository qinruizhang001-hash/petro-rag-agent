/**
 * 当前活动场景的登记处。
 *
 * ## 为什么需要它
 *
 * 性能探针（`?perf=1`）要读 `renderer.info`，而 `GeoScene` 实例住在
 * `GeoSectionCanvas` 的 `useEffect` 闭包里。把它往上提一层当 props 传，
 * 会让"性能诊断"这条支线污染主数据流 —— 而主数据流里已经有一条
 * `data → updateData` 的显式契约，不该再混进一个只在调试时存在的对象。
 *
 * ## 只有一个场景
 *
 * 这块屏上同时只存在一个 3D 画布（中央区两个 Tab 互斥，切走时组件卸载）。
 * 所以登记处是单槽而不是数组 —— 数组会让人以为"可能有好几个"，
 * 于是探针要决定读哪一个，而那个决定没有任何依据。
 *
 * 卸载时**必须**调 `setActiveScene(null)`：不清空的话，探针在
 * 切到"全链"Tab 之后仍然会读到上一帧的 `stats()`，显示出一组
 * 看起来正常、但已经不属于任何活着的场景的数字。
 */
import type { GeoScene } from "./geoscene";

let current: GeoScene | null = null;

export function setActiveScene(scene: GeoScene | null): void {
  current = scene;
}

export function activeScene(): GeoScene | null {
  return current;
}
