/**
 * 场景构建的共享上下文。
 *
 * ## 为什么不是一堆参数
 *
 * 拆 `geoscene.ts` 的时候有两个选择：给每个 builder 传 `(scene, data, anchors,
 * track, uniforms, ...)` 六个参数，或者传一个上下文对象。选后者是因为
 * **参数列表还会长** —— 下一个加进来的是 `renderer`（做遮挡查询）或者
 * `camera`（做视锥裁剪），六个参数变八个，每个调用点都要改。
 *
 * 上下文是只读的契约：builder 往里加东西（图元、锚点），不往里改配置。
 */
import type * as THREE from "three";
import type { SectionData } from "../lib/section";

/**
 * 剖面在 Z 方向的厚度（场景单位）。160 m —— 薄到能读，厚到能看出是个体。
 *
 * 立体感的来源就是这个厚度加上侧壁描边，**不是**透视。相机是正交的，
 * 见 `geoscene.ts` 的注释。
 */
export const THICKNESS = 1.6;

/** 羽流体在 Z 方向的压扁系数。真实羽流是"饼"不是"球"，压扁后不会读成球体。 */
export const PLUME_Z_SQUASH = 0.45;

/**
 * 粒子着色器的 uniform。由 `GeoScene` 持有，每帧只改 `uTime`。
 *
 * **必须是 `type` 不能是 `interface`。** `ShaderMaterial` 的 `uniforms`
 * 参数类型是 `{ [uniform: string]: IUniform }`，而 TypeScript 只给
 * 对象字面量类型别名隐式索引签名，`interface` 没有 —— 写成 interface
 * 会在 `new THREE.ShaderMaterial({ uniforms })` 处报
 * "Index signature for type 'string' is missing"。
 */
export type ParticleUniforms = {
  uTime: { value: number };
  uSize: { value: number };
  uSizeScale: { value: number };
  uPixelRatio: { value: number };
  uOpacity: { value: number };
  uTintCore: { value: THREE.Color };
  uTintEdge: { value: THREE.Color };
}

/** 交给 disposal 表的东西。`Object3D` 自己没有 `dispose()`，所以只能收这两类。 */
export type Disposable = { dispose: () => void };

export interface BuildContext {
  scene: THREE.Scene;
  data: SectionData;
  /** 可以被 HTML 叠加层定位的锚点（世界坐标） */
  anchors: Map<string, THREE.Vector3>;
  /** 登记一个需要释放的资源，原样返回（方便 `new X(this.track(...))`） */
  track: <T extends Disposable>(x: T) => T;
  uniforms: ParticleUniforms;
}
