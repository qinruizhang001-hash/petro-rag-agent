/**
 * 地质剖面 3D 场景 —— 装配层。
 *
 * 这个文件只做四件事：**建渲染器、装配图元、跑渲染循环、释放**。
 * 具体每一笔怎么画在各自的模块里：
 *
 * | 模块 | 内容 | 身份 |
 * |---|---|---|
 * | `strata.ts` | 地层板、地表网格、盖层底界 | 背景 |
 * | `wells.ts` | 井筒、井口、射孔段 | 数据（弱） |
 * | `plume.ts` | 剖面轮廓、3D 包络体、等压环 | 数据 + 装饰 |
 * | `particles.ts` | 3000 个 CO₂ 点 | 装饰 |
 * | `monitors.ts` | 测点球、外环、深度尺锚点 | 数据 |
 * | `camera.ts` | 正交取景与像素换算 | — |
 * | `context.ts` | builder 的共享上下文 | — |
 *
 * ## 为什么不直接用 @react-three/fiber
 *
 * 这个场景只有一个画布、一台固定机位的相机、没有交互拾取，也不需要
 * 每帧把状态同步回 React。r3f 的价值在于"把场景变成可组合的 React 树"，
 * 而这里恰好没有需要组合的东西 —— 引入 reconciler 换来的只是更大的包和
 * 一层要额外解释的抽象。所以直接调 three。
 *
 * ## 三种图元，三种身份
 *
 * 这张图上每一笔都必须是数据，或者必须一眼看出是装饰。混在一起之后，
 * "那个发光的圆是什么"就会变成每次评审的第一个问题。
 *
 * | 图元 | 身份 | 指挥舱 | 工程师 |
 * |---|---|---|---|
 * | 地层板 | 背景（层位参照） | 有 | 有（描边更硬） |
 * | 羽流**剖面轮廓** | **数据**（CO₂ 波及范围） | 有 | 有 |
 * | 羽流**体**（3D 包络） | 装饰（让人看出它是个体） | 有 | **无** |
 * | 粒子 | 装饰（流动感） | 有 | **无** |
 * | 等压虚线环 | **数据**（压力影响范围 > CO₂ 范围） | 有 | 有 |
 * | 测点球 | **数据**（六级色） | 有 | 有 |
 *
 * 工程师模式关掉的是"体"和"粒子"，**不是羽流本身**。第一版把整个羽流
 * 都当特效删掉了，结果工程师模式得到一个空的剖面 —— 而那正是现场最需要
 * 看的东西：CO₂ 到底铺到哪一层、离监测井还有多远。
 */

import * as THREE from "three";
import { FRAME_BUDGET_MS } from "../lib/perf";
import type { MonitorAnchor, SectionData } from "../lib/section";
import { frameCamera } from "./camera";
import type { BuildContext, Disposable, ParticleUniforms } from "./context";
import { buildMonitors, updateMonitorColors } from "./monitors";
import { buildParticles } from "./particles";
import { buildPlume } from "./plume";
import {
  QUALITY_PARTICLES,
  QUALITY_PIXEL_RATIO_CAP,
  QUALITY_VOLUME,
  nextQuality,
} from "./quality";
import type { Quality } from "./quality";
import { buildStrata, buildSurfaceGrid } from "./strata";
import { buildWells } from "./wells";

export interface ScreenAnchor {
  id: string;
  x: number;
  y: number;
  visible: boolean;
}

export type GeoMode = "deck" | "engineer";

/**
 * 构造选项。
 *
 * `lowPower` 存在的唯一理由：**证明集显上也能跑**。
 *
 * 混合显卡的机器上，`powerPreference: "high-performance"` 会把 WebGL 路由到
 * 独显 —— 这对产品是对的（指挥舱就该用独显），但它让"集显 60 FPS"这条
 * 承诺永远无法被测量：你在带独显的机器上量到的永远是独显的数字。
 *
 * 所以留一个 `?gpu=low` 的诊断开关，把 `powerPreference` 钉成 `low-power`，
 * 强制落到集显上，同一份代码、同一套几何、同样的 3000 个粒子。
 */
export interface GeoSceneOptions {
  lowPower?: boolean;
  /**
   * 钉死渲染档位（`?quality=high|medium|low`），关闭自适应。
   *
   * 截图与评审需要确定性：同一份代码在两台机器上产出两张不一样的图，
   * 比对就无从谈起。不传则从 `high` 起步，由帧耗时自己往下走。
   */
  quality?: Quality;
}

/** 一帧的渲染开销。见 `GeoScene.stats()`。 */
export interface SceneStats {
  drawCalls: number;
  triangles: number;
  points: number;
  lines: number;
  geometries: number;
  textures: number;
  programs: number;
  width: number;
  height: number;
  pixelRatio: number;
  /** 渲染循环是否在跑。工程师模式应当为 `false`（只渲染一帧）。 */
  animating: boolean;
  /**
   * 实际承担渲染的 GPU（`WEBGL_debug_renderer_info`）。
   *
   * 为什么这个字段必须在：混合显卡的机器上，同一个页面的帧率可以差三倍，
   * 而"p50 3.3 ms"这个数字离开 GPU 名字就不可解释 —— 它可能是独显的余量，
   * 也可能是集显的勉强及格。不写清楚，两个读数看起来一模一样。
   */
  gpu: string;
  /** 当前渲染档位。`high` → `medium` → `low`，只降不升 */
  quality: Quality;
  /** 档位是否被 `?quality=` 钉死（钉死时不自适应） */
  qualityPinned: boolean;
}

/**
 * 读实际承担渲染的 GPU 名字。
 *
 * 用**已有的**上下文读，不新建一个：新建上下文会占显存、触发一次 GPU 唤醒，
 * 如果这个调用发生在测量之前，测到的就不是稳态了。这条约束是把它做成
 * 一个接收 `renderer` 的纯函数的原因 —— 放在测量脚本里另开 canvas 是错的。
 *
 * 读不到就返回 `masked` / `unknown`，不猜。Safari 与部分隐私模式会屏蔽这个扩展。
 */
function readGpuName(renderer: THREE.WebGLRenderer): string {
  try {
    const gl = renderer.getContext();
    const ext = gl.getExtension("WEBGL_debug_renderer_info");
    if (!ext) return "masked";
    return String(gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) ?? "unknown");
  } catch {
    return "unknown";
  }
}

/**
 * 渲染质量的自动降级。
 *
 * ## 为什么"保证 60 FPS"必须靠降级来实现
 *
 * 集显是个很宽的区间：Intel UHD 620 与 Iris Xe 差三倍，同一块 iGPU 在
 * 插电与省电模式下又差一倍。任何一组写死的参数，在某个端点上一定是错的。
 * 所以不能"调到一个值"，只能"给一个能自己往下走的台阶"（分档见 `quality.ts`）。
 *
 * ## 三条防止它自己变成故障的约束
 *
 * 1. **只降不升。** 升档需要重新评估，而重新评估意味着又一次抖动窗口 ——
 *    在"刚好及格"的机器上，升 → 掉帧 → 降 会变成一个可见的循环，
 *    比一直停在低档难看得多。
 * 2. **最多降两档，降到底就定住。** `degrade()` 到底返回 `null`，
 *    这里据此收手 —— 没有这个终点，"一直降"会变成一个逻辑上永不结束的过程。
 * 3. **`?quality=<档位>` 可以把档位钉死。** 截图与评审需要确定性：
 *    同一份代码在两台机器上产出两张不一样的图，评审时无法比对。
 *
 * ## 判据是**中位数**，不是平均
 *
 * 一次 GC 尖峰会把平均值拉过阈值，于是在一台本来及格的机器上触发降级，
 * 而它其实没问题。中位数对单次尖峰不敏感，对"整体掉到 30 帧"很敏感 ——
 * 这正是我们要区分的那件事。
 */
const SAMPLE_WARMUP_FRAMES = 20;
const SAMPLE_WINDOW_FRAMES = 60;
/** 超过预算 15% 才降级。留一点余量，避免在 16.7 与 16.8 之间反复判定。 */
const DEGRADE_THRESHOLD = FRAME_BUDGET_MS * 1.15;

export class GeoScene {
  private renderer: THREE.WebGLRenderer;
  private scene = new THREE.Scene();
  private camera: THREE.OrthographicCamera;
  private disposables: Disposable[] = [];
  /** 构造时读一次，之后不变（GPU 名字不会在运行期变） */
  private gpuName = "unknown";

  private mode: GeoMode;
  private data: SectionData;
  private clock = 0;
  private raf = 0;
  private width = 1;
  private height = 1;
  /** 每场景单位对应多少 CSS 像素，由 `frameCamera` 维护 */
  private pixelsPerUnit = 40;

  /** 当前渲染档位。由 `sampleFrame()` 自适应下调，或由 `?quality=` 钉死。 */
  private quality: Quality = "high";
  private qualityPinned = false;
  /** 最近一窗的帧耗时（毫秒） */
  private samples: number[] = [];
  private seenFrames = 0;
  private lastFrameAt = 0;
  /** 已经定档（够快，或已降到底），不再采样 */
  private settled = false;
  /** `devicePixelRatio` 原值。降级时只改上限，不改这个 */
  private devicePixelRatio = 1;

  private uniforms: ParticleUniforms = {
    uTime: { value: 0 },
    //: 粒子直径，单位是**场景单位**（0.05 = 5 m）。真正的像素尺寸由
    //: `uSizeScale`（每场景单位多少像素）在片元前换算，见 resize()。
    uSize: { value: 0.05 },
    uSizeScale: { value: 40 },
    uPixelRatio: { value: 1 },
    uOpacity: { value: 0 },
    uTintCore: { value: new THREE.Color() },
    uTintEdge: { value: new THREE.Color() },
  };

  /** 可以被 HTML 叠加层定位的锚点（世界坐标） */
  private anchors = new Map<string, THREE.Vector3>();

  /**
   * 每次渲染之后回调。
   *
   * 标签用 HTML 而不是 WebGL 里的文字：中文字体做成字形图集要几 MB，
   * 而 HTML 标签直接复用系统字体、支持选中、支持 `prefers-reduced-motion`。
   * 代价是相机一动就要重新投影 —— 所以由场景在渲染后主动通知，
   * 而不是让组件再起一个 rAF 循环去猜场景什么时候画完了。
   */
  onAfterRender?: () => void;

  private plumeFace: THREE.Mesh | null = null;
  private plumeVolume: THREE.Mesh | null = null;
  private plumeHatch: THREE.Texture | null = null;
  private particles: THREE.Points | null = null;
  private isobars: THREE.Line[] = [];
  private monitorMeshes = new Map<string, THREE.Mesh>();

  constructor(
    canvas: HTMLCanvasElement,
    data: SectionData,
    mode: GeoMode,
    opts: GeoSceneOptions = {},
  ) {
    this.data = data;
    this.mode = mode;
    if (opts.quality) {
      this.quality = opts.quality;
      this.qualityPinned = true;
    }

    this.renderer = new THREE.WebGLRenderer({
      canvas,
      antialias: mode === "deck",
      alpha: true,
      powerPreference: opts.lowPower ? "low-power" : mode === "deck" ? "high-performance" : "low-power",
    });
    this.renderer.setClearColor(0x000000, 0);
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.gpuName = readGpuName(this.renderer);

    // 正交而不是透视，理由见 `camera.ts`
    this.camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 1, 400);

    this.build();
    this.applyMode(mode);
  }

  // ── 构建 ────────────────────────────────────────────────────────────────

  /**
   * 登记一个需要释放的资源，原样返回（方便 `new X(track(...))`）。
   *
   * 写成箭头属性而不是方法：它要被当成值塞进 `BuildContext` 传给各个
   * builder，方法脱离 `this` 之后会静默丢失绑定 —— 那时 `this.disposables`
   * 是 `undefined`，报错发生在构建阶段，而构建阶段的错误会被
   * `GeoSectionCanvas` 的 catch 吞成一句"3D 视图不可用"。
   */
  private track = <T extends Disposable>(x: T): T => {
    this.disposables.push(x);
    return x;
  };

  private build() {
    const ctx: BuildContext = {
      scene: this.scene,
      data: this.data,
      anchors: this.anchors,
      track: this.track,
      uniforms: this.uniforms,
    };

    buildStrata(ctx);
    buildSurfaceGrid(ctx);
    buildWells(ctx);

    const plume = buildPlume(ctx);
    this.plumeFace = plume.face;
    this.plumeVolume = plume.volume;
    this.plumeHatch = plume.hatch;
    this.isobars = plume.isobars;

    this.particles = buildParticles(ctx);
    this.monitorMeshes = buildMonitors(ctx);
  }

  // ── 质量 ────────────────────────────────────────────────────────────────

  /**
   * 装饰层的可见性 = 模式 **且** 档位。
   *
   * 两个条件必须在这里合并，不能各写一处：`applyMode` 管模式，`setQuality`
   * 管档位，谁后调用谁说了算 —— 于是"先降级再切模式"和"先切模式再降级"
   * 会得到两个不同的画面，而两者看起来都正常。合并成一个纯函数之后，
   * 结果只取决于 (mode, quality)，与调用顺序无关。
   */
  private applyVisibility() {
    const deck = this.mode === "deck";
    if (this.particles) this.particles.visible = deck && QUALITY_PARTICLES[this.quality];
    if (this.plumeVolume) this.plumeVolume.visible = deck && QUALITY_VOLUME[this.quality];
  }

  /** 把 `devicePixelRatio` 按当前档位封顶后应用。 */
  private applyPixelRatio() {
    const ratio = Math.min(this.devicePixelRatio, QUALITY_PIXEL_RATIO_CAP[this.quality], 2);
    this.renderer.setPixelRatio(ratio);
    this.renderer.setSize(this.width, this.height, false);
    this.uniforms.uPixelRatio.value = ratio;
    // 改 pixelRatio 会改变画布的实际像素数，但 left/right/top/bottom 不变，
    // 所以取景不用重算 —— 只有"每单位多少像素"要跟着变（粒子尺寸用它）。
    this.uniforms.uSizeScale.value = this.pixelsPerUnit;
    if (this.mode !== "deck") this.renderOnce();
  }

  private setQuality(q: Quality) {
    if (q === this.quality) return;
    this.quality = q;
    this.applyPixelRatio();
    this.applyVisibility();
  }

  /**
   * 采样一帧，够了就判定。
   *
   * 只在指挥舱的渲染循环里被调用（工程师模式停止循环，没有帧可测，
   * 也不需要测 —— 它已经是静态一帧了）。
   */
  private sampleFrame(now: number) {
    const prev = this.lastFrameAt;
    this.lastFrameAt = now;
    if (this.settled || this.qualityPinned) return;
    if (prev === 0) return;

    const dt = now - prev;
    // 标签页切回来、首帧、GPU 驱动卡顿都会产生几百毫秒的异常值。
    // 它们不是"这一帧很慢"，是"这一帧根本没在画"，不能参与判定。
    if (dt <= 0 || dt > 1000) return;

    // 前若干帧含着色器编译与纹理上传，算进去会得到"启动耗时"
    if (this.seenFrames++ < SAMPLE_WARMUP_FRAMES) return;

    this.samples.push(dt);
    if (this.samples.length < SAMPLE_WINDOW_FRAMES) return;

    const sorted = [...this.samples].sort((a, b) => a - b);
    const median = sorted[Math.floor(sorted.length / 2)] ?? 0;
    this.samples = [];

    if (median <= DEGRADE_THRESHOLD) {
      this.settled = true;
      return;
    }
    const next = nextQuality(this.quality, median, DEGRADE_THRESHOLD);
    if (!next) {
      this.settled = true;
      return;
    }
    this.setQuality(next);
  }

  // ── 模式 ────────────────────────────────────────────────────────────────

  /**
   * 双模切换。
   *
   * 指挥舱：体 + 粒子 + 抗锯齿 + 高功率 GPU 偏好 + 相机缓慢漂移。
   * 工程师：**只保留剖面轮廓与等压环**，关掉粒子与体，相机锁死，
   * 并且停止渲染循环 —— 现场工控机的 GPU 不该为一张静态图持续耗电，
   * 而一张会动的图在强光下读起来更慢。
   */
  applyMode(mode: GeoMode) {
    this.mode = mode;
    const deck = mode === "deck";

    this.applyVisibility();
    this.uniforms.uOpacity.value = deck ? 1 : 0;

    for (const line of this.isobars) {
      const m = line.material as THREE.LineDashedMaterial;
      // 基准值存在 userData 里。直接写 `m.opacity * 1.6` 会在每次切模式时
      // 再乘一次 —— 来回切几次之后等压线要么全白要么全没了，而且不报错。
      const base = (m.userData.baseOpacity as number | undefined) ?? m.opacity;
      m.userData.baseOpacity = base;
      m.opacity = deck ? base : Math.min(0.55, base * 1.9);
    }

    if (this.plumeFace) {
      const mat = this.plumeFace.material as THREE.MeshBasicMaterial;
      // 工程师模式：填充线 + 稍实的底色。没有粒子撑着，0.09 会让羽流
      // 几乎看不见，而它是这张图上唯一一个"CO₂ 到哪儿了"的直接答案。
      mat.map = deck ? null : this.plumeHatch;
      mat.opacity = deck ? 0.09 : 0.34;
      mat.needsUpdate = true;
    }

    if (deck) this.start();
    else this.stopAndRenderOnce();
  }

  updateData(data: SectionData) {    // 几何随数据变（羽流范围、测点等级）。这里只改颜色与可见性，
    // 不重建几何 —— 重建会让相机与锚点全部失效。
    this.data = data;
    updateMonitorColors(this.monitorMeshes, data);
  }

  // ── 相机与渲染 ──────────────────────────────────────────────────────────

  resize(width: number, height: number, pixelRatio: number) {
    this.width = Math.max(1, width);
    this.height = Math.max(1, height);
    this.devicePixelRatio = Math.max(1, pixelRatio);
    // 正交相机没有 `aspect` 字段 —— 长宽比由 left/right/top/bottom 直接表达，
    // 所以这里只调 frameCamera，不要再写 `camera.aspect = ...`
    this.pixelsPerUnit = frameCamera(this.camera, this.width, this.height);
    // 像素比与尺寸都由 `applyPixelRatio` 统一施加（它要按当前档位封顶）
    this.applyPixelRatio();
  }

  private renderOnce() {
    this.renderer.render(this.scene, this.camera);
    this.onAfterRender?.();
  }

  private tick = () => {
    const now = performance.now();
    this.clock += 0.016;
    this.uniforms.uTime.value = this.clock;

    // 相机**不动**。
    //
    // 上一版让相机缓慢漂移，视觉上确实"活"，但代价是整张图每帧平移几个像素：
    // 深度尺刻度、层名、测点标签全都跟着晃。看板上唯一该动的是"数据在流动"
    // 这件事本身，由粒子承担；图框自己动起来，只会让人以为剖面在变。
    this.renderOnce();
    this.sampleFrame(now);
    this.raf = requestAnimationFrame(this.tick);
  };

  start() {
    if (this.raf) return;
    // 上一轮停下时留下的时间戳不能拿来算帧间隔 —— 停多久就算多久，
    // 会得到一个几百毫秒的"帧耗时"，而它会被采样窗口当成真实掉帧。
    this.lastFrameAt = 0;
    this.raf = requestAnimationFrame(this.tick);
  }

  stopAndRenderOnce() {
    if (this.raf) {
      cancelAnimationFrame(this.raf);
      this.raf = 0;
    }
    this.lastFrameAt = 0;
    this.samples = [];
    // 工程师模式：相机回到取景位置并停住，只渲染一帧。
    // 现场工控机的 GPU 不该为一张静态图持续耗电，而会动的图在强光下读得更慢。
    this.pixelsPerUnit = frameCamera(this.camera, this.width, this.height);
    this.uniforms.uSizeScale.value = this.pixelsPerUnit;
    this.renderOnce();
  }

  /**
   * 渲染开销快照。
   *
   * 只读 `renderer.info`，不在渲染路径上，也不触发任何 GPU 工作 ——
   * 所以可以随时调用（`?perf=1` 的探针每秒读一次）。
   *
   * 为什么要在产品代码里留这个方法：`drawCalls` 与 `triangles` 是**唯一**
   * 能证明"这帧很轻"的客观数据。只看源码里的 `COUNT = 3000` 说明不了问题 ——
   * 3000 个点画 1 次和画 3000 次差三个数量级，而两种写法看起来都很"省"。
   */
  stats(): SceneStats {
    const info = this.renderer.info;
    return {
      drawCalls: info.render.calls,
      triangles: info.render.triangles,
      points: info.render.points,
      lines: info.render.lines,
      geometries: info.memory.geometries,
      textures: info.memory.textures,
      programs: info.programs?.length ?? 0,
      width: this.width,
      height: this.height,
      pixelRatio: this.renderer.getPixelRatio(),
      animating: this.raf !== 0,
      gpu: this.gpuName,
      quality: this.quality,
      qualityPinned: this.qualityPinned,
    };
  }

  /** 把世界坐标锚点投影到画布像素坐标，供 HTML 标签定位。 */
  screenAnchors(): ScreenAnchor[] {
    const out: ScreenAnchor[] = [];
    const v = new THREE.Vector3();
    for (const [id, world] of this.anchors) {
      v.copy(world).project(this.camera);
      out.push({
        id,
        x: (v.x * 0.5 + 0.5) * this.width,
        y: (-v.y * 0.5 + 0.5) * this.height,
        visible: v.z < 1 && v.z > -1,
      });
    }
    return out;
  }

  dispose() {
    if (this.raf) cancelAnimationFrame(this.raf);
    this.raf = 0;
    for (const d of this.disposables) d.dispose();
    this.disposables = [];
    this.scene.clear();
    this.renderer.dispose();
    this.monitorMeshes.clear();
    this.anchors.clear();
  }
}

export type { MonitorAnchor };
