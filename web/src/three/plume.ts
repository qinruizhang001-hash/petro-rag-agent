/**
 * 羽流：剖面轮廓（数据）、3D 包络体（装饰）、等压环（数据）。
 *
 * 三者的身份必须分清，否则"那个发光的圆是什么"会变成每次评审的第一个问题：
 *
 * | 图元 | 身份 | 指挥舱 | 工程师 |
 * |---|---|---|---|
 * | 剖面轮廓 | **数据**（CO₂ 波及范围） | 有 | 有 |
 * | 3D 包络体 | 装饰（让人看出它是个体） | 有 | **无** |
 * | 等压虚线环 | **数据**（压力影响范围 > CO₂ 范围） | 有 | 有 |
 *
 * 工程师模式关掉的是"体"，**不是羽流本身**。第一版把整个羽流都当特效
 * 删掉了，结果工程师模式得到一个空的剖面 —— 而那正是现场最需要看的东西：
 * CO₂ 到底铺到哪一层、离监测井还有多远。
 */
import * as THREE from "three";
import { SCALE, plumeRadiusM } from "../lib/section";
import { PLUME_Z_SQUASH, THICKNESS } from "./context";
import type { BuildContext } from "./context";
import { token } from "./theme";

/**
 * 羽流的填充线纹理，只在工程师模式用。
 *
 * 没有粒子撑着的时候，一块半透明的实心色块在剖面上读起来像"容器"——
 * 它有明确的边界、均匀的内部，正好是固体给人的印象。而羽流是一团
 * 饱和度渐变的 CO₂。
 *
 * 工程制图里表示"体"的通用做法是填充线（hatch），它同时传达两件事：
 * 边界在哪里（线到哪儿为止）、内部是什么（有纹理而不是空的）。
 * 这条约定比任何动画都更被现场的人认。
 */
function plumeHatchTexture(): THREE.Texture | null {
  const size = 128;
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = size;
  const g = canvas.getContext("2d");
  if (!g) return null;
  g.clearRect(0, 0, size, size);
  g.strokeStyle = "rgba(88,184,201,0.42)";
  g.lineWidth = 1.6;
  for (let i = -size; i < size * 2; i += 19) {
    g.beginPath();
    g.moveTo(i, 0);
    g.lineTo(i + size, size);
    g.stroke();
  }
  const tex = new THREE.CanvasTexture(canvas);
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  // ShapeGeometry 的 UV 就是世界坐标，所以 repeat 直接是"每多少个场景单位
  // 铺一块"：0.4 → 每 2.5 单位（250 m）一块，线间距约 24 m
  tex.repeat.set(0.4, 0.4);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

/** 交给 `GeoScene.applyMode()` 的三件东西 —— 模式切换要动它们。 */
export interface PlumeHandles {
  /**
   * 剖面轮廓的填充面。
   *
   * 直接持有引用，而不是切模式时用 `scene.traverse` 找 `name === "plume-face"`：
   * 按字符串名字找图元，某次重命名之后会**静默失效** —— 工程师模式下
   * 羽流退回 0.09 透明度，几乎看不见，而没有任何地方会报错。
   */
  face: THREE.Mesh;
  /** 3D 包络体，仅指挥舱可见 */
  volume: THREE.Mesh;
  /** 工程师模式的填充线纹理，切模式时挂到 `face` 上 */
  hatch: THREE.Texture | null;
  /** 等压环，切模式时改透明度（基准值存在 `userData.baseOpacity`） */
  isobars: THREE.Line[];
}

export function buildPlume(ctx: BuildContext): PlumeHandles {
  const { scene, data, anchors, track } = ctx;
  const p = data.plume;
  const topY = -p.apex / SCALE;
  const botY = -p.base / SCALE;
  const cx = p.cx / SCALE;
  const faceZ = THICKNESS / 2 + 0.03;

  // ① 剖面轮廓 —— 这是**数据**，两个模式都留
  const profile: THREE.Vector2[] = [];
  const SEG = 64;
  // 顶缘 4% 的高度里收 10% 半径：CO₂ 在盖层下的边缘不是一刀切，
  // 饱和度是渐变的。不收口的话，顶部是一条硬横线 + 两个直角，
  // 整条羽流会被读成一只玻璃杯，而不是一团气体。
  const rim = (t: number) => 1 - 0.1 * Math.max(0, (t - 0.96) / 0.04);
  for (let i = 0; i <= SEG; i++) {
    const t = i / SEG;
    const r = (plumeRadiusM(p, t) * rim(t)) / SCALE;
    profile.push(new THREE.Vector2(cx + r, botY + (topY - botY) * t));
  }
  for (let i = SEG; i >= 0; i--) {
    const t = i / SEG;
    const r = (plumeRadiusM(p, t) * rim(t)) / SCALE;
    profile.push(new THREE.Vector2(cx - r, botY + (topY - botY) * t));
  }
  const shape = new THREE.Shape(profile);
  const faceGeo = track(new THREE.ShapeGeometry(shape));
  /**
   * ## 三个透明面一律**不写** `side: THREE.DoubleSide`
   *
   * 实测（`renderer.info`，1680×1000）：三个 DoubleSide 的透明网格让
   * draw call 与三角形各自翻倍 —— 33 → 30 calls、8,508 → 5,164 triangles，
   * 多出的 128 + 16 + 3,200 = 3,344 与逐对象求和完全吻合。
   *
   * 集显的瓶颈是像素填充，而这三个恰好是这张图上面积最大的半透明面。
   * 更关键的是：没有任何一处需要看背面 —— 相机固定，只从 +Z 方向看，
   * 而 DoubleSide 换来的"从里面也能看见"在这张图上没有对应的问题。
   */
  const faceMat = track(
    new THREE.MeshBasicMaterial({
      color: token("--acc", "#58b8c9"),
      transparent: true,
      opacity: 0.09,
      depthWrite: false,
    }),
  );
  // 工程师模式用的填充线纹理，切模式时挂上去
  const hatch = plumeHatchTexture();
  if (hatch) track(hatch);
  const face = new THREE.Mesh(faceGeo, faceMat);
  face.position.z = faceZ;
  face.renderOrder = 5;
  face.name = "plume-face";
  scene.add(face);

  // 轮廓描边：数据要有边。没有边，读者分不清哪是羽流哪是地层纹理
  const outlineGeo = track(
    new THREE.BufferGeometry().setFromPoints(
      profile.map((v) => new THREE.Vector3(v.x, v.y, faceZ + 0.005)),
    ),
  );
  const outline = new THREE.Line(
    outlineGeo,
    track(
      new THREE.LineBasicMaterial({
        color: token("--acc", "#58b8c9"),
        transparent: true,
        // 0.75 的实线会把羽流读成"容器"。CO₂ 前缘的饱和度是渐变的，
        // 边界只该是一条参考线，不该是器壁。
        opacity: 0.5,
      }),
    ),
  );
  outline.renderOrder = 6;
  scene.add(outline);

  // ② 3D 包络体 —— 装饰（让人看出羽流是个体），仅指挥舱
  const ring = 40;
  const positions: number[] = [];
  const indices: number[] = [];
  for (let i = 0; i <= ring; i++) {
    const t = i / ring;
    const y = botY + (topY - botY) * t;
    const r = plumeRadiusM(p, t) / SCALE;
    for (let j = 0; j <= ring; j++) {
      const a = (j / ring) * Math.PI * 2;
      positions.push(cx + Math.cos(a) * r, y, Math.sin(a) * r * PLUME_Z_SQUASH);
    }
  }
  for (let i = 0; i < ring; i++) {
    for (let j = 0; j < ring; j++) {
      const a = i * (ring + 1) + j;
      const b = a + ring + 1;
      indices.push(a, b, a + 1, b, b + 1, a + 1);
    }
  }
  const volGeo = track(new THREE.BufferGeometry());
  volGeo.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  volGeo.setIndex(indices);
  volGeo.computeVertexNormals();

  const volume = new THREE.Mesh(
    volGeo,
    track(
      new THREE.MeshBasicMaterial({
        color: token("--acc"),
        transparent: true,
        opacity: 0.075,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
      }),
    ),
  );
  volume.renderOrder = 5;
  scene.add(volume);

  // ③ 等压环 —— 数据：压力影响范围比 CO₂ 波及范围更大，这是两件事
  const centerY = botY + (topY - botY) * 0.72;
  const isobars: THREE.Line[] = [];
  // 只画两圈、且紧贴羽流边缘。第一版画到 1.9 倍半径，三圈大虚线横跨
  // 整个剖面，看起来像三个互不相干的环 —— 读者会以为那是别的东西
  for (const [i, factor] of [1.06, 1.22].entries()) {
    const r = (p.radius * factor) / SCALE;
    const pts: THREE.Vector3[] = [];
    const N = 96;
    for (let k = 0; k <= N; k++) {
      const a = (k / N) * Math.PI * 2;
      pts.push(
        new THREE.Vector3(cx + Math.cos(a) * r, centerY + Math.sin(a) * r * 0.34, faceZ + 0.004),
      );
    }
    const g = track(new THREE.BufferGeometry().setFromPoints(pts));
    const m = track(
      new THREE.LineDashedMaterial({
        color: token("--acc-2"),
        transparent: true,
        opacity: 0.2 - i * 0.07,
        dashSize: 0.09,
        gapSize: 0.13,
        depthWrite: false,
      }),
    );
    const line = new THREE.Line(g, m);
    line.computeLineDistances();
    line.renderOrder = 6;
    isobars.push(line);
    scene.add(line);
  }

  anchors.set("plume-apex", new THREE.Vector3(cx, topY, THICKNESS / 2));
  anchors.set("plume-edge", new THREE.Vector3(cx + p.radius / SCALE, topY, THICKNESS / 2));

  return { face, volume, hatch, isobars };
}
