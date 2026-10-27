/**
 * 地层板与地表网格。
 *
 * 这一层是**背景**，不是数据。它的全部职责是给"层位"一个视觉落点 ——
 * 告警卡说"归因到盖层"，读者要能在这张图上指出盖层是哪一块。
 *
 * ## 为什么纹理是画出来的
 *
 * 一张 8 KB 的 PNG 也是一个要维护的二进制资源，而这两张纹理只有十几行
 * 代码，改起来比换图快，也不占仓库体积（CI 有一道 1 MB 的守卫，
 * 专门抓误提交的二进制）。
 */
import * as THREE from "three";
import { HALF_WIDTH_M, SCALE, SEAL_BASE } from "../lib/section";
import type { Stratum } from "../lib/section";
import { THICKNESS } from "./context";
import type { BuildContext } from "./context";
import { token } from "./theme";

/** 层带的程序化纹理。`null` 表示 canvas 上下文拿不到（极少数降级环境）。 */
function stratumTexture(
  stratum: Stratum,
  widthUnits: number,
  heightUnits: number,
): THREE.Texture | null {
  const size = 256;
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = size;
  const g = canvas.getContext("2d");
  if (!g) return null;

  // 底色：白 + 低 alpha，实际颜色由材质的 color 决定。
  // 0.10 太淡 —— 五个层带只剩边框可见，"归因到层位"就没有视觉落点了
  g.fillStyle = "rgba(255,255,255,0.30)";
  g.fillRect(0, 0, size, size);

  if (stratum.texture === "hatch") {
    // 泥岩斜线：45°，间距 12 px
    g.strokeStyle = "rgba(255,255,255,0.38)";
    g.lineWidth = 1.4;
    for (let i = -size; i < size * 2; i += 12) {
      g.beginPath();
      g.moveTo(i, 0);
      g.lineTo(i + size, size);
      g.stroke();
    }
  } else {
    // 孔隙介质：不规则圆点。用固定种子的伪随机，避免每次刷新纹理都变
    let seed = 20261027;
    const rnd = () => {
      seed = (seed * 1103515245 + 12345) & 0x7fffffff;
      return seed / 0x7fffffff;
    };
    g.fillStyle = "rgba(255,255,255,0.30)";
    for (let i = 0; i < 220; i++) {
      const r = 0.8 + rnd() * 1.8;
      g.beginPath();
      g.arc(rnd() * size, rnd() * size, r, 0, Math.PI * 2);
      g.fill();
    }
  }

  const tex = new THREE.CanvasTexture(canvas);
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  // 一个 tile 覆盖 2 个场景单位（200 m），纹理密度与层厚无关
  tex.repeat.set(Math.max(1, widthUnits / 2), Math.max(0.5, heightUnits / 2));
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

export function buildStrata(ctx: BuildContext): void {
  const { scene, data, anchors, track } = ctx;
  const widthUnits = (HALF_WIDTH_M * 2) / SCALE;

  for (const stratum of data.strata) {
    const topY = -stratum.top / SCALE;
    const bottomY = -stratum.bottom / SCALE;
    const h = topY - bottomY;

    const geo = track(new THREE.BoxGeometry(widthUnits, h, THICKNESS));
    const tex = stratumTexture(stratum, widthUnits, h);
    if (tex) track(tex);

    const mat = track(
      new THREE.MeshBasicMaterial({
        color: token(stratum.colorVar, "#16303c"),
        map: tex,
        transparent: true,
        /**
         * 一律 1.0，层与层之间的明暗只由颜色决定。
         *
         * 上一版写的是 `seal ? 1.0 : 0.82`，本意是让盖层更实，结果
         * **把明暗顺序整个弄反了**：没有纹理的层（覆盖层）材质上没有 map，
         * 于是最终 alpha 就等于 opacity（0.82）；有纹理的层 alpha 是
         * `opacity × 底纹自身的 0.26`。同一条属性在两条代码路径上
         * 差 4 倍 —— 覆盖层比盖层还亮，而盖层是这张图上最该被看见的一层。
         *
         * 现在所有层都挂同一张底纹，alpha 语义统一，盖层靠颜色区分。
         */
        opacity: 1.0,
        depthWrite: false,
      }),
    );

    const mesh = new THREE.Mesh(geo, mat);
    mesh.position.set(0, (topY + bottomY) / 2, 0);
    mesh.renderOrder = 1;
    scene.add(mesh);

    // 只把 geometry / material 交给 disposal 表 —— `Object3D` 自己没有
    // `dispose()`，把整棵树塞进去会让 `dispose()` 在退出时抛错，
    // 而那个错误发生在组件卸载时，几乎不会被看到
    const edgesGeo = track(new THREE.EdgesGeometry(geo));
    const edgesMat = track(
      new THREE.LineBasicMaterial({
        color: token("--line-2", "#7ab2cc"),
        transparent: true,
        opacity: stratum.seal ? 0.85 : 0.45,
        depthWrite: false,
      }),
    );
    const edges = new THREE.LineSegments(edgesGeo, edgesMat);
    edges.position.copy(mesh.position);
    edges.renderOrder = 2;
    scene.add(edges);

    // 层名锚点：左边缘、层中点
    anchors.set(
      `stratum:${stratum.key}`,
      new THREE.Vector3(-widthUnits / 2, (topY + bottomY) / 2, THICKNESS / 2),
    );
  }

  // 盖层底界：一条贯穿的高亮线。羽流离它多远，是这张图上最该被读到的数
  const sealY = -SEAL_BASE / SCALE;
  const sealGeo = track(
    new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(-widthUnits / 2, sealY, THICKNESS / 2 + 0.01),
      new THREE.Vector3(widthUnits / 2, sealY, THICKNESS / 2 + 0.01),
    ]),
  );
  const sealLine = new THREE.Line(
    sealGeo,
    track(
      new THREE.LineBasicMaterial({
        color: token("--acc-2", "#7e9cb8"),
        transparent: true,
        opacity: 0.55,
      }),
    ),
  );
  sealLine.renderOrder = 3;
  scene.add(sealLine);
}

/** 地表网格：纯装饰，但它是"这是地面"的唯一提示，不能省。 */
export function buildSurfaceGrid(ctx: BuildContext): void {
  const grid = new THREE.GridHelper(30, 30, token("--acc").getHex(), token("--acc").getHex());
  grid.position.set(0, 0, -2);
  const mat = grid.material as THREE.LineBasicMaterial;
  mat.transparent = true;
  mat.opacity = 0.1;
  mat.depthWrite = false;
  grid.renderOrder = 0;
  ctx.scene.add(grid);
  ctx.track(grid.geometry);
  ctx.track(mat);
}
