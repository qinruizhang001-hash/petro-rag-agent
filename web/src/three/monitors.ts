/**
 * 测点球 + 外环 + 深度尺锚点。
 *
 * 测点是这张图上唯一一个"会变色"的图元，也是唯一一个六级颜色全都用到的
 * 地方（见 `lib/severity.ts`）。三重信号叠加，是因为深色底上单一信号
 * 不够用：
 *
 * 1. **颜色** —— 六级语义色
 * 2. **形状** —— 不可判定用空心球（"灰点"和"暗青点"很难分辨，空心/实心一眼能分）
 * 3. **外环** —— 非 normal 的测点加一圈虚线，让它在满屏地层纹理里能被扫到
 */
import * as THREE from "three";
import { HALF_WIDTH_M, RULER_DEPTHS, SCALE } from "../lib/section";
import { THICKNESS } from "./context";
import type { BuildContext } from "./context";
import { token } from "./theme";

/** 测点等级 → 语义色。`null` 与 `indeterminate` 都落到 `--sev-unknown`。 */
function severityColor(severity: string | null): THREE.Color {
  if (severity === null || severity === "indeterminate") {
    return token("--sev-unknown", "#6e7c88");
  }
  return token(`--sev-${severity}`, "#6e7c88");
}

export function buildMonitors(ctx: BuildContext): Map<string, THREE.Mesh> {
  const { scene, data, anchors, track } = ctx;
  const faceZ = THICKNESS / 2 + 0.05;
  const meshes = new Map<string, THREE.Mesh>();

  for (const m of data.monitors) {
    const color = severityColor(m.severity);

    // 不可判定用**空心球**：形状是第三重信号（另两重是灰与破折号），
    // 深色底上"灰点"和"暗青点"很难分辨，空心/实心一眼就能分开
    const hollow = m.severity === "indeterminate" || m.severity === null;
    const geo = track(new THREE.SphereGeometry(0.15, 16, 12));
    const mat = track(
      new THREE.MeshBasicMaterial({
        color,
        transparent: true,
        opacity: hollow ? 0.22 : 0.95,
        depthWrite: false,
      }),
    );
    const sphere = new THREE.Mesh(geo, mat);
    sphere.position.set(m.x / SCALE, -m.depth / SCALE, faceZ);
    sphere.renderOrder = 8;
    scene.add(sphere);
    meshes.set(m.id, sphere);

    // 外环：非 normal 的测点加一圈，让它在满屏地层纹理里能被扫到
    const active =
      m.severity !== null && m.severity !== "normal" && m.severity !== "indeterminate";
    if (active || hollow) {
      const ringPts: THREE.Vector3[] = [];
      const N = 48;
      const rad = 0.28;
      for (let k = 0; k <= N; k++) {
        const a = (k / N) * Math.PI * 2;
        ringPts.push(new THREE.Vector3(Math.cos(a) * rad, Math.sin(a) * rad, 0));
      }
      const rg = track(new THREE.BufferGeometry().setFromPoints(ringPts));
      const rm = track(
        new THREE.LineDashedMaterial({
          color,
          transparent: true,
          opacity: hollow ? 0.55 : 0.8,
          // 空心球那圈是"破折号"语义（不确定），所以真的画成短虚线；
          // 有等级的测点画成实线环（dashSize 10 / gapSize 0）
          dashSize: hollow ? 0.06 : 10,
          gapSize: hollow ? 0.05 : 0,
        }),
      );
      const ring = new THREE.Line(rg, rm);
      ring.position.copy(sphere.position);
      ring.computeLineDistances();
      ring.renderOrder = 9;
      scene.add(ring);
    }

    anchors.set(`monitor:${m.id}`, new THREE.Vector3(m.x / SCALE, -m.depth / SCALE, faceZ));
  }

  // 深度尺锚点（右侧）
  const rulerX = HALF_WIDTH_M / SCALE - 0.4;
  for (const d of RULER_DEPTHS) {
    anchors.set(`ruler:${d}`, new THREE.Vector3(rulerX, -d / SCALE, THICKNESS / 2));
  }

  return meshes;
}

/** 数据刷新时只改颜色与透明度，不重建几何（重建会让相机与锚点全部失效）。 */
export function updateMonitorColors(meshes: Map<string, THREE.Mesh>, data: BuildContext["data"]): void {
  for (const m of data.monitors) {
    const mesh = meshes.get(m.id);
    if (!mesh) continue;
    const mat = mesh.material as THREE.MeshBasicMaterial;
    const hollow = m.severity === "indeterminate" || m.severity === null;
    mat.color = severityColor(m.severity);
    mat.opacity = hollow ? 0.22 : 0.95;
  }
}
