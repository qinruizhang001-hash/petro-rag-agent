/**
 * 注入井：井筒线 + 井口 + 射孔段。
 *
 * 三件东西都是**数据**（CO₂ 从哪儿进储层），但都不能抢视线 ——
 * 这张图的主角是羽流和测点。所以全部低透明度、细线，
 * 只在地表附近留一个短竖管告诉读者"从上面灌下去的"。
 */
import * as THREE from "three";
import { INJECTION_DEPTH, SCALE } from "../lib/section";
import { THICKNESS } from "./context";
import type { BuildContext } from "./context";
import { token } from "./theme";

export function buildWells(ctx: BuildContext): void {
  const { scene, anchors, track } = ctx;
  const faceZ = THICKNESS / 2 + 0.02;

  const injectTop = new THREE.Vector3(0, 0, faceZ);
  const injectBottom = new THREE.Vector3(0, -INJECTION_DEPTH / SCALE, faceZ);
  const geo = track(new THREE.BufferGeometry().setFromPoints([injectTop, injectBottom]));
  const line = new THREE.Line(
    geo,
    track(
      new THREE.LineBasicMaterial({
        color: token("--acc", "#58b8c9"),
        transparent: true,
        // 0.5 太亮：井筒线会读成一条网格线，而它是注入路径，不是参考线
        opacity: 0.34,
      }),
    ),
  );
  line.renderOrder = 4;
  scene.add(line);

  // 井口：一个短竖管 + 顶圈，让人知道 CO₂ 是从地面上灌下去的
  const headGeo = track(new THREE.CylinderGeometry(0.045, 0.045, 0.34, 8, 1, true));
  const head = new THREE.Mesh(
    headGeo,
    track(
      new THREE.MeshBasicMaterial({
        color: token("--acc"),
        transparent: true,
        opacity: 0.7,
      }),
    ),
  );
  head.position.set(0, 0.17, faceZ);
  head.renderOrder = 4;
  scene.add(head);

  // 射孔段：一条短横线，标出 CO₂ 的入口
  const perfGeo = track(
    new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(-0.5, -INJECTION_DEPTH / SCALE, faceZ),
      new THREE.Vector3(0.5, -INJECTION_DEPTH / SCALE, faceZ),
    ]),
  );
  const perf = new THREE.Line(
    perfGeo,
    track(
      new THREE.LineBasicMaterial({
        color: token("--acc"),
        transparent: true,
        opacity: 0.8,
      }),
    ),
  );
  perf.renderOrder = 4;
  scene.add(perf);

  anchors.set("wellhead", new THREE.Vector3(0, 0.34, THICKNESS / 2));
  anchors.set("perforation", new THREE.Vector3(0.62, -INJECTION_DEPTH / SCALE, THICKNESS / 2));
}
