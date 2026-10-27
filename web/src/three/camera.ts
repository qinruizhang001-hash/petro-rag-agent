/**
 * 正交相机的取景与缩放。
 *
 * ## 为什么是正交
 *
 * 剖面的深度尺必须**线性**。透视下 0–400 m 会比 900–1300 m 占更多像素，
 * 于是同一张图上两段等长的深度画出来不一样长 —— 而深度尺的全部意义
 * 就是"可以按比例读"。立体感由切块的厚度、侧壁和地表网格承担，
 * 不需要透视来凑。
 *
 * ## 为什么取景要反解而不是目测
 *
 * 把相机推远再目测出来的取景，在 16:9 和 21:9 上会得到两个不同的构图，
 * 而评审时大屏往往是 21:9。所以用固定的需求尺寸反解 left/right/top/bottom。
 */
import * as THREE from "three";
import { DEPTH_MAX, HALF_WIDTH_M, SCALE } from "../lib/section";

/** 相机距离。正交相机下这个值只影响近远裁剪面，不影响成像大小。 */
const CAMERA_DISTANCE = 120;

/**
 * 取景：让剖面**竖着装下**，同时横向留出余量。
 *
 * @returns 每场景单位对应多少 CSS 像素 —— 粒子尺寸、线宽这类"物理尺寸"
 *          都要经过它换算，否则窗口一缩放，粒子就变成一堆点或一片糊。
 */
export function frameCamera(
  camera: THREE.OrthographicCamera,
  width: number,
  height: number,
): number {
  const aspect = width / Math.max(1, height);
  const needH = DEPTH_MAX / SCALE + 2.2; // 剖面 + 地表上方余量
  const needW = ((HALF_WIDTH_M * 2) / SCALE) * 1.04;

  let halfH = needH / 2;
  let halfW = halfH * aspect;
  // 竖着装不下就按宽度反推 —— 21:9 的大屏与 4:3 的工控机都要能读
  if (halfW < needW / 2) {
    halfW = needW / 2;
    halfH = halfW / aspect;
  }

  camera.left = -halfW;
  camera.right = halfW;
  camera.top = halfH;
  camera.bottom = -halfH;

  // 俯角约 9°、偏航约 3°：能看见切块的顶面与一侧侧壁，但不至于把
  // 剖面压成一条斜带。角度再大，层厚就开始"看起来不一样厚"了。
  const target = new THREE.Vector3(0, -DEPTH_MAX / SCALE / 2, 0);
  const dir = new THREE.Vector3(0.055, 0.16, 1).normalize();
  camera.position.copy(target).addScaledVector(dir, CAMERA_DISTANCE);
  camera.lookAt(target);
  camera.updateProjectionMatrix();

  return height / (camera.top - camera.bottom);
}
