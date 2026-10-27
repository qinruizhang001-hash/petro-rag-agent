/**
 * 从 `index.css` 读色值。
 *
 * 不在这里另抄一份十六进制：3D 与 DOM 用同一个令牌，改一次两边一起变。
 * 抄一份的话，某天调整了 `--sev-critical`，左栏的告警卡变了、
 * 剖面上的测点没变，而这种不一致没人会去比对两张截图。
 *
 * 令牌是 `R G B` 三元组（见 index.css 的注释），所以按空白切分即可。
 *
 * ## 为什么是函数而不是常量表
 *
 * `getComputedStyle` 必须在 DOM 就绪后才能读到值，而模块顶层求值时
 * 样式表可能还没挂上 —— 那时读到空串，回退成默认色，而且**不会报错**，
 * 只是整个场景变成一种颜色。所以每次构建时现读。
 */
import * as THREE from "three";

export function token(name: string, fallback = "#58b8c9"): THREE.Color {
  if (typeof document === "undefined") return new THREE.Color(fallback);
  const raw = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const parts = raw.split(/[\s,]+/).map(Number);
  if (parts.length === 3) {
    const [r = 0, g = 0, b = 0] = parts;
    if ([r, g, b].every(Number.isFinite)) {
      return new THREE.Color().setRGB(r / 255, g / 255, b / 255, THREE.SRGBColorSpace);
    }
  }
  return new THREE.Color(fallback);
}
