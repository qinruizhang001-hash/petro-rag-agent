/**
 * CO₂ 粒子流。
 *
 * ## 一个刻意的限制：粒子不能跑出包络
 *
 * 位移上限被压在**局部半径的 12%**，而且只在原地做小幅环绕。
 * 让粒子自由飘看起来更"流体"，但那样一来图上显示的羽流范围就变成了
 * 粒子群的包围盒 —— 一个由随机数决定的值，而不是模型给的范围。
 * 在一个用来判断"CO₂ 有没有到监测井"的界面上，这是不能接受的。
 *
 * ## 性能：3000 个点 = 1 次 draw call
 *
 * 全部粒子在**一个** `THREE.Points` 里，位移在顶点着色器里算，CPU 每帧
 * 只写一个 `uTime`。所以「3000 个粒子」和「1 个粒子」的开销是同一量级 ——
 * 真正决定帧率的是片元数量（叠加混合下每个粒子都要混合一次），
 * 这也是 COUNT 从 5200 降到 3000 的原因：不只是性能，更是亮度。
 */
import * as THREE from "three";
import { SCALE, plumeRadiusM } from "../lib/section";
import { PLUME_Z_SQUASH } from "./context";
import type { BuildContext } from "./context";
import { token } from "./theme";

export function buildParticles(ctx: BuildContext): THREE.Points {
  const { scene, data, track, uniforms } = ctx;
  const p = data.plume;
  const topY = -p.apex / SCALE;
  const botY = -p.base / SCALE;
  const cx = p.cx / SCALE;
  // 3000 而不是 5200：叠加混合下粒子密度直接决定亮度，
  // 上一版在羽流轴心叠成一片实心白，读起来像一块塑料而不是一团气体
  const COUNT = 3000;

  const pos = new Float32Array(COUNT * 3);
  const seed = new Float32Array(COUNT);
  const tint = new Float32Array(COUNT);
  const size = new Float32Array(COUNT);

  // 固定种子：每次刷新看到的粒子分布一致，截图可比对
  let s = 987654321;
  const rnd = () => {
    s = (s * 1103515245 + 12345) & 0x7fffffff;
    return s / 0x7fffffff;
  };

  for (let i = 0; i < COUNT; i++) {
    // t 偏向顶部：CO₂ 在盖层下聚集，那里的密度最高
    const t = Math.pow(rnd(), 0.62);
    const y = botY + (topY - botY) * t;
    const rMax = plumeRadiusM(p, t) / SCALE;
    // 立方根 → 体积均匀；再乘 0.92 留一点边，别贴着轮廓线
    const rr = Math.cbrt(rnd()) * rMax * 0.92;
    const a = rnd() * Math.PI * 2;

    pos[i * 3] = cx + Math.cos(a) * rr;
    pos[i * 3 + 1] = y;
    pos[i * 3 + 2] = Math.sin(a) * rr * PLUME_Z_SQUASH;

    seed[i] = rnd();
    tint[i] = rr / Math.max(1e-6, rMax); // 0 = 轴心，1 = 边缘
    size[i] = 0.55 + rnd() * 0.85;
  }

  const geo = track(new THREE.BufferGeometry());
  geo.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
  geo.setAttribute("aSeed", new THREE.Float32BufferAttribute(seed, 1));
  geo.setAttribute("aTint", new THREE.Float32BufferAttribute(tint, 1));
  geo.setAttribute("aSize", new THREE.Float32BufferAttribute(size, 1));

  uniforms.uTintCore.value = token("--acc");
  uniforms.uTintEdge.value = token("--acc-2");

  const mat = track(
    new THREE.ShaderMaterial({
      uniforms,
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      vertexShader: /* glsl */ `
        attribute float aSeed;
        attribute float aTint;
        attribute float aSize;

        uniform float uTime;
        uniform float uSize;
        uniform float uSizeScale;
        uniform float uPixelRatio;

        varying float vTint;
        varying float vAlpha;

        void main() {
          vec3 p = position;
          float s = aSeed * 6.2831853;

          // 位移上限 = 局部半径的 12%（见 buildParticles 的注释）
          float amp = 0.045;
          p.x += sin(uTime * 0.55 + s) * amp;
          p.z += cos(uTime * 0.42 + s * 1.3) * amp * 0.6;
          p.y += sin(uTime * 0.85 + s * 1.7) * amp * 0.55;

          vec4 mv = modelViewMatrix * vec4(p, 1.0);
          gl_Position = projectionMatrix * mv;

          // 正交相机没有透视，粒子的屏幕尺寸与距离无关 ——
          // 上一版沿用透视的除以 -mv.z，在 120 单位的相机距离下把 5 m 的
          // 粒子算成 0.1 px，粒子其实一直没被看见，看到的只是叠加出来的雾。
          gl_PointSize = aSize * uSize * uSizeScale * uPixelRatio;

          vTint = aTint;
          // 缓慢明暗呼吸，让"数据在流动"这件事被看见，但不足以吸引视线
          vAlpha = 0.30 + 0.70 * (0.5 + 0.5 * sin(uTime * 1.1 + s * 2.3));
          // 径向衰减：羽流边缘的 CO₂ 饱和度本来就低。
          // 不做衰减时，粒子会一直堆到轮廓线上，整条羽流看起来像一袋沙子，
          // 而不是一团边界模糊的气体。
          vAlpha *= 1.0 - 0.62 * vTint * vTint;
        }
      `,
      fragmentShader: /* glsl */ `
        uniform vec3 uTintCore;
        uniform vec3 uTintEdge;
        uniform float uOpacity;

        varying float vTint;
        varying float vAlpha;

        void main() {
          vec2 d = gl_PointCoord - 0.5;
          float r = length(d);
          if (r > 0.5) discard;
          // 软边：硬边粒子在深色底上会读成一堆小方块
          float a = smoothstep(0.5, 0.05, r);
          vec3 c = mix(uTintCore, uTintEdge, vTint);
          gl_FragColor = vec4(c, a * vAlpha * uOpacity * 0.62);
        }
      `,
    }),
  );

  const points = new THREE.Points(geo, mat);
  points.renderOrder = 7;
  points.frustumCulled = false;
  scene.add(points);
  return points;
}
