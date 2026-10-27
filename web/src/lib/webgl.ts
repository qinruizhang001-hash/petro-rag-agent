/**
 * WebGL 可用性探测。
 *
 * 单独一个文件，是为了让调用方**不必**为了做这个判断而静态引入 `three`
 * （那会把 160 KB 的库拖进首屏 chunk，而首屏真正急的是左栏告警）。
 *
 * 探测本身会创建一个上下文再丢掉。这是刻意的：`WebGLRenderingContext` 的
 * 存在只能说明浏览器实现了接口，不能说明这台机器的驱动能让它跑起来 ——
 * 老 Intel 集显 + 远程桌面会话就是典型的"接口在、上下文创建失败"。
 */
export function webglAvailable(): boolean {
  try {
    const canvas = document.createElement("canvas");
    const ctx =
      canvas.getContext("webgl2") ??
      canvas.getContext("webgl") ??
      canvas.getContext("experimental-webgl");
    return Boolean(ctx);
  } catch {
    return false;
  }
}
