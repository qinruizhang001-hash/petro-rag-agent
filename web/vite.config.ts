import react from "@vitejs/plugin-react";
import { defineConfig, loadEnv } from "vite";

// 中文注释：开发时把 /api 代理到后端，避免为了本地调试去开 CORS。
// 生产部署走 nginx 同源（见仓库根目录 nginx.conf），所以这里只在 dev 生效。
//
// 目标地址从 `.env` 的 OPS_API_TARGET 读，而不是 `process.env`：
// 引用 `process` 需要 @types/node，而那会把整套 node 全局塞进浏览器侧的
// 类型检查里 —— 之后在 src/ 里写 `fs.readFileSync` 也能过编译。
// 一个只在前端用的项目不该有那种"能过编译"。
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, ".", "");

  return {
    plugins: [react()],
    server: {
      port: 5180,
      proxy: {
        "/api": {
          target: env.OPS_API_TARGET || "http://127.0.0.1:8000",
          changeOrigin: true,
        },
      },
    },
    build: {
      // 中文注释：首屏体积是这套界面的已知风险（参考项目里有 50 MB 单页的先例）。
      // 图表库与 3D 一律按需 import，这里先把上限压到 600 KB 提醒自己别超。
      chunkSizeWarningLimit: 600,
    },
  };
});
