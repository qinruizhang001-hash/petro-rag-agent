import react from "@vitejs/plugin-react";
import { defineConfig, loadEnv } from "vite";

// 中文注释：开发时把 /api 代理到后端，避免为了本地调试去开 CORS。
// 生产部署走 nginx 同源（见仓库根目录 nginx.conf），所以这里只在 dev / preview 生效。
//
// 目标地址从 `.env` 的 OPS_API_TARGET 读，而不是 `process.env`：
// 引用 `process` 需要 @types/node，而那会把整套 node 全局塞进浏览器侧的
// 类型检查里 —— 之后在 src/ 里写 `fs.readFileSync` 也能过编译。
// 一个只在前端用的项目不该有那种"能过编译"。
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, ".", "");
  const target = env.OPS_API_TARGET || "http://127.0.0.1:8000";

  /*
   * 两条代理：
   * - `/api`：业务接口。
   * - `/health`：后端可达性探针（不鉴权，见 api_server.py）。生产环境由
   *   nginx 的 `location = /health` 转给后端。这里不配的话，探针会打到
   *   vite 自己身上：dev 拿到 404、preview 拿到 500，而界面会显示
   *   "后端不可达" —— 那只说明代理没配，不是后端没起来。
   *
   * dev 与 preview 各要一份：`vite preview` **不继承** `server.proxy`，
   * 而 README 里的截图流程用的正是 preview。少配一份，同一份代码在
   * 两种模式下的底栏会显示两个不同的后端状态。
   */
  const proxy = {
    "/api": { target, changeOrigin: true },
    "/health": { target, changeOrigin: true },
  };

  return {
    plugins: [react()],
    server: { port: 5180, proxy },
    preview: { port: 4173, proxy },
    build: {
      // 中文注释：首屏体积是这套界面的已知风险（参考项目里有 50 MB 单页的先例）。
      // 图表库与 3D 一律按需 import，这里先把上限压到 600 KB 提醒自己别超。
      chunkSizeWarningLimit: 600,
    },
  };
});
