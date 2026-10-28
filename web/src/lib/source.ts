/**
 * 数据源选择：`fixture`（本地快照）还是 `live`（打运行中的后端）。
 *
 * ## 为什么单独抽一个纯函数文件
 *
 * 这件事原本只有一行：`import.meta.env.VITE_OPS_SOURCE ?? "fixture"`，
 * 是**构建期常量**。后果是：一份产物只能是一种模式 —— 想给评审看"真的
 * 连着后端"，就得重新 `npm run build`；build 完 dist 变成 live，下次想
 * 放快照又得再 build 一次。而"现在这份 dist 到底是哪种"只能靠底栏那行
 * 小字分辨，那行字是全屏最容易被忽略的地方。
 *
 * 抽出来的第二个理由：`import.meta.env` 在 Node 里不存在，测试要靠
 * esbuild 的 `define` 把它换掉。判定逻辑只要留在纯函数里，就能直接用
 * `node --test` 覆盖 —— 否则"URL 覆盖到底生效没有"这件事，只能在真实
 * 浏览器里靠肉眼确认，而它在没有后端的那台机器上表现为"整屏加载失败"。
 */

export type DataSource = "fixture" | "live";

export interface SourceChoice {
  source: DataSource;
  /** 这个值是否由 `?source=` 覆盖而来（界面要如实说，不能让人以为改了构建）。 */
  overridden: boolean;
}

/** 构建期默认值。只认 `"live"`，其余（含拼错、大小写不符）一律当 fixture。 */
export function buildSource(raw: string | undefined): DataSource {
  return raw === "live" ? "live" : "fixture";
}

/**
 * 运行时覆盖：`?source=live` / `?source=fixture`。
 *
 * **拼错的值一律回落到构建值，不猜**。`?source=lvie` 若被猜成 live，
 * 一份本意是演示快照的产物就会去打后端；在没有后端的那台机器上，它表现
 * 为整屏"数据加载失败"，而讲的人会以为是后端挂了。回落至少还能看。
 * 大小写同理（`?source=LIVE` 不生效）—— 判定越"聪明"，出错时越难解释。
 *
 * 是否生效不靠猜：底栏的数据来源标记会显示实际生效的那一个，覆盖时
 * 另加一句"本次由 ?source 覆盖"。
 */
export function resolveSource(search: string, build: DataSource): SourceChoice {
  const q = new URLSearchParams(search).get("source");
  if (q === "live" || q === "fixture") return { source: q, overridden: q !== build };
  return { source: build, overridden: false };
}

/**
 * 从当前地址解析（浏览器里才有 `window`；Node 测试里直接返回构建值）。
 *
 * 这个薄壳存在的理由：`api.ts` 与 `kb.ts` 各写一遍 `typeof window` 判断，
 * 迟早有一处忘了改 —— 而那表现为"底栏说 live、知识库还是快照"，同一屏
 * 两个数据源。判定只有一份，读地址的入口也只有一份。
 */
export function choiceFromLocation(build: DataSource): SourceChoice {
  if (typeof window === "undefined") return { source: build, overridden: false };
  return resolveSource(window.location.search, build);
}

/** 底栏那一格的字。三种情形必须能分开读，否则"覆盖生效没有"看不出来。 */
export function sourceLabel(choice: SourceChoice): string {
  const base =
    choice.source === "fixture" ? "fixture（真实响应快照）" : "live · 读数采样回放";
  return choice.overridden ? `${base} · ?source 覆盖` : base;
}

/** 底栏那一格的悬停说明。要同时说清"这次是什么"和"构建默认是什么"。 */
export function sourceHint(choice: SourceChoice, build: DataSource): string {
  const what =
    choice.source === "fixture"
      ? "数据来自 scripts/capture_ops_fixtures.py 抓取的真实响应，不是手写样例。注意：它不代表后端此刻可达。"
      : "告警与归因由运行中的后端实算。输入是抓 fixture 时录下的那批读数（回放），不是实时时序 —— 后端 /api/ops/* 是无状态的，读数必须由调用方送进去。";
  if (!choice.overridden) return `${what}（构建默认，可用 ?source= 临时切换）`;
  const b = build === "fixture" ? "fixture" : "live";
  return `${what}本次由 URL 参数 ?source=${choice.source} 覆盖，这份产物构建时默认是 ${b}。`;
}
