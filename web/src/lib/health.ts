/**
 * 后端可达性探针。
 *
 * ## 为什么它和"数据来源"是两个指示器
 *
 * `VITE_OPS_SOURCE=fixture` 时界面读的是本地快照，后端在不在都不影响显示。
 * 于是出现一种最难解释的状态：**界面全绿、数字齐全、后端已经挂了三天**。
 * 演示时这没问题；一旦有人拿它当"系统正常运行"的证据，就是一个错误的结论。
 * 所以数据来源与后端可达性必须分开显示 —— 一个是"这份数字从哪来"，
 * 一个是"那台机器还活着吗"。
 *
 * ## 探针打的是 `/health`，不是 `/api/*`
 *
 * 后端 `/health` 刻意不鉴权（Docker healthcheck、K8s probe、nginx upstream
 * 检查都不会带 token）。探一个需要 token 的接口，会把"没配 token"报成
 * "后端挂了"，而这两件事的处置完全不同。
 *
 * ## 404 与超时不能合并
 *
 * 404 是"这个路径没有被路由"（多半是 nginx / vite 代理没配），
 * 超时是"包发出去了没人应答"。前者是配置问题、后者是服务问题。
 * 两者都显示成"后端不可达"时，排查会从错误的方向开始。
 */

const HEALTH_URL = import.meta.env.VITE_HEALTH_URL ?? "/health";
const TIMEOUT_MS = 4000;

export type BackendState = "unknown" | "checking" | "ok" | "degraded" | "unreachable";

export interface BackendHealth {
  state: BackendState;
  /** 后端自报的 `status`（`ok` / `degraded`）。拿不到时为空串 */
  reported: string;
  /** 后端自报的鉴权状态（`enabled` / `disabled`） */
  auth: string;
  /** 未通过的依赖检查项名（ollama / chroma / graph…） */
  failing: string[];
  /** 人话原因。成功时为空串 */
  detail: string;
  /** 上次探测时刻（本地时间字符串），`null` = 还没探过 */
  at: string | null;
}

export const HEALTH_INITIAL: BackendHealth = {
  state: "unknown",
  reported: "",
  auth: "",
  failing: [],
  detail: "",
  at: null,
};

interface HealthBody {
  status?: string;
  auth?: string;
  checks?: Record<string, { ok?: boolean; detail?: string } | undefined>;
}

function nowLabel(): string {
  const d = new Date();
  const p = (n: number) => String(n).padStart(2, "0");
  return `${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`;
}

function failingOf(body: HealthBody): string[] {
  const checks = body.checks ?? {};
  return Object.keys(checks).filter((k) => checks[k]?.ok !== true);
}

/**
 * 探一次。**永不抛** —— 探针抛异常会让整个界面进入错误态，
 * 而它只是一条附加信息，不该有能力改变主流程。
 */
export async function probeBackend(): Promise<BackendHealth> {
  const at = nowLabel();
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);

  try {
    const res = await fetch(HEALTH_URL, {
      signal: controller.signal,
      cache: "no-store",
      // 探针不跟随到别的源：跨源重定向会把"本地后端没起来"变成一个
      // 看起来成功的第三方响应。
      redirect: "error",
    });

    if (!res.ok) {
      return {
        state: "unreachable",
        reported: "",
        auth: "",
        failing: [],
        detail:
          res.status === 404
            ? `${HEALTH_URL} 返回 404：这个路径没有被路由（检查 nginx 的 /health 或 vite 的代理）`
            : `${HEALTH_URL} 返回 HTTP ${res.status}`,
        at,
      };
    }

    const body = (await res.json()) as HealthBody;
    const failing = failingOf(body);
    const reported = body.status ?? "";
    // 后端自报 degraded 时**不**降级成 unreachable：服务在，是依赖不在。
    // 这两种状态的处置完全不同（重启 vs 起 Ollama）。
    const state: BackendState =
      reported === "ok" && failing.length === 0 ? "ok" : "degraded";

    return {
      state,
      reported,
      auth: body.auth ?? "",
      failing,
      detail: failing.length > 0 ? `依赖未就绪：${failing.join("、")}` : "",
      at,
    };
  } catch (e: unknown) {
    const aborted = e instanceof Error && e.name === "AbortError";
    return {
      state: "unreachable",
      reported: "",
      auth: "",
      failing: [],
      detail: aborted
        ? `${HEALTH_URL} 超时（${TIMEOUT_MS} ms 无应答）`
        : `${HEALTH_URL} 请求失败：${e instanceof Error ? e.message : String(e)}`,
      at,
    };
  } finally {
    clearTimeout(timer);
  }
}

/** 状态 → 文案。`unknown` 与 `checking` 都不写成"正常"。 */
export const BACKEND_LABEL: Record<BackendState, string> = {
  unknown: "后端未探测",
  checking: "后端探测中",
  ok: "后端可达",
  degraded: "后端降级",
  unreachable: "后端不可达",
};
