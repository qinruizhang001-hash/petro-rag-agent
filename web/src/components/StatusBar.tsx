import { BACKEND_LABEL } from "../lib/health";
import type { BackendHealth, BackendState } from "../lib/health";
import type { PushStatus } from "../lib/push";
import { SEVERITY_CLASS, SEVERITY_META } from "../lib/severity";
import { sourceHint, sourceLabel } from "../lib/source";
import type { DataSource, SourceChoice } from "../lib/source";
import type { ThresholdOrigin } from "../lib/thresholds";
import type { AlarmCounts } from "../lib/types";
import { cn } from "../lib/utils";

interface StatusBarProps {
  counts: AlarmCounts;
  /**
   * 实际生效的数据源与"是不是 URL 覆盖来的"。
   *
   * 两者必须一起传：只传 `source` 的话，"构建默认是 fixture、这次被
   * `?source=live` 覆盖"这件事在界面上无法区分 —— 而它决定了这张截图
   * 能不能当"接了后端"的证据。
   */
  choice: SourceChoice;
  /** 这份产物构建时的默认值。悬停说明里要写出来。 */
  buildSource: DataSource;
  period: [string, string];
  /**
   * 实时推送状态。`idle` 表示**没有配置**推送地址 —— 此时不渲染任何角标。
   *
   * 不渲染而不是显示"未连接"：默认部署本来就没有推送通道，把它画成一个
   * 灰色的警告会让每张截图都带着一个"有问题"的暗示，而它其实只是没配。
   * 但一旦配了（`connecting` / `open` / `closed`），状态就必须可见 ——
   * 推送断了而界面照旧显示旧数字，是最难发现的一类故障。
   */
  push?: PushStatus;
  /**
   * 后端可达性。
   *
   * 与"数据来源"是**两件事**：fixture 模式下界面读本地快照，后端在不在
   * 都不影响显示。于是"界面全绿 + 后端挂了三天"是一个可能的状态，
   * 而它读起来像"系统正常"。两个指示器必须并排出现。
   */
  health: BackendHealth;
  /** 点一下重探。缺省时不渲染成按钮（只显示状态） */
  onProbe?: () => void;
  /**
   * 阈值从哪来、可不可信。
   *
   * 这是底栏里**唯一**说明"判据本身是不是真的"的一格。没有它，读者会
   * 默认每条读数旁边那条线是本项目的设计限值 —— 而本地跑的时候它是
   * 示例文件里的【替换】占位值。读数、余量、分级全部算得对，判据是假的。
   *
   * 传 `describeThresholds()` 的结果，判定逻辑在 `lib/thresholds.ts`。
   */
  thresholdOrigin?: ThresholdOrigin;
}

/** 后端状态 → 配色。字面量，理由同 `severity.ts`。 */
const BACKEND_TONE: Record<BackendState, string> = {
  unknown: "border-line-strong text-ink-2",
  checking: "border-line-strong text-ink-2",
  ok: "border-sev-normal/50 text-sev-normal",
  degraded: "border-sev-warn/60 text-sev-warn",
  unreachable: "border-sev-alert/60 text-sev-alert",
};

/** 推送状态 → 文案与配色。`idle` 不在此表里：它不渲染。 */
const PUSH_META: Record<Exclude<PushStatus, "idle">, { label: string; cls: string }> = {
  connecting: { label: "推送连接中", cls: "border-line-strong text-ink-2" },
  open: { label: "推送已连接", cls: "border-sev-normal/50 text-sev-normal" },
  closed: { label: "推送已断开 · 重试中", cls: "border-sev-warn/60 text-sev-warn" },
};

/** 计数顺序：从严重到正常，最后是不可判定 —— 与左栏排序一致。 */
const ORDER = [
  "critical",
  "alert",
  "warn",
  "indeterminate",
  "notice",
  "normal",
] as const;

/**
 * 底部状态栏。
 *
 * 六态**全部**列出，包括计数为 0 的那些。"0 严重"是有信息量的：
 * 它告诉读者"我看过了，确实没有"，而只显示非零项会让人分不清
 * "没有严重告警"和"这一栏不存在"。
 */
export function StatusBar({
  counts,
  choice,
  buildSource,
  period,
  push = "idle",
  health,
  onProbe,
  thresholdOrigin,
}: StatusBarProps) {
  return (
    <footer className="flex flex-wrap items-center gap-x-3 gap-y-1 border-t border-line bg-surface-1 px-3 py-1.5 text-2xs text-ink-2">
      <span className="tracking-wide">状态计数</span>

      {ORDER.map((sev) => {
        const n = counts[sev];
        const cls = SEVERITY_CLASS[sev];
        const zero = n === 0;
        return (
          <span
            key={sev}
            className={cn("inline-flex items-baseline gap-1", zero && "opacity-40")}
          >
            <span
              aria-hidden
              className={cn("inline-block size-1.5 translate-y-px rounded-full", cls.dot)}
            />
            <span className="text-ink-1">{SEVERITY_META[sev].label}</span>
            <span className={cn("tnum", zero ? "text-ink-2" : cls.text)}>{n}</span>
          </span>
        );
      })}

      <span className="ml-auto flex items-center gap-3">
        <span className="tnum">
          {period[0].slice(5, 16).replace("T", " ")} – {period[1].slice(11, 16)}
        </span>

        {/*
          阈值出处。占位值时描虚线 + 警示色 —— 与"数据源被覆盖"同一套视觉：
          虚线在这块底栏里的含义固定为"这一屏有某个前提不是你以为什么"。

          只在 needsAttention 时描虚线。`label` 有值但不需警告（文件名仍叫
          example 但内容已填过）就普通样式 —— 把一份填好的模板也喊成"假的"，
          下次真出问题时没人会看。
        */}
        {thresholdOrigin?.label ? (
          <span
            className={cn(
              "rounded-panel border px-1.5 py-0.5",
              thresholdOrigin.needsAttention
                ? "border-sev-warn/60 text-sev-warn border-dashed"
                : "border-line-strong text-ink-2",
            )}
            title={thresholdOrigin.hint}
          >
            {thresholdOrigin.label}
          </span>
        ) : null}
        <span
          className={cn(
            "rounded-panel border px-1.5 py-0.5",
            choice.source === "fixture"
              ? "border-line-strong text-ink-2"
              : "border-acc/40 text-acc",
            // 被 URL 覆盖时描一道虚线：它说明"这一屏不是这份产物的默认行为"。
            // 只靠文案区分不够 —— 底栏 12px 字，`· ?source 覆盖` 是全文最末尾
            // 的六个字，截图上常常被裁掉。
            choice.overridden && "border-dashed",
          )}
          title={sourceHint(choice, buildSource)}
        >
          {sourceLabel(choice)}
        </span>

        {/*
          后端可达性。点击重探 —— 探针只跑一次的话，界面会永远停在
          打开页面那一刻的结论，而"后端刚才起来了"这件事没人能看见。
        */}
        <button
          type="button"
          onClick={onProbe}
          disabled={!onProbe || health.state === "checking"}
          className={cn(
            "rounded-panel border px-1.5 py-0.5",
            BACKEND_TONE[health.state],
            onProbe && "cursor-pointer hover:brightness-125",
          )}
          title={[
            health.detail || "后端 /health 正常",
            health.auth ? `鉴权：${health.auth === "disabled" ? "未启用" : "已启用"}` : "",
            health.at ? `探测于 ${health.at}` : "",
            onProbe ? "点一下重新探测" : "",
          ]
            .filter(Boolean)
            .join(" · ")}
        >
          {BACKEND_LABEL[health.state]}
          {health.state === "degraded" && health.failing.length > 0
            ? ` · ${health.failing.length} 项`
            : ""}
        </button>
        {push !== "idle" && (
          <span
            className={cn("rounded-panel border px-1.5 py-0.5", PUSH_META[push].cls)}
            title="WebSocket 推送通道状态。断开时会自动重连，但这段时间的数据不会自动补上。"
          >
            {PUSH_META[push].label}
          </span>
        )}
      </span>
    </footer>
  );
}
