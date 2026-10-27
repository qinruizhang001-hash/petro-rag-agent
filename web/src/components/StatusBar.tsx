import type { PushStatus } from "../lib/push";
import { SEVERITY_CLASS, SEVERITY_META } from "../lib/severity";
import type { AlarmCounts } from "../lib/types";
import { cn } from "../lib/utils";

interface StatusBarProps {
  counts: AlarmCounts;
  source: string;
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
}

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
export function StatusBar({ counts, source, period, push = "idle" }: StatusBarProps) {
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
        <span
          className={cn(
            "rounded-panel border px-1.5 py-0.5",
            source === "fixture"
              ? "border-line-strong text-ink-2"
              : "border-acc/40 text-acc",
          )}
          title={
            source === "fixture"
              ? "数据来自 scripts/capture_ops_fixtures.py 抓取的真实响应，不是手写样例"
              : "数据来自运行中的后端"
          }
        >
          {source === "fixture" ? "fixture（真实响应快照）" : "live"}
        </span>
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
