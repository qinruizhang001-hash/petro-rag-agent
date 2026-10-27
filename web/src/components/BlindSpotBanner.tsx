import { cn } from "../lib/utils";

interface BlindSpotBannerProps {
  /** 已声明但没配阈值的测点（`report.unconfigured`） */
  unconfigured: string[];
  /** 完全没收到读数的测点（`report.no_data`） */
  noData: string[];
  labels: Record<string, string>;
  className?: string;
}

/**
 * 盲区提示条。
 *
 * 这一条是整个界面最不能省的元素。
 *
 * 测点有读数、但没配阈值时，系统**判不了**。如果界面把它画成正常，
 * 看板会一直是绿的 —— 而那是这套系统唯一不会报错的失败：没有异常、
 * 没有日志、没有告警，只是有一块区域永远没人看。
 *
 * 用 `notice` 色而不是 `critical` 色：盲区不是故障，是"我们还没配置"。
 * 用红色会让它和真实告警抢注意力，久了就被忽略 —— 和告警疲劳同一个机制。
 */
export function BlindSpotBanner({
  unconfigured,
  noData,
  labels,
  className,
}: BlindSpotBannerProps) {
  const total = unconfigured.length + noData.length;
  if (total === 0) return null;

  return (
    <div
      className={cn(
        "sev-unknown-ring rounded-panel border border-sev-notice/60 bg-sev-notice/10 px-2.5 py-2",
        className,
      )}
      role="status"
    >
      <div className="flex items-baseline gap-1.5">
        <span className="text-2xs font-semibold tracking-wide text-sev-notice">
          已知盲区
        </span>
        <span className="tnum text-2xs text-ink-2">{total} 个测点</span>
      </div>

      {unconfigured.length > 0 && (
        <p className="mt-1 text-2xs leading-relaxed text-ink-1">
          <span className="text-ink-2">未配置阈值：</span>
          {unconfigured.map((k) => labels[k] ?? k).join("、")}
        </p>
      )}

      {noData.length > 0 && (
        <p className="mt-0.5 text-2xs leading-relaxed text-ink-1">
          <span className="text-ink-2">无读数：</span>
          {noData.map((k) => labels[k] ?? k).join("、")}
        </p>
      )}

      {/*
        这句话必须出现在界面上，而不是只写在文档里。
        "不可判定"与"正常"在视觉上只差一个灰度和一条虚线，
        没有这句话，读者会默认它是正常。
      */}
      <p className="mt-1 text-2xs text-ink-2">
        不可判定 ≠ 正常：这些测点有读数但无法判定，其数据不参与"是否正常"的结论。
      </p>
    </div>
  );
}
