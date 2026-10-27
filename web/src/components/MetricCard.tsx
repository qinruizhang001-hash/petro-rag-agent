import { SEVERITY_CLASS, describeMargin, formatValue, isIndeterminate } from "../lib/severity";
import { STAGE_SHORT } from "../lib/stages";
import type { Alarm } from "../lib/types";
import { cn } from "../lib/utils";
import { SeverityChip } from "./SeverityChip";

interface MetricCardProps {
  alarm: Alarm;
  unit: string;
  /**
   * 测点目录里的 warn 阈值。
   *
   * **不是** `alarm.threshold_value` —— 后者是"命中了哪一级的线"，
   * `normal` 时它是 `null`。拿它当阈值用，会让所有正常测点显示成
   * "无阈值"，与真正的盲区长得一模一样。
   */
  warnThreshold: number | null;
}

/**
 * 指标卡。
 *
 * 带**阈值余量**而不是只显示数值：
 *
 * ```
 * 捕集率            86.8 %
 * 阈值 88.0 · 超 warn 1.2
 * ```
 *
 * "86.8 %" 单独看无法判断好坏，"超 warn 1.2" 才能。方向由 `direction` 决定：
 * `high`（越高越坏）余量 = 阈值 − 当前值，`low`（越低越坏）反之。
 *
 * 这一条是这套界面里最容易被省掉的字段 —— 省掉之后卡片依然好看，
 * 只是读者必须自己心算，而人在连盯 8 小时后不会心算。
 */
export function MetricCard({ alarm, unit, warnThreshold }: MetricCardProps) {
  const cls = SEVERITY_CLASS[alarm.severity];
  const indet = isIndeterminate(alarm.severity);
  const margin = describeMargin(
    alarm.value,
    warnThreshold,
    alarm.threshold_level,
    alarm.threshold_value,
    alarm.direction,
  );

  return (
    <div
      className={cn(
        "rounded-panel border border-line border-l-2 bg-surface-2 px-2.5 py-1.5",
        cls.border,
        indet && "sev-unknown-ring",
      )}
    >
      <div className="flex items-baseline justify-between gap-2">
        <span className="truncate text-xs text-ink-1">{alarm.label}</span>
        <SeverityChip severity={alarm.severity} />
      </div>

      <div className="mt-0.5 flex items-baseline justify-between gap-2">
        {/*
          不可判定时显示破折号，**不显示 0**。
          这一重信号与灰色、虚线边框并列，缺一重就会被读成"正常"。
        */}
        <span className={cn("tnum text-lg leading-none", cls.text)}>
          {indet ? "—" : formatValue(alarm.value)}
          {!indet && <span className="ml-0.5 text-2xs text-ink-2">{unit}</span>}
        </span>
        <span className="text-2xs text-ink-2">{STAGE_SHORT[alarm.stage]}</span>
      </div>

      <div className="mt-0.5 text-2xs">
        {indet ? (
          // 不可判定时不能写"正常"，要把原因说出来：
          // 没配阈值 / 没有读数 / 仪表不可信 —— 三种原因的处理方式完全不同
          <span className="text-sev-unknown">{indeterminateReason(alarm)}</span>
        ) : (
          <span className={margin.breached ? "text-sev-alert" : "text-ink-2"}>
            {margin.referenceValue === null ? (
              "未配置阈值"
            ) : (
              <>
                {/* 越限时基准换成被越过的那条线，否则"阈值 20 · 超 1.6"自相矛盾 */}
                {margin.referenceLabel}{" "}
                <span className="tnum">{formatValue(margin.referenceValue)}</span>
                {" · "}
                <span className={cn("tnum", margin.breached && "font-semibold")}>
                  {margin.text}
                </span>
              </>
            )}
          </span>
        )}
      </div>
    </div>
  );
}

function indeterminateReason(alarm: Alarm): string {
  // 顺序有讲究：先看仪表、再看读数、最后才归到"没配阈值"。
  // 反过来写的话，一个仪表失效的测点会被说成"未配置阈值"，
  // 而这两件事的处置完全不同 —— 一个要找仪表班，一个要找工艺工程师。
  if (alarm.quality !== "good") {
    return `仪表${alarm.quality === "bad" ? "失效" : "可疑"} · 无法判定`;
  }
  if (alarm.value === null) return "无读数 · 无法判定";
  return "未配置阈值 · 无法判定";
}
