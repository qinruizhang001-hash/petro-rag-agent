import { useMemo, useState } from "react";
import { SEVERITY_CLASS, SEVERITY_META, formatValue } from "../lib/severity";
import {
  ROW_HEIGHT,
  bodyHeight,
  buildSensorRows,
  formatMargin,
  visibleRange,
} from "../lib/sensorTable";
import type { Alarm, ThresholdStatus } from "../lib/types";
import { unitLabel } from "../lib/units";
import { cn } from "../lib/utils";

/**
 * 测点全量表 —— 剖面之外，现场还需要一屏能扫完的纯表格。
 *
 * ## 为什么卡片列表不够
 *
 * 左栏的卡片是按**严重度**排的：需要动作的用可展开的告警卡，其余用紧凑
 * 指标卡。那个排法回答的是"我该先处理哪一条"，但它不回答另一个问题：
 *
 * > 18 个测点里，哪几个离自己的线最近？
 *
 * 卡片里每一张的"阈值 / 余量"是同一行文字，横向扫不起来；而 `notice`
 * 那几条被夹在中间，看起来和 `normal` 一样。表格把这些变成同构的列。
 *
 * ## 为什么列是「距 warn 线」而不是「距上限」
 *
 * "上限"对每个测点含义不同（`high` 方向是 warn 上方的线，`low` 方向是
 * 下方的线）。统一成"距 warn 线的余量，负 = 已越线"之后，一列就能横向
 * 比较，不必按方向分两张表。**具体命中了哪一级**由状态列回答。
 *
 * ## 行高必须是常量
 *
 * 窗口化按 `top = index × ROW_HEIGHT` 定位，行高一变，滚动条位置和
 * 渲染区间就对不上 —— 表现为"滚到某处会跳一下"。所以行内文字一律
 * `truncate`，不做换行。
 */
export function SensorTable({
  alarms,
  thresholds,
}: {
  alarms: Alarm[];
  thresholds: ThresholdStatus;
}) {
  const rows = useMemo(() => buildSensorRows(alarms, thresholds), [alarms, thresholds]);
  const [scrollTop, setScrollTop] = useState(0);

  /*
   * 视口高度用**常量**而不是量出来的 `clientHeight`：
   * 量一次就要在 resize / 字体加载后重量，而窗口化只需要"大概几行"。
   * 多算一行不影响正确性（多渲染一行），少算才会留白。
   *
   * 12 行 × 22 px = 264 px。选它是因为 520 px 的窄屏下限里还要放上限校核
   * 与盲区提示（合计约 200 px）—— 表格不能把上面两个结论挤出视口。
   */
  const viewportRows = 12;
  const viewportHeight = viewportRows * ROW_HEIGHT;
  const { start, end } = visibleRange(scrollTop, viewportHeight, rows.length);

  return (
    <div className="text-2xs">
      {/* 表头在滚动区外面：滚的时候列名不该跟着走 */}
      <div className="flex items-center gap-1 border-b border-line px-1 pb-0.5 text-ink-2">
        <span className="min-w-0 flex-1">测点</span>
        <span className="w-[4.5rem] shrink-0 text-right">读数</span>
        <span className="w-16 shrink-0 text-right" title="到 warn 线的余量，负值表示已越线">
          距 warn 线
        </span>
        <span className="w-14 shrink-0 text-right">状态</span>
      </div>

      <div
        className="overflow-y-auto"
        style={{ height: viewportHeight }}
        onScroll={(e) => setScrollTop(e.currentTarget.scrollTop)}
      >
        <div style={{ height: bodyHeight(rows.length) }} className="relative">
          {/* 用两个占位块把可视行推到正确位置，比绝对定位少一次 reflow */}
          <div style={{ height: start * ROW_HEIGHT }} />
          {rows.slice(start, end).map((row) => {
            const meta = SEVERITY_META[row.severity];
            return (
              <div
                key={row.alarmId}
                className="flex items-center gap-1 px-1"
                style={{ height: ROW_HEIGHT }}
                title={
                  row.undecidable
                    ? `${row.label}：${row.undecidableReason}`
                    : `${row.label} · ${row.direction === "high" ? "上限" : "下限"} warn=${row.warn} ${unitLabel(row.unit)}`.trim()
                }
              >
                <span className="min-w-0 flex-1 truncate">
                  <span className="text-ink-0">{row.label}</span>
                  {row.stageLabel && (
                    <span className="ml-1 text-ink-2">{row.stageLabel}</span>
                  )}
                </span>
                <span className="flex w-[4.5rem] shrink-0 items-baseline justify-end gap-0.5 font-num text-ink-0">
                  {/* 数字永不截断，单位可以 —— 截断单位比截断数字更可接受 */}
                  <span className="shrink-0">
                    {row.value === null ? "—" : formatValue(row.value)}
                  </span>
                  {unitLabel(row.unit) && (
                    <span className="min-w-0 truncate text-ink-2">{unitLabel(row.unit)}</span>
                  )}
                </span>
                <span
                  className={cn(
                    "w-16 shrink-0 text-right font-num",
                    row.margin === null
                      ? "text-ink-2"
                      : row.margin < 0
                        ? SEVERITY_CLASS[row.severity].text
                        : "text-ink-1",
                  )}
                >
                  {formatMargin(row)}
                </span>
                <span
                  className={cn(
                    "w-14 shrink-0 text-right",
                    // 不可判定用虚线下划线而不是换个颜色：色觉障碍、强光、
                    // 灰度打印都会让"灰"和"浅灰"变成同一个东西。
                    row.undecidable
                      ? "text-sev-unknown underline decoration-dotted"
                      : SEVERITY_CLASS[row.severity].text,
                  )}
                >
                  {row.undecidable ? "不可判定" : meta.label}
                </span>
              </div>
            );
          })}
          <div style={{ height: Math.max(0, rows.length - end) * ROW_HEIGHT }} />
        </div>
      </div>

      {/*
        表格底部必须说明"没有阈值的测点不是正常" —— 表格里那三行只显示
        一个破折号，而破折号在别处（没有读数）也出现，两种"不知道"混在一起
        读起来像"这三个没数"。它们的区别是：一个有读数但没人定线，
        一个连数都没有。
      */}
      {rows.some((r) => r.undecidable) && (
        <p className="mt-0.5 border-t border-line pt-0.5 text-ink-2">
          不可判定的 {rows.filter((r) => r.undecidable).length} 项与「正常」不同：
          它们是「没人说过多少算有问题」，不是「没问题」。
        </p>
      )}
    </div>
  );
}
