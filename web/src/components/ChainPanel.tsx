import { useMemo } from "react";
import { SEVERITY_CLASS, formatValue, isIndeterminate, worstOf } from "../lib/severity";
import { STAGE_HINT, STAGE_LABELS, STAGE_ORDER } from "../lib/stages";
import type { Alarm, AlarmReport, Stage, ThresholdStatus } from "../lib/types";
import { cn } from "../lib/utils";
import { Panel } from "./Panel";
import { SeverityChip } from "./SeverityChip";

interface ChainPanelProps {
  report: AlarmReport;
  thresholds: ThresholdStatus;
  /**
   * 只渲染内容，不套 `Panel` 外框。
   *
   * 中央区的 Tab 已经在 `GeoSection` 里，再套一层外框会出现两道标题栏。
   * 这个参数不是"为了方便"，而是因为同一个视图现在有两个宿主。
   */
  bare?: boolean;
}

/**
 * 中央区 —— 链路状态矩阵。
 *
 * 设计规范里中央区放的是**地质剖面**（见 `GeoSection`）。这个视图是同一份
 * 数据的另一种读法：按 捕集 → 运输 → 注入 → MVA 四段分组，
 * 回答"这条链上哪一段出了问题"，而剖面回答"问题发生在哪一层"。
 *
 * 两者不是替代关系，所以它被保留成第二个 Tab，而不是删掉。
 */
export function ChainPanel({ report, thresholds, bare = false }: ChainPanelProps) {
  const unitOf = useMemo(() => {
    const map: Record<string, string> = {};
    for (const s of thresholds.sensors) map[s.key] = s.unit;
    return map;
  }, [thresholds]);

  const byStage = useMemo(() => {
    const map = new Map<Stage, Alarm[]>();
    for (const stage of STAGE_ORDER) {
      map.set(
        stage,
        report.alarms.filter((a) => a.stage === stage),
      );
    }
    return map;
  }, [report.alarms]);

  const body = (
    <>
      {STAGE_ORDER.map((stage, i) => {
        const alarms = byStage.get(stage) ?? [];
        const worst = worstOf(alarms.map((a) => a.severity));
        const cls = SEVERITY_CLASS[worst];
        const blind = alarms.filter((a) => isIndeterminate(a.severity)).length;

        return (
          // flex-1 让四段撑满可用高度。第一版让它们按内容高度排列，
          // 结果四个小方块堆在顶部，下面空出一大片 —— 看起来像还没做完，
          // 而它其实是"做完了但没占满"。链路图本来就该纵向铺开：
          // 它表达的是 CO₂ 从上到下的流动方向。
          <div key={stage} className="relative flex min-h-0 flex-1 flex-col">
            <div
              className={cn(
                "flex flex-1 flex-col justify-center rounded-panel border border-line border-l-2 bg-surface-2 px-2.5 py-2",
                cls.border,
              )}
            >
              <div className="flex items-baseline justify-between gap-2">
                <div className="flex items-baseline gap-2">
                  <span className="text-2xs text-ink-2 tnum">
                    {String(i + 1).padStart(2, "0")}
                  </span>
                  <span className="text-sm text-ink-0">{STAGE_LABELS[stage]}</span>
                  <span className="text-2xs text-ink-2">{STAGE_HINT[stage]}</span>
                </div>
                <div className="flex items-center gap-1.5">
                  {blind > 0 && (
                    <span className="sev-unknown-ring rounded-panel border border-sev-unknown/60 px-1.5 py-0.5 text-2xs text-sev-unknown">
                      {blind} 盲
                    </span>
                  )}
                  <SeverityChip severity={worst} />
                </div>
              </div>

              <div className="mt-1.5 flex flex-wrap gap-1">
                {alarms.length === 0 && (
                  <span className="text-2xs text-ink-2">该段没有测点数据</span>
                )}
                {alarms.map((a) => {
                  const ac = SEVERITY_CLASS[a.severity];
                  const indet = isIndeterminate(a.severity);
                  return (
                    <span
                      key={a.alarm_id}
                      title={a.rationale[0] ?? a.label}
                      className={cn(
                        "inline-flex items-baseline gap-1 rounded-panel border border-line px-1.5 py-0.5 text-2xs",
                        indet && "sev-unknown-ring",
                      )}
                    >
                      <span
                        aria-hidden
                        className={cn(
                          "inline-block size-1.5 translate-y-px rounded-full",
                          indet ? "bg-transparent ring-1 ring-sev-unknown" : ac.dot,
                        )}
                      />
                      <span className="text-ink-1">{a.label}</span>
                      <span className={cn("tnum", ac.text)}>
                        {indet ? "—" : formatValue(a.value)}
                      </span>
                      <span className="text-ink-2">{unitOf[a.sensor] ?? ""}</span>
                    </span>
                  );
                })}
              </div>
            </div>

            {i < STAGE_ORDER.length - 1 && (
              <div
                aria-hidden
                className="mx-auto h-1.5 w-px bg-line-strong"
              />
            )}
          </div>
        );
      })}

      <div className="mt-auto space-y-1 border-t border-line pt-2 text-2xs text-ink-2">
        {/*
          分级政策的出处必须跟着数据一起出现在界面上。
          这些线（notice 带、稳态越限、速率升级）都是项目政策选择，
          没有标准依据 —— 不写出来，读者会以为是行业规定。
        */}
        <p>{report.policy}</p>
        <p>
          阈值来源：<span className="text-ink-1">{report.threshold_origin}</span>
        </p>
      </div>
    </>
  );

  if (bare) {
    return <div className="flex h-full min-h-0 flex-col gap-1.5">{body}</div>;
  }

  return (
    <Panel
      title="全链状态"
      aside={<span>捕集 → 运输 → 注入 → MVA</span>}
      bodyClassName="flex flex-col gap-1.5"
    >
      {body}
    </Panel>
  );
}
