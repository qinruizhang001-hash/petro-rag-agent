import { useMemo } from "react";
import { SEVERITY_META, sortAlarms, isIndeterminate } from "../lib/severity";
import type { AlarmReport, Attribution, ThresholdStatus } from "../lib/types";
import { AlarmCard } from "./AlarmCard";
import { BlindSpotBanner } from "./BlindSpotBanner";
import { CeilingCard } from "./CeilingCard";
import { MetricCard } from "./MetricCard";
import { Panel } from "./Panel";

interface LeftRailProps {
  report: AlarmReport;
  thresholds: ThresholdStatus;
}

/**
 * 左栏：报警与关键指标。
 *
 * 排序不是"按测点名字"，也不是"按时间"，而是**按严重度**：
 * 值班的人从上往下读，读到第一个不该忽略的东西之前，不该先读十个 normal。
 *
 * 单个列表而不是"需要动作 / 关键指标"两个分区，是为了避免同一条测点
 * 在同一个屏幕上出现两次。重复出现会让读者以为那是两个不同的测点，
 * 而"环空压力"和"环空压力"并排的界面，第一反应是系统出错了。
 */
export function LeftRail({ report, thresholds }: LeftRailProps) {
  const unitOf = useMemo(() => {
    const map: Record<string, string> = {};
    for (const s of thresholds.sensors) map[s.key] = s.unit;
    return map;
  }, [thresholds]);

  /**
   * 测点目录里的 warn 阈值。
   *
   * 这是"这个测点的阈值是多少"的唯一正确来源。`alarm.threshold_value`
   * 是"命中了哪一级的线"，`normal` 时为 `null` —— 拿它当阈值，
   * 所有正常测点都会显示成"无阈值"，与真正的盲区无法区分。
   */
  const warnOf = useMemo(() => {
    const map: Record<string, number | null> = {};
    for (const s of thresholds.sensors) map[s.key] = s.threshold?.warn ?? null;
    return map;
  }, [thresholds]);

  const labelOf = useMemo(() => {
    const map: Record<string, string> = {};
    for (const s of thresholds.sensors) map[s.key] = s.label;
    return map;
  }, [thresholds]);

  /**
   * 被排除的归因按测点索引。
   *
   * 后端把 `ruled_out` 从单条告警里摘掉了（有测试锁着这个契约），
   * 所以这里从报告级列表重建索引，**只按 basis 里真实出现过的测点**关联。
   * 按 stage 关联会串味：管道结垢的排除理由会跑到气质纯度的卡片里。
   */
  const excludedBySensor = useMemo(() => {
    const map: Record<string, Attribution[]> = {};
    for (const a of report.attribution) {
      if (!a.ruled_out) continue;
      for (const b of a.basis) {
        (map[b.sensor] ??= []).push(a);
      }
    }
    return map;
  }, [report.attribution]);

  const ordered = useMemo(() => sortAlarms(report.alarms), [report.alarms]);
  const actionableCount = ordered.filter((a) => SEVERITY_META[a.severity].actionable).length;
  const indetCount = ordered.filter((a) => isIndeterminate(a.severity)).length;

  return (
    <Panel
      title="报警与关键指标"
      aside={
        <span className="tnum">
          {actionableCount} 需动作 · {indetCount} 不可判定 / 共 {ordered.length}
        </span>
      }
      bodyClassName="space-y-1.5"
    >
      <BlindSpotBanner
        unconfigured={report.unconfigured}
        noData={report.no_data}
        labels={labelOf}
        className="mb-2"
      />

      {/*
        上限校核排在告警列表**之前**：它是唯一一项法规直接给定的判据，
        而告警是项目自己配的线。读者先看到"注入本身合不合规"，
        再看"哪个测点偏离了我们的线" —— 顺序反过来会让合规结论被淹在
        十几条运行告警里。
      */}
      <CeilingCard check={report.ceiling} className="mb-2" />

      {ordered.map((alarm) => {
        // 需要动作的用可展开的告警卡；其余用紧凑指标卡。
        // 判断依据是"这一条是否要求人做点什么"，而不是"它是否 non-normal" ——
        // notice 只要求下个班次复核，给它一张可展开的大卡片会淹没真正的告警。
        const actionable = SEVERITY_META[alarm.severity].actionable;
        return actionable ? (
          <AlarmCard
            key={alarm.alarm_id}
            alarm={alarm}
            unit={unitOf[alarm.sensor] ?? ""}
            warnThreshold={warnOf[alarm.sensor] ?? null}
            excluded={excludedBySensor[alarm.sensor] ?? []}
          />
        ) : (
          <MetricCard
            key={alarm.alarm_id}
            alarm={alarm}
            unit={unitOf[alarm.sensor] ?? ""}
            warnThreshold={warnOf[alarm.sensor] ?? null}
          />
        );
      })}
    </Panel>
  );
}
