import { useMemo } from "react";
import { useAckStorage, useAcks } from "../lib/ack";
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

  // 本机处置标记：存在浏览器里，刷新不丢，但不写后端（见 `lib/ack.ts`）。
  const acks = useAcks();
  const storageOk = useAckStorage();

  /**
   * 排序：严重度优先，**已标记处置的沉底**。
   *
   * 两段稳定分区，不是 `sort` 加比较函数 —— 后者在"未处理"内部会重排，
   * 而读者记住的是"第三条是环空压力"。分区之后，未处理之间的相对顺序
   * 完全不变。
   *
   * 为什么沉底而不是隐藏：藏掉一条 critical，下一个人就再也看不到它。
   * "有人处理过"和"这件事不存在"必须长得不一样。
   */
  const ordered = useMemo(() => {
    const base = sortAlarms(report.alarms);
    const todo: typeof base = [];
    const done: typeof base = [];
    for (const a of base) (acks[a.alarm_id] ? done : todo).push(a);
    return [...todo, ...done];
  }, [report.alarms, acks]);

  const actionableCount = ordered.filter((a) => SEVERITY_META[a.severity].actionable).length;
  const indetCount = ordered.filter((a) => isIndeterminate(a.severity)).length;
  const ackCount = ordered.filter((a) => acks[a.alarm_id]).length;

  return (
    <Panel
      title="报警与关键指标"
      /*
       * 窄屏必须有 `min-h`。
       *
       * 单列布局下行高是 `auto`，而 `Panel` 的 body 是 `flex-1 min-h-0`
       * —— 一个 `flex: 1 1 0%` 的子元素在"由内容决定高度"的父级里，
       * 假设高度是 **0**，于是整个面板缩成只剩标题条：左栏的告警、
       * 上限校核、盲区全都被裁掉，而控制台里没有任何报错。
       * 这正是"关键信息被挤掉"最典型的样子。
       *
       * 520 而不是 420：420 时只能完整显示"盲区 + 上限 + 1 张告警"，
       * 第二张卡正好被切在分割线上，看起来像渲染坏了（其实是内部滚）。
       * 520 能放下三张，代价是单列页面更长 —— 而单列本来就要滚，
       * "多滚一屏"比"关键告警要滚才能看见"划算。
       *
       * `lg` 起行高被钉死（`minmax(0,1fr)`），面板自己滚，不再需要 min-h。
       * 改这个数要同步改 CI 里 `Narrow-screen floors survive the build`。
       */
      className="min-h-[520px] lg:min-h-0"
      aside={
        <span className="tnum">
          {actionableCount} 需动作 · {indetCount} 不可判定
          {ackCount > 0 && ` · ${ackCount} 已标记`} / 共 {ordered.length}
        </span>
      }
      bodyClassName="space-y-1.5"
    >
      {/*
        标记存在本机浏览器里，不在后端。这句话必须写在面板里，而不是只放在
        每个徽标的 tooltip 上：tooltip 要悬浮才出现，而"已确认"这个说法在
        交接时会被当真。

        存储不可用时（无痕模式、配额满）要照实说"刷新就会丢"——
        浏览器里存不住却显示"已保存"，比一开始就说存不住糟得多。
      */}
      {ackCount > 0 && (
        <p className="mb-2 rounded-panel border border-line px-2 py-1 text-2xs text-ink-2">
          {ackCount} 条已标记处置，已排到列表末尾 ——{" "}
          {storageOk
            ? "标记存在本机浏览器，刷新不丢；但没有写入后端审计，跨班次交接不能只凭这个。"
            : "本机浏览器不允许本地存储，刷新就会丢；也没有写入后端审计。"}
        </p>
      )}

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
