import { useMemo, useState } from "react";
import { buildSection, PLUME_SOURCE } from "../lib/section";
import type { AlarmReport, ThresholdStatus } from "../lib/types";
import { webglAvailable } from "../lib/webgl";
import { cn } from "../lib/utils";
import { ChainPanel } from "./ChainPanel";
import GeoSectionCanvas from "./GeoSectionCanvas";
import type { VoiceState } from "./GeoSectionCanvas";
import { GeoSectionFallback } from "./GeoSectionFallback";
import { Panel } from "./Panel";

interface Props {
  report: AlarmReport;
  thresholds: ThresholdStatus;
  mode: "deck" | "engineer";
  voice?: VoiceState;
}

type Tab = "section" | "chain";

/**
 * 中央区：地下地质剖面。
 *
 * ## 为什么中央不是那个发光球
 *
 * 参考图（以及绝大多数"贾维斯风"作品）中央都是一个旋转的球。球不承载信息 ——
 * 它周围的数字才是信息，球只是让画面不空。而这块屏是给油田用的：
 * 中央区占了 55% 的面积，如果它只负责好看，等于把一半屏幕浪费掉。
 *
 * 换成地质剖面之后，中央区同时是三件事：
 * 1. **空间索引**：`soil_co2_flux` 在地表、`gw_*` 在含水层、`reservoir_p` 在储层，
 *    告警卡上说"浅层地下水 pH 异常"时，读者知道异常发生在哪一层；
 * 2. **量级判断**：羽流横向前缘离监测井还有多远，图上直接看得见；
 * 3. **风险落点**：羽流顶距盖层底界多少米，这是唯一一个失效后没有补救手段的环节。
 *
 * 视觉冲击力由"注入点高亮 + 羽流辉光 + 等压虚线环"承担，不靠球。
 *
 * ## 两个 Tab 而不是两个面板
 *
 * 剖面和链路矩阵是同一份数据的两种读法。并排放会让人以为它们各说各的，
 * 而并排的空间代价在 1440 宽下会把每一栏都挤到不能读。
 */
export function GeoSection({ report, thresholds, mode, voice = "idle" }: Props) {
  // `?tab=chain` 供截图与评审直接定位到第二个视图。
  // 一个只能靠点击才能到达的状态，评审时总会有人截错图。
  const [tab, setTab] = useState<Tab>(() => {
    if (typeof window === "undefined") return "section";
    return new URLSearchParams(window.location.search).get("tab") === "chain"
      ? "chain"
      : "section";
  });

  const unitOf = useMemo(() => {
    const map: Record<string, string> = {};
    for (const s of thresholds.sensors) map[s.key] = s.unit;
    return map;
  }, [thresholds]);

  const data = useMemo(() => buildSection(report, unitOf), [report, unitOf]);

  // 探测只做一次：每帧创建/销毁一个 WebGL 上下文会在低端机上直接触发上下文丢失
  const [hasWebgl] = useState(() => webglAvailable());

  return (
    <Panel
      title="地下剖面"
      aside={
        <div className="flex rounded-panel border border-line p-0.5" role="tablist">
          {(
            [
              ["section", "剖面"],
              ["chain", "全链"],
            ] as const
          ).map(([value, label]) => (
            <button
              key={value}
              type="button"
              role="tab"
              aria-selected={tab === value}
              onClick={() => setTab(value)}
              className={cn(
                "rounded-panel px-1.5 py-0.5 text-2xs",
                tab === value ? "bg-acc/15 text-acc" : "text-ink-2 hover:text-ink-1",
              )}
            >
              {label}
            </button>
          ))}
        </div>
      }
      bodyClassName={cn("p-0", tab === "section" ? "overflow-hidden" : "overflow-auto p-3")}
    >
      {tab === "chain" ? (
        <ChainPanel report={report} thresholds={thresholds} bare />
      ) : (
        <div className="flex h-full min-h-0 flex-col">
          <div className="relative min-h-0 flex-1">
            {hasWebgl ? (
              <GeoSectionCanvas data={data} mode={mode} voice={voice} />
            ) : (
              <div className="h-full w-full p-2">
                <GeoSectionFallback data={data} />
              </div>
            )}
          </div>

          {/*
            图例与出处。两件事必须跟着图一起出现：
            ① 每条线是什么意思 —— 等压线不是 CO₂ 边界，两者差得远；
            ② 羽流几何是示意值 —— 它来自储层模拟，不是从压力读数反推的。
            少了 ②，这张图会被截图放进汇报材料，而里面的数没人能复现。
          */}
          <div className="flex flex-wrap items-center gap-x-3 gap-y-1 border-t border-line px-3 py-1.5 text-2xs text-ink-2">
            <Legend className="bg-acc/40" label="CO₂ 波及范围" />
            <Legend className="border border-dashed border-acc-2/60" label="压力影响范围" />
            <Legend className="border-t border-acc-2/60" label="盖层底界" />
            <span className="ml-auto">
              羽流几何：<span className="text-sev-notice">{PLUME_SOURCE}</span>
            </span>
          </div>
        </div>
      )}
    </Panel>
  );
}

function Legend({ className, label }: { className: string; label: string }) {
  return (
    <span className="inline-flex items-center gap-1">
      <span aria-hidden className={cn("inline-block h-2 w-4 rounded-sm", className)} />
      {label}
    </span>
  );
}
