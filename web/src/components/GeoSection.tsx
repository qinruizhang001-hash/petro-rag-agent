import { useMemo, useState } from "react";
import { buildSection, PLUME_SOURCE, STRATA_SOURCE } from "../lib/section";
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
  /**
   * 外层尺寸类。
   *
   * 窄屏时剖面所在的行是 `auto` 高度，画布（`h-full`）会算成 0 px ——
   * 图看起来"消失了"，而控制台里没有任何报错。所以尺寸必须由调用方
   * 用 `min-h` 给死，而 Panel 需要 `h-full` 才能填满那一格。
   */
  className?: string;
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
export function GeoSection({ report, thresholds, mode, voice = "idle", className }: Props) {
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

  /**
   * 实际在跑的渲染路径。
   *
   * 不能只看 `webglAvailable()`：那个函数只问"浏览器有没有 WebGL"，
   * 而真实故障是"有 WebGL，但这台机器的驱动建不出上下文"（工控机、
   * 远程桌面会话、被策略禁用的硬件加速）。所以这里由画布**回报**
   * 失败，然后切到 SVG 版 —— 否则那块屏会变成一句"3D 视图不可用"，
   * 而剖面上的 5 个测点与盖层间距是数据，不是特效。
   */
  const [renderMode, setRenderMode] = useState<"3d" | "svg">(
    () => (hasWebgl ? "3d" : "svg"),
  );
  const [failReason, setFailReason] = useState<string | null>(null);

  return (
    <Panel
      title="地下剖面"
      className={cn("h-full", className)}
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
            {renderMode === "3d" ? (
              <GeoSectionCanvas
                data={data}
                mode={mode}
                voice={voice}
                onFailed={(reason) => {
                  setFailReason(reason);
                  setRenderMode("svg");
                }}
              />
            ) : (
              <div className="h-full w-full p-2">
                <GeoSectionFallback data={data} />
              </div>
            )}
          </div>

          {/*
            图例与限制。三件事必须跟着图一起出现：
            ① 每条线是什么意思 —— 等压线不是 CO₂ 边界，两者差得远；
            ② 羽流几何与地层分层都是示意值 —— 它们不来自本场地的模拟或解释；
            ③ 现在看的是 3D 还是 SVG 降级版 —— 两者数据相同，表现力不同，
               但读者有权利知道自己在看哪一张。
            少了 ②③，这张图会被截图放进汇报材料，而里面的数没人能复现，
            也没人知道它是不是降级版。

            `shrink-0` 不能省：画布是 `flex-1`，一旦画布里的内容把这一格
            撑紧，这个区块会被 flex 压缩到只剩半行 —— 而它是这张图上
            唯一说明"哪些数字不能拿去用"的地方。宁可压画布，不压它。

            这三句话**没有**后端对应字段：`AlarmReport` 里有 `policy` 与
            `threshold_origin`（分级政策的出处），但没有剖面示意性的说明。
            也就是说这里是唯一来源 —— 将来接入储层模拟时，必须同步改
            `lib/section.ts` 里的 `PLUME_SOURCE` / `STRATA_SOURCE`，
            否则界面会继续声称"未接储层模拟"。
          */}
          <div className="shrink-0 border-t border-line px-3 py-1.5 text-2xs text-ink-2">
            <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
              <Legend className="bg-acc/40" label="CO₂ 波及范围" />
              <Legend className="border border-dashed border-acc-2/60" label="压力影响范围" />
              <Legend className="border-t border-acc-2/60" label="盖层底界" />

              <span className="ml-auto flex flex-wrap items-center gap-1">
                <Tag
                  label={renderMode === "3d" ? "视图 3D" : "视图 SVG 降级"}
                  tone={renderMode === "3d" ? "acc" : "notice"}
                  title={
                    renderMode === "3d"
                      ? "WebGL 正常，当前是 three.js 渲染"
                      : `WebGL 不可用，已切到 SVG 版（数据与 3D 版同源）${
                          failReason ? `：${failReason}` : ""
                        }`
                  }
                />
                <Tag label="羽流 示意" title={PLUME_SOURCE} />
                <Tag label="分层 示意" title={STRATA_SOURCE} />
              </span>
            </div>

            <p className="mt-1 leading-relaxed">
              这是一张<span className="text-ink-1">示意剖面</span>：羽流几何
              {PLUME_SOURCE}，地层分层{STRATA_SOURCE}。它能回答"异常发生在哪一层、
              离盖层还有多远"，<span className="text-ink-1">不能</span>
              作为井位设计、封存容量或压力预测的依据。
            </p>
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

/**
 * 小标签。`title` 必填 —— 每一个标签的完整说明都比标签本身长，
 * 而这块地方放不下正文（放下就会把图压扁）。
 */
function Tag({
  label,
  tone = "line",
  title,
}: {
  label: string;
  tone?: "line" | "acc" | "notice";
  title: string;
}) {
  return (
    <span
      title={title}
      className={cn(
        "rounded-panel border px-1",
        tone === "acc"
          ? "border-acc/40 text-acc"
          : tone === "notice"
            ? "border-sev-notice/60 text-sev-notice"
            : "border-line text-ink-2",
      )}
    >
      {label}
    </span>
  );
}
