import { cn } from "../lib/utils";
import type { VoiceState } from "./GeoSectionCanvas";

const STATE_LABEL: Record<VoiceState, string> = {
  idle: "待机",
  listening: "聆听中",
  thinking: "思考中",
};

/**
 * 语音状态环。
 *
 * ## 它不控制任何东西
 *
 * 三个状态只回答一个问题："系统现在在听吗"。环上没有任何可点的元素，
 * 也不承载"确认/执行"——高危操作必须走点击 + 二次确认 + 操作人签名
 * （见 `RightRail` 与设计规范 §5.5）。
 *
 * 把状态环做成可交互的按钮，是这类界面最常见的一个错误：它看起来像
 * "语音入口"，于是有人对着它说话，然后以为说完了就执行了。
 *
 * ## 动效只在指挥舱
 *
 * 工程师模式由 `html[data-mode="engineer"] * { animation: none }` 全局关闭。
 * 现场强光下，一个一直在转的环只会被当成"系统在忙"，
 * 而"系统在忙"和"系统卡住了"必须能分开 —— 所以宁可让它静止。
 */
export function VoiceRing({
  state,
  mode,
}: {
  state: VoiceState;
  mode: "deck" | "engineer";
}) {
  const active = state !== "idle";
  const deck = mode === "deck";

  return (
    <div className="flex flex-col items-center gap-1">
      <div className="relative grid size-16 place-items-center">
        <svg viewBox="0 0 64 64" className="absolute inset-0 size-full" aria-hidden>
          {/* 外圈：待机时是最淡的一圈，说明"它在，但没在听" */}
          <circle
            cx="32"
            cy="32"
            r="30"
            fill="none"
            stroke="currentColor"
            strokeWidth="1"
            className={cn(
              "text-acc/25",
              active && deck && state === "listening" && "animate-pulse text-acc/60",
            )}
          />
          {/* 状态弧：聆听是实线短弧，思考是虚线整圈 */}
          <circle
            cx="32"
            cy="32"
            r="24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeDasharray={
              state === "idle" ? "4 150" : state === "listening" ? "60 92" : "3 6"
            }
            className={cn(
              "text-acc/40",
              active && "text-acc",
              deck && state === "thinking" && "origin-center animate-spin",
            )}
            style={{ transformBox: "fill-box", transformOrigin: "center" }}
          />
          {/* 麦克风：不画波形 —— 波形会让人以为它在录音 */}
          <path
            d="M32 24a4 4 0 0 1 4 4v6a4 4 0 0 1-8 0v-6a4 4 0 0 1 4-4Z"
            className={cn("fill-current", active ? "text-acc" : "text-ink-2")}
          />
          <path
            d="M25 33a7 7 0 0 0 14 0M32 40v4"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.6"
            strokeLinecap="round"
            className={cn(active ? "text-acc" : "text-ink-2")}
          />
        </svg>
      </div>
      <span
        className={cn(
          "rounded-panel px-1.5 py-0.5 text-2xs",
          active ? "bg-acc/10 text-acc" : "text-ink-2",
        )}
      >
        {STATE_LABEL[state]}
      </span>
    </div>
  );
}
