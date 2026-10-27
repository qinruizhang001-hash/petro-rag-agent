import type { Stage } from "./types";

/**
 * 阶段标签。与后端 `ops/contracts.py` 的 `STAGE_LABELS` 逐字一致 ——
 * 界面自己另译一遍，就会出现"注入"和"注入环节"并存，
 * 而这种不一致读者只会当成两个不同的东西。
 */
export const STAGE_LABELS: Record<Stage, string> = {
  capture: "捕集",
  transport: "运输",
  injection: "注入",
  mva: "监测核查与核算（MVA）",
};

export const STAGE_SHORT: Record<Stage, string> = {
  capture: "捕集",
  transport: "运输",
  injection: "注入",
  mva: "MVA",
};

/** 链路顺序。中央区按这个顺序画，因为它是 CO₂ 实际流过的方向。 */
export const STAGE_ORDER: Stage[] = ["capture", "transport", "injection", "mva"];

/** 每个阶段承担什么，用于空态与提示文案。 */
export const STAGE_HINT: Record<Stage, string> = {
  capture: "燃烧后捕集 · 再生 · 溶剂",
  transport: "管输 · 压降 · 气质",
  injection: "井口 · 环空 · 注入能力",
  mva: "储层 · 浅层地下水 · 近地表",
};
