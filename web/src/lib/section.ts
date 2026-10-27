/**
 * 地质剖面的几何真值。
 *
 * three.js 场景与 SVG 降级版都从这里取数。两个渲染器各写一份坐标，
 * 迟早在某次改动之后画出两个不一样的剖面 —— 而"看起来都挺对"是这类
 * 偏差最难被发现的地方。所以坐标、层序、测点深度只允许存在这一份。
 *
 * ## 坐标约定
 *
 * 真实世界单位是米，场景单位是「百米」（`SCALE = 100`）。
 * - `x` 为横向偏移，注入井在 `x = 0`，正方向是远离井的一侧；
 * - `depth` 为向下为正的深度（米），场景里 `y = -depth / SCALE`。
 *
 * 用"向下为正"而不是"向上为正"，是因为现场所有数据都是这么说的：
 * 测点深度 880 m、盖层底界 420 m。渲染层去翻这个号，比在数据层翻要安全 ——
 * 数据层一旦有一个字段的符号约定跟别的不一样，就是一次静默的镜像。
 */

import { worstOf } from "./severity";
import type { Alarm, AlarmReport, Severity } from "./types";

/** 米 → 场景单位。100 m 一个单位，剖面总高 16 个单位。 */
export const SCALE = 100;

/** 剖面最底部深度（米）。深度尺与相机取景都以此为准。 */
export const DEPTH_MAX = 1300;

/** 剖面横向半宽（米）。取景时用它算横向是否装得下。 */
export const HALF_WIDTH_M = 950;

export interface Stratum {
  key: string;
  label: string;
  /** 层顶深度（米） */
  top: number;
  /** 层底深度（米） */
  bottom: number;
  /** 取 `index.css` 里的同名 CSS 变量，保证 3D 与 DOM 用的是同一个色值 */
  colorVar: string;
  /** `hatch` 斜线（泥岩/隔层）、`dots` 点状（孔隙介质）、`none` 均质 */
  texture: "none" | "hatch" | "dots";
  /**
   * 是否是封闭层。
   *
   * 单独标出来而不是让渲染器去猜：它决定"羽流能不能穿过"这件事，
   * 也决定这一层要不要画得更亮 —— 盖层是整套 CCUS 里唯一一个
   * "失效了就没有第二次机会"的部件。
   */
  seal?: boolean;
}

/**
 * 层序：从上到下。
 *
 * 层厚不是随手分的：浅层含水层单独成层，是因为 `gw_ph` / `gw_conductivity`
 * 两个测点在那里取样 —— 如果含水层和覆盖层画成一层，那两个测点就没有
 * 可以指向的层位，"归因到层位"这句话就落空了。
 */
export const STRATA: Stratum[] = [
  { key: "overburden", label: "第四系覆盖层", top: 0, bottom: 180, colorVar: "--strata-1", texture: "none" },
  { key: "aquifer", label: "浅层含水层", top: 180, bottom: 260, colorVar: "--strata-2", texture: "dots" },
  { key: "caprock", label: "盖层（泥岩封闭）", top: 260, bottom: 420, colorVar: "--strata-3", texture: "hatch", seal: true },
  // 储层厚度取 400 m：再厚下去，羽流会变成一个竖长条，
  // 而真实的 CO₂ 在咸水层里是**横向铺开**的 —— 竖长条会让人误判波及方向
  { key: "reservoir", label: "咸水层储层（注入层）", top: 420, bottom: 820, colorVar: "--strata-4", texture: "dots" },
  { key: "basement", label: "基底", top: 820, bottom: DEPTH_MAX, colorVar: "--strata-5", texture: "hatch" },
];

/** 深度尺刻度。放在这里而不是组件里，否则 SVG 降级版会画出另一套刻度。 */
export const RULER_DEPTHS = [0, 300, 600, 900, 1200];

/** 盖层底界深度（米）。羽流顶部离它多远，是这张图上最该被读到的数。 */
export const SEAL_BASE = STRATA.find((s) => s.seal)!.bottom;

/** 注入层顶界（米）。 */
export const RESERVOIR_TOP = STRATA.find((s) => s.key === "reservoir")!.top;

/** 注入射孔段深度（米）。在储层中下部，给 CO₂ 留出向上运移的空间。 */
export const INJECTION_DEPTH = 760;

export type MonitorKind = "reservoir" | "wellbore" | "aquifer" | "nearSurface" | "surface";

export interface MonitorAnchor {
  /** 稳定 id，React 用来给标签定位 */
  id: string;
  /** 这个位置在剖面上的角色 */
  kind: MonitorKind;
  label: string;
  /** 深度（米） */
  depth: number;
  /** 横向偏移（米）。注入井在 0 */
  x: number;
  /**
   * 该位置上的测点。一个位置可能挂多个测点（浅层含水层同时测 pH 与电导率），
   * 此时取其中最坏的那个等级 —— 一个位置只能有一种颜色，
   * 否则读者要在两个色块之间做判断，而图上的位置本身不携带这个信息。
   */
  sensors: string[];
  /** 已解析出的等级；`null` = 该位置没有任何测点数据 */
  severity: Severity | null;
  /** 该位置最坏那个测点的读数，用于标签第二行 */
  value: number | null;
  unit: string;
  /** 读数不可信的原因；非空时标签要显式写出来 */
  caveat: string;
  /**
   * 标签相对测点的纵向偏移（像素）。
   *
   * 地表附近三个测点（井口环空压力、土壤通量、地表形变）深度都在 0–26 m，
   * 标签不挪开就会叠成一团。这个偏移是**标注排布**，不是几何，
   * 所以留在界面层，不进 `depth`。
   */
  dy?: number;
}

/**
 * 测点在剖面上的空间位置。
 *
 * 这五个位置不是装饰：MVA 的意义就是"在 CO₂ 可能到达的路径上布点"。
 * 图上把 `soil_co2_flux` 放在地表、`gw_*` 放在含水层，是为了让
 * "浅部异常"这句话有一个能指的地方 —— 否则告警卡上说"地下水 pH 异常"，
 * 读者不知道异常发生在哪一层。
 */
const MONITOR_LAYOUT: Omit<MonitorAnchor, "severity" | "value" | "unit" | "caveat" | "label">[] = [
  // 储层观测井放在羽流**外侧**：它测的是压力传播，不是 CO₂ 到达。
  // 放进羽流里会让"储层压力"这个点看起来像"CO₂ 已经到这儿了"，是两件事。
  // 所有横向位置都必须落在 ±HALF_WIDTH_M 之内 —— 超出剖面的测点会被画到
  // 画布外面，标签投影成一个负数，看起来像"这个测点消失了"。
  { id: "reservoir", kind: "reservoir", depth: 620, x: 470, sensors: ["reservoir_p"] },
  // 环空压力是**井口**测量，所以画在井口而不是井筒中部。
  // 画在中部会让人以为那里有个井下传感器，而现场没有。
  { id: "wellbore", kind: "wellbore", depth: 26, x: 0, sensors: ["annulus_p"], dy: -14 },
  { id: "aquifer", kind: "aquifer", depth: 220, x: 560, sensors: ["gw_ph", "gw_conductivity"] },
  { id: "nearSurface", kind: "nearSurface", depth: 12, x: 730, sensors: ["soil_co2_flux"], dy: 16 },
  { id: "surface", kind: "surface", depth: 0, x: 880, sensors: ["surface_heave"], dy: -16 },
];

const KIND_LABEL: Record<MonitorKind, string> = {
  reservoir: "储层压力",
  wellbore: "井筒完整性",
  aquifer: "浅层地下水",
  nearSurface: "近地表土壤通量",
  surface: "地表形变",
};

export interface PlumeGeometry {
  /** 羽流顶部深度（米） */
  apex: number;
  /** 羽流底部深度（米） */
  base: number;
  /** 横向中心（米） */
  cx: number;
  /** 顶部横向半径（米） */
  radius: number;
  /** 羽流顶部与盖层底界的间距（米）。0 = 已贴住盖层 */
  sealClearance: number;
}

/**
 * 羽流几何的**示意值**。
 *
 * ## 为什么是示意值，而不是从读数算出来的
 *
 * 羽流的形状来自储层模拟（历史拟合 + 流体运移），不是从压力读数能反推的。
 * 用 `reservoir_p` 去"推算"羽流半径，得到的是一个看起来很像真数的编造值 ——
 * 而它会被截图放进汇报材料。
 *
 * 所以这里把 `source` 一起带出去，界面上照原样显示「示意（未接储层模拟）」，
 * 与阈值来源那句 `【替换】` 的处理方式一致：**不知道就说不知道**，
 * 不要把一个占位值渲染成测量结果。
 *
 * 接上真实模型时，把 `PLUME_PLACEHOLDER` 换成 `ops` 侧下发的对象即可，
 * 渲染器一行都不用改 —— 这也是为什么几何参数要单独成一个类型。
 */
export const PLUME_PLACEHOLDER: PlumeGeometry = {
  apex: SEAL_BASE + 6,
  base: INJECTION_DEPTH,
  cx: 0,
  // 380 m 是让羽流在 1900 m 宽的剖面上占三分之一左右：
  // 太小看不出波及方向，太大就贴到浅层监测井上了（那是一个不该凭空造出来的告警）
  radius: 380,
  sealClearance: 6,
};

export const PLUME_SOURCE = "示意（未接储层模拟）";

/**
 * 羽流在某个相对高度处的横向半径（米）。`t`: 0 = 射孔段，1 = 羽流顶。
 *
 * ## 为什么必须是"喇叭"而不是"高脚杯"
 *
 * CO₂ 比地层水轻，注入后受浮力**向上**运移，到盖层底界停住，然后沿盖层
 * 底面横向铺开 —— 形状是一根供给柱顶着一个贴着盖层的饼。
 *
 * 前两版都画成了高脚杯：张开区间被压在顶部 20% 的高度里，于是一条几乎
 * 水平的肩线 + 一根细针。高脚杯在物理上意味着"从某一深度起突然出现水平
 * 通道"，而图上并没有画这条通道 —— 形状在暗示一个不存在的构造。
 *
 * 现在用 `pow(smoothstep(0.15, 1, t), 1.7)`：从储层下部就开始缓慢张开，
 * 越往上越快，顶部取满半径。曲线连续、没有肩线，顶部仍是平的（贴盖层）。
 *
 * ## 为什么放在这里而不是各自的渲染器里
 *
 * 3D 与 SVG 降级版各写一份，改了一处忘了另一处，就会出现"两个剖面不一样"
 * —— 而且两边看起来都挺对，没人会去比对。这份几何只允许有一个定义。
 */
export function plumeRadiusM(p: PlumeGeometry, t: number): number {
  // 供给柱 150 m：90 m 配 380 m 的顶是 1:4.2，画出来就是一根针顶着一个碗。
  // 真实羽流的底部也有可观宽度 —— CO₂ 一进储层就开始指进，不是先攒成一根柱。
  const feeder = 150;
  // 张开区间铺满整个高度（pow 1.4 让上段张得更快），避免出现一条水平肩线
  const s = Math.min(1, Math.max(0, t));
  const flare = Math.pow(s * s * (3 - 2 * s), 1.4);
  // 射孔段的局部扩散。55 会在底部鼓出一个明显的球，整条羽流被读成
  // 高脚杯的杯脚；真实射孔段的局部扩大远没有这么夸张。
  const bulb = 24 * Math.exp(-Math.pow(t / 0.1, 2));
  return feeder + (p.radius - feeder) * flare + bulb;
}


export interface SectionData {
  strata: Stratum[];
  monitors: MonitorAnchor[];
  plume: PlumeGeometry;
  /** 羽流顶部到盖层底界的距离（米） */
  sealClearance: number;
  /**
   * 浅部（含水层及以上）出现的最坏等级。
   *
   * 非空且 ≥ warn 时，图上画一条**虚线通道 + 问号**，而不是一条红色实线箭头：
   * "浅部有异常"是实测的，"异常是从储层漏上来的"是推测的。
   * 把推测画成实测，正是这套界面最该避免的事（与 `ruled_out` 的处理同理）。
   */
  shallowAnomaly: Severity | null;
  /** 羽流横向前缘距最近监测点的距离（米）；`null` = 没有可比对的点 */
  monitorClearance: number | null;
}

function pick(alarms: Alarm[], sensor: string): Alarm | undefined {
  return alarms.find((a) => a.sensor === sensor);
}

/**
 * 把一份告警报告投影到剖面上。
 *
 * 只做投影，不做推断：测点等级直接取报告里的，羽流几何取示意值，
 * 两者不在这一层混算。混算过一次之后，"图上那个红点是实测还是推的"
 * 就再也说不清了。
 */
export function buildSection(report: AlarmReport, unitOf: Record<string, string>): SectionData {
  const monitors: MonitorAnchor[] = MONITOR_LAYOUT.map((layout) => {
    const found = layout.sensors
      .map((s) => pick(report.alarms, s))
      .filter((a): a is Alarm => Boolean(a));

    const first = found[0];
    if (first === undefined) {
      return {
        ...layout,
        label: KIND_LABEL[layout.kind],
        severity: null,
        value: null,
        unit: unitOf[layout.sensors[0] ?? ""] ?? "",
        caveat: "无测点数据",
      };
    }

    // 一个位置一种颜色：取最坏的那个测点
    const worst = worstOf(found.map((a) => a.severity));
    // 读数与等级必须来自同一个测点，否则会出现"标着 warn 却显示正常测点的数"
    const driver = found.find((a) => a.severity === worst) ?? first;

    const caveats: string[] = [];
    if (found.some((a) => a.severity === "indeterminate")) caveats.push("有测点不可判定");
    if (driver.quality !== "good") caveats.push(`仪表${driver.quality === "bad" ? "失效" : "可疑"}`);

    return {
      ...layout,
      label: KIND_LABEL[layout.kind],
      severity: worst,
      value: driver.value,
      unit: unitOf[driver.sensor] ?? "",
      caveat: caveats.join(" · "),
    };
  });

  const shallow = monitors.filter(
    (m) => m.kind === "aquifer" || m.kind === "nearSurface" || m.kind === "surface",
  );
  const shallowSeverity = shallow.some((m) => m.severity !== null)
    ? worstOf(shallow.map((m) => m.severity ?? "indeterminate"))
    : null;

  const plume = PLUME_PLACEHOLDER;

  // 羽流前缘与浅层监测井的横向距离。
  //
  // 只算 `aquifer` 及以上，**不含储层内的观测井**：储层里的观测井本来就在
  // 羽流波及范围内，拿它算"余量"会得到一个负数，读起来像"已经漏了"。
  // 这个数要回答的是"CO₂ 还有多远到达封闭失效的判定点"。
  const sentinels = monitors.filter((m) => m.kind === "aquifer" || m.kind === "nearSurface");
  const distances = sentinels.map((m) => Math.abs(m.x - plume.cx) - plume.radius);
  const monitorClearance = distances.length ? Math.min(...distances) : null;

  return {
    strata: STRATA,
    monitors,
    plume,
    sealClearance: plume.apex - SEAL_BASE,
    shallowAnomaly: shallowSeverity,
    monitorClearance,
  };
}
