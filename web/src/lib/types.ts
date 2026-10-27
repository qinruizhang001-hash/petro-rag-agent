/**
 * `/api/ops/*` 的响应类型。
 *
 * 这些类型不是照着文档手写的，而是照着 `web/src/fixtures/*.json` 反推的，
 * 而 fixture 由 `scripts/capture_ops_fixtures.py` 从真实接口抓取。
 * 后端字段改名时，`python scripts/capture_ops_fixtures.py --check` 会先报出来，
 * 然后这里的类型编译不过 —— 两道闸，而不是等运行时某个数字变成 `undefined`。
 *
 * 刻意**不**把 severity 写成 string：六态里少一态，界面上就会有一个状态
 * 悄悄渲染成默认色，而默认色是灰的，读起来像"没问题"。
 */

/** 五个有序等级 + 一个不属于这个序列的第六态。 */
export const SEVERITIES = ["normal", "notice", "warn", "alert", "critical"] as const;

export type GradedSeverity = (typeof SEVERITIES)[number];

/**
 * 第六态。它**不在** `SEVERITIES` 里，因为"不可判定"不是"严重度为 0"，
 * 而是否定"能判定"这件事本身。把它塞进序列会让 `top_severity` 的比较
 * 变得没有意义。
 */
export const INDETERMINATE = "indeterminate" as const;

export type Severity = GradedSeverity | typeof INDETERMINATE;

export type Stage = "capture" | "transport" | "injection" | "mva";

export type Direction = "high" | "low";

export type Confidence = "strong" | "partial" | "weak";

export interface BasisEntry {
  sensor: string;
  /** 规则期望的状态，`deviated` / `ok` */
  expected: string;
  /** 实测状态；`null` 表示该测点没有读数 */
  observed: string | null;
}

export interface Attribution {
  cause: string;
  stage: Stage;
  confidence: Confidence;
  /** 给人读的措辞，已经由后端翻译成中文 */
  evidence: string[];
  /**
   * 被排除的候选仍然保留，并给出理由。
   *
   * 读者需要看到"我们考虑过这个原因、因为某条读数排除了它"，
   * 否则会怀疑是不是漏了。所以界面上用删除线显示，**不隐藏**。
   */
  ruled_out: boolean;
  ruled_out_reason: string;
  /** 结构化的判据，用于把归因关联到具体测点 */
  basis: BasisEntry[];
}

export interface ActionItem {
  key: string;
  action: string;
  /** 责任岗位，不是人名 */
  owner: string;
}

export interface Alarm {
  alarm_id: string;
  sensor: string;
  label: string;
  stage: Stage;
  severity: Severity;
  /** 后端已经给了中文标签，但界面用它时仍以 `severity` 为准做样式 */
  severity_label: string;
  direction: Direction;
  /** `null` 表示没有可信读数 —— 此时不能显示 0 */
  value: number | null;
  ts: string;
  /**
   * 逐条判据。**不折叠成一句"超阈值"**：
   * 运维要问的是"为什么是这一级"，三条规则命中就写三行。
   */
  rationale: string[];
  /** 只包含未被排除的归因（后端的契约，见 test_ruled_out_attributions_are_not_attached_to_alarms） */
  attribution: Attribution[];
  actions: ActionItem[];
  threshold_level: string;
  threshold_value: number | null;
  provenance: string;
  quality: "good" | "suspect" | "bad";
}

export interface AlarmCounts {
  normal: number;
  notice: number;
  warn: number;
  alert: number;
  critical: number;
  indeterminate: number;
}

/**
 * 注入压力上限校核（`ops/ceiling.py`）。
 *
 * 四种状态而不是三种：`indeterminate` 与 `ok` 的区别就是
 * "没查成"与"查过了没问题"。把两者合并，界面会把一个没接井底压力计的
 * 场地画成合规 —— 这是这套界面最不能犯的错。
 */
export type CeilingState = "ok" | "near" | "exceeded" | "indeterminate";

export interface CeilingCheck {
  state: CeilingState;
  /** 后端给的中文标签，与 `state` 同时下发 */
  label: string;
  /** 是否应当阻断报告提交。只有确证超限才为真 */
  blocks: boolean;
  /** 窗口内**峰值**井底压力，不是末值 */
  bhp: number | null;
  bhp_ts: string | null;
  /** 峰值时刻**生效**的那次 LOT/XLOT（时间戳不晚于峰值），不是"最新一次" */
  frac_p: number | null;
  frac_ts: string | null;
  /** 井底峰值 / 破裂压力；`null` = 缺数据算不出来 */
  ratio: number | null;
  /** 距上限的余量，负值表示已超出 */
  headroom_bar: number | null;
  /** 井底峰值与破裂压力读数的时间差（天） */
  skew_days: number | null;
  /** 法规上限比例，例如 0.9 */
  ceiling_ratio: number;
  /** 法规出处（含"换辖区时须同时替换"这类维护提示，不适合进正文） */
  source: string;
  /** 可引用的出处，正文用这一句 */
  citation: string;
  /** 逐条依据，给人读 */
  basis: string[];
  next_step: string;
}

export interface AlarmReport {
  /** 一句话摘要，含最高等级、盲区数量、各态计数 */
  summary: string;
  top_severity: Severity;
  counts: AlarmCounts;
  /** 分级政策的出处说明 —— 这些线没有标准依据，必须随数据一起下发 */
  policy: string;
  threshold_origin: string;
  /** 已声明但没配阈值的测点。**不是空数组时看板上有盲区** */
  unconfigured: string[];
  /** 完全没收到读数的已声明测点 */
  no_data: string[];
  alarms: Alarm[];
  /**
   * 报告级归因列表，含被排除的候选。
   *
   * 单条告警的 `attribution` 里没有被排除项，所以界面上"已排除"那一栏
   * 要从这里按 `basis` 里的测点关联过去 —— 不能按 stage 关联，
   * 那会让"管道结垢已排除"出现在气质纯度的卡片里。
   */
  attribution: Attribution[];
  /** `[startISO, endISO]` */
  period: [string, string];
  /**
   * 注入压力上限校核。
   *
   * 可选：`/api/ops/run` 的响应里它是独立字段，`/api/ops/alarms` 里也是；
   * 但老版本 fixture 没有这一项，所以类型上允许缺省 —— 缺省时界面显示
   * "本项未执行"，**不是**"合规"。这一点由组件负责，不由调用方负责。
   */
  ceiling?: CeilingCheck | null;
}

export interface SensorThreshold {
  warn: number;
  alert: number;
  critical: number;
  source: string;
  note?: string;
}

export interface SensorCatalogEntry {
  key: string;
  label: string;
  unit: string;
  stage: Stage;
  stage_label: string;
  direction: Direction;
  /** 该测点的判定依据说明 */
  basis: string;
  /** `null` = 已声明但没配阈值，界面上必须与"正常"区分 */
  threshold: SensorThreshold | null;
}

export interface ThresholdStatus {
  origin: string;
  configured: string[];
  missing: string[];
  sensors: SensorCatalogEntry[];
}

export interface SensorStats {
  sensor: string;
  label: string;
  unit: string;
  n: number;
  start: string;
  end: string;
  min: number;
  max: number;
  mean: number;
  /** nearest-rank 分位：取实测点，不插值 —— 所以它永远不会是一个没测到的数 */
  p95: number;
  p5: number;
  last: number;
  slope_per_hour: number;
  /** 窗口覆盖率，< 1 说明有缺口 */
  coverage: number;
  gaps: string[];
  over_threshold: number;
  threshold_source: string;
}

export interface ConditionAnswer {
  question: string;
  window: { hours: number; label: string; explicit: boolean; note: string };
  now: string;
  sensors: string[];
  stats: SensorStats[];
  /** 事实块。模型措辞里出现的数字必须能在这里找到，否则整篇回退模板 */
  facts: string[];
  text: string;
  /** 问题里认不出来的部分。非空时**不能**假装回答了 —— 见 ops/condition.py */
  unresolved: string[];
  /**
   * 模型措辞里出现但事实块中找不到的数字。
   * 非空 = 守卫触发过，此时 `text` 已整篇回退到模板。
   * 界面要把它显示出来，那是一次真实的守卫触发，不是噪音。
   */
  unsourced_numbers: string[];
  alarm_summary: string;
}
