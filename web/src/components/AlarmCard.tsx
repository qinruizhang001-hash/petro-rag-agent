import { useState } from "react";
import {
  SEVERITY_CLASS,
  describeMargin,
  formatValue,
  isIndeterminate,
} from "../lib/severity";
import { STAGE_SHORT } from "../lib/stages";
import type { Alarm, Attribution } from "../lib/types";
import { cn } from "../lib/utils";
import { SeverityChip } from "./SeverityChip";

interface AlarmCardProps {
  alarm: Alarm;
  unit: string;
  /**
   * 测点目录里的 warn 阈值。**不是** `alarm.threshold_value` ——
   * 后者是"命中了哪一级的线"，`normal` 时它是 `null`。
   */
  warnThreshold: number | null;
  /**
   * 被排除的归因。
   *
   * 后端的契约是"`ruled_out` 不挂到单条告警上"（有测试锁着），
   * 所以这里从报告级列表按 `basis` 里是否真的用到了这个测点关联过来。
   * **不能按 stage 关联**：那会让"管道结垢已排除"出现在气质纯度的卡片里，
   * 读者会以为系统把两个不相干的测点混在一起了。
   */
  excluded: Attribution[];
}

/**
 * 告警卡。
 *
 * 三个刻意的设计，都来自 `ops/contracts.py` 的字段语义：
 *
 * 1. **`rationale` 逐条列出**，不折叠成一句"超阈值"。运维要问的是
 *    "为什么是这一级"，三条规则命中就写三行。
 * 2. **`ruled_out` 用删除线保留，不隐藏**。读者需要看到"我们考虑过这个
 *    原因、因为某条读数排除了它"，否则会怀疑是漏了。
 * 3. **`provenance` 放在展开区第一屏**，不用 tooltip。现场工程师戴手套、
 *    环境嘈杂，悬浮提示根本点不到。
 */
export function AlarmCard({ alarm, unit, warnThreshold, excluded }: AlarmCardProps) {
  const [open, setOpen] = useState(false);
  const cls = SEVERITY_CLASS[alarm.severity];
  const indet = isIndeterminate(alarm.severity);
  const margin = describeMargin(
    alarm.value,
    warnThreshold,
    alarm.threshold_level,
    alarm.threshold_value,
    alarm.direction,
  );
  const hasDetail =
    alarm.rationale.length > 0 ||
    alarm.attribution.length > 0 ||
    excluded.length > 0 ||
    alarm.actions.length > 0 ||
    Boolean(alarm.provenance);

  return (
    <article
      className={cn(
        "rounded-panel border border-line bg-surface-2",
        // 左侧色条承担等级识别；不可判定时整卡虚线，与"正常"再拉开一层
        "border-l-2",
        cls.border,
        indet && "sev-unknown-ring",
      )}
    >
      <button
        type="button"
        onClick={() => hasDetail && setOpen((v) => !v)}
        aria-expanded={open}
        className={cn(
          "flex w-full items-start gap-2 px-2.5 py-2 text-left",
          hasDetail && "cursor-pointer hover:bg-surface-3/60",
        )}
      >
        <div className="min-w-0 flex-1">
          <div className="flex items-baseline gap-2">
            <SeverityChip severity={alarm.severity} />
            <span className="truncate text-sm text-ink-0">{alarm.label}</span>
          </div>
          <div className="mt-1 flex flex-wrap items-baseline gap-x-2 gap-y-0.5 text-2xs text-ink-2">
            <span>{STAGE_SHORT[alarm.stage]}</span>
            <span aria-hidden>·</span>
            <span>
              {margin.referenceValue === null ? (
                "未配置阈值"
              ) : (
                <>
                  {/* 越限时基准换成被越过的那条线，否则"阈值 20 · 超 1.6"自相矛盾 */}
                  {margin.referenceLabel}{" "}
                  <span className="tnum">{formatValue(margin.referenceValue)}</span>{" "}
                  {unit}
                </>
              )}
            </span>
            <span aria-hidden>·</span>
            <span className="tnum">{alarm.ts.slice(11, 16)}</span>
            {alarm.quality !== "good" && (
              <>
                <span aria-hidden>·</span>
                <span className="text-sev-warn">
                  仪表{alarm.quality === "bad" ? "失效" : "可疑"}
                </span>
              </>
            )}
          </div>
        </div>

        <div className="shrink-0 text-right">
          {/*
            不可判定时**不显示数值**，显示破折号。
            这里若回退成 0 或上一帧的旧值，卡片会读成"读数是零"。
          */}
          <div className={cn("tnum text-lg leading-none", cls.text)}>
            {indet ? "—" : formatValue(alarm.value)}
          </div>
          <div className="mt-1 text-2xs text-ink-2">
            {indet ? "无判定依据" : `${margin.text} ${unit}`}
          </div>
        </div>
      </button>

      {open && (
        <div className="border-t border-line px-2.5 py-2 text-2xs leading-relaxed">
          {alarm.rationale.length > 0 && (
            <Block label="判据">
              <ul className="space-y-0.5">
                {alarm.rationale.map((r) => (
                  <li key={r} className="text-ink-1">
                    {r}
                  </li>
                ))}
              </ul>
            </Block>
          )}

          {alarm.provenance && (
            <Block label="出处">
              <p className="text-ink-1">{alarm.provenance}</p>
            </Block>
          )}

          {alarm.attribution.length > 0 && (
            <Block label="归因">
              <ul className="space-y-1.5">
                {alarm.attribution.map((a) => (
                  <AttributionItem key={a.cause} attribution={a} />
                ))}
              </ul>
            </Block>
          )}

          {excluded.length > 0 && (
            <Block label="已排除">
              <ul className="space-y-1">
                {excluded.map((a) => (
                  <li key={a.cause}>
                    {/*
                      删除线 + 仍可读。不隐藏是因为"考虑过并排除了"本身是信息：
                      看不到它，读者无法判断是排除了还是漏了。
                    */}
                    <span className="text-ink-2 line-through">{a.cause}</span>
                    {a.ruled_out_reason && (
                      <span className="ml-1 text-ink-2">（{a.ruled_out_reason}）</span>
                    )}
                  </li>
                ))}
              </ul>
            </Block>
          )}

          {alarm.actions.length > 0 && (
            <Block label="处置">
              <ul className="space-y-0.5">
                {alarm.actions.map((a) => (
                  <li key={a.key} className="text-ink-1">
                    {a.action}
                    <span className="ml-1 text-ink-2">（{a.owner}）</span>
                  </li>
                ))}
              </ul>
            </Block>
          )}
        </div>
      )}
    </article>
  );
}

function Block({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="mb-1.5 flex gap-2 last:mb-0">
      <span className="w-7 shrink-0 text-ink-2">{label}</span>
      <div className="min-w-0 flex-1">{children}</div>
    </div>
  );
}

function AttributionItem({ attribution }: { attribution: Attribution }) {
  // 强度用文字而不是星级：`partial` 的含义是"3 条判据里命中 2 条"，
  // 而三颗星表达不了"缺的是哪一条"。
  const confidenceLabel =
    attribution.confidence === "strong"
      ? "强"
      : attribution.confidence === "partial"
        ? "中"
        : "弱";

  return (
    <li>
      <div className="flex items-baseline gap-1.5">
        <span className="text-ink-0">{attribution.cause}</span>
        <span className="shrink-0 text-2xs text-ink-2">[{confidenceLabel}]</span>
      </div>
      <ul className="mt-0.5 space-y-0.5 border-l border-line pl-2">
        {attribution.evidence.map((e) => (
          <li key={e} className="text-ink-2">
            {e}
          </li>
        ))}
      </ul>
    </li>
  );
}
