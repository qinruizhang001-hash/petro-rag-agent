import { useState } from "react";
import { formatValue, SEVERITY_CLASS } from "../lib/severity";
import type { CeilingCheck, CeilingState, Severity } from "../lib/types";
import { cn } from "../lib/utils";

/**
 * 注入压力上限校核卡片。
 *
 * ## 为什么它单独一张卡，而不是并进测点列表
 *
 * 测点告警问的是"这个数偏离了配置的运行窗口吗"，上限校核问的是
 * "井底压力占破裂压力多少"。后者是**两个读数之比**，而且上限是法规给的、
 * 不能按项目调。把它塞进某个测点的卡片里，会出现"井口压力正常、
 * 井底已经超破裂压力"这种组合看不出来的情况。
 *
 * ## 为什么 `indeterminate` 必须有独立样式
 *
 * "没接井底压力计"与"压力在限内"在数据上都是一个没有告警的测点。
 * 如果两者长得一样，一个从未做过上限校核的场地会和校核通过的场地
 * 完全无法区分 —— 而监管材料里这一项是**必填**的。所以这一态用
 * 灰底 + 虚线框 + 破折号，三重信号，与六级告警的"不可判定"同源。
 */

const TONE: Record<CeilingState, Severity> = {
  ok: "normal",
  near: "warn",
  exceeded: "critical",
  indeterminate: "indeterminate",
};

interface CeilingCardProps {
  check: CeilingCheck | null | undefined;
  className?: string;
}

export function CeilingCard({ check, className }: CeilingCardProps) {
  const [open, setOpen] = useState(false);

  // 没有这一项 ≠ 合规。缺省时按"未执行"渲染，理由见组件文档。
  if (!check) {
    return (
      <div
        className={cn(
          "rounded-panel border border-dashed border-sev-unknown/60 bg-sev-unknown/10 px-2.5 py-2",
          className,
        )}
        role="status"
      >
        <div className="flex items-baseline gap-1.5">
          <span className="text-2xs font-semibold tracking-wide text-sev-unknown">
            注入压力上限
          </span>
          <span className="tnum text-2xs text-ink-2">未执行</span>
        </div>
        <p className="mt-1 text-2xs leading-relaxed text-ink-1">
          本批次数据未包含上限校核结果。
          <span className="text-ink-2">未执行 ≠ 合规</span>：
          这一项在许可条件里是必填的，缺它意味着本期没有做过校核。
        </p>
      </div>
    );
  }

  const tone = TONE[check.state];
  const cls = SEVERITY_CLASS[tone];
  const unknown = check.state === "indeterminate";
  const ratioPct = check.ratio === null ? null : check.ratio * 100;
  const limitPct = check.ceiling_ratio * 100;
  // 折叠区已经显示了 basis[0]（仅无比值那一态），展开区从下一条开始 ——
  // 同一条不重复出现两次
  const restBasis = ratioPct === null ? check.basis.slice(1) : check.basis;

  // 条形的满量程取"上限的 1.15 倍"：让 90% 那条刻度落在条内 78% 处，
  // 越限后仍有可见的超出长度。若按 100% 满量程画，越限只表现为"条满了"，
  // 看不出超了多少 —— 而超了多少正是唯一需要处置的信息。
  const fullScale = limitPct * 1.15;
  const barPct = ratioPct === null ? 0 : Math.min(100, (ratioPct / fullScale) * 100);
  const markPct = (limitPct / fullScale) * 100;

  return (
    <div
      className={cn(
        "rounded-panel border px-2.5 py-2",
        unknown ? "border-dashed" : "",
        cls.border,
        cls.softBg,
        className,
      )}
      role="status"
    >
      <div className="flex items-baseline justify-between gap-2">
        <span className={cn("text-2xs font-semibold tracking-wide", cls.text)}>
          注入压力上限
        </span>
        <span className={cn("tnum text-2xs font-medium", cls.text)}>
          {check.label}
        </span>
      </div>

      {ratioPct === null ? (
        <p className="mt-1 text-2xs leading-relaxed text-ink-1">
          {/* 缺数据时**不画条形**：一条 0% 的进度条会被读成"压力很低"，
              而实际是"没算出来"。

              原因直接取 `basis[0]`，不在这里重写一句。后端新增一种不可判定的
              情形时，卡片上的枚举就会悄悄变成不完整的 —— 而它读起来同样确定。
              展开区因此从 `basis[1]` 开始渲染，同一条不会出现两次。 */}
          {check.basis[0] ?? "本项未通过校核。"}
        </p>
      ) : (
        <>
          <div className="mt-1.5 flex items-baseline gap-1.5">
            <span className={cn("tnum text-base font-semibold leading-none", cls.text)}>
              {ratioPct.toFixed(1)}
              <span className="ml-0.5 text-2xs font-normal">%</span>
            </span>
            <span className="tnum text-2xs text-ink-2">
              / 上限 {limitPct.toFixed(0)}%
            </span>
          </div>

          <div className="relative mt-1.5 h-1.5 w-full overflow-hidden rounded-full bg-line">
            <div
              className={cn("h-full rounded-full", cls.bg)}
              style={{ width: `${barPct}%` }}
            />
            {/* 上限刻度。它不是装饰：没有这条线，"74%" 无法判断是宽裕还是贴着上限 */}
            <div
              className="absolute top-0 h-full w-px bg-ink-1/70"
              style={{ left: `${markPct}%` }}
              aria-hidden
            />
          </div>

          {/*
            两个读数**分两行**，不并排。并排时 "…bar @10:00 破裂压力… bar @10-27"
            读起来像一句话，而这两个 @ 的含义完全不同：一个是本期峰值时刻，
            一个是最近一次 LOT 的日期。分成两行之后，"哪个时刻对应哪个数"
            不需要读者去推断。
          */}
          <dl className="mt-1.5 space-y-0.5 text-2xs">
            <div className="flex items-baseline justify-between gap-2">
              <dt className="text-ink-2">井底峰值</dt>
              <dd className="tnum text-ink-1">
                {formatValue(check.bhp)} bar
                {check.bhp_ts && (
                  <span className="ml-1.5 text-ink-2">{shortTime(check.bhp_ts)}</span>
                )}
              </dd>
            </div>
            <div className="flex items-baseline justify-between gap-2">
              <dt className="text-ink-2">破裂压力</dt>
              <dd className="tnum text-ink-1">
                {formatValue(check.frac_p)} bar
                {check.frac_ts && (
                  <span className="ml-1.5 text-ink-2">{shortDate(check.frac_ts)}</span>
                )}
              </dd>
            </div>
          </dl>

          {check.headroom_bar !== null && (
            <p className={cn("tnum mt-1 text-2xs", check.headroom_bar < 0 ? cls.text : "text-ink-1")}>
              {check.headroom_bar >= 0
                ? `距上限余量 ${formatValue(check.headroom_bar)} bar`
                : `已超出上限 ${formatValue(Math.abs(check.headroom_bar))} bar`}
            </p>
          )}
        </>
      )}

      {/* 时间差超过 30 天时后端会在 basis 里给出说明。这里把它显示在展开区，
          默认收起：它是"该怎么读这个比值"，不是"要不要处置"。 */}
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="mt-1.5 text-2xs text-ink-2 underline-offset-2 hover:text-ink-1 hover:underline"
        aria-expanded={open}
      >
        {open ? "收起依据" : "查看依据与出处"}
      </button>

      {open && (
        <div className="mt-1.5 space-y-1 border-t border-line pt-1.5">
          {/* `basis` 同时被渲染进 Markdown 报告，所以里面带 `**`。这里是
              JSX 文本，星号会原样显示 —— 统一去掉，不指望作者记得少写。 */}
          {restBasis.map((line) => (
            <p key={line} className="text-2xs leading-relaxed text-ink-2">
              {line.replaceAll("**", "")}
            </p>
          ))}
          {check.next_step && (
            <p className="text-2xs leading-relaxed text-ink-1">
              <span className="text-ink-2">下一步：</span>
              {check.next_step}
            </p>
          )}
        </div>
      )}
    </div>
  );
}

/** `2026-10-27T10:00:00` → `10:00`。解析失败时原样返回，不吞掉信息。 */
function shortTime(iso: string): string {
  const m = /T(\d{2}:\d{2})/.exec(iso);
  return m?.[1] ?? iso;
}

/** `2026-10-27T06:00:00` → `10-27`。破裂压力是周期性作业结果，日期比时刻重要。 */
function shortDate(iso: string): string {
  const m = /^(\d{4})-(\d{2})-(\d{2})/.exec(iso);
  return m ? `${m[2] ?? ""}-${m[3] ?? ""}` : iso;
}
