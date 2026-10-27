import { SEVERITY_CLASS, SEVERITY_META, isIndeterminate } from "../lib/severity";
import type { Severity } from "../lib/types";
import { cn } from "../lib/utils";

interface SeverityChipProps {
  severity: Severity;
  className?: string;
}

/**
 * 六级状态标签。
 *
 * 三重信号在这里落地：
 * 1. **颜色** —— 六级语义色，与品牌色隔离；
 * 2. **边框** —— 不可判定用虚线，可判定用实线；
 * 3. **文字** —— 不可判定写"不可判定"而不是"正常"。
 *
 * 只做颜色是不够的：`--sev-normal` 与 `--sev-unknown` 在压暗的深色界面上
 * 对比度接近。三重信号是给"看不清颜色"的现场准备的，不是只给色觉障碍。
 */
export function SeverityChip({ severity, className }: SeverityChipProps) {
  const meta = SEVERITY_META[severity];
  const cls = SEVERITY_CLASS[severity];
  const indet = isIndeterminate(severity);

  return (
    <span
      className={cn(
        "inline-flex shrink-0 items-center gap-1.5 rounded-panel border px-1.5 py-0.5 text-2xs leading-tight",
        indet ? "sev-unknown-ring" : "border-transparent",
        cls.text,
        cls.softBg,
        className,
      )}
    >
      <span
        aria-hidden
        className={cn(
          "inline-block size-1.5 rounded-full",
          indet ? "bg-transparent ring-1 ring-sev-unknown" : cls.dot,
        )}
      />
      {meta.label}
    </span>
  );
}
