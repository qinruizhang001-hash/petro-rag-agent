import { STRENGTH_META, strengthOf } from "../../lib/evidence";
import { cn } from "../../lib/utils";

/**
 * 证据强度徽标：可溯源 / 无文件血缘。
 *
 * 出现在三处（证据卡、引用详情、对话里的引用清单），所以单独成文件 ——
 * 三处各写一份文案，迟早在某处变成"有出处"/"可引用"/"已溯源"三种说法，
 * 而它们描述的是同一件事，读者会以为是三种不同的状态。
 */
export function StrengthChip({
  source,
  className,
}: {
  source: string | null | undefined;
  className?: string;
}) {
  const meta = STRENGTH_META[strengthOf(source)];
  return (
    <span
      className={cn(
        "shrink-0 rounded-panel border px-1 text-2xs",
        meta.cls,
        className,
      )}
      title={meta.title}
    >
      {meta.label}
    </span>
  );
}
