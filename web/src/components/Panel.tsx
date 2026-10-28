import type { ReactNode } from "react";
import { cn } from "../lib/utils";

interface PanelProps {
  title?: ReactNode;
  /** 右侧的补充信息：计数、单位、时间范围 */
  aside?: ReactNode;
  children: ReactNode;
  className?: string;
  bodyClassName?: string;
  /**
   * 是否画对角角标。
   *
   * 设计规范 §2.4 只保留两处 HUD 装饰，这是其二：12 px 的 L 形线替代粗边框。
   * 工程师模式关掉 —— 现场看的是数字，角标只是噪声。
   */
  ticks?: boolean;
}

/**
 * 面板外框。
 *
 * 刻意不用 `border` 包一圈：四角角标已经说明了边界，再加一整圈实线
 * 会在三栏布局里形成大量平行线，眼睛没有落点。这条经验来自参考图 ——
 * 它们的面板之所以不糊，是因为几乎没有完整的矩形边框。
 */
export function Panel({
  title,
  aside,
  children,
  className,
  bodyClassName,
  ticks = true,
}: PanelProps) {
  return (
    <section
      className={cn(
        "relative flex min-h-0 flex-col bg-surface-1",
        className,
      )}
    >
      {ticks && <CornerTicks />}
      {(title || aside) && (
        /*
         * 标题行必须能换行，标题本身不许被压断。
         *
         * 默认的 `flex` 让两个子项一起缩，而 flex 子项会缩到内容宽度以下 ——
         * 于是「报警与关键指标」被挤成「报警与关键指 / 标」，旁边那串
         * 「4 需动作 · 3 不可判定 / 共 18」被压成竖排单字。两个元素都在，
         * 但都读不出来，而控制台里没有任何报错。
         *
         * 判据：`flex-wrap` + 标题 `whitespace-nowrap`。挤不下时换行，
         * 不缩字。aside 内部自己再决定是"计数 + 按钮"一行还是两行。
         */
        <header className="flex flex-wrap items-baseline justify-between gap-x-3 gap-y-1 border-b border-line px-3 py-2">
          <h2 className="whitespace-nowrap text-xs font-normal tracking-wide text-ink-1">
            {title}
          </h2>
          {aside && <div className="min-w-0 text-2xs text-ink-2">{aside}</div>}
        </header>
      )}
      <div className={cn("min-h-0 flex-1 overflow-auto p-3", bodyClassName)}>
        {children}
      </div>
    </section>
  );
}

function CornerTicks() {
  // 12 px 的 L 形线。用 4 个绝对定位的 div 而不是伪元素，
  // 因为伪元素在 Tailwind 里要写 arbitrary variants，可读性更差。
  const common = "pointer-events-none absolute h-3 w-3 border-acc/40";
  return (
    <>
      <span className={cn(common, "left-0 top-0 border-l border-t")} />
      <span className={cn(common, "right-0 top-0 border-r border-t")} />
      <span className={cn(common, "bottom-0 left-0 border-b border-l")} />
      <span className={cn(common, "bottom-0 right-0 border-b border-r")} />
    </>
  );
}
