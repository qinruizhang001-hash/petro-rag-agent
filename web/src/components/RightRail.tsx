import { useState } from "react";
import { cn } from "../lib/utils";
import { DialogPanel } from "./dialog/DialogPanel";
import { KnowledgePanel } from "./kb/KnowledgePanel";
import { Panel } from "./Panel";

type Tab = "kb" | "dialog";

/**
 * 右栏：知识库 + 对话。
 *
 * ## 为什么知识库放在右栏而不是中央
 *
 * 中央是**空间索引**（剖面回答"这件事发生在哪一层"），右栏是**查阅与对话**。
 * 知识库要回答的是"这条结论出自哪篇论文"，它天然是对话的支撑面 ——
 * 引用角标点开之后落在这里，读者不必把视线从结论上移开半个屏幕。
 *
 * 中央区已经有两个 Tab（剖面 / 全链），再加第三个会把"中央是空间视图"
 * 这个前提打掉：知识库是文字，放进中央等于用 55% 的面积显示一个列表。
 *
 * ## 默认停在知识库
 *
 * 知识库能回答"库里有什么"，对话能回答"现在怎么样"。值班时先看告警，
 * 需要追出处时进知识库，需要问一句时进对话 —— 而对话在 fixture 模式下
 * 只有两段快照，默认停在一个能持续用的页更诚实。
 *
 * ## 两个 Tab 都是整高、内部各自滚
 *
 * 外层 `overflow-hidden` + 内部 `min-h-0 flex-1 overflow-auto`。
 * 让外层滚的话，输入框会跟着列表一起滚出视口 —— 而输入框是这一页
 * 唯一需要一直够得着的东西。
 */
export function RightRail() {
  // `?rail=dialog` 供截图与评审直接定位 —— 只能点两下才能到达的状态，
  // 评审时总会有人截错图。
  const [tab, setTab] = useState<Tab>(() => {
    if (typeof window === "undefined") return "kb";
    return new URLSearchParams(window.location.search).get("rail") === "dialog"
      ? "dialog"
      : "kb";
  });

  return (
    <Panel
      title={tab === "kb" ? "知识库" : "对话"}
      className="h-full"
      aside={
        <div className="flex rounded-panel border border-line p-0.5" role="tablist">
          {(
            [
              ["kb", "知识库"],
              ["dialog", "对话"],
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
      bodyClassName="flex min-h-0 flex-col overflow-hidden p-2"
    >
      {tab === "kb" ? <KnowledgePanel /> : <DialogPanel />}
    </Panel>
  );
}
