import { useState } from "react";
import type { Citation } from "../lib/chat";
import type { FocusDoc } from "../lib/kb-types";
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
 * ## 它是引用跳转的中转站
 *
 * 对话里的角标 → `jumpToCitation` → 切到知识库页 + 把"看哪篇的哪一片"
 * 交给 `KnowledgePanel`。这段状态只能放在这里：`DialogPanel` 和
 * `KnowledgePanel` 是兄弟节点，一个要说话，另一个要听见，共同的父级
 * 就是这一层。再往上提到 `App` 会把"右栏内部怎么分页"变成全应用的事。
 *
 * ## 两个 Tab 都是整高、内部各自滚
 *
 * 外层 `overflow-hidden` + 内部 `min-h-0 flex-1 overflow-auto`。
 * 让外层滚的话，输入框会跟着列表一起滚出视口 —— 而输入框是这一页
 * 唯一需要一直够得着的东西。
 */
export function RightRail({ onCollapse }: { onCollapse?: () => void }) {
  // `?rail=dialog` 供截图与评审直接定位 —— 只能点两下才能到达的状态，
  // 评审时总会有人截错图。
  const [tab, setTab] = useState<Tab>(() => {
    if (typeof window === "undefined") return "kb";
    return new URLSearchParams(window.location.search).get("rail") === "dialog"
      ? "dialog"
      : "kb";
  });
  const [focusDoc, setFocusDoc] = useState<FocusDoc | null>(null);

  /**
   * 对话角标 → 知识库详情。
   *
   * 无文件血缘的引用（`source === ""`）不跳：那个落点会是一页
   * 没有标题的空白，而"点开什么都没有"比"不能点"更像故障。
   * 调用方已经保证只有可溯源的角度会调用这里，这里再挡一次。
   */
  const jumpToCitation = (c: Citation) => {
    if (!c.source) return;
    setFocusDoc({
      name: c.source,
      focus: c.chunk_id,
      fallback: { chunk_id: c.chunk_id, chars: c.chars, content: c.content },
    });
    setTab("kb");
  };

  return (
    <Panel
      title={tab === "kb" ? "知识库" : "对话"}
      className="h-full"
      aside={
        <div className="flex items-center gap-1">
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
          {/*
            收起。只在调用方给了回调时才出现 —— 单独把 `RightRail` 放进
            一个窄容器里渲染（比如测试或某个抽屉）时，"收起"没有去处。
          */}
          {onCollapse && (
            <button
              type="button"
              onClick={onCollapse}
              title="收起右栏，把宽度让给剖面。顶部「知识库」按钮可以再展开。"
              aria-label="收起右栏"
              className="rounded-panel border border-line px-1 text-2xs text-ink-2 hover:border-acc/40 hover:text-acc"
            >
              ›
            </button>
          )}
        </div>
      }
      bodyClassName="flex min-h-0 flex-col overflow-hidden p-2"
    >
      {/*
        两个面板**都挂载**，用 `hidden` 切换。
        
        这不是为了省一次重新加载，是为了别把对话弄丢：条件渲染时，
        点一个引用角标切到知识库，`DialogPanel` 会被卸载，它 `useState`
        里的对话历史随之消失 —— 读者为了核对一句出处，回来发现自己
        刚才问的话没了。而这个动作恰恰是这一页最主要的操作。
        
        代价是两个面板同时活着：`KnowledgePanel` 挂载时会取概览与文档
        清单，`DialogPanel` 会读演示快照。都是小请求，且都能被 `alive`
        标记取消，比"核对一次出处就丢一次上下文"划算得多。
      */}
      <div className={cn("flex min-h-0 flex-1 flex-col", tab !== "kb" && "hidden")}>
        <KnowledgePanel focusDoc={focusDoc} />
      </div>
      <div className={cn("flex min-h-0 flex-1 flex-col", tab !== "dialog" && "hidden")}>
        <DialogPanel onOpenCitation={jumpToCitation} />
      </div>
    </Panel>
  );
}
