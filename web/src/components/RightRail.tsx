import { Suspense, lazy, useEffect, useState } from "react";
import type { Citation } from "../lib/chat";
import type { FocusDoc } from "../lib/kb-types";
import { cn } from "../lib/utils";
import { KnowledgePanel } from "./kb/KnowledgePanel";
import { Panel } from "./Panel";
/**
 * 审批面板按需加载。
 *
 * 它是右栏三个页签里最重的一个：草稿正文、审批契约、正文解析器全在它
 * 那一支依赖里。而它**不是首屏必需的** —— 首屏要看的是左栏告警与中央剖面，
 * "要不要叫人"这件事不该等审批面板的代码。
 *
 * 与 `geoscene` 用同一个理由（3D 库不该阻塞告警），CI 有守卫断言它确实
 * 落在一个独立 chunk 里。
 */
const ApprovalPanel = lazy(() =>
  import("./reports/ApprovalPanel").then((m) => ({ default: m.ApprovalPanel })),
);

/**
 * 对话面板也走懒加载。
 *
 * 它是这个文件里最大的一块（611 行 + 引用角标 + 语音按钮），而默认页签是
 * 「知识库」—— 打开界面只想看告警的人，不该先下载对话面板的代码。
 *
 * 加语音之前首屏余量是 7.0 KB，加完只剩 1.6 KB：一个 226 行的麦克风按钮
 * 就把守卫线顶到"任何改动都红"。按 CI 里写好的规矩（撞线先找原因，不要
 * 改数字）查出真原因 —— 语音代码本身没问题，问题是它被静态 import 进了
 * 首屏，而它所在的整个面板本来就不该在首屏。
 */
const DialogPanel = lazy(() =>
  import("./dialog/DialogPanel").then((m) => ({ default: m.DialogPanel })),
);

type Tab = "kb" | "dialog" | "reports";

const TABS: ReadonlyArray<readonly [Tab, string]> = [
  ["kb", "知识库"],
  ["dialog", "对话"],
  ["reports", "报告"],
];

const TAB_TITLE: Record<Tab, string> = {
  kb: "知识库",
  dialog: "对话",
  reports: "报告审批",
};

/**
 * 右栏：知识库 + 对话 + 报告审批。
 *
 * ## 为什么知识库放在右栏而不是中央
 *
 * 中央是**空间索引**（剖面回答"这件事发生在哪一层"），右栏是**查阅与对话**。
 * 知识库要回答的是"这条结论出自哪篇论文"，它天然是对话的支撑面 ——
 * 引用角标点开之后落在这里，读者不必把视线从结论上移开半个屏幕。
 *
 * ## 报告审批为什么也在右栏
 *
 * 审批要看的是同一批东西：告警、剖面、出处。把它做成一个独立页面，
 * 审批人就得在两页之间来回切 —— 而"核一下这个数字对不对"是审批的
 * 主要动作，每核一次切一次页，人会干脆不核，直接点批准。
 *
 * ## 它是引用跳转的中转站
 *
 * 对话里的角标 → `jumpToCitation` → 切到知识库页 + 把"看哪篇的哪一片"
 * 交给 `KnowledgePanel`。这段状态只能放在这里：`DialogPanel` 和
 * `KnowledgePanel` 是兄弟节点，一个要说话，另一个要听见，共同的父级
 * 就是这一层。再往上提到 `App` 会把"右栏内部怎么分页"变成全应用的事。
 *
 * ## 三个 Tab 都是整高、内部各自滚
 *
 * 外层 `overflow-hidden` + 内部 `min-h-0 flex-1 overflow-auto`。
 * 让外层滚的话，输入框会跟着列表一起滚出视口 —— 而输入框是这一页
 * 唯一需要一直够得着的东西。
 */
export function RightRail({ onCollapse }: { onCollapse?: () => void }) {
  // `?rail=dialog|reports` 供截图与评审直接定位 —— 只能点两下才能到达的
  // 状态，评审时总会有人截错图。
  const [tab, setTab] = useState<Tab>(() => {
    if (typeof window === "undefined") return "kb";
    const v = new URLSearchParams(window.location.search).get("rail");
    return v === "dialog" || v === "reports" ? v : "kb";
  });
  const [focusDoc, setFocusDoc] = useState<FocusDoc | null>(null);

  /**
   * 审批面板是否已经挂载过。
   *
   * 首次切到「报告」时才挂载，之后**一直留着** —— 与另外两个面板同样的理由：
   * 写了一半的驳回意见，切一下标签就没了，而驳回意见是审计记录的一部分。
   * 只在"从未打开过"时不加载，省掉首屏那几十 KB。
   */
  const [reportsMounted, setReportsMounted] = useState(tab === "reports");
  useEffect(() => {
    if (tab === "reports") setReportsMounted(true);
  }, [tab]);

  /*
   * 对话面板同理：**代码**延迟到首次激活才下载，但一旦挂载就一直留着。
   *
   * 两件事必须分开 —— 卸载会丢对话历史（见下面那段注释），而"不下载代码"
   * 只是首屏的事。用 mounted 标志而不是 `tab === "dialog" &&`，就是把
   * 这两件事分开：切走时它还在，只是 hidden。
   */
  const [dialogMounted, setDialogMounted] = useState(tab === "dialog");
  useEffect(() => {
    if (tab === "dialog") setDialogMounted(true);
  }, [tab]);

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
      title={TAB_TITLE[tab]}
      className="h-full"
      aside={
        <div className="flex items-center gap-1">
          <div className="flex rounded-panel border border-line p-0.5" role="tablist">
            {TABS.map(([value, label]) => (
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
              title="收起右栏，把宽度让给剖面。顶部「知识库 · 对话 · 报告」按钮可以再展开。"
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
        三个面板**都挂载**，用 `hidden` 切换。
        
        这不是为了省一次重新加载，是为了别把对话弄丢：条件渲染时，
        点一个引用角标切到知识库，`DialogPanel` 会被卸载，它 `useState`
        里的对话历史随之消失 —— 读者为了核对一句出处，回来发现自己
        刚才问的话没了。而这个动作恰恰是这一页最主要的操作。
        
        审批面板同理：写了一半的驳回意见，切一下标签就没了 ——
        而驳回意见是审计记录的一部分，重打一遍往往就写得比第一次短。
        
        代价是三个面板同时活着：`KnowledgePanel` 挂载时会取概览与文档
        清单，`DialogPanel` 会读演示快照，`ApprovalPanel` 会读草稿列表。
        都是小请求，且都能被 `alive` 标记取消，比"核对一次出处就丢一次
        上下文"划算得多。
      */}
      <div className={cn("flex min-h-0 flex-1 flex-col", tab !== "kb" && "hidden")}>
        <KnowledgePanel focusDoc={focusDoc} />
      </div>
      {dialogMounted && (
        <div className={cn("flex min-h-0 flex-1 flex-col", tab !== "dialog" && "hidden")}>
          <Suspense fallback={<p className="text-2xs text-ink-2">载入对话面板…</p>}>
            <DialogPanel onOpenCitation={jumpToCitation} />
          </Suspense>
        </div>
      )}
      {reportsMounted && (
        <div className={cn("flex min-h-0 flex-1 flex-col", tab !== "reports" && "hidden")}>
          <Suspense
            fallback={<p className="text-2xs text-ink-2">载入审批面板…</p>}
          >
            <ApprovalPanel />
          </Suspense>
        </div>
      )}
    </Panel>
  );
}
