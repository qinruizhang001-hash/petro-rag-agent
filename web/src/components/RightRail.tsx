import { useState } from "react";
import { cn } from "../lib/utils";
import { KnowledgePanel } from "./kb/KnowledgePanel";
import { Panel } from "./Panel";

type Tab = "kb" | "dialog";

/**
 * 右栏：知识库 + 对话与操作确认。
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
 * 对话还是占位（依赖 `/api/ops/condition` 的流式回答），知识库是真的能用的。
 * 默认停在能用的一页，比默认停在一个写着"第 3 周"的占位更诚实。
 *
 * ## 语音权限边界
 *
 * 下面那段表格是这套界面里唯一一处"做错了会伤到人"的地方，而它不依赖
 * 任何后端。放在占位区而不是写在文档里，是因为下一个接手的人会先看界面。
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
      title={tab === "kb" ? "知识库" : "对话与操作确认"}
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
      bodyClassName={cn(
        "p-2",
        tab === "kb" ? "flex min-h-0 flex-col overflow-hidden" : "overflow-auto",
      )}
    >
      {tab === "kb" ? <KnowledgePanel /> : <DialogTab />}
    </Panel>
  );
}

function DialogTab() {
  return (
    <div className="flex flex-col gap-3 text-2xs leading-relaxed">
      <section>
        <h3 className="text-xs text-ink-1">语音权限边界</h3>
        <p className="mt-1 text-ink-2">
          这条边界不是产品策略，是安全约束。它先于任何对话功能落地。
        </p>

        <table className="mt-2 w-full border-collapse text-left">
          <thead>
            <tr className="text-ink-2">
              <th className="w-1/2 border-b border-line pb-1 font-normal">语音可以做</th>
              <th className="w-1/2 border-b border-line pb-1 pl-2 font-normal text-sev-alert">
                语音不可以做
              </th>
            </tr>
          </thead>
          <tbody className="align-top">
            <tr>
              <td className="py-1 pr-2 text-ink-1">查询工况、打开某个测点</td>
              <td className="py-1 pl-2 text-ink-1">直接开关阀门</td>
            </tr>
            <tr>
              <td className="py-1 pr-2 text-ink-1">请求生成报告草稿</td>
              <td className="py-1 pl-2 text-ink-1">直接调泵速</td>
            </tr>
            <tr>
              <td className="py-1 pr-2 text-ink-1">发起"建议降速"的讨论</td>
              <td className="py-1 pl-2 text-ink-1">任何改变物理状态的操作</td>
            </tr>
          </tbody>
        </table>

        <p className="mt-2 text-ink-2">
          语音最多做到"打开确认面板"。落笔必须是人手点击 + 签名 ——
          确认弹窗里的签名框为空时，提交按钮不放行。
        </p>
      </section>

      <section className="rounded-panel border border-line bg-surface-2 px-2.5 py-2">
        <h3 className="text-xs text-ink-1">确认弹窗必须包含三样东西</h3>
        <ol className="mt-1 list-inside list-decimal space-y-0.5 text-ink-2">
          <li>
            <span className="text-ink-1">动作的具体参数</span>
            （42 t/h → 35 t/h），不是"降速"这种模糊描述
          </li>
          <li>
            <span className="text-ink-1">系统已核验 / 未核验的边界</span>
            （"环空超阈值已确认，但盖层完整性未确认"）
          </li>
          <li>
            <span className="text-ink-1">操作人签名输入框</span>（必填）
          </li>
        </ol>
        <p className="mt-1.5 text-ink-2">
          第 2 条最容易漏。只写"确认执行吗"等于把责任推给人；
          写明"系统知道什么、不知道什么"才是把决策权交还给人。
        </p>
      </section>

      <section>
        <h3 className="text-xs text-ink-1">尚未实现</h3>
        <ul className="mt-1 space-y-0.5 text-ink-2">
          <li>对话输入与流式回答（依赖 /api/ops/condition）</li>
          <li>
            对话正文里的引用角标跳转 —— 知识库页已能做（点文件名定位到切片），
            对话还没接
          </li>
          <li>审批状态机 UI（draft → submitted → approved / rejected → reopened）</li>
          <li>语音状态环（待机 / 聆听 / 思考 / 回复，四个状态，没有第五个）</li>
        </ul>
      </section>
    </div>
  );
}
