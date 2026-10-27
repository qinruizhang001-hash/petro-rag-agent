/**
 * 引用角标 `[1] [2]`。
 *
 * ## 为什么单独一个文件
 *
 * 它被 `KbResultView`（来源表）与 `KbChunkCard`（证据卡）同时用到，
 * 而后者是前者的子组件。放在 `KbResultView` 里再让 `KbChunkCard` 反向
 * import，会形成一条看着能跑、但改一次就可能变成循环依赖的边。
 * 十行的原子组件独立成文件，代价比这条边小。
 *
 * ## 这个编号不能当文档 ID
 *
 * `reference_id` 是**响应内编号**：同一次检索里按出现频率排序，
 * 换一个问题就重新排。所以长期指向只能用 `file_path`（见 `kb-types.ts`）。
 * `title` 里写清楚这一点，是因为角标长得就像一个稳定的引用编号。
 */
export function RefBadge({ id }: { id: string }) {
  return (
    <span
      className="shrink-0 rounded-panel border border-acc/40 px-1 font-num text-acc"
      title={`本次检索的第 ${id} 个来源（响应内编号，不是文档 ID）`}
    >
      {id}
    </span>
  );
}
