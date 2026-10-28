import { useCallback, useEffect, useState } from "react";
import { runOpsReport } from "../../lib/api";
import {
  ACTION_HINT,
  ACTION_LABEL,
  KIND_LABEL,
  STATUS_LABEL,
  blockedReason,
  formatPeriod,
  isTerminal,
  requiredFields,
} from "../../lib/approval";
import type {
  ApprovalAction,
  DraftStatus,
  ReportDraft,
  ReportSummary,
} from "../../lib/approval";
import {
  draftSource,
  draftStorageAvailable,
  localDropReason,
  rehearsal,
  resetLocalDrafts,
} from "../../lib/reports";
import {
  RUN_KIND_LABEL,
  conflictLines,
  failedAgents,
  generateBlocked,
  replayNote,
  runHeadline,
  runSuccessNote,
} from "../../lib/run";
import type { OpsRunResult, ReportKind } from "../../lib/run";
import { cn } from "../../lib/utils";
import { ReportBody } from "./ReportBody";

/**
 * 报告审批。
 *
 * ## 这一页要回答的问题
 *
 * 不是"稿子写得好不好"，是**「谁在什么时候基于什么理由批准了什么」**。
 * 所以三样东西必须同时在场：当前状态、可执行的动作、以及审批轨迹。
 * 少了轨迹，批准就只是一次点击；少了状态，读者不知道自己在批第几稿。
 *
 * ## 为什么按钮是"后端下发"的
 *
 * `draft.allowed` 由后端算（`ops/reports.py::allowed_actions`）。前端不抄
 * 那张表 —— 抄的那一刻就有了两份真值，后端加一条迁移时前端不会跟着变，
 * 表现是按钮点不动、或点下去 409，两种情况都不报错。详见 `lib/approval.ts`。
 *
 * ## 为什么禁用的按钮旁边要写字
 *
 * 一个灰按钮只能表达"不行"。读者需要知道**差什么**：是没署名、没写意见，
 * 还是稿子里还有三个数字没出处。所以每个不能做的动作都把原因显示出来，
 * 而不是塞进 `title`（触屏上根本没有 hover）。
 *
 * ## 两种模式必须写在脸上
 *
 * `fixture` 下审批轨迹存在浏览器里，`live` 下写进后端 sqlite。前者刷新不丢
 * 但不是审计记录、不跨班次。把这句话藏起来，等于让一次演练看起来像一次
 * 真实审批 —— 而审批的全部价值就在于它能被追溯。
 */
export function ApprovalPanel() {
  const [list, setList] = useState<ReportSummary[] | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [draft, setDraft] = useState<ReportDraft | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [flash, setFlash] = useState<string | null>(null);
  const [busy, setBusy] = useState<ApprovalAction | null>(null);
  const [reviewer, setReviewer] = useState("");
  const [note, setNote] = useState("");

  /** 生成动作自己的状态，与审批动作分开 —— 两者的失败含义完全不同。 */
  const [running, setRunning] = useState(false);
  const [kind, setKind] = useState<ReportKind>("daily");
  const [permitId, setPermitId] = useState("");
  const [runResult, setRunResult] = useState<OpsRunResult | null>(null);
  const [runError, setRunError] = useState<string | null>(null);

  const storageOk = useState(() => draftStorageAvailable())[0];
  const [dropped] = useState(() => (rehearsal ? localDropReason() : null));

  const refreshList = useCallback(async () => {
    try {
      const rows = await draftSource.list();
      setList(rows);
      setSelected((prev) =>
        prev && rows.some((r) => r.draft_id === prev) ? prev : (rows[0]?.draft_id ?? null),
      );
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  useEffect(() => {
    void refreshList();
    // 本机源在每次动作后通知；后端源不通知（动作的返回值就是最新状态），
    // 空订阅让两边的接口保持一致。
    return draftSource.subscribe(() => void refreshList());
  }, [refreshList]);

  useEffect(() => {
    if (!selected) {
      setDraft(null);
      return;
    }
    let alive = true;
    draftSource
      .get(selected)
      .then((d) => {
        if (alive) setDraft(d);
      })
      .catch((e: unknown) => {
        if (alive) setError(e instanceof Error ? e.message : String(e));
      });
    return () => {
      alive = false;
    };
  }, [selected]);

  /**
   * 生成一份草稿。
   *
   * 三件事必须一起发生，否则读者会以为它成功了：拿到 `draft_id`、
   * 列表刷新、新草稿被选中。少最后一步时界面上什么也没变 —— 而草稿其实
   * 已经落盘了，下一次刷新会突然多出一条，看起来像别人写的。
   */
  const generate = async () => {
    setRunning(true);
    setRunError(null);
    setRunResult(null);
    setFlash(null);
    setError(null);
    try {
      const result = await runOpsReport(kind, permitId.trim());
      setRunResult(result);
      setFlash(runSuccessNote(result));
      await refreshList();
      if (result.draft_id) setSelected(result.draft_id);
    } catch (e) {
      // 后端的 400 文案原样透出：`permit_id` 缺失那一条讲的是"没有许可证号
      // 的监管材料无法归档"，换成「生成失败」用户只能重试，而他需要的是补信息。
      setRunError(e instanceof Error ? e.message : String(e));
    } finally {
      setRunning(false);
    }
  };

  const act = async (action: ApprovalAction) => {
    if (!draft) return;
    setBusy(action);
    setError(null);
    setFlash(null);
    try {
      const next = await draftSource.act(draft.draft_id, action, {
        // 起草人优先用署名框里的名字：一个人在自己电脑上起草又提交，
        // 记成"起草方"会让轨迹里出现一个谁都不是的名字。
        actor: reviewer.trim() || "起草方",
        reviewer,
        note,
      });
      setDraft(next);
      setFlash(`${ACTION_LABEL[action]}完成 → ${STATUS_LABEL[next.status]}`);
      // 意见提交后清空。留着会让下一次动作**误带**上一次的意见，
      // 而驳回意见是审计记录的一部分，带错了没人能发现。
      setNote("");
      await refreshList();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  };

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-1.5">
      <ModeBanner storageOk={storageOk} dropped={dropped} />

      <GenerateBar
        kind={kind}
        permitId={permitId}
        running={running}
        blocked={generateBlocked(rehearsal)}
        onKind={(k) => {
          setKind(k);
          setRunResult(null);
          setRunError(null);
        }}
        onPermitId={setPermitId}
        onGenerate={() => void generate()}
      />

      {runError && (
        <p className="shrink-0 rounded-panel border border-sev-critical/60 bg-sev-critical/10 px-2 py-1 text-2xs leading-relaxed text-sev-critical">
          {runError}
        </p>
      )}

      {runResult && <RunResult result={runResult} />}

      {error && (
        <p className="shrink-0 rounded-panel border border-sev-critical/60 bg-sev-critical/10 px-2 py-1 text-2xs text-sev-critical">
          {error}
        </p>
      )}
      {flash && (
        <p className="shrink-0 rounded-panel border border-sev-normal/50 bg-sev-normal/10 px-2 py-1 text-2xs text-sev-normal">
          {flash}
        </p>
      )}

      <DraftList
        list={list}
        selected={selected}
        onSelect={(id) => {
          setSelected(id);
          setFlash(null);
          setError(null);
        }}
      />

      <div className="min-h-0 flex-1 overflow-auto border-t border-line pt-1.5">
        {draft ? (
          <DraftDetail
            draft={draft}
            busy={busy}
            reviewer={reviewer}
            note={note}
            onReviewer={setReviewer}
            onNote={setNote}
            onAct={(a) => void act(a)}
          />
        ) : (
          <p className="text-2xs text-ink-2">
            {list && list.length === 0
              ? "还没有报告草稿。生成一份之后才会出现在这里。"
              : "选择上方的一份草稿查看内容与审批轨迹。"}
          </p>
        )}
      </div>
    </div>
  );
}

/**
 * 生成栏：唯一会新建记录的动作。
 *
 * ## 为什么按钮旁边必须写清"它要跑什么"
 *
 * 「生成报告」四个字让人以为是导出，而它实际上要现场跑两个 agent 再做一次
 * 跨 agent 冲突检测。不知道这件事的人，会在等的时候以为卡住了，或者更糟：
 * 以为它只是把已有的数字排版一下，于是不核对结论。
 *
 * ## 为什么监管材料要多一个许可证号
 *
 * 后端硬性要求（`permit_id` 为空直接 400）。界面上先在**动手之前**拦住，
 * 并把原因写在按钮旁边 —— 让用户打完字点一下、看到一句 400 的 JSON，
 * 是同一件事的更差版本。
 */
function GenerateBar({
  kind,
  permitId,
  running,
  blocked,
  onKind,
  onPermitId,
  onGenerate,
}: {
  kind: ReportKind;
  permitId: string;
  running: boolean;
  blocked: string | null;
  onKind: (k: ReportKind) => void;
  onPermitId: (v: string) => void;
  onGenerate: () => void;
}) {
  const [elapsed, setElapsed] = useState(0);
  useEffect(() => {
    if (!running) {
      setElapsed(0);
      return;
    }
    const t0 = Date.now();
    const id = setInterval(() => setElapsed(Math.round((Date.now() - t0) / 1000)), 500);
    return () => clearInterval(id);
  }, [running]);

  const needsPermit = kind === "regulatory";
  const permitMissing = needsPermit && !permitId.trim();
  const disabled = blocked !== null || running || permitMissing;

  return (
    <section className="shrink-0 rounded-panel border border-line px-1.5 py-1">
      <div className="flex items-center gap-1">
        <select
          value={kind}
          onChange={(e) => onKind(e.target.value as ReportKind)}
          aria-label="报告种类"
          className="rounded-panel border border-line bg-surface-2 px-1 py-0.5 text-2xs text-ink-1"
        >
          {(Object.keys(RUN_KIND_LABEL) as ReportKind[]).map((k) => (
            <option key={k} value={k}>
              {RUN_KIND_LABEL[k]}
            </option>
          ))}
        </select>
        <button
          type="button"
          disabled={disabled}
          onClick={onGenerate}
          className={cn(
            "ml-auto rounded-panel border px-2 py-0.5 text-2xs",
            disabled
              ? "cursor-not-allowed border-line text-ink-2 opacity-60"
              : "border-acc/50 text-acc hover:bg-acc/10",
          )}
        >
          {running ? `生成中… ${elapsed}s` : "生成草稿"}
        </button>
      </div>

      {needsPermit && (
        <input
          value={permitId}
          onChange={(e) => onPermitId(e.target.value)}
          placeholder="许可证号（监管材料必填）"
          aria-label="许可证号"
          className="mt-1 w-full rounded-panel border border-line bg-surface-0 px-1.5 py-0.5 text-2xs text-ink-0 placeholder:text-ink-2 focus:border-acc/60 focus:outline-none"
        />
      )}

      {/* 灰按钮旁边必须写字：触屏上没有 hover */}
      <p className="mt-0.5 text-[0.65rem] leading-snug text-ink-2">
        {blocked
          ? blocked
          : running
            ? "主管 agent 正在跑「告警判定 → 报告撰写 → 冲突检测」。不调模型，通常一两秒。"
            : permitMissing
              ? "监管材料必须填许可证号 —— 没有许可证号的监管材料无法归档，也核不清对应哪个场地。"
              : "会现场跑一遍 agent 并落盘成草稿；读数取回放样本，不是当期实测。"}
      </p>
    </section>
  );
}

/**
 * 这一趟跑成了什么样。
 *
 * 三样东西一起显示，缺一样都会让"已生成"变成一句无法核对的话：
 * 哪几个 agent 跑了（及各自耗时）、有没有失败、有没有阻断性冲突。
 */
function RunResult({ result }: { result: OpsRunResult }) {
  const failures = failedAgents(result);
  const conflicts = conflictLines(result);
  const blocking = result.conflicts.some((c) => c.blocking);

  return (
    <section className="shrink-0 space-y-0.5 rounded-panel border border-line px-2 py-1">
      <p className="text-2xs text-ink-1">{runHeadline(result)}</p>
      <ul className="space-y-0.5">
        {result.runs.map((r) => (
          <li key={r.agent} className="text-[0.65rem] leading-snug text-ink-2">
            <span className={r.ok ? "text-ink-1" : "text-sev-critical"}>
              {r.ok ? "✓" : "✕"} {r.agent}
            </span>{" "}
            · {r.elapsed_s.toFixed(2)}s · {r.reason}
          </li>
        ))}
      </ul>
      {failures.length > 0 && (
        <ul className="space-y-0.5">
          {failures.map((f, i) => (
            <li key={i} className="text-[0.65rem] leading-snug text-sev-critical">
              {f}
            </li>
          ))}
        </ul>
      )}
      {conflicts.length > 0 && (
        <ul className="space-y-0.5 border-t border-line pt-0.5">
          {conflicts.map((c, i) => (
            <li key={i} className="text-[0.65rem] leading-snug text-ink-2">
              {c}
            </li>
          ))}
        </ul>
      )}
      {blocking && (
        <p className="text-[0.65rem] leading-snug text-sev-warn">
          有阻断性冲突：这份稿子不能提交，先按冲突里的建议处理。
        </p>
      )}
      <p className="text-[0.65rem] leading-snug text-ink-2">
        {replayNote(result.period)}
      </p>
    </section>
  );
}

function ModeBanner({
  storageOk,
  dropped,
}: {
  storageOk: boolean;
  dropped: "empty" | "corrupt" | "stale" | null;
}) {
  if (!rehearsal) {
    return (
      <p className="shrink-0 rounded-panel border border-line px-2 py-1 text-2xs leading-relaxed text-ink-2">
        <span className="text-acc">已连后端</span> · 审批轨迹写入{" "}
        <code className="text-ink-1">ops_reports.db</code>，跨请求、跨班次可见。
      </p>
    );
  }
  return (
    <div className="shrink-0 rounded-panel border border-sev-notice/50 bg-sev-notice/5 px-2 py-1 text-2xs leading-relaxed text-ink-2">
      <p>
        <span className="text-sev-notice">本机演练</span>（fixture）· 轨迹存在这个浏览器里，
        刷新不丢；<span className="text-ink-1">但不是审计记录</span>，不跨班次、不跨设备。
      </p>
      {!storageOk && (
        <p className="mt-0.5 text-sev-warn">
          本机浏览器不允许本地存储，<span className="text-ink-1">刷新就会丢</span>。
        </p>
      )}
      {dropped === "stale" && (
        <p className="mt-0.5 text-ink-2">种子草稿已更新，本机演练数据已重置。</p>
      )}
      {dropped === "corrupt" && (
        <p className="mt-0.5 text-sev-warn">
          本机存的数据坏了，已整体丢弃。不逐条抢救 —— 一份只恢复一半的审批表，
          会让某几条显示成「已批准」而实际没人批过。
        </p>
      )}
      <div className="mt-0.5 flex items-center gap-2">
        <span className="text-ink-2">要真审批请把 VITE_OPS_SOURCE 设成 live。</span>
        <button
          type="button"
          onClick={() => {
            resetLocalDrafts();
          }}
          className="ml-auto rounded-panel border border-line px-1 text-2xs text-ink-2 hover:border-acc/40 hover:text-acc"
        >
          重置演练数据
        </button>
      </div>
    </div>
  );
}

function DraftList({
  list,
  selected,
  onSelect,
}: {
  list: ReportSummary[] | null;
  selected: string | null;
  onSelect: (id: string) => void;
}) {
  if (list === null) {
    return <p className="shrink-0 text-2xs text-ink-2">载入草稿…</p>;
  }
  if (list.length === 0) return null;
  return (
    <ul className="max-h-[34%] shrink-0 space-y-0.5 overflow-auto">
      {list.map((r) => {
        const active = r.draft_id === selected;
        return (
          <li key={r.draft_id}>
            <button
              type="button"
              onClick={() => onSelect(r.draft_id)}
              aria-current={active}
              className={cn(
                "w-full rounded-panel border px-1.5 py-1 text-left text-2xs",
                active
                  ? "border-acc/50 bg-acc/10"
                  : "border-transparent hover:border-line hover:bg-surface-2",
              )}
            >
              <span className="flex items-baseline gap-1">
                <StatusChip status={r.status} />
                <span className="shrink-0 text-ink-2">{KIND_LABEL[r.kind]}</span>
                {r.problem_count > 0 && (
                  <span className="ml-auto shrink-0 text-sev-warn">
                    引用检查 {r.problem_count} 项未过
                  </span>
                )}
              </span>
              <span className="mt-0.5 block truncate text-ink-1">{r.title}</span>
            </button>
          </li>
        );
      })}
    </ul>
  );
}

function StatusChip({ status }: { status: DraftStatus }) {
  return (
    <span
      className={cn(
        "shrink-0 rounded-panel border px-1",
        STATUS_CLASS[status] ?? "border-line text-ink-2",
      )}
    >
      {STATUS_LABEL[status] ?? status}
    </span>
  );
}

/**
 * 状态色。**写成字面量**，不能拼 `border-sev-${token}` ——
 * 拼接出来的类名运行时存在、编译期扫不到，颜色会静默失效。
 */
const STATUS_CLASS: Record<DraftStatus, string> = {
  draft: "border-line text-ink-2",
  pending_review: "border-sev-warn/60 text-sev-warn",
  approved: "border-sev-normal/60 text-sev-normal",
  rejected: "border-sev-alert/60 text-sev-alert",
};

function DraftDetail({
  draft,
  busy,
  reviewer,
  note,
  onReviewer,
  onNote,
  onAct,
}: {
  draft: ReportDraft;
  busy: ApprovalAction | null;
  reviewer: string;
  note: string;
  onReviewer: (v: string) => void;
  onNote: (v: string) => void;
  onAct: (a: ApprovalAction) => void;
}) {
  const allowed = draft.allowed ?? [];
  const needsReviewer = allowed.some((a) => requiredFields(a).includes("reviewer"));
  const needsNote = allowed.includes("reject");
  const optionalNote = allowed.includes("approve");

  return (
    <div className="space-y-1.5">
      <header className="space-y-0.5">
        <h3 className="text-xs leading-snug text-ink-0">{draft.title}</h3>
        <p className="flex flex-wrap items-center gap-x-2 gap-y-0.5 text-2xs text-ink-2">
          <StatusChip status={draft.status} />
          <span>{KIND_LABEL[draft.kind]}</span>
          <span>{formatPeriod(draft.period)}</span>
          <span>{draft.citation_count} 处引用</span>
          {draft.permit_id && <span>许可 {draft.permit_id}</span>}
        </p>
        <p className="font-mono text-[0.65rem] text-ink-2">{draft.draft_id}</p>
      </header>

      {draft.seed_note && (
        <p className="rounded-panel border border-sev-notice/50 bg-sev-notice/5 px-2 py-1 text-2xs leading-relaxed text-ink-2">
          <span className="text-sev-notice">演示样本</span>：{draft.seed_note}
        </p>
      )}

      {draft.problems.length > 0 && (
        <section className="rounded-panel border border-sev-warn/50 bg-sev-warn/5 px-2 py-1">
          <p className="text-2xs text-sev-warn">
            引用检查未通过（{draft.problems.length} 项）—— 不允许提交审批
          </p>
          <ul className="mt-0.5 space-y-0.5">
            {draft.problems.map((p, i) => (
              <li key={i} className="text-2xs leading-relaxed text-ink-1">
                · {p}
              </li>
            ))}
          </ul>
          <p className="mt-0.5 text-2xs leading-relaxed text-ink-2">
            审批人无法核对没有出处的数字，批准这样的稿子等于背书。
          </p>
        </section>
      )}

      {draft.blind_spots.length > 0 && (
        <section className="rounded-panel border border-line px-2 py-1">
          <p className="text-2xs text-ink-2">已披露的监测盲区（{draft.blind_spots.length} 项）</p>
          <ul className="mt-0.5 space-y-0.5">
            {draft.blind_spots.map((b, i) => (
              <li key={i} className="text-2xs leading-relaxed text-ink-1">
                · {b}
              </li>
            ))}
          </ul>
        </section>
      )}

      <section className="rounded-panel border border-line px-2 py-1.5">
        <ReportBody body={draft.body} citations={draft.citations ?? {}} />
      </section>

      <Trail draft={draft} />

      <section className="space-y-1 border-t border-line pt-1.5">
        {isTerminal(draft.status) ? (
          <p className="text-2xs leading-relaxed text-ink-2">
            当前状态为<span className="text-ink-1">{STATUS_LABEL[draft.status]}</span>，
            是终态，没有可执行的动作。审批轨迹已完整记录在上方。
          </p>
        ) : (
          <>
            {needsReviewer && (
              <label className="block">
                <span className="text-2xs text-ink-2">审批人（必填，不预填）</span>
                <input
                  value={reviewer}
                  onChange={(e) => onReviewer(e.target.value)}
                  placeholder="你的姓名"
                  className="mt-0.5 w-full rounded-panel border border-line bg-surface-0 px-1.5 py-1 text-2xs text-ink-0 placeholder:text-ink-2 focus:border-acc/60 focus:outline-none"
                />
                <span className="mt-0.5 block text-[0.65rem] leading-snug text-ink-2">
                  不预填上次的署名 —— 批准人必须自己确认这次是谁在批。
                </span>
              </label>
            )}
            {(needsNote || optionalNote) && (
              <label className="block">
                <span className="text-2xs text-ink-2">
                  意见{needsNote ? "（驳回必填）" : "（可选）"}
                </span>
                <textarea
                  value={note}
                  onChange={(e) => onNote(e.target.value)}
                  rows={2}
                  placeholder={needsNote ? "要改什么？否则下一稿会原样再来一遍" : "核对结论"}
                  className="mt-0.5 w-full resize-none rounded-panel border border-line bg-surface-0 px-1.5 py-1 text-2xs text-ink-0 placeholder:text-ink-2 focus:border-acc/60 focus:outline-none"
                />
              </label>
            )}

            <div className="flex flex-wrap gap-1">
              {allowed.map((action) => {
                const reason = blockedReason(draft, action, { reviewer, note });
                return (
                  <button
                    key={action}
                    type="button"
                    disabled={reason !== null || busy !== null}
                    title={reason ?? ACTION_HINT[action]}
                    onClick={() => onAct(action)}
                    className={cn(
                      "rounded-panel border px-2 py-1 text-2xs",
                      reason !== null
                        ? "cursor-not-allowed border-line text-ink-2 opacity-60"
                        : "border-acc/50 text-acc hover:bg-acc/10",
                    )}
                  >
                    {busy === action ? `${ACTION_LABEL[action]}中…` : ACTION_LABEL[action]}
                  </button>
                );
              })}
            </div>

            {/* 灰按钮旁边必须写字：触屏上没有 hover，`title` 谁也看不到 */}
            {allowed.map((action) => {
              const reason = blockedReason(draft, action, { reviewer, note });
              if (!reason) return null;
              return (
                <p key={action} className="text-[0.65rem] leading-snug text-ink-2">
                  「{ACTION_LABEL[action]}」不可用：{reason}
                </p>
              );
            })}
          </>
        )}
      </section>
    </div>
  );
}

function Trail({ draft }: { draft: ReportDraft }) {
  return (
    <section className="rounded-panel border border-line px-2 py-1">
      <p className="text-2xs text-ink-2">审批轨迹（{draft.trail.length} 步）</p>
      {draft.trail.length === 0 ? (
        <p className="mt-0.5 text-2xs leading-relaxed text-ink-2">
          还没有任何流转。提交之后这里会记录每一步的时间、操作人与意见。
        </p>
      ) : (
        <ol className="mt-0.5 space-y-0.5">
          {draft.trail.map((t, i) => (
            <li key={i} className="text-2xs leading-relaxed text-ink-1">
              <span className="font-mono text-[0.65rem] text-ink-2">{t.at}</span>{" "}
              <span className="text-ink-0">{t.actor}</span>{" "}
              <span className="text-ink-2">
                {STATUS_LABEL[t.from]} → {STATUS_LABEL[t.to]}
              </span>
              {t.note && <span className="block pl-2 text-ink-2">「{t.note}」</span>}
            </li>
          ))}
        </ol>
      )}
      {rehearsal && draft.trail.length > 0 && (
        <p className="mt-0.5 text-[0.65rem] leading-snug text-ink-2">
          轨迹由本机记录，不是审计记录 —— 换一台机器就看不到。
        </p>
      )}
    </section>
  );
}
