"""
汇总节点的报告收尾（纯文本后处理）
==================================
终稿生成之后，还有一串"代码保证、不靠模型自觉"的动作：

1. 工具算出的数值没写进正文 → 补一段带区间的结论
2. 决策等级为 warn / alert → 补一个带概率与阈值的高亮块
3. 已计入井间干扰 → 补一句"已考虑井间干扰"及 f 值
4. 降级说明 → 追加「数据完整性提示」
5. 图表引用 → 补 Markdown 相对路径
6. 无法溯源的百分比 → 整句删除

这些动作原来直接写在 ``aggregate_report`` 里，和模型调用、流式输出、记忆写入
交替出现，改一处收尾逻辑得在一屏里找半天。抽出来之后它们变成**对字符串的
确定性变换**：输入一段报告，输出一段报告，中间没有模型、没有 I/O。

为什么必须是代码而不是 prompt：本地 7B 实测会忽略"必须写进正文"的指令，
而"编造的 12% 概率"更是照着 prompt 里的示例抄的。收尾这一步的可靠性来自
它是代码，不是来自措辞。
"""
from __future__ import annotations

from utils.probability_guard import strip_unsupported_probabilities
from utils.branch_merge import degradation_block

from utils.logging_setup import get_logger

log = get_logger(__name__)


def finalize_report(
    report_text: str,
    *,
    density_result: dict | None = None,
    uq_summary: dict | None = None,
    pinn_result: dict | None = None,
    branch: dict | None = None,
    chart_result: dict | None = None,
    chart_ok: bool = False,
    probability_source: str = "",
) -> tuple[str, list[str]]:
    """对终稿做全部确定性收尾，返回 ``(报告, 被删掉的百分比)``。"""
    injection_uq = (uq_summary or {}).get("injection") or {}
    branch = branch or {}
    # ─── 数值兜底：工具算出来的数不能"没写进报告"───────────
    # 模型偶尔会漏掉注入的数值。数值来自工具是核心承诺，漏了就等于没算，
    # 所以这里补一行显式数值，而不是让它自生自灭。
    #
    # v1.6 起判据从"均值出现过吗"改成"**区间上界**出现过吗"：
    # 模型很容易只抄一个 1.35 就交差，而"1.35"恰恰是这次改造要消灭的
    # 那种伪确定性。上界没出现，说明区间没写进去。
    if density_result and density_result.get("mean") is not None and density_result.get("p95") is not None:
        mean_txt = f"{density_result['mean']:.2f}"
        hi_txt = f"{density_result['p95']:.2f}"
        if hi_txt not in report_text:
            log.info(
                f"  [System] 报告缺少密度置信区间（上界 {hi_txt} 未出现）"
                f" → 追加概率性结论"
            )
            report_text = report_text.rstrip() + (
                f"\n\n> **CO2 密度（概率性结论）**：{density_result['text']}。"
                f"均值 {mean_txt} kg/m³ 不是唯一可能值，"
                f"上述区间覆盖 90% 的模拟结果。"
            )

    # ─── 注入风险概率兜底：同上，模型倾向于只写"有风险/无风险" ───
    if injection_uq.get("text"):
        labels = ("注入可行", "压裂风险", "压力不足")
        if not any(lbl in report_text for lbl in labels):
            log.info("  [System] 报告缺少注入风险概率 → 追加概率性结论")
            report_text = report_text.rstrip() + (
                f"\n\n> **注入风险（概率性结论）**：{injection_uq['text']}。"
                f"概率来自 100 次蒙特卡洛抽样，不是主观判断。"
            )

    # ─── PINN 数值兜底：与密度同理，工具算了但没写进报告就等于没算 ───
    # 实测本地 7B 会忽略 facts_block 里"必须写进正文"的指令（报告里只字未提
    # 109.95 bar），所以这里和密度一样由代码补。判断依据是数值本身是否出现，
    # 而不是"模型有没有提到压力" —— 后者无法可靠判断。
    if pinn_result is not None:
        pnum = f"{pinn_result['pressure_bar']:.2f}"
        if pnum not in report_text:
            log.info(f"  [System] 报告未包含 PINN 预测 {pnum} bar → 追加数值说明")
            report_text = report_text.rstrip() + f"\n\n> {pinn_result['text']}。"

    # ─── 风险决策：warn / alert 必须带概率与阈值出现在正文里 ──────
    # 判据是"概率串 + 阈值"两样都在。只查一个"有风险"之类的词不行：
    # 模型完全可能写"存在压裂风险"却不给数字，而那正是最需要数字的地方。
    # 反过来，两样都在就说明它确实写了，不必重复。
    decision = (uq_summary or {}).get("decision") or {}
    level = decision.get("level")
    if level in ("warn", "alert") and decision.get("p") is not None:
        th = decision.get("thresholds") or {}
        p_txt = f"{decision['p'] * 100:.1f}%"
        if p_txt not in report_text or "阈值" not in report_text:
            log.info(
                f"  [System] 决策等级 {level} 但正文缺少概率/阈值"
                f"（{p_txt} / 阈值）→ 追加高亮决策块"
            )
            ci = decision.get("p_ci") or []
            ci_txt = (
                f"（95% 区间 {ci[0] * 100:.1f}%–{ci[1] * 100:.1f}%，"
                f"{decision.get('n')} 次模拟）"
                if len(ci) == 2 else ""
            )
            label = decision.get("label", "压裂风险")
            warn_txt = f"{th.get('warn', 0) * 100:.0f}%"
            alert_txt = f"{th.get('alert', 0) * 100:.0f}%"
            # 阈值来路必须跟着数字走：环境变量覆盖过的阈值与默认值不是一回事，
            # 报告写"5%"而实际用的是 8% 是这类系统最难发现的错之一。
            prov = (
                "（阈值由环境变量覆盖，非默认值）"
                if th.get("source") == "env" else ""
            )
            report_text = report_text.rstrip() + (
                f"\n\n> **风险决策（{decision.get('level_label', level)}）**："
                f"P({label}) = {p_txt}{ci_txt}，"
                f"{'达到告警阈值' if level == 'alert' else '达到提示阈值'} "
                f"{alert_txt if level == 'alert' else warn_txt}。"
                f"判定规则：P({label}) ≥ {alert_txt} → 告警；"
                f"≥ {warn_txt} → 提示{prov}。"
            )

    # ─── 井间干扰：算过就必须写出来 ────────────────
    # "已考虑干扰"与"没考虑"给出的是两个不同的速率，而正文里看不出来。
    # 这里只保证**声明与 f 值**出现；具体论证仍在 notes / 面板里。
    plan_physics = ((uq_summary or {}).get("plan") or {}).get("physics") or {}
    if plan_physics.get("interference_active"):
        f_txt = f"{plan_physics['interference_factor']:.4f}"
        if "井间干扰" not in report_text:
            log.info("  [System] 报告未提井间干扰（已计入）→ 追加声明")
            how = (
                "按给定的假设干扰系数"
                if plan_physics.get("interference_source") == "user_coefficient"
                else f"按等效井距 {plan_physics.get('well_spacing_m'):g} m"
            )
            report_text = report_text.rstrip() + (
                f"\n\n> **已考虑井间干扰**：{how}，f = {f_txt}。"
                f"注入速率按 q = n·J·(p−P_h)/(1+(n−1)f) 计算，"
                f"因此加井不按线性增注。"
            )
        elif f_txt not in report_text:
            log.info("  [System] 报告提到干扰但缺 f 值 → 追加 f")
            report_text = report_text.rstrip() + (
                f"\n\n> 井间干扰系数 f = {f_txt}"
                f"（q = n·J·(p−P_h)/(1+(n−1)f)），供回代核对。"
            )

    # ─── 降级说明：代码追加，不依赖模型照做 ────────────────
    # 这段必须放在图片引用**之前**吗？不必 —— 但要放在正文之后、且保证唯一。
    # 重写轮次（feedback 非空）会重新生成本节点，所以这里每次都重新追加，
    # 不会因为上一轮的追加而重复（report_text 是本轮模型输出，不含旧尾巴）。
    note = degradation_block(branch)
    if note:
        report_text = report_text.rstrip() + f"\n\n{note}\n"

    # ─── 报告末尾补齐图片引用（代码保证，不靠模型自觉）──────
    # 报告本体存 Markdown 相对路径，便于归档与其它消费者；
    # 前端用 chart 事件里的真实路径调 st.image，不会重复渲染。
    if chart_ok:
        ref = f"![地层压力随深度变化曲线]({chart_result['rel_path']})"
        if chart_result["rel_path"] not in report_text:
            report_text = report_text.rstrip() + f"\n\n{ref}\n"

    # ─── 概率溯源：删掉无法溯源的百分比（代码保证，不靠模型自觉）──
    # 评审节点会因此扣分并打回重写，但重写预算只有两次；预算用尽后整份报告
    # 会被判失败（见 core.gates）。与其把"我们知道它没有依据"的数字发出去，
    # 不如在这里直接删掉。
    report_text, stripped_probabilities = strip_unsupported_probabilities(
        report_text, probability_source
    )
    if stripped_probabilities:
        log.warning(
            f"  [System] 删除 {len(stripped_probabilities)} 处无法溯源的百分比："
            f"{stripped_probabilities[:3]}"
        )

    return report_text, stripped_probabilities
