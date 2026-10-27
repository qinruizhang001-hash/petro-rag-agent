"""全链条运维接口（ops/）。

为什么单独一个模块，而不是写进 ``api_server.py``
------------------------------------------------
``api_server.py`` 顶部 ``from workflow import ...`` 会加载 LangGraph 全图与
2.2 GB 的重排模型。把运维端点写在那里，意味着**任何一次接口测试都要先把
模型拉起来** —— 在 CI 的零依赖 job 里根本跑不了，只能靠读代码猜行为。

拆出来之后本模块只依赖 ``ops/``（纯 stdlib）与 FastAPI，测试可以直接
``import api_ops`` 挂一个裸 ``FastAPI()`` 跑，不需要模型、不需要 Chroma、
不需要 Ollama。这是本文件存在的唯一理由，不是风格偏好。

鉴权不在本模块里
----------------
本模块的 ``router`` **不带** ``Depends(require_token)``，因为 ``require_token``
定义在 ``api_server.py``，反向导入会成环。鉴权在 ``api_server.py`` 的
``include_router`` 上统一施加。这个分工有个副作用：谁把
``api.include_router(ops_router)`` 写成不带 dependencies 的形式，
运维接口就裸奔了 —— 所以 ``tests/test_api_ops.py`` 里有一条静态断言盯着它。

与 ``/api/analyze*`` 的分工
---------------------------
``/api/analyze*`` 回答"这个项目有什么风险"——离线、一问一答、分钟级，
产出是一篇评估报告。
``/api/ops/*`` 回答"现在链上发生了什么、要不要叫人、要写什么"——在线、
连续、秒级，产出是告警清单与审批材料。

三条贯穿全组的约束（与 ``ops/`` 包一致）：
  1. 没有出处的阈值不参与判定 → 未配阈值的测点返回 indeterminate，不是 normal；
  2. 归因给候选不给结论 → 响应里同时带 evidence 与 ruled_out_reason；
  3. 每个数字可回代 → 报告正文里的测量值都带 ``[alarm:...]`` 引用。

响应形状直接来自 ``ops/`` 的 ``to_dict()``，不再包一层。多包一层意味着
前端要维护两份字段名，而两份迟早会不一致。
"""
from __future__ import annotations

import asyncio
from datetime import datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict

from ops import alarms as ops_alarms
from ops import ceiling as ops_ceiling
from ops import condition as ops_condition
from ops import reports as ops_reports
from ops import store as ops_store
from ops import supervisor as ops_supervisor
from ops import thresholds as ops_thresholds
from ops.contracts import QUALITIES, STAGE_LABELS, Reading
from utils.logging_setup import get_logger

log = get_logger(__name__)

#: 不带鉴权依赖，由 api_server 在 include 时统一施加（见模块文档）。
router = APIRouter(tags=["ops"])


# ══════════════════════════════════════════════════════════
# 请求模型
# ══════════════════════════════════════════════════════════
class ReadingIn(BaseModel):
    """一条测点读数。

    ``ts`` 必须是带时间的 ISO 字符串。**不接受"省略 ts 用服务器当前时间"**：
    离线复盘时数据可能是三天前的，用当前时间会让"最近 1 小时"的窗口取到空集，
    而空集会被渲染成"没有异常"。
    """

    #: 多余字段一律拒绝。默认的"忽略未知字段"会让 ``{"readngs": [...]}``
    #: 这种拼写错误变成一个 200 + 空读数，然后报"没有数据"——
    #: 排查方向完全错了。宁可 422 让它当场失败。
    model_config = ConfigDict(extra="forbid")

    sensor: str
    value: float
    ts: datetime
    quality: str = "good"


class PeriodIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start: datetime
    end: datetime


class OpsReadingsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    readings: list[ReadingIn] = []
    period: PeriodIn | None = None


class OpsAlarmsRequest(OpsReadingsRequest):
    # 三个政策参数可覆盖，但**没有标准依据**，见 ops/alarms.py 的说明。
    # 客户端不传就用默认值，默认值的出处写在响应的 policy 字段里一起返回。
    notice_margin: float = ops_alarms.DEFAULT_NOTICE_MARGIN
    persist_points: int = ops_alarms.DEFAULT_PERSIST_POINTS
    rapid_intervals: float = ops_alarms.DEFAULT_RAPID_INTERVALS_PER_HOUR


class OpsConditionRequest(OpsReadingsRequest):
    question: str
    #: "现在" = 数据最新时刻，默认由服务端从读数里取。允许显式指定以便复盘。
    now: datetime | None = None


class OpsRunRequest(OpsReadingsRequest):
    question: str = ""
    #: 离线复盘时显式指定"现在"；不传则由读数最新时刻决定。
    now: datetime | None = None
    want_alarms: bool = True
    want_condition: bool = False
    want_report: bool = False
    report_kind: str = "daily"
    permit_id: str = ""
    periods: list[PeriodIn] = []


class DraftSubmitRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    actor: str = "起草方"


class DraftReviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reviewer: str
    note: str = ""


# ══════════════════════════════════════════════════════════
# 内部工具
# ══════════════════════════════════════════════════════════
def _ops_registry():
    """装载阈值表。

    **每次请求重读配置文件**，不做进程级缓存。理由：运维调整阈值后必须立刻
    生效，而"改一个数要重启服务"会让人干脆不改。配置文件只有几 KB，
    一次读盘的成本远低于让一套阈值长期停留在错误值上。
    """
    try:
        return ops_thresholds.load_registry()
    except ValueError as exc:
        # 配置文件本身坏了是服务端问题，不是调用方的错 —— 500 而不是 400。
        log.error(f"[OPS] 阈值配置无法装载：{exc}")
        raise HTTPException(status_code=500, detail=f"阈值配置无法装载：{exc}")


def _ops_readings(rows: list[ReadingIn]) -> list[Reading]:
    out: list[Reading] = []
    for row in rows:
        if row.quality not in QUALITIES:
            raise HTTPException(
                status_code=400,
                detail=f"读数 {row.sensor!r} 的 quality={row.quality!r} 必须是 {QUALITIES} 之一",
            )
        try:
            # Reading 的 __post_init__ 会拒绝 NaN / Inf —— NaN 与任何数比较都为假，
            # 会被当成"没超阈值"，是最危险的一类静默错误。
            out.append(
                Reading(sensor=row.sensor, value=row.value, ts=row.ts, quality=row.quality)
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"读数不合法：{exc}")
    return out


def _ops_period(period: PeriodIn | None, readings: list[Reading]) -> tuple[datetime, datetime]:
    """时间范围：优先用调用方给的；否则从读数里取最早/最晚。

    两者都没有时返回 400 而不是"默认最近一小时"—— 一个空窗口会产出一份
    "未发现异常"的报告，而真实原因是没给数据。
    """
    if period is not None:
        if period.end < period.start:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"period.end（{period.end.isoformat()}）"
                    f"早于 period.start（{period.start.isoformat()}）"
                ),
            )
        return (period.start, period.end)
    if not readings:
        raise HTTPException(
            status_code=400,
            detail="既没有提供 period，也没有读数可以用来推断时间范围",
        )
    stamps = [r.ts for r in readings]
    return (min(stamps), max(stamps))


def _ops_call(fn, *args, **kwargs):
    """把 ops 层的 ValueError 统一映射成 400。

    ops/ 的 ValueError 都是**调用方可修正**的问题（未知测点、参数越界、
    状态迁移不合法），把它们变成 500 会让调用方以为是自己写错了代码。
    """
    try:
        return fn(*args, **kwargs)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


def _load_draft(draft_id: str) -> ops_reports.ReportDraft:
    draft = ops_store.load(draft_id)
    if draft is None:
        raise HTTPException(status_code=404, detail=f"未找到报告草稿 {draft_id}")
    return draft


# ══════════════════════════════════════════════════════════
# 阈值与测点目录
# ══════════════════════════════════════════════════════════
@router.get("/api/ops/thresholds")
async def ops_threshold_status():
    """阈值注册表状态 + 全部测点目录。

    前端据此渲染"测点表"与"盲区提示"。``missing`` 不是装饰字段：
    它列出的是**已声明但没配阈值**的测点，这些测点有读数却无法判定，
    必须与"正常"在界面上区分开。
    """
    registry = _ops_registry()
    sensors = []
    for spec in ops_thresholds.DEFAULT_SENSORS:
        th = registry.get(spec.key)
        sensors.append(
            {
                "key": spec.key,
                "label": spec.label,
                "unit": spec.unit,
                "stage": spec.stage,
                "stage_label": STAGE_LABELS[spec.stage],
                "direction": spec.direction,
                "basis": spec.basis,
                "threshold": th.to_dict() if th else None,
            }
        )
    return {
        "origin": registry.origin,
        "configured": registry.configured(),
        "missing": registry.missing(),
        "sensors": sensors,
    }


# ══════════════════════════════════════════════════════════
# 告警 / 工况问答
# ══════════════════════════════════════════════════════════
@router.post("/api/ops/alarms")
async def ops_alarms_endpoint(request: OpsAlarmsRequest):
    """只跑告警分级与归因（最常用，也最快）。

    响应里同时带 ``ceiling``：注入压力上限校核不依赖阈值表，也不依赖告警
    分级，它只依赖 bhp 与 frac_p 两个读数。放进这个端点是因为前端拿到告警
    清单时最想同时知道的一件事就是"注入本身合不合规"。

    请求里的 ``readings`` 除窗口内的时序外，**还须带上最近一次 LOT/XLOT 的
    ``frac_p`` 读数**，哪怕它的时间戳在窗口之外。只发窗口数据会让 ``ceiling``
    恒为 ``indeterminate`` —— 而它读起来和"校核没通过"一样，实际是"没送进来"。
    """
    readings = _ops_readings(request.readings)
    period = _ops_period(request.period, readings)
    registry = _ops_registry()

    report = _ops_call(
        ops_alarms.evaluate,
        readings,
        registry,
        notice_margin=request.notice_margin,
        persist_points=request.persist_points,
        rapid_intervals=request.rapid_intervals,
    )
    payload = report.to_dict()
    payload["period"] = [period[0].isoformat(), period[1].isoformat()]
    payload["ceiling"] = _ops_call(ops_ceiling.check, readings).to_dict()
    return payload


@router.post("/api/ops/condition")
async def ops_condition_endpoint(request: OpsConditionRequest):
    """对时序的工况问答。

    ``unsourced_numbers`` 恒为空是**正常**的：它非空表示模型生成的措辞里
    出现了无法在事实块中找到的数字，此时 ``text`` 会整篇回退到模板。
    前端应当把它显示出来而不是隐藏 —— 那是一次真实的守卫触发。
    """
    readings = _ops_readings(request.readings)
    registry = _ops_registry()
    answer = _ops_call(
        ops_condition.answer,
        request.question,
        readings,
        registry,
        now=request.now,
    )
    return answer.to_dict()


# ══════════════════════════════════════════════════════════
# Supervisor 全链编排
# ══════════════════════════════════════════════════════════
@router.post("/api/ops/run")
async def ops_run_endpoint(request: OpsRunRequest):
    """任务拆分 → 依次执行 → 跨 Agent 冲突检测。

    报告产出后立即落盘并返回 ``draft_id``，后续 ``/submit`` ``/approve``
    都按这个号操作。**不落盘就无法跨请求审批** —— 审批是跨班次的流程。
    """
    readings = _ops_readings(request.readings)
    period = _ops_period(request.period, readings)
    registry = _ops_registry()

    if request.report_kind not in ("daily", "weekly", "regulatory"):
        raise HTTPException(
            status_code=400,
            detail=f"report_kind={request.report_kind!r} 必须是 daily / weekly / regulatory",
        )
    if request.report_kind == "regulatory" and not request.permit_id.strip():
        raise HTTPException(
            status_code=400,
            detail=(
                "监管材料必须提供 permit_id：没有许可证号的监管材料无法归档，"
                "也核不清对应哪个场地"
            ),
        )

    ops_request = ops_supervisor.OpsRequest(
        readings=readings,
        period=period,
        question=request.question,
        want_alarms=request.want_alarms,
        want_condition=request.want_condition,
        want_report=request.want_report,
        report_kind=request.report_kind,
        permit_id=request.permit_id,
        periods=[(p.start, p.end) for p in request.periods],
        now=request.now,
    )

    result = _ops_call(ops_supervisor.run, ops_request, registry)

    payload = result.to_dict()
    payload["brief"] = ops_supervisor.render_brief(result)
    payload["draft_id"] = ""

    if result.report is not None:
        draft_id = ops_store.new_draft_id(result.report.kind)
        await asyncio.to_thread(ops_store.save, draft_id, result.report)
        payload["draft_id"] = draft_id

    return payload


# ══════════════════════════════════════════════════════════
# 报告草稿与人工审批
# ══════════════════════════════════════════════════════════
@router.get("/api/ops/reports")
async def ops_reports_list(limit: int = 20):
    """最近的报告草稿摘要（不含正文，正文可能几十 KB）。"""
    rows = await asyncio.to_thread(ops_store.list_recent, limit)
    return {"reports": rows, "count": len(rows)}


@router.get("/api/ops/reports/{draft_id}")
async def ops_report_detail(draft_id: str):
    """读取一份草稿的完整内容与审批轨迹。"""
    draft = await asyncio.to_thread(_load_draft, draft_id)
    return draft.to_dict()


@router.post("/api/ops/reports/{draft_id}/submit")
async def ops_report_submit(draft_id: str, request: DraftSubmitRequest):
    """提交审批。

    引用检查不通过时返回 **409**，``detail`` 里同时给可读消息与结构化的
    ``problems`` 列表 —— 前端要能逐条显示"第几行的哪个数字没有出处"，
    而不是只弹一句"提交失败"。
    """
    draft = await asyncio.to_thread(_load_draft, draft_id)
    try:
        ops_reports.submit(draft, actor=request.actor or "起草方")
    except ValueError as exc:
        raise HTTPException(
            status_code=409,
            detail={
                "message": str(exc),
                "problems": [p.render() for p in draft.problems],
                "status": draft.status,
            },
        )
    await asyncio.to_thread(ops_store.save, draft_id, draft)
    log.info(f"[OPS] 报告 {draft_id} 已提交审批（{request.actor or '起草方'}）")
    return draft.to_dict()


@router.post("/api/ops/reports/{draft_id}/approve")
async def ops_report_approve(draft_id: str, request: DraftReviewRequest):
    """批准。**必须署名** —— 无署名的批准无法追溯，等于没人负责。"""
    draft = await asyncio.to_thread(_load_draft, draft_id)
    if not request.reviewer.strip():
        raise HTTPException(status_code=400, detail="批准必须填写审批人")
    try:
        ops_reports.approve(draft, reviewer=request.reviewer, note=request.note)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    await asyncio.to_thread(ops_store.save, draft_id, draft)
    log.info(f"[OPS] 报告 {draft_id} 已批准（{request.reviewer}）")
    return draft.to_dict()


@router.post("/api/ops/reports/{draft_id}/reject")
async def ops_report_reject(draft_id: str, request: DraftReviewRequest):
    """驳回。**必须写意见** —— 没有意见的驳回会让下一稿原样再来一遍。"""
    draft = await asyncio.to_thread(_load_draft, draft_id)
    if not request.reviewer.strip():
        raise HTTPException(status_code=400, detail="驳回必须填写审批人")
    if not request.note.strip():
        raise HTTPException(
            status_code=400,
            detail="驳回必须写意见：起草方需要知道要改什么，否则下一稿会原样再来",
        )
    try:
        ops_reports.reject(draft, reviewer=request.reviewer, note=request.note)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    await asyncio.to_thread(ops_store.save, draft_id, draft)
    log.info(f"[OPS] 报告 {draft_id} 已驳回（{request.reviewer}）")
    return draft.to_dict()


@router.post("/api/ops/reports/{draft_id}/reopen")
async def ops_report_reopen(draft_id: str, request: DraftSubmitRequest):
    """驳回后重新打开为草稿，保留完整审批轨迹。"""
    draft = await asyncio.to_thread(_load_draft, draft_id)
    try:
        ops_reports.reopen(draft, actor=request.actor or "起草方")
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    await asyncio.to_thread(ops_store.save, draft_id, draft)
    return draft.to_dict()
