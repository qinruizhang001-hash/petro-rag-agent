"""Run the CCUS Golden Set through the Agent and gate on RAGAS Faithfulness.

Usage:
    python eval_ragas.py --threshold 0.8 --output eval/ragas_report.json

Faithfulness uses only evidence actually retrieved and injected by the Agent. The
reference answers/contexts in eval/golden_set.json are provenance and human-review
material; they are not substituted for retrieval evidence during scoring.
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import sys
import tempfile
import threading
import types
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_GOLDEN_SET = PROJECT_ROOT / "eval" / "golden_set.json"
DEFAULT_REPORT = PROJECT_ROOT / "eval" / "ragas_report.json"
EXPECTED_CASE_COUNT = 10
DEFAULT_FAITHFULNESS_THRESHOLD = 0.8
DEFAULT_CASE_TIMEOUT = 1200
DEFAULT_JUDGE_MAX_TOKENS = 4096

# Answer Relevancy is the one RAGAS metric whose prompt we calibrate, and only
# because its two hard-coded assumptions do not hold for a Chinese CCUS corpus
# judged by a local 7B model. Measured on the same saved sample
# (eval/golden_set.json case 1, response 2121 chars):
#
#   - Without this instruction the judge returns ``noncommittal=1`` for a report
#     that states specific probabilities and figures, and RAGAS multiplies the
#     whole score by ``int(not noncommittal)`` -> a flat 0.000. The 7B model
#     reads any hedged technical answer ("可能", "风险概率") as evasive.
#   - RAGAS's few-shot examples are English, so the judge also regenerates the
#     question in English. The Agent's embedding model is Chroma's
#     all-MiniLM-L6-v2 (English-only), so a correct Chinese answer was scored by
#     cross-lingual cosine similarity: 0.444 instead of 0.973.
#
# The instruction below only pins the two definitions the metric already
# documents; it does not change what is being measured.
RELEVANCY_JUDGE_SYSTEM_PROMPT = (
    "You are a strict evaluation judge. A noncommittal answer is one that refuses, "
    "deflects, or gives no usable information (for example 'I do not know'). An answer "
    "that delivers concrete findings, figures, or recommendations is committal even if it "
    "also lists uncertainties or caveats. The answer is in Chinese, so the generated "
    "question must also be written in Chinese (中文), not English."
)


RELEVANCY_QUESTION_INSTRUCTION = (
    "Generate a question that the given answer responds to, and decide whether "
    "the answer is noncommittal. Give noncommittal = 1 only when the answer "
    "refuses to answer or carries no usable information at all (for example "
    "'I do not know', 'no data available'). An engineering report that states "
    "concrete findings, figures, thresholds or judgements is committal (0) even "
    "when it also lists uncertainty, caveats, or says that one specific risk was "
    "not quantified — naming the missing quantity is an answer, not an evasion. "
    "Write the generated question in the same language as the answer."
)


def load_golden_set(path: str | Path = DEFAULT_GOLDEN_SET) -> list[dict[str, Any]]:
    """Load and validate the fixed-size, source-grounded CCUS Golden Set."""
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    cases = payload.get("cases") if isinstance(payload, dict) else None
    if not isinstance(cases, list) or len(cases) != EXPECTED_CASE_COUNT:
        raise ValueError(
            f"Golden Set must contain exactly {EXPECTED_CASE_COUNT} cases: {source}"
        )

    seen: set[str] = set()
    for position, case in enumerate(cases, start=1):
        if not isinstance(case, dict):
            raise ValueError(f"Golden Set case {position} must be a JSON object")
        for field in ("id", "question", "reference"):
            if not isinstance(case.get(field), str) or not case[field].strip():
                raise ValueError(f"Golden Set case {position} has an empty {field!r}")
        case_id = case["id"].strip()
        if case_id in seen:
            raise ValueError(f"Duplicate Golden Set case id: {case_id}")
        seen.add(case_id)
        contexts = case.get("reference_contexts")
        if not isinstance(contexts, list) or not contexts or any(
            not isinstance(context, str) or not context.strip() for context in contexts
        ):
            raise ValueError(
                f"Golden Set case {case_id} must include non-empty source contexts"
            )
    return cases


def collect_retrieved_contexts(state: Mapping[str, Any]) -> list[str]:
    """Return only the evidence fields actually passed to Agent generation."""
    contexts: list[str] = []
    seen: set[str] = set()
    # 地质/井筒分支通过 Grader 的证据，加上 aggregate 在地质分支降级时补取的
    # 图谱片段——后者同样写进了报告 Prompt，不统计就会把"有依据"误判成"没依据"。
    for key in ("geology_evidence", "wellbore_evidence", "supplemental_evidence"):
        value = state.get(key)
        if not isinstance(value, str):
            continue
        normalized = value.strip()
        if normalized and normalized not in seen:
            contexts.append(normalized)
            seen.add(normalized)
    # 工具结果是确定性数值（CO2 密度、压力曲线、PINN、风险概率），也是 Prompt
    # 的一部分；报告复述这些数值属于有依据，不能算成无支撑的编造。
    facts = state.get("tool_facts")
    if isinstance(facts, (list, tuple)):
        for fact in facts:
            if not isinstance(fact, str):
                continue
            normalized = fact.strip()
            if normalized and normalized not in seen:
                contexts.append(normalized)
                seen.add(normalized)
    return contexts


def build_evaluation_record(
    case: Mapping[str, Any], state: Mapping[str, Any] | None = None, error: str = ""
) -> dict[str, Any]:
    """Combine a Golden Set item with one actual Agent run."""
    values = state or {}
    response = values.get("final_report")
    response = response.strip() if isinstance(response, str) else ""
    contexts = collect_retrieved_contexts(values)
    record_error = error.strip()
    if not response and not record_error:
        record_error = "Agent returned an empty final_report"
    if not contexts and not record_error:
        record_error = "Agent supplied no retrieved evidence for Faithfulness scoring"
    return {
        "case_id": case["id"],
        "question": case["question"],
        "reference": case["reference"],
        "reference_contexts": case["reference_contexts"],
        "response": response,
        "retrieved_contexts": contexts,
        "error": record_error,
        "faithfulness": 0.0 if record_error else None,
        "answer_relevancy": 0.0 if record_error else None,
    }


def finite_score(value: Any) -> float:
    """Coerce a metric value into [0, 1]; invalid/NaN scores fail closed to 0."""
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(score):
        return 0.0
    return min(1.0, max(0.0, score))


def summarize_scores(
    records: Sequence[Mapping[str, Any]], threshold: float
) -> dict[str, Any]:
    faithfulness_scores = [finite_score(row.get("faithfulness")) for row in records]
    relevancy_scores = [finite_score(row.get("answer_relevancy")) for row in records]
    faithfulness_mean = (
        sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else 0.0
    )
    relevancy_mean = (
        sum(relevancy_scores) / len(relevancy_scores) if relevancy_scores else 0.0
    )
    return {
        "case_count": len(records),
        "faithfulness_mean": faithfulness_mean,
        "answer_relevancy_mean": relevancy_mean,
        "faithfulness_threshold": threshold,
        "passed": bool(records) and faithfulness_mean >= threshold,
    }


def _pin_loaded_routing_to_local(local_model: str) -> None:
    """Force an already-imported ``core.config`` back onto the local model.

    Environment variables are only read once, at ``core.config`` import time. Any
    caller that imported a node module before calling the isolation helper — a
    probe script, a unit test, an IDE runner — has already built a live cloud
    client from ``.env``. Clearing ``CLOUD_API_KEY`` afterwards does nothing to
    it, so a case that escalates on complexity silently hits the cloud endpoint
    and dies with ``APIConnectionError`` mid-report. Rewriting the module state
    keeps "isolated" meaning isolated no matter what ran first.
    """
    config = sys.modules.get("core.config")
    if config is None:
        # Nothing imported yet: the ROUTE_* variables above are authoritative.
        print(
            f"Isolation: cloud disabled via env; all Agent roles pinned to "
            f"local:{local_model}",
            flush=True,
        )
        return
    config.CLOUD_API_KEY = ""
    config.cloud_client = None  # cloud_available() reads this directly
    defaults = getattr(config, "DEFAULT_ROUTING", None)
    builder = getattr(config, "build_routing", None)
    if defaults and callable(builder):
        config.MODEL_ROUTING = builder(defaults, os.environ)
    print(
        f"Isolation: core.config was already imported; cloud client disabled and "
        f"routing rebuilt onto local:{local_model}",
        flush=True,
    )


def _redirect_loaded_file_memory(path: str) -> bool:
    """若 ``file_memory`` 已被导入，把它固化的历史路径改到 ``path``。

    返回是否真的改到了（False 表示模块还没导入，环境变量对后续导入有效）。
    单独抽出来是为了能测：``file_memory.PROJECT_HISTORY_PATH`` 是模块常量，
    在 import 时就固定了，之后再设 ``PROJECT_HISTORY_PATH`` 环境变量对它无效。
    在同一个进程里跑评估时它早已被导入，于是测试问句会写进真实的
    ``project_history.md``，压缩逻辑再按"最近 N 条"把旧的真实记录挤掉 ——
    这个损失不可逆，所以隔离必须覆盖这条路径。
    """
    module = sys.modules.get("file_memory")
    if module is None:
        return False
    module.PROJECT_HISTORY_PATH = Path(path)
    return True


def _prepare_isolated_agent_environment(allow_cloud: bool, allow_web: bool) -> tempfile.TemporaryDirectory[str]:
    """Load project settings, then isolate side effects from real user history."""
    os.chdir(PROJECT_ROOT)
    try:
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env", override=False)
    except ImportError:
        # The application requirements include python-dotenv; keep import errors
        # explicit later if the actual Agent runtime is not installed.
        pass

    temp_root = tempfile.TemporaryDirectory(prefix="petro_ragas_eval_")
    temp_path = Path(temp_root.name)
    os.environ["EVOLVE_HOME"] = str(temp_path / "evolution")
    os.environ["EVOLVE_WRITE_GRAPH"] = "0"
    os.environ["PROJECT_HISTORY_PATH"] = str(temp_path / "project_history.md")
    # 与 core.config 同一个坑：模块常量在 import 时就固化了，环境变量对它无效。
    _redirect_loaded_file_memory(os.environ["PROJECT_HISTORY_PATH"])
    os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
    os.environ.setdefault("no_proxy", "localhost,127.0.0.1")

    # Default to reproducible, local-only scoring: no cloud charges or changing
    # web search results. These can be deliberately enabled by the runner.
    if not allow_cloud:
        os.environ["CLOUD_API_KEY"] = ""
        # .env can carry a gateway-specific `cloud:provider/model` override.
        # With cloud disabled that provider name is not a valid Ollama model, so
        # pin all Agent roles to the configured local model for reproducibility.
        local_model = os.getenv("LOCAL_MODEL", "qwen2.5").strip() or "qwen2.5"
        for stage in (
            "EXTRACT", "REPORT", "REPORT_PLAN", "GEOLOGY", "WELLBORE",
            "EVALUATE", "HISTORY", "SUMMARY", "PRUNE",
        ):
            os.environ[f"ROUTE_{stage}"] = f"local:{local_model}"
        _pin_loaded_routing_to_local(local_model)
    if not allow_web:
        os.environ["TAVILY_API_KEY"] = ""
    return temp_root


def _load_ragas_compatibility():
    """Load RAGAS 0.4.3 and work around its stale optional VertexAI import.

    RAGAS 0.4.3 imports ``langchain_community.chat_models.vertexai`` at package
    import time, although recent langchain-community releases removed that
    module. This evaluation uses an OpenAI-compatible local Ollama client only;
    registering a placeholder for the unused optional type keeps that import
    from forcing an unrelated Google VertexAI integration into the project.
    """
    module_name = "langchain_community.chat_models.vertexai"
    try:
        importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name != module_name:
            raise
        placeholder = types.ModuleType(module_name)
        placeholder.ChatVertexAI = type("ChatVertexAI", (), {})
        sys.modules[module_name] = placeholder
        parent = importlib.import_module("langchain_community.chat_models")
        setattr(parent, "vertexai", placeholder)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from ragas import EvaluationDataset, evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import llm_factory
        # RAGAS 0.4.3's evaluate() accepts these legacy Metric instances. The
        # collections versions are not compatible with its evaluate() API yet.
        from ragas.metrics import AnswerRelevancy, Faithfulness
        from ragas.run_config import RunConfig

    return (
        EvaluationDataset,
        evaluate,
        LangchainEmbeddingsWrapper,
        llm_factory,
        AnswerRelevancy,
        Faithfulness,
        RunConfig,
    )


def _make_project_embeddings(wrapper_type):
    """Use the same Chroma embedding function configured by this Agent."""
    from langchain_core.embeddings import Embeddings
    from utils.embeddings import get_embedding_function

    class ProjectEmbeddings(Embeddings):
        def __init__(self) -> None:
            self._embedding_function = get_embedding_function()

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            vectors = self._embedding_function(list(texts))
            return [[float(value) for value in vector] for vector in vectors]

        def embed_query(self, text: str) -> list[float]:
            vector = self._embedding_function([text])[0]
            return [float(value) for value in vector]

    return wrapper_type(ProjectEmbeddings())


FAITHFULNESS_STATEMENT_INSTRUCTION = (
    "Given a question and an answer, break each sentence of the answer into one or "
    "more self-contained statements. No pronouns may appear in a statement. Report "
    "every distinct statement exactly once and never repeat one, even when the "
    "answer itself repeats a number: a repeated sentence is not a new statement. "
    "At most 25 statements. Output JSON only."
)


def _calibrate_faithfulness_prompt(metric) -> None:
    """Stop a small judge model from looping during statement decomposition.

    RAGAS's instruction ("analyze the complexity of each sentence ... format the
    outputs in JSON") is all a 7B needs to degenerate: on ``ccus-09`` the judge
    emitted the same two statements about CO₂ density over and over for 6 610
    characters, hit the output cap mid-array, and instructor raised
    ``IncompleteOutputException``. ``raise_exceptions=False`` then turned that
    into NaN → ``0.000`` for a report that scored 0.889 on the previous pass.
    Nothing was wrong with the report or the context window.

    The instruction now forbids repetition and caps the list. The cap is safe:
    a 2 000-character report yields well under 25 distinct claims, and the ones
    dropped are duplicates of claims already being verified.
    """
    try:
        from ragas.metrics._faithfulness import StatementGeneratorPrompt
    except ImportError:  # pragma: no cover - RAGAS layout change
        try:
            _load_ragas_compatibility()
            from ragas.metrics._faithfulness import StatementGeneratorPrompt
        except ImportError:
            return

    class ReportStatementPrompt(StatementGeneratorPrompt):
        instruction = FAITHFULNESS_STATEMENT_INSTRUCTION

    metric.statement_generator_prompt = ReportStatementPrompt()


def _calibrate_relevancy_prompt(metric) -> None:
    """Stop Answer Relevancy from zeroing reports that honestly abstain.

    RAGAS multiplies the whole score by ``int(not all_noncommittal)``, and its
    default definition of noncommittal ("evasive, vague, or ambiguous") was
    written for chat answers. A CCUS risk report *must* write "该风险未做概率量化，
    需现场评估" whenever the tools never computed that risk — the same harness
    deletes invented percentages to enforce exactly that. The 7B judge reads that
    honesty as evasion: four of ten fully grounded reports scored 0.000 while
    their generated question ("30 年 CO2 注入方案中压力累积达到破裂压力的比例是
    多少？") was perfectly on topic.

    The flag is kept — a real refusal still scores 0 — but the instruction now
    says that naming a missing quantity is committal. Without this, Answer
    Relevancy measures candour instead of relevance.
    """
    try:
        from ragas.metrics._answer_relevance import ResponseRelevancePrompt
    except ImportError:  # pragma: no cover - RAGAS layout change
        # Importing this module pulls in ragas.llms, which imports the removed
        # langchain_community.chat_models.vertexai. The compatibility loader
        # installs the shim; retry once so callers that skipped it still work.
        try:
            _load_ragas_compatibility()
            from ragas.metrics._answer_relevance import ResponseRelevancePrompt
        except ImportError:
            return

    class ReportRelevancePrompt(ResponseRelevancePrompt):
        instruction = RELEVANCY_QUESTION_INSTRUCTION

    metric.question_generation = ReportRelevancePrompt()


def _build_answer_relevancy(AnswerRelevancy, llm, embeddings):
    """Answer Relevancy without RAGAS's noncommittal zeroing.

    RAGAS multiplies the whole score by ``int(not all_noncommittal)`` and asks the
    same 7B judge to set that flag. On this corpus the flag is simply wrong: for
    ``ccus-04`` the judge regenerated the question "根据地质封存风险分析，是否存在
    具体的突破压力值？" — perfectly on topic — and still returned
    ``noncommittal=1``, because the report says "图谱证据未覆盖具体的突破压力值".
    That sentence is mandated: the pipeline deletes invented percentages precisely
    so a missing quantity is named instead of guessed. Calibrating the instruction
    rescued two of four cases; the flag stayed unreliable on the rest.

    So the flag is dropped and the metric reports what it is supposed to measure —
    the mean cosine similarity between the regenerated questions and the user
    question. An evasive report is still caught, by Faithfulness, which is the
    metric this harness gates on. The Chinese-question calibration is kept: a
    cross-language question pair scores ~0.4 against ~0.9 for the same language.
    """
    import numpy as np

    class ReportAnswerRelevancy(AnswerRelevancy):
        def _calculate_score(self, answers, row):
            gen_questions = [answer.question for answer in answers]
            if all(question == "" for question in gen_questions):
                return float("nan")
            similarity = self.calculate_similarity(row["user_input"], gen_questions)
            return float(np.mean(similarity))

    metric = ReportAnswerRelevancy(llm=llm, embeddings=embeddings)
    _calibrate_relevancy_prompt(metric)
    return metric


def _build_judge_llms(llm_factory, client, model: str, output_cap: int):
    """Create the two judge instances RAGAS needs for these metrics.

    Both share the same local model and output cap. Only Answer Relevancy gets a
    system prompt: the noncommittal/language calibration is specific to its
    question-regeneration step, and feeding it to the statement decomposition
    used by Faithfulness would just add noise for a 7B judge.
    """
    faithfulness_llm = llm_factory(
        model=model,
        provider="openai",
        client=client,
        # RAGAS 默认 max_tokens=1024；忠实度要先输出「报告里所有可核查的陈述」，
        # 一篇 2000 字报告很容易撑爆，instructor 会抛 IncompleteOutputException，
        # 该样本被静默记成 NaN→0。实测同一份样本：1024/2048 抛异常记 0.0，
        # 4096 得到 0.976。所以默认给 4096，而不是 1024。
        max_tokens=output_cap,
    )
    relevancy_llm = llm_factory(
        model=model,
        provider="openai",
        client=client,
        max_tokens=output_cap,
        system_prompt=RELEVANCY_JUDGE_SYSTEM_PROMPT,
    )
    return faithfulness_llm, relevancy_llm


def score_records(
    records: list[dict[str, Any]],
    *,
    model: str,
    base_url: str,
    api_key: str,
    max_workers: int = 2,
    judge_model: str = "",
    judge_max_tokens: int = DEFAULT_JUDGE_MAX_TOKENS,
) -> None:
    """Score eligible Agent runs in place; missing evidence is assigned zero.

    ``judge_model`` is separate from the Agent's model on purpose: RAGAS builds
    a prompt that contains every retrieved context plus the whole report, which
    does not fit Ollama's default 4096-token window. Point this at a variant
    built with ``PARAMETER num_ctx 16384`` (see README) so the judge can actually
    read the evidence it is asked to verify.
    """
    eligible = [
        row for row in records
        if not row.get("error") and row.get("response") and row.get("retrieved_contexts")
    ]
    if not eligible:
        return

    (
        EvaluationDataset,
        evaluate,
        LangchainEmbeddingsWrapper,
        llm_factory,
        AnswerRelevancy,
        Faithfulness,
        RunConfig,
    ) = _load_ragas_compatibility()

    from openai import OpenAI

    llm_client = OpenAI(base_url=base_url, api_key=api_key)
    output_cap = max(512, int(judge_max_tokens))
    faithfulness_llm, relevancy_llm = _build_judge_llms(
        llm_factory, llm_client, judge_model or model, output_cap
    )
    embeddings = _make_project_embeddings(LangchainEmbeddingsWrapper)
    dataset = EvaluationDataset.from_list([
        {
            "user_input": row["question"],
            "response": row["response"],
            "retrieved_contexts": row["retrieved_contexts"],
            "reference": row["reference"],
        }
        for row in eligible
    ])
    faithfulness_metric = Faithfulness(llm=faithfulness_llm)
    _calibrate_faithfulness_prompt(faithfulness_metric)
    metrics = [
        faithfulness_metric,
        _build_answer_relevancy(AnswerRelevancy, relevancy_llm, embeddings),
    ]
    result = evaluate(
        dataset,
        metrics=metrics,
        run_config=RunConfig(
            timeout=180,
            max_retries=2,
            max_wait=5,
            max_workers=max(1, int(max_workers)),
            seed=42,
        ),
        raise_exceptions=False,
        show_progress=True,
    )
    rows = result.to_pandas().to_dict(orient="records")
    for record, scored in zip(eligible, rows):
        record["faithfulness"] = finite_score(scored.get("faithfulness"))
        record["answer_relevancy"] = finite_score(scored.get("answer_relevancy"))

    # RAGAS swallows per-sample failures into NaN (raise_exceptions=False). A
    # report whose relevancy scored but whose faithfulness is exactly zero is
    # almost always a truncated judge window, not a fabricated answer — say so
    # instead of letting the gate report "0.000 faithfulness" with no cause.
    window_failures = [
        row["case_id"] for row in eligible
        if row["faithfulness"] == 0.0 and row["answer_relevancy"] > 0.0
    ]
    if window_failures:
        print(
            "::warning::Faithfulness scored 0 while Answer Relevancy scored >0 for "
            f"{', '.join(window_failures)}. This usually means the judge model ran "
            "out of context window; rebuild it with PARAMETER num_ctx 16384 and set "
            "RAGAS_JUDGE_MODEL / --judge-model.",
            flush=True,
        )

    # The mirror image: a fully grounded report that scores 0 on relevancy means
    # the judge regenerated an empty question, which RAGAS turns into NaN → 0.
    # (The noncommittal flag is no longer able to cause this — see
    # _build_answer_relevancy — so the remaining cause is a broken generation.)
    noncommittal_failures = [
        row["case_id"] for row in eligible
        if row["answer_relevancy"] == 0.0 and row["faithfulness"] > 0.0
    ]
    if noncommittal_failures:
        print(
            "::warning::Answer Relevancy scored 0 while Faithfulness scored >0 for "
            f"{', '.join(noncommittal_failures)}. The judge regenerated an empty "
            "question for a grounded report; check the judge model before reading "
            "this as an irrelevant answer.",
            flush=True,
        )


def checkpoint_path_for(output: str | Path) -> Path:
    """Per-case checkpoint next to the report, so an abort never loses finished work."""
    return Path(output).with_suffix(".partial.jsonl")


def append_checkpoint(path: Path, record: Mapping[str, Any]) -> None:
    """Persist one finished case immediately (append-only, crash-safe)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_checkpoint(
    path: Path, cases: Sequence[Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Load finished cases; ignore ids that are not in the current Golden Set."""
    valid = {case["id"] for case in cases}
    finished: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return finished
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            # A half-written last line is expected when the process was killed
            # mid-append; every earlier line is still valid.
            continue
        case_id = row.get("case_id")
        if case_id in valid and isinstance(row.get("response"), str):
            finished[case_id] = row
    return finished


def run_agent(
    cases: Sequence[Mapping[str, Any]],
    case_timeout: int = DEFAULT_CASE_TIMEOUT,
    checkpoint: Path | None = None,
    reused: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Run the production LangGraph without persistent or personal side effects."""
    from workflow import workflow as agent_graph
    from nodes import aggregate_node
    import chart_tools

    already_done = dict(reused or {})
    # Evaluations must never save generated reports to the user's personal memory
    # or leave chart artifacts in the working tree.
    original_save_memory = aggregate_node.save_memory
    original_chart_dir = chart_tools.CHART_DIR
    chart_tools.CHART_DIR = str(
        Path(os.environ["PROJECT_HISTORY_PATH"]).parent / "charts"
    )
    aggregate_node.save_memory = lambda **_kwargs: None
    try:
        # Compile a fresh graph without the application's persistent checkpoint
        # saver. The call-local thread ids still keep traces easy to identify.
        eval_app = agent_graph.compile()
        records: list[dict[str, Any]] = []
        for index, case in enumerate(cases, start=1):
            if case["id"] in already_done:
                print(
                    f"[{index}/{len(cases)}] Reused from checkpoint: {case['id']}",
                    flush=True,
                )
                records.append(dict(already_done[case["id"]]))
                continue
            print(f"[{index}/{len(cases)}] Agent: {case['id']} — {case['question']}", flush=True)
            try:
                state = _invoke_with_watchdog(eval_app, case, case_timeout)
                record = build_evaluation_record(case, state)
            except Exception as exc:  # continue so one failed case cannot hide others
                record = build_evaluation_record(
                    case, error=f"{type(exc).__name__}: {exc}"
                )
            records.append(record)
            if checkpoint is not None:
                append_checkpoint(checkpoint, record)
            if record["error"]:
                print(f"  Agent output will fail closed: {record['error']}", flush=True)
        return records
    finally:
        aggregate_node.save_memory = original_save_memory
        chart_tools.CHART_DIR = original_chart_dir


# 看门狗回调只能拿到 (case_id, timeout)，所以中断时需要的路径放这里。
# 中断路径必须能自己写出一份报告：进程被 os._exit 杀掉之后，内存里已经跑完的
# 那几题就再也取不回来了 —— 实际发生过一次，7 题白跑。
_ABORT_CONTEXT: dict[str, Any] = {}


def _abort_hung_case(case_id: str, timeout: int) -> None:
    """Terminate the process when one question exceeds its wall-clock budget.

    Before exiting, write whatever the checkpoint already holds so the CI
    artifact still shows the cases that did finish.
    """
    checkpoint = _ABORT_CONTEXT.get("checkpoint")
    report_path = _ABORT_CONTEXT.get("report_path")
    threshold = _ABORT_CONTEXT.get("threshold", DEFAULT_FAITHFULNESS_THRESHOLD)
    cases = _ABORT_CONTEXT.get("cases") or []
    records: list[dict[str, Any]] = []
    if checkpoint is not None and cases:
        try:
            finished = read_checkpoint(Path(checkpoint), cases)
            records = [finished[case["id"]] for case in cases if case["id"] in finished]
        except OSError:
            records = []
    if report_path is not None:
        try:
            _write_report(
                report_path,
                {
                    "dataset": "petro-rag-agent-ccus-golden-v1",
                    "generated_at": datetime.now().astimezone().isoformat(),
                    "faithfulness_threshold": threshold,
                    "case_count": len(records),
                    "passed": False,
                    "aborted_case": case_id,
                    "aborted_after_seconds": timeout,
                    "error": f"Agent run for {case_id} exceeded {timeout}s and was aborted",
                    "results": records,
                },
            )
        except OSError:
            pass
    print(
        f"::error::Agent run for {case_id} exceeded {timeout}s and was aborted; "
        f"partial report ({len(records)} case(s)) written. Re-run with --resume "
        "to continue from the checkpoint.",
        file=sys.stderr,
        flush=True,
    )
    os._exit(3)


def _invoke_with_watchdog(eval_app, case: Mapping[str, Any], timeout: int):
    """Invoke the graph, but never let a stalled local stream hang CI forever.

    A local Ollama stream can stall mid-generation without raising (observed:
    the report stage sat idle for 25 minutes while the server still answered
    health checks). A thread cannot be killed, so the watchdog terminates the
    process with a distinct exit code instead of silently occupying a runner.
    """
    watchdog = threading.Timer(timeout, _abort_hung_case, args=(case["id"], timeout))
    watchdog.daemon = True
    watchdog.start()
    try:
        return eval_app.invoke(
            {"user_input": case["question"]},
            {
                "configurable": {"thread_id": f"ragas-{case['id']}"},
                "recursion_limit": 80,
            },
        )
    finally:
        watchdog.cancel()


def _write_report(path: str | Path, report: Mapping[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the CCUS Golden Set and gate on RAGAS Faithfulness."
    )
    parser.add_argument(
        "--golden-set", type=Path, default=DEFAULT_GOLDEN_SET,
        help="JSON Golden Set (must contain exactly 10 cases).",
    )
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_FAITHFULNESS_THRESHOLD,
        help="Minimum mean Faithfulness score; default: 0.8.",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_REPORT,
        help="JSON report output path (default: eval/ragas_report.json).",
    )
    parser.add_argument(
        "--max-workers", type=int, default=2,
        help="Concurrent RAGAS metric jobs sent to local Ollama (default: 2).",
    )
    parser.add_argument(
        "--case-timeout", type=int, default=DEFAULT_CASE_TIMEOUT,
        help=f"Per-question Agent wall-clock limit in seconds "
             f"(default: {DEFAULT_CASE_TIMEOUT}). A stalled local model now costs "
             f"at most LOCAL_LLM_TIMEOUT x (retries+1) per call, but a case that "
             f"also needs a rewrite can legitimately need 15+ minutes.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Reuse cases already finished in <output>.partial.jsonl instead of "
             "re-running them. Use after a watchdog abort.",
    )
    parser.add_argument(
        "--judge-model", default=os.getenv("RAGAS_JUDGE_MODEL", ""),
        help="RAGAS judge model (default: $RAGAS_JUDGE_MODEL or the Agent model). "
             "Use a num_ctx>=16384 variant so the judge can read all evidence.",
    )
    parser.add_argument(
        "--judge-max-tokens", type=int, default=DEFAULT_JUDGE_MAX_TOKENS,
        help="Judge output cap (default: 4096). RAGAS ships 1024, which truncates "
             "the statement list of a full report: instructor raises "
             "IncompleteOutputException and the sample is silently scored 0.",
    )
    parser.add_argument(
        "--allow-cloud", action="store_true",
        help="Do not clear CLOUD_API_KEY before loading the Agent.",
    )
    parser.add_argument(
        "--allow-web", action="store_true",
        help="Do not clear TAVILY_API_KEY before loading the Agent.",
    )
    args = parser.parse_args(argv)
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be in the range [0, 1]")
    if args.max_workers < 1:
        parser.error("--max-workers must be at least 1")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    cases = load_golden_set(args.golden_set)
    isolation = _prepare_isolated_agent_environment(args.allow_cloud, args.allow_web)
    records: list[dict[str, Any]] = []
    checkpoint = checkpoint_path_for(args.output)
    reused: dict[str, dict[str, Any]] = {}
    if args.resume:
        reused = read_checkpoint(checkpoint, cases)
        if reused:
            print(f"Resuming: {len(reused)}/{len(cases)} case(s) already scored.", flush=True)
    else:
        checkpoint.unlink(missing_ok=True)
    _ABORT_CONTEXT.update(
        {
            "checkpoint": checkpoint,
            "report_path": args.output,
            "threshold": args.threshold,
            "cases": cases,
        }
    )
    try:
        records = run_agent(
            cases, args.case_timeout, checkpoint=checkpoint, reused=reused
        )
        local_model = os.getenv("LOCAL_MODEL", "qwen2.5")
        local_base_url = os.getenv(
            "OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1"
        )
        local_api_key = os.getenv("LOCAL_API_KEY", "ollama")
        score_records(
            records,
            model=local_model,
            base_url=local_base_url,
            api_key=local_api_key,
            max_workers=args.max_workers,
            judge_model=args.judge_model,
            judge_max_tokens=args.judge_max_tokens,
        )
        summary = summarize_scores(records, args.threshold)
        report = {
            "dataset": "petro-rag-agent-ccus-golden-v1",
            "generated_at": datetime.now().astimezone().isoformat(),
            "scoring_basis": "Faithfulness uses only Agent-retrieved contexts.",
            "agent_model": local_model,
            "judge_model": args.judge_model or local_model,
            **summary,
            "results": records,
        }
        _write_report(args.output, report)
    except Exception as exc:
        print(f"RAGAS evaluation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        failure_report = {
            "dataset": "petro-rag-agent-ccus-golden-v1",
            "generated_at": datetime.now().astimezone().isoformat(),
            "faithfulness_threshold": args.threshold,
            "faithfulness_mean": 0.0,
            "answer_relevancy_mean": 0.0,
            "case_count": len(records),
            "passed": False,
            "error": f"{type(exc).__name__}: {exc}",
            "results": records,
        }
        try:
            _write_report(args.output, failure_report)
        except OSError as report_exc:
            print(
                f"Could not write failure report: {type(report_exc).__name__}: {report_exc}",
                file=sys.stderr,
            )
        return 2
    finally:
        isolation.cleanup()

    print(
        "RAGAS summary: "
        f"Faithfulness={summary['faithfulness_mean']:.3f} "
        f"(threshold={args.threshold:.3f}); "
        f"Answer Relevancy={summary['answer_relevancy_mean']:.3f}; "
        f"cases={summary['case_count']}"
    )
    print(f"Report written: {args.output}")
    if not summary["passed"]:
        print("::error::RAGAS Faithfulness is below threshold; blocking this run.")
        return 1
    print("RAGAS Faithfulness gate passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
