"""Flask API для многодиалогового DeepSeek Agent."""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import uuid
from copy import deepcopy
from contextvars import ContextVar
from io import BytesIO
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from flask import Flask, g, jsonify, render_template, request, send_file

from artifacts import (
    ArtifactError,
    active_artifacts,
    artifact_file_path,
    copy_artifact_files,
    delete_artifact_files,
    ensure_artifacts,
    missing_required_artifacts,
    public_artifact,
    required_artifacts_are_validated,
    save_response_artifacts,
    verify_required_artifacts,
)

from agent import (
    DEEPSEEK_MODELS,
    PROVIDER_CAPABILITIES,
    Agent,
    AgentResult,
    AgentSettings,
    DeepSeekProvider,
    dialogue_token_totals,
    normalize_token_usage,
)
from context_manager import (
    active_path_messages,
    add_fact,
    append_tree_message,
    begin_branch,
    branch_points,
    completed_exchanges,
    current_stage_exchanges,
    current_stage_facts,
    delete_fact,
    edit_fact,
    ensure_context_management,
    facts_token_totals,
    request_history,
    select_branch,
    set_context_mode,
    set_window_sizes,
    summary_token_totals,
    update_sticky_facts,
    update_summaries,
)
from logging_setup import reset_request_id, set_request_id
from invariants import InvariantStore
from memory_manager import MemoryManager
from document_index import (
    EvidenceValidationError,
    RAG_PIPELINE_MODES,
    RAG_STRATEGIES,
    RAG_UNSUPPORTED_RESPONSE,
    RagIndexService,
    apply_relevance_gate,
    build_evidence_guidance,
    build_rag_context,
    parse_semantic_evaluation,
    repair_prompt,
    retry_unknown_prompt,
    semantic_evaluation_prompt,
    validate_and_format_evidence,
)
from document_index.evaluations import RagEvaluationStore
from mcp_control import PersistentFeatureControl
from mcp_manager import (
    MCPManager,
    MediaWikiMCPManager,
    OpenMeteoMCPManager,
    SchedulerMCPManager,
    StdioMCPManager,
    WorldBankMCPManager,
)
from presets import PresetManager
from scheduler import SchedulerRepository, SchedulerService, TIMEZONE_NAME, scheduler_now
from storage import JsonStorage, now_iso
from task_state import (
    STAGE_DEFAULTS,
    TaskTransitionError,
    allowed_task_events,
    apply_task_event,
    ensure_task_state,
    task_is_paused,
    task_state_report,
    update_task_state,
)
from task_policy import (
    PolicyValidationError,
    blocked_content,
    generation_guidance,
    parse_validation_result,
    policy_audit,
    validation_prompt,
)
from task_handoff import (
    activate_stage_handoff,
    active_task_handoffs,
    build_stage_handoff,
    ensure_task_handoffs,
    handoff_token_totals,
    task_handoff_context,
)
from task_memory import (
    apply_task_memory_result,
    ensure_task_memory,
    task_memory_context,
    update_task_memory_settings,
)
from voice import WhisperService


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
MAX_MESSAGE_LENGTH = 50_000
logger = logging.getLogger("deepseek_agent.app")
_profile_override: ContextVar[str | None] = ContextVar("scheduler_profile_override", default=None)
_automation_run_override: ContextVar[dict[str, Any] | None] = ContextVar("automation_run_override", default=None)
_scheduled_tool_allowlist: ContextVar[set[str] | None] = ContextVar("scheduled_tool_allowlist", default=None)
SCHEDULER_MODEL_TOOLS = {
    "create_scheduled_task", "list_scheduled_tasks", "get_scheduled_task",
    "get_scheduled_task_runs", "get_scheduler_summary", "get_scheduler_capabilities",
}
FILESYSTEM_MODEL_TOOLS = {"write_file"}
MEDIAWIKI_MODEL_TOOLS = {
    "search-page", "search-page-by-prefix", "get-page", "get-pages",
    "get-category-members", "get-site-info",
}
WORLDBANK_MODEL_TOOLS = {
    "worldbank_list_topics", "worldbank_list_sources", "worldbank_list_countries",
    "worldbank_get_country", "worldbank_search_indicators", "worldbank_get_indicator",
    "worldbank_get_data", "worldbank_get_poverty", "worldbank_search_projects",
}
_FILE_EXPORT_NEGATIONS = (
    "не сохраняй", "не сохранять", "без сохранения", "не записывай",
    "не записывать", "не создавай файл", "не создавай отчёт", "не создавай отчет",
)
_FILE_EXPORT_PATTERNS = (
    r"\bсохран\w*\b", r"\bвыгруз\w*\b", r"\bэкспорт\w*\b",
    r"\bзапиш\w*\b.{0,30}\b(?:файл|документ|отч[её]т)\b",
    r"\bсозда\w*\b.{0,30}\b(?:файл|документ|отч[её]т)\b",
    r"\b(?:файл|документ|отч[её]т)\b.{0,20}\b(?:на диск|в файл|в документ)\b",
    r"\b(?:save|write|export)\b.{0,30}\b(?:file|document|report)\b",
)
_OUTSIDE_RAG_PATTERNS = (
    r"\bответь\w*\s+(?:прямо\s+)?вне\s+(?:локальной\s+)?базы",
    r"\bответь\w*\s+без\s+(?:rag|рага|базы\s+знаний)",
    r"\bиспользуй\w*\s+(?:свои|собственные)\s+знания",
    r"\bиспользуй\w*\s+(?:доступные|разреш[её]нные)\s+инструменты",
)
_OUTSIDE_RAG_NEGATIONS = ("не используй свои знания", "не отвечай вне базы", "не используй инструменты")
OUTSIDE_RAG_PREFIX = "Ответ вне локальной базы знаний."
OUTSIDE_RAG_SUFFIX = "Источники локальной базы: не использовались."
RAG_MISS_NOTICE = "В локальной базе знаний релевантная информация не найдена."
RAG_MISS_ANSWER_LABEL = "Ответ модели вне локальной базы знаний:"
AUTO_RAG_ROUTES = {"rag", "task", "tools", "hybrid", "general"}
_AUTO_TASK_PATTERNS = (
    r"\b(?:что|чем)\s+(?:мы\s+)?(?:сейчас|теперь)\s+(?:делаем|занимаемся)\b",
    r"\b(?:текущ(?:ий|ая)|следующ(?:ий|ая))\s+(?:этап|шаг|действие|задач)\b",
    r"\b(?:цель|статус|состояние)\s+(?:текущей\s+)?задачи\b",
    r"\b(?:план|этап)\s+(?:уже\s+)?(?:утвержден|утверждён|завершен|завершён)\b",
)
_AUTO_TOOL_PATTERNS = (
    r"\bapi\b", r"\bmcp\b", r"\b(?:доступн\w*|разрешенн\w*|разрешённ\w*|внешн\w*)\s+инструмент\w*\b",
    r"\b(?:получи|загрузи|найди|проверь)\w*\b.{0,35}\b(?:через|из)\s+(?:api|интернет|внешн)",
    r"\b(?:погод\w*|world\s*bank|всемирн\w*\s+банк\w*|wikipedia|википеди\w*)\b",
    r"\b(?:сохрани|запиши|экспортируй)\w*\b.{0,30}\b(?:файл|отчет|отчёт|документ)\b",
)
_AUTO_RAG_PATTERNS = (
    r"\b(?:в|по|из)\s+(?:документ\w*|стать\w*|пособи\w*|pdf|баз[еуы]\s+знаний)\b",
    r"\b(?:локальн\w*\s+баз\w*|чанк\w*|цитат\w*|chunk_id|источник\w*)\b",
    r"\b(?:согласно|упомянут\w*|написан\w*|сказан\w*)\b.{0,35}\b(?:документ\w*|стать\w*|пособи\w*)",
    r"\bстать\w*\s+(?:относит|считает|указывает|приводит|описывает)\b",
)
_TASK_DOCUMENT_PATTERNS = _AUTO_RAG_PATTERNS + (
    r"\b(?:rag|рага|документальн\w*\s+контекст\w*|локальн\w*\s+индекс\w*)\b",
    r"\b(?:источник\w*|цитат\w*|chunk_id|чанк\w*)\b",
)
_PLAN_APPROVAL_PATTERNS = (
    r"^\s*/?approve_plan\s*$",
    r"\b(?:план\s+)?утвержда(?:ю|ем)\b",
    r"\bплан\s+(?:утвержден|утверждён|принят)\b",
    r"\b(?:согласен|согласна)\s+(?:с\s+)?(?:этим\s+)?планом\b",
    r"\bпринимаю\s+(?:этот\s+)?план\b",
)
_PLAN_APPROVAL_NEGATIONS = (
    "не утверждаю", "не утверждаем", "пока не утверждаю", "нельзя утверждать",
    "не согласен с планом", "не согласна с планом", "не принимаю план",
    "с поправками", "после исправления", "после изменений",
)


def requests_file_export(text: str) -> bool:
    """Распознаёт только явную просьбу создать локальный файловый результат."""
    lowered = str(text or "").lower().replace("ё", "е")
    if any(marker.replace("ё", "е") in lowered for marker in _FILE_EXPORT_NEGATIONS):
        return False
    return any(re.search(pattern, lowered, flags=re.DOTALL) for pattern in _FILE_EXPORT_PATTERNS)


def requests_outside_rag(text: str) -> bool:
    """Разрешает одноразовый выход из RAG только по явной положительной команде."""
    lowered = str(text or "").lower().replace("ё", "е")
    if any(marker.replace("ё", "е") in lowered for marker in _OUTSIDE_RAG_NEGATIONS):
        return False
    return any(re.search(pattern, lowered, flags=re.DOTALL) for pattern in _OUTSIDE_RAG_PATTERNS)


def deterministic_auto_route(text: str) -> dict[str, Any] | None:
    """Бесплатно маршрутизирует только однозначные формулировки."""
    lowered = str(text or "").lower().replace("ё", "е")
    task = any(re.search(pattern.replace("ё", "е"), lowered, flags=re.DOTALL) for pattern in _AUTO_TASK_PATTERNS)
    tools = any(re.search(pattern.replace("ё", "е"), lowered, flags=re.DOTALL) for pattern in _AUTO_TOOL_PATTERNS)
    rag = any(re.search(pattern.replace("ё", "е"), lowered, flags=re.DOTALL) for pattern in _AUTO_RAG_PATTERNS)
    if tools and rag:
        route = "hybrid"
    elif task:
        route = "task"
    elif tools:
        route = "tools"
    elif rag:
        route = "rag"
    else:
        return None
    return {"route": route, "confidence": 1.0, "reason": "deterministic_rule", "classifier": "local"}


def task_text_requires_documents(text: str) -> bool:
    """Определяет явную зависимость этапа задачи от локальных документов/RAG."""
    lowered = str(text or "").lower().replace("ё", "е")
    return any(
        re.search(pattern.replace("ё", "е"), lowered, flags=re.DOTALL)
        for pattern in _TASK_DOCUMENT_PATTERNS
    )


def explicitly_approves_plan(text: str) -> bool:
    """Распознаёт однозначное утверждение уже показанного плана без правок."""
    lowered = str(text or "").lower().replace("ё", "е")
    if any(marker.replace("ё", "е") in lowered for marker in _PLAN_APPROVAL_NEGATIONS):
        return False
    return any(
        re.search(pattern.replace("ё", "е"), lowered, flags=re.DOTALL)
        for pattern in _PLAN_APPROVAL_PATTERNS
    )


def strip_rag_evidence_appendix(text: str) -> str:
    """Оставляет смысл ответа, не возвращая старые источники и цитаты в новый контекст."""
    value = str(text or "")
    marker = re.search(r"\n\s*Источники(?: маршрута)?:\s*\n", value, flags=re.IGNORECASE)
    if marker:
        value = value[:marker.start()]
        value = re.sub(r"\s*\[\d+]", "", value)
    return value.strip()


def compact_rag_chunk(chunk: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "chunk_id", "source", "title", "section", "page", "strategy", "token_count",
        "text_hash", "score", "reranker_score", "direct_score", "context_score", "fusion_score",
        "relevance_score", "channel_gate_passed", "ledger_reused",
    )
    return {key: deepcopy(chunk.get(key)) for key in fields if chunk.get(key) is not None}


def normalize_rag_options(value: Any) -> dict[str, Any]:
    raw = value if isinstance(value, dict) else {}
    enabled = raw.get("enabled") is True
    strategy = str(raw.get("strategy") or "structural").strip().lower()
    if strategy not in RAG_STRATEGIES:
        raise ValueError("Стратегия RAG должна быть fixed, structural или combined.")
    mode = str(raw.get("mode") or "baseline").strip().lower()
    if mode not in RAG_PIPELINE_MODES:
        raise ValueError("Режим RAG должен быть baseline, rewrite, filter, rerank или combined.")
    try:
        final_k = int(raw.get("final_k", raw.get("top_k", 5)))
        candidate_k = int(raw.get("candidate_k", final_k))
        similarity_threshold = float(raw.get("similarity_threshold", 0.83))
    except (TypeError, ValueError) as error:
        raise ValueError("Параметры top-K и similarity threshold имеют неверный формат.") from error
    if not 1 <= candidate_k <= 100:
        raise ValueError("Candidate top-K должен быть от 1 до 100.")
    if not 1 <= final_k <= 20:
        raise ValueError("Final top-K должен быть от 1 до 20.")
    if final_k > candidate_k:
        raise ValueError("Final top-K не может быть больше candidate top-K.")
    if not -1 <= similarity_threshold <= 1:
        raise ValueError("Порог similarity должен быть от -1 до 1.")
    routing_mode = str(raw.get("routing_mode") or "strict").strip().lower()
    if routing_mode not in {"auto", "strict"}:
        raise ValueError("Маршрутизация RAG должна быть auto или strict.")
    return {
        "enabled": enabled,
        "verified": raw.get("verified") is True,
        "routing_mode": routing_mode,
        "strategy": strategy,
        "mode": mode,
        "top_k": final_k,
        "candidate_k": candidate_k,
        "final_k": final_k,
        "similarity_threshold": similarity_threshold,
    }


def rag_snapshot(
    options: dict[str, Any],
    retrieval: dict | None = None,
    *,
    include_chunk_text: bool = True,
) -> dict[str, Any]:
    snapshot = deepcopy(options)
    if retrieval:
        snapshot.update({
            "run_id": retrieval.get("run_id"),
            "query": retrieval.get("query"),
            "search_query": retrieval.get("search_query", retrieval.get("query")),
            "candidate_count": retrieval.get("candidate_count"),
            "after_filter_count": retrieval.get("after_filter_count"),
            "pre_gate_count": retrieval.get("pre_gate_count"),
            "post_gate_count": retrieval.get("post_gate_count"),
            "max_similarity": retrieval.get("max_similarity"),
            "preflight_max_similarity": retrieval.get("preflight_max_similarity"),
            "relevance_gate_passed": retrieval.get("relevance_gate_passed"),
            "gate_phase": retrieval.get("gate_phase"),
            "rewrite": deepcopy(retrieval.get("rewrite")),
            "direct_query": retrieval.get("direct_query"),
            "contextual_query": retrieval.get("contextual_query"),
            "direct_weight": retrieval.get("direct_weight"),
            "context_weight": retrieval.get("context_weight"),
            "contextual_threshold": retrieval.get("contextual_threshold"),
            "direct_candidate_count": retrieval.get("direct_candidate_count"),
            "context_candidate_count": retrieval.get("context_candidate_count"),
            "ledger_candidate_count": retrieval.get("ledger_candidate_count"),
            "chunks": (
                deepcopy(retrieval.get("chunks") or [])
                if include_chunk_text
                else [compact_rag_chunk(item) for item in retrieval.get("chunks") or []]
            ),
        })
    else:
        snapshot["chunks"] = []
    return snapshot


def unique_report_path(workspace: Path, requested_path: Any) -> Path:
    """Создаёт уникальный Markdown-путь строго внутри workspace/reports."""
    reports_dir = (workspace.resolve() / "reports").resolve()
    reports_dir.mkdir(parents=True, exist_ok=True)
    raw_name = Path(str(requested_path or "weather-report").replace("\\", "/")).name
    stem = Path(raw_name).stem.strip() or "weather-report"
    safe_stem = re.sub(r"[^0-9A-Za-zА-Яа-яЁё._-]+", "-", stem).strip(" .-_")[:80]
    safe_stem = safe_stem or "weather-report"
    timestamp = scheduler_now().strftime("%Y-%m-%d-%H-%M-%S")
    candidate = reports_dir / f"{safe_stem}-{timestamp}.md"
    if candidate.exists():
        candidate = reports_dir / f"{safe_stem}-{timestamp}-{uuid.uuid4().hex[:8]}.md"
    return candidate


def create_app(
    data_dir: Path | None = None,
    agent_instance: Agent | None = None,
    voice_service: WhisperService | None = None,
    mcp_manager: MCPManager | None = None,
    weather_mcp_manager: StdioMCPManager | None = None,
    scheduler_mcp_manager: StdioMCPManager | None = None,
    scheduler_service: SchedulerService | None = None,
    scheduler_autostart: bool = False,
    mediawiki_mcp_manager: StdioMCPManager | None = None,
    worldbank_mcp_manager: StdioMCPManager | None = None,
    orchestration_autostart: bool = False,
    rag_index_service: RagIndexService | None = None,
) -> Flask:
    flask_app = Flask(__name__)
    flask_app.json.ensure_ascii = False
    flask_app.config["MAX_CONTENT_LENGTH"] = 26 * 1024 * 1024
    storage = JsonStorage(data_dir or BASE_DIR / "data")
    preset_manager = PresetManager(storage)
    chat_agent = agent_instance or Agent(DeepSeekProvider())
    memory_manager = MemoryManager(storage, BASE_DIR / "config" / "agent_policy.json")
    invariant_store = InvariantStore(storage.data_dir)
    whisper = voice_service or WhisperService(BASE_DIR, autostart=data_dir is None)
    mcp = mcp_manager or MCPManager(BASE_DIR)
    weather_mcp = weather_mcp_manager or OpenMeteoMCPManager(BASE_DIR)
    scheduler_repository = (
        scheduler_service.repository if scheduler_service is not None
        else SchedulerRepository(storage.data_dir / "scheduler.sqlite3")
    )
    scheduler_mcp = scheduler_mcp_manager or SchedulerMCPManager(BASE_DIR, scheduler_repository.database_path)
    mediawiki_mcp = mediawiki_mcp_manager or MediaWikiMCPManager(BASE_DIR)
    worldbank_mcp = worldbank_mcp_manager or WorldBankMCPManager(BASE_DIR)
    scheduler = scheduler_service or SchedulerService(scheduler_repository)
    rag_index = rag_index_service or RagIndexService(
        (BASE_DIR / "rag_documents") if data_dir is None else (storage.data_dir / "rag_documents"),
        storage.data_dir / "rag_index.sqlite3",
    )
    rag_evaluations = RagEvaluationStore(storage.data_dir / "rag_evaluations.json")
    mcp_control = PersistentFeatureControl(storage.data_dir / "mcp_control.json")
    task_control = PersistentFeatureControl(storage.data_dir / "task_control.json")
    mcp_activity_lock = threading.RLock()
    mcp_activities: dict[str, dict[str, Any]] = {}

    flask_app.extensions["json_storage"] = storage
    flask_app.extensions["preset_manager"] = preset_manager
    flask_app.extensions["chat_agent"] = chat_agent
    flask_app.extensions["memory_manager"] = memory_manager
    flask_app.extensions["invariant_store"] = invariant_store
    flask_app.extensions["whisper_service"] = whisper
    flask_app.extensions["mcp_manager"] = mcp
    flask_app.extensions["weather_mcp_manager"] = weather_mcp
    flask_app.extensions["scheduler_mcp_manager"] = scheduler_mcp
    flask_app.extensions["mediawiki_mcp_manager"] = mediawiki_mcp
    flask_app.extensions["worldbank_mcp_manager"] = worldbank_mcp
    flask_app.extensions["scheduler_service"] = scheduler
    flask_app.extensions["rag_index_service"] = rag_index
    flask_app.extensions["rag_evaluation_store"] = rag_evaluations
    flask_app.extensions["mcp_control"] = mcp_control
    flask_app.extensions["task_control"] = task_control
    logger.info("Flask-приложение создано | data_dir=%s", storage.data_dir)

    def current_profile_id() -> str:
        return _profile_override.get() or storage.active_profile_id()

    def mcp_blocked_response():
        return api_error(
            "Все MCP принудительно отключены. Сначала нажмите «Разрешить MCP».",
            409,
        )

    def task_control_blocked_response():
        return api_error(
            "Машина задач принудительно отключена. Сначала нажмите «Включить машину задач».",
            409,
        )

    def rewritten_query(
        question: str,
        settings: AgentSettings,
        retrieval_context: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        result = chat_agent.rewrite_rag_query(question, settings, retrieval_context)
        value = " ".join(str(result.content or "").strip().strip('"').split())
        if not value:
            raise RuntimeError("DeepSeek не вернул поисковую формулировку.")
        return value[:2_000], {
            "query": value[:2_000],
            "technical": deepcopy(result.technical),
        }

    def conversational_retrieval_context(
        conversation: dict[str, Any],
        *,
        task_enabled: bool,
    ) -> dict[str, Any]:
        """Строит ограниченный контекст для rewrite без старых RAG-чанков и приложений цитат."""
        history, summary, facts, context_snapshot = request_history(
            conversation, stage_scoped=task_enabled,
        )
        completed = [
            item for item in history
            if item.get("role") in {"user", "assistant"}
            and item.get("technical", {}).get("request_status") in {None, "completed"}
        ]
        previous_users = [
            str(item.get("content") or "")[:1_500]
            for item in completed if item.get("role") == "user"
        ][-2:]
        previous_assistant = next((
            strip_rag_evidence_appendix(str(item.get("content") or ""))[:2_000]
            for item in reversed(completed) if item.get("role") == "assistant"
        ), "")
        payload: dict[str, Any] = {}
        if previous_users:
            payload["previous_user_messages"] = previous_users
        if previous_assistant:
            payload["previous_assistant_answer"] = previous_assistant
        if summary:
            cleaned_summary = strip_rag_evidence_appendix(str(summary))[:4_000]
            if cleaned_summary:
                payload["active_summary"] = cleaned_summary
        if facts:
            payload["sticky_facts"] = deepcopy(facts[:20])
        if task_enabled:
            state = ensure_task_state(conversation)
            default_step, default_action = STAGE_DEFAULTS[state.get("stage", "planning")]
            state_payload = {}
            if state.get("description"):
                state_payload["description"] = deepcopy(state["description"])
            if state.get("plan"):
                state_payload["plan"] = deepcopy(state["plan"])
            if state.get("current_step") and state.get("current_step") != default_step:
                state_payload["current_step"] = deepcopy(state["current_step"])
            if state.get("expected_action") and state.get("expected_action") != default_action:
                state_payload["expected_action"] = deepcopy(state["expected_action"])
            if state.get("stage") != "planning" or state.get("transition_history"):
                state_payload["stage"] = state.get("stage")
            if state_payload:
                payload["task_state"] = state_payload
        memory = task_memory_context(conversation)
        if memory:
            payload["task_memory"] = {
                key: deepcopy(memory.get(key))
                for key in ("goal", "clarifications", "constraints", "terms", "decisions", "open_questions")
                if memory.get(key)
            }
        if payload:
            payload["context_mode"] = context_snapshot.get("mode")
        return payload

    def automatic_task_document_query(
        conversation: dict[str, Any],
        state: dict[str, Any],
    ) -> str:
        """Возвращает содержательный RAG-запрос автопилота, если этап зависит от документов."""
        if state.get("stage") not in {"execution", "validation"}:
            return ""
        parts = [
            str(state.get("description") or "").strip(),
            str(state.get("plan") or "").strip(),
            str(state.get("current_step") or "").strip(),
            str(state.get("expected_action") or "").strip(),
        ]
        for handoff in active_task_handoffs(conversation)[-2:]:
            parts.append(str(handoff.get("summary") or "").strip())
            for field in (
                "approved_plan", "decisions", "constraints", "acceptance_criteria",
                "completed_work", "validation_findings", "open_questions",
            ):
                values = handoff.get(field)
                if isinstance(values, list):
                    parts.extend(str(item).strip() for item in values if str(item).strip())
        unique_parts = list(dict.fromkeys(item for item in parts if item))
        task_text = "\n".join(unique_parts)
        if not task_text_requires_documents(task_text):
            return ""
        return task_text[:4_000]

    def resolve_auto_route(
        question: str,
        settings: AgentSettings,
        retrieval_context: dict[str, Any],
        probe: dict[str, Any],
    ) -> dict[str, Any]:
        deterministic = deterministic_auto_route(question)
        if deterministic:
            return deterministic
        chunks = list(probe.get("chunks") or [])
        max_similarity = max((float(item.get("score", 0)) for item in chunks), default=None)
        prompt = {
            "question": question,
            "bounded_conversation_context": retrieval_context,
            "rag_probe": {
                "max_similarity": max_similarity,
                "sources": list(dict.fromkeys(
                    str(item.get("source")) for item in chunks if item.get("source")
                ))[:5],
                "sections": [str(item.get("section") or "")[:200] for item in chunks[:5]],
            },
            "capabilities": {
                "mcp_globally_enabled": mcp_control.is_enabled(),
                "task_state_enabled": task_control.is_enabled(),
            },
        }
        try:
            result = chat_agent.route_rag_request(
                json.dumps(prompt, ensure_ascii=False), settings,
            )
            value = json.loads(str(result.content or ""))
            route = str(value.get("route") or "").strip().lower()
            if route not in AUTO_RAG_ROUTES:
                raise ValueError("Неизвестный маршрут классификатора.")
            try:
                confidence = min(1.0, max(0.0, float(value.get("confidence", 0))))
            except (TypeError, ValueError):
                confidence = 0.0
            return {
                "route": route,
                "confidence": confidence,
                "reason": str(value.get("reason") or "llm_classifier")[:500],
                "classifier": "deepseek",
                "technical": deepcopy(result.technical),
            }
        except Exception as error:
            fallback = "rag" if max_similarity is not None and max_similarity >= 0.88 else "general"
            return {
                "route": fallback,
                "confidence": 0.0,
                "reason": f"classifier_fallback: {error}",
                "classifier": "fallback",
            }

    def append_auto_route_sources(
        result: AgentResult,
        route: str,
        retrieval: dict[str, Any] | None,
    ) -> AgentResult:
        """Добавляет проверяемую маркировку происхождения для нестрогих маршрутов."""
        if result.technical.get("request_status", "completed") != "completed":
            return result
        content = str(result.content or "").rstrip()
        evidence = result.technical.get("evidence") or {}
        if route == "rag" and evidence.get("status") == "answer" and evidence.get("valid") is True:
            return result
        lines: list[str] = []
        chunks = list((retrieval or {}).get("chunks") or [])
        local_sources = list(dict.fromkeys(
            str(item.get("source")) for item in chunks if item.get("source")
        ))
        successful_tools = [
            item for item in (result.technical.get("mcp_tool_calls") or [])
            if isinstance(item, dict) and not item.get("is_error")
        ]
        tool_sources = list(dict.fromkeys(
            str(item.get("server") or item.get("name") or "MCP") for item in successful_tools
        ))
        if route == "task":
            lines.append("- Task State текущего диалога и история активной ветки.")
        elif route == "general":
            lines.append("- Общие знания модели; локальная база не использовалась.")
        elif route == "tools":
            if tool_sources:
                lines.append("- Внешние MCP-инструменты: " + ", ".join(tool_sources) + ".")
            else:
                state = "принудительно отключены" if not mcp_control.is_enabled() else "не подключены или не были вызваны"
                lines.append(f"- Внешние данные не получены: MCP {state}.")
        elif route == "hybrid":
            lines.append("- Локальная база: " + (", ".join(local_sources) if local_sources else "подходящие чанки не найдены") + ".")
            lines.append("- Внешние MCP-инструменты: " + (", ".join(tool_sources) if tool_sources else "данные не получены") + ".")
        elif route == "rag":
            if local_sources:
                lines.append("- Локальная база: " + ", ".join(local_sources) + ".")
            else:
                lines.append("- Локальная база: релевантных фрагментов недостаточно.")
        if not lines:
            return result
        return AgentResult(
            content=content + "\n\nИсточники маршрута:\n" + "\n".join(lines),
            reasoning_content=result.reasoning_content,
            technical=deepcopy(result.technical),
        )

    def sanitize_history_for_generation(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
        result = deepcopy(history)
        for item in result:
            if item.get("role") == "assistant" and isinstance(item.get("content"), str):
                item["content"] = strip_rag_evidence_appendix(item["content"])
        return result

    def retrieve_rag(
        question: str,
        options: dict[str, Any],
        settings: AgentSettings,
        *,
        prepared_rewrite: tuple[str, dict[str, Any]] | None = None,
        retrieval_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        mode = options.get("mode", "baseline")
        if mode == "baseline":
            return rag_index.retrieve(question, options["strategy"], options["final_k"])
        rewrite_value = None
        rewrite_audit = None
        if mode in {"rewrite", "combined"}:
            rewrite_value, rewrite_audit = prepared_rewrite or rewritten_query(
                question, settings, retrieval_context,
            )
        if mode == "combined" and retrieval_context:
            retrieval = rag_index.retrieve_conversational(
                question,
                rewrite_value or question,
                strategy=options["strategy"],
                candidate_k=options["candidate_k"],
                final_k=options["final_k"],
                similarity_threshold=options["similarity_threshold"],
                direct_weight=0.7,
            )
        else:
            retrieval = rag_index.retrieve_pipeline(
                question,
                strategy=options["strategy"],
                mode=mode,
                candidate_k=options["candidate_k"],
                final_k=options["final_k"],
                similarity_threshold=options["similarity_threshold"],
                rewritten_query=rewrite_value,
            )
        if rewrite_audit:
            retrieval["rewrite"] = rewrite_audit
        return retrieval

    def retrieve_verified_rag(
        question: str,
        options: dict[str, Any],
        settings: AgentSettings,
        retrieval_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Проверяет direct и contextual каналы до разрешения основного ответа."""
        if options.get("mode") in {"rewrite", "combined"}:
            retrieval = retrieve_rag(
                question, options, settings, retrieval_context=retrieval_context,
            )
            gated = apply_relevance_gate(retrieval, options["similarity_threshold"])
            gated["gate_phase"] = "contextual_preflight"
            return gated
        preflight = rag_index.retrieve(question, options["strategy"], options["candidate_k"])
        gated_preflight = apply_relevance_gate(preflight, options["similarity_threshold"])
        gated_preflight["gate_phase"] = "preflight"
        if not gated_preflight["chunks"]:
            gated_preflight.update({
                "mode": options.get("mode", "baseline"),
                "candidate_k": options["candidate_k"],
                "final_k": options["final_k"],
                "candidate_count": len(preflight.get("chunks") or []),
                "after_filter_count": 0,
            })
            return gated_preflight
        retrieval = retrieve_rag(
            question, options, settings, retrieval_context=retrieval_context,
        )
        gated = apply_relevance_gate(retrieval, options["similarity_threshold"])
        gated["gate_phase"] = "final"
        gated["preflight_max_similarity"] = gated_preflight.get("max_similarity")
        return gated

    def rag_miss_fallback(draft: AgentResult, retrieval: dict[str, Any]) -> AgentResult:
        """Маркирует общий ответ модели, когда relevance-gate не пропустил чанки."""
        return AgentResult(
            content=(
                f"{RAG_MISS_NOTICE}\n\n{RAG_MISS_ANSWER_LABEL}\n\n"
                + draft.content.strip()
            ),
            reasoning_content=draft.reasoning_content,
            technical={
                **deepcopy(draft.technical),
                "rag_fallback_to_model": True,
                "evidence": {
                    "status": "general_fallback",
                    "valid": True,
                    "sources_present": False,
                    "quotes_present": False,
                    "citations": [],
                    "errors": [],
                    "reason": "below_similarity_threshold",
                    "max_similarity": retrieval.get("max_similarity"),
                    "threshold": retrieval.get("relevance_threshold"),
                },
            },
        )

    def finalize_verified_rag(
        question: str,
        draft: AgentResult,
        retrieval: dict[str, Any],
        settings: AgentSettings,
    ) -> AgentResult:
        """Проверяет JSON/цитаты и делает не более одной исправляющей попытки."""
        repair_result = None
        first_error = None
        try:
            content, evidence = validate_and_format_evidence(draft.content, retrieval)
            if evidence.get("status") == "unknown" and retrieval.get("chunks"):
                repair_result = chat_agent.repair_rag_evidence(
                    retry_unknown_prompt(question, draft.content, retrieval), settings
                )
                try:
                    retry_content, retry_evidence = validate_and_format_evidence(
                        repair_result.content, retrieval,
                    )
                    if retry_evidence.get("status") == "answer":
                        content, evidence = retry_content, retry_evidence
                    else:
                        evidence["retry_reason"] = "model_confirmed_unknown"
                except EvidenceValidationError as retry_error:
                    evidence["retry_reason"] = "unknown_retry_invalid"
                    evidence["retry_error"] = str(retry_error)
        except EvidenceValidationError as error:
            first_error = str(error)
            repair_result = chat_agent.repair_rag_evidence(
                repair_prompt(question, draft.content, retrieval, first_error), settings
            )
            try:
                content, evidence = validate_and_format_evidence(repair_result.content, retrieval)
            except EvidenceValidationError as repair_error:
                content = RAG_UNSUPPORTED_RESPONSE
                evidence = {
                    "status": "unknown",
                    "valid": False,
                    "sources_present": False,
                    "quotes_present": False,
                    "citations": [],
                    "errors": [first_error, str(repair_error)],
                    "reason": "citation_validation_failed",
                }
        technical = deepcopy(draft.technical)
        technical["evidence"] = evidence
        if repair_result is not None:
            technical["evidence_repair"] = deepcopy(repair_result.technical)
        return AgentResult(
            content=content,
            reasoning_content=draft.reasoning_content,
            technical=technical,
        )

    def is_synthesis_question(text: str) -> bool:
        lowered = str(text or "").lower().replace("ё", "е")
        return any(marker in lowered for marker in (
            "итог", "собери", "сводк", "обобщ", "вывод по", "заверши", "сопоставь",
        ))

    def evidence_ledger_ids(conversation: dict[str, Any]) -> list[str]:
        ledger = conversation.get("rag_evidence_ledger")
        if not isinstance(ledger, list):
            return []
        return list(dict.fromkeys(
            str(item.get("chunk_id")) for item in ledger
            if isinstance(item, dict) and item.get("chunk_id")
        ))[-30:]

    def update_evidence_ledger(
        conversation: dict[str, Any], evidence: dict[str, Any], exchange_id: str,
    ) -> None:
        if evidence.get("status") != "answer" or evidence.get("valid") is not True:
            return
        ledger = conversation.setdefault("rag_evidence_ledger", [])
        existing = {str(item.get("chunk_id")): item for item in ledger if isinstance(item, dict)}
        for citation in evidence.get("citations") or []:
            chunk_id = str(citation.get("chunk_id") or "")
            if not chunk_id:
                continue
            existing[chunk_id] = {
                "chunk_id": chunk_id,
                "source": citation.get("source"),
                "section": citation.get("section"),
                "page": citation.get("page"),
                "exchange_id": exchange_id,
                "verified": True,
            }
        conversation["rag_evidence_ledger"] = list(existing.values())[-30:]

    def known_rag_questions() -> dict[str, dict[str, Any]]:
        questions: dict[str, dict[str, Any]] = {}
        for filename in ("day22_control_questions.json", "day24_control_questions.json"):
            path = BASE_DIR / "docs" / filename
            if not path.exists():
                continue
            value = json.loads(path.read_text(encoding="utf-8"))
            for item in value.get("questions", []) if isinstance(value, dict) else []:
                if isinstance(item, dict) and item.get("id"):
                    questions[str(item["id"])] = item
        return questions

    def begin_mcp_activity(activity_id: str | None) -> str | None:
        if not activity_id or not re.fullmatch(r"[0-9a-f]{32}", activity_id):
            return None
        with mcp_activity_lock:
            if len(mcp_activities) >= 100:
                oldest = min(mcp_activities, key=lambda key: mcp_activities[key].get("created_at", ""))
                mcp_activities.pop(oldest, None)
            mcp_activities[activity_id] = {
                "id": activity_id, "phase": "thinking", "created_at": now_iso(), "calls": [],
            }
        return activity_id

    def public_tool_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
        safe: dict[str, Any] = {}
        for key, value in arguments.items():
            lowered = str(key).lower()
            if any(marker in lowered for marker in ("content", "password", "secret", "token", "key")):
                safe[key] = f"<{len(str(value))} символов>" if value is not None else None
                continue
            encoded = json.dumps(value, ensure_ascii=False)
            safe[key] = value if len(encoded) <= 500 else encoded[:497] + "…"
        return safe

    def set_mcp_activity_phase(activity_id: str | None, phase: str, error: str | None = None) -> None:
        if not activity_id:
            return
        with mcp_activity_lock:
            activity = mcp_activities.get(activity_id)
            if activity:
                activity["phase"] = phase
                activity["updated_at"] = now_iso()
                if error:
                    activity["error"] = error

    def begin_tool_activity(activity_id: str | None, server: str | None, name: str,
                            arguments: dict[str, Any]) -> int | None:
        if not activity_id:
            return None
        with mcp_activity_lock:
            activity = mcp_activities.get(activity_id)
            if not activity:
                return None
            call_id = len(activity["calls"]) + 1
            activity["phase"] = "tools"
            activity["calls"].append({
                "id": call_id, "server": server, "name": name,
                "arguments": public_tool_arguments(arguments), "status": "running",
                "started_at": now_iso(),
            })
            activity["updated_at"] = now_iso()
            return call_id

    def finish_tool_activity(activity_id: str | None, call_id: int | None, *, is_error: bool,
                             error: str | None = None) -> None:
        if not activity_id or call_id is None:
            return
        with mcp_activity_lock:
            activity = mcp_activities.get(activity_id)
            if not activity:
                return
            call = next((item for item in activity["calls"] if item["id"] == call_id), None)
            if call:
                call["status"] = "error" if is_error else "completed"
                call["finished_at"] = now_iso()
                if error:
                    call["error"] = error
            activity["updated_at"] = now_iso()

    def invariant_context(conversation: dict[str, Any], profile_id: str) -> dict[str, Any]:
        policy = memory_manager.policy()
        return invariant_store.bundle(
            profile_id=profile_id,
            project_id=conversation.get("project_id"),
            conversation_id=conversation["id"],
            system_rules=policy.get("rules", []),
        )

    def controlled_agent_reply(
        *,
        history: list[dict[str, Any]],
        user_text: str,
        settings: AgentSettings,
        summary: str,
        facts: list[dict[str, str]],
        memory_context: dict[str, Any],
        task_memory_snapshot: dict[str, Any],
        task_state: dict[str, Any],
        handoff_context: dict[str, Any],
        invariants: dict[str, Any],
        rag_context: str = "",
        tool_context: dict[str, Any] | None = None,
    ) -> AgentResult:
        """Генерирует черновик и не выпускает его без независимой проверки политики."""
        available_tools: list[dict[str, Any]] = []
        tool_owners: dict[str, str] = {}
        created_scheduler_tasks: list[tuple[str, str]] = []
        created_report_paths: list[Path] = []
        mcp_enabled = mcp_control.is_enabled()
        task_control_enabled = task_control.is_enabled()
        auto_route = str((tool_context or {}).get("auto_route") or "")
        if mcp_enabled and weather_mcp.status().get("connected"):
            for tool in weather_mcp.tools_for_model():
                available_tools.append(tool)
                tool_owners[tool["name"]] = "open-meteo"
        if mcp_enabled and mediawiki_mcp.status().get("connected"):
            for tool in mediawiki_mcp.tools_for_model():
                if tool["name"] in MEDIAWIKI_MODEL_TOOLS:
                    available_tools.append(tool)
                    tool_owners[tool["name"]] = "mediawiki"
        if mcp_enabled and worldbank_mcp.status().get("connected"):
            for tool in worldbank_mcp.tools_for_model():
                if tool["name"] in WORLDBANK_MODEL_TOOLS:
                    available_tools.append(tool)
                    tool_owners[tool["name"]] = "worldbank"
        if mcp_enabled and scheduler_mcp.status().get("connected") and _scheduled_tool_allowlist.get() is None:
            for tool in scheduler_mcp.tools_for_model():
                if tool["name"] in SCHEDULER_MODEL_TOOLS:
                    available_tools.append(tool)
                    tool_owners[tool["name"]] = "scheduler"
        allowlist = _scheduled_tool_allowlist.get()
        file_export_enabled = (
            mcp_enabled
            and bool((tool_context or {}).get("enable_file_export"))
            and allowlist is None
        )
        if file_export_enabled:
            if not mcp.status().get("connected"):
                mcp.start()
            filesystem_tools = mcp.list_tools().get("tools") or []
            for tool in filesystem_tools:
                if tool.get("name") not in FILESYSTEM_MODEL_TOOLS:
                    continue
                safe_tool = deepcopy(tool)
                safe_tool["description"] = (
                    "Сохранить подготовленный Markdown-отчёт. Инструмент доступен только при явной просьбе "
                    "пользователя сохранить файл. Передайте полное содержимое отчёта и осмысленное базовое имя; "
                    "приложение принудительно сохранит новый .md-файл в workspace/reports и добавит дату и время."
                )
                schema = deepcopy(safe_tool.get("input_schema") or {})
                properties = schema.setdefault("properties", {})
                properties.setdefault("path", {})["description"] = (
                    "Базовое имя отчёта, например weather-novosibirsk.md. Каталог будет заменён на workspace/reports."
                )
                properties.setdefault("content", {})["description"] = "Полное содержимое Markdown-отчёта."
                schema["required"] = ["path", "content"]
                safe_tool["input_schema"] = schema
                available_tools.append(safe_tool)
                tool_owners[safe_tool["name"]] = "filesystem"
        if allowlist is not None:
            available_tools = [tool for tool in available_tools if tool["name"] in allowlist]
        if auto_route and auto_route not in {"tools", "hybrid"}:
            available_tools = []

        def execute_owned_tool(name: str, arguments: dict[str, Any], owner: str | None) -> dict[str, Any]:
            if not mcp_control.is_enabled():
                raise PermissionError("Все MCP принудительно отключены пользователем.")
            if owner == "open-meteo":
                return weather_mcp.call_tool(name, arguments)
            if owner == "mediawiki":
                return mediawiki_mcp.call_tool(name, arguments)
            if owner == "worldbank":
                return worldbank_mcp.call_tool(name, arguments)
            if owner == "filesystem":
                if name != "write_file" or not file_export_enabled:
                    raise PermissionError("Запись файлов не разрешена для этого запроса.")
                content = arguments.get("content")
                if not isinstance(content, str) or not content.strip():
                    raise ValueError("Для отчёта требуется непустое текстовое содержимое.")
                if len(content) > 500_000:
                    raise ValueError("Отчёт превышает допустимый размер 500 000 символов.")
                workspace_value = mcp.status().get("workspace")
                if not workspace_value:
                    raise RuntimeError("Filesystem MCP не сообщил разрешённую рабочую папку.")
                workspace_path = Path(str(workspace_value)).resolve()
                report_path = unique_report_path(workspace_path, arguments.get("path"))
                output = mcp.call_tool(name, {"path": str(report_path), "content": content})
                if not output.get("is_error"):
                    created_report_paths.append(report_path)
                    output["result"] = {
                        "mcp_result": output.get("result"),
                        "relative_path": report_path.relative_to(workspace_path).as_posix(),
                        "absolute_path": str(report_path),
                    }
                return output
            if owner != "scheduler":
                raise RuntimeError(f"MCP-инструмент {name!r} не принадлежит подключённому серверу.")
            profile_id = str((tool_context or {}).get("profile_id") or current_profile_id())
            task_id = str(arguments.get("task_id") or "")
            if task_id:
                existing = scheduler_repository.get_task(task_id)
                if existing.get("owner_profile_id") not in {None, profile_id}:
                    raise PermissionError("Запланированное задание принадлежит другому профилю.")
            output = scheduler_mcp.call_tool(name, arguments)
            if output.get("is_error"):
                return output
            if name == "create_scheduled_task":
                created = output.get("result")
                if not isinstance(created, dict) or not created.get("id"):
                    raise RuntimeError("Scheduler MCP не вернул созданное задание.")
                try:
                    conversation = storage.create_conversation(
                        f"Автоматизация · {created.get('title', 'Задание')}",
                        (tool_context or {}).get("project_id"), profile_id,
                    )
                    attached = scheduler_repository.attach_context(
                        str(created["id"]), owner_profile_id=profile_id,
                        conversation_id=conversation["id"],
                        source_conversation_id=(tool_context or {}).get("source_conversation_id"),
                        project_id=(tool_context or {}).get("project_id"),
                        settings=(tool_context or {}).get("settings") or {},
                    )
                except Exception:
                    scheduler_repository.delete_task(str(created["id"]))
                    raise
                created_scheduler_tasks.append((str(created["id"]), conversation["id"]))
                output["result"] = attached
            elif name == "list_scheduled_tasks":
                tasks = scheduler_repository.list_tasks(profile_id)
                output["result"] = {"count": len(tasks), "timezone": TIMEZONE_NAME, "tasks": tasks}
            elif name == "get_scheduler_summary":
                output["result"] = scheduler_repository.summary(profile_id)
            return output

        def execute_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
            owner = tool_owners.get(name)
            activity_id = str((tool_context or {}).get("mcp_activity_id") or "") or None
            call_id = begin_tool_activity(activity_id, owner, name, arguments)
            try:
                output = execute_owned_tool(name, arguments, owner)
                output["mcp_server"] = owner
                finish_tool_activity(activity_id, call_id, is_error=bool(output.get("is_error")))
                return output
            except Exception as error:
                finish_tool_activity(activity_id, call_id, is_error=True, error=str(error))
                raise

        def rollback_created_schedules() -> None:
            for task_id, automation_conversation_id in created_scheduler_tasks:
                try:
                    scheduler_repository.delete_task(task_id)
                    storage.delete_conversation(automation_conversation_id)
                except (FileNotFoundError, ValueError):
                    logger.warning("Не удалось откатить Scheduler MCP | task_id=%s", task_id)

        def rollback_created_reports() -> None:
            for report_path in created_report_paths:
                try:
                    report_path.unlink(missing_ok=True)
                except OSError:
                    logger.warning("Не удалось откатить файловый отчёт | path=%s", report_path)

        try:
            draft_result = chat_agent.reply(
                history,
                user_text,
                settings,
                summary=summary,
                facts=facts,
                memory_context=memory_context,
                task_memory=task_memory_snapshot,
                task_handoff_context=handoff_context if task_control_enabled else {},
                task_state=task_state if task_control_enabled else {},
                policy_guidance=(generation_guidance(task_state, invariants) if task_control_enabled else "") + rag_context + (
                    "\n\nКОМПОЗИЦИЯ ОТЧЁТА:\n"
                    "Пользователь явно попросил сохранить результат. Сначала получи необходимые исходные данные "
                    "через доступные MCP-инструменты, затем самостоятельно проанализируй их строго по запросу "
                    "пользователя и вызови write_file ровно с готовым полным Markdown-отчётом. Не утверждай, что "
                    "файл сохранён, пока write_file не вернул успешный результат.\n"
                    if file_export_enabled else ""
                ) + (
                    "\n\nОРКЕСТРАЦИЯ MCP:\n"
                    "Выбирай только необходимые инструменты. Если результат одного MCP-сервера определяет "
                    "аргументы следующего, сначала получи первый результат, затем вызови следующий инструмент "
                    "с фактически найденными значениями. Не подменяй вызов догадкой.\n"
                ) + (
                    "\n\nАВТОМАТИЧЕСКИЙ МАРШРУТ ИНСТРУМЕНТОВ:\n"
                    "Запрос требует внешних данных или действия, но доступных MCP-инструментов сейчас нет. "
                    "Не выдумывай результат API. Прямо сообщи, что MCP выключены либо нужный сервер не подключён.\n"
                    if auto_route in {"tools", "hybrid"} and not available_tools else ""
                ),
                tools=available_tools or None,
                tool_executor=execute_tool if available_tools else None,
            )
        except Exception:
            set_mcp_activity_phase(
                str((tool_context or {}).get("mcp_activity_id") or "") or None,
                "error",
            )
            rollback_created_schedules()
            rollback_created_reports()
            raise
        if not task_control_enabled:
            set_mcp_activity_phase(
                str((tool_context or {}).get("mcp_activity_id") or "") or None,
                "completed",
            )
            return AgentResult(
                content=draft_result.content,
                reasoning_content=draft_result.reasoning_content,
                technical={
                    **draft_result.technical,
                    "request_status": "completed",
                    "task_control": {"enabled": False},
                },
            )
        validation = None
        validation_result = None
        validation_retry_result = None
        accepted = False
        reason = ""
        performed_actions = [
            str(item.get("name")) for item in draft_result.technical.get("mcp_tool_calls", [])
            if isinstance(item, dict) and (
                str(item.get("name", "")).startswith(("create_scheduled_", "list_scheduled_", "get_scheduled_", "get_scheduler_"))
                or str(item.get("name", "")) == "write_file"
            )
        ]
        validator_prompt = validation_prompt(
            user_text, draft_result.content, task_state, invariants,
            performed_actions=performed_actions,
        )
        try:
            validation_result = chat_agent.validate_task_response(
                validator_prompt, settings
            )
            validation = parse_validation_result(validation_result.content, task_state, invariants)
            accepted = validation["allowed"]
            if not accepted:
                reason = validation.get("explanation") or "Фактическое действие ответа запрещено текущей политикой."
        except PolicyValidationError as error:
            reason = str(error)
            try:
                validation_retry_result = chat_agent.validate_task_response(
                    validator_prompt
                    + "\n\nПредыдущая попытка не прошла проверку формата: "
                    + reason
                    + " Верни заново полный JSON со всеми обязательными полями.",
                    settings,
                )
                validation = parse_validation_result(
                    validation_retry_result.content, task_state, invariants,
                )
                accepted = validation["allowed"]
                reason = "" if accepted else (
                    validation.get("explanation") or "Фактическое действие ответа запрещено текущей политикой."
                )
            except PolicyValidationError as retry_error:
                reason = f"{reason} Повторная проверка: {retry_error}"

        if not accepted:
            rollback_created_schedules()
            rollback_created_reports()

        set_mcp_activity_phase(
            str((tool_context or {}).get("mcp_activity_id") or "") or None,
            "completed" if accepted else "blocked",
        )

        content = draft_result.content if accepted else blocked_content(
            reason or "Ответ не прошёл обязательную проверку.",
            task_state,
            invariants,
            (validation or {}).get("violated_invariant_ids", []),
        )
        generation_usage = normalize_token_usage(draft_result.technical.get("usage"))
        validation_usage = normalize_token_usage(
            validation_result.technical.get("usage") if validation_result else {}
        )
        retry_validation_usage = normalize_token_usage(
            validation_retry_result.technical.get("usage") if validation_retry_result else {}
        )
        validation_usage = {
            key: validation_usage[key] + retry_validation_usage[key]
            for key in validation_usage
        }
        combined_usage = {key: generation_usage[key] + validation_usage[key] for key in generation_usage}
        audit = policy_audit(
            validation, task_state, invariants,
            accepted=accepted,
            reason=reason,
        )
        return AgentResult(
            content=content,
            reasoning_content=draft_result.reasoning_content if accepted else "",
            technical={
                **draft_result.technical,
                "usage": combined_usage,
                "policy_audit": audit,
                "policy_generation_usage": generation_usage,
                "policy_validation_usage": validation_usage,
                "policy_validation_retried": validation_retry_result is not None,
                "request_status": "completed" if accepted else "blocked",
            },
        )

    def automatic_event_is_authorized(conversation: dict[str, Any], event: str) -> bool:
        """Принимает автопереход только из последнего проверенного ответа текущего запуска этапа."""
        if not task_control.is_enabled():
            return False
        state = ensure_task_state(conversation)
        if state.get("transition_mode") != "automatic" or event not in allowed_task_events(state):
            return False
        exchanges = current_stage_exchanges(conversation)
        if not exchanges:
            return False
        assistant = next(
            (item for item in reversed(exchanges[-1].get("messages", [])) if item.get("role") == "assistant"),
            None,
        )
        technical = (assistant or {}).get("technical", {})
        audit = technical.get("policy_audit", {}) if isinstance(technical, dict) else {}
        snapshot = technical.get("task_state", {}) if isinstance(technical, dict) else {}
        return bool(
            technical.get("request_status") == "completed"
            and audit.get("accepted") is True
            and audit.get("stage_complete") is True
            and audit.get("recommended_event") == event
            and snapshot.get("stage_run_id") == state.get("stage_run_id")
        )

    def apply_explicit_plan_approval(
        conversation: dict[str, Any],
        user_text: str,
        task_state: dict[str, Any],
        audit: dict[str, Any],
    ) -> None:
        """Не даёт вероятностному валидатору потерять явное утверждение плана."""
        if (
            task_state.get("stage") != "planning"
            or task_state.get("transition_mode") != "automatic"
            or audit.get("accepted") is not True
            or not explicitly_approves_plan(user_text)
        ):
            return
        prior_approval_request = None
        for message in reversed(active_path_messages(conversation)):
            if message.get("role") != "assistant":
                continue
            technical = message.get("technical") if isinstance(message.get("technical"), dict) else {}
            prior_audit = technical.get("policy_audit") if isinstance(technical.get("policy_audit"), dict) else {}
            if (
                technical.get("request_status") == "completed"
                and prior_audit.get("accepted") is True
                and prior_audit.get("detected_action_type") in {"planning", "request_plan_approval"}
                and (technical.get("task_state") or {}).get("stage_run_id") == task_state.get("stage_run_id")
            ):
                prior_approval_request = prior_audit
                break
        if prior_approval_request is None:
            return
        audit.update({
            "stage_complete": True,
            "recommended_event": "approve_plan",
            "required_artifacts": list(prior_approval_request.get("required_artifacts", [])),
            "completion_source": "explicit_user_plan_approval",
            "reason": "Пользователь явно утвердил ранее предложенный план в режиме автопилота.",
        })

    def automatic_continuation_text(state: dict[str, Any]) -> str:
        """Формирует видимое служебное сообщение автопилота без пользовательского ввода."""
        return (
            "[Автопилот] Продолжи задачу на текущем этапе. Используй активный handoff, состояние задачи, "
            "память и инварианты. Выполни весь доступный объём этапа. Если для безопасного продолжения нужен "
            "критический выбор пользователя или отсутствуют обязательные данные, явно запроси их и не считай "
            f"этап завершённым. Текущий этап: {state['stage']}; шаг: {state['current_step']}."
        )

    def assert_artifact_transition(conversation: dict[str, Any], event: str) -> None:
        if event == "complete_execution":
            missing = missing_required_artifacts(conversation, storage.data_dir)
            if missing:
                raise TaskTransitionError(
                    "Нельзя завершить выполнение: отсутствуют обязательные артефакты: " + ", ".join(missing) + "."
                )
        if event == "pass_validation" and not required_artifacts_are_validated(conversation, storage.data_dir):
            raise TaskTransitionError(
                "Нельзя завершить задачу: обязательные артефакты не подтверждены текущим этапом валидации."
            )

    def append_completion_result(conversation: dict[str, Any]) -> None:
        """Запрашивает у LLM итог этапа done и прикладывает серверный реестр файлов."""
        artifacts = [public_artifact(item) for item in active_artifacts(conversation)]
        final_handoff = next(
            (
                item for item in reversed(conversation.get("task_handoffs", []))
                if item.get("target_stage") == "done"
            ),
            {},
        )
        summary = str(final_handoff.get("summary") or "").strip()
        completed_work = [str(item) for item in final_handoff.get("completed_work", []) if str(item).strip()]
        artifact_lines = [
            f"- {item['path']} · версия {item['version']} · {item['size_bytes']} байт · SHA-256 {item['sha256'][:12]}…"
            for item in artifacts
        ]
        prompt = (
            "[Завершение задачи] Сформируй последнее итоговое сообщение пользователю. "
            "Задача уже прошла валидацию и находится на этапе done. Используй только авторитетный handoff, "
            "состояние задачи, память, инварианты и приведённый ниже точный реестр артефактов. "
            "Кратко сообщи конечный результат, перечисли фактически выполненную работу и обязательно перечисли "
            "каждый созданный артефакт по точному пути. Не придумывай отсутствующие файлы, проверки или ссылки; "
            "полное содержимое файлов повторять не нужно — интерфейс приложит кнопки открытия и скачивания.\n\n"
            f"Итог handoff: {summary or 'не указан'}\n"
            "Выполнено по handoff:\n"
            + ("\n".join(f"- {item}" for item in completed_work) or "- не указано")
            + "\nТочный реестр артефактов:\n"
            + ("\n".join(artifact_lines) or "- файловых артефактов нет")
        )
        settings_source = last_configuration(active_path_messages(conversation)) or {}
        settings = AgentSettings.from_dict(settings_source.get("settings", AgentSettings().to_dict()))
        configuration = deepcopy(settings_source.get("configuration_source")) or {
            "type": "custom", "preset_id": None, "preset_name": None,
        }
        history, summary_text, facts, context_snapshot = request_history(conversation)
        profile_id = current_profile_id()
        memory_snapshot = memory_manager.context_snapshot(conversation, context_snapshot["mode"], profile_id)
        task_memory_snapshot = task_memory_context(conversation)
        state_snapshot = deepcopy(ensure_task_state(conversation))
        state_snapshot["registered_artifacts"] = artifacts
        handoff_snapshot = task_handoff_context(conversation)
        invariant_snapshot = invariant_context(conversation, profile_id)
        result = controlled_agent_reply(
            history=history,
            user_text=prompt,
            settings=settings,
            summary=summary_text,
            facts=facts,
            memory_context=memory_snapshot,
            task_memory_snapshot=task_memory_snapshot,
            task_state=state_snapshot,
            handoff_context=handoff_snapshot,
            invariants=invariant_snapshot,
        )
        if result.technical.get("request_status") != "completed":
            reason = result.technical.get("policy_audit", {}).get("reason") or "финальный ответ заблокирован контроллером"
            raise TaskTransitionError(f"Переход в done не сохранён: {reason}.")
        exact_manifest = "\n".join([
            "",
            "Созданные артефакты:",
            *(artifact_lines or ["- файловых артефактов нет"]),
        ])
        usage = normalize_token_usage(result.technical.get("usage"))
        if artifacts:
            exact_manifest += "\nОткрыть или скачать файлы можно кнопками в карточке итогового результата."
        message = {
            "id": uuid.uuid4().hex,
            "role": "assistant",
            "content": result.content.rstrip() + "\n" + exact_manifest,
            "reasoning_content": result.reasoning_content,
            "created_at": now_iso(),
            "technical": {
                **result.technical,
                "request_status": "completed",
                "final_completion": True,
                "configuration_source": configuration,
                "settings": settings.to_dict(),
                "context_management": context_snapshot,
                "memory_context": memory_snapshot,
                "profile_snapshot": deepcopy(memory_snapshot.get("profile_snapshot", {})),
                "task_state": state_snapshot,
                "task_handoff_context": handoff_snapshot,
                "invariants": invariant_snapshot,
                "result_manifest": {
                    "status": "done",
                    "summary": summary,
                    "completed_work": completed_work,
                    "artifacts": artifacts,
                },
            },
        }
        append_tree_message(conversation, message)
        totals = dialogue_token_totals(conversation.get("messages", []))
        conversation["token_totals"] = totals
        message["technical"]["dialogue_totals"] = totals

    def accessible_project(project_id: str, *, owner_only: bool = False) -> dict[str, Any]:
        project = storage.get_project(project_id)
        profile_id = current_profile_id()
        allowed = profile_id == project.get("owner_profile_id") if owner_only else storage.profile_can_access_project(project, profile_id)
        if not allowed:
            raise PermissionError(project_id)
        return project

    def accessible_conversation(conversation_id: str) -> dict[str, Any]:
        conversation = storage.get_conversation(conversation_id)
        if conversation.get("project_id"):
            accessible_project(str(conversation["project_id"]))
        elif conversation.get("owner_profile_id") != current_profile_id():
            raise PermissionError(conversation_id)
        return conversation

    def paused_task_response(conversation: dict[str, Any]):
        return jsonify({
            "ok": False,
            "error": "Задача приостановлена. Нажмите «Продолжить», чтобы снова выполнять действия агента.",
            "conversation": with_token_totals(conversation),
        }), 409

    def append_task_state_report(conversation: dict[str, Any], content: str) -> dict[str, Any]:
        """Сохраняет локальную пару /state, не вызывая провайдера и не меняя контекст LLM."""
        ensure_context_management(conversation)
        exchange_id = uuid.uuid4().hex
        timestamp = now_iso()
        snapshot = deepcopy(ensure_task_state(conversation))
        user_message = {
            "id": uuid.uuid4().hex,
            "exchange_id": exchange_id,
            "role": "user",
            "content": content,
            "created_at": timestamp,
            "author_profile_id": current_profile_id(),
            "technical": {"request_status": "local", "local_command": "task_state"},
        }
        append_tree_message(conversation, user_message)
        append_tree_message(conversation, {
            "id": uuid.uuid4().hex,
            "exchange_id": exchange_id,
            "role": "assistant",
            "content": task_state_report(snapshot),
            "reasoning_content": "",
            "created_at": timestamp,
            "technical": {
                "request_status": "local",
                "local_command": "task_state",
                "task_state": snapshot,
            },
        }, parent_id=user_message["id"])
        if conversation.get("title") == "Новый диалог":
            conversation["title"] = "Состояние задачи"
        return storage.save_conversation(conversation)

    @flask_app.before_request
    def begin_request_log() -> None:
        g.request_started = time.perf_counter()
        g.request_id = uuid.uuid4().hex[:12]
        g.logging_request_token = set_request_id(g.request_id)

    @flask_app.after_request
    def finish_request_log(response):
        response.headers["X-Request-ID"] = g.get("request_id", "")
        path = request.path
        level = logging.DEBUG if path == "/api/voice/status" else (
            logging.WARNING if response.status_code >= 400 else logging.INFO
        )
        if path.startswith("/api/"):
            logger.log(
                level,
                "HTTP | request_id=%s | method=%s | path=%s | status=%s | elapsed_ms=%.1f",
                g.get("request_id", "-"), request.method, path, response.status_code,
                (time.perf_counter() - g.get("request_started", time.perf_counter())) * 1000,
            )
        return response

    @flask_app.teardown_request
    def log_unhandled_request_error(error: BaseException | None) -> None:
        if error is not None:
            logger.error(
                "Необработанная ошибка HTTP | request_id=%s | method=%s | path=%s",
                g.get("request_id", "-"), request.method, request.path,
                exc_info=(type(error), error, error.__traceback__),
            )
        token = g.get("logging_request_token")
        if token is not None:
            reset_request_id(token)

    @flask_app.get("/")
    def index():
        return render_template("index.html")

    @flask_app.get("/api/state")
    def state():
        default_model = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
        if default_model not in DEEPSEEK_MODELS:
            default_model = "deepseek-v4-flash"
        profile_id = current_profile_id()
        return jsonify({
            "ok": True,
            "provider": PROVIDER_CAPABILITIES,
            "default_settings": AgentSettings(model=default_model).to_dict(),
            "conversations": storage.list_conversations(profile_id),
            "projects": storage.list_projects(profile_id),
            "profiles": storage.list_profiles(),
            "active_profile_id": profile_id,
            "presets": preset_manager.list(),
        })

    @flask_app.get("/api/rag/state")
    def rag_state():
        return jsonify({"ok": True, **rag_index.state()})

    @flask_app.post("/api/rag/index")
    def rag_build_index():
        try:
            return jsonify({"ok": True, "job": rag_index.start_build(json_body())}), 202
        except RuntimeError as error:
            return api_error(str(error), 409)
        except ValueError as error:
            return api_error(str(error), 400)

    @flask_app.get("/api/rag/chunks")
    def rag_chunks():
        try:
            strategy = str(request.args.get("strategy", "fixed"))
            page = int(request.args.get("page", 1))
            page_size = int(request.args.get("page_size", 20))
            source = str(request.args.get("source", ""))
            return jsonify({"ok": True, **rag_index.list_chunks(strategy, page, page_size, source)})
        except (TypeError, ValueError) as error:
            return api_error(str(error), 400)

    @flask_app.post("/api/rag/search")
    def rag_search():
        try:
            data = json_body()
            return jsonify({"ok": True, **rag_index.search(data.get("query", ""), data.get("top_k", 5))})
        except ValueError as error:
            return api_error(str(error), 400)
        except RuntimeError as error:
            return api_error(str(error), 409)

    @flask_app.get("/api/rag/evaluations")
    def rag_evaluation_state():
        return jsonify({
            "ok": True,
            "questions": list(known_rag_questions().values()),
            "items": rag_evaluations.list(),
        })

    @flask_app.post("/api/conversations/<conversation_id>/rag-compare")
    def compare_rag_answers(conversation_id: str):
        """Два изолированных LLM-вызова с одинаковыми настройками; историю не изменяет."""
        try:
            data = json_body()
            conversation = accessible_conversation(conversation_id)
            question = require_message(data.get("question"))
            settings = AgentSettings.from_dict(data.get("settings"))
            options = normalize_rag_options({
                "enabled": True,
                "strategy": data.get("strategy", "structural"),
                "top_k": data.get("top_k", 5),
            })
            retrieval = rag_index.retrieve(question, options["strategy"], options["top_k"])
            history, summary, facts, context_snapshot = request_history(conversation)
            profile_id = current_profile_id()
            memory_snapshot = memory_manager.context_snapshot(conversation, context_snapshot["mode"], profile_id)
            state_snapshot = deepcopy(ensure_task_state(conversation))
            state_snapshot["registered_artifacts"] = [
                public_artifact(item) for item in active_artifacts(conversation)
            ]
            handoff_snapshot = task_handoff_context(conversation)
            invariant_snapshot = invariant_context(conversation, profile_id)
            common = {
                "history": history,
                "user_text": question,
                "settings": settings,
                "summary": summary,
                "facts": facts,
                "memory_context": memory_snapshot,
                "task_handoff_context": handoff_snapshot,
                "task_state": state_snapshot,
            }
            guidance = generation_guidance(state_snapshot, invariant_snapshot)
            without_rag = chat_agent.reply(**common, policy_guidance=guidance)
            with_rag = chat_agent.reply(
                **common,
                policy_guidance=guidance + build_rag_context(retrieval),
            )
            control = known_rag_questions().get(str(data.get("question_id") or ""))
            item = rag_evaluations.add({
                "conversation_id": conversation_id,
                "question_id": (control or {}).get("id"),
                "question": question,
                "expectation": deepcopy((control or {}).get("expectation")),
                "expected_sources": deepcopy((control or {}).get("expected_sources", [])),
                "settings": settings.to_dict(),
                "retrieval": rag_snapshot(options, retrieval),
                "without_rag": {
                    "content": without_rag.content,
                    "reasoning_content": without_rag.reasoning_content,
                    "technical": without_rag.technical,
                },
                "with_rag": {
                    "content": with_rag.content,
                    "reasoning_content": with_rag.reasoning_content,
                    "technical": with_rag.technical,
                },
            })
            return jsonify({"ok": True, "evaluation": item})
        except (FileNotFoundError, PermissionError):
            return api_error("Диалог не найден.", 404)
        except (ValueError, RuntimeError, KeyError) as error:
            return api_error(str(error).strip("'"), 400)
        except Exception as error:
            logger.exception("Ошибка сравнения RAG | conversation_id=%s", conversation_id)
            return api_error(friendly_api_error(error), 502)

    @flask_app.post("/api/conversations/<conversation_id>/rag-pipeline-compare")
    def compare_rag_pipelines(conversation_id: str):
        """Сравнивает baseline, rewrite, filter, reranker и их совместное применение."""
        try:
            data = json_body()
            conversation = accessible_conversation(conversation_id)
            question = require_message(data.get("question"))
            settings = AgentSettings.from_dict(data.get("settings"))
            base_options = normalize_rag_options({
                "enabled": True,
                "strategy": data.get("strategy", "combined"),
                "mode": "baseline",
                "candidate_k": data.get("candidate_k", 20),
                "final_k": data.get("final_k", 5),
                "similarity_threshold": data.get("similarity_threshold", 0.83),
            })
            prepared_rewrite = rewritten_query(question, settings)
            history, summary, facts, context_snapshot = request_history(
                conversation, stage_scoped=task_control.is_enabled(),
            )
            profile_id = current_profile_id()
            memory_snapshot = memory_manager.context_snapshot(conversation, context_snapshot["mode"], profile_id)
            state_snapshot = deepcopy(ensure_task_state(conversation))
            state_snapshot["registered_artifacts"] = [
                public_artifact(item) for item in active_artifacts(conversation)
            ]
            task_enabled = task_control.is_enabled()
            common = {
                "history": history,
                "user_text": question,
                "settings": settings,
                "summary": summary,
                "facts": facts,
                "memory_context": memory_snapshot,
                "task_handoff_context": task_handoff_context(conversation) if task_enabled else {},
                "task_state": state_snapshot if task_enabled else {},
            }
            guidance = generation_guidance(state_snapshot, invariant_context(conversation, profile_id)) if task_enabled else ""
            results: dict[str, Any] = {}
            for mode in ("baseline", "rewrite", "filter", "rerank", "combined"):
                options = {**base_options, "mode": mode}
                retrieval = retrieve_rag(
                    question,
                    options,
                    settings,
                    prepared_rewrite=prepared_rewrite if mode in {"rewrite", "combined"} else None,
                )
                answer = chat_agent.reply(
                    **common,
                    policy_guidance=guidance + build_rag_context(retrieval),
                )
                results[mode] = {
                    "retrieval": rag_snapshot(options, retrieval),
                    "answer": {
                        "content": answer.content,
                        "reasoning_content": answer.reasoning_content,
                        "technical": answer.technical,
                    },
                }
            control = known_rag_questions().get(str(data.get("question_id") or ""))
            item = rag_evaluations.add({
                "evaluation_mode": "day23-pipeline",
                "conversation_id": conversation_id,
                "question_id": (control or {}).get("id"),
                "question": question,
                "expectation": deepcopy((control or {}).get("expectation")),
                "expected_sources": deepcopy((control or {}).get("expected_sources", [])),
                "settings": settings.to_dict(),
                "pipeline_settings": {
                    "strategy": base_options["strategy"],
                    "candidate_k": base_options["candidate_k"],
                    "final_k": base_options["final_k"],
                    "similarity_threshold": base_options["similarity_threshold"],
                },
                "rewrite": prepared_rewrite[1],
                "pipeline_results": results,
            })
            return jsonify({"ok": True, "evaluation": item})
        except (FileNotFoundError, PermissionError):
            return api_error("Диалог не найден.", 404)
        except (ValueError, RuntimeError, KeyError) as error:
            return api_error(str(error).strip("'"), 400)
        except Exception as error:
            logger.exception("Ошибка сравнения RAG-конвейеров | conversation_id=%s", conversation_id)
            return api_error(friendly_api_error(error), 502)

    @flask_app.post("/api/conversations/<conversation_id>/rag-evidence-check")
    def check_rag_evidence(conversation_id: str):
        """Один изолированный проверяемый RAG-ответ и независимая LLM-оценка цитат."""
        try:
            data = json_body()
            conversation = accessible_conversation(conversation_id)
            question = require_message(data.get("question"))
            settings = AgentSettings.from_dict(data.get("settings"))
            options = normalize_rag_options({
                "enabled": True,
                "verified": True,
                "strategy": data.get("strategy", "combined"),
                "mode": "combined",
                "candidate_k": data.get("candidate_k", 20),
                "final_k": data.get("final_k", 5),
                "similarity_threshold": data.get("similarity_threshold", 0.83),
            })
            retrieval = retrieve_verified_rag(question, options, settings)
            control = known_rag_questions().get(str(data.get("question_id") or ""))
            history, summary, facts, context_snapshot = request_history(
                conversation, stage_scoped=task_control.is_enabled(),
            )
            profile_id = current_profile_id()
            memory_snapshot = memory_manager.context_snapshot(
                conversation, context_snapshot["mode"], profile_id,
            )
            state_snapshot = deepcopy(ensure_task_state(conversation))
            state_snapshot["registered_artifacts"] = [
                public_artifact(item) for item in active_artifacts(conversation)
            ]
            task_enabled = task_control.is_enabled()
            guidance = (
                generation_guidance(state_snapshot, invariant_context(conversation, profile_id))
                if task_enabled else ""
            )
            rag_miss = not retrieval["chunks"]
            guidance += (
                "\n\nЛОКАЛЬНЫЙ RAG НЕ НАШЁЛ РЕЛЕВАНТНЫХ ФРАГМЕНТОВ:\n"
                "Ответь на вопрос на основе общих знаний. Не придумывай цитаты, chunk_id или локальные источники."
                if rag_miss else build_rag_context(retrieval) + build_evidence_guidance(retrieval)
            )
            draft = chat_agent.reply(
                history,
                question,
                settings,
                summary=summary,
                facts=facts,
                memory_context=memory_snapshot,
                task_handoff_context=task_handoff_context(conversation) if task_enabled else {},
                task_state=state_snapshot if task_enabled else {},
                policy_guidance=guidance,
            )
            answer = (
                rag_miss_fallback(draft, retrieval)
                if rag_miss else finalize_verified_rag(question, draft, retrieval, settings)
            )

            evidence = deepcopy(answer.technical.get("evidence") or {})
            judge_result = chat_agent.evaluate_rag_evidence(
                semantic_evaluation_prompt(
                    question,
                    answer.content,
                    evidence,
                    deepcopy((control or {}).get("expectation")),
                ),
                settings,
            )
            try:
                semantic = parse_semantic_evaluation(judge_result.content)
            except EvidenceValidationError as error:
                semantic = {
                    "meaning_supported": False,
                    "unsupported_claims": [],
                    "notes": f"Не удалось разобрать LLM-оценку: {error}",
                }
            semantic["technical"] = deepcopy(judge_result.technical)
            item = rag_evaluations.add({
                "evaluation_mode": "day24-evidence",
                "conversation_id": conversation_id,
                "question_id": (control or {}).get("id"),
                "question": question,
                "expectation": deepcopy((control or {}).get("expectation")),
                "expected_sources": deepcopy((control or {}).get("expected_sources", [])),
                "expected_behavior": deepcopy((control or {}).get("expected_behavior", "answer")),
                "settings": settings.to_dict(),
                "pipeline_settings": {
                    "strategy": options["strategy"],
                    "mode": options["mode"],
                    "candidate_k": options["candidate_k"],
                    "final_k": options["final_k"],
                    "similarity_threshold": options["similarity_threshold"],
                },
                "rewrite": deepcopy(retrieval.get("rewrite")),
                "retrieval": rag_snapshot(options, retrieval),
                "answer": {
                    "content": answer.content,
                    "reasoning_content": answer.reasoning_content,
                    "technical": answer.technical,
                },
                "evidence": evidence,
                "semantic_evaluation": semantic,
            })
            return jsonify({"ok": True, "evaluation": item})
        except (FileNotFoundError, PermissionError):
            return api_error("Диалог не найден.", 404)
        except (ValueError, RuntimeError, KeyError) as error:
            return api_error(str(error).strip("'"), 400)
        except Exception as error:
            logger.exception("Ошибка проверки цитат RAG | conversation_id=%s", conversation_id)
            return api_error(friendly_api_error(error), 502)

    @flask_app.get("/api/profiles")
    def list_profiles():
        return jsonify({
            "ok": True,
            "profiles": storage.list_profiles(),
            "active_profile_id": current_profile_id(),
        })

    @flask_app.post("/api/profiles")
    def create_profile():
        try:
            profile = memory_manager.create_profile(json_body())
            return jsonify({"ok": True, "profile": profile}), 201
        except ValueError as error:
            return api_error(str(error), 400)

    @flask_app.patch("/api/profiles/<profile_id>")
    def update_profile(profile_id: str):
        try:
            return jsonify({"ok": True, "profile": memory_manager.update_profile(profile_id, json_body())})
        except (FileNotFoundError, PermissionError):
            return api_error("Профиль не найден.", 404)
        except ValueError as error:
            return api_error(str(error), 400)

    @flask_app.patch("/api/profiles/active")
    def change_active_profile():
        try:
            profile = memory_manager.set_active_profile(str(json_body().get("profile_id", "")))
            return jsonify({"ok": True, "profile": profile})
        except FileNotFoundError:
            return api_error("Профиль не найден.", 404)

    @flask_app.get("/api/voice/status")
    def voice_status():
        return jsonify({"ok": True, "voice": whisper.status()})

    @flask_app.post("/api/voice/start")
    def voice_start():
        whisper.start()
        return jsonify({"ok": True, "voice": whisper.status()}), 202

    @flask_app.post("/api/voice/transcribe")
    def voice_transcribe():
        uploaded = request.files.get("audio")
        if uploaded is None:
            return api_error("Аудиозапись не передана.", 400)
        try:
            result = whisper.transcribe(
                uploaded.read(),
                uploaded.filename or "recording.webm",
                uploaded.mimetype or "application/octet-stream",
            )
            return jsonify({"ok": True, **result})
        except ValueError as error:
            return api_error(str(error), 400)
        except RuntimeError as error:
            return api_error(str(error), 503)

    @flask_app.get("/api/mcp/status")
    def mcp_status():
        return jsonify({"ok": True, "mcp": mcp.status()})

    @flask_app.get("/api/task-control")
    def task_control_status():
        return jsonify({"ok": True, "control": task_control.state()})

    @flask_app.post("/api/task-control/disable")
    def task_control_disable():
        control = task_control.set_enabled(False)
        logger.info("Машина задач принудительно отключена")
        return jsonify({"ok": True, "control": control})

    @flask_app.post("/api/task-control/enable")
    def task_control_enable():
        control = task_control.set_enabled(True)
        logger.info("Машина задач снова включена")
        return jsonify({"ok": True, "control": control})

    @flask_app.get("/api/mcp-control")
    def mcp_control_status():
        return jsonify({"ok": True, "control": mcp_control.state()})

    @flask_app.post("/api/mcp-control/disable")
    def mcp_control_disable():
        control = mcp_control.set_enabled(False)
        managers = {
            "filesystem": mcp,
            "open-meteo": weather_mcp,
            "scheduler": scheduler_mcp,
            "mediawiki": mediawiki_mcp,
            "worldbank": worldbank_mcp,
        }
        stopped: dict[str, dict[str, Any]] = {}
        errors: dict[str, str] = {}
        for name, manager in managers.items():
            try:
                stopped[name] = manager.stop()
            except RuntimeError as error:
                errors[name] = str(error)
                stopped[name] = manager.status()
        logger.info("Все MCP принудительно отключены | errors=%s", sorted(errors))
        return jsonify({
            "ok": True,
            "control": control,
            "servers": stopped,
            "errors": errors,
        })

    @flask_app.post("/api/mcp-control/enable")
    def mcp_control_enable():
        control = mcp_control.set_enabled(True)
        logger.info("Запуск MCP снова разрешён; серверы остаются остановленными")
        return jsonify({"ok": True, "control": control})

    @flask_app.post("/api/mcp/start")
    def mcp_start():
        if not mcp_control.is_enabled():
            return mcp_blocked_response()
        try:
            return jsonify({"ok": True, "mcp": mcp.start()})
        except RuntimeError as error:
            logger.warning("MCP не запущен | reason=%s", error)
            return jsonify({"ok": False, "error": str(error), "mcp": mcp.status()}), 503

    @flask_app.get("/api/mcp/tools")
    def mcp_tools():
        if not mcp_control.is_enabled():
            return mcp_blocked_response()
        try:
            return jsonify({"ok": True, "mcp": mcp.list_tools()})
        except RuntimeError as error:
            logger.warning("Не удалось получить MCP-инструменты | reason=%s", error)
            return jsonify({"ok": False, "error": str(error), "mcp": mcp.status()}), 409

    @flask_app.post("/api/mcp/stop")
    def mcp_stop():
        try:
            return jsonify({"ok": True, "mcp": mcp.stop()})
        except RuntimeError as error:
            logger.warning("MCP не остановлен штатно | reason=%s", error)
            return jsonify({"ok": False, "error": str(error), "mcp": mcp.status()}), 503

    @flask_app.get("/api/weather-mcp/status")
    def weather_mcp_status():
        return jsonify({"ok": True, "mcp": weather_mcp.status()})

    @flask_app.post("/api/weather-mcp/start")
    def weather_mcp_start():
        if not mcp_control.is_enabled():
            return mcp_blocked_response()
        try:
            status = weather_mcp.start()
            tools = weather_mcp.list_tools()
            return jsonify({"ok": True, "mcp": {**status, **tools}})
        except RuntimeError as error:
            logger.warning("Open-Meteo MCP не запущен: %s", error)
            return jsonify({"ok": False, "error": str(error), "mcp": weather_mcp.status()}), 503

    @flask_app.get("/api/weather-mcp/tools")
    def weather_mcp_tools():
        if not mcp_control.is_enabled():
            return mcp_blocked_response()
        try:
            return jsonify({"ok": True, "mcp": weather_mcp.list_tools()})
        except RuntimeError as error:
            return jsonify({"ok": False, "error": str(error), "mcp": weather_mcp.status()}), 409

    @flask_app.post("/api/weather-mcp/stop")
    def weather_mcp_stop():
        try:
            return jsonify({"ok": True, "mcp": weather_mcp.stop()})
        except RuntimeError as error:
            return jsonify({"ok": False, "error": str(error), "mcp": weather_mcp.status()}), 503

    orchestration_managers = {
        "mediawiki": mediawiki_mcp,
        "worldbank": worldbank_mcp,
    }

    @flask_app.get("/api/orchestration-mcp/<server_name>/status")
    def orchestration_mcp_status(server_name: str):
        manager = orchestration_managers.get(server_name)
        if manager is None:
            return api_error("Неизвестный MCP-сервер.", 404)
        return jsonify({"ok": True, "mcp": manager.status()})

    @flask_app.route("/api/orchestration-mcp/<server_name>/<action>", methods=["GET", "POST"])
    def orchestration_mcp_action(server_name: str, action: str):
        manager = orchestration_managers.get(server_name)
        if manager is None or action not in {"start", "tools", "stop"}:
            return api_error("Неизвестная операция MCP.", 404)
        try:
            if action == "start":
                if not mcp_control.is_enabled():
                    return mcp_blocked_response()
                status = manager.start()
                tools = manager.list_tools()
                return jsonify({"ok": True, "mcp": {**status, **tools}})
            if action == "tools":
                if not mcp_control.is_enabled():
                    return mcp_blocked_response()
                return jsonify({"ok": True, "mcp": manager.list_tools()})
            return jsonify({"ok": True, "mcp": manager.stop()})
        except RuntimeError as error:
            status_code = 409 if action == "tools" else 503
            return jsonify({"ok": False, "error": str(error), "mcp": manager.status()}), status_code

    @flask_app.get("/api/mcp/activity/<activity_id>")
    def mcp_activity(activity_id: str):
        if not re.fullmatch(r"[0-9a-f]{32}", activity_id):
            return api_error("Некорректный идентификатор выполнения.", 400)
        with mcp_activity_lock:
            activity = deepcopy(mcp_activities.get(activity_id))
        if activity is None:
            return api_error("Выполнение ещё не зарегистрировано.", 404)
        return jsonify({"ok": True, "activity": activity})

    def owned_scheduled_task(task_id: str) -> dict[str, Any]:
        task = scheduler_repository.get_task(task_id)
        if task.get("owner_profile_id") != current_profile_id():
            raise PermissionError("Запланированное задание принадлежит другому профилю.")
        return task

    @flask_app.get("/api/scheduler/state")
    def scheduler_state():
        profile_id = current_profile_id()
        return jsonify({
            "ok": True,
            "scheduler": {
                "running": scheduler.status()["running"],
                "timezone": TIMEZONE_NAME,
            },
            "mcp": scheduler_mcp.status(),
            "tasks": scheduler_repository.list_tasks(profile_id),
            "runs": [
                run for run in scheduler_repository.list_runs(limit=100)
                if scheduler_repository.get_task(run["task_id"]).get("owner_profile_id") == profile_id
            ],
            "summary": scheduler_repository.summary(profile_id),
        })

    @flask_app.get("/api/scheduler/tools")
    def scheduler_tools():
        if not mcp_control.is_enabled():
            return mcp_blocked_response()
        try:
            return jsonify({"ok": True, "mcp": scheduler_mcp.list_tools()})
        except RuntimeError as error:
            return jsonify({"ok": False, "error": str(error), "mcp": scheduler_mcp.status()}), 409

    @flask_app.post("/api/scheduler/tasks")
    def create_scheduled_task_api():
        data = json_body()
        project_id = str(data.get("project_id") or "") or None
        try:
            if project_id:
                accessible_project(project_id)
            settings = AgentSettings.from_dict(data.get("settings")).to_dict()
            task = scheduler_repository.create_task(
                title=data.get("title", "Автоматизация"),
                prompt=data.get("prompt", ""),
                schedule_type=data.get("schedule_type", "once"),
                run_at=data.get("run_at"),
                delay_minutes=data.get("delay_minutes"),
                interval_minutes=data.get("interval_minutes"),
                mcp_servers=data.get("mcp_servers") or [],
                allowed_tools=data.get("allowed_tools") or [],
                owner_profile_id=current_profile_id(),
                source_conversation_id=data.get("source_conversation_id"),
                project_id=project_id,
                settings=settings,
            )
            try:
                conversation = storage.create_conversation(
                    f"Автоматизация · {task['title']}", project_id, current_profile_id()
                )
                task = scheduler_repository.attach_context(
                    task["id"], owner_profile_id=current_profile_id(),
                    conversation_id=conversation["id"],
                    source_conversation_id=data.get("source_conversation_id"),
                    project_id=project_id, settings=settings,
                )
            except Exception:
                scheduler_repository.delete_task(task["id"])
                raise
            return jsonify({"ok": True, "task": task, "conversation": with_token_totals(conversation)}), 201
        except (ValueError, KeyError) as error:
            return api_error(str(error).strip("'"), 400)
        except (FileNotFoundError, PermissionError):
            return api_error("Проект не найден или недоступен.", 404)

    @flask_app.patch("/api/scheduler/tasks/<task_id>")
    def update_scheduled_task_api(task_id: str):
        try:
            current = owned_scheduled_task(task_id)
            data = json_body()
            task = scheduler_repository.update_task(task_id, **data)
            if data.get("title") and current.get("conversation_id"):
                storage.rename_conversation(current["conversation_id"], f"Автоматизация · {task['title']}")
            return jsonify({"ok": True, "task": task})
        except FileNotFoundError:
            return api_error("Запланированное задание не найдено.", 404)
        except PermissionError:
            return api_error("Нет доступа к запланированному заданию.", 403)
        except (ValueError, KeyError) as error:
            return api_error(str(error).strip("'"), 400)

    @flask_app.post("/api/scheduler/tasks/<task_id>/<action>")
    def scheduled_task_action(task_id: str, action: str):
        try:
            owned_scheduled_task(task_id)
            if action == "pause":
                task = scheduler_repository.set_status(task_id, "paused")
            elif action == "resume":
                task = scheduler_repository.set_status(task_id, "enabled")
            elif action == "run-now":
                task = scheduler_repository.request_run_now(task_id)
            else:
                return api_error("Неизвестное действие расписания.", 404)
            return jsonify({"ok": True, "task": task})
        except FileNotFoundError:
            return api_error("Запланированное задание не найдено.", 404)
        except PermissionError:
            return api_error("Нет доступа к запланированному заданию.", 403)
        except ValueError as error:
            return api_error(str(error), 409)

    @flask_app.delete("/api/scheduler/tasks/<task_id>")
    def delete_scheduled_task_api(task_id: str):
        try:
            owned_scheduled_task(task_id)
            scheduler_repository.delete_task(task_id)
            return jsonify({"ok": True})
        except FileNotFoundError:
            return api_error("Запланированное задание не найдено.", 404)
        except PermissionError:
            return api_error("Нет доступа к запланированному заданию.", 403)
        except ValueError as error:
            return api_error(str(error), 409)

    @flask_app.get("/api/scheduler/tasks/<task_id>/runs")
    def scheduled_task_runs(task_id: str):
        try:
            owned_scheduled_task(task_id)
            return jsonify({"ok": True, "runs": scheduler_repository.list_runs(task_id, 100)})
        except FileNotFoundError:
            return api_error("Запланированное задание не найдено.", 404)
        except PermissionError:
            return api_error("Нет доступа к запланированному заданию.", 403)

    @flask_app.post("/api/scheduler/notifications/read")
    def read_scheduler_notifications():
        count = scheduler_repository.mark_notifications_read(current_profile_id())
        return jsonify({"ok": True, "read_count": count, "summary": scheduler_repository.summary(current_profile_id())})

    @flask_app.post("/api/conversations")
    def create_conversation():
        try:
            data = json_body()
            project_id = data.get("project_id")
            if project_id:
                accessible_project(str(project_id))
            conversation = storage.create_conversation(
                data.get("title", "Новый диалог"), str(project_id) if project_id else None,
                current_profile_id(),
            )
            return jsonify({"ok": True, "conversation": with_token_totals(conversation)}), 201
        except (FileNotFoundError, PermissionError):
            return api_error("Проект не найден.", 404)
        except ValueError as error:
            return api_error(str(error), 400)

    @flask_app.get("/api/conversations/<conversation_id>")
    def get_conversation(conversation_id: str):
        try:
            return jsonify({"ok": True, "conversation": with_token_totals(accessible_conversation(conversation_id))})
        except (ValueError, FileNotFoundError, PermissionError):
            return api_error("Диалог не найден.", 404)

    @flask_app.patch("/api/conversations/<conversation_id>")
    def rename_conversation(conversation_id: str):
        try:
            accessible_conversation(conversation_id)
            conversation = storage.rename_conversation(conversation_id, json_body().get("title", ""))
            return jsonify({"ok": True, "conversation": with_token_totals(conversation)})
        except (FileNotFoundError, PermissionError):
            return api_error("Диалог не найден.", 404)
        except ValueError as error:
            return api_error(str(error), 400)

    @flask_app.patch("/api/conversations/<conversation_id>/task-state")
    def update_conversation_task_state(conversation_id: str):
        if not task_control.is_enabled():
            return task_control_blocked_response()
        try:
            conversation = accessible_conversation(conversation_id)
            changes = json_body()
            update_task_state(conversation, changes)
            conversation = storage.save_conversation(conversation)
            return jsonify({"ok": True, "conversation": with_token_totals(conversation)})
        except (FileNotFoundError, PermissionError):
            return api_error("Диалог не найден.", 404)
        except ValueError as error:
            return api_error(str(error), 400)
        except Exception as error:
            logger.exception("Ошибка формирования handoff при ручной смене этапа | conversation_id=%s", conversation_id)
            return api_error(friendly_api_error(error), 502)

    @flask_app.get("/api/conversations/<conversation_id>/task-state")
    def get_conversation_task_state(conversation_id: str):
        """Читает авторитетное состояние без обращения к модели."""
        try:
            conversation = accessible_conversation(conversation_id)
            state = ensure_task_state(conversation)
            return jsonify({
                "ok": True,
                "task_state": {**deepcopy(state), "allowed_events": allowed_task_events(state)},
                "report": task_state_report(state),
            })
        except (ValueError, FileNotFoundError, PermissionError):
            return api_error("Диалог не найден.", 404)

    @flask_app.patch("/api/conversations/<conversation_id>/task-memory")
    def update_conversation_task_memory(conversation_id: str):
        """Включает или выключает независимую память текущей задачи."""
        try:
            conversation = accessible_conversation(conversation_id)
            memory = update_task_memory_settings(conversation, json_body())
            conversation = storage.save_conversation(conversation)
            return jsonify({
                "ok": True,
                "task_memory": deepcopy(memory),
                "conversation": with_token_totals(conversation),
            })
        except (FileNotFoundError, PermissionError):
            return api_error("Диалог не найден.", 404)
        except ValueError as error:
            return api_error(str(error), 400)

    @flask_app.post("/api/conversations/<conversation_id>/task-state/events")
    def apply_conversation_task_event(conversation_id: str):
        """Применяет событие только через серверный граф переходов."""
        if not task_control.is_enabled():
            return task_control_blocked_response()
        try:
            conversation = accessible_conversation(conversation_id)
            data = json_body()
            event = data.get("event")
            automatic = data.get("automatic") is True
            if automatic and not automatic_event_is_authorized(conversation, str(event)):
                raise TaskTransitionError("Автопереход не подтверждён последним проверенным ответом текущего этапа.")
            assert_artifact_transition(conversation, str(event))
            preview = deepcopy(conversation)
            before = deepcopy(ensure_task_state(conversation))
            apply_task_event(
                preview,
                event,
                source="autopilot" if automatic else "ui",
                reason=data.get("reason", ""),
            )
            target = ensure_task_state(preview)
            handoff = None
            if before["stage"] != target["stage"]:
                handoff = build_stage_handoff(
                    conversation, chat_agent, target_stage=target["stage"], event=str(event),
                )
            if handoff:
                activate_stage_handoff(preview, handoff)
            if ensure_task_state(preview)["stage"] == "done":
                append_completion_result(preview)
            conversation = storage.save_conversation(preview)
            return jsonify({"ok": True, "conversation": with_token_totals(conversation)})
        except (FileNotFoundError, PermissionError):
            return api_error("Диалог не найден.", 404)
        except TaskTransitionError as error:
            payload = {"ok": False, "error": str(error)}
            if "conversation" in locals():
                payload["conversation"] = with_token_totals(conversation)
            return jsonify(payload), 409
        except ValueError as error:
            return api_error(str(error), 400)
        except Exception as error:
            logger.exception("Ошибка handoff или финального ответа при переходе | conversation_id=%s", conversation_id)
            return api_error(friendly_api_error(error), 502)

    @flask_app.get("/api/projects")
    def list_projects():
        return jsonify({"ok": True, "projects": storage.list_projects(current_profile_id())})

    @flask_app.post("/api/projects")
    def create_project():
        try:
            data = json_body()
            project = memory_manager.create_project(data.get("name"), data.get("description", ""), current_profile_id())
            payload: dict[str, Any] = {"ok": True, "project": project}
            if data.get("create_dialog") is True:
                payload["conversation"] = with_token_totals(storage.create_conversation("Новый диалог", project["id"], current_profile_id()))
            return jsonify(payload), 201
        except ValueError as error:
            return api_error(str(error), 400)

    @flask_app.get("/api/projects/<project_id>")
    def get_project(project_id: str):
        try:
            project = accessible_project(project_id)
            project["conversations"] = [item for item in storage.list_conversations(current_profile_id()) if item.get("project_id") == project_id]
            return jsonify({"ok": True, "project": project})
        except (ValueError, FileNotFoundError, PermissionError):
            return api_error("Проект не найден.", 404)

    @flask_app.patch("/api/projects/<project_id>")
    def update_project(project_id: str):
        try:
            accessible_project(project_id, owner_only=True)
            return jsonify({"ok": True, "project": memory_manager.update_project(project_id, json_body())})
        except (FileNotFoundError, PermissionError):
            return api_error("Проект не найден.", 404)
        except ValueError as error:
            return api_error(str(error), 400)

    @flask_app.patch("/api/conversations/<conversation_id>/project")
    def set_conversation_project(conversation_id: str):
        try:
            conversation = accessible_conversation(conversation_id)
            project_id = json_body().get("project_id")
            if project_id is not None:
                accessible_project(str(project_id))
            conversation["project_id"] = project_id
            conversation = storage.save_conversation(conversation)
            return jsonify({"ok": True, "conversation": with_token_totals(conversation)})
        except (ValueError, FileNotFoundError, PermissionError):
            return api_error("Диалог или проект не найден.", 404)

    @flask_app.post("/api/conversations/<conversation_id>/copy-to-project")
    def copy_conversation_to_project(conversation_id: str):
        try:
            source = accessible_conversation(conversation_id)
            project_id = str(json_body().get("project_id", ""))
            accessible_project(project_id)
            copied = storage.copy_conversation_to_project(conversation_id, project_id)
            copy_artifact_files(storage.data_dir, source, copied)
            copied = storage.save_conversation(copied)
            return jsonify({"ok": True, "conversation": with_token_totals(copied)}), 201
        except (ValueError, FileNotFoundError, PermissionError):
            return api_error("Диалог или проект не найден.", 404)

    @flask_app.post("/api/conversations/<conversation_id>/extract-project-memory")
    def extract_project_memory_from_history(conversation_id: str):
        try:
            conversation = accessible_conversation(conversation_id)
            if task_is_paused(conversation):
                return paused_task_response(conversation)
            if not conversation.get("project_id"):
                raise ValueError("Сначала подключите диалог к проекту.")
            path = active_path_messages(conversation)
            exchanges = completed_exchanges(path)
            successful_messages = [
                deepcopy(message)
                for exchange in exchanges
                for message in exchange.get("messages", [])
            ]
            extraction_id = uuid.uuid4().hex
            result = chat_agent.extract_project_memories_from_history(successful_messages)
            revision = memory_manager.route_candidates(
                result, conversation, extraction_id, force_scopes={"project"},
            )
            revision["purpose"] = "project_history_extraction"
            revision["exchange_count"] = len(exchanges)
            conversation.setdefault("memory_extraction_revisions", []).append(revision)
            conversation = storage.save_conversation(conversation)
            if revision.get("status") != "completed":
                return api_error("Не удалось извлечь память из истории диалога.", 502)
            return jsonify({
                "ok": True, "revision": revision,
                "conversation": with_token_totals(conversation),
                "project": accessible_project(conversation["project_id"]),
            })
        except (FileNotFoundError, PermissionError):
            return api_error("Диалог или проект не найден.", 404)
        except ValueError as error:
            return api_error(str(error), 400)
        except Exception as error:
            logger.warning(
                "Ошибка явного извлечения памяти из истории | conversation_id=%s | error=%s",
                conversation_id, type(error).__name__, exc_info=True,
            )
            return api_error("Не удалось извлечь память из истории диалога.", 502)

    @flask_app.get("/api/projects/<project_id>/memory")
    def get_project_memory(project_id: str):
        try:
            accessible_project(project_id)
            return jsonify({"ok": True, **memory_manager.list_entries("project", project_id)})
        except (ValueError, FileNotFoundError, PermissionError):
            return api_error("Проект не найден.", 404)

    @flask_app.post("/api/projects/<project_id>/memory")
    def create_project_memory(project_id: str):
        try:
            accessible_project(project_id)
            entry = memory_manager.add_manual("project", json_body(), project_id)
            return jsonify({"ok": True, "memory": entry}), 201
        except (FileNotFoundError, PermissionError):
            return api_error("Проект не найден.", 404)
        except ValueError as error:
            return api_error(str(error), 400)

    @flask_app.put("/api/projects/<project_id>/memory/<memory_id>")
    def update_project_memory(project_id: str, memory_id: str):
        try:
            accessible_project(project_id)
            return jsonify({"ok": True, "memory": memory_manager.update_entry("project", memory_id, json_body(), project_id)})
        except (FileNotFoundError, KeyError, PermissionError):
            return api_error("Проект или запись памяти не найдены.", 404)
        except ValueError as error:
            return api_error(str(error), 400)

    @flask_app.delete("/api/projects/<project_id>/memory/<memory_id>")
    def delete_project_memory(project_id: str, memory_id: str):
        try:
            accessible_project(project_id)
            memory_manager.delete_entry("project", memory_id, project_id)
            return jsonify({"ok": True})
        except (FileNotFoundError, KeyError, PermissionError):
            return api_error("Проект или запись памяти не найдены.", 404)

    @flask_app.get("/api/memory/user")
    def get_user_memory():
        return jsonify({"ok": True, **memory_manager.list_entries("user", profile_id=current_profile_id())})

    @flask_app.post("/api/memory/user")
    def create_user_memory():
        try:
            data = json_body()
            if set(data) == {"automatic_extraction"}:
                return jsonify({"ok": True, "document": memory_manager.set_user_extraction(data["automatic_extraction"], current_profile_id())})
            return jsonify({"ok": True, "memory": memory_manager.add_manual("user", data, profile_id=current_profile_id())}), 201
        except ValueError as error:
            return api_error(str(error), 400)

    @flask_app.put("/api/memory/user/<memory_id>")
    def update_user_memory(memory_id: str):
        try:
            return jsonify({"ok": True, "memory": memory_manager.update_entry("user", memory_id, json_body(), profile_id=current_profile_id())})
        except KeyError:
            return api_error("Запись памяти не найдена.", 404)
        except ValueError as error:
            return api_error(str(error), 400)

    @flask_app.delete("/api/memory/user/<memory_id>")
    def delete_user_memory(memory_id: str):
        try:
            memory_manager.delete_entry("user", memory_id, profile_id=current_profile_id())
            return jsonify({"ok": True})
        except KeyError:
            return api_error("Запись памяти не найдена.", 404)

    @flask_app.get("/api/memory/policy")
    def get_memory_policy():
        return jsonify({"ok": True, "policy": memory_manager.policy()})

    def authorize_invariant_owner(scope: str, owner_id: str) -> None:
        if scope == "user":
            if owner_id != current_profile_id():
                raise PermissionError(owner_id)
        elif scope == "project":
            accessible_project(owner_id, owner_only=True)
        elif scope == "task":
            accessible_conversation(owner_id)
        else:
            raise ValueError("Область инвариантов должна быть user, project или task.")

    @flask_app.get("/api/invariants/context/<conversation_id>")
    def get_invariant_context(conversation_id: str):
        try:
            conversation = accessible_conversation(conversation_id)
            profile_id = current_profile_id()
            bundle = invariant_context(conversation, profile_id)
            editable_sets = {
                "user": invariant_store.list("user", profile_id),
                "project": invariant_store.list("project", conversation.get("project_id")) if conversation.get("project_id") else [],
                "task": invariant_store.list("task", conversation_id),
            }
            return jsonify({"ok": True, "bundle": bundle, "sets": editable_sets})
        except (ValueError, FileNotFoundError, PermissionError):
            return api_error("Диалог не найден.", 404)

    @flask_app.post("/api/invariants")
    def create_invariant_set():
        try:
            data = json_body()
            authorize_invariant_owner(str(data.get("scope", "")), str(data.get("owner_id", "")))
            return jsonify({"ok": True, "invariant_set": invariant_store.create(data)}), 201
        except PermissionError:
            return api_error("Недостаточно прав для этой области инвариантов.", 403)
        except (ValueError, FileNotFoundError) as error:
            return api_error(str(error), 400)

    @flask_app.put("/api/invariants/<set_id>")
    def update_invariant_set(set_id: str):
        try:
            existing = invariant_store.get(set_id)
            if existing is None:
                raise FileNotFoundError(set_id)
            authorize_invariant_owner(existing["scope"], existing["owner_id"])
            data = json_body()
            data.update({"scope": existing["scope"], "owner_id": existing["owner_id"]})
            return jsonify({"ok": True, "invariant_set": invariant_store.update(set_id, data)})
        except PermissionError:
            return api_error("Недостаточно прав для этой области инвариантов.", 403)
        except FileNotFoundError:
            return api_error("Набор инвариантов не найден.", 404)
        except ValueError as error:
            return api_error(str(error), 400)

    @flask_app.delete("/api/invariants/<set_id>")
    def delete_invariant_set(set_id: str):
        try:
            existing = invariant_store.get(set_id)
            if existing is None:
                raise FileNotFoundError(set_id)
            authorize_invariant_owner(existing["scope"], existing["owner_id"])
            invariant_store.delete(set_id)
            return jsonify({"ok": True})
        except PermissionError:
            return api_error("Недостаточно прав для этой области инвариантов.", 403)
        except FileNotFoundError:
            return api_error("Набор инвариантов не найден.", 404)

    @flask_app.get("/api/conversations/<conversation_id>/memory-context")
    def get_memory_context(conversation_id: str):
        try:
            conversation = accessible_conversation(conversation_id)
            snapshots = [
                {"message_id": item.get("id"), "created_at": item.get("created_at"), "memory_context": item.get("technical", {}).get("memory_context")}
                for item in conversation.get("messages", [])
                if item.get("role") == "assistant" and item.get("technical", {}).get("memory_context")
            ]
            return jsonify({"ok": True, "snapshots": snapshots})
        except (ValueError, FileNotFoundError, PermissionError):
            return api_error("Диалог не найден.", 404)

    @flask_app.patch("/api/conversations/<conversation_id>/context")
    def update_conversation_context(conversation_id: str):
        try:
            conversation = accessible_conversation(conversation_id)
            previous_mode = ensure_context_management(conversation)["mode"]
            data = json_body()
            mode = data.get("mode")
            set_context_mode(conversation, mode)
            set_window_sizes(
                conversation,
                sliding=data.get("sliding_window_exchanges"),
                facts=data.get("facts_window_exchanges"),
            )
            if previous_mode != mode:
                labels = {
                    "full": "Полная история",
                    "summary": "Summary + последние обмены",
                    "sliding": "Sliding Window",
                    "facts": "Sticky Facts",
                }
                conversation["messages"].append({
                    "id": uuid.uuid4().hex,
                    "role": "event",
                    "parent_id": conversation["context_management"].get("active_leaf_id"),
                    "content": f"Режим контекста: {labels[mode]}.",
                    "created_at": now_iso(),
                })
            conversation = storage.save_conversation(conversation)
            return jsonify({"ok": True, "conversation": with_token_totals(conversation)})
        except (FileNotFoundError, PermissionError):
            return api_error("Диалог не найден.", 404)
        except ValueError as error:
            return api_error(str(error), 400)

    @flask_app.delete("/api/conversations/<conversation_id>")
    def delete_conversation(conversation_id: str):
        try:
            accessible_conversation(conversation_id)
            delete_artifact_files(storage.data_dir, conversation_id)
            storage.delete_conversation(conversation_id)
            return jsonify({"ok": True})
        except (ValueError, FileNotFoundError, PermissionError):
            return api_error("Диалог не найден.", 404)

    @flask_app.post("/api/conversations/<conversation_id>/messages")
    def send_message(conversation_id: str):
        try:
            data = json_body()
            mcp_activity_id = begin_mcp_activity(str(data.get("mcp_activity_id") or "") or None)
            settings = AgentSettings.from_dict(data.get("settings"))
            rag_options = normalize_rag_options(data.get("rag"))
            source = configuration_source(data.get("preset_id"), settings, preset_manager)
            conversation = accessible_conversation(conversation_id)
            automatic = data.get("automatic") is True
            task_control_enabled = task_control.is_enabled()
            state = ensure_task_state(conversation)
            if automatic:
                if not task_control_enabled:
                    raise TaskTransitionError("Машина задач отключена; автоматическое продолжение запрещено.")
                if state.get("transition_mode") != "automatic":
                    raise TaskTransitionError("Автопилот остановлен или переключён в ручной режим.")
                content = automatic_continuation_text(state)
            else:
                content = require_message(data.get("content"))
            outside_rag = bool(rag_options["enabled"] and requests_outside_rag(content))
            rag_options["outside_knowledge_requested"] = outside_rag
            if content.strip().lower() in {"/state", "/task-state"}:
                conversation = append_task_state_report(conversation, content)
                return jsonify({"ok": True, "conversation": with_token_totals(conversation)})
            if task_control_enabled and task_is_paused(conversation):
                return paused_task_response(conversation)
        except (FileNotFoundError, PermissionError):
            return api_error("Диалог не найден.", 404)
        except TaskTransitionError as error:
            return api_error(str(error), 409)
        except (ValueError, KeyError) as error:
            return api_error(str(error).strip("'"), 400)

        exchange_id = uuid.uuid4().hex
        created_at = now_iso()
        settings_snapshot = settings.to_dict()
        ensure_context_management(conversation)
        previous_source = last_configuration(active_path_messages(conversation))
        if previous_source is not None and (
            previous_source.get("settings") != settings_snapshot
            or previous_source.get("configuration_source") != source
        ):
            conversation["messages"].append({
                "id": uuid.uuid4().hex,
                "role": "event",
                "parent_id": conversation["context_management"].get("active_leaf_id"),
                "content": configuration_event_text(source),
                "created_at": created_at,
            })

        user_message = {
            "id": uuid.uuid4().hex,
            "exchange_id": exchange_id,
            "role": "user",
            "content": content,
            "created_at": created_at,
            "author_profile_id": current_profile_id(),
            "technical": {
                "request_status": "pending",
                "configuration_source": source,
                "settings": settings_snapshot,
                "rag": rag_snapshot(rag_options),
                "automatic_continuation": automatic,
                "scheduled_automation": deepcopy(_automation_run_override.get()),
                "task_control": {"enabled": task_control_enabled},
            },
        }
        append_tree_message(conversation, user_message)
        if conversation.get("title") == "Новый диалог":
            conversation["title"] = content.replace("\n", " ")[:60]
        conversation = storage.save_conversation(conversation)
        logger.info(
            "Сообщение принято | conversation_id=%s | exchange_id=%s | model=%s | mode=%s | chars=%s",
            conversation_id, exchange_id, settings.model,
            conversation["context_management"]["mode"], len(content),
        )

        try:
            auto_routing = bool(
                rag_options["enabled"] and rag_options.get("routing_mode") == "auto"
            )
            automatic_document_query = (
                automatic_task_document_query(conversation, state)
                if automatic and auto_routing and task_control_enabled
                else ""
            )
            retrieval_question = automatic_document_query or content
            retrieval_context = (
                conversational_retrieval_context(
                    conversation, task_enabled=task_control_enabled,
                )
                if rag_options["enabled"] and (
                    auto_routing or rag_options.get("mode") in {"rewrite", "combined"}
                )
                else None
            )
            retrieval = None
            route_info: dict[str, Any] = {
                "route": "strict" if rag_options["enabled"] else "disabled",
                "confidence": 1.0,
                "reason": "manual_mode",
                "classifier": "manual",
            }
            if auto_routing:
                probe = rag_index.retrieve(
                    retrieval_question, rag_options["strategy"], rag_options["final_k"],
                )
                probe["route_probe_only"] = True
                probe["candidate_count"] = len(probe.get("chunks") or [])
                probe["after_filter_count"] = len(probe.get("chunks") or [])
                probe["max_similarity"] = max(
                    (float(item.get("score", 0)) for item in probe.get("chunks") or []),
                    default=None,
                )
                if outside_rag:
                    route_info = {
                        "route": "general", "confidence": 1.0,
                        "reason": "explicit_outside_rag", "classifier": "explicit",
                    }
                    retrieval = probe
                elif automatic_document_query:
                    route_info = {
                        "route": "rag",
                        "confidence": 1.0,
                        "reason": "automatic_task_requires_documents",
                        "classifier": "task_policy",
                    }
                    retrieval = (
                        retrieve_verified_rag(
                            retrieval_question, rag_options, settings, retrieval_context,
                        )
                        if rag_options["verified"]
                        else retrieve_rag(
                            retrieval_question, rag_options, settings,
                            retrieval_context=retrieval_context,
                        )
                    )
                else:
                    route_info = resolve_auto_route(
                        content, settings, retrieval_context or {}, probe,
                    )
                    selected_route = route_info["route"]
                    if selected_route in {"rag", "hybrid"}:
                        retrieval = (
                            retrieve_verified_rag(
                                content, rag_options, settings, retrieval_context,
                            )
                            if rag_options["verified"] and selected_route == "rag"
                            else retrieve_rag(
                                content, rag_options, settings,
                                retrieval_context=retrieval_context,
                            )
                        )
                    else:
                        retrieval = probe
            elif rag_options["enabled"] and not outside_rag:
                retrieval = (
                    retrieve_verified_rag(
                        content, rag_options, settings,
                        retrieval_context,
                    )
                    if rag_options["verified"]
                    else retrieve_rag(
                        content, rag_options, settings,
                        retrieval_context=retrieval_context,
                    )
                )
            selected_route = str(route_info.get("route") or "strict")
            document_route = selected_route in {"rag", "hybrid", "strict"}
            strict_evidence = bool(
                rag_options["enabled"] and not outside_rag and rag_options["verified"]
                and selected_route in {"rag", "strict"}
            )
            if retrieval is not None:
                retrieval["automatic_routing"] = deepcopy(route_info)
                if document_route and is_synthesis_question(content):
                    ledger_ids = evidence_ledger_ids(conversation)
                    if ledger_ids:
                        retrieval = rag_index.augment_with_chunk_ids(
                            content, retrieval or {}, ledger_ids,
                            final_k=rag_options["final_k"],
                        )
            request_rag_snapshot = rag_snapshot(
                rag_options, retrieval, include_chunk_text=False,
            )
            request_rag_snapshot["routing"] = deepcopy(route_info)
            request_rag_snapshot["chunks_used_for_answer"] = bool(
                document_route and not outside_rag and (retrieval or {}).get("chunks")
            )
            request_rag_snapshot["route_probe_only"] = bool((retrieval or {}).get("route_probe_only"))
            request_rag_snapshot["automatic_task_document_context"] = bool(automatic_document_query)
            request_rag_snapshot["retrieval_query_source"] = (
                "task_state_handoff" if automatic_document_query else "message"
            )
            local_refusal = bool(
                strict_evidence
                and not (retrieval or {}).get("chunks")
            )
            update_sticky_facts(conversation, chat_agent, content, stage_scoped=task_control_enabled)
            update_summaries(conversation, chat_agent, stage_scoped=task_control_enabled)
            history_for_request, active_summary_text, active_facts, context_snapshot = request_history(
                conversation, stage_scoped=task_control_enabled,
            )
            history_for_request = sanitize_history_for_generation(history_for_request)
            active_summary_text = strip_rag_evidence_appendix(active_summary_text)
            profile_id = user_message["author_profile_id"]
            memory_snapshot = memory_manager.context_snapshot(conversation, context_snapshot["mode"], profile_id)
            task_memory_snapshot = task_memory_context(conversation)
            task_state_snapshot = deepcopy(ensure_task_state(conversation))
            task_state_snapshot["registered_artifacts"] = [
                public_artifact(item) for item in active_artifacts(conversation)
            ]
            handoff_snapshot = task_handoff_context(conversation)
            invariant_snapshot = invariant_context(conversation, profile_id)
            conversation = storage.save_conversation(conversation)
            if local_refusal:
                result = controlled_agent_reply(
                    history=history_for_request,
                    user_text=content,
                    settings=settings,
                    summary=active_summary_text,
                    facts=active_facts,
                    memory_context=memory_snapshot,
                    task_memory_snapshot=task_memory_snapshot,
                    task_state=task_state_snapshot,
                    handoff_context=handoff_snapshot,
                    invariants=invariant_snapshot,
                    rag_context=(
                        "\n\nЛОКАЛЬНЫЙ RAG НЕ НАШЁЛ РЕЛЕВАНТНЫХ ФРАГМЕНТОВ:\n"
                        "Ответь на вопрос на основе общих знаний. Не придумывай цитаты, chunk_id или локальные источники."
                    ),
                    tool_context={
                        "profile_id": profile_id,
                        "source_conversation_id": conversation_id,
                        "project_id": conversation.get("project_id"),
                        "settings": settings_snapshot,
                        "enable_file_export": requests_file_export(content),
                        "mcp_activity_id": mcp_activity_id,
                        "auto_route": "general",
                    },
                )
                if result.technical.get("request_status", "completed") == "completed":
                    result = rag_miss_fallback(result, retrieval or {})
                set_mcp_activity_phase(mcp_activity_id, "completed")
            else:
                rag_guidance = ""
                if outside_rag:
                    rag_guidance = (
                        "\n\nОДНОРАЗОВЫЙ ОТВЕТ ВНЕ ЛОКАЛЬНОЙ БАЗЫ:\n"
                        "Пользователь явно разрешил для этого сообщения ответить вне локального RAG. "
                        f"Начни ответ точной строкой «{OUTSIDE_RAG_PREFIX}». "
                        "Используй собственные знания или доступные MCP-инструменты согласно запросу. "
                        f"Заверши ответ точной строкой «{OUTSIDE_RAG_SUFFIX}». "
                        "Не утверждай, что локальная база подтверждает этот ответ."
                    )
                elif document_route and retrieval:
                    rag_guidance = build_rag_context(retrieval)
                if auto_routing and selected_route == "task":
                    rag_guidance += (
                        "\n\nАВТОМАРШРУТ TASK STATE:\nОтветь по авторитетному состоянию текущей задачи "
                        "и истории активной ветки. Документальные цитаты для этого служебного ответа не требуются."
                    )
                elif auto_routing and selected_route == "general":
                    rag_guidance += (
                        "\n\nАВТОМАРШРУТ ОБЩЕГО ОТВЕТА:\nЛокальные документы не являются источником этого "
                        "ответа. Не приписывай им сведения и не изображай вызов внешнего API."
                    )
                elif auto_routing and selected_route == "tools":
                    rag_guidance += (
                        "\n\nАВТОМАРШРУТ ВНЕШНИХ ДАННЫХ:\nИспользуй только реально доступные инструменты. "
                        "Не подменяй результат API собственными знаниями."
                    )
                elif auto_routing and selected_route == "hybrid":
                    rag_guidance += (
                        "\n\nАВТОМАРШРУТ HYBRID:\nСочетай локальные документы только с фактически "
                        "полученными результатами инструментов и явно разделяй их происхождение."
                    )
                if strict_evidence:
                    rag_guidance += build_evidence_guidance(retrieval or {})
                result = controlled_agent_reply(
                    history=history_for_request,
                    user_text=content,
                    settings=settings,
                    summary=active_summary_text,
                    facts=active_facts,
                    memory_context=memory_snapshot,
                    task_memory_snapshot=task_memory_snapshot,
                    task_state=task_state_snapshot,
                    handoff_context=handoff_snapshot,
                    invariants=invariant_snapshot,
                    rag_context=rag_guidance,
                    tool_context={
                        "profile_id": profile_id,
                        "source_conversation_id": conversation_id,
                        "project_id": conversation.get("project_id"),
                        "settings": settings_snapshot,
                        "enable_file_export": requests_file_export(content),
                        "mcp_activity_id": mcp_activity_id,
                        "auto_route": (
                            selected_route if auto_routing
                            else ("rag" if rag_options["enabled"] and not outside_rag else "")
                        ),
                    },
                )
                if (
                    strict_evidence
                    and result.technical.get("request_status", "completed") == "completed"
                ):
                    result = finalize_verified_rag(content, result, retrieval or {}, settings)
                if outside_rag and result.technical.get("request_status", "completed") == "completed":
                    clean = result.content.strip()
                    if not clean.startswith(OUTSIDE_RAG_PREFIX):
                        clean = f"{OUTSIDE_RAG_PREFIX}\n\n{clean}"
                    if not clean.endswith(OUTSIDE_RAG_SUFFIX):
                        clean = f"{clean}\n\n{OUTSIDE_RAG_SUFFIX}"
                    result = AgentResult(
                        content=clean,
                        reasoning_content=result.reasoning_content,
                        technical={**result.technical, "outside_local_rag": True},
                    )
            result.technical["automatic_routing"] = deepcopy(route_info)
            if strict_evidence:
                result_evidence = deepcopy(result.technical.get("evidence") or {})
                request_rag_snapshot["evidence"] = result_evidence
                update_evidence_ledger(conversation, result_evidence, exchange_id)
            if auto_routing and not outside_rag:
                result = append_auto_route_sources(result, selected_route, retrieval)
            usage = normalize_token_usage(result.technical.get("usage"))
            request_status = result.technical.get("request_status", "completed")
            assistant_message_id = uuid.uuid4().hex
            audit = result.technical.get("policy_audit", {})
            if task_control_enabled and request_status == "completed":
                apply_explicit_plan_approval(
                    conversation,
                    content,
                    task_state_snapshot,
                    audit,
                )
            if task_control_enabled and request_status == "completed" and task_state_snapshot.get("stage") == "planning" and audit.get("stage_complete"):
                ensure_task_state(conversation)["required_artifacts"] = list(audit.get("required_artifacts", []))
                task_state_snapshot["required_artifacts"] = list(audit.get("required_artifacts", []))
            if task_control_enabled and request_status == "completed" and task_state_snapshot.get("stage") == "execution":
                try:
                    created_artifacts = save_response_artifacts(
                        storage.data_dir,
                        conversation,
                        result.content,
                        source_message_id=assistant_message_id,
                        stage_run_id=str(task_state_snapshot.get("stage_run_id", "")),
                    )
                    result.technical["artifacts"] = [public_artifact(item) for item in created_artifacts]
                    missing = missing_required_artifacts(conversation, storage.data_dir)
                    if audit.get("stage_complete") and missing:
                        audit.update({
                            "stage_complete": False,
                            "recommended_event": None,
                            "reason": "Ответ не создал обязательные артефакты: " + ", ".join(missing) + ".",
                        })
                except ArtifactError as artifact_error:
                    audit.update({
                        "stage_complete": False,
                        "recommended_event": None,
                        "reason": str(artifact_error),
                    })
            if (
                task_control_enabled
                and request_status == "completed"
                and task_state_snapshot.get("stage") == "validation"
                and audit.get("stage_complete")
                and audit.get("recommended_event") == "pass_validation"
            ):
                try:
                    verified = verify_required_artifacts(
                        conversation,
                        storage.data_dir,
                        validation_stage_run_id=str(task_state_snapshot.get("stage_run_id", "")),
                    )
                    result.technical["verified_artifacts"] = [public_artifact(item) for item in verified]
                except ArtifactError as artifact_error:
                    audit.update({
                        "stage_complete": False,
                        "recommended_event": None,
                        "reason": str(artifact_error),
                    })
            for message in conversation["messages"]:
                if message.get("id") == user_message["id"]:
                    message["technical"].update({
                        "request_status": request_status,
                        "token_usage": {
                            "context_tokens": usage["input_tokens"],
                            "cached_context_tokens": usage["cached_input_tokens"],
                            "uncached_context_tokens": usage["uncached_input_tokens"],
                        },
                        "context_management": context_snapshot,
                        "task_memory": task_memory_snapshot,
                        "task_state": task_state_snapshot,
                        "task_handoff_context": handoff_snapshot,
                        "invariants": invariant_snapshot,
                        "rag": request_rag_snapshot,
                        "task_control": {"enabled": task_control_enabled},
                    })
                    break
            assistant_message = {
                "id": assistant_message_id,
                "exchange_id": exchange_id,
                "role": "assistant",
                "content": result.content,
                "reasoning_content": result.reasoning_content,
                "created_at": now_iso(),
                "technical": {
                    **result.technical,
                    "request_status": request_status,
                    "configuration_source": source,
                    "settings": settings_snapshot,
                    "context_management": context_snapshot,
                    "memory_context": memory_snapshot,
                    "profile_snapshot": deepcopy(memory_snapshot.get("profile_snapshot", {})),
                    "task_memory": task_memory_snapshot,
                    "task_state": task_state_snapshot,
                    "task_handoff_context": handoff_snapshot,
                    "invariants": invariant_snapshot,
                    "rag": request_rag_snapshot,
                    "scheduled_automation": deepcopy(_automation_run_override.get()),
                    "task_control": {"enabled": task_control_enabled},
                },
            }
            append_tree_message(conversation, assistant_message, parent_id=user_message["id"])
            totals = dialogue_token_totals(conversation["messages"])
            conversation["token_totals"] = totals
            assistant_message["technical"]["dialogue_totals"] = totals
            conversation = storage.save_conversation(conversation)
            if request_status == "completed" and task_memory_snapshot.get("enabled"):
                try:
                    task_memory_result = chat_agent.update_task_memory(
                        task_memory_snapshot, content, result.content,
                    )
                    apply_task_memory_result(conversation, task_memory_result, exchange_id)
                except Exception as task_memory_error:
                    logger.warning(
                        "Ошибка обновления памяти задачи | conversation_id=%s | error=%s",
                        conversation_id, type(task_memory_error).__name__,
                    )
                    conversation.setdefault("task_memory_revisions", []).append({
                        "id": uuid.uuid4().hex,
                        "created_at": now_iso(),
                        "exchange_id": exchange_id,
                        "status": "failed",
                        "error": type(task_memory_error).__name__,
                        "technical": {"usage": normalize_token_usage({})},
                    })
                conversation = storage.save_conversation(conversation)
            project = None
            if conversation.get("project_id"):
                project = storage.get_project(conversation["project_id"])
            allow_project = bool(project and project.get("settings", {}).get("automatic_extraction", True))
            allow_user = bool(memory_manager.user_document(profile_id).get("settings", {}).get("automatic_extraction", False))
            if (
                request_status == "completed"
                and not result.technical.get("local_rag_refusal")
                and (allow_project or allow_user)
            ):
                try:
                    extraction_result = chat_agent.extract_memories(
                        content, result.content, allow_project=allow_project, allow_user=allow_user,
                    )
                    revision = memory_manager.route_candidates(extraction_result, conversation, exchange_id, profile_id=profile_id)
                except Exception as extraction_error:
                    logger.warning("Ошибка извлечения памяти | conversation_id=%s | error=%s", conversation_id, type(extraction_error).__name__)
                    revision = {
                        "id": uuid.uuid4().hex, "created_at": now_iso(), "status": "failed",
                        "error": type(extraction_error).__name__, "technical": {"usage": normalize_token_usage({})},
                    }
                conversation.setdefault("memory_extraction_revisions", []).append(revision)
                conversation = storage.save_conversation(conversation)
            logger.info(
                "Ответ сохранён | conversation_id=%s | exchange_id=%s | request_id=%s | total_tokens=%s",
                conversation_id, exchange_id, result.technical.get("request_id"), usage["total_tokens"],
            )
            return jsonify({"ok": True, "conversation": with_token_totals(conversation)})
        except Exception as error:
            set_mcp_activity_phase(mcp_activity_id, "error", friendly_api_error(error))
            logger.exception(
                "Ошибка обработки сообщения | conversation_id=%s | exchange_id=%s | model=%s",
                conversation_id, exchange_id, settings.model,
            )
            message = friendly_api_error(error)
            for item in conversation["messages"]:
                if item.get("id") == user_message["id"]:
                    item["technical"].update({"request_status": "failed", "error": message})
                    break
            conversation = storage.save_conversation(conversation)
            return jsonify({"ok": False, "error": message, "conversation": with_token_totals(conversation)}), 502

    @flask_app.get("/api/conversations/<conversation_id>/export")
    def export_conversation(conversation_id: str):
        try:
            accessible_conversation(conversation_id)
            bundle = storage.export_bundle(conversation_id)
            bundle["conversation"] = with_token_totals(bundle["conversation"])
        except (ValueError, FileNotFoundError, PermissionError):
            return api_error("Диалог не найден.", 404)
        payload = json.dumps(bundle, ensure_ascii=False, indent=2).encode("utf-8")
        return send_file(
            BytesIO(payload),
            mimetype="application/json; charset=utf-8",
            as_attachment=True,
            download_name=f"deepseek-dialog-{conversation_id[:8]}.json",
        )

    @flask_app.get("/api/conversations/<conversation_id>/artifacts/<artifact_id>")
    def download_artifact(conversation_id: str, artifact_id: str):
        """Отдаёт только зарегистрированный артефакт доступного пользователю диалога."""
        try:
            conversation = accessible_conversation(conversation_id)
            record = next(
                (item for item in conversation.get("artifacts", []) if item.get("id") == artifact_id),
                None,
            )
            if record is None:
                raise FileNotFoundError
            path = artifact_file_path(storage.data_dir, record)
            if not path.is_file():
                raise FileNotFoundError
            inline = request.args.get("disposition") == "inline"
            response = send_file(
                path,
                mimetype=record.get("mime_type") or "application/octet-stream",
                as_attachment=not inline,
                download_name=record.get("filename") or path.name,
            )
            response.headers["X-Content-Type-Options"] = "nosniff"
            if inline:
                response.headers["Content-Security-Policy"] = (
                    "sandbox; default-src 'none'; style-src 'unsafe-inline'; img-src data:; font-src data:"
                )
            return response
        except (ArtifactError, FileNotFoundError, PermissionError):
            return api_error("Артефакт не найден.", 404)

    @flask_app.post("/api/conversations/<conversation_id>/branches")
    def create_branch(conversation_id: str):
        """Начинает ветку после сообщения; после user сразу генерирует новый ответ."""
        try:
            conversation = accessible_conversation(conversation_id)
            task_control_enabled = task_control.is_enabled()
            if task_control_enabled and task_is_paused(conversation):
                return paused_task_response(conversation)
            checkpoint_id = json_body().get("checkpoint_id")
            source_message = next(
                (item for item in conversation.get("messages", []) if item.get("id") == checkpoint_id),
                None,
            )
            if source_message and source_message.get("technical", {}).get("local_command"):
                raise ValueError("Локальную команду состояния нельзя разветвить через модель.")
            if task_control_enabled and source_message:
                source_snapshot = source_message.get("technical", {}).get("task_state", {})
                current_state = ensure_task_state(conversation)
                stale_checkpoint = (
                    source_snapshot.get("stage_run_id") != current_state.get("stage_run_id")
                    if conversation.get("task_handoffs")
                    else bool(source_snapshot.get("stage") and source_snapshot.get("stage") != current_state.get("stage"))
                )
                if stale_checkpoint:
                    raise TaskTransitionError(
                        "Нельзя продолжить или перегенерировать сообщение из предыдущего этапа. "
                        "Используйте handoff текущего этапа."
                    )
            checkpoint = begin_branch(conversation, checkpoint_id)
            if checkpoint["role"] == "assistant":
                conversation = storage.save_conversation(conversation)
                return jsonify({"ok": True, "conversation": with_token_totals(conversation)}), 201

            technical = checkpoint.get("technical", {})
            settings = AgentSettings.from_dict(technical.get("settings"))
            source = technical.get("configuration_source") or {
                "type": "custom", "preset_id": None, "preset_name": None,
            }
            update_summaries(conversation, chat_agent, stage_scoped=task_control_enabled)
            history, summary, facts, snapshot = request_history(conversation, stage_scoped=task_control_enabled)
            profile_id = checkpoint.get("author_profile_id") or conversation.get("owner_profile_id") or current_profile_id()
            memory_snapshot = memory_manager.context_snapshot(conversation, snapshot["mode"], profile_id)
            task_memory_snapshot = task_memory_context(conversation)
            task_state_snapshot = deepcopy(ensure_task_state(conversation))
            handoff_snapshot = task_handoff_context(conversation)
            invariant_snapshot = invariant_context(conversation, profile_id)
            result = controlled_agent_reply(
                history=history,
                user_text=checkpoint["content"],
                settings=settings,
                summary=summary,
                facts=facts,
                memory_context=memory_snapshot,
                task_memory_snapshot=task_memory_snapshot,
                task_state=task_state_snapshot,
                handoff_context=handoff_snapshot,
                invariants=invariant_snapshot,
            )
            request_status = result.technical.get("request_status", "completed")
            retry_after_policy_block = technical.get("request_status") == "blocked"
            if retry_after_policy_block:
                usage = normalize_token_usage(result.technical.get("usage"))
                technical.update({
                    "request_status": request_status,
                    "token_usage": {
                        "context_tokens": usage["input_tokens"],
                        "cached_context_tokens": usage["cached_input_tokens"],
                        "uncached_context_tokens": usage["uncached_input_tokens"],
                    },
                    "context_management": snapshot,
                    "task_memory": task_memory_snapshot,
                    "task_state": task_state_snapshot,
                    "task_handoff_context": handoff_snapshot,
                    "invariants": invariant_snapshot,
                })
            assistant = {
                "id": uuid.uuid4().hex,
                "exchange_id": checkpoint.get("exchange_id"),
                "role": "assistant",
                "content": result.content,
                "reasoning_content": result.reasoning_content,
                "created_at": now_iso(),
                "technical": {
                    **result.technical,
                    "request_status": request_status,
                    "configuration_source": source,
                    "settings": settings.to_dict(),
                    "context_management": snapshot,
                    "regenerated_from_checkpoint": checkpoint["id"],
                    "retry_after_policy_block": retry_after_policy_block,
                    "memory_context": memory_snapshot,
                    "profile_snapshot": deepcopy(memory_snapshot.get("profile_snapshot", {})),
                    "task_memory": task_memory_snapshot,
                    "task_state": task_state_snapshot,
                    "task_handoff_context": handoff_snapshot,
                    "invariants": invariant_snapshot,
                    "task_control": {"enabled": task_control_enabled},
                },
            }
            append_tree_message(conversation, assistant, parent_id=checkpoint["id"])
            totals = dialogue_token_totals(conversation["messages"])
            conversation["token_totals"] = totals
            assistant["technical"]["dialogue_totals"] = totals
            conversation = storage.save_conversation(conversation)
            return jsonify({"ok": True, "conversation": with_token_totals(conversation)}), 201
        except (FileNotFoundError, PermissionError):
            return api_error("Диалог не найден.", 404)
        except TaskTransitionError as error:
            return api_error(str(error), 409)
        except (ValueError, KeyError) as error:
            return api_error(str(error).strip("'"), 400)
        except Exception as error:
            logger.exception("Ошибка DeepSeek API при создании ветки | conversation_id=%s", conversation_id)
            return api_error(friendly_api_error(error), 502)

    @flask_app.patch("/api/conversations/<conversation_id>/branches/active")
    def change_active_branch(conversation_id: str):
        try:
            conversation = accessible_conversation(conversation_id)
            data = json_body()
            select_branch(conversation, data.get("checkpoint_id"), data.get("child_id"))
            conversation = storage.save_conversation(conversation)
            return jsonify({"ok": True, "conversation": with_token_totals(conversation)})
        except (FileNotFoundError, PermissionError):
            return api_error("Диалог не найден.", 404)
        except ValueError as error:
            return api_error(str(error), 400)

    @flask_app.post("/api/conversations/<conversation_id>/facts")
    def create_manual_fact(conversation_id: str):
        try:
            conversation = accessible_conversation(conversation_id)
            data = json_body()
            fact = add_fact(conversation, data.get("key"), data.get("value"))
            conversation = storage.save_conversation(conversation)
            return jsonify({"ok": True, "fact": fact, "conversation": with_token_totals(conversation)}), 201
        except (FileNotFoundError, PermissionError):
            return api_error("Диалог не найден.", 404)
        except ValueError as error:
            return api_error(str(error), 400)

    @flask_app.put("/api/conversations/<conversation_id>/facts/<fact_id>")
    def update_manual_fact(conversation_id: str, fact_id: str):
        try:
            conversation = accessible_conversation(conversation_id)
            data = json_body()
            fact = edit_fact(conversation, fact_id, data.get("key"), data.get("value"), data.get("locked", True))
            conversation = storage.save_conversation(conversation)
            return jsonify({"ok": True, "fact": fact, "conversation": with_token_totals(conversation)})
        except (FileNotFoundError, PermissionError):
            return api_error("Диалог не найден.", 404)
        except KeyError:
            return api_error("Факт не найден.", 404)
        except ValueError as error:
            return api_error(str(error), 400)

    @flask_app.delete("/api/conversations/<conversation_id>/facts/<fact_id>")
    def remove_manual_fact(conversation_id: str, fact_id: str):
        try:
            conversation = accessible_conversation(conversation_id)
            delete_fact(conversation, fact_id)
            conversation = storage.save_conversation(conversation)
            return jsonify({"ok": True, "conversation": with_token_totals(conversation)})
        except (FileNotFoundError, PermissionError):
            return api_error("Диалог не найден.", 404)
        except KeyError:
            return api_error("Факт не найден.", 404)

    @flask_app.post("/api/import")
    def import_conversation():
        try:
            conversation = storage.import_bundle(json_body(), current_profile_id())
            return jsonify({"ok": True, "conversation": with_token_totals(conversation)}), 201
        except (ValueError, OSError, json.JSONDecodeError) as error:
            return api_error(f"Не удалось импортировать файл: {error}", 400)

    @flask_app.post("/api/presets")
    def create_preset():
        try:
            return jsonify({"ok": True, "preset": preset_manager.create(json_body())}), 201
        except ValueError as error:
            return api_error(str(error), 400)

    @flask_app.put("/api/presets/<preset_id>")
    def update_preset(preset_id: str):
        try:
            return jsonify({"ok": True, "preset": preset_manager.update(preset_id, json_body())})
        except KeyError:
            return api_error("Пресет не найден.", 404)
        except ValueError as error:
            return api_error(str(error), 400)

    @flask_app.delete("/api/presets/<preset_id>")
    def delete_preset(preset_id: str):
        try:
            preset_manager.delete(preset_id)
            return jsonify({"ok": True})
        except KeyError:
            return api_error("Пресет не найден.", 404)

    def run_scheduled_automation(task: dict[str, Any], run: dict[str, Any]) -> dict[str, Any]:
        if mcp_control.is_enabled() and "open-meteo" in task.get("mcp_servers", []):
            weather_mcp.start()
            weather_mcp.list_tools()
        profile_token = _profile_override.set(str(task["owner_profile_id"]))
        automation_token = _automation_run_override.set({
            "task_id": task["id"], "run_id": run["id"],
            "scheduled_for": run["scheduled_for"],
        })
        tools_token = _scheduled_tool_allowlist.set(set(task.get("allowed_tools", [])))
        try:
            client = flask_app.test_client()
            response = client.post(
                f"/api/conversations/{task['conversation_id']}/messages",
                json={"content": task["prompt"], "settings": task.get("settings") or {}},
            )
            payload = response.get_json(silent=True) or {}
            if response.status_code != 200 or not payload.get("ok"):
                raise RuntimeError(payload.get("error") or f"Внутренний запрос завершился с HTTP {response.status_code}.")
            messages = payload.get("conversation", {}).get("messages", [])
            assistant = next((item for item in reversed(messages) if item.get("role") == "assistant"), None)
            if not assistant:
                raise RuntimeError("Запланированный запуск не вернул ответ агента.")
            if assistant.get("technical", {}).get("request_status") != "completed":
                raise RuntimeError(assistant.get("content") or "Ответ автоматизации заблокирован.")
            return {"content": assistant.get("content", ""), "message_id": assistant.get("id")}
        finally:
            _scheduled_tool_allowlist.reset(tools_token)
            _automation_run_override.reset(automation_token)
            _profile_override.reset(profile_token)

    scheduler.set_runner(run_scheduled_automation)
    if scheduler_autostart:
        if mcp_control.is_enabled():
            scheduler_mcp.start()
            scheduler_mcp.list_tools()
        scheduler.start()
    if orchestration_autostart and mcp_control.is_enabled():
        for manager in (mediawiki_mcp, worldbank_mcp):
            manager.start()
            manager.list_tools()

    return flask_app


def json_body() -> dict[str, Any]:
    value = request.get_json(silent=True)
    if not isinstance(value, dict):
        raise ValueError("Ожидался JSON-объект.")
    return value


def require_message(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Введите сообщение.")
    content = value.strip()
    if len(content) > MAX_MESSAGE_LENGTH:
        raise ValueError("Сообщение длиннее 50 000 символов.")
    return content


def configuration_source(preset_id: Any, settings: AgentSettings, manager: PresetManager) -> dict[str, Any]:
    if not preset_id:
        return {"type": "custom", "preset_id": None, "preset_name": None}
    preset = manager.get(str(preset_id))
    source_type = "preset" if preset["settings"] == settings.to_dict() else "preset_modified"
    return {"type": source_type, "preset_id": preset["id"], "preset_name": preset["name"]}


def last_configuration(messages: list[dict[str, Any]]) -> dict[str, Any] | None:
    for message in reversed(messages):
        if message.get("role") == "user" and isinstance(message.get("technical"), dict):
            return message["technical"]
    return None


def configuration_event_text(source: dict[str, Any]) -> str:
    if source["type"] == "preset":
        return f"Выбран пресет «{source['preset_name']}»."
    if source["type"] == "preset_modified":
        return f"Настройки пресета «{source['preset_name']}» изменены для следующего запроса."
    return "Для следующего запроса выбраны разовые настройки."


def api_error(message: str, status: int):
    return jsonify({"ok": False, "error": message}), status


def with_token_totals(conversation: dict[str, Any]) -> dict[str, Any]:
    """Добавляет вычисляемые представления, не изменяя сохранённый оригинал."""
    result = deepcopy(conversation)
    ensure_context_management(result)
    ensure_task_memory(result)
    state = ensure_task_state(result)
    ensure_task_handoffs(result)
    stored_artifacts = ensure_artifacts(result)
    result["artifacts"] = [public_artifact(item) for item in stored_artifacts]
    result["active_artifacts"] = [public_artifact(item) for item in stored_artifacts if item.get("active") is True]
    result["token_totals"] = dialogue_token_totals(result.get("messages", []))
    visible = active_path_messages(result)
    result["visible_messages"] = visible
    result["active_path_token_totals"] = dialogue_token_totals(visible)
    result["branch_points"] = branch_points(result)
    result["summary_token_totals"] = summary_token_totals(result.get("summaries", []))
    result["facts_token_totals"] = facts_token_totals(result.get("fact_revisions", []))
    result["task_handoff_token_totals"] = handoff_token_totals(result.get("task_handoffs", []))
    result["active_task_handoffs"] = active_task_handoffs(result)
    result["active_stage_facts"] = deepcopy(current_stage_facts(result))
    result["current_stage_exchange_count"] = len(current_stage_exchanges(result))
    result["memory_token_totals"] = auxiliary_usage_totals(result.get("memory_extraction_revisions", []))
    result["task_memory_token_totals"] = auxiliary_usage_totals(result.get("task_memory_revisions", []))
    result["task_state"]["allowed_events"] = allowed_task_events(result["task_state"])
    return result


def auxiliary_usage_totals(revisions: list[dict[str, Any]]) -> dict[str, int]:
    totals = {key: 0 for key in ("input_tokens", "output_tokens", "total_tokens", "cached_input_tokens", "uncached_input_tokens", "reasoning_tokens")}
    totals["request_count"] = 0
    for revision in revisions:
        technical = revision.get("technical", {}) if isinstance(revision, dict) else {}
        if revision.get("status") != "completed" or not isinstance(technical.get("usage"), dict):
            continue
        usage = normalize_token_usage(technical["usage"])
        totals["request_count"] += 1
        for key in usage:
            totals[key] += usage[key]
    return totals


def friendly_api_error(error: Exception) -> str:
    text = str(error)
    lowered = text.lower()
    context_markers = (
        "context length",
        "context_length",
        "maximum context",
        "max context",
        "too many tokens",
        "token limit",
    )
    if any(marker in lowered for marker in context_markers):
        return (
            "Контекст диалога превысил лимит модели. "
            "Создайте новый диалог или сократите историю и повторите запрос."
        )
    if "api key" in lowered or "authentication" in lowered or "401" in lowered:
        return "DeepSeek отклонил API-ключ. Проверьте DEEPSEEK_API_KEY в файле .env."
    if "429" in lowered or "rate limit" in lowered:
        return "DeepSeek временно ограничил количество запросов. Попробуйте немного позже."
    if "timeout" in lowered or "timed out" in lowered:
        return "DeepSeek не успел ответить. Повторите запрос."
    if "connection" in lowered or "connect" in lowered or "network" in lowered:
        return "Не удалось соединиться с DeepSeek. Проверьте интернет, VPN или прокси."
    return f"Не удалось получить ответ DeepSeek: {text}"


_direct_log_paths = None
if __name__ == "__main__":
    from logging_setup import configure_logging

    _direct_log_paths = configure_logging(BASE_DIR / "logs")

app = create_app(voice_service=WhisperService(BASE_DIR, autostart=False), scheduler_autostart=False)


if __name__ == "__main__":
    app.extensions["whisper_service"].start()
    if app.extensions["mcp_control"].is_enabled():
        app.extensions["scheduler_mcp_manager"].start()
        app.extensions["scheduler_mcp_manager"].list_tools()
    app.extensions["scheduler_service"].start()
    if app.extensions["mcp_control"].is_enabled():
        for extension_name in ("mediawiki_mcp_manager", "worldbank_mcp_manager"):
            app.extensions[extension_name].start()
            app.extensions[extension_name].list_tools()
    logger.info("Приложение запущено напрямую через app.py | url=http://127.0.0.1:5000")
    print(f"Общий журнал: {_direct_log_paths['app']}")
    print(f"Журнал ошибок: {_direct_log_paths['errors']}")
    try:
        app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
    finally:
        app.extensions["scheduler_service"].stop()
        app.extensions["scheduler_mcp_manager"].stop()
        app.extensions["weather_mcp_manager"].stop()
        app.extensions["mediawiki_mcp_manager"].stop()
        app.extensions["worldbank_mcp_manager"].stop()
        app.extensions["mcp_manager"].stop()
        app.extensions["whisper_service"].stop()
        logger.info("Приложение остановлено")
