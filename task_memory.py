"""Структурированная память текущей задачи для production-like RAG-чата."""

from __future__ import annotations

import json
import re
import uuid
from copy import deepcopy
from datetime import datetime
from typing import Any

from agent import normalize_token_usage


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


LIST_FIELDS = ("clarifications", "constraints", "decisions", "open_questions")
MAX_ITEMS = 30
MAX_ITEM_CHARS = 1_000
MAX_GOAL_CHARS = 2_000
SECRET_RE = re.compile(
    r"(?i)(api[_ -]?key|password|парол|секрет|token)\s*[:=]\s*\S+|sk-[A-Za-z0-9_-]{12,}"
)


def initial_task_memory(timestamp: str | None = None) -> dict[str, Any]:
    return {
        "enabled": False,
        "goal": "",
        "clarifications": [],
        "constraints": [],
        "terms": [],
        "decisions": [],
        "open_questions": [],
        "revision": 0,
        "updated_at": timestamp,
    }


def ensure_task_memory(conversation: dict[str, Any]) -> dict[str, Any]:
    raw = conversation.get("task_memory")
    source = raw if isinstance(raw, dict) else {}
    normalized = initial_task_memory(source.get("updated_at"))
    normalized["enabled"] = source.get("enabled") is True
    normalized["goal"] = _clean_text(source.get("goal"), MAX_GOAL_CHARS, allow_empty=True)
    for field in LIST_FIELDS:
        normalized[field] = _clean_list(source.get(field))
    normalized["terms"] = _clean_terms(source.get("terms"))
    try:
        normalized["revision"] = max(0, int(source.get("revision", 0)))
    except (TypeError, ValueError):
        normalized["revision"] = 0
    conversation["task_memory"] = normalized
    revisions = conversation.get("task_memory_revisions")
    conversation["task_memory_revisions"] = revisions if isinstance(revisions, list) else []
    return normalized


def update_task_memory_settings(conversation: dict[str, Any], values: dict[str, Any]) -> dict[str, Any]:
    memory = ensure_task_memory(conversation)
    if set(values) - {"enabled"}:
        raise ValueError("Разрешено изменять только переключатель памяти задачи.")
    if "enabled" not in values or not isinstance(values["enabled"], bool):
        raise ValueError("Переключатель памяти задачи должен быть логическим.")
    memory["enabled"] = values["enabled"]
    memory["updated_at"] = now_iso()
    return memory


def task_memory_context(conversation: dict[str, Any]) -> dict[str, Any]:
    memory = ensure_task_memory(conversation)
    if not memory["enabled"]:
        return {}
    return deepcopy(memory)


def apply_task_memory_result(
    conversation: dict[str, Any],
    result: Any,
    exchange_id: str,
) -> dict[str, Any]:
    before = deepcopy(ensure_task_memory(conversation))
    revision = {
        "id": uuid.uuid4().hex,
        "created_at": now_iso(),
        "exchange_id": exchange_id,
        "status": "completed",
        "changed_fields": [],
        "technical": {"usage": normalize_token_usage(result.technical.get("usage"))},
    }
    try:
        value = _json_object(result.content)
        payload = value.get("task_memory") if isinstance(value.get("task_memory"), dict) else value
        after = initial_task_memory(now_iso())
        after["enabled"] = before["enabled"]
        after["goal"] = _clean_text(payload.get("goal"), MAX_GOAL_CHARS, allow_empty=True)
        for field in LIST_FIELDS:
            after[field] = _clean_list(payload.get(field))
        after["terms"] = _clean_terms(payload.get("terms"))
        tracked = ("goal", *LIST_FIELDS, "terms")
        revision["changed_fields"] = [field for field in tracked if before.get(field) != after.get(field)]
        after["revision"] = before.get("revision", 0) + (1 if revision["changed_fields"] else 0)
        if not revision["changed_fields"]:
            after["updated_at"] = before.get("updated_at")
        conversation["task_memory"] = after
        revision["snapshot"] = deepcopy(after)
    except Exception as error:
        revision.update({"status": "failed", "error": type(error).__name__})
        conversation["task_memory"] = before
    revisions = conversation.setdefault("task_memory_revisions", [])
    revisions.append(revision)
    if len(revisions) > 100:
        del revisions[:-100]
    return revision


def _clean_text(value: Any, limit: int, *, allow_empty: bool = False) -> str:
    text = str(value or "").strip()
    if not text and allow_empty:
        return ""
    if not text:
        raise ValueError("Пустое значение памяти задачи.")
    text = text[:limit]
    if SECRET_RE.search(text):
        raise ValueError("Память задачи не должна содержать секреты.")
    return text


def _clean_list(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("Поле памяти задачи должно быть массивом строк.")
    result = []
    for item in value[:MAX_ITEMS]:
        text = _clean_text(item, MAX_ITEM_CHARS, allow_empty=True)
        if text and text not in result:
            result.append(text)
    return result


def _clean_terms(value: Any) -> list[dict[str, str]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("Термины памяти задачи должны быть массивом.")
    result = []
    seen = set()
    for item in value[:MAX_ITEMS]:
        if not isinstance(item, dict):
            raise ValueError("Термин памяти задачи должен быть объектом.")
        term = _clean_text(item.get("term"), 200, allow_empty=True)
        meaning = _clean_text(item.get("meaning"), MAX_ITEM_CHARS, allow_empty=True)
        key = term.casefold()
        if term and meaning and key not in seen:
            result.append({"term": term, "meaning": meaning})
            seen.add(key)
    return result


def _json_object(content: Any) -> dict[str, Any]:
    text = str(content or "").strip().lstrip("\ufeff")
    if text.startswith("```") and text.endswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("Модель не вернула объект памяти задачи.")
    return value
