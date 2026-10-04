"""Проверяемые источники, цитаты и отказ при слабом RAG-контексте."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from typing import Any


RAG_UNKNOWN_RESPONSE = (
    "Не знаю: в базе знаний не найдено достаточно релевантной информации. "
    "Уточните вопрос или добавьте подходящий документ. Если нужен ответ вне базы, "
    "напишите: «Ответь вне базы знаний» или «Используй доступные инструменты»."
)
RAG_UNSUPPORTED_RESPONSE = (
    "Не знаю: найденные фрагменты не позволяют подтвердить ответ дословными цитатами. "
    "Уточните вопрос или добавьте подходящий документ. Если нужен ответ вне базы, "
    "напишите: «Ответь вне базы знаний» или «Используй доступные инструменты»."
)


class EvidenceValidationError(ValueError):
    """Ответ модели нельзя связать с фактически переданными чанками."""


def _pdf_normalized_with_offsets(value: str) -> tuple[str, list[int]]:
    """Нормализует только пробелы и перенос слова вида ``сло- во``.

    Смещения позволяют после однозначного совпадения вернуть исходный дословный
    фрагмент, а не нормализованный текст модели.
    """
    text = str(value or "")
    normalized: list[str] = []
    offsets: list[int] = []
    index = 0
    while index < len(text):
        char = text[index]
        if char == "-" and index > 0 and text[index - 1].isalnum():
            next_index = index + 1
            while next_index < len(text) and text[next_index].isspace():
                next_index += 1
            if next_index > index + 1 and next_index < len(text) and text[next_index].isalnum():
                index = next_index
                continue
        if char.isspace():
            if normalized and normalized[-1] != " ":
                normalized.append(" ")
                offsets.append(index)
            index += 1
            continue
        normalized.append(char)
        offsets.append(index)
        index += 1
    return "".join(normalized), offsets


def _canonical_quote(chunk_text: str, quote: str) -> tuple[str | None, bool]:
    """Возвращает точный исходный фрагмент при прямом или однозначном PDF-совпадении."""
    if quote and quote in chunk_text:
        return quote, False
    normalized_text, offsets = _pdf_normalized_with_offsets(chunk_text)
    normalized_quote, _ = _pdf_normalized_with_offsets(quote)
    normalized_quote = normalized_quote.strip()
    if not normalized_quote:
        return None, False
    first = normalized_text.find(normalized_quote)
    if first < 0 or normalized_text.find(normalized_quote, first + 1) >= 0:
        return None, False
    last = first + len(normalized_quote) - 1
    if first >= len(offsets) or last >= len(offsets):
        return None, False
    return chunk_text[offsets[first]:offsets[last] + 1], True


def apply_relevance_gate(retrieval: dict[str, Any], threshold: float) -> dict[str, Any]:
    """Всегда применяет порог Дня 24, независимо от выбранного режима Дня 23."""
    result = deepcopy(retrieval)
    chunks = list(result.get("chunks") or [])
    result["pre_gate_count"] = len(chunks)
    def score(item: dict[str, Any]) -> float:
        return float(item.get("relevance_score", item.get("score", 0)))

    result["max_similarity"] = max((score(item) for item in chunks), default=None)
    result["chunks"] = [
        item for item in chunks
        if item.get("channel_gate_passed") is True or score(item) >= float(threshold)
    ]
    result["relevance_threshold"] = float(threshold)
    result["relevance_gate_passed"] = bool(result["chunks"])
    result["post_gate_count"] = len(result["chunks"])
    return result


def build_evidence_guidance(retrieval: dict[str, Any]) -> str:
    allowed = [
        {
            "ref": index,
            "chunk_id": chunk.get("chunk_id"),
            "source": chunk.get("source"),
            "section": chunk.get("section"),
        }
        for index, chunk in enumerate(retrieval.get("chunks") or [], 1)
    ]
    return (
        "\n\nПРОВЕРЯЕМЫЙ RAG — ОБЯЗАТЕЛЬНЫЙ ФОРМАТ:\n"
        "Отвечай только по документальному контексту. Каждый фактический вывод снабди ссылкой [N]. "
        "Верни только один JSON-объект без Markdown-ограждения: "
        '{"status":"answer","answer":"Текст с ссылками [1]","citations":'
        '[{"ref":1,"chunk_id":"точный id","quote":"дословная непрерывная цитата"}]}. '
        "ref — номер чанка из списка ниже. quote должна быть дословным непрерывным фрагментом этого чанка: "
        "не исправляй регистр, пунктуацию и пробелы, не используй многоточие вместо пропуска. "
        "Каждая запись citations должна быть использована в answer как [N], а каждая ссылка в answer должна иметь citation. "
        "Не включай source и section самостоятельно: приложение подставит их из метаданных. "
        "Если найденные чанки не содержат ответа, верни только "
        '{"status":"unknown","answer":"Не знаю: найденные фрагменты не содержат подтверждённого ответа. '
        'Уточните вопрос.","citations":[]}.\n'
        "Разрешённые чанки:\n"
        f"{json.dumps(allowed, ensure_ascii=False)}"
    )


def parse_json_object(content: str) -> dict[str, Any]:
    text = str(content or "").strip().lstrip("\ufeff")
    candidates = [text]
    if text.startswith("```") and text.endswith("```"):
        first_newline = text.find("\n")
        if first_newline >= 0:
            candidates.append(text[first_newline + 1:-3].strip())
    opening = text.find("{")
    closing = text.rfind("}")
    if opening >= 0 and closing > opening:
        candidates.append(text[opening:closing + 1])
    for candidate in dict.fromkeys(candidates):
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise EvidenceValidationError("Модель не вернула корректный JSON проверяемого ответа.")


def validate_and_format_evidence(content: str, retrieval: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    payload = parse_json_object(content)
    status = str(payload.get("status") or "").strip().lower()
    answer = str(payload.get("answer") or "").strip()
    citations = payload.get("citations")
    if status == "unknown":
        if "не знаю" not in answer.lower() or not any(word in answer.lower() for word in ("уточн", "добав")):
            raise EvidenceValidationError("Отказ должен содержать «не знаю» и просьбу уточнить вопрос.")
        if citations not in (None, []):
            raise EvidenceValidationError("Отказ не должен содержать неподтверждённые цитаты.")
        return answer, {
            "status": "unknown",
            "valid": True,
            "sources_present": False,
            "quotes_present": False,
            "citations": [],
            "errors": [],
        }
    if status != "answer":
        raise EvidenceValidationError("Поле status должно быть answer или unknown.")
    if not answer:
        raise EvidenceValidationError("Проверяемый ответ пуст.")
    if not isinstance(citations, list) or not citations:
        raise EvidenceValidationError("Для ответа требуется хотя бы одна цитата.")

    chunks_by_id = {
        str(chunk.get("chunk_id")): (index, chunk)
        for index, chunk in enumerate(retrieval.get("chunks") or [], 1)
        if chunk.get("chunk_id")
    }
    validated = []
    used_refs: set[int] = set()
    declared_to_expected: dict[int, int] = {}
    for item in citations:
        if not isinstance(item, dict):
            raise EvidenceValidationError("Каждая citation должна быть объектом.")
        try:
            ref = int(item.get("ref"))
        except (TypeError, ValueError) as error:
            raise EvidenceValidationError("У citation отсутствует числовой ref.") from error
        chunk_id = str(item.get("chunk_id") or "")
        quote = str(item.get("quote") or "")
        if chunk_id not in chunks_by_id:
            raise EvidenceValidationError(f"Chunk {chunk_id!r} не передавался модели.")
        expected_ref, chunk = chunks_by_id[chunk_id]
        if ref in declared_to_expected and declared_to_expected[ref] != expected_ref:
            raise EvidenceValidationError(f"Ссылка [{ref}] назначена нескольким разным chunk_id.")
        if expected_ref in used_refs:
            raise EvidenceValidationError(f"Chunk {chunk_id!r} повторяется в citations.")
        quote, quote_canonicalized = _canonical_quote(str(chunk.get("text") or ""), quote)
        if not quote:
            raise EvidenceValidationError(f"Цитата для chunk_id {chunk_id!r} не является дословным фрагментом чанка.")
        declared_to_expected[ref] = expected_ref
        used_refs.add(expected_ref)
        validated.append({
            "ref": expected_ref,
            "chunk_id": chunk_id,
            "source": str(chunk.get("source") or ""),
            "section": str(chunk.get("section") or "Без раздела"),
            "page": chunk.get("page"),
            "quote": quote,
            "quote_canonicalized": quote_canonicalized,
            "similarity": float(chunk.get("score", 0)),
        })

    for declared, expected in declared_to_expected.items():
        answer = answer.replace(f"[{declared}]", f"[REF-{expected}]")
    answer = re.sub(r"\[REF-(\d+)]", r"[\1]", answer)
    inline_refs = {int(value) for value in re.findall(r"\[(\d+)]", answer)}
    if inline_refs != used_refs:
        raise EvidenceValidationError(
            "Ссылки в тексте ответа и список citations должны совпадать: "
            f"в тексте {sorted(inline_refs)}, в citations {sorted(used_refs)}."
        )
    validated.sort(key=lambda item: item["ref"])
    source_lines = []
    quote_lines = []
    for item in validated:
        page = f", стр. {item['page']}" if item.get("page") else ""
        source_lines.append(
            f"[{item['ref']}] {item['source']} — {item['section']}{page}; chunk_id: `{item['chunk_id']}`"
        )
        quote_lines.append(f"[{item['ref']}] «{item['quote']}»")
    formatted = (
        f"{answer}\n\nИсточники:\n" + "\n".join(source_lines)
        + "\n\nЦитаты:\n" + "\n".join(quote_lines)
    )
    return formatted, {
        "status": "answer",
        "valid": True,
        "sources_present": True,
        "quotes_present": True,
        "citations": validated,
        "errors": [],
    }


def repair_prompt(question: str, content: str, retrieval: dict[str, Any], error: str) -> str:
    chunks = [
        {
            "ref": index,
            "chunk_id": chunk.get("chunk_id"),
            "text": chunk.get("text"),
        }
        for index, chunk in enumerate(retrieval.get("chunks") or [], 1)
    ]
    return (
        "Исправь только структуру, ссылки и дословные цитаты проверяемого RAG-ответа. "
        "Не добавляй новых фактов. Верни только JSON строго по схеме: "
        '{"status":"answer","answer":"Текст с [1]","citations":'
        '[{"ref":1,"chunk_id":"точный id","quote":"дословная непрерывная цитата"}]}. '
        "Если доказательств недостаточно, верни status=unknown, фразу «Не знаю» и просьбу уточнить вопрос.\n\n"
        f"Вопрос:\n{question}\n\nОшибка проверки:\n{error}\n\nЧерновик:\n{content}\n\n"
        f"Разрешённые чанки:\n{json.dumps(chunks, ensure_ascii=False)}"
    )


def retry_unknown_prompt(question: str, content: str, retrieval: dict[str, Any]) -> str:
    chunks = [
        {
            "ref": index,
            "chunk_id": chunk.get("chunk_id"),
            "source": chunk.get("source"),
            "section": chunk.get("section"),
            "text": chunk.get("text"),
        }
        for index, chunk in enumerate(retrieval.get("chunks") or [], 1)
    ]
    return (
        "Повторно проверь отказ, используя только разрешённые чанки. Контекст уже прошёл relevance gate. "
        "Если в чанках есть прямое подтверждение ответа, верни только JSON строго по схеме "
        '{"status":"answer","answer":"Текст с [1]","citations":'
        '[{"ref":1,"chunk_id":"точный id","quote":"дословная непрерывная цитата"}]}. '
        "сославшись на дословные непрерывные цитаты. Если ответа действительно нет, сохрани status=unknown. "
        "Не добавляй внешних знаний и новых фактов.\n\n"
        f"Вопрос:\n{question}\n\nПервый ответ:\n{content}\n\n"
        f"Разрешённые чанки:\n{json.dumps(chunks, ensure_ascii=False)}"
    )


def semantic_evaluation_prompt(
    question: str,
    answer: str,
    evidence: dict[str, Any],
    expectation: str | None = None,
) -> str:
    return (
        "Оцени только смысловую подтверждённость ответа приведёнными дословными цитатами. "
        "Не используй внешние знания. Если evidence.status=unknown, считай корректным только осторожный отказ "
        "без фактического ответа, содержащий «не знаю» и просьбу уточнить вопрос; отсутствие цитат у такого отказа допустимо. "
        "Верни JSON: "
        '{"meaning_supported":true,"unsupported_claims":[],"notes":"краткое объяснение"}. '
        "meaning_supported=true только если все существенные фактические утверждения ответа следуют из цитат.\n\n"
        f"Вопрос:\n{question}\n\nОжидание контрольного набора:\n{expectation or 'не задано'}\n\n"
        f"Ответ:\n{answer}\n\nПроверенные доказательства:\n"
        f"{json.dumps(evidence.get('citations', []), ensure_ascii=False)}"
    )


def parse_semantic_evaluation(content: str) -> dict[str, Any]:
    value = parse_json_object(content)
    unsupported = value.get("unsupported_claims", [])
    if not isinstance(unsupported, list):
        raise EvidenceValidationError("LLM-оценка должна вернуть unsupported_claims как массив.")
    return {
        "meaning_supported": value.get("meaning_supported") is True,
        "unsupported_claims": [str(item) for item in unsupported],
        "notes": str(value.get("notes") or "").strip(),
    }
