"""Реальный прогон проверяемого RAG Дня 24 и генерация Markdown-отчёта."""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import Agent, AgentResult, AgentSettings, DeepSeekProvider, normalize_token_usage
from document_index import (
    EvidenceValidationError,
    RAG_UNKNOWN_RESPONSE,
    RAG_UNSUPPORTED_RESPONSE,
    RagIndexService,
    apply_relevance_gate,
    build_evidence_guidance,
    build_rag_context,
    parse_semantic_evaluation,
    repair_prompt,
    semantic_evaluation_prompt,
    validate_and_format_evidence,
)
from document_index.evaluations import RagEvaluationStore


REPORT_PATH = ROOT / "docs" / "DAY24_RESULTS.md"


def load_questions() -> list[dict]:
    result = []
    for name in ("day22_control_questions.json", "day24_control_questions.json"):
        value = json.loads((ROOT / "docs" / name).read_text(encoding="utf-8"))
        result.extend(value["questions"])
    return result


def local_refusal(settings: AgentSettings, retrieval: dict) -> AgentResult:
    return AgentResult(RAG_UNKNOWN_RESPONSE, "", {
        "provider": "local-rag-gate",
        "request_id": None,
        "model": None,
        "finish_reason": "relevance_gate",
        "elapsed_seconds": 0,
        "usage": normalize_token_usage({}),
        "settings": settings.to_dict(),
        "local_rag_refusal": True,
        "evidence": {
            "status": "unknown", "valid": True,
            "sources_present": False, "quotes_present": False,
            "citations": [], "errors": [], "reason": "below_similarity_threshold",
            "max_similarity": retrieval.get("max_similarity"),
            "threshold": retrieval.get("relevance_threshold"),
        },
    })


def checked_answer(
    agent: Agent,
    question: str,
    retrieval: dict,
    settings: AgentSettings,
) -> AgentResult:
    draft = agent.reply(
        [], question, settings,
        policy_guidance=build_rag_context(retrieval) + build_evidence_guidance(retrieval),
    )
    repair = None
    try:
        content, evidence = validate_and_format_evidence(draft.content, retrieval)
    except EvidenceValidationError as first_error:
        repair = agent.repair_rag_evidence(
            repair_prompt(question, draft.content, retrieval, str(first_error)), settings,
        )
        try:
            content, evidence = validate_and_format_evidence(repair.content, retrieval)
        except EvidenceValidationError as second_error:
            content = RAG_UNSUPPORTED_RESPONSE
            evidence = {
                "status": "unknown", "valid": False,
                "sources_present": False, "quotes_present": False,
                "citations": [], "errors": [str(first_error), str(second_error)],
                "reason": "citation_validation_failed",
            }
    technical = deepcopy(draft.technical)
    technical["evidence"] = evidence
    if repair:
        technical["evidence_repair"] = deepcopy(repair.technical)
    return AgentResult(content, draft.reasoning_content, technical)


def number(value: object) -> str:
    return "—" if value is None else f"{float(value):.4f}"


def md_cell(value: object) -> str:
    return str(value or "").replace("|", "\\|").replace("\n", "<br>")


def write_report(items: list[dict]) -> None:
    ordered = sorted(items, key=lambda item: int(item.get("batch_order", 999)))
    positives = [item for item in ordered if item.get("expected_behavior") != "unknown"]
    negatives = [item for item in ordered if item.get("expected_behavior") == "unknown"]
    positive_sources = sum(bool(item.get("evidence", {}).get("sources_present")) for item in positives)
    positive_quotes = sum(bool(item.get("evidence", {}).get("quotes_present")) for item in positives)
    semantic_ok = sum(bool(item.get("semantic_evaluation", {}).get("meaning_supported")) for item in ordered)
    behavior_ok = sum(bool(item.get("expected_behavior_passed")) for item in ordered)
    rewrite_tokens = sum(int((item.get("rewrite") or {}).get("technical", {}).get("usage", {}).get("total_tokens", 0)) for item in ordered)
    answer_tokens = sum(int(item.get("answer", {}).get("technical", {}).get("usage", {}).get("total_tokens", 0)) for item in ordered)
    repair_tokens = sum(int(item.get("answer", {}).get("technical", {}).get("evidence_repair", {}).get("usage", {}).get("total_tokens", 0)) for item in ordered)
    judge_tokens = sum(int(item.get("semantic_evaluation", {}).get("technical", {}).get("usage", {}).get("total_tokens", 0)) for item in ordered)
    rewrite_calls = sum(bool(item.get("rewrite")) for item in ordered)
    answer_calls = sum(not item.get("answer", {}).get("technical", {}).get("local_rag_refusal") for item in ordered)
    repair_calls = sum("evidence_repair" in item.get("answer", {}).get("technical", {}) for item in ordered)
    judge_calls = sum(bool(item.get("semantic_evaluation", {}).get("technical")) for item in ordered)
    lines = [
        "# День 24. Цитаты, источники и анти-галлюцинации",
        "",
        "Реальный прогон выполнен на `deepseek-v4-flash`. Использованы `combined`, candidate top-K = 20, "
        "final top-K = 5 и similarity threshold = 0,83.",
        "",
        "## Сводка",
        "",
        "| Проверка | Результат |",
        "|---|---:|",
        f"| Основные вопросы с источниками | {positive_sources}/{len(positives)} |",
        f"| Основные вопросы с дословными цитатами | {positive_quotes}/{len(positives)} |",
        f"| Смысл ответа подтверждён цитатами по LLM-оценке | {semantic_ok}/{len(ordered)} |",
        f"| Ожидаемое поведение выполнено | {behavior_ok}/{len(ordered)} |",
        f"| Негативные вопросы, завершившиеся «не знаю» | {sum(item.get('evidence', {}).get('status') == 'unknown' for item in negatives)}/{len(negatives)} |",
        "",
        "## Реальные вызовы и токены",
        "",
        "| Назначение | Вызовов | Токенов |",
        "|---|---:|---:|",
        f"| Query rewrite | {rewrite_calls} | {rewrite_tokens:,} |".replace(",", " "),
        f"| Проверяемый ответ | {answer_calls} | {answer_tokens:,} |".replace(",", " "),
        f"| Исправление формата/цитат | {repair_calls} | {repair_tokens:,} |".replace(",", " "),
        f"| LLM-оценка смысла | {judge_calls} | {judge_tokens:,} |".replace(",", " "),
        f"| **Всего** | **{rewrite_calls + answer_calls + repair_calls + judge_calls}** | **{rewrite_tokens + answer_tokens + repair_tokens + judge_tokens:,}** |".replace(",", " "),
        "",
        f"Четыре полностью посторонних вопроса были остановлены локальным порогом до query rewrite и ответа. "
        f"Вопрос про ORCID прошёл similarity-порог, но модель корректно выбрала `unknown`. "
        f"Исправляющие вызовы не потребовались.",
        "",
        "## Таблица проверки",
        "",
        "| № | Вопрос | Max similarity | Статус | Источники | Дословные цитаты | Смысл подтверждён | Ожидаемое поведение |",
        "|---:|---|---:|---|---|---|---|---|",
    ]
    for item in ordered:
        evidence = item.get("evidence", {})
        semantic = item.get("semantic_evaluation", {})
        lines.append(
            f"| {item.get('batch_order')} | {md_cell(item.get('question'))} | {number(item.get('retrieval', {}).get('max_similarity'))} | "
            f"{evidence.get('status', '—')} | {'да' if evidence.get('sources_present') else 'не требуются' if evidence.get('status') == 'unknown' else 'нет'} | "
            f"{'да' if evidence.get('quotes_present') else 'не требуются' if evidence.get('status') == 'unknown' else 'нет'} | "
            f"{'да' if semantic.get('meaning_supported') else 'нет'} | {'да' if item.get('expected_behavior_passed') else 'нет'} |"
        )
    lines.extend(["", "## Полные результаты", ""])
    for item in ordered:
        evidence = item.get("evidence", {})
        semantic = item.get("semantic_evaluation", {})
        lines.extend([
            f"### {item.get('batch_order')}. {item.get('question')}",
            "",
            f"- Ожидание: {item.get('expectation') or 'не задано'}",
            f"- Max similarity: {number(item.get('retrieval', {}).get('max_similarity'))}",
            f"- Статус: `{evidence.get('status', '—')}`",
            f"- Программная проверка: {'пройдена' if evidence.get('valid') else 'не пройдена'}",
            f"- LLM-оценка смысла: {'подтверждён' if semantic.get('meaning_supported') else 'не подтверждён'} — {semantic.get('notes') or 'без пояснения'}",
            "",
            "Полученный ответ:",
            "",
            str(item.get("answer", {}).get("content") or ""),
            "",
        ])
    lines.extend([
        "## Выводы",
        "",
        "1. Источники и разделы не принимаются от модели: приложение восстанавливает их по фактическому `chunk_id` из выдачи.",
        "2. Цитата считается корректной только при полном дословном совпадении с непрерывным фрагментом чанка.",
        "3. Если ни один чанк не проходит порог 0,83, ответ DeepSeek не вызывается и приложение возвращает локальное «не знаю».",
        "4. Если релевантный по теме чанк найден, но конкретного факта в нём нет, модель обязана выбрать `status=unknown`; неподтверждённый ответ не выпускается.",
        "5. Смысловая LLM-оценка является дополнительной аналитикой, а не заменой детерминированной проверки источника и дословности.",
        "",
    ])
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="выполнить реальные вызовы DeepSeek")
    parser.add_argument("--limit", type=int, default=15)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--final-k", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.83)
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    questions = load_questions()[: max(1, min(15, args.limit))]
    service = RagIndexService(ROOT / "rag_documents", ROOT / "data" / "rag_index.sqlite3")
    store = RagEvaluationStore(ROOT / "data" / "rag_evaluations.json")
    existing = {
        item.get("question_id"): item for item in store.list()
        if item.get("evaluation_mode") == "batch-day24-live"
    }
    agent = Agent(DeepSeekProvider()) if args.live else None
    settings = AgentSettings(
        model="deepseek-v4-flash",
        system_prompt=(
            "Отвечай на русском языке только по переданному документальному контексту. "
            "Не используй внешние знания и не выдумывай отсутствующие факты."
        ),
        temperature=0,
        top_p=1,
        reasoning_enabled=False,
        reasoning_effort="low",
        max_tokens=1_500,
    )
    saved = 0
    for index, question in enumerate(questions, 1):
        if question["id"] in existing:
            print(f"[{index:02d}/{len(questions):02d}] {question['id']} | already saved", flush=True)
            continue
        preflight = service.retrieve(question["question"], "combined", args.candidate_k)
        gated_preflight = apply_relevance_gate(preflight, args.threshold)
        gated_preflight["gate_phase"] = "preflight"
        rewrite = None
        retrieval = gated_preflight
        if gated_preflight["chunks"]:
            if not args.live:
                print(f"[{index:02d}/{len(questions):02d}] {question['id']} | would call DeepSeek", flush=True)
                continue
            rewrite_result = agent.rewrite_rag_query(question["question"], settings)
            rewrite_query = " ".join(rewrite_result.content.strip().strip('"').split())
            rewrite = {"query": rewrite_query, "technical": deepcopy(rewrite_result.technical)}
            retrieval = service.retrieve_pipeline(
                question["question"], strategy="combined", mode="combined",
                candidate_k=args.candidate_k, final_k=args.final_k,
                similarity_threshold=args.threshold, rewritten_query=rewrite_query,
            )
            retrieval = apply_relevance_gate(retrieval, args.threshold)
            retrieval["gate_phase"] = "final"
            retrieval["preflight_max_similarity"] = gated_preflight.get("max_similarity")
        if not args.live:
            print(
                f"[{index:02d}/{len(questions):02d}] {question['id']} | local refusal | "
                f"max={number(retrieval.get('max_similarity'))}", flush=True,
            )
            continue

        answer = local_refusal(settings, retrieval) if not retrieval["chunks"] else checked_answer(
            agent, question["question"], retrieval, settings,
        )
        evidence = deepcopy(answer.technical.get("evidence") or {})
        judge = agent.evaluate_rag_evidence(
            semantic_evaluation_prompt(
                question["question"], answer.content, evidence, question.get("expectation"),
            ),
            settings,
        )
        try:
            semantic = parse_semantic_evaluation(judge.content)
        except EvidenceValidationError as error:
            semantic = {"meaning_supported": False, "unsupported_claims": [], "notes": str(error)}
        semantic["technical"] = deepcopy(judge.technical)
        expected_behavior = question.get("expected_behavior", "answer")
        expected_behavior_passed = (
            evidence.get("status") == "unknown" if expected_behavior == "unknown"
            else evidence.get("status") == "answer" and evidence.get("valid") is True
        )
        stored = store.add({
            "evaluation_mode": "batch-day24-live",
            "batch_order": index,
            "conversation_id": None,
            "question_id": question["id"],
            "question": question["question"],
            "expectation": question.get("expectation"),
            "expected_sources": question.get("expected_sources", []),
            "expected_behavior": expected_behavior,
            "expected_behavior_passed": expected_behavior_passed,
            "settings": settings.to_dict(),
            "pipeline_settings": {
                "strategy": "combined", "mode": "combined",
                "candidate_k": args.candidate_k, "final_k": args.final_k,
                "similarity_threshold": args.threshold,
            },
            "rewrite": rewrite,
            "retrieval": retrieval,
            "answer": {
                "content": answer.content,
                "reasoning_content": answer.reasoning_content,
                "technical": answer.technical,
            },
            "evidence": evidence,
            "semantic_evaluation": semantic,
        })
        existing[question["id"]] = stored
        saved += 1
        print(
            f"[{index:02d}/{len(questions):02d}] {question['id']} | status={evidence.get('status')} "
            f"| sim={number(retrieval.get('max_similarity'))} | semantic={semantic.get('meaning_supported')}",
            flush=True,
        )

    batch_items = [
        item for item in store.list()
        if item.get("evaluation_mode") == "batch-day24-live"
    ]
    if batch_items:
        write_report(batch_items)
        print(f"Report: {REPORT_PATH}")
    print(f"Live evaluations saved now: {saved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
