"""Реальный прогон двух длинных сценариев Дня 25 через штатный Flask API."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
REPORT_PATH = ROOT / "docs" / "DAY25_RESULTS_AFTER_OPTIMIZATION.md"
RAW_PATH = ROOT / "data" / "day25_live_results_after_optimization.json"

CONTROL_REPORT_PATH = ROOT / "docs" / "DAY25_CONTROL_AFTER_FIX.md"
CONTROL_RAW_PATH = ROOT / "data" / "day25_control_after_fix.json"


SCENARIOS = [
    {
        "id": "portfolio-methods",
        "title": "Портфельные модели и оценка фондов",
        "goal": (
            "Подготовить по локальной базе краткую сравнительную справку о моделях формирования "
            "портфеля и методах оценки российских ПИФов, не теряя исходную цель при уточнениях."
        ),
        "plan": (
            "Сначала разобрать модели и диверсификацию из пособия ИТМО, затем перейти к выборке "
            "и показателям оценки ПИФов, после чего собрать общий вывод только по источникам."
        ),
        "turns": [
            ("Какие три модели формирования оптимального портфеля вынесены в отдельные подразделы пособия ИТМО?", "itmo.pdf"),
            ("Чем вторая из них отличается от первой в постановке задачи?", "itmo.pdf"),
            ("Как пособие интерпретирует положительную и отрицательную ковариацию доходностей двух бумаг?", "itmo.pdf"),
            ("Как это объяснение связано с диверсификацией портфеля?", "itmo.pdf"),
            ("При каком состоянии рынка рациональна пассивная стратегия и что она предполагает?", "itmo.pdf"),
            ("Сформулируй промежуточный вывод по первой части нашей задачи в двух пунктах.", "itmo.pdf"),
            ("Теперь перейдём ко второй части: на какой выборке и за какой период тестировали методы оценки российских ПИФов?", "metody.pdf"),
            ("А какую безрисковую ставку и какой бенчмарк там использовали?", "metody.pdf"),
            ("Какой вывод получен о рэнкингах по двусторонним и односторонним мерам риска?", "metody.pdf"),
            ("Собери итоговую сравнительную справку по нашей цели: три коротких пункта, только подтверждённые базой выводы.", None),
        ],
    },
    {
        "id": "alternatives-and-boundary",
        "title": "Альтернативные инвестиции и граница базы",
        "goal": (
            "Собрать по локальной базе компактную записку об альтернативных инвестициях и сопоставить "
            "её с данными исследования ПИФов; отдельно проверить одноразовый ответ вне базы."
        ),
        "plan": (
            "Извлечь автора и числовые оценки альтернативных инвестиций, затем сведения о ПИФах, "
            "один раз явно выйти из базы и убедиться, что следующий вопрос снова использует RAG."
        ),
        "turns": [
            ("Кто написал статью об альтернативных инвестициях и с какой кафедрой и вузом связан автор?", "alternativnye.pdf"),
            ("Какие инструменты он относит к категории 1 и каков объём вложений в неё по итогам III квартала 2021 года?", "alternativnye.pdf"),
            ("Какую оценку доли и объёма иностранных структурных продуктов он приводит?", "alternativnye.pdf"),
            ("Сопоставь упомянутые 18%, 75%, 0,234 и 0,24 трлн рублей, не смешивая их значения.", "alternativnye.pdf"),
            ("На какие категории были разделены ПИФы перед случайным отбором фондов?", "metody.pdf"),
            ("Как в этом исследовании получили безрисковую ставку 7,2%?", "metody.pdf"),
            ("Какой индекс использовали как бенчмарк и для чего он был нужен в сравнении?", "metody.pdf"),
            ("Ответь вне базы знаний: напомни цель текущей задачи и назови столицу Австралии.", None),
            ("Теперь вернись к базе: повтори категории ПИФов, но сгруппируй их в одной строке.", "metody.pdf"),
            ("Заверши нашу записку тремя пунктами: два вывода из статьи об альтернативных инвестициях и один из исследования ПИФов.", None),
        ],
    },
]

CONTROL_SCENARIOS = [
    {
        "id": "evidence-control",
        "title": "Контроль исправлений доказательного RAG",
        "goal": (
            "Проверить исправления ссылок, повторной генерации и накопления доказательств "
            "на связанных вопросах по пособию ИТМО и исследованию российских ПИФов."
        ),
        "plan": (
            "Получить четыре проверяемых факта из двух документов, затем составить краткий "
            "итог только по уже подтверждённым в диалоге сведениям."
        ),
        "turns": [
            ("Как пособие интерпретирует положительную и отрицательную ковариацию доходностей двух бумаг?", "itmo.pdf"),
            ("При каком состоянии рынка рациональна пассивная стратегия и что она предполагает?", "itmo.pdf"),
            ("На какой выборке и за какой период тестировали методы оценки российских ПИФов?", "metody.pdf"),
            ("Какую безрисковую ставку и какой бенчмарк использовали в исследовании ПИФов?", "metody.pdf"),
            ("Собери итог проверки в четырёх коротких пунктах, используя только уже подтверждённые сведения этого диалога.", None),
        ],
    },
]


def source_names(snapshot: dict[str, Any]) -> list[str]:
    return sorted({str(item.get("source") or "") for item in snapshot.get("chunks") or [] if item.get("source")})


def usage_total(message: dict[str, Any]) -> int:
    return int((message.get("technical") or {}).get("usage", {}).get("total_tokens", 0) or 0)


def write_json(items: list[dict[str, Any]], raw_path: Path) -> None:
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(json.dumps({"scenarios": items}, ensure_ascii=False, indent=2), encoding="utf-8")


def yes(value: bool) -> str:
    return "да" if value else "нет"


def report(
    items: list[dict[str, Any]],
    settings: dict[str, Any],
    report_path: Path,
    *,
    control: bool = False,
) -> None:
    turns = [turn for scenario in items for turn in scenario["turns"]]
    regular = [turn for turn in turns if not turn["outside_requested"]]
    answered = [turn for turn in regular if turn["evidence_status"] == "answer"]
    unknown = [turn for turn in regular if turn["evidence_status"] == "unknown"]
    blocked = [turn for turn in regular if turn["request_status"] == "blocked"]
    valid = [turn for turn in answered if turn["evidence_valid"]]
    expected_source_turns = [turn for turn in regular if turn.get("expected_source")]
    source_matches = [turn for turn in expected_source_turns if turn["expected_source_found"]]
    contextual = [turn for turn in regular if turn.get("contextual_query")]
    compact = all(not chunk.get("has_text") for turn in turns for chunk in turn["chunks"])
    state_ok = all(turn["task_goal_preserved"] for turn in turns)
    outside = [turn for turn in turns if turn["outside_requested"]]
    outside_labeled = all(turn["outside_labeled"] for turn in outside)
    after_outside = next((turn for turn in turns if turn.get("after_outside")), None)
    total_tokens = sum(turn["tokens"] for turn in turns)
    rewrite_tokens = sum(turn.get("rewrite_tokens", 0) for turn in turns)

    lines = [
        "# День 25. Короткий контроль после финальных исправлений" if control else "# День 25. Повторный прогон после оптимизации RAG",
        "",
        f"> Выполнено: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"> Модель: `{settings['model']}`; стратегия контекста: `sliding`, окно 5 обменов.",
        "> RAG: `combined`, candidate top-K 20, final top-K 5, similarity 0.83, проверяемые цитаты включены.",
        "",
        "## Сводка",
        "",
        "| Проверка | Результат |",
        "| --- | --- |",
        f"| Сценарии / сообщения | {len(items)} / {len(turns)} |",
        f"| Успешные документальные ответы | {len(answered)}/{len(regular)} |",
        f"| Безопасные отказы / блокировки Task State | {len(unknown)} / {len(blocked)} |",
        f"| Программно валидные источники и цитаты | {len(valid)}/{len(answered)} |",
        f"| Ожидаемый PDF присутствует в retrieval | {len(source_matches)}/{len(expected_source_turns)} |",
        f"| Contextual-поиск зафиксирован | {len(contextual)}/{len(regular)} документальных ходов |",
        f"| Цель Task State сохранилась | {yes(state_ok)} |",
        f"| В JSON-снимках нет полного текста чанков | {yes(compact)} |",
        f"| Одноразовый выход из базы маркирован | {'не проверялся' if control else yes(outside_labeled)} |",
        f"| Следующий ход вернулся к RAG | {'не проверялся' if control else yes(bool(after_outside and after_outside['rag_used']))} |",
        f"| Суммарные токены штатных ответов и валидаторов | {total_tokens} |",
        f"| Дополнительные токены query rewrite | {rewrite_tokens} |",
        "| Базовый прогон до оптимизации | 5/19 ответов; 12/17 ожидаемых источников; 13 отказов; 1 блокировка |",
        "",
        "`unknown` является безопасным результатом пороговой проверки, но для вопросов, намеренно составленных по базе, он считается потерей полезного ответа.",
        "",
    ]

    for scenario in items:
        lines.extend([
            f"## Сценарий: {scenario['title']}",
            "",
            f"**Task State — цель:** {scenario['goal']}",
            "",
            f"**План:** {scenario['plan']}",
            "",
            "| № | Статус | Источники | Ожидаемый источник | Контекстный поиск | Токены |",
            "| ---: | --- | --- | --- | --- | ---: |",
        ])
        for turn in scenario["turns"]:
            status = "вне базы" if turn["outside_requested"] else turn["evidence_status"] or turn["request_status"]
            sources = ", ".join(turn["sources"]) or "—"
            expected = turn.get("expected_source") or "—"
            expected_mark = f"{expected} ({'да' if turn['expected_source_found'] else 'нет'})" if turn.get("expected_source") else expected
            lines.append(
                f"| {turn['number']} | {status} | {sources} | {expected_mark} | "
                f"{yes(bool(turn.get('contextual_query')))} | {turn['tokens']} |"
            )
        lines.append("")

        for turn in scenario["turns"]:
            lines.extend([
                f"### {turn['number']}. {turn['question']}",
                "",
                turn["answer"],
                "",
                f"- Request status: `{turn['request_status']}`.",
                f"- Evidence: `{turn['evidence_status'] or 'не применялся'}`; valid: `{turn['evidence_valid']}`.",
                f"- Причина evidence: `{turn.get('evidence_reason') or '—'}`; ошибки: `{turn.get('evidence_errors') or []}`.",
                f"- Источники retrieval: {', '.join(turn['sources']) or 'не использовались'}.",
                f"- Gate: `{turn.get('gate_phase') or '—'}`; max similarity: `{turn.get('max_similarity')}`.",
                f"- Direct query: `{turn.get('direct_query') or '—'}`.",
                f"- Contextual query: `{turn.get('contextual_query') or '—'}`.",
                f"- Evidence ledger candidates: `{turn.get('ledger_candidate_count') or 0}`.",
                f"- Task validator retry: `{turn.get('policy_validation_retried')}`; причина блокировки: `{turn.get('policy_reason') or '—'}`.",
                f"- Task State сохранил цель: {yes(turn['task_goal_preserved'])}.",
                f"- Полный текст чанков попал в JSON: {yes(any(chunk['has_text'] for chunk in turn['chunks']))}.",
                "",
            ])

    lines.extend([
        "## Автоматический вывод",
        "",
        f"1. Task State сохранял исходную цель на всех {len(turns)} ходах: **{yes(state_ok)}**.",
        f"2. Серверная проверка источников и дословных цитат прошла для {len(valid)} из {len(answered)} выданных документальных ответов.",
        f"3. На {len(unknown)} документальных вопросах сработал безопасный ответ «не знаю».",
        f"4. Ожидаемый файл оказался среди найденных чанков в {len(source_matches)} из {len(expected_source_turns)} проверяемых ходов.",
        ("5. Одноразовый ответ вне базы в короткий контроль не входил."
         if control else
         f"5. Одноразовый ответ вне базы был маркирован, а следующий ход вернулся к RAG: **{yes(outside_labeled and bool(after_outside and after_outside['rag_used']))}**."),
        "6. Это техническая и детерминированная проверка контракта. Смысловую полноту длинных итоговых ответов следует оценить по приведённому выше тексту.",
        "",
    ])
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="разрешить реальные вызовы DeepSeek")
    parser.add_argument("--control", action="store_true", help="выполнить короткий контроль из пяти сообщений")
    args = parser.parse_args()
    if not args.live:
        print("Добавьте --live: сценарий выполняет реальные платные/сетевые вызовы DeepSeek.")
        return 2

    load_dotenv(ROOT / ".env")
    from agent import Agent, AgentSettings, DeepSeekProvider
    from app import create_app
    from document_index import RagIndexService

    settings = AgentSettings(
        model="deepseek-v4-flash",
        system_prompt=(
            "Отвечай на русском языке. При наличии проверяемого RAG строго следуй документальному контексту. "
            "Сохраняй цель текущей задачи и явно учитывай уточнения из диалога."
        ),
        temperature=0,
        top_p=1,
        reasoning_enabled=False,
        reasoning_effort="low",
        max_tokens=1_500,
    )
    service = RagIndexService(ROOT / "rag_documents", ROOT / "data" / "rag_index.sqlite3")
    scenarios = CONTROL_SCENARIOS if args.control else SCENARIOS
    report_path = CONTROL_REPORT_PATH if args.control else REPORT_PATH
    raw_path = CONTROL_RAW_PATH if args.control else RAW_PATH
    results: list[dict[str, Any]] = []
    previous_was_outside = False

    with tempfile.TemporaryDirectory(prefix="day25-rag-") as directory:
        app = create_app(Path(directory), Agent(DeepSeekProvider()), rag_index_service=service)
        app.config.update(TESTING=True)
        client = app.test_client()
        for scenario_index, definition in enumerate(scenarios, 1):
            created = client.post("/api/conversations", json={"title": definition["title"]})
            if created.status_code != 201:
                raise RuntimeError(created.get_json())
            conversation_id = created.get_json()["conversation"]["id"]
            context = client.patch(f"/api/conversations/{conversation_id}/context", json={
                "mode": "sliding", "sliding_window_exchanges": 5, "facts_window_exchanges": 5,
            })
            if context.status_code != 200:
                raise RuntimeError(context.get_json())
            task = client.patch(f"/api/conversations/{conversation_id}/task-state", json={
                "description": definition["goal"],
                "current_step": "Собирать подтверждённые сведения и удерживать цель длинного диалога",
                "expected_action": "Ответить на текущий вопрос с источниками и продолжить общий план",
                "plan": definition["plan"],
            })
            if task.status_code != 200:
                raise RuntimeError(task.get_json())

            scenario_result = {
                "id": definition["id"], "title": definition["title"],
                "goal": definition["goal"], "plan": definition["plan"], "turns": [],
            }
            previous_was_outside = False
            for turn_index, (question, expected_source) in enumerate(definition["turns"], 1):
                outside_requested = "вне базы знаний" in question.lower()
                response = client.post(f"/api/conversations/{conversation_id}/messages", json={
                    "content": question,
                    "settings": settings.to_dict(),
                    "rag": {
                        "enabled": True, "verified": True, "strategy": "combined", "mode": "combined",
                        "routing_mode": "auto", "candidate_k": 20, "final_k": 5, "similarity_threshold": 0.83,
                    },
                })
                payload = response.get_json() or {}
                if response.status_code != 200:
                    raise RuntimeError(f"{definition['id']} turn {turn_index}: {payload}")
                conversation = payload["conversation"]
                assistant = conversation["visible_messages"][-1]
                technical = assistant.get("technical") or {}
                rag = technical.get("rag") or {}
                evidence = rag.get("evidence") or technical.get("evidence") or {}
                task_state = technical.get("task_state") or {}
                sources = source_names(rag)
                chunks = [
                    {
                        "chunk_id": item.get("chunk_id"), "source": item.get("source"),
                        "score": item.get("score"), "fusion_score": item.get("fusion_score"),
                        "reranker_score": item.get("reranker_score"), "has_text": "text" in item,
                    }
                    for item in rag.get("chunks") or []
                ]
                result = {
                    "number": turn_index,
                    "question": question,
                    "answer": assistant.get("content") or "",
                    "expected_source": expected_source,
                    "expected_source_found": bool(expected_source and any(source.endswith(expected_source) for source in sources)),
                    "request_status": technical.get("request_status"),
                    "evidence_status": evidence.get("status"),
                    "evidence_valid": evidence.get("valid") is True,
                    "evidence_reason": evidence.get("reason") or evidence.get("retry_reason"),
                    "evidence_errors": evidence.get("errors") or ([evidence.get("retry_error")] if evidence.get("retry_error") else []),
                    "sources": sources,
                    "chunks": chunks,
                    "direct_query": rag.get("direct_query"),
                    "contextual_query": rag.get("contextual_query"),
                    "gate_phase": rag.get("gate_phase"),
                    "max_similarity": rag.get("max_similarity"),
                    "ledger_candidate_count": rag.get("ledger_candidate_count"),
                    "outside_requested": outside_requested,
                    "outside_labeled": bool(
                        technical.get("outside_local_rag")
                        and str(assistant.get("content") or "").startswith("Ответ вне локальной базы знаний.")
                        and str(assistant.get("content") or "").endswith("Источники локальной базы: не использовались.")
                    ),
                    "after_outside": previous_was_outside,
                    "rag_used": bool(rag.get("chunks")),
                    "task_goal_preserved": task_state.get("description") == definition["goal"],
                    "policy_validation_retried": technical.get("policy_validation_retried") is True,
                    "policy_reason": (technical.get("policy_audit") or {}).get("reason"),
                    "rewrite_tokens": int((((rag.get("rewrite") or {}).get("technical") or {}).get("usage") or {}).get("total_tokens", 0) or 0),
                    "tokens": usage_total(assistant),
                }
                scenario_result["turns"].append(result)
                previous_was_outside = outside_requested
                write_json(results + [scenario_result], raw_path)
                print(
                    f"[{scenario_index}/{len(scenarios)}:{turn_index}/{len(definition['turns'])}] "
                    f"{definition['id']} | {result['evidence_status'] or 'outside'} | "
                    f"sources={len(sources)} | tokens={result['tokens']}",
                    flush=True,
                )
            results.append(scenario_result)
            write_json(results, raw_path)

    report(results, settings.to_dict(), report_path, control=args.control)
    print(f"Report: {report_path}")
    print(f"Raw results: {raw_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
