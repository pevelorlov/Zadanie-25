"""Два реальных длинных сценария Дня 25 для автоматической маршрутизации."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REPORT_PATH = ROOT / "docs" / "DAY25_AUTO_ROUTING_RESULTS.md"
RAW_PATH = ROOT / "data" / "day25_auto_routing_results.json"

SCENARIOS = [
    {
        "id": "portfolio-plan",
        "title": "Портфельный план из двух документов",
        "goal": (
            "Подготовить подтверждённую локальной базой справку о формировании портфеля "
            "и оценке российских ПИФов, сохраняя цель при уточнениях и служебных вопросах."
        ),
        "plan": (
            "Разобрать модели и диверсификацию по пособию ИТМО, проверить состояние задачи, "
            "затем добавить методику исследования ПИФов и собрать итог со ссылками."
        ),
        "turns": [
            {"q": "Какие три модели формирования оптимального портфеля выделены в пособии ИТМО?", "route": "rag", "source": "itmo.pdf"},
            {"q": "Чем вторая из них отличается от первой по постановке задачи?", "route": "rag", "source": "itmo.pdf"},
            {"q": "Как в пособии интерпретируется положительная и отрицательная ковариация доходностей двух бумаг?", "route": "rag", "source": "itmo.pdf"},
            {"q": "Почему это важно именно для диверсификации портфеля?", "route": "rag", "source": "itmo.pdf"},
            {"q": "Что мы сейчас делаем и какова цель текущей задачи?", "route": "task", "source": None},
            {"q": "По пособию ИТМО, при каком состоянии рынка рациональна пассивная стратегия и что она предполагает?", "route": "rag", "source": "itmo.pdf"},
            {"q": "Собери промежуточный вывод по пособию в трёх коротких пунктах, используя уже подтверждённые сведения.", "route": "rag", "source": "itmo.pdf"},
            {"q": "Теперь по исследованию российских ПИФов: на какой выборке и за какой период тестировали методы оценки?", "route": "rag", "source": "metody.pdf"},
            {"q": "А какую безрисковую ставку там приняли и как её получили?", "route": "rag", "source": "metody.pdf"},
            {"q": "Какой индекс использовали как бенчмарк?", "route": "rag", "source": "metody.pdf"},
            {"q": "Какой вывод сделан о рэнкингах по двусторонним и односторонним мерам риска?", "route": "rag", "source": "metody.pdf"},
            {"q": "Собери итоговую справку по нашей цели в четырёх пунктах только по двум использованным документам.", "route": "rag", "source": None},
        ],
    },
    {
        "id": "alternatives-boundary",
        "title": "Альтернативные инвестиции, ПИФы и внешний запрос",
        "goal": (
            "Подготовить записку об альтернативных инвестициях и сопоставить её с исследованием "
            "ПИФов, не потеряв цель после служебного и внешнего запроса."
        ),
        "plan": (
            "Извлечь автора, категории и числовые оценки из статьи, проверить Task State, "
            "добавить сведения о ПИФах, проверить внешний API-маршрут и вернуться к базе."
        ),
        "turns": [
            {"q": "Кто написал статью об альтернативных инвестициях и с какой кафедрой и вузом связан автор?", "route": "rag", "source": "alternativnye.pdf"},
            {"q": "Какие инструменты статья относит к категории 1?", "route": "rag", "source": "alternativnye.pdf"},
            {"q": "Каков объём вложений в эту категорию по итогам III квартала 2021 года?", "route": "rag", "source": "alternativnye.pdf"},
            {"q": "Какую долю и объём иностранных структурных продуктов приводит автор?", "route": "rag", "source": "alternativnye.pdf"},
            {"q": "Сопоставь числа 18%, 75%, 0,234 и 0,24 трлн рублей, не смешивая их смысл.", "route": "rag", "source": "alternativnye.pdf"},
            {"q": "Какова текущая цель задачи и что мы уже успели уточнить?", "route": "task", "source": None},
            {"q": "По исследованию ПИФов, на какие категории разделили фонды перед случайным отбором?", "route": "rag", "source": "metody.pdf"},
            {"q": "Как в этом исследовании получили безрисковую ставку 7,2%?", "route": "rag", "source": "metody.pdf"},
            {"q": "Получи через доступный API актуальную ключевую ставку России и не подменяй вызов догадкой.", "route": "tools", "source": None},
            {"q": "Теперь вернись к локальной базе: какой индекс использовали как бенчмарк и зачем?", "route": "rag", "source": "metody.pdf"},
            {"q": "Сравни только подтверждённые базой сведения из статьи об альтернативных инвестициях и исследования ПИФов.", "route": "rag", "source": None},
            {"q": "Что является целью текущего диалога и какие ограничения мы соблюдаем?", "route": "task", "source": None},
            {"q": "Заверши записку четырьмя пунктами с источниками, не добавляя неподтверждённых актуальных данных.", "route": "rag", "source": None},
        ],
    },
]


def usage(message: dict[str, Any]) -> int:
    return int((((message.get("technical") or {}).get("usage") or {}).get("total_tokens", 0)) or 0)


def write_raw(results: list[dict[str, Any]]) -> None:
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    RAW_PATH.write_text(json.dumps({"scenarios": results}, ensure_ascii=False, indent=2), encoding="utf-8")


def build_report(results: list[dict[str, Any]], model: str) -> None:
    turns = [turn for scenario in results for turn in scenario["turns"]]
    document_turns = [turn for turn in turns if turn["expected_route"] == "rag"]
    route_matches = [turn for turn in turns if turn["route_match"]]
    source_marked = [turn for turn in turns if turn["source_marker"]]
    expected_sources = [turn for turn in turns if turn.get("expected_source")]
    source_matches = [turn for turn in expected_sources if turn["expected_source_found"]]
    answered_docs = [turn for turn in document_turns if turn["evidence_status"] == "answer"]
    valid_docs = [turn for turn in answered_docs if turn["evidence_valid"]]
    unknown_docs = [turn for turn in document_turns if turn["evidence_status"] == "unknown"]
    goals_kept = all(turn["goal_preserved"] for turn in turns)
    route_counts = Counter(turn["actual_route"] for turn in turns)
    total_tokens = sum(turn["tokens"] for turn in turns)
    router_tokens = sum(turn["router_tokens"] for turn in turns)

    lines = [
        "# День 25. Два длинных сценария с автоматической маршрутизацией",
        "",
        f"> Выполнено: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"> Модель: `{model}`. Выполнены реальные сетевые вызовы DeepSeek API.",
        "> RAG: Auto routing, verified, combined, candidate top-K 20, final top-K 5, threshold 0.83.",
        "",
        "## Общая сводка",
        "",
        "| Проверка | Результат |",
        "| --- | --- |",
        f"| Сценарии / сообщения | {len(results)} / {len(turns)} |",
        f"| Ожидаемый маршрут выбран | {len(route_matches)}/{len(turns)} |",
        f"| В каждом ответе есть маркировка источника | {len(source_marked)}/{len(turns)} |",
        f"| Документальные ответы / безопасные отказы | {len(answered_docs)}/{len(document_turns)} / {len(unknown_docs)} |",
        f"| Валидные цитаты у документальных ответов | {len(valid_docs)}/{len(answered_docs)} |",
        f"| Ожидаемый PDF найден в retrieval | {len(source_matches)}/{len(expected_sources)} |",
        f"| Цель Task State сохранилась на всех ходах | {'да' if goals_kept else 'нет'} |",
        f"| Маршруты | {dict(route_counts)} |",
        f"| Токены основных ответов и валидаторов | {total_tokens} |",
        f"| Токены неоднозначного route-классификатора | {router_tokens} |",
        "",
    ]

    for scenario in results:
        lines.extend([
            f"## Сценарий: {scenario['title']}",
            "",
            f"**Цель:** {scenario['goal']}",
            "",
            f"**План:** {scenario['plan']}",
            "",
            "| № | Ожидался | Выбран | Источник указан | PDF найден | Evidence |",
            "| ---: | --- | --- | --- | --- | --- |",
        ])
        for turn in scenario["turns"]:
            lines.append(
                f"| {turn['number']} | {turn['expected_route']} | {turn['actual_route']} | "
                f"{'да' if turn['source_marker'] else 'нет'} | "
                f"{'да' if turn['expected_source_found'] else ('—' if not turn.get('expected_source') else 'нет')} | "
                f"{turn['evidence_status'] or 'не требуется'} |"
            )
        lines.append("")
        for turn in scenario["turns"]:
            lines.extend([
                f"### {turn['number']}. {turn['question']}",
                "",
                turn["answer"],
                "",
                f"- Маршрут: ожидался `{turn['expected_route']}`, выбран `{turn['actual_route']}` "
                f"(`{turn['classifier']}`, confidence `{turn['confidence']}`).",
                f"- Причина маршрута: `{turn['route_reason']}`.",
                f"- Retrieval sources: {', '.join(turn['retrieval_sources']) or 'нет'}.",
                f"- Evidence: `{turn['evidence_status'] or 'не применялся'}`, valid: `{turn['evidence_valid']}`.",
                f"- Источник маркирован в ответе: {'да' if turn['source_marker'] else 'нет'}.",
                f"- Task State сохранил цель: {'да' if turn['goal_preserved'] else 'нет'}.",
                f"- Токены ответа: {turn['tokens']}; роутера: {turn['router_tokens']}.",
                "",
            ])

    lines.extend([
        "## Автоматическое сравнение сценариев",
        "",
    ])
    for scenario in results:
        scenario_turns = scenario["turns"]
        docs = [turn for turn in scenario_turns if turn["expected_route"] == "rag"]
        valid = [turn for turn in docs if turn["evidence_status"] == "answer" and turn["evidence_valid"]]
        lines.append(
            f"- **{scenario['title']}**: маршруты {sum(t['route_match'] for t in scenario_turns)}/{len(scenario_turns)}, "
            f"источники {sum(t['source_marker'] for t in scenario_turns)}/{len(scenario_turns)}, "
            f"проверенные документальные ответы {len(valid)}/{len(docs)}."
        )
    lines.extend([
        "",
        "`unknown` считается безопасным поведением анти-галлюцинационного режима, но для вопроса, "
        "заранее составленного по базе, одновременно означает потерю полезного ответа. Маршрут `tools` "
        "считается корректным даже при недоступном MCP, если ассистент не выдумал внешние данные и явно "
        "указал отсутствие фактического источника.",
        "",
    ])
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="разрешить реальные платные вызовы DeepSeek")
    args = parser.parse_args()
    if not args.live:
        print("Добавьте --live: сценарии выполняют реальные платные вызовы DeepSeek.")
        return 2

    load_dotenv(ROOT / ".env")
    from agent import Agent, AgentSettings, DeepSeekProvider
    from app import create_app
    from document_index import RagIndexService

    settings = AgentSettings(
        model="deepseek-v4-flash",
        system_prompt=(
            "Отвечай на русском языке. Удерживай цель длинного диалога. Для локальных документов "
            "используй только переданный RAG-контекст; внешние данные не выдумывай."
        ),
        temperature=0,
        top_p=1,
        reasoning_enabled=False,
        reasoning_effort="low",
        max_tokens=1_500,
    )
    service = RagIndexService(ROOT / "rag_documents", ROOT / "data" / "rag_index.sqlite3")
    results: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="day25-auto-") as directory:
        app = create_app(Path(directory), Agent(DeepSeekProvider()), rag_index_service=service)
        app.config.update(TESTING=True)
        client = app.test_client()
        for scenario_index, definition in enumerate(SCENARIOS, 1):
            created = client.post("/api/conversations", json={"title": definition["title"]})
            conversation_id = created.get_json()["conversation"]["id"]
            client.patch(f"/api/conversations/{conversation_id}/context", json={
                "mode": "sliding", "sliding_window_exchanges": 5, "facts_window_exchanges": 5,
            })
            task_response = client.patch(f"/api/conversations/{conversation_id}/task-state", json={
                "description": definition["goal"],
                "current_step": "Последовательно собирать подтверждённые сведения и удерживать цель",
                "expected_action": "Ответить на текущий вопрос и продолжить согласованный план",
                "plan": definition["plan"],
            })
            if task_response.status_code != 200:
                raise RuntimeError(task_response.get_json())

            scenario_result = {
                "id": definition["id"], "title": definition["title"],
                "goal": definition["goal"], "plan": definition["plan"], "turns": [],
            }
            for turn_index, definition_turn in enumerate(definition["turns"], 1):
                response = client.post(f"/api/conversations/{conversation_id}/messages", json={
                    "content": definition_turn["q"],
                    "settings": settings.to_dict(),
                    "rag": {
                        "enabled": True, "verified": True, "routing_mode": "auto",
                        "strategy": "combined", "mode": "combined",
                        "candidate_k": 20, "final_k": 5, "similarity_threshold": 0.83,
                    },
                })
                payload = response.get_json() or {}
                if response.status_code != 200:
                    raise RuntimeError(f"{definition['id']} turn {turn_index}: {payload}")
                conversation = payload["conversation"]
                assistant = conversation["visible_messages"][-1]
                technical = assistant.get("technical") or {}
                rag = technical.get("rag") or {}
                routing = rag.get("routing") or {}
                evidence = rag.get("evidence") or technical.get("evidence") or {}
                chunks = rag.get("chunks") or []
                sources = sorted({str(item.get("source")) for item in chunks if item.get("source")})
                answer = str(assistant.get("content") or "")
                expected_source = definition_turn.get("source")
                route_technical = routing.get("technical") or {}
                result = {
                    "number": turn_index,
                    "question": definition_turn["q"],
                    "answer": answer,
                    "expected_route": definition_turn["route"],
                    "actual_route": routing.get("route") or "missing",
                    "route_match": routing.get("route") == definition_turn["route"],
                    "classifier": routing.get("classifier"),
                    "confidence": routing.get("confidence"),
                    "route_reason": routing.get("reason"),
                    "expected_source": expected_source,
                    "expected_source_found": bool(
                        expected_source and any(source.endswith(expected_source) for source in sources)
                    ),
                    "retrieval_sources": sources,
                    "source_marker": "Источники:" in answer or "Источники маршрута:" in answer,
                    "evidence_status": evidence.get("status"),
                    "evidence_valid": evidence.get("valid") is True,
                    "goal_preserved": (technical.get("task_state") or {}).get("description") == definition["goal"],
                    "tokens": usage(assistant),
                    "router_tokens": int(((route_technical.get("usage") or {}).get("total_tokens", 0)) or 0),
                }
                scenario_result["turns"].append(result)
                write_raw(results + [scenario_result])
                print(
                    f"[{scenario_index}/{len(SCENARIOS)}:{turn_index}/{len(definition['turns'])}] "
                    f"{definition['id']} | route={result['actual_route']} | "
                    f"evidence={result['evidence_status'] or '-'} | source={result['source_marker']}",
                    flush=True,
                )
            results.append(scenario_result)
            write_raw(results)

    build_report(results, settings.model)
    print(f"Report: {REPORT_PATH}")
    print(f"Raw: {RAW_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
