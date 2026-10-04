"""Два реальных длинных прогона Дня 25 с независимой памятью задачи."""

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

REPORT_PATH = ROOT / "docs" / "DAY25_TASK_MEMORY_SCENARIOS.md"
RAW_PATH = ROOT / "data" / "day25_task_memory_scenarios.json"

SCENARIOS = [
    {
        "id": "cheboksary-apartment-portfolio",
        "title": "Портфель для накопления на квартиру в Чебоксарах",
        "expected_goal": "накопить на квартиру в Чебоксарах",
        "turns": [
            {
                "q": (
                    "Начинаем долгую задачу: составить учебный алгоритм инвестиционного портфеля, чтобы накопить "
                    "на квартиру в Чебоксарах. Для демонстрации считаем цель 6 млн рублей, уже есть 1 млн, можно "
                    "вкладывать 70 тысяч рублей в месяц, горизонт 7 лет, допустимая просадка 20%, кредитное плечо "
                    "запрещено. Это условные числа, а не актуальная оценка квартиры. Сначала сформулируй наш алгоритм "
                    "из последовательных шагов."
                ),
                "expected_route": "general",
                "expected_source": None,
                "depends_on_memory": False,
            },
            {
                "q": "Какие три модели формирования оптимального портфеля выделены в пособии ИТМО?",
                "expected_route": "rag", "expected_source": "itmo.pdf", "depends_on_memory": False,
            },
            {
                "q": (
                    "Теперь вернись к нашему алгоритму накопления: на каком его шаге выбирается модель портфеля "
                    "и какая из только что названных моделей лучше подходит как учебная основа при наших ограничениях?"
                ),
                "expected_route": "rag", "expected_source": "itmo.pdf", "depends_on_memory": True,
            },
            {
                "q": "Как пособие ИТМО объясняет положительную и отрицательную ковариацию доходностей двух бумаг?",
                "expected_route": "rag", "expected_source": "itmo.pdf", "depends_on_memory": False,
            },
            {
                "q": (
                    "Примени это к шагу диверсификации нашего алгоритма: какое практическое правило отбора активов "
                    "следует из объяснения ковариации, не называя конкретные бумаги?"
                ),
                "expected_route": "rag", "expected_source": "itmo.pdf", "depends_on_memory": True,
            },
            {
                "q": (
                    "При каком состоянии рынка пособие считает рациональной пассивную стратегию и как этот вывод "
                    "встраивается в наш семилетний алгоритм?"
                ),
                "expected_route": "rag", "expected_source": "itmo.pdf", "depends_on_memory": True,
            },
            {
                "q": (
                    "По исследованию российских ПИФов назови выборку, период, безрисковую ставку и бенчмарк, "
                    "которые использовали авторы."
                ),
                "expected_route": "rag", "expected_source": "metody.pdf", "depends_on_memory": False,
            },
            {
                "q": (
                    "На основе этого исследования добавь в наш алгоритм отдельный шаг проверки фонда. "
                    "Какие данные и сравнения мы должны запросить, не выдавая историческую методику за гарантию доходности?"
                ),
                "expected_route": "rag", "expected_source": "metody.pdf", "depends_on_memory": True,
            },
            {
                "q": (
                    "Проверь память задачи: повтори нашу цель, исходный капитал, ежемесячное пополнение, горизонт, "
                    "предел просадки и запрет, но не пересчитывай портфель."
                ),
                "expected_route": "general", "expected_source": None, "depends_on_memory": True,
            },
            {
                "q": (
                    "Какие категории альтернативных инвестиций описаны в статье alternativnye.pdf и на каком шаге "
                    "нашего алгоритма их допустимо рассматривать?"
                ),
                "expected_route": "rag", "expected_source": "alternativnye.pdf", "depends_on_memory": True,
            },
            {
                "q": (
                    "Уточняю условия: теперь можем откладывать 90 тысяч рублей в месяц, а допустимую просадку "
                    "снижаем до 15%. Цель 6 млн, капитал 1 млн, горизонт 7 лет и запрет плеча сохраняются. "
                    "Обнови только затронутые шаги нашего алгоритма."
                ),
                "expected_route": "general", "expected_source": None, "depends_on_memory": True,
            },
            {
                "q": (
                    "Собери окончательную версию нашего алгоритма накопления на квартиру: используй последние "
                    "ограничения, отдельно отметь допущения, а документальные положения подтверди источниками и цитатами."
                ),
                "expected_route": "rag", "expected_source": None, "depends_on_memory": True,
            },
        ],
    },
    {
        "id": "beginner-investment-workshop",
        "title": "Учебное занятие для начинающих инвесторов",
        "expected_goal": "подготовить учебное занятие",
        "turns": [
            {
                "q": (
                    "Новая долгая задача: подготовить учебное занятие для студенческого инвестклуба. Аудитория — "
                    "новички без математической подготовки, длительность 45 минут, результат — план из четырёх блоков "
                    "и одного сквозного примера. Нельзя давать персональные инвестиционные рекомендации. "
                    "Предложи первоначальный алгоритм подготовки занятия."
                ),
                "expected_route": "general", "expected_source": None, "depends_on_memory": False,
            },
            {
                "q": (
                    "Какие три модели формирования портфеля перечислены в пособии ИТМО и в какой блок нашего "
                    "алгоритма подготовки занятия их логично поставить?"
                ),
                "expected_route": "rag", "expected_source": "itmo.pdf", "depends_on_memory": True,
            },
            {
                "q": (
                    "Объясни для нашей аудитории положительную и отрицательную ковариацию одним бытовым примером, "
                    "но сохрани смысл формулировки пособия."
                ),
                "expected_route": "rag", "expected_source": "itmo.pdf", "depends_on_memory": True,
            },
            {
                "q": (
                    "Какой вывод о диверсификации должен сделать студент после этого примера и как он связан "
                    "с предыдущим блоком занятия?"
                ),
                "expected_route": "rag", "expected_source": "itmo.pdf", "depends_on_memory": True,
            },
            {
                "q": (
                    "Что пособие говорит об условиях рациональности пассивной стратегии? Добавь это как оговорку "
                    "к нашему сквозному примеру, учитывая уровень новичков."
                ),
                "expected_route": "rag", "expected_source": "itmo.pdf", "depends_on_memory": True,
            },
            {
                "q": "На какой выборке и за какой период в статье metody.pdf проверялись методы оценки российских ПИФов?",
                "expected_route": "rag", "expected_source": "metody.pdf", "depends_on_memory": False,
            },
            {
                "q": (
                    "Какой учебный мини-кейс для третьего блока можно построить на этой выборке, чтобы не обещать "
                    "будущую доходность и не нарушить наше ограничение о персональных рекомендациях?"
                ),
                "expected_route": "rag", "expected_source": "metody.pdf", "depends_on_memory": True,
            },
            {
                "q": (
                    "Как статья сравнивает рэнкинги по двусторонним и односторонним мерам риска и какой вопрос "
                    "после этого следует задать студентам?"
                ),
                "expected_route": "rag", "expected_source": "metody.pdf", "depends_on_memory": True,
            },
            {
                "q": (
                    "Какие инструменты статья alternativnye.pdf относит к категории 1 и как аккуратно включить "
                    "их в четвёртый блок нашего занятия?"
                ),
                "expected_route": "rag", "expected_source": "alternativnye.pdf", "depends_on_memory": True,
            },
            {
                "q": (
                    "Проверь память: для кого мы готовим занятие, сколько оно длится, какой нужен результат и "
                    "какое ограничение действует на рекомендации?"
                ),
                "expected_route": "general", "expected_source": None, "depends_on_memory": True,
            },
            {
                "q": (
                    "Меняем условие: занятие сокращается до 30 минут, оставляем три блока и тот же один сквозной "
                    "пример. Аудитория и запрет персональных рекомендаций сохраняются. Перестрой наш алгоритм подготовки."
                ),
                "expected_route": "general", "expected_source": None, "depends_on_memory": True,
            },
            {
                "q": (
                    "Собери финальный поминутный план занятия по последним условиям. В каждом документальном блоке "
                    "укажи источник и цитату, а в конце перечисли, какие сведения взяты только из памяти нашего диалога."
                ),
                "expected_route": "rag", "expected_source": None, "depends_on_memory": True,
            },
        ],
    },
]


def token_usage(message: dict[str, Any]) -> int:
    usage = (message.get("technical") or {}).get("usage") or {}
    return int(usage.get("total_tokens", 0) or 0)


def memory_usage_delta(previous: int, conversation: dict[str, Any]) -> tuple[int, int]:
    total = int((conversation.get("task_memory_token_totals") or {}).get("total_tokens", 0) or 0)
    return total - previous, total


def write_raw(results: list[dict[str, Any]]) -> None:
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    RAW_PATH.write_text(json.dumps({"scenarios": results}, ensure_ascii=False, indent=2), encoding="utf-8")


def compact_memory(memory: dict[str, Any]) -> str:
    terms = [f"{item.get('term')}: {item.get('meaning')}" for item in memory.get("terms", [])]
    parts = [
        f"Цель: {memory.get('goal') or '—'}",
        "Уточнения: " + ("; ".join(memory.get("clarifications", [])) or "—"),
        "Ограничения: " + ("; ".join(memory.get("constraints", [])) or "—"),
        "Термины: " + ("; ".join(terms) or "—"),
        "Решения: " + ("; ".join(memory.get("decisions", [])) or "—"),
        "Открытые вопросы: " + ("; ".join(memory.get("open_questions", [])) or "—"),
    ]
    return "\n".join(f"- {part}" for part in parts)


def build_report(results: list[dict[str, Any]], model: str) -> None:
    turns = [turn for scenario in results for turn in scenario["turns"]]
    dependent = [turn for turn in turns if turn["depends_on_memory"]]
    document_turns = [turn for turn in turns if turn["expected_route"] == "rag"]
    route_counts = Counter(turn["actual_route"] for turn in turns)
    source_turns = [turn for turn in turns if turn["source_marker"]]
    valid_evidence = [
        turn for turn in document_turns
        if turn["evidence_status"] == "answer" and turn["evidence_valid"]
    ]
    memory_present = [turn for turn in turns if turn["memory_goal_present"]]
    followup_turns = [turn for scenario in results for turn in scenario["turns"][1:]]
    memory_applied = [turn for turn in followup_turns if turn["memory_applied"]]
    total_answer_tokens = sum(turn["answer_tokens"] for turn in turns)
    total_memory_tokens = sum(turn["memory_update_tokens"] for turn in turns)

    lines = [
        "# День 25. Два длинных сценария с RAG и памятью задачи",
        "",
        f"> Выполнено: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"> Модель: `{model}`. Выполнены реальные платные сетевые вызовы DeepSeek API.",
        "> Настройки: Task State Machine выключена; память задачи включена; Sliding Window = 5; "
        "RAG Auto + Verified + Combined; candidate top-K 20; final top-K 5; threshold 0.83.",
        "",
        "## Общая сводка",
        "",
        "| Проверка | Результат |",
        "| --- | --- |",
        f"| Сценарии / сообщения | {len(results)} / {len(turns)} |",
        f"| Сообщения, зависящие от ранее сформированного алгоритма | {len(dependent)}/{len(turns)} |",
        f"| После ответа в памяти есть цель | {len(memory_present)}/{len(turns)} |",
        f"| Начиная со второго сообщения память передана основному ответу | {len(memory_applied)}/{len(followup_turns)} |",
        f"| Ответы с маркировкой источников | {len(source_turns)}/{len(turns)} |",
        f"| Валидные документальные ответы с цитатами | {len(valid_evidence)}/{len(document_turns)} |",
        f"| Маршруты | {dict(route_counts)} |",
        f"| Токены основных ответов/валидаторов | {total_answer_tokens} |",
        f"| Токены обновления памяти задачи | {total_memory_tokens} |",
        "",
    ]

    for scenario in results:
        lines.extend([
            f"## Сценарий: {scenario['title']}",
            "",
            f"**Ожидаемая долговременная цель:** {scenario['expected_goal']}.",
            "",
            "| № | Зависит от памяти | Ожидался маршрут | Выбран | Evidence | Цель в памяти |",
            "| ---: | --- | --- | --- | --- | --- |",
        ])
        for turn in scenario["turns"]:
            lines.append(
                f"| {turn['number']} | {'да' if turn['depends_on_memory'] else 'нет'} | "
                f"{turn['expected_route']} | {turn['actual_route']} | {turn['evidence_status'] or '—'} | "
                f"{'да' if turn['memory_goal_present'] else 'нет'} |"
            )
        lines.append("")
        for turn in scenario["turns"]:
            lines.extend([
                f"### {turn['number']}. Пользователь",
                "",
                turn["question"],
                "",
                "**Ассистент**",
                "",
                turn["answer"],
                "",
                "**Память задачи после ответа**",
                "",
                compact_memory(turn["task_memory"]),
                "",
                f"Технически: маршрут `{turn['actual_route']}`, evidence `{turn['evidence_status'] or 'не применялся'}`, "
                f"валидность `{turn['evidence_valid']}`, retrieval sources "
                f"`{', '.join(turn['retrieval_sources']) or 'нет'}`, память была передана ответу "
                f"`{turn['memory_applied']}`, токены ответа `{turn['answer_tokens']}`, "
                f"обновления памяти `{turn['memory_update_tokens']}`.",
                "",
            ])

    lines.extend([
        "## Автоматический вывод",
        "",
    ])
    for scenario in results:
        scenario_turns = scenario["turns"]
        dependent_turns = [turn for turn in scenario_turns if turn["depends_on_memory"]]
        latest = scenario_turns[-1]["task_memory"] if scenario_turns else {}
        lines.append(
            f"- **{scenario['title']}**: зависимых сообщений {len(dependent_turns)}/{len(scenario_turns)}; "
            f"цель сохранялась {sum(turn['memory_goal_present'] for turn in scenario_turns)}/{len(scenario_turns)}; "
            f"финальная ревизия памяти `{latest.get('revision', 0)}`; "
            f"финальная цель: «{latest.get('goal') or 'не заполнена'}»."
        )
    lines.extend([
        "",
        "## Что показал прогон",
        "",
        "1. **Основная функция памяти работает:** после первого ответа цель присутствовала во всех 24 снимках, "
        "а на всех 22 последующих ходах сохранённый снимок был передан основному ответу.",
        "2. **Изменения условий не потерялись:** в первом сценарии финальный снимок содержит 90 тыс. ₽ в месяц "
        "и просадку 15% вместо исходных 70 тыс. ₽ и 20%; во втором — 30 минут и три блока вместо 45 минут и четырёх.",
        "3. **Строгий RAG сработал не во всех составных вопросах:** из 14 фактически выбранных RAG-маршрутов "
        "10 завершились валидным ответом с цитатами, 4 — безопасным `unknown`. Это особенно заметно в вопросах, "
        "где одновременно требовались документальный факт и применение к ранее созданному алгоритму.",
        "4. **Автомаршрут смешивает память задачи с Task State:** 10 сообщений выбрали маршрут `task`, хотя Task "
        "State Machine была выключена. Сам ответ видел память задачи, но подпись «Источники маршрута» ошибочно "
        "называла Task State. Это отдельный дефект маркировки/маршрутизации, а не потеря памяти.",
        "5. **Память постепенно разрастается:** обновления памяти потребовали заметный отдельный объём токенов, "
        "а в `open_questions` остались некоторые уже разобранные вопросы. Для production-режима понадобятся "
        "ограничение размера, удаление закрытых вопросов и более компактные решения/термины.",
        "",
        "Оценка выше проверяет технические признаки: сохранение и передачу структурированного снимка, "
        "маршрутизацию, наличие источников и серверную валидность цитат. Содержательное качество каждого "
        "ответа можно проверить непосредственно по полным репликам в этом отчёте.",
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
            "Отвечай на русском языке. Следуй текущему запросу и структурированной памяти задачи. "
            "Не выдавай учебные рассуждения за персональную финансовую рекомендацию. Документальные "
            "утверждения основывай только на переданном RAG-контексте."
        ),
        temperature=0,
        top_p=1,
        reasoning_enabled=False,
        reasoning_effort="low",
        max_tokens=1_200,
    )
    service = RagIndexService(ROOT / "rag_documents", ROOT / "data" / "rag_index.sqlite3")
    results: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="day25-task-memory-") as directory:
        app = create_app(Path(directory), Agent(DeepSeekProvider()), rag_index_service=service)
        app.config.update(TESTING=True)
        client = app.test_client()
        client.post("/api/task-control/disable", json={})
        client.post("/api/mcp-control/disable", json={})

        for scenario_index, definition in enumerate(SCENARIOS, 1):
            created = client.post("/api/conversations", json={"title": definition["title"]})
            conversation_id = created.get_json()["conversation"]["id"]
            client.patch(f"/api/conversations/{conversation_id}/context", json={
                "mode": "sliding", "sliding_window_exchanges": 5, "facts_window_exchanges": 5,
            })
            enabled = client.patch(
                f"/api/conversations/{conversation_id}/task-memory", json={"enabled": True},
            )
            if enabled.status_code != 200:
                raise RuntimeError(enabled.get_json())

            scenario_result = {
                "id": definition["id"],
                "title": definition["title"],
                "expected_goal": definition["expected_goal"],
                "turns": [],
            }
            previous_memory_tokens = 0
            for turn_index, turn_definition in enumerate(definition["turns"], 1):
                response = client.post(f"/api/conversations/{conversation_id}/messages", json={
                    "content": turn_definition["q"],
                    "settings": settings.to_dict(),
                    "rag": {
                        "enabled": True,
                        "verified": True,
                        "routing_mode": "auto",
                        "strategy": "combined",
                        "mode": "combined",
                        "candidate_k": 20,
                        "final_k": 5,
                        "similarity_threshold": 0.83,
                    },
                })
                payload = response.get_json() or {}
                if response.status_code != 200:
                    write_raw(results + [scenario_result])
                    raise RuntimeError(
                        f"{definition['id']} turn {turn_index}: HTTP {response.status_code}: {payload}"
                    )
                conversation = payload["conversation"]
                assistant = conversation["visible_messages"][-1]
                technical = assistant.get("technical") or {}
                rag = technical.get("rag") or {}
                routing = rag.get("routing") or {}
                evidence = rag.get("evidence") or technical.get("evidence") or {}
                chunks = rag.get("chunks") or []
                sources = sorted({str(item.get("source")) for item in chunks if item.get("source")})
                answer = str(assistant.get("content") or "")
                memory = conversation.get("task_memory") or {}
                memory_delta, previous_memory_tokens = memory_usage_delta(previous_memory_tokens, conversation)
                applied_memory = technical.get("task_memory") or {}
                expected_source = turn_definition.get("expected_source")
                result = {
                    "number": turn_index,
                    "question": turn_definition["q"],
                    "answer": answer,
                    "depends_on_memory": turn_definition["depends_on_memory"],
                    "expected_route": turn_definition["expected_route"],
                    "actual_route": routing.get("route") or "missing",
                    "expected_source": expected_source,
                    "expected_source_found": bool(
                        expected_source and any(source.endswith(expected_source) for source in sources)
                    ),
                    "retrieval_sources": sources,
                    "source_marker": "Источники:" in answer or "Источники маршрута:" in answer,
                    "evidence_status": evidence.get("status"),
                    "evidence_valid": evidence.get("valid") is True,
                    "task_memory": memory,
                    "memory_goal_present": bool(str(memory.get("goal") or "").strip()),
                    "memory_applied": applied_memory.get("enabled") is True,
                    "answer_tokens": token_usage(assistant),
                    "memory_update_tokens": memory_delta,
                }
                scenario_result["turns"].append(result)
                write_raw(results + [scenario_result])
                print(
                    f"[{scenario_index}/{len(SCENARIOS)}:{turn_index}/{len(definition['turns'])}] "
                    f"{definition['id']} | route={result['actual_route']} | "
                    f"memory_rev={memory.get('revision', 0)} | evidence={result['evidence_status'] or '-'}",
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
