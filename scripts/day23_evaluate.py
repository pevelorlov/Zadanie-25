"""Реальный сравнительный прогон пяти RAG-конвейеров Дня 23."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from agent import Agent, AgentSettings, DeepSeekProvider
from document_index import RagIndexService, build_rag_context
from document_index.evaluations import RagEvaluationStore


ROOT = Path(__file__).resolve().parents[1]
MODES = ("baseline", "rewrite", "filter", "rerank", "combined")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="rewrite и пять реальных ответов DeepSeek на вопрос")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--strategy", choices=("fixed", "structural", "combined"), default="combined")
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--final-k", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.83)
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    questions = json.loads((ROOT / "docs" / "day22_control_questions.json").read_text(encoding="utf-8"))["questions"]
    questions = questions[: max(1, min(len(questions), args.limit))]
    service = RagIndexService(ROOT / "rag_documents", ROOT / "data" / "rag_index.sqlite3")
    store = RagEvaluationStore(ROOT / "data" / "rag_evaluations.json")
    existing_ids = {
        item.get("question_id") for item in store.list()
        if item.get("evaluation_mode") == "batch-day23-live"
    }
    agent = Agent(DeepSeekProvider()) if args.live else None
    settings = AgentSettings(
        model="deepseek-v4-flash",
        system_prompt=(
            "Отвечай на вопрос точно и по существу на русском языке. Используй переданный документальный контекст. "
            "Не выдумывай точные числа, названия или выводы, если их нет в найденных фрагментах."
        ),
        temperature=0.1,
        top_p=1.0,
        reasoning_enabled=False,
        reasoning_effort="low",
        max_tokens=900,
    )
    coverage = {mode: 0 for mode in MODES}
    completed = 0
    for index, question in enumerate(questions, 1):
        rewrite_result = agent.rewrite_rag_query(question["question"], settings) if args.live else None
        rewrite = " ".join((rewrite_result.content if rewrite_result else question["question"]).strip().strip('"').split())
        expected = {Path(value).name.lower() for value in question.get("expected_sources", [])}
        pipeline_results = {}
        for mode in MODES:
            retrieval = service.retrieve_pipeline(
                question["question"],
                strategy=args.strategy,
                mode=mode,
                candidate_k=args.candidate_k,
                final_k=args.final_k,
                similarity_threshold=args.threshold,
                rewritten_query=rewrite if mode in {"rewrite", "combined"} else None,
            )
            found = {Path(item["source"]).name.lower() for item in retrieval["chunks"]}
            covered = expected.issubset(found)
            coverage[mode] += int(covered)
            print(
                f"[{index:02d}/{len(questions):02d}] {question['id']} | {mode:8s} "
                f"source={'OK' if covered else 'MISS'} | candidates={retrieval['candidate_count']} "
                f"filtered={retrieval['after_filter_count']} | final={len(retrieval['chunks'])}",
                flush=True,
            )
            answer = None
            if args.live and question["id"] not in existing_ids:
                answer = agent.reply([], question["question"], settings, policy_guidance=build_rag_context(retrieval))
            pipeline_results[mode] = {
                "retrieval": retrieval,
                "answer": ({
                    "content": answer.content,
                    "reasoning_content": answer.reasoning_content,
                    "technical": answer.technical,
                } if answer else None),
            }
        if args.live and question["id"] not in existing_ids:
            store.add({
                "evaluation_mode": "batch-day23-live",
                "conversation_id": None,
                "question_id": question["id"],
                "question": question["question"],
                "expectation": question["expectation"],
                "expected_sources": question.get("expected_sources", []),
                "settings": settings.to_dict(),
                "pipeline_settings": {
                    "strategy": args.strategy,
                    "candidate_k": args.candidate_k,
                    "final_k": args.final_k,
                    "similarity_threshold": args.threshold,
                },
                "rewrite": {"query": rewrite, "technical": rewrite_result.technical},
                "pipeline_results": pipeline_results,
            })
            completed += 1
            print("  saved", flush=True)
    print("Coverage: " + ", ".join(f"{mode}={count}/{len(questions)}" for mode, count in coverage.items()))
    print(f"Live comparisons saved now: {completed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
