"""Measure a staged revision's retrieval quality against the active corpus on fixed questions."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Callable

MODES = ("hybrid", "graph_hybrid")
MAXIMUM_QUESTIONS = 20

Answer = Callable[[str, str], dict[str, Any]]
Judge = Callable[[str, str, dict[str, Any]], dict[str, Any]]


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def normalize_questions(raw: list[dict[str, Any]]) -> list[dict[str, str]]:
    if not raw or len(raw) > MAXIMUM_QUESTIONS:
        raise ValueError(f"Staged evaluation needs 1 to {MAXIMUM_QUESTIONS} fixed questions.")
    questions = [
        {
            "question_id": str(item.get("question_id") or "").strip(),
            "text": str(item.get("text") or "").strip(),
            "category": str(item.get("category") or "general").strip(),
        }
        for item in raw
    ]
    identifiers = [question["question_id"] for question in questions]
    if any(not question["question_id"] or not question["text"] for question in questions):
        raise ValueError("Every staged-evaluation question needs a question_id and text.")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Staged-evaluation question IDs must be unique.")
    return questions


def measure_staged_revision(
    questions: list[dict[str, Any]],
    *,
    staged_document_id: str,
    baseline_answer: Answer,
    preview_answer: Answer,
    judge: Judge,
) -> dict[str, Any]:
    """Answer each question on the active corpus and on the staged preview, then judge both.

    An answer whose text and citations equal one already judged for the same question and mode
    reuses that judgment, so unchanged answers cannot move the ratios through judge noise alone.
    """
    from scripts.run_corpus_evaluation import citation_validity
    from scripts.validate_compact_evaluation import compare_evaluations

    questions = normalize_questions(questions)
    judgments: dict[str, dict[str, Any]] = {}
    reused = 0

    def judged(question: str, mode: str, answer: dict[str, Any]) -> dict[str, Any]:
        nonlocal reused
        key = _digest([question, mode, answer.get("answer"), answer.get("citations")])
        if key in judgments:
            reused += 1
        else:
            judgments[key] = judge(question, mode, answer)
        return judgments[key]

    reports: dict[str, list[dict[str, Any]]] = {"baseline": [], "candidate": []}
    summary_rows = []
    staged_citations = 0
    for question in questions:
        summary: dict[str, Any] = {
            "question_id": question["question_id"],
            "category": question["category"],
            "modes": {},
        }
        for condition, answer_fn in (("baseline", baseline_answer), ("candidate", preview_answer)):
            answers = {mode: answer_fn(question["text"], mode) for mode in MODES}
            reports[condition].append(
                {
                    "question_id": question["question_id"],
                    "question": question["text"],
                    "category": question["category"],
                    "answers": answers,
                    "citation_validity": {
                        mode: citation_validity(answer) for mode, answer in answers.items()
                    },
                    "judgments": {
                        mode: judged(question["text"], mode, answer)
                        for mode, answer in answers.items()
                    },
                }
            )
        baseline_row, candidate_row = reports["baseline"][-1], reports["candidate"][-1]
        for mode in MODES:
            cites_staged = any(
                citation.get("document_id") == staged_document_id
                for citation in candidate_row["answers"][mode].get("citations") or []
            )
            staged_citations += int(cites_staged)
            summary["modes"][mode] = {
                "baseline_score": baseline_row["judgments"][mode]["total_score"],
                "candidate_score": candidate_row["judgments"][mode]["total_score"],
                "baseline_unsupported_claims": baseline_row["judgments"][mode]["unsupported_claim_count"],
                "candidate_unsupported_claims": candidate_row["judgments"][mode]["unsupported_claim_count"],
                "candidate_citation_validity": candidate_row["citation_validity"][mode],
                "candidate_cites_staged_document": cites_staged,
            }
        summary_rows.append(summary)
    comparison = compare_evaluations(
        {"questions": reports["baseline"]}, {"questions": reports["candidate"]}
    )
    if not comparison["gates"]["same_questions"]:
        raise RuntimeError("Baseline and preview evaluations did not answer the same questions.")
    return {
        "question_set_sha256": _digest(questions),
        "metrics": {
            # The :evaluate schema bounds the ratio at 2; the raw ratio stays in diagnostics.
            "baseline_quality_ratio": min(2.0, comparison["quality_ratio"]),
            "effective_citation_ratio": comparison["candidate_citation_validity"],
            "unsupported_claim_delta": float(
                comparison["candidate_unsupported_claims"] - comparison["baseline_unsupported_claims"]
            ),
        },
        "diagnostics": {
            "baseline_mean_score": comparison["baseline_mean_score"],
            "candidate_mean_score": comparison["candidate_mean_score"],
            "quality_ratio": comparison["quality_ratio"],
            "baseline_unsupported_claims": comparison["baseline_unsupported_claims"],
            "candidate_unsupported_claims": comparison["candidate_unsupported_claims"],
            "staged_document_citations": staged_citations,
            "judged_answers": len(judgments),
            "reused_judgments": reused,
        },
        "questions": summary_rows,
        "report": reports,
    }
