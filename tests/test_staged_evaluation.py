from __future__ import annotations

import pytest

from notebooklm_graph_pipe.service.staged_evaluation import measure_staged_revision

QUESTIONS = [
    {"question_id": "F01", "text": "Unchanged question", "category": "factual"},
    {"question_id": "S01", "text": "Staged question", "category": "source_canary"},
]


def _answer(document_id: str) -> dict:
    return {
        "answer": f"Supported by {document_id} [S1]",
        "citations": [{"id": "S1", "document_id": document_id, "source_uri": f"uri:{document_id}"}],
        "retrieval": {"graph_candidates": 0},
    }


def test_measurement_compares_preview_with_active_corpus_and_reuses_identical_judgments() -> None:
    judged: list[tuple[str, str, str]] = []

    def judge(question, mode, answer):
        judged.append((question, mode, answer["answer"]))
        staged = "staged" in answer["answer"]
        return {"total_score": 16 if staged else 12, "unsupported_claim_count": 0}

    def baseline(question, mode):
        return _answer("active")

    def preview(question, mode):
        return _answer("staged" if question == "Staged question" else "active")

    result = measure_staged_revision(
        QUESTIONS,
        staged_document_id="staged",
        baseline_answer=baseline,
        preview_answer=preview,
        judge=judge,
    )

    # Unchanged answers are judged once per question and mode, never twice.
    assert len(judged) == 6
    assert result["diagnostics"]["reused_judgments"] == 2
    assert result["diagnostics"]["staged_document_citations"] == 2
    assert result["metrics"] == {
        "baseline_quality_ratio": pytest.approx(56 / 48),
        "effective_citation_ratio": 1.0,
        "unsupported_claim_delta": 0.0,
    }
    staged_row = result["questions"][1]["modes"]["graph_hybrid"]
    assert staged_row["candidate_cites_staged_document"] is True
    assert staged_row["baseline_score"] == 12 and staged_row["candidate_score"] == 16
    assert len(result["question_set_sha256"]) == 64


def test_measurement_reports_unsupported_claim_increase_and_invalid_citations() -> None:
    def judge(question, mode, answer):
        return {"total_score": 12, "unsupported_claim_count": 2 if "bad" in answer["answer"] else 0}

    def preview(question, mode):
        return {"answer": "bad claim", "citations": [{"id": "S1"}], "retrieval": {}}

    result = measure_staged_revision(
        QUESTIONS[:1],
        staged_document_id="staged",
        baseline_answer=lambda question, mode: _answer("active"),
        preview_answer=preview,
        judge=judge,
    )

    assert result["metrics"]["unsupported_claim_delta"] == 4.0
    assert result["metrics"]["effective_citation_ratio"] == 0.0


@pytest.mark.parametrize(
    "questions",
    [[], [{"question_id": "Q", "text": ""}], [{"question_id": "Q", "text": "a"}] * 2],
)
def test_measurement_rejects_unbounded_or_ambiguous_question_sets(questions) -> None:
    with pytest.raises(ValueError):
        measure_staged_revision(
            questions,
            staged_document_id="staged",
            baseline_answer=lambda question, mode: {},
            preview_answer=lambda question, mode: {},
            judge=lambda question, mode, answer: {},
        )
