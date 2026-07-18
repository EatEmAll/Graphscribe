from __future__ import annotations

from scripts.validate_compact_evaluation import compare_evaluations


def _report(score: int, *, graph_candidates: int, unsupported: int = 0) -> dict:
    rows = []
    for question_id, category in (("F1", "factual"), ("G1", "entity_bridge")):
        answers = {
            mode: {
                "answer": "supported [S1]",
                "citations": [{"id": "S1"}],
                "retrieval": {"graph_candidates": graph_candidates if mode == "graph_hybrid" else 0},
            }
            for mode in ("hybrid", "graph_hybrid")
        }
        rows.append(
            {
                "question_id": question_id,
                "question": f"Question {question_id}",
                "category": category,
                "reserve": False,
                "answers": answers,
                "citation_validity": {"hybrid": 1.0, "graph_hybrid": 1.0},
                "judgments": {
                    mode: {"total_score": score, "unsupported_claim_count": unsupported}
                    for mode in answers
                },
            }
        )
    return {"questions": rows}


def test_compact_evaluation_passes_all_gates() -> None:
    result = compare_evaluations(_report(20, graph_candidates=1), _report(19, graph_candidates=1))

    assert result["quality_ratio"] == 0.95
    assert result["passed"] is True


def test_compact_evaluation_fails_quality_and_graph_gates() -> None:
    result = compare_evaluations(_report(20, graph_candidates=1), _report(18, graph_candidates=0))

    assert result["gates"]["quality"] is False
    assert result["gates"]["graph_expansion"] is False
    assert result["passed"] is False
