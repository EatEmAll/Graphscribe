from __future__ import annotations

import json

from notebooklm_graph_pipe.runtime.llm_routing import PromptRoleConfig
from scripts.run_corpus_evaluation import (
    EvaluationModel,
    EvaluationQuestion,
    citation_validity,
    evaluate,
    load_completed_rows,
    load_questions,
    write_report,
)


class Service:
    def answer(self, corpus_key, payload):
        return {
            "answer": f"{payload['mode']} answer [S1]",
            "citations": [{"id": "S1", "document_id": "doc", "source_uri": "source.md"}],
        }


def test_load_questions_accepts_wrapped_payload(tmp_path) -> None:
    path = tmp_path / "questions.json"
    path.write_text(json.dumps({"questions": [{"question_id": "Q1", "text": "Question?"}]}), encoding="utf-8")
    questions = load_questions(path)
    assert questions[0].id == "Q1"
    assert questions[0].text == "Question?"


def test_evaluation_compares_text_and_graph_hybrid() -> None:
    report = evaluate(Service(), "demo", [EvaluationQuestion("Q1", "Question?")])
    assert report["conditions"] == ["hybrid", "graph_hybrid"]
    assert report["summary"]["mean_citation_validity"] == {"hybrid": 1.0, "graph_hybrid": 1.0}


def test_evaluation_checkpoints_and_resumes_completed_questions(tmp_path) -> None:
    output = tmp_path / "evaluation.json"
    questions = [EvaluationQuestion("Q1", "First?"), EvaluationQuestion("Q2", "Second?")]
    first = evaluate(
        Service(),
        "demo",
        questions[:1],
        lambda *_: {"total_score": 10},
        checkpoint=lambda report: write_report(output, report),
    )
    completed = load_completed_rows(output, "demo", questions)
    report = evaluate(
        Service(),
        "demo",
        questions,
        lambda *_: {"total_score": 10},
        completed_rows=completed,
    )
    assert first["summary"]["question_count"] == 1
    assert [row["question_id"] for row in report["questions"]] == ["Q1", "Q2"]


def test_citation_validity_rejects_incomplete_targets() -> None:
    assert citation_validity({"citations": []}) == 0.0
    assert citation_validity({"citations": [{"id": "S1", "document_id": "doc"}]}) == 0.0


def test_evaluation_model_validates_judge_scores() -> None:
    def generator(client, **kwargs):
        return {
            "correctness": 4,
            "completeness": 3,
            "evidence_quality": 5,
            "cross_document_synthesis": 2,
            "unsupported_claim_count": 1,
            "rationale": "ok",
            "weakness_tags": ["thin"],
        }, ""

    model = EvaluationModel(
        PromptRoleConfig("genai", "questions"),
        PromptRoleConfig("genai", "judge"),
        {"genai": object()},
        generator,
    )
    result = model.judge(
        "Question?",
        "hybrid",
        {"answer": "Answer [S1]", "citations": [{"id": "S1", "document_id": "doc", "source_uri": "a.md"}]},
    )
    assert result["total_score"] == 14
    assert result["normalized_score"] == 7.0
    assert result["citation_validity"] == 1.0
