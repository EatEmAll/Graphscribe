from __future__ import annotations

from pathlib import Path

import pytest

import scripts.run_ab_evaluation as ab


def test_parse_jsonl_events_collects_mcp_tool_calls() -> None:
    raw_output = "\n".join(
        [
            "warning: ignored",
            '{"type":"item.started","item":{"type":"mcp_tool_call","server":"notebooklm-mcp","tool":"notebook_query"}}',
            '{"type":"item.completed","item":{"type":"mcp_tool_call","server":"neo4j","tool":"read_neo4j_cypher"}}',
            "{not-json}",
        ]
    )

    payloads, tools_used = ab.parse_jsonl_events(raw_output)

    assert len(payloads) == 2
    assert tools_used == ["notebooklm-mcp:notebook_query", "neo4j:read_neo4j_cypher"]


def test_validate_tools_enforces_condition_contract() -> None:
    assert ab.validate_tools(ab.NOTEBOOK_ONLY, ["notebooklm-mcp:notebook_query"])
    assert not ab.validate_tools(ab.NOTEBOOK_ONLY, ["notebooklm-mcp:notebook_query", "neo4j:read_neo4j_cypher"])
    assert ab.validate_tools(ab.HYBRID, ["notebooklm-mcp:notebook_query", "neo4j:read_neo4j_cypher"])
    assert not ab.validate_tools(ab.HYBRID, ["notebooklm-mcp:notebook_query"])


def test_replacement_candidates_prefers_weak_ties_and_notebook_only_wins() -> None:
    primary_questions = [
        ab.QuestionSpec("Q1", "q1"),
        ab.QuestionSpec("Q2", "q2"),
        ab.QuestionSpec("Q3", "q3"),
    ]
    notebook_scores = {
        "Q1": ab.AnswerScore("d", ab.NOTEBOOK_ONLY, "Q1", 3, 3, 3, 2, 11, 5.5, "r", [], Path("q1n.json")),
        "Q2": ab.AnswerScore("d", ab.NOTEBOOK_ONLY, "Q2", 4, 4, 4, 1, 13, 6.5, "r", [], Path("q2n.json")),
        "Q3": ab.AnswerScore("d", ab.NOTEBOOK_ONLY, "Q3", 4, 4, 4, 4, 16, 8.0, "r", [], Path("q3n.json")),
    }
    hybrid_scores = {
        "Q1": ab.AnswerScore("d", ab.HYBRID, "Q1", 3, 3, 3, 2, 11, 5.5, "r", [], Path("q1h.json")),
        "Q2": ab.AnswerScore("d", ab.HYBRID, "Q2", 3, 3, 3, 1, 10, 5.0, "r", [], Path("q2h.json")),
        "Q3": ab.AnswerScore("d", ab.HYBRID, "Q3", 4, 4, 4, 2, 14, 7.0, "r", [], Path("q3h.json")),
    }
    comparisons = {
        "Q1": ab.ComparisonScore("d", "Q1", "tie", "no", "flat", Path("q1c.json")),
        "Q2": ab.ComparisonScore("d", "Q2", ab.NOTEBOOK_ONLY, "no", "flat", Path("q2c.json")),
        "Q3": ab.ComparisonScore("d", "Q3", "tie", "no", "flat", Path("q3c.json")),
    }

    candidates = ab.replacement_candidates(primary_questions, notebook_scores, hybrid_scores, comparisons)

    assert candidates == ["Q2", "Q1"]


def test_aggregate_overall_reports_hybrid_win() -> None:
    aggregate = ab.aggregate_overall(
        [
            {
                "rows": [
                    {"notebook_only_total": 10, "hybrid_total": 14},
                    {"notebook_only_total": 11, "hybrid_total": 15},
                ],
                "wins": 2,
                "losses": 0,
                "ties": 0,
            }
        ]
    )

    assert aggregate["notebook_only_rating_10"] == 5.25
    assert aggregate["hybrid_rating_10"] == 7.25
    assert aggregate["wins"] == 2
    assert "materially better overall" in aggregate["verdict"]


def test_run_structured_judge_rejects_tool_usage(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fake_run_codex_exec(**_: object) -> tuple[str, list[str], str]:
        return '{"dataset":"d","condition":"c","question_id":"q","correctness":4,"completeness":4,"evidence_quality":4,"cross_document_synthesis":4,"total_score":16,"normalized_score":8.0,"rationale":"ok","weakness_tags":[]}', ["notebooklm-mcp:notebook_query"], "cmd"

    monkeypatch.setattr(ab, "run_codex_exec", fake_run_codex_exec)

    with pytest.raises(ab.BenchmarkError, match="unexpectedly used tools"):
        ab.run_structured_judge(
            prompt_text="judge",
            temp_root=tmp_path / "temp",
            output_path=tmp_path / "judge.json",
            model="gpt-5.4",
        )


def test_load_existing_answer_artifact_reuses_valid_trace(tmp_path: Path) -> None:
    dataset = ab.dataset_specs()["bench-openalex-rag"]
    question = ab.QuestionSpec("OA01", "Question text")
    prompt_path, answer_path, events_path = ab.answer_artifact_paths(
        tmp_path, dataset.key, ab.NOTEBOOK_ONLY, question.question_id, 1
    )
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text("prompt", encoding="utf-8")
    answer_path.write_text("answer", encoding="utf-8")
    events_path.write_text(
        '{"type":"item.completed","item":{"type":"mcp_tool_call","server":"notebooklm-mcp","tool":"notebook_query"}}\n',
        encoding="utf-8",
    )

    artifact = ab.load_existing_answer_artifact(tmp_path, dataset, question, ab.NOTEBOOK_ONLY)

    assert artifact is not None
    assert artifact.answer_text == "answer"
    assert artifact.tools_used == ["notebooklm-mcp:notebook_query"]
    assert artifact.command_redacted == "<reused>"


def test_load_existing_answer_artifact_ignores_empty_answer(tmp_path: Path) -> None:
    dataset = ab.dataset_specs()["bench-openalex-rag"]
    question = ab.QuestionSpec("OA01", "Question text")
    prompt_path, answer_path, events_path = ab.answer_artifact_paths(
        tmp_path, dataset.key, ab.NOTEBOOK_ONLY, question.question_id, 1
    )
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text("prompt", encoding="utf-8")
    answer_path.write_text("", encoding="utf-8")
    events_path.write_text(
        '{"type":"item.completed","item":{"type":"mcp_tool_call","server":"notebooklm-mcp","tool":"notebook_query"}}\n',
        encoding="utf-8",
    )

    artifact = ab.load_existing_answer_artifact(tmp_path, dataset, question, ab.NOTEBOOK_ONLY)

    assert artifact is None


def test_load_existing_answer_score_round_trips(tmp_path: Path) -> None:
    score_path = tmp_path / "score.json"
    score_path.write_text(
        '{"dataset":"d","condition":"notebook_only","question_id":"Q1","correctness":4,"completeness":3,"evidence_quality":5,"cross_document_synthesis":2,"total_score":14,"normalized_score":7.0,"rationale":"ok","weakness_tags":["thin_evidence"]}',
        encoding="utf-8",
    )

    score = ab.load_existing_answer_score(score_path)

    assert score is not None
    assert score.total_score == 14
    assert score.weakness_tags == ["thin_evidence"]
    assert score.normalized_score == 7.0


def test_load_existing_answer_score_salvages_judge_text(tmp_path: Path) -> None:
    score_path = tmp_path / "score.json"
    judge_path = tmp_path / "score.judge.txt"
    judge_path.write_text(
        '{"dataset":"d","condition":"hybrid","question_id":"Q2","correctness":3,"completeness":4,"evidence_quality":3,"cross_document_synthesis":4,"total_score":14,"normalized_score":0.7,"rationale":"ok","weakness_tags":[]}',
        encoding="utf-8",
    )

    score = ab.load_existing_answer_score(score_path)

    assert score is not None
    assert score.total_score == 14
    assert score.normalized_score == 7.0
    assert score_path.exists()


def test_run_answer_generation_retries_when_codex_exits_without_final_message(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dataset = ab.dataset_specs()["bench-openalex-rag"]
    question = ab.QuestionSpec("OA01", "Question text")
    entry = ab.load_dataset_entry("bench-openalex-rag")
    calls: list[int] = []

    def fake_run_codex_exec(**kwargs: object) -> tuple[str, list[str], str]:
        calls.append(1)
        answer_path = Path(kwargs["answer_path"])
        events_path = Path(kwargs["events_path"])
        if len(calls) == 1:
            answer_path.parent.mkdir(parents=True, exist_ok=True)
            answer_path.write_text("", encoding="utf-8")
            events_path.write_text("", encoding="utf-8")
            raise ab.BenchmarkError("no last agent message")
        answer_path.parent.mkdir(parents=True, exist_ok=True)
        answer_path.write_text("answer", encoding="utf-8")
        events_path.write_text(
            '{"type":"item.completed","item":{"type":"mcp_tool_call","server":"notebooklm-mcp","tool":"notebook_query"}}\n',
            encoding="utf-8",
        )
        return "answer", ["notebooklm-mcp:notebook_query"], "cmd"

    monkeypatch.setattr(ab, "run_codex_exec", fake_run_codex_exec)

    artifact = ab.run_answer_generation(
        run_dir=tmp_path / "run",
        temp_root=tmp_path / "temp",
        dataset=dataset,
        entry=entry,
        question=question,
        condition=ab.NOTEBOOK_ONLY,
        model="gpt-5.4",
    )

    assert artifact.answer_text == "answer"
    assert artifact.attempt_count == 2
