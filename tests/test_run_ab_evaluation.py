from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

import scripts.run_ab_evaluation as ab


def test_ab_evaluation_defaults_to_sol_low_effort() -> None:
    assert ab.MODEL_NAME == "gpt-5.6-sol"
    assert ab.MODEL_REASONING_EFFORT == "low"


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


def test_load_manifest_dataset_builds_generic_entry(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "project_slug": "demo-corpus",
                "notebook": {"id": "nb-1", "title": "Demo Corpus"},
                "neo4j": {
                    "uri": "bolt://127.0.0.1:7687",
                    "username": "neo4j",
                    "password": "secret",
                    "database": "neo4j",
                    "container_name": "neo4j-demo",
                },
            }
        ),
        encoding="utf-8",
    )

    dataset, entry = ab.load_manifest_dataset(manifest_path)

    assert dataset.key == "demo-corpus"
    assert dataset.title == "Demo Corpus"
    assert entry.notebook.id == "nb-1"
    assert entry.neo4j.uri == "bolt://127.0.0.1:7687"
    assert entry.neo4j.container_name == "neo4j-demo"


def test_build_hybrid_overrides_uses_manifest_runtime(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "project_slug": "demo-corpus",
                "notebook": {"id": "nb-1", "title": "Demo Corpus"},
                "neo4j": {"uri": "bolt://127.0.0.1:7687", "username": "neo4j", "password": "secret", "database": "neo4j"},
            }
        ),
        encoding="utf-8",
    )

    _, entry = ab.load_manifest_dataset(manifest_path)

    overrides = ab.build_hybrid_overrides(entry)

    assert overrides == [
        "mcp_servers.neo4j.env={}",
        'mcp_servers.neo4j.env_vars=["NEO4J_URI","NEO4J_USERNAME","NEO4J_PASSWORD","NEO4J_DATABASE"]',
    ]
    assert "secret" not in " ".join(overrides)


def test_run_codex_exec_passes_neo4j_secret_only_in_child_environment(monkeypatch, tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "project_slug": "hosted",
                "notebook": {"id": "nb-1", "title": "Hosted"},
                "neo4j": {
                "uri": "neo4j+s://hosted.databases.neo4j.io",
                "username": "neo4j",
                "password": "environment-only-secret",
                "database": "neo4j",
            },
            }
        ),
        encoding="utf-8",
    )
    _, entry = ab.load_manifest_dataset(manifest_path)
    captured: dict[str, object] = {}
    answer_path = tmp_path / "answer.md"

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["env"] = kwargs["env"]
        answer_path.write_text("answer", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(ab, "resolve_codex_launcher", lambda: ["codex"])
    monkeypatch.setattr(ab.subprocess, "run", fake_run)

    ab.run_codex_exec(
        prompt_text="prompt",
        temp_root=tmp_path / "temp",
        answer_path=answer_path,
        events_path=tmp_path / "events.jsonl",
        model="gpt-5.4",
        config_overrides=ab.build_hybrid_overrides(entry),
        dataset_entry=entry,
        timeout_seconds=30,
    )

    assert "environment-only-secret" not in " ".join(captured["command"])
    assert captured["command"][captured["command"].index("--model") + 1] == "gpt-5.4"
    assert 'model_reasoning_effort="low"' in captured["command"]
    assert captured["env"]["NEO4J_PASSWORD"] == "environment-only-secret"


def test_load_manifest_dataset_rejects_missing_neo4j(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps({"project_slug": "demo-corpus", "notebook": {"id": "nb-1", "title": "Demo Corpus"}}),
        encoding="utf-8",
    )

    with pytest.raises(ab.BenchmarkError, match="Neo4j connection fields"):
        ab.load_manifest_dataset(manifest_path)


def test_load_manifest_dataset_resolves_password_environment(monkeypatch, tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "version": 3,
                "project_slug": "hosted",
                "notebook": {"id": "nb-1", "title": "Hosted"},
                "neo4j": {
                    "uri": "neo4j+s://hosted.databases.neo4j.io",
                    "username": "neo4j",
                    "database": "neo4j",
                    "password_env": "HOSTED_NEO4J_PASSWORD",
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("HOSTED_NEO4J_PASSWORD", "secret-from-env")

    _, entry = ab.load_manifest_dataset(manifest_path)

    assert entry.neo4j.password == "secret-from-env"
    assert entry.neo4j.password_env == "HOSTED_NEO4J_PASSWORD"


def test_load_manifest_dataset_recovers_managed_container_password(monkeypatch, tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "version": 3,
                "project_slug": "local",
                "notebook": {"id": "nb-1", "title": "Local"},
                "neo4j": {
                    "uri": "bolt://127.0.0.1:17687",
                    "username": "neo4j",
                    "database": "neo4j",
                    "container_name": "managed-neo4j",
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("NEO4J_PASSWORD", raising=False)
    monkeypatch.setattr(
        ab.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=args[0],
            returncode=0,
            stdout=json.dumps([{"Config": {"Env": ["NEO4J_AUTH=neo4j/container-secret"]}}]),
            stderr="",
        ),
    )

    _, entry = ab.load_manifest_dataset(manifest_path)

    assert entry.neo4j.password == "container-secret"


def test_load_question_plan_file_normalizes_ids_and_defaults(tmp_path: Path) -> None:
    questions_path = tmp_path / "questions.json"
    questions_path.write_text(
        json.dumps(
            {
                "questions": [
                    {"question_id": "x1", "text": f"Question {index}", "reserve": index > 8}
                    for index in range(1, 11)
                ]
            }
        ),
        encoding="utf-8",
    )
    dataset = ab.DatasetSpec(key="demo", title="Demo Corpus", notebook_id="nb-1", background="", rationale="", questions=[])

    background, rationale, questions = ab.load_question_plan_file(questions_path, dataset)

    assert "Demo Corpus" in background
    assert "Demo Corpus" in rationale
    assert [question.question_id for question in questions] == [f"Q{index:02d}" for index in range(1, 11)]
    assert sum(int(question.reserve) for question in questions) == 2


def test_load_question_plan_file_rejects_partial_reserve_shape(tmp_path: Path) -> None:
    questions_path = tmp_path / "questions.json"
    questions_path.write_text(
        json.dumps(
            {
                "questions": [
                    {"question_id": f"x{index}", "text": f"Question {index}", **({"reserve": index > 8} if index != 5 else {})}
                    for index in range(1, 11)
                ]
            }
        ),
        encoding="utf-8",
    )
    dataset = ab.DatasetSpec(key="demo", title="Demo Corpus", notebook_id="nb-1", background="", rationale="", questions=[])

    with pytest.raises(ab.BenchmarkError, match="include 'reserve' for every question"):
        ab.load_question_plan_file(questions_path, dataset)


def test_generate_question_plan_uses_hybrid_tools_and_writes_normalized_json(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dataset = ab.DatasetSpec(key="demo", title="Demo Corpus", notebook_id="nb-1", background="", rationale="", questions=[])
    entry = ab.DatasetRegistryEntry(
        key="demo",
        notebook=ab.RegistryNotebook(id="nb-1", title="Demo Corpus"),
        neo4j=ab.RegistryNeo4j(uri="bolt://127.0.0.1:7687", username="neo4j", password="secret", database="neo4j"),
    )

    def fake_run_codex_exec(**_: object) -> tuple[str, list[str], str]:
        payload = {
            "background": "Synthetic background",
            "rationale": "Synthetic rationale",
            "questions": [{"question_id": f"x{index}", "text": f"Question {index}", "reserve": index > 8} for index in range(1, 11)],
        }
        return json.dumps(payload), ["notebooklm-mcp:notebook_query", "neo4j:read_neo4j_cypher"], "cmd"

    monkeypatch.setattr(ab, "run_codex_exec", fake_run_codex_exec)

    background, rationale, questions = ab.generate_question_plan(tmp_path / "run", tmp_path / "temp", dataset, entry, "gpt-5.4")

    assert background == "Synthetic background"
    assert rationale == "Synthetic rationale"
    assert questions[0].question_id == "Q01"
    assert (tmp_path / "run" / "demo" / "question_plan" / "generated.json").exists()


def test_main_generic_mode_does_not_consult_dataset_specs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "project_slug": "demo-corpus",
                "notebook": {"id": "nb-1", "title": "Demo Corpus"},
                "neo4j": {"uri": "bolt://127.0.0.1:7687", "username": "neo4j", "password": "secret", "database": "neo4j"},
            }
        ),
        encoding="utf-8",
    )

    questions = [ab.QuestionSpec(f"Q{index:02d}", f"Question {index}", reserve=index > 8) for index in range(1, 11)]

    monkeypatch.setattr(ab, "dataset_specs", lambda: (_ for _ in ()).throw(AssertionError("dataset_specs should not be used")))
    monkeypatch.setattr(ab, "preflight_notebooks", lambda entries: None)
    monkeypatch.setattr(ab, "generate_question_plan", lambda *args, **kwargs: ("Generic background", "Generic rationale", questions))
    monkeypatch.setattr(
        ab,
        "benchmark_dataset",
            lambda **kwargs: (
                {
                    "dataset": kwargs["dataset"].key,
                    "rows": [{"question_id": "Q01", "notebook_only_total": 10, "hybrid_total": 12, "winner": ab.HYBRID, "reason": "better"}],
                    "notebook_only_mean_total": 10.0,
                    "hybrid_mean_total": 12.0,
                    "notebook_only_rating_10": 5.0,
                "hybrid_rating_10": 6.0,
                "wins": 1,
                "losses": 0,
                "ties": 0,
                "hybrid_helped": ["Q01"],
                "hybrid_little_or_none": [],
            },
            ["Q01"],
            {ab.NOTEBOOK_ONLY: {"Q01": ab.AnswerArtifact("demo-corpus", ab.NOTEBOOK_ONLY, "Q01", "Question 1", tmp_path / "a.md", tmp_path / "a.jsonl", tmp_path / "a.txt", "answer", ["notebooklm-mcp:notebook_query"], "cmd", 1)}, ab.HYBRID: {"Q01": ab.AnswerArtifact("demo-corpus", ab.HYBRID, "Q01", "Question 1", tmp_path / "b.md", tmp_path / "b.jsonl", tmp_path / "b.txt", "answer", ["notebooklm-mcp:notebook_query", "neo4j:read_neo4j_cypher"], "cmd", 1)}},
            {"Q01": ab.AnswerScore("demo-corpus", ab.NOTEBOOK_ONLY, "Q01", 3, 3, 2, 2, 10, 5.0, "r", [], tmp_path / "n.json")},
            {"Q01": ab.AnswerScore("demo-corpus", ab.HYBRID, "Q01", 3, 3, 3, 3, 12, 6.0, "r", [], tmp_path / "h.json")},
            {"Q01": ab.ComparisonScore("demo-corpus", "Q01", ab.HYBRID, "yes", "r", tmp_path / "c.json")},
            [],
        ),
    )
    monkeypatch.setattr(ab, "write_json", lambda *args, **kwargs: None)
    monkeypatch.setattr(ab, "write_text", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        ab.sys,
        "argv",
        [
            "run_ab_evaluation.py",
            "--manifest-path",
            str(manifest_path),
            "--runs-root",
            str(tmp_path / "runs"),
            "--temp-root",
            str(tmp_path / "temp"),
        ],
    )

    assert ab.main() == 0


def test_main_generic_mode_uses_questions_file_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "project_slug": "demo-corpus",
                "notebook": {"id": "nb-1", "title": "Demo Corpus"},
                "neo4j": {"uri": "bolt://127.0.0.1:7687", "username": "neo4j", "password": "secret", "database": "neo4j"},
            }
        ),
        encoding="utf-8",
    )
    questions_path = tmp_path / "questions.json"
    questions_path.write_text(
        json.dumps(
            {
                "background": "Override background",
                "rationale": "Override rationale",
                "questions": [
                    {"question_id": f"x{index}", "text": f"Question {index}", "reserve": index > 8}
                    for index in range(1, 11)
                ],
            }
        ),
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

    monkeypatch.setattr(ab, "preflight_notebooks", lambda entries: None)
    monkeypatch.setattr(ab, "generate_question_plan", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("generate_question_plan should not run")))

    def fake_benchmark_dataset(**kwargs: object):
        dataset = kwargs["dataset"]
        captured["background"] = dataset.background
        captured["rationale"] = dataset.rationale
        captured["questions"] = dataset.questions
        return (
            {
                "dataset": dataset.key,
                "rows": [{"question_id": "Q01", "notebook_only_total": 10, "hybrid_total": 12, "winner": ab.HYBRID, "reason": "better"}],
                "notebook_only_mean_total": 10.0,
                "hybrid_mean_total": 12.0,
                "notebook_only_rating_10": 5.0,
                "hybrid_rating_10": 6.0,
                "wins": 1,
                "losses": 0,
                "ties": 0,
                "hybrid_helped": ["Q01"],
                "hybrid_little_or_none": [],
            },
            ["Q01"],
            {ab.NOTEBOOK_ONLY: {"Q01": ab.AnswerArtifact("demo-corpus", ab.NOTEBOOK_ONLY, "Q01", "Question 1", tmp_path / "a.md", tmp_path / "a.jsonl", tmp_path / "a.txt", "answer", ["notebooklm-mcp:notebook_query"], "cmd", 1)}, ab.HYBRID: {"Q01": ab.AnswerArtifact("demo-corpus", ab.HYBRID, "Q01", "Question 1", tmp_path / "b.md", tmp_path / "b.jsonl", tmp_path / "b.txt", "answer", ["notebooklm-mcp:notebook_query", "neo4j:read_neo4j_cypher"], "cmd", 1)}},
            {"Q01": ab.AnswerScore("demo-corpus", ab.NOTEBOOK_ONLY, "Q01", 3, 3, 2, 2, 10, 5.0, "r", [], tmp_path / "n.json")},
            {"Q01": ab.AnswerScore("demo-corpus", ab.HYBRID, "Q01", 3, 3, 3, 3, 12, 6.0, "r", [], tmp_path / "h.json")},
            {"Q01": ab.ComparisonScore("demo-corpus", "Q01", ab.HYBRID, "yes", "r", tmp_path / "c.json")},
            [],
        )

    monkeypatch.setattr(ab, "benchmark_dataset", fake_benchmark_dataset)
    monkeypatch.setattr(ab, "write_json", lambda *args, **kwargs: None)
    monkeypatch.setattr(ab, "write_text", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        ab.sys,
        "argv",
        [
            "run_ab_evaluation.py",
            "--manifest-path",
            str(manifest_path),
            "--questions-file",
            str(questions_path),
            "--runs-root",
            str(tmp_path / "runs"),
            "--temp-root",
            str(tmp_path / "temp"),
        ],
    )

    assert ab.main() == 0
    assert captured["background"] == "Override background"
    assert captured["rationale"] == "Override rationale"
    assert [question.question_id for question in captured["questions"]] == [f"Q{index:02d}" for index in range(1, 11)]


def test_main_benchmark_mode_without_docker_uses_direct_neo4j(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dataset = ab.dataset_specs()["bench-openalex-rag"]
    entry = ab.DatasetRegistryEntry(
        key=dataset.key,
        notebook=ab.RegistryNotebook(id=dataset.notebook_id, title=dataset.title),
        neo4j=ab.RegistryNeo4j(
            uri="bolt://127.0.0.1:7687",
            username="neo4j",
            password="secret",
            database="neo4j",
            container_name="bench-container",
        ),
    )
    captured: dict[str, object] = {}

    monkeypatch.setattr(ab, "docker_available", lambda: False)
    monkeypatch.setattr(ab, "load_dataset_entry", lambda dataset_key, registry_path=None: entry)
    monkeypatch.setattr(ab, "preflight_notebooks", lambda entries: None)
    monkeypatch.setattr(ab, "ensure_container_exists", lambda name: (_ for _ in ()).throw(AssertionError("container preflight should not run")))

    def fake_benchmark_dataset(**kwargs: object):
        captured["manage_container"] = kwargs["manage_container"]
        return (
            {
                "dataset": dataset.key,
                "rows": [{"question_id": "OA01", "notebook_only_total": 10, "hybrid_total": 12, "winner": ab.HYBRID, "reason": "better"}],
                "notebook_only_mean_total": 10.0,
                "hybrid_mean_total": 12.0,
                "notebook_only_rating_10": 5.0,
                "hybrid_rating_10": 6.0,
                "wins": 1,
                "losses": 0,
                "ties": 0,
                "hybrid_helped": ["OA01"],
                "hybrid_little_or_none": [],
            },
            ["OA01"],
            {ab.NOTEBOOK_ONLY: {"OA01": ab.AnswerArtifact(dataset.key, ab.NOTEBOOK_ONLY, "OA01", "Question 1", tmp_path / "a.md", tmp_path / "a.jsonl", tmp_path / "a.txt", "answer", ["notebooklm-mcp:notebook_query"], "cmd", 1)}, ab.HYBRID: {"OA01": ab.AnswerArtifact(dataset.key, ab.HYBRID, "OA01", "Question 1", tmp_path / "b.md", tmp_path / "b.jsonl", tmp_path / "b.txt", "answer", ["notebooklm-mcp:notebook_query", "neo4j:read_neo4j_cypher"], "cmd", 1)}},
            {"OA01": ab.AnswerScore(dataset.key, ab.NOTEBOOK_ONLY, "OA01", 3, 3, 2, 2, 10, 5.0, "r", [], tmp_path / "n.json")},
            {"OA01": ab.AnswerScore(dataset.key, ab.HYBRID, "OA01", 3, 3, 3, 3, 12, 6.0, "r", [], tmp_path / "h.json")},
            {"OA01": ab.ComparisonScore(dataset.key, "OA01", ab.HYBRID, "yes", "r", tmp_path / "c.json")},
            [],
        )

    monkeypatch.setattr(ab, "benchmark_dataset", fake_benchmark_dataset)
    monkeypatch.setattr(ab, "write_json", lambda *args, **kwargs: None)
    monkeypatch.setattr(ab, "write_text", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        ab.sys,
        "argv",
        [
            "run_ab_evaluation.py",
            "--datasets",
            dataset.key,
            "--runs-root",
            str(tmp_path / "runs"),
            "--temp-root",
            str(tmp_path / "temp"),
        ],
    )

    assert ab.main() == 0
    assert captured["manage_container"] is False


def test_main_benchmark_mode_with_docker_and_container_manages_container(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dataset = ab.dataset_specs()["bench-openalex-rag"]
    entry = ab.DatasetRegistryEntry(
        key=dataset.key,
        notebook=ab.RegistryNotebook(id=dataset.notebook_id, title=dataset.title),
        neo4j=ab.RegistryNeo4j(
            uri="bolt://127.0.0.1:7687",
            username="neo4j",
            password="secret",
            database="neo4j",
            container_name="bench-container",
        ),
    )
    captured: dict[str, object] = {}
    container_checks: list[str] = []

    monkeypatch.setattr(ab, "docker_available", lambda: True)
    monkeypatch.setattr(ab, "load_dataset_entry", lambda dataset_key, registry_path=None: entry)
    monkeypatch.setattr(ab, "preflight_notebooks", lambda entries: None)
    monkeypatch.setattr(ab, "ensure_container_exists", lambda name: container_checks.append(name))

    def fake_benchmark_dataset(**kwargs: object):
        captured["manage_container"] = kwargs["manage_container"]
        return (
            {
                "dataset": dataset.key,
                "rows": [{"question_id": "OA01", "notebook_only_total": 10, "hybrid_total": 12, "winner": ab.HYBRID, "reason": "better"}],
                "notebook_only_mean_total": 10.0,
                "hybrid_mean_total": 12.0,
                "notebook_only_rating_10": 5.0,
                "hybrid_rating_10": 6.0,
                "wins": 1,
                "losses": 0,
                "ties": 0,
                "hybrid_helped": ["OA01"],
                "hybrid_little_or_none": [],
            },
            ["OA01"],
            {ab.NOTEBOOK_ONLY: {"OA01": ab.AnswerArtifact(dataset.key, ab.NOTEBOOK_ONLY, "OA01", "Question 1", tmp_path / "a.md", tmp_path / "a.jsonl", tmp_path / "a.txt", "answer", ["notebooklm-mcp:notebook_query"], "cmd", 1)}, ab.HYBRID: {"OA01": ab.AnswerArtifact(dataset.key, ab.HYBRID, "OA01", "Question 1", tmp_path / "b.md", tmp_path / "b.jsonl", tmp_path / "b.txt", "answer", ["notebooklm-mcp:notebook_query", "neo4j:read_neo4j_cypher"], "cmd", 1)}},
            {"OA01": ab.AnswerScore(dataset.key, ab.NOTEBOOK_ONLY, "OA01", 3, 3, 2, 2, 10, 5.0, "r", [], tmp_path / "n.json")},
            {"OA01": ab.AnswerScore(dataset.key, ab.HYBRID, "OA01", 3, 3, 3, 3, 12, 6.0, "r", [], tmp_path / "h.json")},
            {"OA01": ab.ComparisonScore(dataset.key, "OA01", ab.HYBRID, "yes", "r", tmp_path / "c.json")},
            [],
        )

    monkeypatch.setattr(ab, "benchmark_dataset", fake_benchmark_dataset)
    monkeypatch.setattr(ab, "write_json", lambda *args, **kwargs: None)
    monkeypatch.setattr(ab, "write_text", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        ab.sys,
        "argv",
        [
            "run_ab_evaluation.py",
            "--datasets",
            dataset.key,
            "--runs-root",
            str(tmp_path / "runs"),
            "--temp-root",
            str(tmp_path / "temp"),
        ],
    )

    assert ab.main() == 0
    assert container_checks == ["bench-container"]
    assert captured["manage_container"] is True


def test_main_benchmark_mode_without_container_name_uses_direct_neo4j(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dataset = ab.dataset_specs()["bench-openalex-rag"]
    entry = ab.DatasetRegistryEntry(
        key=dataset.key,
        notebook=ab.RegistryNotebook(id=dataset.notebook_id, title=dataset.title),
        neo4j=ab.RegistryNeo4j(
            uri="bolt://127.0.0.1:7687",
            username="neo4j",
            password="secret",
            database="neo4j",
            container_name=None,
        ),
    )
    captured: dict[str, object] = {}

    monkeypatch.setattr(ab, "docker_available", lambda: True)
    monkeypatch.setattr(ab, "load_dataset_entry", lambda dataset_key, registry_path=None: entry)
    monkeypatch.setattr(ab, "preflight_notebooks", lambda entries: None)
    monkeypatch.setattr(ab, "ensure_container_exists", lambda name: (_ for _ in ()).throw(AssertionError("container preflight should not run")))

    def fake_benchmark_dataset(**kwargs: object):
        captured["manage_container"] = kwargs["manage_container"]
        return (
            {
                "dataset": dataset.key,
                "rows": [{"question_id": "OA01", "notebook_only_total": 10, "hybrid_total": 12, "winner": ab.HYBRID, "reason": "better"}],
                "notebook_only_mean_total": 10.0,
                "hybrid_mean_total": 12.0,
                "notebook_only_rating_10": 5.0,
                "hybrid_rating_10": 6.0,
                "wins": 1,
                "losses": 0,
                "ties": 0,
                "hybrid_helped": ["OA01"],
                "hybrid_little_or_none": [],
            },
            ["OA01"],
            {ab.NOTEBOOK_ONLY: {"OA01": ab.AnswerArtifact(dataset.key, ab.NOTEBOOK_ONLY, "OA01", "Question 1", tmp_path / "a.md", tmp_path / "a.jsonl", tmp_path / "a.txt", "answer", ["notebooklm-mcp:notebook_query"], "cmd", 1)}, ab.HYBRID: {"OA01": ab.AnswerArtifact(dataset.key, ab.HYBRID, "OA01", "Question 1", tmp_path / "b.md", tmp_path / "b.jsonl", tmp_path / "b.txt", "answer", ["notebooklm-mcp:notebook_query", "neo4j:read_neo4j_cypher"], "cmd", 1)}},
            {"OA01": ab.AnswerScore(dataset.key, ab.NOTEBOOK_ONLY, "OA01", 3, 3, 2, 2, 10, 5.0, "r", [], tmp_path / "n.json")},
            {"OA01": ab.AnswerScore(dataset.key, ab.HYBRID, "OA01", 3, 3, 3, 3, 12, 6.0, "r", [], tmp_path / "h.json")},
            {"OA01": ab.ComparisonScore(dataset.key, "OA01", ab.HYBRID, "yes", "r", tmp_path / "c.json")},
            [],
        )

    monkeypatch.setattr(ab, "benchmark_dataset", fake_benchmark_dataset)
    monkeypatch.setattr(ab, "write_json", lambda *args, **kwargs: None)
    monkeypatch.setattr(ab, "write_text", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        ab.sys,
        "argv",
        [
            "run_ab_evaluation.py",
            "--datasets",
            dataset.key,
            "--runs-root",
            str(tmp_path / "runs"),
            "--temp-root",
            str(tmp_path / "temp"),
        ],
    )

    assert ab.main() == 0
    assert captured["manage_container"] is False


def test_render_technical_blog_uses_generic_wording() -> None:
    dataset = ab.DatasetSpec(
        key="demo",
        title="Demo Corpus",
        notebook_id="nb-1",
        background="Background",
        rationale="Rationale",
        questions=[ab.QuestionSpec("Q01", "Question 1")],
    )

    rendered = ab.render_technical_blog(
        Path("run"),
        [
            {
                "dataset": "demo",
                "rows": [{"question_id": "Q01", "notebook_only_total": 10, "hybrid_total": 12, "winner": "hybrid", "reason": "better"}],
                "notebook_only_mean_total": 10.0,
                "hybrid_mean_total": 12.0,
                "notebook_only_rating_10": 5.0,
                "hybrid_rating_10": 6.0,
                "wins": 1,
                "losses": 0,
                "ties": 0,
                "hybrid_helped": ["Q01"],
                "hybrid_little_or_none": [],
            }
        ],
        {"notebook_only_rating_10": 5.0, "hybrid_rating_10": 6.0, "wins": 1, "losses": 0, "ties": 0, "verdict": "Across this corpus, hybrid was better."},
        [dataset],
        report_mode=ab.GENERIC_MODE,
    )

    assert "Across 3 Benchmark Corpora" not in rendered
    assert "## Why This Corpus" in rendered
    assert "Demo Corpus" in rendered
