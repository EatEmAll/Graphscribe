import json
import subprocess
from pathlib import Path

import pytest

import notebooklm_graph_pipe.consolidation.self_improving as csi


@pytest.fixture(autouse=True)
def _stub_live_labels(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        csi,
        "_fetch_live_entity_labels",
        lambda params: set(csi.DEFAULT_TIER2_LABELS) | {"Execution Concept", "Resource", "Group"},
    )
    monkeypatch.setattr(csi, "_compute_live_concept_only_without_taxonomy_ratio", lambda params: 0.0)


def _valid_review_payload() -> dict:
    return {
        "is_consolidated": False,
        "diagnosis": "mixed_debt",
        "kpis": {
            "entity_count": 1000,
            "concept_only_count": 200,
            "concept_only_ratio": 0.2,
            "duplicate_anchor_count": 40,
            "duplicate_candidate_rate": 0.04,
            "subclass_rel_count": 25,
        },
        "taxonomy_kpis": {
            "concept_only_without_taxonomy_count": 160,
            "concept_only_degree_le_2_count": 90,
            "concept_only_degree_le_3_count": 140,
            "concept_only_with_similarity_or_alias_count": 20,
        },
        "focus_examples": {
            "high_degree_concept_only": ["Leverage", "Volatility"],
            "low_degree_concept_only": ["Net Fee", "Epoch"],
        },
        "proposed_tier2": {
            "batch_size": 80,
            "sleep_seconds": 0.5,
            "max_nodes": 1200,
        },
        "proposed_tier3": {
            "threshold": 0.88,
            "max_candidates": 900,
            "max_merges": 120,
        },
        "proposed_tier2_catalog": {
            "labels": list(csi.DEFAULT_TIER2_LABELS),
            "add": [],
            "remove": [],
            "rename_map": {},
            "guidance": {"Trading Concept": "Core trading ideas like leverage or liquidity."},
            "rationale": "Keep the current catalog.",
        },
        "rationale": "Needs more relabeling and dedup passes.",
        "confidence": 0.81,
    }


def test_combined_stop_gate_requires_semantic_pass_and_consecutive_passes() -> None:
    review = _valid_review_payload()
    review["kpis"]["concept_only_ratio"] = 0.03
    review["kpis"]["duplicate_candidate_rate"] = 0.0
    review["taxonomy_kpis"]["concept_only_without_taxonomy_count"] = 140
    review["taxonomy_kpis"]["concept_only_with_similarity_or_alias_count"] = 0

    assert csi.passes_consolidation_gate(
        review["kpis"],
        target_concept_ratio=0.05,
        target_duplicate_rate=0.015,
    )
    assert csi.passes_semantic_gate(
        {**review, "kpis": {**review["kpis"], "concept_only_count": 0}},
        target_concept_ratio=0.05,
        target_duplicate_rate=0.015,
        target_concept_without_taxonomy_ratio=0.60,
    )
    assert csi.passes_semantic_gate(
        {**review, "kpis": {**review["kpis"], "concept_only_count": 200}, "taxonomy_kpis": {**review["taxonomy_kpis"], "concept_only_without_taxonomy_count": 120}},
        target_concept_ratio=0.05,
        target_duplicate_rate=0.015,
        target_concept_without_taxonomy_ratio=0.60,
    )
    assert not csi.passes_stop_gate(
        review,
        target_concept_ratio=0.05,
        target_duplicate_rate=0.015,
        target_concept_without_taxonomy_ratio=0.60,
    )

    review["taxonomy_kpis"]["concept_only_without_taxonomy_count"] = 120
    assert csi.passes_stop_gate(
        review,
        target_concept_ratio=0.05,
        target_duplicate_rate=0.015,
        target_concept_without_taxonomy_ratio=0.60,
    )

    consecutive = 0
    consecutive = csi.update_consecutive_passes(consecutive, gate_passed=True)
    consecutive = csi.update_consecutive_passes(consecutive, gate_passed=True)
    assert consecutive == 2
    assert csi.update_consecutive_passes(consecutive, gate_passed=False) == 0


def test_apply_guardrails_clamps_and_blocks_threshold_drop() -> None:
    review = _valid_review_payload()
    review["proposed_tier2"] = {"batch_size": 10000, "sleep_seconds": -3, "max_nodes": 1}
    review["proposed_tier3"] = {"threshold": 0.82, "max_candidates": 5, "max_merges": 5000}

    current_t2 = csi.Tier2Params(batch_size=50, sleep_seconds=1.0, max_nodes=1000)
    current_t3 = csi.Tier3Params(threshold=0.90, max_candidates=600, max_merges=200)

    next_t2, next_t3 = csi.apply_guardrails(
        review=review,
        current_tier2=current_t2,
        current_tier3=current_t3,
        target_concept_ratio=0.15,
        target_duplicate_rate=0.015,
        target_concept_without_taxonomy_ratio=0.60,
        last_tier3_summary={"judged_pairs": 10, "alias_acceptance_rate": 0.05},
    )

    assert next_t2.batch_size == 200
    assert next_t2.sleep_seconds == 0.0
    assert next_t2.max_nodes == 50
    assert next_t3.max_candidates == 100
    assert next_t3.max_merges == 400
    assert next_t3.threshold == current_t3.threshold


def test_apply_guardrails_freezes_tier3_for_taxonomy_debt() -> None:
    review = _valid_review_payload()
    review["diagnosis"] = "taxonomy_debt"
    review["kpis"]["duplicate_candidate_rate"] = 0.0
    review["kpis"]["duplicate_anchor_count"] = 0
    review["taxonomy_kpis"]["concept_only_with_similarity_or_alias_count"] = 0

    current_t2 = csi.Tier2Params(batch_size=50, sleep_seconds=1.0, max_nodes=1000)
    current_t3 = csi.Tier3Params(threshold=0.90, max_candidates=600, max_merges=200)
    next_t2, next_t3 = csi.apply_guardrails(
        review=review,
        current_tier2=current_t2,
        current_tier3=current_t3,
        target_concept_ratio=0.15,
        target_duplicate_rate=0.015,
        target_concept_without_taxonomy_ratio=0.60,
        last_tier3_summary=None,
    )

    assert next_t2.batch_size == 80
    assert next_t3 == current_t3


def test_classify_failure_mode_overrides_ungrounded_taxonomy_diagnosis() -> None:
    review = _valid_review_payload()
    review["diagnosis"] = "taxonomy_debt"
    review["kpis"]["duplicate_candidate_rate"] = 0.0
    review["taxonomy_kpis"]["concept_only_with_similarity_or_alias_count"] = 60

    assert csi.classify_failure_mode(
        review=review,
        target_concept_ratio=0.15,
        target_duplicate_rate=0.015,
        target_concept_without_taxonomy_ratio=0.60,
    ) == "mixed_debt"


def test_classify_failure_mode_flags_taxonomy_debt_when_without_taxonomy_ratio_is_high() -> None:
    review = _valid_review_payload()
    review["diagnosis"] = "balanced"
    review["kpis"]["concept_only_ratio"] = 0.03
    review["kpis"]["duplicate_candidate_rate"] = 0.0
    review["taxonomy_kpis"]["concept_only_without_taxonomy_count"] = 150
    review["taxonomy_kpis"]["concept_only_with_similarity_or_alias_count"] = 0

    assert csi.classify_failure_mode(
        review=review,
        target_concept_ratio=0.05,
        target_duplicate_rate=0.015,
        target_concept_without_taxonomy_ratio=0.60,
    ) == "taxonomy_debt"


def test_should_run_tier3_uses_local_failure_mode() -> None:
    review = _valid_review_payload()
    review["diagnosis"] = "taxonomy_debt"
    review["kpis"]["duplicate_candidate_rate"] = 0.0
    review["taxonomy_kpis"]["concept_only_with_similarity_or_alias_count"] = 60

    assert csi.should_run_tier3(
        review,
        {"judged_pairs": 10, "alias_acceptance_rate": 0.01},
        target_concept_ratio=0.15,
        target_duplicate_rate=0.015,
        target_concept_without_taxonomy_ratio=0.60,
    ) is True


def test_should_run_tier2_uses_concept_ratio_only() -> None:
    review = _valid_review_payload()
    review["kpis"]["concept_only_ratio"] = 0.051

    assert csi.should_run_tier2(review, target_concept_ratio=0.05) is True

    review["kpis"]["concept_only_ratio"] = 0.05
    assert csi.should_run_tier2(review, target_concept_ratio=0.05) is False


def test_should_run_taxonomy_uses_taxonomy_kpis() -> None:
    review = _valid_review_payload()
    review["kpis"]["concept_only_ratio"] = 0.03
    review["kpis"]["concept_only_count"] = 100
    review["taxonomy_kpis"]["concept_only_without_taxonomy_count"] = 70

    assert csi.should_run_taxonomy(
        review,
        target_concept_ratio=0.05,
        target_concept_without_taxonomy_ratio=0.60,
    ) is True

    review["taxonomy_kpis"]["concept_only_without_taxonomy_count"] = 60
    assert csi.should_run_taxonomy(
        review,
        target_concept_ratio=0.05,
        target_concept_without_taxonomy_ratio=0.60,
    ) is False


def test_parse_and_validate_codex_payload_round_trip() -> None:
    payload = _valid_review_payload()
    validated = csi.validate_review_payload(csi.parse_codex_review_output(json.dumps([payload])))
    assert validated["diagnosis"] == "mixed_debt"
    assert validated["taxonomy_kpis"]["concept_only_without_taxonomy_count"] == 160
    assert validated["focus_examples"]["high_degree_concept_only"][0] == "Leverage"
    assert validated["proposed_tier3"]["threshold"] == pytest.approx(0.88)


def test_parse_codex_review_output_strips_whitespace_from_keys() -> None:
    payload = csi.parse_codex_review_output('{" diagnosis":"balanced"," kpis":{" entity_count":1}}')
    assert payload["diagnosis"] == "balanced"
    assert payload["kpis"]["entity_count"] == 1


def test_run_codex_review_retries_invalid_payload(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    outputs = iter(
        [
            '{"is_consolidated":true,"diagnosis":"balanced","kpis":{"entity_count":1,"concept_only_count":0,"concept_only_ratio":0.bad}}',
            json.dumps([_valid_review_payload()]),
        ]
    )

    def fake_run_with_input(command, log_file, stdin_text):
        raw_path = Path(command[-2])
        raw_path.write_text(next(outputs), encoding="utf-8")
        existing = log_file.read_text(encoding="utf-8") if log_file.exists() else ""
        log_file.write_text(existing + "ok", encoding="utf-8")

    monkeypatch.setattr(csi, "_run_command_with_input", fake_run_with_input)
    monkeypatch.setattr(csi, "CODEX_REVIEW_MAX_ATTEMPTS", 2)

    review = csi.run_codex_review(
        codex_bin="codex",
        target_concept_ratio=0.15,
        target_duplicate_rate=0.015,
        target_concept_without_taxonomy_ratio=0.60,
        current_tier2=csi.Tier2Params(),
        current_tier3=csi.Tier3Params(),
        current_catalog=csi._default_tier2_catalog(),
        iteration_dir=tmp_path,
    )

    assert review["diagnosis"] == "mixed_debt"
    assert "REVIEW_PARSE_ATTEMPT_1_ERROR" in (tmp_path / "codex_review_exec.log").read_text(encoding="utf-8")


def test_validate_codex_taxonomy_tail_payload_round_trip() -> None:
    payload = {
        "summary": {
            "queue_count": 2,
            "processed_count": 2,
            "recommended_relabels": 1,
            "recommended_relations": 1,
            "deprioritized": 0,
            "blocked": 0,
            "rationale": "Prioritize the two strongest residual fixes.",
            "confidence": 0.78,
        },
        "decisions": [
            {
                "eid": "eid-1",
                "name": "Talent",
                "action": "relabel",
                "label": "Condition",
                "relation": "NONE",
                "target_eid": None,
                "reason": "Talent behaves like a quality.",
                "priority": 1,
                "confidence": 0.8,
            },
            {
                "eid": "eid-2",
                "name": "problem",
                "action": "add_relation",
                "label": None,
                "relation": "SUBCLASS_OF",
                "target_eid": "eid-parent",
                "reason": "General Problems is the broader parent.",
                "priority": 2,
                "confidence": 0.74,
            },
        ],
    }

    validated = csi.validate_codex_taxonomy_tail_payload(csi.parse_codex_taxonomy_tail_output(json.dumps([payload])))
    assert validated["summary"]["queue_count"] == 2
    assert validated["decisions"][0]["action"] == "relabel"
    assert validated["decisions"][1]["relation"] == "SUBCLASS_OF"


def test_run_state_omits_neo4j_password_and_restores_it_from_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("NEO4J_PASSWORD", "hosted-secret")
    state = csi._initialize_run_state(csi.OrchestratorConfig(), tmp_path)

    assert "hosted-secret" not in json.dumps(state)
    assert all("neo4j_password" not in values for values in state["current_params"].values())

    tier2, taxonomy, tier3, *_ = csi._restore_runtime_state(state)
    assert {tier2.neo4j_password, taxonomy.neo4j_password, tier3.neo4j_password} == {"hosted-secret"}


def test_prepare_state_scrubs_legacy_neo4j_password(tmp_path: Path) -> None:
    state = csi._initialize_run_state(csi.OrchestratorConfig(), tmp_path)
    state["current_params"]["tier2"]["neo4j_password"] = "legacy-secret"
    state_path = tmp_path / "run_state.json"

    csi._prepare_state(state, state_path)

    assert "legacy-secret" not in state_path.read_text(encoding="utf-8")

def test_run_codex_taxonomy_tail_retries_invalid_payload(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    queue_path = tmp_path / "taxonomy_codex_queue.jsonl"
    queue_path.write_text(
        json.dumps({"eid": "eid-1", "name": "Talent", "candidate_targets": []}) + "\n",
        encoding="utf-8",
    )
    outputs = iter(
        [
            '{"summary":{"queue_count":1,"processed_count":"bad"}}',
            json.dumps(
                {
                    "summary": {
                        "queue_count": 1,
                        "processed_count": 1,
                        "recommended_relabels": 1,
                        "recommended_relations": 0,
                        "deprioritized": 0,
                        "blocked": 0,
                        "rationale": "One safe relabel.",
                        "confidence": 0.82,
                    },
                    "decisions": [
                        {
                            "eid": "eid-1",
                            "name": "Talent",
                            "action": "relabel",
                            "label": "Condition",
                            "relation": "NONE",
                            "target_eid": None,
                            "reason": "Talent behaves like a quality.",
                            "priority": 1,
                            "confidence": 0.8,
                        }
                    ],
                }
            ),
        ]
    )

    def fake_run_with_input(command, log_file, stdin_text):
        raw_path = Path(command[-2])
        raw_path.write_text(next(outputs), encoding="utf-8")
        existing = log_file.read_text(encoding="utf-8") if log_file.exists() else ""
        log_file.write_text(existing + "ok", encoding="utf-8")

    monkeypatch.setattr(csi, "_run_command_with_input", fake_run_with_input)
    monkeypatch.setattr(csi, "CODEX_REVIEW_MAX_ATTEMPTS", 2)

    payload = csi.run_codex_taxonomy_tail(
        codex_bin="codex",
        current_catalog=csi._default_tier2_catalog(),
        target_concept_without_taxonomy_ratio=0.60,
        iteration_dir=tmp_path,
        prior_review_path=None,
    )

    assert payload["summary"]["processed_count"] == 1
    assert "TAIL_PARSE_ATTEMPT_1_ERROR" in (tmp_path / "codex_taxonomy_tail_exec.log").read_text(encoding="utf-8")


def test_build_agent_command_for_codex_uses_exec_and_output_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(csi, "_resolve_cli_executable", lambda executable: executable)
    command, stdin_text = csi._build_agent_command(
        role_config=csi.AgentRoleConfig(client="codex", model="gpt-5.4", executable="codex"),
        raw_output_file=tmp_path / "raw.txt",
        schema_file=None,
        prompt="review prompt",
    )

    assert command == [
        "codex",
        "exec",
        "--ephemeral",
        "-m",
        "gpt-5.4",
        "-o",
        str(tmp_path / "raw.txt"),
        "-",
    ]
    assert stdin_text == "review prompt"


def test_build_agent_command_for_claude_uses_json_schema_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(csi, "_resolve_cli_executable", lambda executable: executable)
    command, stdin_text = csi._build_agent_command(
        role_config=csi.AgentRoleConfig(
            client="claude",
            model="claude-sonnet-4",
            executable="claude",
            args=("--mcp-config", "C:\\temp\\claude-mcp.json"),
        ),
        raw_output_file=tmp_path / "raw.txt",
        schema_file=tmp_path / "schema.json",
        prompt="review prompt",
    )

    assert command == [
        "claude",
        "--mcp-config",
        "C:\\temp\\claude-mcp.json",
        "-p",
        "--permission-mode",
        "bypassPermissions",
        "--model",
        "claude-sonnet-4",
        "--output-format",
        "json",
    ]
    assert stdin_text == "review prompt"


def test_build_agent_command_for_opencode_uses_run_mode(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(csi, "_resolve_cli_executable", lambda executable: executable)
    command, stdin_text = csi._build_agent_command(
        role_config=csi.AgentRoleConfig(
            client="opencode",
            model="gpt-5.4-mini",
            executable="opencode",
            args=("--format", "json"),
        ),
        raw_output_file=tmp_path / "raw.txt",
        schema_file=None,
        prompt="review prompt",
    )

    assert command == [
        "opencode",
        "run",
        "--format",
        "json",
        "--model",
        "gpt-5.4-mini",
        "review prompt",
    ]
    assert stdin_text is None


def test_resolve_cli_executable_prefers_cmd_over_bare_shim_on_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeCompletedProcess:
        def __init__(self, stdout: str) -> None:
            self.returncode = 0
            self.stdout = stdout

    monkeypatch.setattr(csi.os, "name", "nt")
    monkeypatch.setattr(csi.shutil, "which", lambda executable: f"C:\\tools\\{executable}.cmd")
    monkeypatch.setattr(
        csi.subprocess,
        "run",
        lambda *args, **kwargs: FakeCompletedProcess("C:\\tools\\opencode\nC:\\tools\\opencode.cmd\n"),
    )

    assert csi._resolve_cli_executable("opencode") == "C:\\tools\\opencode.cmd"


def test_extract_opencode_text_prefers_text_events() -> None:
    raw = "\n".join(
        [
            json.dumps({"type": "step_start", "part": {"type": "step-start"}}),
            json.dumps({"type": "text", "part": {"text": '{" count": 91}'}}),
            json.dumps({"type": "step_finish", "part": {"type": "step-finish"}}),
        ]
    )

    assert csi._extract_opencode_text(raw) == '{" count": 91}'


def test_extract_claude_text_prefers_result_payload_and_strips_fence() -> None:
    raw = json.dumps(
        {
            "type": "result",
            "result": "```json\n{\"diagnosis\": \"balanced\"}\n```",
        }
    )

    assert csi._extract_claude_text(raw) == '{"diagnosis": "balanced"}'


def test_build_codex_prompt_includes_machine_output_contract_for_opencode() -> None:
    prompt = csi._build_codex_prompt(
        review_client="opencode",
        target_concept_ratio=0.05,
        target_duplicate_rate=0.015,
        target_concept_without_taxonomy_ratio=0.6,
        current_tier2=csi.Tier2Params(),
        current_tier3=csi.Tier3Params(),
        current_catalog=csi._default_tier2_catalog(),
    )

    assert 'First character must be "{" and last character must be "}"' in prompt
    assert "Machine-consumed response: output exactly one JSON object." in prompt
    assert "You are not writing a human-readable report." in prompt


def test_run_agent_command_raises_timeout_and_logs_partial_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(csi, "AGENT_COMMAND_TIMEOUT_SECONDS", 7)

    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=kwargs.get("args", args[0]), timeout=7, output="partial", stderr="boom")

    monkeypatch.setattr(csi.subprocess, "run", fake_run)

    log_file = tmp_path / "agent.log"
    raw_output_file = tmp_path / "raw.txt"
    with pytest.raises(RuntimeError, match="timed out after 7s"):
        csi._run_agent_command(["claude", "-p"], log_file, raw_output_file, stdin_text="review")

    logged = log_file.read_text(encoding="utf-8")
    assert "TIMEOUT_SECONDS: 7" in logged
    assert "partial" in logged
    assert "boom" in logged


def test_apply_codex_taxonomy_tail_applies_valid_actions_and_skips_invalid(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "taxonomy_codex_queue.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "eid": "eid-1",
                        "name": "Talent",
                        "current_labels": ["Concept"],
                        "candidate_targets": [],
                    }
                ),
                json.dumps(
                    {
                        "eid": "eid-2",
                        "name": "problem",
                        "current_labels": ["Concept"],
                        "candidate_targets": [
                            {"eid": "eid-parent", "name": "General Problems", "labels": ["Concept"], "description": "Parent"}
                        ],
                    }
                ),
                json.dumps(
                    {
                        "eid": "eid-3",
                        "name": "Noise",
                        "current_labels": ["Concept"],
                        "candidate_targets": [],
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "codex_taxonomy_tail.json").write_text(
        json.dumps(
            {
                "summary": {
                    "queue_count": 3,
                    "processed_count": 3,
                    "recommended_relabels": 1,
                    "recommended_relations": 1,
                    "deprioritized": 1,
                    "blocked": 0,
                    "rationale": "Apply two safe repairs.",
                    "confidence": 0.79,
                },
                "decisions": [
                    {
                        "eid": "eid-1",
                        "name": "Talent",
                        "action": "relabel",
                        "label": "Condition",
                        "relation": "NONE",
                        "target_eid": None,
                        "reason": "Quality-like node.",
                        "priority": 1,
                        "confidence": 0.8,
                    },
                    {
                        "eid": "eid-2",
                        "name": "problem",
                        "action": "add_relation",
                        "label": None,
                        "relation": "SUBCLASS_OF",
                        "target_eid": "eid-parent",
                        "reason": "Clear parent.",
                        "priority": 2,
                        "confidence": 0.75,
                    },
                    {
                        "eid": "eid-3",
                        "name": "Noise",
                        "action": "relabel",
                        "label": "Imaginary Label",
                        "relation": "NONE",
                        "target_eid": None,
                        "reason": "Invalid label should be skipped.",
                        "priority": 3,
                        "confidence": 0.6,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    executed: list[str] = []

    class FakeResult:
        def __init__(self, row=None):
            self._row = row or {}

        def single(self):
            return self._row

    class FakeSession:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def run(self, query, **params):
            if "reverse_count" in query:
                return FakeResult({"reverse_count": 0})
            if "would_cycle" in query:
                return FakeResult({"would_cycle": False})
            if "conflicting_count" in query:
                return FakeResult({"conflicting_count": 0})
            executed.append(query.strip())
            return FakeResult({})

    class FakeDriver:
        def session(self, database=None):
            return FakeSession()

        def close(self):
            return None

    monkeypatch.setattr(csi.GraphDatabase, "driver", lambda *args, **kwargs: FakeDriver())

    applied = csi.apply_codex_taxonomy_tail(
        params=csi.TaxonomyParams(),
        current_catalog=csi._default_tier2_catalog(),
        iteration_dir=tmp_path,
        dry_run=False,
    )

    assert applied["summary"]["applied_relabels"] == 1
    assert applied["summary"]["applied_relations"] == 1
    assert applied["summary"]["skipped_invalid"] == 1
    assert len(executed) >= 2

def test_apply_catalog_guardrails_accepts_graph_native_additions_only() -> None:
    review = _valid_review_payload()
    review["proposed_tier2_catalog"] = {
        "labels": [label for label in csi.DEFAULT_TIER2_LABELS if label != "Market Feature"] + ["Execution Concept"],
        "add": ["Execution Concept"],
        "remove": [],
        "rename_map": {"Market Feature": "Execution Concept"},
        "guidance": {"Execution Concept": "Execution-specific ideas."},
        "rationale": "Add one label.",
    }

    catalog = csi._default_tier2_catalog()
    next_catalog, proposal_artifact, applied_artifact = csi.apply_catalog_guardrails(
        review=review,
        current_catalog=catalog,
        live_labels=set(csi.DEFAULT_TIER2_LABELS) | {"Execution Concept"},
    )

    assert proposal_artifact["is_valid"] is True
    assert proposal_artifact["structural_changes_applied"] is True
    assert next_catalog.labels == [*catalog.labels, "Execution Concept"]
    assert next_catalog.preferred_examples["Market Feature"] == catalog.preferred_examples["Market Feature"]
    assert next_catalog.preferred_examples["Execution Concept"] == "Execution-specific ideas."
    assert applied_artifact["structural_changes_applied"] is True


def test_apply_catalog_guardrails_applies_guidance_for_existing_labels_only() -> None:
    review = _valid_review_payload()
    review["proposed_tier2_catalog"] = {
        "labels": list(csi.DEFAULT_TIER2_LABELS),
        "add": [],
        "remove": [],
        "rename_map": {},
        "guidance": {"Trading Concept": "Use for leverage, liquidity, and execution ideas."},
        "rationale": "Tighten an existing label hint.",
    }

    catalog = csi._default_tier2_catalog()
    next_catalog, proposal_artifact, applied_artifact = csi.apply_catalog_guardrails(
        review=review,
        current_catalog=catalog,
    )

    assert proposal_artifact["is_valid"] is True
    assert proposal_artifact["structural_changes_applied"] is False
    assert next_catalog.labels == catalog.labels
    assert next_catalog.preferred_examples["Trading Concept"] == "Use for leverage, liquidity, and execution ideas."
    assert applied_artifact["is_fallback"] is False


def test_apply_catalog_guardrails_rejects_non_graph_native_additions() -> None:
    review = _valid_review_payload()
    review["proposed_tier2_catalog"] = {
        "labels": [*csi.DEFAULT_TIER2_LABELS, "Execution Concept"],
        "add": ["Execution Concept"],
        "remove": [],
        "rename_map": {},
        "guidance": {"Execution Concept": "Execution-specific ideas."},
        "rationale": "Add one label.",
    }

    catalog = csi._default_tier2_catalog()
    next_catalog, proposal_artifact, applied_artifact = csi.apply_catalog_guardrails(
        review=review,
        current_catalog=catalog,
        live_labels=set(csi.DEFAULT_TIER2_LABELS),
    )

    assert next_catalog.labels == catalog.labels
    assert proposal_artifact["accepted_additions"] == []
    assert proposal_artifact["rejected_additions"] == ["Execution Concept"]
    assert applied_artifact["structural_changes_applied"] is False


def test_update_plateau_streak_detects_flat_taxonomy_debt() -> None:
    prev = _valid_review_payload()
    prev["diagnosis"] = "taxonomy_debt"
    prev["kpis"]["concept_only_ratio"] = 0.221
    curr = _valid_review_payload()
    curr["diagnosis"] = "taxonomy_debt"
    curr["kpis"]["concept_only_ratio"] = 0.2205

    assert csi.update_plateau_streak(
        previous_review=prev,
        current_review=curr,
        current_streak=1,
        min_delta=0.002,
    ) == 2


def test_update_plateau_streak_resets_on_without_taxonomy_improvement() -> None:
    prev = _valid_review_payload()
    prev["diagnosis"] = "taxonomy_debt"
    prev["kpis"]["concept_only_ratio"] = 0.0300
    prev["taxonomy_kpis"]["concept_only_without_taxonomy_count"] = 160
    curr = _valid_review_payload()
    curr["diagnosis"] = "taxonomy_debt"
    curr["kpis"]["concept_only_ratio"] = 0.0299
    curr["taxonomy_kpis"]["concept_only_without_taxonomy_count"] = 140

    assert csi.update_plateau_streak(
        previous_review=prev,
        current_review=curr,
        current_streak=1,
        min_delta=0.002,
    ) == 0


def test_review_diagnostics_include_taxonomy_fields(tmp_path: Path) -> None:
    iteration_dir = tmp_path / "iteration_1"
    iteration_dir.mkdir()
    (iteration_dir / "tier2_decisions.jsonl").write_text(
        json.dumps({"name": "Weather", "status": "classified", "confidence": 0.44, "reason": "weak fit"}) + "\n",
        encoding="utf-8",
    )
    (iteration_dir / "taxonomy_decisions.jsonl").write_text(
        json.dumps(
            {
                "name": "Weather",
                "status": "classified",
                "action": "relabel",
                "confidence": 0.42,
                "reason": "weak fit",
                "skipped_reason": "Relabel confidence below apply threshold",
                "suspicious": True,
                "suspicious_reason": "Relabel confidence below apply threshold",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    csi._write_review_diagnostics(
        iteration_dir=iteration_dir,
        review=_valid_review_payload(),
        effective_diagnosis="taxonomy_debt",
        tier2_summary={"relabeled_without_taxonomy_support": 12, "dry_run": True},
        taxonomy_summary={
            "relations_added": {"SUBCLASS_OF": 2},
            "suspicious_relabels": 1,
            "residual_seed_count": 9,
            "carry_forward_seed_count": 4,
            "review_focus_seed_count": 2,
            "post_relabel_relation_attempts": 3,
            "post_relabel_relations_added": 1,
            "dry_run": True,
        },
    )
    payload = json.loads((iteration_dir / "review_diagnostics.json").read_text(encoding="utf-8"))
    assert payload["kpi_source"] == "live_graph_review"
    assert payload["execution_artifact_mode"] == "dry_run"
    assert payload["concept_only_without_taxonomy_ratio"] == pytest.approx(0.8)
    assert payload["consolidation_gate_pass"] is None
    assert payload["semantic_gate_pass"] is None
    assert payload["taxonomy_relation_counts_by_type"]["SUBCLASS_OF"] == 2
    assert payload["suspicious_relabel_count"] == 1
    assert payload["suspicious_keep_label_concept_count"] == 0
    assert payload["relabeled_nodes_missing_taxonomy_support"] == 12
    assert payload["residual_seed_count"] == 9
    assert payload["carry_forward_seed_count"] == 4
    assert payload["review_focus_seed_count"] == 2
    assert payload["post_relabel_relation_attempts"] == 3
    assert payload["post_relabel_relations_added"] == 1
    assert payload["suspicious_relabel_examples"][0]["name"] == "Weather"


def test_review_diagnostics_surface_suspicious_keep_label_examples(tmp_path: Path) -> None:
    iteration_dir = tmp_path / "iteration_1"
    iteration_dir.mkdir()
    (iteration_dir / "taxonomy_decisions.jsonl").write_text(
        json.dumps(
            {
                "name": "Logic",
                "status": "classified",
                "action": "keep_label",
                "confidence": 1.0,
                "reason": "broad concept",
                "suspicious": True,
                "suspicious_reason": "High-confidence keep_label on concept-only node without taxonomy support",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    csi._write_review_diagnostics(
        iteration_dir=iteration_dir,
        review=_valid_review_payload(),
        effective_diagnosis="taxonomy_debt",
        tier2_summary={"relabeled_without_taxonomy_support": 2},
        taxonomy_summary={"relations_added": {"SUBCLASS_OF": 1}, "suspicious_relabels": 0, "suspicious_keep_label_concepts": 1},
    )
    payload = json.loads((iteration_dir / "review_diagnostics.json").read_text(encoding="utf-8"))
    assert payload["suspicious_keep_label_concept_count"] == 1
    assert payload["suspicious_relabel_examples"][0]["name"] == "Logic"


def test_orchestrator_does_not_complete_when_only_consolidation_gate_passes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fake_tier1(*, params: csi.Tier2Params, dry_run: bool, iteration_dir: Path) -> None:
        return None

    def fake_tier2(*, params: csi.Tier2Params, catalog: csi.Tier2LabelCatalog, dry_run: bool, iteration_dir: Path) -> dict:
        return {"tier": 2, "processed_nodes": params.max_nodes}

    def fake_taxonomy(
        *,
        params: csi.TaxonomyParams,
        catalog: csi.Tier2LabelCatalog,
        dry_run: bool,
        iteration_dir: Path,
        prior_taxonomy_decisions_jsonl: str | None = None,
        prior_review_json: str | None = None,
    ) -> dict:
        return {"tier": "taxonomy", "relations_added": {"SUBCLASS_OF": 1}, "suspicious_relabels": 0}

    def fake_tier3(*, params: csi.Tier3Params, dry_run: bool, iteration_dir: Path) -> dict:
        return {"tier": 3, "judged_pairs": 5, "alias_acceptance_rate": 0.0}

    def fake_review(**kwargs) -> dict:
        payload = _valid_review_payload()
        payload["diagnosis"] = "balanced"
        payload["kpis"]["concept_only_ratio"] = 0.03
        payload["kpis"]["duplicate_candidate_rate"] = 0.0
        payload["kpis"]["duplicate_anchor_count"] = 0
        payload["taxonomy_kpis"]["concept_only_without_taxonomy_count"] = 150
        payload["taxonomy_kpis"]["concept_only_with_similarity_or_alias_count"] = 0
        return payload

    monkeypatch.setattr(csi, "run_tier1", fake_tier1)
    monkeypatch.setattr(csi, "run_tier2", fake_tier2)
    monkeypatch.setattr(csi, "run_taxonomy", fake_taxonomy)
    monkeypatch.setattr(csi, "run_tier3", fake_tier3)
    monkeypatch.setattr(csi, "run_codex_review", fake_review)


    state = csi.run_self_improving(
        csi.OrchestratorConfig(
            max_iterations=1,
            target_concept_ratio=0.05,
            target_duplicate_rate=0.015,
            target_concept_without_taxonomy_ratio=0.60,
            required_consecutive_passes=1,
            run_dir=str(tmp_path / "semantic_gate_run"),
            dry_run=True,
        )
    )

    assert state["status"] == "max_iterations_reached"
    iteration_1 = state["iterations"][1]
    assert iteration_1["consolidation_gate_pass"] is True
    assert iteration_1["semantic_gate_pass"] is False
    assert iteration_1["gate_pass"] is False
    assert state["consolidation_gate_pass"] is True
    assert state["semantic_gate_pass"] is False
    assert state["concept_only_without_taxonomy_ratio"] == pytest.approx(0.75)


def test_integration_smoke_dry_run_max_iterations(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[str] = []

    def fake_tier1(*, params: csi.Tier2Params, dry_run: bool, iteration_dir: Path) -> None:
        calls.append("tier1")

    def fake_tier2(*, params: csi.Tier2Params, catalog: csi.Tier2LabelCatalog, dry_run: bool, iteration_dir: Path) -> dict:
        calls.append("tier2")
        return {"tier": 2, "processed_nodes": params.max_nodes, "relabeled_without_taxonomy_support": 5}

    def fake_taxonomy(
        *,
        params: csi.TaxonomyParams,
        catalog: csi.Tier2LabelCatalog,
        dry_run: bool,
        iteration_dir: Path,
        prior_taxonomy_decisions_jsonl: str | None = None,
        prior_review_json: str | None = None,
    ) -> dict:
        calls.append("taxonomy")
        return {"tier": "taxonomy", "relations_added": {"SUBCLASS_OF": 2}, "suspicious_relabels": 0}

    def fake_tier3(*, params: csi.Tier3Params, dry_run: bool, iteration_dir: Path) -> dict:
        calls.append("tier3")
        return {"tier": 3, "judged_pairs": 30, "alias_acceptance_rate": 0.25}

    review_count = {"n": 0}

    def fake_review(
        *,
        codex_bin: str,
        target_concept_ratio: float,
        target_duplicate_rate: float,
        target_concept_without_taxonomy_ratio: float,
        current_tier2: csi.Tier2Params,
        current_tier3: csi.Tier3Params,
        current_catalog: csi.Tier2LabelCatalog,
        iteration_dir: Path,
    ) -> dict:
        review_count["n"] += 1
        calls.append(f"review{review_count['n']}")
        payload = _valid_review_payload()
        payload["proposed_tier2_catalog"] = {
            "labels": [label for label in current_catalog.labels if label != "Market Feature"] + ["Execution Concept"],
            "add": ["Execution Concept"],
            "remove": [],
            "rename_map": {"Market Feature": "Execution Concept"},
            "guidance": {"Trading Concept": "Refine existing guidance only."},
            "rationale": "Suggestion should be frozen structurally.",
        }
        return payload

    monkeypatch.setattr(csi, "run_tier1", fake_tier1)
    monkeypatch.setattr(csi, "run_tier2", fake_tier2)
    monkeypatch.setattr(csi, "run_taxonomy", fake_taxonomy)
    monkeypatch.setattr(csi, "run_tier3", fake_tier3)
    monkeypatch.setattr(csi, "run_codex_review", fake_review)


    run_dir = tmp_path / "run"
    state = csi.run_self_improving(
        csi.OrchestratorConfig(
            max_iterations=2,
            target_concept_ratio=0.15,
            target_duplicate_rate=0.015,
            required_consecutive_passes=2,
            run_dir=str(run_dir),
            codex_bin="codex",
            dry_run=True,
            resume=False,
        )
    )

    assert calls == ["tier1", "tier2", "taxonomy", "tier3", "review1", "tier2", "taxonomy", "tier3", "review2"]
    assert state["status"] == "max_iterations_reached"
    assert state["current_tier2_catalog"]["labels"] == [*csi.DEFAULT_TIER2_LABELS, "Execution Concept"]
    assert state["last_taxonomy_summary"]["relations_added"]["SUBCLASS_OF"] == 2


def test_orchestrator_passes_previous_taxonomy_and_review_artifacts_to_next_taxonomy_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    taxonomy_calls: list[tuple[str | None, str | None]] = []

    def fake_tier1(*, params: csi.Tier2Params, dry_run: bool, iteration_dir: Path) -> None:
        return None

    def fake_tier2(*, params: csi.Tier2Params, catalog: csi.Tier2LabelCatalog, dry_run: bool, iteration_dir: Path) -> dict:
        return {"tier": 2, "processed_nodes": params.max_nodes}

    def fake_taxonomy(
        *,
        params: csi.TaxonomyParams,
        catalog: csi.Tier2LabelCatalog,
        dry_run: bool,
        iteration_dir: Path,
        prior_taxonomy_decisions_jsonl: str | None = None,
        prior_review_json: str | None = None,
    ) -> dict:
        taxonomy_calls.append((prior_taxonomy_decisions_jsonl, prior_review_json))
        (iteration_dir / "taxonomy_decisions.jsonl").write_text("{}", encoding="utf-8")
        return {
            "tier": "taxonomy",
            "relations_added": {"SUBCLASS_OF": 1},
            "suspicious_relabels": 0,
        }

    def fake_tier3(*, params: csi.Tier3Params, dry_run: bool, iteration_dir: Path) -> dict:
        return {"tier": 3, "judged_pairs": 10, "alias_acceptance_rate": 0.02}

    def fake_review(**kwargs) -> dict:
        iteration_dir = kwargs["iteration_dir"]
        payload = _valid_review_payload()
        payload["diagnosis"] = "mixed_debt"
        payload["kpis"]["concept_only_ratio"] = 0.2
        (iteration_dir / "codex_review.json").write_text(json.dumps(payload), encoding="utf-8")
        return payload

    monkeypatch.setattr(csi, "run_tier1", fake_tier1)
    monkeypatch.setattr(csi, "run_tier2", fake_tier2)
    monkeypatch.setattr(csi, "run_taxonomy", fake_taxonomy)
    monkeypatch.setattr(csi, "run_tier3", fake_tier3)
    monkeypatch.setattr(csi, "run_codex_review", fake_review)


    run_dir = tmp_path / "run_with_prior_taxonomy"
    csi.run_self_improving(
        csi.OrchestratorConfig(
            max_iterations=3,
            target_concept_ratio=0.05,
            target_duplicate_rate=0.015,
            target_concept_without_taxonomy_ratio=0.60,
            required_consecutive_passes=2,
            run_dir=str(run_dir),
            dry_run=True,
        )
    )

    assert taxonomy_calls[0] == (None, None)
    assert taxonomy_calls[1][0] == str(run_dir / "iteration_0" / "taxonomy_decisions.jsonl")
    assert taxonomy_calls[1][1] is None
    assert taxonomy_calls[2][0] == str(run_dir / "iteration_1" / "taxonomy_decisions.jsonl")
    assert taxonomy_calls[2][1] == str(run_dir / "iteration_1" / "codex_review.json")


def test_orchestrator_skips_tier2_but_runs_taxonomy_when_only_taxonomy_ratio_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[str] = []

    def fake_tier1(*, params: csi.Tier2Params, dry_run: bool, iteration_dir: Path) -> None:
        calls.append("tier1")

    def fake_tier2(*, params: csi.Tier2Params, catalog: csi.Tier2LabelCatalog, dry_run: bool, iteration_dir: Path) -> dict:
        calls.append("tier2")
        return {"tier": 2, "processed_nodes": params.max_nodes}

    def fake_taxonomy(
        *,
        params: csi.TaxonomyParams,
        catalog: csi.Tier2LabelCatalog,
        dry_run: bool,
        iteration_dir: Path,
        prior_taxonomy_decisions_jsonl: str | None = None,
        prior_review_json: str | None = None,
    ) -> dict:
        calls.append("taxonomy")
        return {"tier": "taxonomy", "relations_added": {"SUBCLASS_OF": 1}, "suspicious_relabels": 0}

    def fake_tier3(*, params: csi.Tier3Params, dry_run: bool, iteration_dir: Path) -> dict:
        calls.append("tier3")
        return {"tier": 3, "judged_pairs": 10, "alias_acceptance_rate": 0.01}

    def fake_review(**kwargs) -> dict:
        calls.append("review")
        payload = _valid_review_payload()
        payload["diagnosis"] = "taxonomy_debt"
        payload["kpis"]["concept_only_ratio"] = 0.03
        payload["kpis"]["concept_only_count"] = 100
        payload["kpis"]["duplicate_candidate_rate"] = 0.0
        payload["kpis"]["duplicate_anchor_count"] = 0
        payload["taxonomy_kpis"]["concept_only_without_taxonomy_count"] = 70
        payload["taxonomy_kpis"]["concept_only_with_similarity_or_alias_count"] = 0
        return payload

    monkeypatch.setattr(csi, "run_tier1", fake_tier1)
    monkeypatch.setattr(csi, "run_tier2", fake_tier2)
    monkeypatch.setattr(csi, "run_taxonomy", fake_taxonomy)
    monkeypatch.setattr(csi, "run_tier3", fake_tier3)
    monkeypatch.setattr(csi, "run_codex_review", fake_review)

    state = csi.run_self_improving(
        csi.OrchestratorConfig(
            max_iterations=2,
            target_concept_ratio=0.05,
            target_duplicate_rate=0.015,
            target_concept_without_taxonomy_ratio=0.60,
            required_consecutive_passes=2,
            run_dir=str(tmp_path / "taxonomy_only_rerun"),
            dry_run=True,
        )
    )

    assert calls == ["tier1", "tier2", "taxonomy", "tier3", "review", "taxonomy", "review"]
    iteration_1 = state["iterations"][1]
    assert iteration_1["tier2_skipped"] is True
    assert iteration_1["tier2_skip_reason"] == csi.TIER2_SKIP_REASON
    assert "tier2_summary" not in iteration_1
    assert iteration_1["taxonomy_skipped"] is False
    assert "taxonomy_summary" in iteration_1


def test_orchestrator_runs_codex_taxonomy_tail_when_taxonomy_queue_exists(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(csi, "_compute_live_concept_only_without_taxonomy_ratio", lambda params: 0.7)

    def fake_tier1(*, params: csi.Tier2Params, dry_run: bool, iteration_dir: Path) -> None:
        calls.append("tier1")

    def fake_tier2(*, params: csi.Tier2Params, catalog: csi.Tier2LabelCatalog, dry_run: bool, iteration_dir: Path) -> dict:
        calls.append("tier2")
        return {"tier": 2, "processed_nodes": params.max_nodes}

    def fake_taxonomy(
        *,
        params: csi.TaxonomyParams,
        catalog: csi.Tier2LabelCatalog,
        dry_run: bool,
        iteration_dir: Path,
        prior_taxonomy_decisions_jsonl: str | None = None,
        prior_review_json: str | None = None,
    ) -> dict:
        calls.append("taxonomy")
        if iteration_dir.name == "iteration_1":
            (iteration_dir / "taxonomy_codex_queue.jsonl").write_text(
                json.dumps(
                    {
                        "eid": "eid-1",
                        "name": "problem",
                        "current_labels": ["Concept"],
                        "candidate_targets": [{"eid": "eid-parent", "name": "General Problems", "labels": ["Concept"]}],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        return {
            "tier": "taxonomy",
            "relations_added": {"SUBCLASS_OF": 1},
            "suspicious_relabels": 0,
            "codex_queue_count": 1 if iteration_dir.name == "iteration_1" else 0,
        }

    def fake_tier3(*, params: csi.Tier3Params, dry_run: bool, iteration_dir: Path) -> dict:
        calls.append("tier3")
        return {"tier": 3, "judged_pairs": 10, "alias_acceptance_rate": 0.01}

    def fake_review(**kwargs) -> dict:
        calls.append("review")
        payload = _valid_review_payload()
        payload["diagnosis"] = "taxonomy_debt"
        payload["kpis"]["concept_only_ratio"] = 0.03
        payload["kpis"]["concept_only_count"] = 100
        payload["kpis"]["duplicate_candidate_rate"] = 0.0
        payload["kpis"]["duplicate_anchor_count"] = 0
        payload["taxonomy_kpis"]["concept_only_without_taxonomy_count"] = 70
        payload["taxonomy_kpis"]["concept_only_with_similarity_or_alias_count"] = 0
        return payload

    def fake_codex_tail(**kwargs) -> dict:
        calls.append("codex_tail")
        payload = {
            "summary": {
                "queue_count": 1,
                "processed_count": 1,
                "recommended_relabels": 0,
                "recommended_relations": 1,
                "deprioritized": 0,
                "blocked": 0,
                "rationale": "One strong parent fix.",
                "confidence": 0.8,
            },
            "decisions": [
                {
                    "eid": "eid-1",
                    "name": "problem",
                    "action": "add_relation",
                    "label": None,
                    "relation": "SUBCLASS_OF",
                    "target_eid": "eid-parent",
                    "reason": "General Problems is the broader parent.",
                    "priority": 1,
                    "confidence": 0.8,
                }
            ],
        }
        (kwargs["iteration_dir"] / "codex_taxonomy_tail.json").write_text(json.dumps(payload), encoding="utf-8")
        return payload

    def fake_apply_codex_tail(
        *,
        params: csi.TaxonomyParams,
        current_catalog: csi.Tier2LabelCatalog,
        iteration_dir: Path,
        dry_run: bool,
    ) -> dict:
        calls.append("codex_tail_apply")
        payload = {
            "summary": {
                "queue_count": 1,
                "processed_count": 1,
                "applied_relabels": 0,
                "applied_relations": 1,
                "deprioritized": 0,
                "blocked": 0,
                "skipped_invalid": 0,
            },
            "results": [],
        }
        (iteration_dir / "applied_codex_taxonomy_tail.json").write_text(json.dumps(payload), encoding="utf-8")
        return payload

    monkeypatch.setattr(csi, "run_tier1", fake_tier1)
    monkeypatch.setattr(csi, "run_tier2", fake_tier2)
    monkeypatch.setattr(csi, "run_taxonomy", fake_taxonomy)
    monkeypatch.setattr(csi, "run_tier3", fake_tier3)
    monkeypatch.setattr(csi, "run_codex_review", fake_review)
    monkeypatch.setattr(csi, "run_codex_taxonomy_tail", fake_codex_tail)
    monkeypatch.setattr(csi, "apply_codex_taxonomy_tail", fake_apply_codex_tail)

    state = csi.run_self_improving(
        csi.OrchestratorConfig(
            max_iterations=2,
            target_concept_ratio=0.05,
            target_duplicate_rate=0.015,
            target_concept_without_taxonomy_ratio=0.60,
            required_consecutive_passes=2,
            run_dir=str(tmp_path / "codex_tail_run"),
            dry_run=True,
        )
    )

    assert calls == ["tier1", "tier2", "taxonomy", "tier3", "review", "taxonomy", "codex_tail", "codex_tail_apply", "review"]
    iteration_1 = state["iterations"][1]
    assert iteration_1["codex_taxonomy_tail_skipped"] is False
    assert iteration_1["codex_taxonomy_tail_summary"]["application"]["applied_relations"] == 1
    assert iteration_1["taxonomy_summary"]["codex_tail_applied_relations"] == 1


def test_initial_iteration_runs_codex_taxonomy_tail_before_tier3(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(csi, "_compute_live_concept_only_without_taxonomy_ratio", lambda params: 0.7)

    def fake_tier1(*, params: csi.Tier2Params, dry_run: bool, iteration_dir: Path) -> None:
        calls.append("tier1")

    def fake_tier2(*, params: csi.Tier2Params, catalog: csi.Tier2LabelCatalog, dry_run: bool, iteration_dir: Path) -> dict:
        calls.append("tier2")
        return {"tier": 2, "processed_nodes": params.max_nodes}

    def fake_taxonomy(
        *,
        params: csi.TaxonomyParams,
        catalog: csi.Tier2LabelCatalog,
        dry_run: bool,
        iteration_dir: Path,
        prior_taxonomy_decisions_jsonl: str | None = None,
        prior_review_json: str | None = None,
    ) -> dict:
        calls.append("taxonomy")
        (iteration_dir / "taxonomy_codex_queue.jsonl").write_text(
            json.dumps(
                {
                    "eid": "eid-1",
                    "name": "problem",
                    "current_labels": ["Concept"],
                    "candidate_targets": [{"eid": "eid-parent", "name": "General Problems", "labels": ["Concept"]}],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return {
            "tier": "taxonomy",
            "relations_added": {"SUBCLASS_OF": 1},
            "suspicious_relabels": 0,
            "codex_queue_count": 1,
        }

    def fake_tier3(*, params: csi.Tier3Params, dry_run: bool, iteration_dir: Path) -> dict:
        calls.append("tier3")
        return {"tier": 3, "judged_pairs": 10, "alias_acceptance_rate": 0.01}

    def fake_review(**kwargs) -> dict:
        calls.append("review")
        payload = _valid_review_payload()
        payload["diagnosis"] = "balanced"
        payload["kpis"]["concept_only_ratio"] = 0.03
        payload["kpis"]["concept_only_count"] = 100
        payload["kpis"]["duplicate_candidate_rate"] = 0.0
        payload["kpis"]["duplicate_anchor_count"] = 0
        payload["taxonomy_kpis"]["concept_only_without_taxonomy_count"] = 60
        payload["taxonomy_kpis"]["concept_only_with_similarity_or_alias_count"] = 0
        return payload

    def fake_codex_tail(**kwargs) -> dict:
        calls.append("codex_tail")
        payload = {
            "summary": {
                "queue_count": 1,
                "processed_count": 1,
                "recommended_relabels": 0,
                "recommended_relations": 1,
                "deprioritized": 0,
                "blocked": 0,
                "rationale": "One strong parent fix.",
                "confidence": 0.8,
            },
            "decisions": [
                {
                    "eid": "eid-1",
                    "name": "problem",
                    "action": "add_relation",
                    "label": None,
                    "relation": "SUBCLASS_OF",
                    "target_eid": "eid-parent",
                    "reason": "General Problems is the broader parent.",
                    "priority": 1,
                    "confidence": 0.8,
                }
            ],
        }
        (kwargs["iteration_dir"] / "codex_taxonomy_tail.json").write_text(json.dumps(payload), encoding="utf-8")
        return payload

    def fake_apply_codex_tail(
        *,
        params: csi.TaxonomyParams,
        current_catalog: csi.Tier2LabelCatalog,
        iteration_dir: Path,
        dry_run: bool,
    ) -> dict:
        calls.append("codex_tail_apply")
        payload = {
            "summary": {
                "queue_count": 1,
                "processed_count": 1,
                "applied_relabels": 0,
                "applied_relations": 1,
                "deprioritized": 0,
                "blocked": 0,
                "skipped_invalid": 0,
            },
            "results": [],
        }
        (iteration_dir / "applied_codex_taxonomy_tail.json").write_text(json.dumps(payload), encoding="utf-8")
        return payload

    monkeypatch.setattr(csi, "run_tier1", fake_tier1)
    monkeypatch.setattr(csi, "run_tier2", fake_tier2)
    monkeypatch.setattr(csi, "run_taxonomy", fake_taxonomy)
    monkeypatch.setattr(csi, "run_tier3", fake_tier3)
    monkeypatch.setattr(csi, "run_codex_review", fake_review)
    monkeypatch.setattr(csi, "run_codex_taxonomy_tail", fake_codex_tail)
    monkeypatch.setattr(csi, "apply_codex_taxonomy_tail", fake_apply_codex_tail)

    state = csi.run_self_improving(
        csi.OrchestratorConfig(
            max_iterations=1,
            target_concept_ratio=0.05,
            target_duplicate_rate=0.015,
            target_concept_without_taxonomy_ratio=0.60,
            required_consecutive_passes=1,
            run_dir=str(tmp_path / "initial_codex_tail_run"),
            dry_run=True,
        )
    )

    assert calls[:6] == ["tier1", "tier2", "taxonomy", "codex_tail", "codex_tail_apply", "tier3"]
    iteration_0 = state["iterations"][0]
    assert iteration_0["codex_taxonomy_tail_skipped"] is False
    assert iteration_0["codex_taxonomy_tail_summary"]["application"]["applied_relations"] == 1


def test_orchestrator_skips_tier2_and_taxonomy_for_duplicate_only_debt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[str] = []

    def fake_tier1(*, params: csi.Tier2Params, dry_run: bool, iteration_dir: Path) -> None:
        calls.append("tier1")

    def fake_tier2(*, params: csi.Tier2Params, catalog: csi.Tier2LabelCatalog, dry_run: bool, iteration_dir: Path) -> dict:
        calls.append("tier2")
        return {"tier": 2, "processed_nodes": params.max_nodes}

    def fake_taxonomy(
        *,
        params: csi.TaxonomyParams,
        catalog: csi.Tier2LabelCatalog,
        dry_run: bool,
        iteration_dir: Path,
        prior_taxonomy_decisions_jsonl: str | None = None,
        prior_review_json: str | None = None,
    ) -> dict:
        calls.append("taxonomy")
        return {"tier": "taxonomy", "relations_added": {"SUBCLASS_OF": 1}, "suspicious_relabels": 0}

    def fake_tier3(*, params: csi.Tier3Params, dry_run: bool, iteration_dir: Path) -> dict:
        calls.append("tier3")
        return {"tier": 3, "judged_pairs": 30, "alias_acceptance_rate": 0.25}

    def fake_review(**kwargs) -> dict:
        calls.append("review")
        payload = _valid_review_payload()
        payload["diagnosis"] = "duplicate_debt"
        payload["kpis"]["concept_only_ratio"] = 0.03
        payload["kpis"]["concept_only_count"] = 100
        payload["kpis"]["duplicate_candidate_rate"] = 0.04
        payload["taxonomy_kpis"]["concept_only_without_taxonomy_count"] = 60
        payload["taxonomy_kpis"]["concept_only_with_similarity_or_alias_count"] = 15
        return payload

    monkeypatch.setattr(csi, "run_tier1", fake_tier1)
    monkeypatch.setattr(csi, "run_tier2", fake_tier2)
    monkeypatch.setattr(csi, "run_taxonomy", fake_taxonomy)
    monkeypatch.setattr(csi, "run_tier3", fake_tier3)
    monkeypatch.setattr(csi, "run_codex_review", fake_review)

    state = csi.run_self_improving(
        csi.OrchestratorConfig(
            max_iterations=2,
            target_concept_ratio=0.05,
            target_duplicate_rate=0.015,
            target_concept_without_taxonomy_ratio=0.60,
            required_consecutive_passes=2,
            run_dir=str(tmp_path / "duplicate_only_rerun"),
            dry_run=True,
        )
    )

    assert calls == ["tier1", "tier2", "taxonomy", "tier3", "review", "tier3", "review"]
    iteration_1 = state["iterations"][1]
    assert iteration_1["tier2_skipped"] is True
    assert iteration_1["tier2_skip_reason"] == csi.TIER2_SKIP_REASON
    assert iteration_1["taxonomy_skipped"] is True
    assert iteration_1["taxonomy_skip_reason"] == csi.TAXONOMY_SKIP_REASON
    assert iteration_1["tier3_skipped"] is False


def test_taxonomy_debt_skips_tier3_and_can_plateau(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[str] = []

    def fake_tier1(*, params: csi.Tier2Params, dry_run: bool, iteration_dir: Path) -> None:
        calls.append("tier1")

    def fake_tier2(*, params: csi.Tier2Params, catalog: csi.Tier2LabelCatalog, dry_run: bool, iteration_dir: Path) -> dict:
        calls.append("tier2")
        return {"tier": 2, "processed_nodes": params.max_nodes, "relabeled_without_taxonomy_support": 3}

    def fake_taxonomy(
        *,
        params: csi.TaxonomyParams,
        catalog: csi.Tier2LabelCatalog,
        dry_run: bool,
        iteration_dir: Path,
        prior_taxonomy_decisions_jsonl: str | None = None,
        prior_review_json: str | None = None,
    ) -> dict:
        calls.append("taxonomy")
        return {"tier": "taxonomy", "relations_added": {"SUBCLASS_OF": 1}, "suspicious_relabels": 0}

    def fake_tier3(*, params: csi.Tier3Params, dry_run: bool, iteration_dir: Path) -> dict:
        calls.append("tier3")
        return {"tier": 3, "judged_pairs": 30, "alias_acceptance_rate": 0.25}

    review_payloads = [
        {
            **_valid_review_payload(),
            "diagnosis": "taxonomy_debt",
            "kpis": {
                "entity_count": 100,
                "concept_only_count": 22,
                "concept_only_ratio": 0.22,
                "duplicate_anchor_count": 0,
                "duplicate_candidate_rate": 0.0,
                "subclass_rel_count": 1,
            },
            "taxonomy_kpis": {
                "concept_only_without_taxonomy_count": 20,
                "concept_only_degree_le_2_count": 10,
                "concept_only_degree_le_3_count": 15,
                "concept_only_with_similarity_or_alias_count": 0,
            },
        },
        {
            **_valid_review_payload(),
            "diagnosis": "taxonomy_debt",
            "kpis": {
                "entity_count": 100,
                "concept_only_count": 219,
                "concept_only_ratio": 0.2195,
                "duplicate_anchor_count": 0,
                "duplicate_candidate_rate": 0.0,
                "subclass_rel_count": 1,
            },
            "taxonomy_kpis": {
                "concept_only_without_taxonomy_count": 200,
                "concept_only_degree_le_2_count": 100,
                "concept_only_degree_le_3_count": 150,
                "concept_only_with_similarity_or_alias_count": 0,
            },
        },
    ]

    def fake_review(**kwargs) -> dict:
        calls.append("review")
        return review_payloads.pop(0)

    monkeypatch.setattr(csi, "run_tier1", fake_tier1)
    monkeypatch.setattr(csi, "run_tier2", fake_tier2)
    monkeypatch.setattr(csi, "run_taxonomy", fake_taxonomy)
    monkeypatch.setattr(csi, "run_tier3", fake_tier3)
    monkeypatch.setattr(csi, "run_codex_review", fake_review)


    run_dir = tmp_path / "taxonomy_run"
    state = csi.run_self_improving(
        csi.OrchestratorConfig(
            max_iterations=5,
            taxonomy_plateau_reviews=1,
            taxonomy_plateau_min_delta=0.002,
            run_dir=str(run_dir),
            dry_run=True,
        )
    )

    assert state["status"] == "plateau_detected"
    assert calls == ["tier1", "tier2", "taxonomy", "tier3", "review", "tier2", "taxonomy", "review"]
    iteration_1 = state["iterations"][1]
    assert iteration_1["tier2_skipped"] is False
    assert iteration_1["taxonomy_skipped"] is False
    assert iteration_1["tier3_skipped"] is True
    assert iteration_1["tier3_skip_reason"] == "taxonomy_debt"


def test_resume_does_not_retroactively_run_skipped_tier2_or_taxonomy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[str] = []

    def fake_tier1(*, params: csi.Tier2Params, dry_run: bool, iteration_dir: Path) -> None:
        calls.append("tier1")

    def fake_tier2(*, params: csi.Tier2Params, catalog: csi.Tier2LabelCatalog, dry_run: bool, iteration_dir: Path) -> dict:
        calls.append("tier2")
        return {"tier": 2, "processed_nodes": params.max_nodes}

    def fake_taxonomy(
        *,
        params: csi.TaxonomyParams,
        catalog: csi.Tier2LabelCatalog,
        dry_run: bool,
        iteration_dir: Path,
        prior_taxonomy_decisions_jsonl: str | None = None,
        prior_review_json: str | None = None,
    ) -> dict:
        calls.append("taxonomy")
        return {"tier": "taxonomy", "relations_added": {"SUBCLASS_OF": 1}}

    def fake_tier3(*, params: csi.Tier3Params, dry_run: bool, iteration_dir: Path) -> dict:
        calls.append("tier3")
        return {"tier": 3, "judged_pairs": 30, "alias_acceptance_rate": 0.25}

    def fake_review(**kwargs) -> dict:
        calls.append("review")
        payload = _valid_review_payload()
        payload["diagnosis"] = "balanced"
        payload["kpis"]["concept_only_ratio"] = 0.03
        payload["kpis"]["concept_only_count"] = 100
        payload["kpis"]["duplicate_candidate_rate"] = 0.0
        payload["kpis"]["duplicate_anchor_count"] = 0
        payload["taxonomy_kpis"]["concept_only_without_taxonomy_count"] = 60
        payload["taxonomy_kpis"]["concept_only_with_similarity_or_alias_count"] = 0
        return payload

    monkeypatch.setattr(csi, "run_tier1", fake_tier1)
    monkeypatch.setattr(csi, "run_tier2", fake_tier2)
    monkeypatch.setattr(csi, "run_taxonomy", fake_taxonomy)
    monkeypatch.setattr(csi, "run_tier3", fake_tier3)
    monkeypatch.setattr(csi, "run_codex_review", fake_review)

    run_dir = tmp_path / "resume_skipped_tiers"
    run_dir.mkdir()
    previous_review = _valid_review_payload()
    previous_review["diagnosis"] = "duplicate_debt"
    previous_review["kpis"]["concept_only_ratio"] = 0.03
    previous_review["kpis"]["concept_only_count"] = 100
    previous_review["kpis"]["duplicate_candidate_rate"] = 0.04
    previous_review["taxonomy_kpis"]["concept_only_without_taxonomy_count"] = 60
    previous_review["taxonomy_kpis"]["concept_only_with_similarity_or_alias_count"] = 10
    state = {
        "status": "running",
        "started_at_utc": "2026-03-06T13:56:33+00:00",
        "config": csi.asdict(csi.OrchestratorConfig(max_iterations=2, run_dir=str(run_dir), dry_run=True, resume=True)),
        "next_iteration": 1,
        "consecutive_passes": 0,
        "plateau_streak": 0,
        "active_step": "idle",
        "last_error": None,
        "current_params": {
            "tier2": csi.asdict(csi.Tier2Params()),
            "taxonomy": csi.asdict(csi.TaxonomyParams()),
            "tier3": csi.asdict(csi.Tier3Params()),
        },
        "current_tier2_catalog": csi._serialize_catalog(csi._default_tier2_catalog()),
        "last_tier2_catalog_proposal": None,
        "last_tier2_summary": {"tier": 2, "processed_nodes": 10},
        "last_taxonomy_summary": {"tier": "taxonomy", "relations_added": {"SUBCLASS_OF": 1}},
        "last_tier3_summary": {"judged_pairs": 30, "alias_acceptance_rate": 0.25},
        "last_review": {**previous_review, "diagnosis": "duplicate_debt"},
        "last_diagnosis": "duplicate_debt",
        "iterations": [
            {
                "iteration": 1,
                "phase": "review_and_rerun",
                "review": previous_review,
                "raw_diagnosis": "duplicate_debt",
                "diagnosis": "duplicate_debt",
                "gate_pass": False,
                "consecutive_passes": 0,
                "plateau_streak": 0,
                "applied_params": {
                    "tier2": csi.asdict(csi.Tier2Params()),
                    "taxonomy": csi.asdict(csi.TaxonomyParams()),
                    "tier3": csi.asdict(csi.Tier3Params()),
                },
                "tier2_catalog_applied": csi._serialize_catalog(csi._default_tier2_catalog()),
                "tier2_skipped": True,
                "tier2_skip_reason": csi.TIER2_SKIP_REASON,
                "taxonomy_skipped": True,
                "taxonomy_skip_reason": csi.TAXONOMY_SKIP_REASON,
                "tier3_summary": {"tier": 3, "judged_pairs": 30, "alias_acceptance_rate": 0.25},
            }
        ],
        "run_dir": str(run_dir),
    }
    (run_dir / "run_state.json").write_text(json.dumps(state), encoding="utf-8")

    resumed = csi.run_self_improving(
        csi.OrchestratorConfig(
            max_iterations=2,
            target_concept_ratio=0.05,
            target_duplicate_rate=0.015,
            target_concept_without_taxonomy_ratio=0.60,
            required_consecutive_passes=1,
            run_dir=str(run_dir),
            dry_run=True,
            resume=True,
        )
    )

    assert calls == ["review"]
    assert resumed["status"] == "completed"


def test_resume_does_not_retroactively_run_skipped_tier3(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[str] = []

    def fake_tier1(*, params: csi.Tier2Params, dry_run: bool, iteration_dir: Path) -> None:
        calls.append("tier1")

    def fake_tier2(*, params: csi.Tier2Params, catalog: csi.Tier2LabelCatalog, dry_run: bool, iteration_dir: Path) -> dict:
        calls.append("tier2")
        return {"tier": 2, "processed_nodes": params.max_nodes}

    def fake_taxonomy(
        *,
        params: csi.TaxonomyParams,
        catalog: csi.Tier2LabelCatalog,
        dry_run: bool,
        iteration_dir: Path,
        prior_taxonomy_decisions_jsonl: str | None = None,
        prior_review_json: str | None = None,
    ) -> dict:
        calls.append("taxonomy")
        return {"tier": "taxonomy", "relations_added": {"SUBCLASS_OF": 1}}

    def fake_tier3(*, params: csi.Tier3Params, dry_run: bool, iteration_dir: Path) -> dict:
        calls.append("tier3")
        return {"tier": 3, "judged_pairs": 30, "alias_acceptance_rate": 0.25}

    def fake_review(**kwargs) -> dict:
        calls.append("review")
        payload = _valid_review_payload()
        payload["diagnosis"] = "balanced"
        payload["kpis"]["concept_only_ratio"] = 0.03
        payload["kpis"]["concept_only_count"] = 100
        payload["kpis"]["duplicate_candidate_rate"] = 0.0
        payload["kpis"]["duplicate_anchor_count"] = 0
        payload["taxonomy_kpis"]["concept_only_without_taxonomy_count"] = 60
        payload["taxonomy_kpis"]["concept_only_with_similarity_or_alias_count"] = 0
        return payload

    monkeypatch.setattr(csi, "run_tier1", fake_tier1)
    monkeypatch.setattr(csi, "run_tier2", fake_tier2)
    monkeypatch.setattr(csi, "run_taxonomy", fake_taxonomy)
    monkeypatch.setattr(csi, "run_tier3", fake_tier3)
    monkeypatch.setattr(csi, "run_codex_review", fake_review)


    run_dir = tmp_path / "resume_run"
    run_dir.mkdir()
    previous_review = _valid_review_payload()
    previous_review["diagnosis"] = "duplicate_debt"
    previous_review["kpis"]["concept_only_ratio"] = 0.03
    previous_review["kpis"]["concept_only_count"] = 100
    previous_review["kpis"]["duplicate_candidate_rate"] = 0.04
    previous_review["taxonomy_kpis"]["concept_only_without_taxonomy_count"] = 60
    previous_review["taxonomy_kpis"]["concept_only_with_similarity_or_alias_count"] = 15
    state = {
        "status": "running",
        "started_at_utc": "2026-03-06T13:56:33+00:00",
        "config": csi.asdict(csi.OrchestratorConfig(max_iterations=3, run_dir=str(run_dir), dry_run=True, resume=True)),
        "next_iteration": 2,
        "consecutive_passes": 0,
        "plateau_streak": 1,
        "active_step": "idle",
        "last_error": None,
        "current_params": {
            "tier2": csi.asdict(csi.Tier2Params()),
            "taxonomy": csi.asdict(csi.TaxonomyParams()),
            "tier3": csi.asdict(csi.Tier3Params()),
        },
        "current_tier2_catalog": csi._serialize_catalog(csi._default_tier2_catalog()),
        "last_tier2_catalog_proposal": None,
        "last_tier2_catalog_migration": {"dry_run": True, "operations": [], "applied": True},
        "last_tier2_summary": {"tier": 2, "processed_nodes": 10},
        "last_taxonomy_summary": {"tier": "taxonomy", "relations_added": {"SUBCLASS_OF": 1}},
        "last_tier3_summary": {"judged_pairs": 30, "alias_acceptance_rate": 0.25},
        "last_review": {**previous_review, "diagnosis": "duplicate_debt"},
        "last_diagnosis": "duplicate_debt",
        "iterations": [
            {
                "iteration": 1,
                "phase": "review_and_rerun",
                "review": previous_review,
                "raw_diagnosis": "duplicate_debt",
                "diagnosis": "duplicate_debt",
                "gate_pass": False,
                "consecutive_passes": 0,
                "plateau_streak": 1,
                "tier2_summary": {"tier": 2, "processed_nodes": 10},
                "taxonomy_summary": {"tier": "taxonomy", "relations_added": {"SUBCLASS_OF": 1}},
                "tier2_catalog": csi._serialize_catalog(csi._default_tier2_catalog()),
                "tier3_skipped": True,
                "tier3_skip_reason": "taxonomy_debt",
            }
        ],
        "run_dir": str(run_dir),
    }
    (run_dir / "run_state.json").write_text(json.dumps(state), encoding="utf-8")

    resumed = csi.run_self_improving(
        csi.OrchestratorConfig(
            max_iterations=3,
            taxonomy_plateau_reviews=2,
            taxonomy_plateau_min_delta=0.002,
            required_consecutive_passes=1,
            run_dir=str(run_dir),
            dry_run=True,
            resume=True,
        )
    )

    assert calls == ["review"]
    assert resumed["status"] == "completed"
    assert resumed["last_diagnosis"] == "balanced"


def test_resume_reuses_existing_codex_taxonomy_tail_artifacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(csi, "_compute_live_concept_only_without_taxonomy_ratio", lambda params: 0.7)

    def fake_review(**kwargs) -> dict:
        calls.append("review")
        payload = _valid_review_payload()
        payload["diagnosis"] = "taxonomy_debt"
        payload["kpis"]["concept_only_ratio"] = 0.03
        payload["kpis"]["concept_only_count"] = 100
        payload["kpis"]["duplicate_candidate_rate"] = 0.0
        payload["kpis"]["duplicate_anchor_count"] = 0
        payload["taxonomy_kpis"]["concept_only_without_taxonomy_count"] = 70
        payload["taxonomy_kpis"]["concept_only_with_similarity_or_alias_count"] = 0
        return payload

    def fail_codex_tail(**kwargs):
        raise AssertionError("Codex taxonomy tail should not rerun when artifacts already exist")

    def fail_apply_codex_tail(**kwargs):
        raise AssertionError("Codex taxonomy tail should not reapply when artifacts already exist")

    monkeypatch.setattr(csi, "run_codex_review", fake_review)
    monkeypatch.setattr(csi, "run_codex_taxonomy_tail", fail_codex_tail)
    monkeypatch.setattr(csi, "apply_codex_taxonomy_tail", fail_apply_codex_tail)

    run_dir = tmp_path / "resume_codex_tail"
    iteration_dir = run_dir / "iteration_1"
    iteration_dir.mkdir(parents=True)
    (iteration_dir / "taxonomy_codex_queue.jsonl").write_text(
        json.dumps(
            {
                "eid": "eid-1",
                "name": "problem",
                "current_labels": ["Concept"],
                "candidate_targets": [{"eid": "eid-parent", "name": "General Problems", "labels": ["Concept"]}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (iteration_dir / "codex_taxonomy_tail.json").write_text(
        json.dumps(
            {
                "summary": {
                    "queue_count": 1,
                    "processed_count": 1,
                    "recommended_relabels": 0,
                    "recommended_relations": 1,
                    "deprioritized": 0,
                    "blocked": 0,
                    "rationale": "Existing artifact.",
                    "confidence": 0.8,
                },
                "decisions": [
                    {
                        "eid": "eid-1",
                        "name": "problem",
                        "action": "blocked",
                        "label": None,
                        "relation": "NONE",
                        "target_eid": None,
                        "reason": "Existing artifact.",
                        "priority": 1,
                        "confidence": 0.7,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (iteration_dir / "applied_codex_taxonomy_tail.json").write_text(
        json.dumps(
            {
                "summary": {
                    "queue_count": 1,
                    "processed_count": 1,
                    "applied_relabels": 0,
                    "applied_relations": 0,
                    "deprioritized": 0,
                    "blocked": 1,
                    "skipped_invalid": 0,
                },
                "results": [],
            }
        ),
        encoding="utf-8",
    )
    previous_review = _valid_review_payload()
    previous_review["diagnosis"] = "taxonomy_debt"
    previous_review["kpis"]["concept_only_ratio"] = 0.03
    previous_review["kpis"]["concept_only_count"] = 100
    previous_review["kpis"]["duplicate_candidate_rate"] = 0.0
    previous_review["kpis"]["duplicate_anchor_count"] = 0
    previous_review["taxonomy_kpis"]["concept_only_without_taxonomy_count"] = 70
    previous_review["taxonomy_kpis"]["concept_only_with_similarity_or_alias_count"] = 0
    state = {
        "status": "running",
        "started_at_utc": "2026-03-06T13:56:33+00:00",
        "config": csi.asdict(csi.OrchestratorConfig(max_iterations=2, run_dir=str(run_dir), dry_run=True, resume=True)),
        "next_iteration": 1,
        "consecutive_passes": 0,
        "plateau_streak": 0,
        "active_step": "idle",
        "last_error": None,
        "current_params": {
            "tier2": csi.asdict(csi.Tier2Params()),
            "taxonomy": csi.asdict(csi.TaxonomyParams()),
            "tier3": csi.asdict(csi.Tier3Params()),
        },
        "current_tier2_catalog": csi._serialize_catalog(csi._default_tier2_catalog()),
        "last_tier2_summary": {"tier": 2, "processed_nodes": 10},
        "last_taxonomy_summary": {"tier": "taxonomy", "relations_added": {"SUBCLASS_OF": 1}},
        "last_tier3_summary": {"judged_pairs": 10, "alias_acceptance_rate": 0.01},
        "last_review": {**previous_review, "diagnosis": "taxonomy_debt"},
        "last_diagnosis": "taxonomy_debt",
        "iterations": [
            {
                "iteration": 1,
                "phase": "review_and_rerun",
                "review": previous_review,
                "raw_diagnosis": "taxonomy_debt",
                "diagnosis": "taxonomy_debt",
                "gate_pass": False,
                "consecutive_passes": 0,
                "plateau_streak": 0,
                "applied_params": {
                    "tier2": csi.asdict(csi.Tier2Params()),
                    "taxonomy": csi.asdict(csi.TaxonomyParams()),
                    "tier3": csi.asdict(csi.Tier3Params()),
                },
                "tier2_catalog_applied": csi._serialize_catalog(csi._default_tier2_catalog()),
                "tier2_skipped": True,
                "tier2_skip_reason": csi.TIER2_SKIP_REASON,
                "taxonomy_skipped": False,
                "taxonomy_summary": {
                    "tier": "taxonomy",
                    "relations_added": {"SUBCLASS_OF": 1},
                    "codex_queue_count": 1,
                },
                "tier3_skipped": True,
                "tier3_skip_reason": "taxonomy_debt",
            }
        ],
        "run_dir": str(run_dir),
    }
    (run_dir / "run_state.json").write_text(json.dumps(state), encoding="utf-8")

    resumed = csi.run_self_improving(
        csi.OrchestratorConfig(
            max_iterations=2,
            target_concept_ratio=0.05,
            target_duplicate_rate=0.015,
            target_concept_without_taxonomy_ratio=0.60,
            required_consecutive_passes=2,
            run_dir=str(run_dir),
            dry_run=True,
            resume=True,
        )
    )

    assert calls == ["review"]
    assert resumed["iterations"][0]["codex_taxonomy_tail_summary"]["application"]["blocked"] == 1


def test_resume_returns_terminal_state_without_reopening_completed_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "completed_run"
    run_dir.mkdir()
    state = {
        "status": "completed",
        "started_at_utc": "2026-03-06T13:56:33+00:00",
        "completed_at_utc": "2026-03-06T14:10:00+00:00",
        "stop_reason": csi.COMBINED_STOP_REASON,
        "config": csi.asdict(csi.OrchestratorConfig(run_dir=str(run_dir), dry_run=True, resume=True)),
        "next_iteration": 2,
        "consecutive_passes": 1,
        "plateau_streak": 0,
        "active_step": "idle",
        "last_error": None,
        "current_params": {
            "tier2": csi.asdict(csi.Tier2Params()),
            "taxonomy": csi.asdict(csi.TaxonomyParams()),
            "tier3": csi.asdict(csi.Tier3Params()),
        },
        "current_tier2_catalog": csi._serialize_catalog(csi._default_tier2_catalog()),
        "last_review": {"diagnosis": "balanced", "kpis": {"concept_only_ratio": 0.03, "duplicate_candidate_rate": 0.0, "concept_only_count": 10}, "taxonomy_kpis": {"concept_only_without_taxonomy_count": 5, "concept_only_degree_le_2_count": 3, "concept_only_degree_le_3_count": 4, "concept_only_with_similarity_or_alias_count": 0}},
        "last_diagnosis": "balanced",
        "consolidation_gate_pass": True,
        "semantic_gate_pass": True,
        "concept_only_without_taxonomy_ratio": 0.5,
        "iterations": [],
        "run_dir": str(run_dir),
    }
    state_path = run_dir / "run_state.json"
    original = json.dumps(state, indent=2)
    state_path.write_text(original, encoding="utf-8")

    resumed = csi.run_self_improving(
        csi.OrchestratorConfig(
            max_iterations=3,
            run_dir=str(run_dir),
            dry_run=True,
            resume=True,
        )
    )

    assert resumed["status"] == "completed"
    assert resumed["stop_reason"] == csi.COMBINED_STOP_REASON
    assert state_path.read_text(encoding="utf-8") == original


def test_live_state_reflects_next_tier2_params_before_rerun(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[str] = []

    def fake_tier1(*, params: csi.Tier2Params, dry_run: bool, iteration_dir: Path) -> None:
        calls.append("tier1")

    def fake_tier2(*, params: csi.Tier2Params, catalog: csi.Tier2LabelCatalog, dry_run: bool, iteration_dir: Path) -> dict:
        calls.append("tier2")
        if iteration_dir.name == "iteration_1":
            state = json.loads((iteration_dir.parent / "run_state.json").read_text(encoding="utf-8"))
            assert state["active_step"] == "iteration_1_tier2"
            assert state["current_params"]["tier2"]["batch_size"] == 80
            assert state["current_params"]["tier2"]["max_nodes"] == 1200
            assert state["current_params"]["taxonomy"]["max_nodes"] == csi.TaxonomyParams().max_nodes
        return {"tier": 2, "processed_nodes": params.max_nodes, "relabeled_without_taxonomy_support": 7}

    def fake_taxonomy(
        *,
        params: csi.TaxonomyParams,
        catalog: csi.Tier2LabelCatalog,
        dry_run: bool,
        iteration_dir: Path,
        prior_taxonomy_decisions_jsonl: str | None = None,
        prior_review_json: str | None = None,
    ) -> dict:
        calls.append("taxonomy")
        return {"tier": "taxonomy", "relations_added": {"SUBCLASS_OF": 1}, "suspicious_relabels": 0}

    def fake_tier3(*, params: csi.Tier3Params, dry_run: bool, iteration_dir: Path) -> dict:
        calls.append("tier3")
        return {"tier": 3, "judged_pairs": 30, "alias_acceptance_rate": 0.25}

    def fake_review(**kwargs) -> dict:
        calls.append("review")
        payload = _valid_review_payload()
        payload["proposed_tier2"] = {"batch_size": 80, "sleep_seconds": 0.5, "max_nodes": 1200}
        return payload

    monkeypatch.setattr(csi, "run_tier1", fake_tier1)
    monkeypatch.setattr(csi, "run_tier2", fake_tier2)
    monkeypatch.setattr(csi, "run_taxonomy", fake_taxonomy)
    monkeypatch.setattr(csi, "run_tier3", fake_tier3)
    monkeypatch.setattr(csi, "run_codex_review", fake_review)


    state = csi.run_self_improving(
        csi.OrchestratorConfig(
            max_iterations=2,
            target_concept_ratio=0.15,
            target_duplicate_rate=0.015,
            required_consecutive_passes=2,
            run_dir=str(tmp_path / "state_live_run"),
            codex_bin="codex",
            dry_run=True,
            resume=False,
        )
    )

    assert calls[:8] == ["tier1", "tier2", "taxonomy", "tier3", "review", "tier2", "taxonomy", "tier3"]
    assert state["status"] == "max_iterations_reached"


def test_failure_persists_failed_state(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fake_tier1(*, params: csi.Tier2Params, dry_run: bool, iteration_dir: Path) -> None:
        raise RuntimeError("tier1 exploded")

    monkeypatch.setattr(csi, "run_tier1", fake_tier1)

    run_dir = tmp_path / "run_failed"
    with pytest.raises(RuntimeError):
        csi.run_self_improving(csi.OrchestratorConfig(max_iterations=1, run_dir=str(run_dir), dry_run=True))

    state = json.loads((run_dir / "run_state.json").read_text(encoding="utf-8"))
    assert state["status"] == "failed"
    assert state["last_error"]["message"] == "tier1 exploded"
