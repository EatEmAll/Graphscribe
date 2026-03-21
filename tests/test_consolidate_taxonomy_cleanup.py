import json

import pytest

import consolidate_taxonomy_cleanup as taxonomy


class _DummyResponse:
    def __init__(self, text: str) -> None:
        self.text = text


class _DummyModels:
    def __init__(self, outputs):
        self._outputs = iter(outputs)

    def generate_content(self, **kwargs):
        output = next(self._outputs)
        if isinstance(output, Exception):
            raise output
        return _DummyResponse(output)


class _DummyClient:
    def __init__(self, outputs) -> None:
        self.models = _DummyModels(outputs)


class _Result:
    def __init__(self, payload):
        self._payload = payload

    def single(self):
        return self._payload


class _Session:
    def __init__(self, reverse_count=0, would_cycle=False, conflicting_count=0):
        self.reverse_count = reverse_count
        self.would_cycle = would_cycle
        self.conflicting_count = conflicting_count

    def run(self, query, **kwargs):
        if "reverse_count" in query:
            return _Result({"reverse_count": self.reverse_count})
        if "would_cycle" in query:
            return _Result({"would_cycle": self.would_cycle})
        if "conflicting_count" in query:
            return _Result({"conflicting_count": self.conflicting_count})
        raise AssertionError(f"Unexpected query: {query}")


def _catalog() -> dict:
    return taxonomy.load_label_catalog(None)


def _node() -> dict:
    return {
        "eid": "node-1",
        "name": "Leverage",
        "description": "Borrowed capital usage.",
        "labels": ["Concept"],
        "degree": 4,
        "outgoing_taxonomy_count": 0,
        "relation_types": ["MENTIONS", "USES"],
        "neighbor_ids": ["Liquidity"],
        "neighbor_labels": ["Trading Concept"],
        "taxonomy_targets": [],
    }


def _candidate() -> dict:
    return {
        "eid": "cand-1",
        "name": "Trading Concept",
        "description": "General trading idea.",
        "labels": ["Trading Concept"],
        "lexical_overlap": 0,
        "shared_neighbors": 1,
        "compatible_label_count": 1,
        "candidate_has_outgoing_taxonomy": 1,
        "candidate_is_taxonomy_target": 1,
        "embedding_similarity": 0.87,
        "candidate_degree": 5,
    }


def test_normalize_relation_maps_is_a_to_type_of() -> None:
    assert taxonomy._normalize_relation("IS_A") == "TYPE_OF"
    assert taxonomy._normalize_relation("subclass_of") == "SUBCLASS_OF"
    assert taxonomy._normalize_relation("bogus") == "NONE"

def test_fetch_parent_candidates_uses_precomputed_pool_shortlist() -> None:
    node = _node() | {"embedding": [1.0, 0.0]}
    candidate_pool = [
        _candidate()
        | {
            "neighbor_ids": ["Liquidity"],
            "embedding": [0.9, 0.1],
            "degree": 8,
            "outgoing_taxonomy_count": 1,
            "incoming_taxonomy_target_count": 1,
        },
        {
            "eid": "cand-2",
            "name": "Completely Different",
            "description": "No overlap",
            "labels": ["Asset"],
            "neighbor_ids": [],
            "neighbor_labels": [],
            "degree": 2,
            "outgoing_taxonomy_count": 0,
            "incoming_taxonomy_target_count": 0,
            "embedding": [0.0, 1.0],
        },
    ]

    results = taxonomy.fetch_parent_candidates(
        _Session(),
        node=node,
        candidate_pool=candidate_pool,
        compatible_labels=["Trading Concept"],
        candidate_limit=5,
        embedding_threshold=0.84,
    )

    assert [row["eid"] for row in results] == ["cand-1"]


def test_fetch_parent_candidates_prefers_taxonomy_bearing_candidate_on_tie() -> None:
    node = _node() | {"embedding": [1.0, 0.0]}
    candidate_pool = [
        {
            "eid": "cand-tax",
            "name": "Risk Concept",
            "description": "Broader risk idea.",
            "labels": ["Trading Concept"],
            "neighbor_ids": ["Liquidity"],
            "neighbor_labels": ["Trading Concept"],
            "degree": 7,
            "outgoing_taxonomy_count": 1,
            "incoming_taxonomy_target_count": 1,
            "embedding": [0.95, 0.05],
        },
        {
            "eid": "cand-peer",
            "name": "Market Idea",
            "description": "Another broad idea.",
            "labels": ["Trading Concept"],
            "neighbor_ids": ["Liquidity"],
            "neighbor_labels": ["Trading Concept"],
            "degree": 6,
            "outgoing_taxonomy_count": 0,
            "incoming_taxonomy_target_count": 0,
            "embedding": [0.95, 0.05],
        },
    ]

    results = taxonomy.fetch_parent_candidates(
        _Session(),
        node=node,
        candidate_pool=candidate_pool,
        compatible_labels=["Trading Concept"],
        candidate_limit=5,
        embedding_threshold=0.84,
    )

    assert [row["eid"] for row in results] == ["cand-tax", "cand-peer"]


def test_apply_label_replaces_old_primary_label() -> None:
    calls: list[str] = []

    class _Recorder:
        def run(self, query, **kwargs):
            calls.append(query)
            return None

    taxonomy.apply_label(
        _Recorder(),
        source_eid="node-1",
        old_labels=["Concept", "Method"],
        new_label="Trading Concept",
    )

    assert "SET n:`Trading Concept`" in calls[0]
    assert "REMOVE n:`Concept`" in calls[0]
    assert "REMOVE n:`Method`" in calls[0]


def test_normalize_decision_clears_relation_fields_for_keep_label() -> None:
    normalized = taxonomy._normalize_decision(
        {
            "action": "keep_label",
            "label": "Concept",
            "relation": "SUBCLASS_OF",
            "target_eid": "cand-1",
            "confidence": 0.93,
            "reason": "keep it",
        },
        node=_node(),
        label_catalog=_catalog(),
    )
    assert normalized["action"] == "keep_label"
    assert normalized["relation"] == "NONE"
    assert normalized["target_eid"] is None
    assert normalized["label"] == "Concept"


def test_seed_eids_from_tier2_picks_relabels_low_confidence_and_unresolved() -> None:
    rows = [
        {"eid": "1", "old_label": "Concept", "new_label": "Trading Concept", "confidence": 0.91},
        {"eid": "2", "old_label": "Concept", "new_label": "Concept", "confidence": 0.42},
        {"eid": "3", "unresolved": True},
    ]
    assert taxonomy._seed_eids_from_tier2(rows) == ["1", "2", "3"]


def test_seed_eids_from_prior_taxonomy_picks_carry_forward_rows() -> None:
    rows = [
        {"eid": "1", "status": "unresolved"},
        {"eid": "2", "suspicious": True},
        {"eid": "3", "skipped_reason": "weak evidence"},
        {"eid": "4", "applied": True, "outgoing_taxonomy_count_after": 0},
        {"eid": "5", "applied": True, "outgoing_taxonomy_count_after": 1},
    ]

    assert taxonomy._seed_eids_from_prior_taxonomy(rows) == ["1", "2", "3", "4"]


def test_review_focus_name_set_reads_prior_review_examples() -> None:
    payload = {
        "focus_examples": {
            "high_degree_concept_only": ["Free Will"],
            "low_degree_concept_only": ["Academic Piece"],
        }
    }

    assert taxonomy._review_focus_name_set(payload) == {"free will", "academic piece"}


def test_prioritize_taxonomy_nodes_uses_residual_priority_buckets() -> None:
    nodes = [
        _node() | {"eid": "concept-no-tax", "name": "Problem", "degree": 9, "outgoing_taxonomy_count": 0},
        _node()
        | {
            "eid": "carry",
            "name": "Control",
            "labels": ["Condition"],
            "degree": 8,
            "outgoing_taxonomy_count": 0,
        },
        _node()
        | {
            "eid": "tier2",
            "name": "Liquidity",
            "labels": ["Trading Concept"],
            "degree": 7,
            "outgoing_taxonomy_count": 0,
        },
        _node() | {"eid": "focus", "name": "Idea", "degree": 6, "outgoing_taxonomy_count": 1},
        _node() | {"eid": "plain", "name": "Talent", "degree": 5, "outgoing_taxonomy_count": 1},
    ]

    ordered, residual_seed_count, carry_forward_seed_count, review_focus_seed_count = taxonomy._prioritize_taxonomy_nodes(
        nodes,
        tier2_seed_eids={"tier2"},
        carry_forward_seed_eids={"carry"},
        review_focus_names={"idea"},
        max_nodes=10,
    )

    assert [node["eid"] for node in ordered] == ["concept-no-tax", "carry", "tier2", "focus"]
    assert residual_seed_count == 3
    assert carry_forward_seed_count == 1
    assert review_focus_seed_count == 1


def test_classify_taxonomy_action_uses_second_stage_for_low_confidence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(taxonomy, "LOW_CONFIDENCE_THRESHOLD", 0.7)
    client = _DummyClient(
        [
            json.dumps(
                {
                    "action": "add_relation",
                    "label": "Concept",
                    "relation": "SUBCLASS_OF",
                    "target_eid": "cand-1",
                    "confidence": 0.4,
                    "reason": "weak fit",
                }
            ),
            json.dumps(
                {
                    "action": "add_relation",
                    "label": "Concept",
                    "relation": "SUBCLASS_OF",
                    "target_eid": "cand-1",
                    "confidence": 0.92,
                    "reason": "strong parent fit",
                }
            ),
        ]
    )

    result = taxonomy.classify_taxonomy_action(
        client,
        label_catalog=_catalog(),
        node=_node(),
        candidates=[_candidate()],
    )

    assert result["action"] == "add_relation"
    assert result["relation"] == "SUBCLASS_OF"
    assert result["target_eid"] == "cand-1"
    assert result["used_second_stage"] is True


def test_classify_taxonomy_action_keeps_high_confidence_keep_label_without_second_stage(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _DummyClient(
        [
            json.dumps(
                {
                    "action": "keep_label",
                    "label": "Concept",
                    "relation": "NONE",
                    "target_eid": None,
                    "confidence": 1.0,
                    "reason": "broad concept",
                }
            )
        ]
    )

    result = taxonomy.classify_taxonomy_action(
        client,
        label_catalog=_catalog(),
        node=_node(),
        candidates=[_candidate()],
    )

    assert result["action"] == "keep_label"
    assert result["attempted_second_stage"] is False
    assert result["used_second_stage"] is False


def test_classify_taxonomy_action_falls_back_to_safe_primary_when_second_stage_is_invalid() -> None:
    client = _DummyClient(
        [
            json.dumps(
                {
                    "action": "keep_label",
                    "label": "Concept",
                    "relation": "NONE",
                    "target_eid": None,
                    "confidence": 0.88,
                    "reason": "broad concept",
                }
            ),
            "",
        ]
    )

    result = taxonomy.classify_taxonomy_action(
        client,
        label_catalog=_catalog(),
        node=_node(),
        candidates=[_candidate()],
    )

    assert result["action"] == "keep_label"
    assert result["status"] == "classified"
    assert result["used_second_stage"] is False
    assert result["attempted_second_stage"] is False


def test_classify_once_retries_and_returns_unresolved_on_transport_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(taxonomy, "MODEL_MAX_ATTEMPTS", 2)
    monkeypatch.setattr(taxonomy, "MODEL_RETRY_SLEEP_SECONDS", 0.0)
    client = _DummyClient([RuntimeError("socket reset"), RuntimeError("socket reset")])

    result = taxonomy._classify_once(
        client,
        model_name=taxonomy.MODEL_NAME,
        label_catalog=_catalog(),
        node=_node(),
        candidates=[_candidate()],
    )

    assert result["status"] == "unresolved"
    assert "Model request failed after retries" in result["reason"]


def test_can_apply_relation_rejects_cycle_and_conflict() -> None:
    allowed, reason = taxonomy.can_apply_relation(
        _Session(would_cycle=True),
        source_eid="a",
        relation="SUBCLASS_OF",
        target_eid="b",
    )
    assert allowed is False
    assert "cycle" in reason.lower()

    allowed, reason = taxonomy.can_apply_relation(
        _Session(conflicting_count=1),
        source_eid="a",
        relation="TYPE_OF",
        target_eid="b",
    )
    assert allowed is False
    assert "existing type_of target".lower() in reason.lower()


def test_relation_direction_guard_rejects_narrower_target() -> None:
    allowed, reason = taxonomy._relation_direction_is_plausible(
        node={"name": "Thinking"},
        candidate={"name": "Mathematical Thinking"},
        relation="SUBCLASS_OF",
    )
    assert allowed is False
    assert "narrower" in reason.lower() or "specialize" in reason.lower()


def test_apply_relation_decision_normalizes_and_validates_once() -> None:
    result = taxonomy._apply_relation_decision(
        _Session(),
        node=_node(),
        relation="IS_A",
        target_eid="cand-1",
        confidence=0.91,
        candidate_by_eid={"cand-1": _candidate()},
        dry_run=True,
    )

    assert result["applied"] is True
    assert result["relation"] == "TYPE_OF"
    assert result["target_eid"] == "cand-1"
    assert result["skipped_reason"] == ""


def test_apply_relation_decision_returns_same_metadata_in_dry_and_live_modes(monkeypatch: pytest.MonkeyPatch) -> None:
    applied_calls: list[tuple[str, str, str]] = []

    monkeypatch.setattr(
        taxonomy,
        "add_relation",
        lambda session, source_eid, relation, target_eid: applied_calls.append((source_eid, relation, target_eid)),
    )

    kwargs = {
        "node": _node(),
        "relation": "SUBCLASS_OF",
        "target_eid": "cand-1",
        "confidence": 0.91,
        "candidate_by_eid": {"cand-1": _candidate()},
    }
    dry_result = taxonomy._apply_relation_decision(_Session(), dry_run=True, **kwargs)
    live_result = taxonomy._apply_relation_decision(_Session(), dry_run=False, **kwargs)

    assert dry_result["relation"] == live_result["relation"] == "SUBCLASS_OF"
    assert dry_result["target_eid"] == live_result["target_eid"] == "cand-1"
    assert dry_result["skipped_reason"] == live_result["skipped_reason"] == ""
    assert dry_result["applied"] is True
    assert live_result["applied"] is True
    assert applied_calls == [("node-1", "SUBCLASS_OF", "cand-1")]


def test_run_skips_relation_when_target_is_not_in_candidate_list(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    class _Driver:
        def session(self, database=None):
            class _Ctx:
                def __enter__(self_inner):
                    return _Session()

                def __exit__(self_inner, exc_type, exc, tb):
                    return False

            return _Ctx()

        def close(self):
            return None

    monkeypatch.setattr(taxonomy, "GOOGLE_API_KEY", "test-key")
    monkeypatch.setattr(taxonomy.genai, "Client", lambda api_key: object())
    monkeypatch.setattr(taxonomy.GraphDatabase, "driver", lambda *args, **kwargs: _Driver())
    monkeypatch.setattr(taxonomy, "fetch_taxonomy_candidates", lambda session, seed_eids, max_nodes: [_node()])
    monkeypatch.setattr(taxonomy, "fetch_candidate_pool", lambda session: [_candidate()])
    monkeypatch.setattr(
        taxonomy,
        "fetch_parent_candidates",
        lambda session, node, candidate_pool, compatible_labels, candidate_limit, embedding_threshold: [_candidate()],
    )
    monkeypatch.setattr(
        taxonomy,
        "classify_taxonomy_action",
        lambda client, label_catalog, node, candidates: {
            "status": "classified",
            "action": "add_relation",
            "label": None,
            "relation": "SUBCLASS_OF",
            "target_eid": "missing-candidate",
            "confidence": 0.99,
            "reason": "bad target",
            "model_name": taxonomy.MODEL_NAME,
            "stage": "primary",
            "used_second_stage": False,
            "attempted_second_stage": False,
        },
    )

    summary = taxonomy.run(
        dry_run=True,
        max_nodes=10,
        candidate_limit=5,
        embedding_threshold=0.8,
        neo4j_uri="bolt://test",
        neo4j_user="neo4j",
        neo4j_password="pw",
        neo4j_database="neo4j",
        labels_json=None,
        tier2_decisions_jsonl=None,
        prior_taxonomy_decisions_jsonl=None,
        prior_review_json=None,
        summary_json=str(tmp_path / "taxonomy_summary.json"),
        decisions_jsonl=str(tmp_path / "taxonomy_decisions.jsonl"),
    )

    assert summary["relations_added"]["SUBCLASS_OF"] == 0
    decision = json.loads((tmp_path / "taxonomy_decisions.jsonl").read_text(encoding="utf-8").strip())
    assert decision["applied"] is False
    assert decision["skipped_reason"] == "Target candidate not in provided list"


def test_run_marks_high_confidence_keep_label_concept_as_suspicious(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    class _Driver:
        def session(self, database=None):
            class _Ctx:
                def __enter__(self_inner):
                    return _Session()

                def __exit__(self_inner, exc_type, exc, tb):
                    return False

            return _Ctx()

        def close(self):
            return None

    monkeypatch.setattr(taxonomy, "GOOGLE_API_KEY", "test-key")
    monkeypatch.setattr(taxonomy.genai, "Client", lambda api_key: object())
    monkeypatch.setattr(taxonomy.GraphDatabase, "driver", lambda *args, **kwargs: _Driver())
    monkeypatch.setattr(taxonomy, "fetch_taxonomy_candidates", lambda session, seed_eids, max_nodes: [_node()])
    monkeypatch.setattr(taxonomy, "fetch_candidate_pool", lambda session: [_candidate()])
    monkeypatch.setattr(
        taxonomy,
        "fetch_parent_candidates",
        lambda session, node, candidate_pool, compatible_labels, candidate_limit, embedding_threshold: [_candidate()],
    )
    monkeypatch.setattr(
        taxonomy,
        "classify_taxonomy_action",
        lambda client, label_catalog, node, candidates: {
            "status": "classified",
            "action": "keep_label",
            "label": "Concept",
            "relation": "NONE",
            "target_eid": None,
            "confidence": 1.0,
            "reason": "broad concept",
            "model_name": taxonomy.MODEL_NAME,
            "stage": "primary",
            "used_second_stage": False,
            "attempted_second_stage": False,
        },
    )

    summary = taxonomy.run(
        dry_run=True,
        max_nodes=10,
        candidate_limit=5,
        embedding_threshold=0.8,
        neo4j_uri="bolt://test",
        neo4j_user="neo4j",
        neo4j_password="pw",
        neo4j_database="neo4j",
        labels_json=None,
        tier2_decisions_jsonl=None,
        prior_taxonomy_decisions_jsonl=None,
        prior_review_json=None,
        summary_json=str(tmp_path / "taxonomy_summary.json"),
        decisions_jsonl=str(tmp_path / "taxonomy_decisions.jsonl"),
    )

    assert summary["suspicious_keep_label_concepts"] == 1
    decision = json.loads((tmp_path / "taxonomy_decisions.jsonl").read_text(encoding="utf-8").strip())
    assert decision["suspicious"] is True
    assert "High-confidence keep_label" in decision["suspicious_reason"]


def test_run_attempts_follow_up_relation_after_high_confidence_relabel(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    class _Driver:
        def session(self, database=None):
            class _Ctx:
                def __enter__(self_inner):
                    return _Session()

                def __exit__(self_inner, exc_type, exc, tb):
                    return False

            return _Ctx()

        def close(self):
            return None

    decisions = iter(
        [
            {
                "status": "classified",
                "action": "relabel",
                "label": "Trading Concept",
                "relation": "NONE",
                "target_eid": None,
                "confidence": 0.96,
                "reason": "Specific trading concept.",
                "model_name": taxonomy.MODEL_NAME,
                "stage": "primary",
                "used_second_stage": False,
                "attempted_second_stage": False,
            },
            {
                "status": "classified",
                "action": "add_relation",
                "label": "Trading Concept",
                "relation": "SUBCLASS_OF",
                "target_eid": "cand-1",
                "confidence": 0.92,
                "reason": "Broader parent exists.",
                "model_name": taxonomy.MODEL_NAME,
                "stage": "primary",
                "used_second_stage": False,
                "attempted_second_stage": False,
            },
        ]
    )

    monkeypatch.setattr(taxonomy, "GOOGLE_API_KEY", "test-key")
    monkeypatch.setattr(taxonomy.genai, "Client", lambda api_key: object())
    monkeypatch.setattr(taxonomy.GraphDatabase, "driver", lambda *args, **kwargs: _Driver())
    monkeypatch.setattr(taxonomy, "fetch_taxonomy_candidates", lambda session, seed_eids, max_nodes: [_node()])
    monkeypatch.setattr(
        taxonomy,
        "fetch_candidate_pool",
        lambda session: [
            {
                "eid": "cand-1",
                "name": "Trading Concept",
                "description": "General trading idea.",
                "labels": ["Trading Concept"],
                "neighbor_ids": ["Liquidity"],
                "neighbor_labels": ["Trading Concept"],
                "degree": 7,
                "outgoing_taxonomy_count": 1,
                "incoming_taxonomy_target_count": 1,
                "embedding": [0.9, 0.1],
            }
        ],
    )
    monkeypatch.setattr(
        taxonomy,
        "fetch_parent_candidates",
        lambda session, node, candidate_pool, compatible_labels, candidate_limit, embedding_threshold: [_candidate()],
    )
    monkeypatch.setattr(
        taxonomy,
        "classify_taxonomy_action",
        lambda client, label_catalog, node, candidates, relation_only=False: next(decisions),
    )

    summary = taxonomy.run(
        dry_run=True,
        max_nodes=10,
        candidate_limit=5,
        embedding_threshold=0.8,
        neo4j_uri="bolt://test",
        neo4j_user="neo4j",
        neo4j_password="pw",
        neo4j_database="neo4j",
        labels_json=None,
        tier2_decisions_jsonl=None,
        prior_taxonomy_decisions_jsonl=None,
        prior_review_json=None,
        summary_json=str(tmp_path / "taxonomy_summary.json"),
        decisions_jsonl=str(tmp_path / "taxonomy_decisions.jsonl"),
    )

    decision = json.loads((tmp_path / "taxonomy_decisions.jsonl").read_text(encoding="utf-8").strip())
    assert summary["relabels_applied"] == 1
    assert summary["post_relabel_relation_attempts"] == 1
    assert summary["post_relabel_relations_added"] == 1
    assert summary["relabeled_without_taxonomy_support"] == 0
    assert decision["follow_up_action"] == "add_relation"
    assert decision["follow_up_relation"] == "SUBCLASS_OF"
    assert decision["follow_up_applied"] is True
    assert decision["outgoing_taxonomy_count_after"] == 1


def test_run_skips_follow_up_when_relabel_is_below_apply_threshold(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    class _Driver:
        def session(self, database=None):
            class _Ctx:
                def __enter__(self_inner):
                    return _Session()

                def __exit__(self_inner, exc_type, exc, tb):
                    return False

            return _Ctx()

        def close(self):
            return None

    monkeypatch.setattr(taxonomy, "GOOGLE_API_KEY", "test-key")
    monkeypatch.setattr(taxonomy.genai, "Client", lambda api_key: object())
    monkeypatch.setattr(taxonomy.GraphDatabase, "driver", lambda *args, **kwargs: _Driver())
    monkeypatch.setattr(taxonomy, "fetch_taxonomy_candidates", lambda session, seed_eids, max_nodes: [_node()])
    monkeypatch.setattr(taxonomy, "fetch_candidate_pool", lambda session: [_candidate()])
    monkeypatch.setattr(
        taxonomy,
        "fetch_parent_candidates",
        lambda session, node, candidate_pool, compatible_labels, candidate_limit, embedding_threshold: [_candidate()],
    )
    monkeypatch.setattr(
        taxonomy,
        "classify_taxonomy_action",
        lambda client, label_catalog, node, candidates, relation_only=False: {
            "status": "classified",
            "action": "relabel",
            "label": "Trading Concept",
            "relation": "NONE",
            "target_eid": None,
            "confidence": 0.50,
            "reason": "Weak fit.",
            "model_name": taxonomy.MODEL_NAME,
            "stage": "primary",
            "used_second_stage": False,
            "attempted_second_stage": False,
        },
    )

    summary = taxonomy.run(
        dry_run=True,
        max_nodes=10,
        candidate_limit=5,
        embedding_threshold=0.8,
        neo4j_uri="bolt://test",
        neo4j_user="neo4j",
        neo4j_password="pw",
        neo4j_database="neo4j",
        labels_json=None,
        tier2_decisions_jsonl=None,
        prior_taxonomy_decisions_jsonl=None,
        prior_review_json=None,
        summary_json=str(tmp_path / "taxonomy_summary.json"),
        decisions_jsonl=str(tmp_path / "taxonomy_decisions.jsonl"),
    )

    decision = json.loads((tmp_path / "taxonomy_decisions.jsonl").read_text(encoding="utf-8").strip())
    assert summary["post_relabel_relation_attempts"] == 0
    assert summary["post_relabel_relations_added"] == 0
    assert decision["follow_up_action"] is None
    assert decision["follow_up_applied"] is False
