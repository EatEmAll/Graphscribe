import json

import numpy as np
import pytest

import notebooklm_graph_pipe.consolidation.tier3_semantic as t3


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


def _entity(name: str, labels: list[str], taxonomy_neighbors: list[str] | None = None) -> dict:
    return {
        "eid": f"eid-{name}",
        "name": name,
        "description": f"{name} description",
        "labels": labels,
        "degree": 3,
        "relation_types": ["MENTIONS"],
        "neighbor_labels": labels,
        "taxonomy_neighbor_eids": taxonomy_neighbors or [],
    }


def test_labels_are_clearly_incompatible() -> None:
    assert t3._labels_are_clearly_incompatible(["Method"], ["Asset"]) is True
    assert t3._labels_are_clearly_incompatible(["Metric"], ["Financial Metric"]) is False
    assert t3._labels_are_clearly_incompatible(["Concept"], ["Asset"]) is False

def test_judge_pair_uses_second_stage_for_low_confidence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(t3, "LOW_CONFIDENCE_THRESHOLD", 0.72)
    client = _DummyClient(
        [
            json.dumps({"verdict": "ALIAS", "confidence": 0.21, "reason": "weak"}),
            json.dumps({"verdict": "ALIAS", "confidence": 0.93, "reason": "strong"}),
        ]
    )

    result = t3.judge_pair(client, _entity("P&L", ["Financial Metric"]), _entity("Profit And Loss", ["Financial Metric"]))

    assert result["verdict"] == "ALIAS"
    assert result["used_second_stage"] is True
    assert result["model_name"] == t3.SECOND_STAGE_JUDGE_MODEL


def test_judge_pair_retries_transient_primary_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(t3, "MODEL_MAX_ATTEMPTS", 2)
    monkeypatch.setattr(t3, "MODEL_RETRY_SLEEP_SECONDS", 0.0)
    client = _DummyClient(
        [
            RuntimeError("socket reset"),
            json.dumps({"verdict": "DIFFERENT", "confidence": 0.91, "reason": "stable after retry"}),
        ]
    )

    result = t3.judge_pair(client, _entity("Leverage", ["Trading Concept"]), _entity("Liquidity", ["Trading Concept"]))

    assert result["status"] == "classified"
    assert result["verdict"] == "DIFFERENT"


def test_judge_pair_falls_back_to_primary_when_second_stage_is_invalid() -> None:
    client = _DummyClient(
        [
            json.dumps({"verdict": "DIFFERENT", "confidence": 0.2, "reason": "weak primary"}),
            "",
        ]
    )

    result = t3.judge_pair(client, _entity("Leverage", ["Trading Concept"]), _entity("Liquidity", ["Trading Concept"]))

    assert result["verdict"] == "DIFFERENT"
    assert result["used_second_stage"] is False
    assert result["attempted_second_stage"] is True


def test_judge_pair_reuses_persistent_cache(tmp_path) -> None:
    cache_path = tmp_path / "tier3_judge_cache.json"
    cache = t3.JsonDiskCache(cache_path)
    first_client = _DummyClient(
        [
            json.dumps({"verdict": "DIFFERENT", "confidence": 0.91, "reason": "cached"}),
        ]
    )

    first = t3.judge_pair(
        first_client,
        _entity("Leverage", ["Trading Concept"]),
        _entity("Liquidity", ["Trading Concept"]),
        cache=cache,
    )
    cache.save()

    second_client = _DummyClient([])
    second = t3.judge_pair(
        second_client,
        _entity("Leverage", ["Trading Concept"]),
        _entity("Liquidity", ["Trading Concept"]),
        cache=t3.JsonDiskCache(cache_path),
    )

    assert first["verdict"] == "DIFFERENT"
    assert second["verdict"] == "DIFFERENT"
    assert second["confidence"] == pytest.approx(0.91)


def test_should_skip_pair_skips_existing_taxonomy_or_incompatible_labels() -> None:
    left = _entity("Leverage", ["Trading Concept"], taxonomy_neighbors=["eid-Liquidity"])
    right = _entity("Liquidity", ["Market Feature"])
    right["eid"] = "eid-Liquidity"
    assert t3._should_skip_pair(left, right) is True

    assert t3._should_skip_pair(_entity("Stop Loss", ["Method"]), _entity("AAPL", ["Asset"])) is True


def test_run_merges_aliases_and_never_adds_relations(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    class _Driver:
        def session(self, database=None):
            class _Ctx:
                def __enter__(self_inner):
                    return object()

                def __exit__(self_inner, exc_type, exc, tb):
                    return False

            return _Ctx()

        def close(self):
            return None

    merges: list[tuple[str, str, str]] = []

    monkeypatch.setattr(t3, "GOOGLE_API_KEY", "test-key")
    monkeypatch.setattr(t3.genai, "Client", lambda api_key: object())
    monkeypatch.setattr(t3.GraphDatabase, "driver", lambda *args, **kwargs: _Driver())
    monkeypatch.setattr(
        t3,
        "fetch_entities",
        lambda session: [
            _entity("P&L", ["Financial Metric"]),
            _entity("Profit And Loss", ["Financial Metric"]),
            _entity("AAPL", ["Asset"]),
        ],
    )
    monkeypatch.setattr(
        t3,
        "embed_batch",
        lambda client, texts, cache_file: [
            np.array([1.0, 0.0]),
            np.array([0.99, 0.01]),
            np.array([0.0, 1.0]),
        ],
    )
    monkeypatch.setattr(
        t3,
        "judge_pair",
        lambda client, entity_a, entity_b, cache=None: {
            "status": "classified",
            "verdict": "ALIAS" if {"P&L", "Profit And Loss"} == {entity_a["name"], entity_b["name"]} else "DIFFERENT",
            "confidence": 0.95,
            "reason": "test",
            "model_name": t3.PRIMARY_JUDGE_MODEL,
            "stage": "primary",
            "used_second_stage": False,
            "attempted_second_stage": False,
        },
    )
    monkeypatch.setattr(t3, "merge_pair", lambda session, eid_a, eid_b, canonical_name: merges.append((eid_a, eid_b, canonical_name)))

    summary = t3.run(
        dry_run=False,
        threshold=0.85,
        max_candidates=10,
        max_merges=5,
        sleep_seconds=0.0,
        cache_file=str(tmp_path / "embeddings_cache.pkl"),
        neo4j_uri="bolt://test",
        neo4j_user="neo4j",
        neo4j_password="pw",
        neo4j_database="neo4j",
        summary_json=str(tmp_path / "tier3_summary.json"),
    )

    assert summary["confirmed_merges"] == 1
    assert summary["relations_added"] == 0
    assert summary["judge_counts"]["ALIAS"] == 1
    assert summary["second_stage_attempts"] == 0
    assert merges and merges[0][2] == "P&L"
