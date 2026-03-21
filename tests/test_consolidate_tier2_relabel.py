import json

import pytest

import consolidate_tier2_relabel as t2


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


def _catalog() -> dict:
    return t2.load_label_catalog(None)


def _node() -> dict:
    return {
        "eid": "1",
        "name": "Leverage",
        "description": "A trading idea about using borrowed capital.",
        "labels": ["Concept"],
        "degree": 4,
        "relation_types": ["MENTIONS", "USES"],
        "neighbor_ids": ["Liquidity", "Portfolio"],
        "neighbor_labels": ["Trading Concept", "Strategy"],
        "outgoing_taxonomy_count": 0,
    }


def test_normalize_label_uses_aliases() -> None:
    catalog = _catalog()
    assert t2.normalize_label("financial", catalog) == "Financial Metric"
    assert t2.normalize_label("trading", catalog) == "Trading Concept"
    assert t2.normalize_label("state", catalog) == "Condition"


def test_build_prompt_includes_neighborhood_context() -> None:
    prompt = t2._build_prompt(_node())
    assert 'Entity: "Leverage"' in prompt
    assert "Current labels: Concept" in prompt
    assert "Neighbor labels: Trading Concept, Strategy" in prompt
    assert "Relation types: MENTIONS, USES" in prompt
    assert "Neighbor ids: Liquidity, Portfolio" in prompt


def test_classify_entity_uses_second_stage_for_low_confidence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(t2, "LOW_CONFIDENCE_THRESHOLD", 0.65)
    client = _DummyClient(
        [
            json.dumps({"label": "financial", "confidence": 0.2, "reason": "weak fit"}),
            json.dumps({"label": "Financial Metric", "confidence": 0.91, "reason": "measured quantity"}),
        ]
    )

    result = t2.classify_entity(client, _node(), _catalog())

    assert result["status"] == "classified"
    assert result["label"] == "Financial Metric"
    assert result["model_name"] == t2.SECOND_STAGE_MODEL_NAME
    assert result["confidence"] == pytest.approx(0.91)
    assert result["used_second_stage"] is True


def test_classify_entity_retries_transient_failures_and_returns_unresolved(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(t2, "MAX_RETRIES", 2)
    monkeypatch.setattr(t2, "INITIAL_RETRY_DELAY_SECONDS", 0.0)
    monkeypatch.setattr(t2.time, "sleep", lambda *_args, **_kwargs: None)
    client = _DummyClient(
        [
            RuntimeError("getaddrinfo failed"),
            RuntimeError("getaddrinfo failed"),
        ]
    )

    result = t2.classify_entity(client, _node(), _catalog())

    assert result["status"] == "unresolved"
    assert result["label"] is None
    assert "getaddrinfo failed" in result["reason"]


def test_classify_entity_reuses_persistent_cache(tmp_path) -> None:
    cache_path = tmp_path / "tier2_cache.json"
    cache = t2.JsonDiskCache(cache_path)
    first_client = _DummyClient(
        [
            json.dumps({"label": "Trading Concept", "confidence": 0.88, "reason": "cached"}),
        ]
    )

    first = t2.classify_entity(first_client, _node(), _catalog(), cache=cache)
    cache.save()

    second_client = _DummyClient([])
    second = t2.classify_entity(second_client, _node(), _catalog(), cache=t2.JsonDiskCache(cache_path))

    assert first["label"] == "Trading Concept"
    assert second["label"] == "Trading Concept"
    assert second["confidence"] == pytest.approx(0.88)


def test_append_decision_jsonl_writes_expected_fields(tmp_path) -> None:
    path = tmp_path / "tier2_decisions.jsonl"
    t2._append_decision_jsonl(
        path,
        {
            "eid": "1",
            "name": "Leverage",
            "old_label": "Concept",
            "new_label": "Trading Concept",
            "status": "classified",
            "confidence": 0.88,
            "reason": "Market idea",
            "model_name": t2.MODEL_NAME,
            "stage": "primary",
            "used_second_stage": False,
            "attempted_second_stage": False,
            "unresolved": False,
            "labels": ["Concept"],
            "degree": 4,
            "relation_types": ["MENTIONS"],
            "neighbor_ids": ["Liquidity"],
            "neighbor_labels": ["Trading Concept"],
            "outgoing_taxonomy_count": 0,
        },
    )
    payload = json.loads(path.read_text(encoding="utf-8").strip())
    assert payload["new_label"] == "Trading Concept"
    assert payload["used_second_stage"] is False
