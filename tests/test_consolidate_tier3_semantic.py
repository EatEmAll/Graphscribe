import json

import numpy as np
import pytest

import notebooklm_graph_pipe.consolidation.tier3_semantic as t3


class _DummyResponses:
    """Stands in for an OpenRouter Responses client: one queued output per request."""

    def __init__(self, outputs):
        self._outputs = iter(outputs)

    def create(self, **kwargs):
        output = next(self._outputs)
        if isinstance(output, Exception):
            raise output
        return type("Response", (), {"output_text": output})()


class _DummyClient:
    def __init__(self, outputs) -> None:
        self.responses = _DummyResponses(outputs)


PROMPT_ROLE = t3.PromptRoleConfig(client="openrouter_json", model="minimax/minimax-m3")


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

def test_judge_pair_merges_only_at_or_above_the_alias_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(t3, "ALIAS_THRESHOLD", 0.8)
    client = _DummyClient(
        [
            json.dumps({"verdict": "ALIAS", "confidence": 0.79, "reason": "weak"}),
            json.dumps({"verdict": "ALIAS", "confidence": 0.8, "reason": "strong"}),
            json.dumps({"verdict": "DIFFERENT", "confidence": 0.1, "reason": "unsure"}),
        ]
    )
    pairs = [("P&L", "Profit And Loss"), ("PnL", "Profit And Loss"), ("P/L", "Profit And Loss")]

    results = [t3.judge_pair(client, _entity(a, ["Financial Metric"]), _entity(b, ["Financial Metric"]), primary_role_config=PROMPT_ROLE)
               for a, b in pairs]

    assert [r["verdict"] for r in results] == ["DIFFERENT", "ALIAS", "ALIAS"]
    assert results[2]["p_alias"] == pytest.approx(0.9)
    assert all("used_second_stage" not in r for r in results)


def test_judge_pair_retries_transient_primary_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(t3, "MODEL_MAX_ATTEMPTS", 2)
    monkeypatch.setattr(t3, "MODEL_RETRY_SLEEP_SECONDS", 0.0)
    client = _DummyClient(
        [
            RuntimeError("socket reset"),
            json.dumps({"verdict": "DIFFERENT", "confidence": 0.91, "reason": "stable after retry"}),
        ]
    )

    result = t3.judge_pair(client, _entity("Leverage", ["Trading Concept"]), _entity("Liquidity", ["Trading Concept"]), primary_role_config=PROMPT_ROLE)

    assert result["status"] == "classified"
    assert result["verdict"] == "DIFFERENT"


def test_judge_pair_keeps_an_unresolved_pair_separate() -> None:
    client = _DummyClient([""] * 3)

    result = t3.judge_pair(client, _entity("Leverage", ["Trading Concept"]), _entity("Liquidity", ["Trading Concept"]), primary_role_config=PROMPT_ROLE)

    assert (result["status"], result["verdict"], result["p_alias"]) == ("unresolved", "DIFFERENT", 0.0)


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
        primary_role_config=PROMPT_ROLE,
        cache=cache,
    )
    cache.save()

    second_client = _DummyClient([])
    second = t3.judge_pair(
        second_client,
        _entity("Leverage", ["Trading Concept"]),
        _entity("Liquidity", ["Trading Concept"]),
        primary_role_config=PROMPT_ROLE,
        cache=t3.JsonDiskCache(cache_path),
    )

    assert first["verdict"] == "DIFFERENT"
    assert second["verdict"] == "DIFFERENT"
    assert second["p_alias"] == pytest.approx(0.09)


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
            _entity("AR(1) Model", ["Model"]),
            _entity("AR(2) Model", ["Model"]),
        ],
    )
    monkeypatch.setattr(
        t3,
        "embed_batch",
        lambda client, texts, cache_file: [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.99, 0.01, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            np.array([0.0, 0.01, 0.99]),
        ],
    )
    monkeypatch.setattr(
        t3,
        "judge_pair",
        lambda client, entity_a, entity_b, cache=None: {
            "status": "classified",
            "verdict": "ALIAS" if {entity_a["name"], entity_b["name"]} in ({"P&L", "Profit And Loss"}, {"AR(1) Model", "AR(2) Model"}) else "DIFFERENT",
            "p_alias": 0.95,
            "reason": "test",
            "model_name": t3.PRIMARY_JUDGE_MODEL,
            "stage": "primary",
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
    assert summary["judge_counts"]["ALIAS"] == 2
    assert summary["digit_guard_blocked"] == [["AR(1) Model", "AR(2) Model"]]
    assert merges == [("eid-P&L", "eid-Profit And Loss", "P&L")]


class _EmbedRecorder:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.models = self

    def embed_content(self, **kwargs):
        self.calls.append(kwargs)
        return type("Result", (), {"embeddings": [type("Embedding", (), {"values": [0.6, 0.8]})()]})()


def test_embed_text_puts_the_task_in_the_prompt_for_gemini_embedding_2() -> None:
    client = _EmbedRecorder()
    role = t3.EmbeddingRoleConfig(client="genai", model="gemini-embedding-2")

    vector = t3._embed_text(client, embedding_role_config=role, text="Sharpe Ratio: risk-adjusted return")

    assert vector.tolist() == [0.6, 0.8]
    assert client.calls == [
        {"model": "gemini-embedding-2", "contents": "task: sentence similarity | query: Sharpe Ratio: risk-adjusted return"}
    ]


def test_embed_text_keeps_task_type_for_gemini_embedding_001() -> None:
    client = _EmbedRecorder()
    role = t3.EmbeddingRoleConfig(client="genai", model="gemini-embedding-001")

    t3._embed_text(client, embedding_role_config=role, text="Sharpe Ratio")

    assert client.calls[0]["contents"] == "Sharpe Ratio"
    assert client.calls[0]["config"].task_type == "SEMANTIC_SIMILARITY"


def test_run_defaults_route_gemini_embeddings_and_the_openrouter_primary_judge(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setattr(t3, "build_single_prompt_clients", lambda *names: {name: object() for name in names})
    monkeypatch.setattr(t3, "fetch_entities", lambda session, scope_revision_ids=None: [])

    summary = t3.run(dry_run=True, threshold=0.85, max_candidates=10, max_merges=1, sleep_seconds=0.0,
                     judge_cache_file=str(tmp_path / "judge.json"),
                     neo4j_uri="bolt://127.0.0.1:1", neo4j_password="unused")

    assert (summary["embed_client_name"], summary["embed_model"]) == ("genai", "gemini-embedding-2")
    assert (summary["judge_client_name_primary"], summary["judge_model_primary"]) == ("openrouter_decisions", "typesafe/jev-1.13")


class _ResponsesRecorder:
    def __init__(self, text: str) -> None:
        self.calls: list[dict] = []
        self.responses = self
        self._text = text

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return type("Response", (), {"output_text": self._text})()


def test_prompt_model_judge_requests_json_with_room_for_reasoning() -> None:
    client = _ResponsesRecorder(json.dumps({"verdict": "ALIAS", "confidence": 0.9, "reason": "same"}))
    role = t3.PromptRoleConfig(client="openrouter_json", model="minimax/minimax-m3")

    result = t3._judge_once({role.client: client}, role_config=role, entity_a=_entity("P&L", ["Financial Metric"]),
                            entity_b=_entity("Profit And Loss", ["Financial Metric"]))

    assert result["status"] == "classified"
    request = client.calls[0]
    assert request["max_output_tokens"] == 2048
    assert request["text"] == {"format": {"type": "json_object"}}
    assert request["extra_body"] == {"provider": {"data_collection": "deny"}}


def test_names_differ_only_in_digits() -> None:
    assert t3._names_differ_only_in_digits("AR(1)", "AR(2)") is True
    assert t3._names_differ_only_in_digits("Iron Man", "Iron Man 3") is True
    assert t3._names_differ_only_in_digits("S&P 500", "SP500") is False
    assert t3._names_differ_only_in_digits("Add Objective Method", "add_objective_method") is False


def test_decision_judge_sends_the_pair_and_reads_the_alias_probability(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict] = []

    def fake_request(client, *, model_name, state, questions):
        calls.append({"model": model_name, "state": state, "questions": questions})
        return {"same": {"noul": 0.97}}

    monkeypatch.setattr(t3, "request_decisions", fake_request)
    role = t3.PromptRoleConfig(client="openrouter_decisions", model="liquid/d1")

    result = t3._judge_once({"openrouter_decisions": object()}, role_config=role, entity_a=_entity("P&L", ["Financial Metric"]),
                            entity_b=_entity("Profit And Loss", ["Financial Metric"]))

    assert (result["status"], result["p_alias"]) == ("classified", 0.97)
    assert calls[0]["questions"] == t3.DECISION_QUESTION
    assert list(calls[0]["state"]["entity_a"]) == list(t3.DECISION_STATE_FIELDS)
    assert calls[0]["state"]["entity_a"]["name"] == "P&L"


def test_decision_judge_without_a_probability_is_unresolved(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(t3, "request_decisions", lambda client, **kwargs: {"same": {}})
    role = t3.PromptRoleConfig(client="openrouter_decisions", model="liquid/d1")

    result = t3.judge_pair({"openrouter_decisions": object()}, _entity("A", ["Model"]), _entity("B", ["Model"]), primary_role_config=role)

    assert (result["status"], result["verdict"]) == ("unresolved", "DIFFERENT")


def test_default_judge_merges_at_the_calibrated_jev_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    probabilities = iter([0.65, 0.64])
    monkeypatch.setattr(t3, "request_decisions", lambda client, **kwargs: {"same": {"noul": next(probabilities)}})
    clients = {"openrouter_decisions": object()}

    verdicts = [t3.judge_pair(clients, _entity("P&L", ["Financial Metric"]), _entity("Profit And Loss", ["Financial Metric"]))["verdict"]
                for _ in range(2)]

    assert (t3.PRIMARY_JUDGE_CLIENT, t3.PRIMARY_JUDGE_MODEL, t3.ALIAS_THRESHOLD) == ("openrouter_decisions", "typesafe/jev-1.13", 0.65)
    assert verdicts == ["ALIAS", "DIFFERENT"]
