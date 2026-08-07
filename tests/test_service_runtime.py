from __future__ import annotations

from types import SimpleNamespace

import pytest

from notebooklm_graph_pipe.service.runtime import CorpusRuntime, RuntimeFactory


def test_answerer_is_created_lazily(monkeypatch, tmp_path) -> None:
    runtime = CorpusRuntime(driver=object(), backend=object(), retriever=object())
    factory = RuntimeFactory()
    monkeypatch.setattr(factory, "get", lambda entry: runtime)
    calls = []

    from notebooklm_graph_pipe.service import runtime as runtime_module

    monkeypatch.setattr(
        runtime_module.GroundedAnswerer,
        "from_routing_config",
        lambda retriever, config, **kwargs: calls.append((retriever, config, kwargs)) or object(),
    )
    entry = SimpleNamespace(
        manifest=SimpleNamespace(execution={}),
        manifest_path=tmp_path / "manifest.json",
    )
    first = factory.get_answerer(entry)
    second = factory.get_answerer(entry)
    assert first is second
    assert len(calls) == 1


def test_runtime_is_rebuilt_when_registry_target_changes(monkeypatch) -> None:
    from notebooklm_graph_pipe.service import runtime as runtime_module

    closed: list[str] = []

    class Driver:
        def __init__(self, uri):
            self.uri = uri

        def close(self):
            closed.append(self.uri)

    monkeypatch.setattr(runtime_module.GraphDatabase, "driver", lambda uri, auth: Driver(uri))
    monkeypatch.setattr(runtime_module, "Neo4jRetrievalBackend", lambda *args, **kwargs: object())
    monkeypatch.setattr(runtime_module, "MiniLMEmbedder", lambda: object())
    monkeypatch.setattr(runtime_module, "CrossEncoderReranker", lambda: object())
    monkeypatch.setattr(runtime_module, "load_minilm_tokenizer", lambda: object())
    monkeypatch.setattr(runtime_module, "HybridRetriever", lambda *args, **kwargs: object())
    monkeypatch.setattr(runtime_module, "verify_corpus_connection", lambda *args, **kwargs: {})
    monkeypatch.setenv("DEMO_NEO4J_PASSWORD", "pw")

    manifest = SimpleNamespace(
        corpus_id="corpus",
        neo4j={
            "uri": "bolt://blue",
            "username": "neo4j",
            "database": "blue",
            "password_env": "DEMO_NEO4J_PASSWORD",
        },
        embedding_provider="sentence-transformer",
        embedding_model="all-MiniLM-L6-v2",
        embedding_dimension=384,
        embedding_normalized=True,
    )
    entry = SimpleNamespace(key="demo", manifest=manifest)
    factory = RuntimeFactory()

    blue = factory.get(entry)
    assert factory.get(entry) is blue
    manifest.neo4j = {**manifest.neo4j, "uri": "bolt://green", "database": "green"}
    green = factory.get(entry)

    assert green is not blue
    assert closed == ["bolt://blue"]
    factory.close()
    assert closed == ["bolt://blue", "bolt://green"]


def test_runtime_rejects_embedding_metadata_drift() -> None:
    manifest = SimpleNamespace(
        corpus_id="corpus",
        neo4j={"uri": "bolt://test", "username": "neo4j", "password": "pw", "database": "neo4j"},
        embedding_provider="sentence-transformer",
        embedding_model="different-model",
        embedding_dimension=384,
        embedding_normalized=True,
    )
    entry = SimpleNamespace(key="demo", manifest=manifest)

    with pytest.raises(ValueError, match="blue-green"):
        RuntimeFactory().get(entry)
