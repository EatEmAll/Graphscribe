from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from filelock import FileLock
from fastapi.testclient import TestClient

from notebooklm_graph_pipe.ingestion.ids import corpus_id
from notebooklm_graph_pipe.ingestion.manifest import CorpusManifest, SourceManifestEntry, load_manifest, save_manifest
from notebooklm_graph_pipe.service.api import create_app
from notebooklm_graph_pipe.service.core import CorpusService
from notebooklm_graph_pipe.service import core as core_module
from notebooklm_graph_pipe.service.registry import CorpusRegistry
from notebooklm_graph_pipe.service.security import load_or_create_token


class Service:
    def list_corpora(self):
        return [{"key": "demo"}]

    def get_corpus(self, key):
        if key != "demo":
            raise KeyError(key)
        return {"key": key}

    def submit_sync(self, key):
        return {"id": "job", "corpus_key": key}

    def get_job(self, job_id):
        return {"id": job_id}

    def search(self, key, payload):
        return {"key": key, "query": payload["query"]}

    def answer(self, key, payload):
        return {"answer": payload["question"], "citations": []}

    def answer_stream(self, key, payload):
        yield {"event": "answer_delta", "data": {"text": payload["question"]}}
        yield {"event": "done", "data": {"cancelled": False}}

    def list_documents(self, key):
        return []

    def get_document(self, key, document_id):
        raise KeyError(document_id)

    def delete_document(self, key, document_id):
        return {"status": "deleted"}

    def close(self):
        return None

    def graph_neighbors(self, key, entity_id, hops=1, limit=50):
        return []


def test_api_requires_token_except_health() -> None:
    client = TestClient(create_app(Service(), "x" * 32))
    assert client.get("/health").status_code == 200
    assert client.get("/v1/corpora").status_code == 401
    assert client.get("/v1/corpora", headers={"Authorization": f"Bearer {'x' * 32}"}).json() == [{"key": "demo"}]


def test_search_endpoint_validates_and_delegates() -> None:
    client = TestClient(create_app(Service(), "x" * 32))
    headers = {"Authorization": f"Bearer {'x' * 32}"}
    response = client.post("/v1/corpora/demo/search", headers=headers, json={"query": "hello"})
    assert response.status_code == 200
    assert response.json() == {"key": "demo", "query": "hello"}


def test_answer_endpoint_accepts_global_mode() -> None:
    client = TestClient(create_app(Service(), "x" * 32))
    headers = {"Authorization": f"Bearer {'x' * 32}"}

    response = client.post(
        "/v1/corpora/demo/answer",
        headers=headers,
        json={"question": "themes", "mode": "global"},
    )

    assert response.status_code == 200
    assert response.json()["answer"] == "themes"


def test_answer_stream_endpoint_emits_sse_events() -> None:
    client = TestClient(create_app(Service(), "x" * 32))
    headers = {"Authorization": f"Bearer {'x' * 32}"}

    response = client.post(
        "/v1/corpora/demo/answer/stream",
        headers=headers,
        json={"question": "stream me"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "event: answer_delta" in response.text
    assert '"text": "stream me"' in response.text
    assert "event: done" in response.text


def test_registry_reads_manifests_and_validates_root(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    manifest_path = tmp_path / "registry" / "demo" / "manifest.json"
    save_manifest(
        manifest_path,
        CorpusManifest(
            corpus_id("demo"),
            "demo",
            "Demo",
            {"uri": "bolt://test"},
            dataset_root=str(dataset),
            sources={
                "a.txt": SourceManifestEntry("doc", "revision", "sum", "text", "1")
            },
        ),
    )
    registry = CorpusRegistry(tmp_path / "registry")
    assert registry.get("demo").manifest.title == "Demo"
    assert registry.validate_dataset_root(registry.get("demo")) == dataset.resolve()


def test_local_token_is_created_and_reused(tmp_path: Path) -> None:
    path = tmp_path / "token"
    first = load_or_create_token(path)
    assert len(first) >= 32
    assert load_or_create_token(path) == first


def test_corpus_metadata_never_exposes_neo4j_password() -> None:
    manifest = CorpusManifest(
        corpus_id("demo"),
        "demo",
        "Demo",
        {"uri": "bolt://test", "username": "neo4j", "password": "secret", "database": "neo4j"},
    )
    registry = SimpleNamespace(get=lambda key: SimpleNamespace(manifest=manifest))
    service = CorpusService(registry, SimpleNamespace(), SimpleNamespace())

    payload = service.get_corpus("demo")

    assert payload["neo4j"]["uri"] == "bolt://test"
    assert "password" not in payload["neo4j"]


def test_corpus_manifest_never_persists_neo4j_password(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    save_manifest(
        path,
        CorpusManifest(
            corpus_id("demo"),
            "demo",
            "Demo",
            {
                "uri": "neo4j+s://hosted.example.com",
                "username": "neo4j",
                "password": "do-not-write",
                "database": "neo4j",
                "password_env": "DEMO_NEO4J_PASSWORD",
            },
        ),
    )

    saved = path.read_text(encoding="utf-8")
    assert "do-not-write" not in saved
    assert '"password"' not in saved
    assert '"password_env": "DEMO_NEO4J_PASSWORD"' in saved


def test_document_delete_is_locked_suppressed_and_garbage_collected(monkeypatch, tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    manifest_path = tmp_path / "registry" / "demo" / "manifest.json"
    save_manifest(
        manifest_path,
        CorpusManifest(
            corpus_id("demo"),
            "demo",
            "Demo",
            {"uri": "bolt://test", "username": "neo4j", "password": "pw", "database": "neo4j"},
            dataset_root=str(dataset),
            sources={"a.txt": SourceManifestEntry("doc", "revision", "sum", "text", "1")},
        ),
    )
    calls = []

    class Store:
        def __init__(self, driver, database, *, corpus_id=None):
            pass

        def deactivate_document(self, document_id):
            calls.append(("deactivate", document_id))

        def garbage_collect(self):
            calls.append(("gc", None))

        def restore_revision(self, document_id, revision_id):
            calls.append(("restore", revision_id))

    monkeypatch.setattr(core_module, "Neo4jCorpusStore", Store)
    registry = CorpusRegistry(tmp_path / "registry")
    service = CorpusService(
        registry,
        SimpleNamespace(get=lambda entry: SimpleNamespace(driver=object())),
        SimpleNamespace(),
    )

    result = service.delete_document("demo", "doc")
    saved = load_manifest(manifest_path)

    assert result == {"document_id": "doc", "status": "deleted"}
    assert calls == [("deactivate", "doc"), ("gc", None)]
    assert saved.sources == {}
    assert saved.suppressed_sources == ["a.txt"]

    with FileLock(str(manifest_path.parent / "sync.lock"), timeout=0):
        with pytest.raises(RuntimeError, match="mutating job"):
            service.delete_document("demo", "missing")
