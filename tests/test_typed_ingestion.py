from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from notebooklm_graph_pipe.ingestion.adapters import ExtractionContext, SourcePackage, SourcePackageAdapter
from notebooklm_graph_pipe.ingestion.neo4j_store import Neo4jCorpusStore
from notebooklm_graph_pipe.ingestion.ids import corpus_id
from notebooklm_graph_pipe.ingestion.source_ledger import (
    SourceIdentity,
    SourceIdentityConflict,
    identity_from_document,
)
from notebooklm_graph_pipe.service.ingestions import CorpusIngestionManager, package_digest
from notebooklm_graph_pipe.service.registry import CorpusRegistry


def make_package(root: Path) -> Path:
    root.mkdir(parents=True)
    content = b"# Synthetic paper\n\nMechanism and failure evidence.\n"
    (root / "content.md").write_bytes(content)
    (root / "source.json").write_text(
        json.dumps(
            {
                "provider": "doi",
                "provider_source_id": "10.1000/synthetic",
                "canonical_uri": "https://example.invalid/paper",
                "title": "Synthetic paper",
                "source_type": "paper",
                "language": "en",
                "content_sha256": hashlib.sha256(content).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    return root


def test_source_package_uses_declared_exact_identity(tmp_path: Path) -> None:
    package = make_package(tmp_path / "package")
    document = SourcePackageAdapter().extract(
        SourcePackage(package), ExtractionContext(corpus_id("corpus"), tmp_path)
    )
    identity = identity_from_document(document)

    assert identity.provider == "doi"
    assert identity.provider_source_id == "10.1000/synthetic"
    assert identity.canonical_uri == "https://example.invalid/paper"
    assert document.text.startswith("Synthetic paper")


def test_source_package_rejects_changed_content(tmp_path: Path) -> None:
    package = make_package(tmp_path / "package")
    (package / "content.md").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        SourcePackageAdapter().extract(SourcePackage(package), ExtractionContext("corpus", tmp_path))


def test_package_digest_binds_paths_and_bytes(tmp_path: Path) -> None:
    package = make_package(tmp_path / "package")
    before = package_digest(package)
    (package / "content.md").write_text("different", encoding="utf-8")
    assert package_digest(package) != before


def test_ingestion_root_rejects_absolute_and_traversal_paths(tmp_path: Path) -> None:
    ingestion_root = tmp_path / "ingestion"
    ingestion_root.mkdir()
    manager = CorpusIngestionManager(
        CorpusRegistry(tmp_path / "registry"),
        SimpleNamespace(),
        ingestion_root,
        chunker_factory=lambda: SimpleNamespace(),
    )
    try:
        with pytest.raises(ValueError, match="relative"):
            manager._package_path(str(tmp_path.resolve()))
        with pytest.raises(ValueError, match="escapes"):
            manager._package_path("../outside")
    finally:
        manager.close()


def test_staging_marks_vector_ready_without_creating_active_revision() -> None:
    calls: list[str] = []

    class Result:
        def single(self):
            return {"actual": 1}

    class Session:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def run(self, query, **parameters):
            calls.append(query)
            return Result()

    store = Neo4jCorpusStore(SimpleNamespace(session=lambda **kwargs: Session()))
    store.stage_compact_revision("document", "revision", 1)

    query = calls[0]
    assert "status = 'STAGED'" in query
    assert "vector_ready = true" in query
    assert "MERGE (document)-[:ACTIVE_REVISION]" not in query


def test_staged_evaluation_context_is_measured_from_graph_state() -> None:
    class Store:
        def staged_revision_state(self, document_id, revision_id):
            assert (document_id, revision_id) == ("document", "revision")
            return {
                "status": "STAGED",
                "vector_ready": True,
                "graph_ready": True,
                "parent_count": 2,
                "completed_parents": 2,
                "retrievable_parents": 2,
                "is_active": False,
            }

        def capacity_counts(self):
            return {"nodes": 50_000, "relationships": 100_000}

        def staged_revision_failures(self, document_id, revision_id):
            assert (document_id, revision_id) == ("document", "revision")
            return [
                {
                    "parent_id": "parent-2",
                    "attempts": 2,
                    "graph_error": "RuntimeError: upstream rejected request",
                }
            ]

        def close(self):
            pass

    record = SimpleNamespace(
        id="ingestion",
        status="staged",
        corpus_key="demo",
        document_id="document",
        revision_id="revision",
        expected_parents=2,
    )
    manager = object.__new__(CorpusIngestionManager)
    manager._records = {record.id: record}
    manager.registry = SimpleNamespace(
        get=lambda _key: SimpleNamespace(
            manifest=SimpleNamespace(neo4j={"database": "neo4j"}, corpus_id="corpus")
        )
    )
    manager.runtimes = SimpleNamespace(
        get=lambda _entry: SimpleNamespace(driver="driver")
    )
    manager.store_factory = lambda *_args, **_kwargs: Store()
    manager.maximum_nodes = 200_000
    manager.maximum_relationships = 400_000

    context = manager.evaluation_context(record.id)

    assert context["metrics"] == {
        "graph_expansion_ratio": 1.0,
        "capacity_headroom_ratio": 0.75,
        "source_canary_retrieved": True,
    }
    assert context["capacity"] == {
        "nodes": 50_000,
        "relationships": 100_000,
        "maximum_nodes": 200_000,
        "maximum_relationships": 400_000,
    }
    assert context["failures"] == [
        {
            "parent_id": "parent-2",
            "attempts": 2,
            "error_type": "RuntimeError",
            "error_sha256": hashlib.sha256(
                b"RuntimeError: upstream rejected request"
            ).hexdigest(),
            "message": "RuntimeError: upstream rejected request",
        }
    ]


def test_evaluate_persists_against_the_ingestion_corpus_entry() -> None:
    entry = object()
    saved: list[tuple[object, object]] = []
    record = SimpleNamespace(
        id="ingestion",
        status="staged",
        corpus_key="demo",
        document_id="document",
        revision_id="revision",
        expected_parents=2,
        evaluation={},
    )
    manager = object.__new__(CorpusIngestionManager)
    manager._records = {record.id: record}
    manager.registry = SimpleNamespace(get=lambda key: entry if key == "demo" else None)
    manager.evaluation_context = lambda _record_id: {
        "state": {"graph_ready": True, "completed_parents": 2},
        "metrics": {
            "graph_expansion_ratio": 1.0,
            "capacity_headroom_ratio": 0.75,
            "source_canary_retrieved": True,
        },
    }
    manager._save = lambda saved_entry, saved_record: saved.append(
        (saved_entry, saved_record)
    )

    result = manager.evaluate(
        record.id,
        {
            "baseline_quality_ratio": 1.0,
            "effective_citation_ratio": 1.0,
            "unsupported_claim_delta": 0.0,
        },
    )

    assert result.status == "evaluated"
    assert result.evaluation["passed"] is True
    assert saved == [(entry, record)]


def test_staged_revision_failures_are_scoped_and_failed_only() -> None:
    calls: list[tuple[str, dict]] = []

    class Session:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def run(self, query, **parameters):
            calls.append((query, parameters))
            return [
                {
                    "parent_id": "parent",
                    "attempts": 3,
                    "graph_error": "ValueError: invalid graph",
                }
            ]

    store = Neo4jCorpusStore(SimpleNamespace(session=lambda **kwargs: Session()))

    assert store.staged_revision_failures("document", "revision") == [
        {
            "parent_id": "parent",
            "attempts": 3,
            "graph_error": "ValueError: invalid graph",
        }
    ]
    query, parameters = calls[0]
    assert "Document {id: $document_id}" in query
    assert "DocumentRevision {id: $revision_id}" in query
    assert "parent.graph_status = 'FAILED'" in query
    assert parameters == {"document_id": "document", "revision_id": "revision"}


def test_capacity_counts_preserve_empty_relationship_inventory() -> None:
    calls = []

    class Result:
        def single(self):
            return {"nodes": 3, "relationships": 0}

    class Session:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def run(self, query, **_parameters):
            calls.append(query)
            return Result()

    store = Neo4jCorpusStore(SimpleNamespace(session=lambda **kwargs: Session()))

    assert store.capacity_counts() == {"nodes": 3, "relationships": 0}
    assert "OPTIONAL MATCH ()-[relationship]->()" in calls[0]


def test_acceptance_can_require_graph_ready_staged_revision() -> None:
    calls: list[tuple[str, dict]] = []

    class Result:
        def single(self):
            return {"actual": 1}

    class Session:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def run(self, query, **parameters):
            calls.append((query, parameters))
            return Result()

    store = Neo4jCorpusStore(SimpleNamespace(session=lambda **kwargs: Session()), corpus_id="corpus")
    store.activate_compact_revision(
        "document", "revision", 1, ledger=None, require_staged=True
    )

    query, parameters = calls[0]
    assert "revision.status = 'STAGED'" in query
    assert "revision.graph_ready = true" in query
    assert "WITH document, revision, actual" in query
    assert parameters["require_staged"] is True


def test_conflicting_exact_ledger_identities_are_quarantined() -> None:
    class Session:
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def run(self, query, **parameters):
            return iter(
                [
                    {"ledger_source_id": "source-a", "provider": "doi",
                     "provider_source_id": "10.1/a", "canonical_uri": "https://example/a",
                     "content_checksum": "f" * 64},
                    {"ledger_source_id": "source-b", "provider": "doi",
                     "provider_source_id": "10.1/b", "canonical_uri": "https://example/b",
                     "content_checksum": "f" * 64},
                ]
            )

    store = Neo4jCorpusStore(SimpleNamespace(session=lambda **kwargs: Session()))
    identity = SourceIdentity(
        corpus_id("corpus"), "doi", "10.1/a", "Synthetic", "paper",
        "https://example/b", "f" * 64,
    )
    with pytest.raises(SourceIdentityConflict):
        store.resolve_ledger_source(identity)


def test_failed_accept_rollback_restores_previous_revision_and_removes_new_ledger() -> None:
    calls: list[tuple[str, dict]] = []

    class Result:
        def single(self):
            return {"document_id": "document"}

    class Session:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def run(self, query, **parameters):
            calls.append((query, parameters))
            return Result()

    store = Neo4jCorpusStore(SimpleNamespace(session=lambda **kwargs: Session()))
    store.rollback_failed_accept(
        "document", "new", "old", "ledger", remove_new_ledger=True,
        previous_ledger={"title": "Old title"},
        previous_document={"title": "Old document title"},
    )

    query, parameters = calls[0]
    assert "previous.status = 'ACTIVE'" in query
    assert "DELETE source" in query
    assert "MERGE (source)-[:MATERIALIZED_AS]->(document)" in query
    assert "source.title = coalesce($previous_ledger.title" in query
    assert parameters["remove_new_ledger"] is True
    assert parameters["previous_ledger"]["title"] == "Old title"
    assert parameters["previous_document"]["title"] == "Old document title"
