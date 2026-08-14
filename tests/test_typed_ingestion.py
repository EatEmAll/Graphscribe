from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from notebooklm_graph_pipe.ingestion.adapters import ExtractionContext, SourcePackage, SourcePackageAdapter
from notebooklm_graph_pipe.ingestion.neo4j_store import Neo4jCorpusStore
from notebooklm_graph_pipe.ingestion.ids import corpus_id
from notebooklm_graph_pipe.ingestion.source_ledger import identity_from_document
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
    assert parameters["require_staged"] is True


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
    store.rollback_failed_accept("document", "new", "old", "ledger", remove_new_ledger=True)

    query, parameters = calls[0]
    assert "previous.status = 'ACTIVE'" in query
    assert "DELETE source" in query
    assert "MERGE (source)-[:MATERIALIZED_AS]->(document)" in query
    assert parameters["remove_new_ledger"] is True
