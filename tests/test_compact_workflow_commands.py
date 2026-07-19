from __future__ import annotations

from types import SimpleNamespace

import pytest

from notebooklm_graph_pipe.ingestion.neo4j_store import Neo4jCorpusStore
from scripts import finalize_compact_update
from scripts.consolidation import consolidate_delta


class Result:
    def __init__(self, rows):
        self.rows = rows

    def __iter__(self):
        return iter(self.rows)

    def single(self):
        return self.rows[0] if self.rows else None


class Driver:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def session(self, database=None):
        driver = self

        class Session:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def run(self, query, **parameters):
                driver.calls.append((query, parameters))
                return Result(driver.rows)

        return Session()


def test_delta_preflight_rejects_revision_without_active_relation_and_scopes_corpus() -> None:
    driver = Driver(
        [
            {
                "revision_id": "revision-1",
                "status": "ACTIVE",
                "graph_ready": True,
                "is_active": False,
                "parents": 2,
                "entities": 3,
            }
        ]
    )

    with pytest.raises(RuntimeError, match="ACTIVE, graph-ready"):
        consolidate_delta._preflight(
            driver,
            "neo4j",
            "corpus-1",
            ["revision-1"],
            require_apoc=False,
        )

    query, parameters = driver.calls[0]
    assert "Corpus {id: $corpus_id}" in query
    assert parameters["corpus_id"] == "corpus-1"


def test_delta_execute_requires_existing_aura_backup(tmp_path) -> None:
    with pytest.raises(SystemExit):
        consolidate_delta.main(
            [
                "--manifest-path",
                str(tmp_path / "manifest.json"),
                "--revision-id",
                "revision-1",
                "--confirm-target",
                "neo4j+s://example|neo4j",
                "--execute",
                "--backup-file",
                str(tmp_path / "missing.backup"),
            ]
        )


def test_finalize_revision_lookup_is_scoped_to_manifest_corpus() -> None:
    driver = Driver(
        [
            {
                "document_id": "document-1",
                "revision_id": "revision-1",
                "status": "ACTIVE",
                "vector_ready": True,
                "graph_ready": True,
                "checksum": "sum",
                "extractor": "text",
                "extractor_version": "1",
                "is_active": True,
            }
        ]
    )

    row = finalize_compact_update._revision(driver, "neo4j", "corpus-1", "revision-1")

    assert row["document_id"] == "document-1"
    query, parameters = driver.calls[0]
    assert "Corpus {id: $corpus_id}" in query
    assert parameters == {"corpus_id": "corpus-1", "revision_id": "revision-1"}


def test_document_scoped_garbage_collection_skips_global_orphan_deletion() -> None:
    calls = []

    class Session:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def run(self, query, **parameters):
            calls.append((query, parameters))
            if "RETURN size(retired)" in query:
                return Result([{"relationships": 0}])
            return Result([{"revisions": 1, "chunks": 0, "parents": 2}])

    store = Neo4jCorpusStore(SimpleNamespace(session=lambda **kwargs: Session()))

    result = store.garbage_collect(["document-1"])

    assert result["entities"] == 0
    assert all("MATCH (entity:__Entity__)" not in query for query, _ in calls)
    assert all(parameters["document_ids"] == ["document-1"] for _, parameters in calls)
    assert all(parameters["revision_ids"] is None for _, parameters in calls)


def test_revision_scoped_garbage_collection_only_deletes_entities_created_by_revision() -> None:
    calls = []

    class Session:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def run(self, query, **parameters):
            calls.append((query, parameters))
            if "RETURN size(retired)" in query:
                return Result([{"relationships": 0}])
            if "MATCH (entity:__Entity__)" in query:
                return Result([{"entities": 2}])
            return Result([{"revisions": 1, "chunks": 0, "parents": 2}])

    store = Neo4jCorpusStore(SimpleNamespace(session=lambda **kwargs: Session()))

    result = store.garbage_collect(revision_ids=["revision-1"])

    assert result["entities"] == 2
    entity_query, parameters = next(
        (query, parameters) for query, parameters in calls if "MATCH (entity:__Entity__)" in query
    )
    assert "entity.created_in_revision IN $revision_ids" in entity_query
    assert parameters == {"revision_ids": ["revision-1"]}
