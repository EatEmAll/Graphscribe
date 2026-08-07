from __future__ import annotations

from notebooklm_graph_pipe.retrieval.lancedb_store import LanceDBVectorStore
from notebooklm_graph_pipe.ingestion.embeddings import EmbeddingConfig, MiniLMEmbedder
from notebooklm_graph_pipe.runtime.contracts import VectorQuery, VectorRecord


def record(record_id: str, corpus: str, revision: str, vector: tuple[float, ...]) -> VectorRecord:
    return VectorRecord(
        record_id,
        corpus,
        revision,
        "document",
        f"parent-{record_id}",
        record_id,
        "chunk",
        "embedding-v1",
        vector,
        f"text {record_id}",
    )


def test_lancedb_queries_only_active_corpus_revisions_and_fingerprint(tmp_path) -> None:
    store = LanceDBVectorStore(tmp_path / "vectors", dimension=2)
    store.upsert(
        [
            record("a", "corpus", "active", (1.0, 0.0)),
            record("stale", "corpus", "stale", (1.0, 0.0)),
            record("foreign", "other", "active", (1.0, 0.0)),
        ]
    )

    hits = store.query(
        VectorQuery("corpus", ("active",), "embedding-v1", (1.0, 0.0), 10, {})
    )

    assert [hit.record_id for hit in hits] == ["a"]
    assert store.revisions("corpus") == {"active", "stale"}
    assert store.health()["rows"] == 3


def test_lancedb_upsert_and_revision_deletion_are_idempotent(tmp_path) -> None:
    store = LanceDBVectorStore(tmp_path / "vectors", dimension=2)
    store.upsert([record("a", "corpus", "revision", (1.0, 0.0))])
    store.upsert([record("a", "corpus", "revision", (0.0, 1.0))])

    assert store.health()["rows"] == 1
    store.delete_revisions("corpus", ["revision"])
    store.delete_revisions("corpus", ["revision"])
    assert store.health()["rows"] == 0


def test_default_manifest_model_alias_has_same_fingerprint_as_runtime_default() -> None:
    manifest_embedder = MiniLMEmbedder(EmbeddingConfig(model="all-MiniLM-L6-v2"), model=object())
    runtime_embedder = MiniLMEmbedder(model=object())

    assert manifest_embedder.fingerprint == runtime_embedder.fingerprint
