from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from notebooklm_graph_pipe.ingestion.adapters import ExtractionContext, TextAdapter
from notebooklm_graph_pipe.ingestion.chunking import ChunkingConfig, HierarchicalChunker
from notebooklm_graph_pipe.ingestion.embeddings import EmbeddingConfig, EmbeddingError, MiniLMEmbedder
from notebooklm_graph_pipe.ingestion.ids import corpus_id
from notebooklm_graph_pipe.ingestion.manifest import CorpusManifest, load_manifest
from notebooklm_graph_pipe.ingestion.neo4j_store import Neo4jCorpusStore, _cypher_identifier
from notebooklm_graph_pipe.ingestion.sync import CorpusSynchronizer, discover_local_sources
from notebooklm_graph_pipe.ingestion import sync as sync_module


class WordTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool = False):
        return list(range(len(text.split())))

    def decode(self, token_ids, *, skip_special_tokens: bool = True):
        return " ".join(f"word{value}" for value in token_ids)


class FakeModel:
    def __init__(self, dimension=4):
        self.dimension = dimension
        self.calls = []

    def encode(self, texts, **kwargs):
        self.calls.append(list(texts))
        return [[float(index == 0) for index in range(self.dimension)] for _ in texts]


class FakeStore:
    def __init__(self):
        self.revisions = []
        self.activated = []
        self.deactivated = []
        self.gc_calls = 0
        self.failed = []
        self.restored = []

    def ensure_schema(self, dimension):
        return None

    def assert_embedding_fingerprint(self, corpus_key, fingerprint):
        return None

    def begin_revision(self, **kwargs):
        self.revisions.append(kwargs)

    def activate_revision(self, document_id, revision_id, expected_chunks):
        self.activated.append((document_id, revision_id, expected_chunks))

    def deactivate_document(self, document_id):
        self.deactivated.append(document_id)

    def garbage_collect(self):
        self.gc_calls += 1
        return {}

    def fail_revision(self, revision_id, message):
        self.failed.append((revision_id, message))

    def restore_revision(self, document_id, revision_id):
        self.restored.append((document_id, revision_id))


def test_embedder_batches_and_validates_vectors() -> None:
    model = FakeModel()
    embedder = MiniLMEmbedder(EmbeddingConfig(dimension=4, batch_size=2), model=model)
    vectors = embedder.embed_documents(["a", "b", "c"])
    assert len(vectors) == 3
    assert [len(call) for call in model.calls] == [2, 1]
    assert embedder.embed_query("query") == [1.0, 0.0, 0.0, 0.0]


def test_embedder_rejects_wrong_dimension() -> None:
    embedder = MiniLMEmbedder(EmbeddingConfig(dimension=4, max_retries=1), model=FakeModel(dimension=3))
    with pytest.raises(EmbeddingError, match="dimensions"):
        embedder.embed_documents(["a"])


def test_sync_rejects_manifest_embedder_drift_before_schema_creation(tmp_path: Path) -> None:
    manifest = CorpusManifest(
        corpus_id("demo"),
        "demo",
        "Demo",
        {"uri": "bolt://test"},
        embedding_dimension=8,
    )
    store = FakeStore()
    synchronizer = CorpusSynchronizer(
        store=store,
        embedder=MiniLMEmbedder(EmbeddingConfig(dimension=4), model=FakeModel()),
        chunker=HierarchicalChunker(WordTokenizer()),
        adapters=(TextAdapter(),),
    )

    with pytest.raises(ValueError, match="blue-green"):
        synchronizer.sync(
            corpus_root=tmp_path,
            manifest=manifest,
            manifest_path=tmp_path / "manifest.json",
            artifact_root=tmp_path / "normalized",
        )


def test_discovery_only_returns_supported_non_hidden_files(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    (tmp_path / "b.md").write_text("b", encoding="utf-8")
    (tmp_path / "c.bin").write_bytes(b"c")
    hidden = tmp_path / ".hidden"
    hidden.mkdir()
    (hidden / "secret.txt").write_text("secret", encoding="utf-8")
    assert [path.name for path in discover_local_sources(tmp_path)] == ["a.txt", "b.md"]


def test_sync_is_incremental_and_deactivates_removed_sources(tmp_path: Path) -> None:
    source = tmp_path / "a.txt"
    source.write_text("one two three four five six seven eight", encoding="utf-8")
    manifest_path = tmp_path / "state" / "manifest.json"
    artifact_root = tmp_path / "state" / "normalized"
    manifest = CorpusManifest(corpus_id("demo"), "demo", "Demo", {"uri": "bolt://test"}, embedding_dimension=4)
    store = FakeStore()
    embedder = MiniLMEmbedder(EmbeddingConfig(dimension=4), model=FakeModel())
    chunker = HierarchicalChunker(
        WordTokenizer(),
        ChunkingConfig(child_target_tokens=4, child_max_tokens=6, child_overlap_tokens=1, child_min_tokens=1),
    )
    synchronizer = CorpusSynchronizer(store=store, embedder=embedder, chunker=chunker, adapters=(TextAdapter(),))

    first = synchronizer.sync(
        corpus_root=tmp_path,
        manifest=manifest,
        manifest_path=manifest_path,
        artifact_root=artifact_root,
    )
    second = synchronizer.sync(
        corpus_root=tmp_path,
        manifest=manifest,
        manifest_path=manifest_path,
        artifact_root=artifact_root,
    )
    source.unlink()
    third = synchronizer.sync(
        corpus_root=tmp_path,
        manifest=manifest,
        manifest_path=manifest_path,
        artifact_root=artifact_root,
    )

    assert first.added == 1
    assert second.unchanged == 1
    assert third.removed == 1
    assert len(store.revisions) == 1
    assert len(store.activated) == 1
    assert len(store.deactivated) == 1
    assert load_manifest(manifest_path) is not None


def test_failed_activation_keeps_manifest_on_previous_revision(tmp_path: Path) -> None:
    source = tmp_path / "a.txt"
    source.write_text("one two three four", encoding="utf-8")
    manifest_path = tmp_path / "state" / "manifest.json"
    manifest = CorpusManifest(corpus_id("demo"), "demo", "Demo", {"uri": "bolt://test"}, embedding_dimension=4)
    store = FakeStore()
    synchronizer = CorpusSynchronizer(
        store=store,
        embedder=MiniLMEmbedder(EmbeddingConfig(dimension=4), model=FakeModel()),
        chunker=HierarchicalChunker(
            WordTokenizer(),
            ChunkingConfig(child_target_tokens=4, child_max_tokens=6, child_overlap_tokens=1, child_min_tokens=1),
        ),
        adapters=(TextAdapter(),),
    )
    first = synchronizer.sync(
        corpus_root=tmp_path,
        manifest=manifest,
        manifest_path=manifest_path,
        artifact_root=tmp_path / "state" / "normalized",
    )
    old_revision = manifest.sources["a.txt"].active_revision_id
    source.write_text("changed content creates a new revision", encoding="utf-8")

    def fail_activation(document_id, revision_id, expected_chunks):
        raise RuntimeError("activation failed")

    store.activate_revision = fail_activation
    second = synchronizer.sync(
        corpus_root=tmp_path,
        manifest=manifest,
        manifest_path=manifest_path,
        artifact_root=tmp_path / "state" / "normalized",
    )

    assert first.added == 1
    assert second.failed == 1
    assert manifest.sources["a.txt"].active_revision_id == old_revision
    assert store.failed and "activation failed" in store.failed[-1][1]


def test_manifest_write_failure_restores_previous_active_revision(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "a.txt"
    source.write_text("one two three four", encoding="utf-8")
    manifest_path = tmp_path / "state" / "manifest.json"
    manifest = CorpusManifest(corpus_id("demo"), "demo", "Demo", {"uri": "bolt://test"}, embedding_dimension=4)
    store = FakeStore()
    synchronizer = CorpusSynchronizer(
        store=store,
        embedder=MiniLMEmbedder(EmbeddingConfig(dimension=4), model=FakeModel()),
        chunker=HierarchicalChunker(
            WordTokenizer(),
            ChunkingConfig(child_target_tokens=4, child_max_tokens=6, child_overlap_tokens=1, child_min_tokens=1),
        ),
        adapters=(TextAdapter(),),
    )
    synchronizer.sync(
        corpus_root=tmp_path,
        manifest=manifest,
        manifest_path=manifest_path,
        artifact_root=tmp_path / "state" / "normalized",
    )
    previous = manifest.sources["a.txt"]
    source.write_text("changed content creates a new revision", encoding="utf-8")
    real_save = sync_module.save_manifest
    calls = 0

    def fail_once(path, value):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("disk full")
        return real_save(path, value)

    monkeypatch.setattr(sync_module, "save_manifest", fail_once)
    report = synchronizer.sync(
        corpus_root=tmp_path,
        manifest=manifest,
        manifest_path=manifest_path,
        artifact_root=tmp_path / "state" / "normalized",
    )

    assert report.failed == 1
    assert manifest.sources["a.txt"] == previous
    assert store.restored[-1] == (previous.document_id, previous.active_revision_id)


def test_suppressed_source_is_not_reingested(tmp_path: Path) -> None:
    source = tmp_path / "a.txt"
    source.write_text("one two three four", encoding="utf-8")
    manifest_path = tmp_path / "state" / "manifest.json"
    manifest = CorpusManifest(
        corpus_id("demo"),
        "demo",
        "Demo",
        {"uri": "bolt://test"},
        suppressed_sources=["a.txt"],
        embedding_dimension=4,
    )
    store = FakeStore()
    synchronizer = CorpusSynchronizer(
        store=store,
        embedder=MiniLMEmbedder(EmbeddingConfig(dimension=4), model=FakeModel()),
        chunker=HierarchicalChunker(
            WordTokenizer(),
            ChunkingConfig(child_target_tokens=4, child_max_tokens=6, child_overlap_tokens=1, child_min_tokens=1),
        ),
        adapters=(TextAdapter(),),
    )

    report = synchronizer.sync(
        corpus_root=tmp_path,
        manifest=manifest,
        manifest_path=manifest_path,
        artifact_root=tmp_path / "state" / "normalized",
    )

    assert report.added == 0
    assert not store.revisions
    assert [(event.source, event.status) for event in report.events] == [("a.txt", "suppressed")]
    assert load_manifest(manifest_path).suppressed_sources == ["a.txt"]


def test_parent_graph_mentions_are_not_copied_to_every_child() -> None:
    calls = []

    class Result:
        def consume(self):
            return None

    class Session:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def run(self, query, **parameters):
            calls.append((query, parameters))
            return Result()

    driver = SimpleNamespace(session=lambda **kwargs: Session())
    store = Neo4jCorpusStore(driver)
    node = SimpleNamespace(id="entity", type="Concept", properties={})
    target = SimpleNamespace(id="target", type="Concept", properties={})
    relationship = SimpleNamespace(source=node, target=target, type="RELATED_TO", properties={})
    graph = SimpleNamespace(nodes=[node, target], relationships=[relationship])

    store.persist_parent_graph("parent", ["child-1", "child-2"], graph)

    mention_query, mention_parameters = next(
        (query, parameters) for query, parameters in calls if "extraction_scope" in query
    )
    assert "MATCH (parent:ParentChunk" in mention_query
    assert "MATCH (chunk:Chunk" not in mention_query
    assert "child_ids" not in mention_parameters
    assert all("apoc." not in query for query, _ in calls)
    relationship_query, relationship_parameters = next(
        (query, parameters) for query, parameters in calls if "source_parent_ids" in query and "MERGE (source)" in query
    )
    assert "$parent_id" in relationship_query
    assert relationship_parameters["parent_id"] == "parent"


def test_graph_types_are_normalized_to_safe_cypher_identifiers() -> None:
    assert _cypher_identifier("Trading Concept", "Entity") == "Trading_Concept"
    assert _cypher_identifier("1); MATCH (n) DETACH DELETE n //", "Entity").startswith("Entity_1_MATCH")


def test_revision_build_preserves_existing_active_document_availability(tmp_path: Path) -> None:
    source = tmp_path / "a.txt"
    source.write_text("one two three four", encoding="utf-8")
    document = TextAdapter().extract(source, ExtractionContext(corpus_id("demo"), tmp_path))
    chunks = HierarchicalChunker(
        WordTokenizer(),
        ChunkingConfig(child_target_tokens=4, child_max_tokens=6, child_overlap_tokens=1, child_min_tokens=1),
    ).chunk(document)
    calls = []

    class Result:
        def consume(self):
            return None

    class Session:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def run(self, query, **parameters):
            calls.append((query, parameters))
            return Result()

    store = Neo4jCorpusStore(SimpleNamespace(session=lambda **kwargs: Session()))
    store.begin_revision(
        corpus_key="demo",
        corpus_title="Demo",
        embedding_fingerprint="test",
        document=document,
        chunks=chunks,
        embeddings=[[1.0, 0.0, 0.0, 0.0] for _ in chunks.children],
    )

    first_query = calls[0][0]
    assert "OPTIONAL MATCH (document)-[:ACTIVE_REVISION]->(active_revision" in first_query
    assert "CASE WHEN active_revision IS NULL THEN 'BUILDING' ELSE 'READY' END" in first_query


def test_graph_queue_queries_are_scoped_to_store_corpus() -> None:
    calls = []

    class Result:
        def __iter__(self):
            return iter(())

        def single(self):
            return {"count": 0}

    class Session:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def run(self, query, **parameters):
            calls.append((query, parameters))
            return Result()

    store = Neo4jCorpusStore(
        SimpleNamespace(session=lambda **kwargs: Session()),
        corpus_id="corpus-id",
    )

    assert store.pending_graph_parents() == []
    assert store.finalize_graph_revisions() == 0
    assert all("(:Corpus {id: $corpus_id})" in query for query, _ in calls)
    assert all(parameters["corpus_id"] == "corpus-id" for _, parameters in calls)
