from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from notebooklm_graph_pipe.ingestion.adapters import ExtractionContext, TextAdapter
from notebooklm_graph_pipe.ingestion.chunking import ChunkingConfig, HierarchicalChunker
from notebooklm_graph_pipe.ingestion.compact_sync import CompactCorpusUpdater
from notebooklm_graph_pipe.ingestion.embeddings import EmbeddingConfig, MiniLMEmbedder
from notebooklm_graph_pipe.ingestion.ids import corpus_id
from notebooklm_graph_pipe.ingestion.manifest import CorpusManifest, load_manifest
from notebooklm_graph_pipe.ingestion.neo4j_store import Neo4jCorpusStore


class WordTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool = False):
        return list(range(len(text.split())))

    def decode(self, token_ids, *, skip_special_tokens: bool = True):
        return " ".join(f"word{value}" for value in token_ids)


class FakeModel:
    def encode(self, texts, **kwargs):
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]


class FakeStore:
    def __init__(self):
        self.compact_revisions = []
        self.activated = []

    def assert_embedding_fingerprint(self, corpus_key, fingerprint):
        return None

    def assert_existing_corpus(self, corpus_id, corpus_key):
        return None

    def active_revision_for_document(self, document_id):
        return None

    def begin_compact_revision(self, **kwargs):
        self.compact_revisions.append(kwargs)

    def activate_compact_revision(self, document_id, revision_id, expected_parents):
        self.activated.append((document_id, revision_id, expected_parents))

    def fail_revision(self, revision_id, message):
        raise AssertionError(message)

    def garbage_collect(self, document_ids=None, *, revision_ids=None):
        return {}


def test_compact_update_persists_only_parent_embeddings_and_keeps_other_sources(tmp_path: Path) -> None:
    source = tmp_path / "new.txt"
    source.write_text("one two three four five six seven eight", encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest = CorpusManifest(
        corpus_id("demo"),
        "demo",
        "Demo",
        {"uri": "neo4j+s://example.databases.neo4j.io"},
        embedding_dimension=4,
        retrieval_unit="parent",
        retrieval_vector_index="parent_embedding_v1",
        retrieval_keyword_index="parent_keyword_v1",
    )
    manifest.removed_sources = ["historical.txt"]
    store = FakeStore()
    updater = CompactCorpusUpdater(
        store=store,
        embedder=MiniLMEmbedder(EmbeddingConfig(dimension=4), model=FakeModel()),
        chunker=HierarchicalChunker(
            WordTokenizer(),
            ChunkingConfig(
                child_target_tokens=4,
                child_max_tokens=6,
                child_overlap_tokens=1,
                child_min_tokens=1,
            ),
        ),
        adapters=(TextAdapter(),),
    )

    report = updater.update(
        sources=[source],
        corpus_root=tmp_path,
        manifest=manifest,
        manifest_path=manifest_path,
    )

    assert report.added == 1
    assert report.events[0].transient_child_chunks > 0
    saved = store.compact_revisions[0]
    assert "parent_embeddings" in saved
    assert "embeddings" not in saved
    assert len(saved["parent_embeddings"]) == len(saved["chunks"].parents)
    assert store.activated[0][2] == len(saved["chunks"].parents)
    assert load_manifest(manifest_path).removed_sources == ["historical.txt"]


def test_compact_update_rejects_child_retrieval_manifest(tmp_path: Path) -> None:
    manifest = CorpusManifest(corpus_id("demo"), "demo", "Demo", {"uri": "bolt://test"})
    updater = CompactCorpusUpdater(
        store=FakeStore(),
        embedder=MiniLMEmbedder(EmbeddingConfig(dimension=4), model=FakeModel()),
        chunker=HierarchicalChunker(WordTokenizer()),
        adapters=(TextAdapter(),),
    )

    try:
        updater.update(sources=[], corpus_root=tmp_path, manifest=manifest, manifest_path=tmp_path / "m.json")
    except ValueError as exc:
        assert "retrieval profile" in str(exc)
    else:
        raise AssertionError("Expected compact profile validation to fail.")


def test_compact_store_never_creates_child_chunk_nodes(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    source.write_text("one two three four five six", encoding="utf-8")
    document = TextAdapter().extract(source, ExtractionContext(corpus_id("demo"), tmp_path))
    chunks = HierarchicalChunker(WordTokenizer()).chunk(document)
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
    store.begin_compact_revision(
        corpus_key="demo",
        corpus_title="Demo",
        embedding_fingerprint="fingerprint",
        document=document,
        chunks=chunks,
        parent_embeddings=[[1.0, 0.0, 0.0, 0.0] for _ in chunks.parents],
    )

    assert all("MERGE (chunk:Chunk" not in query for query, _ in calls)
    parent_call = next((query, params) for query, params in calls if "MERGE (parent:ParentChunk" in query)
    assert all("embedding" in row for row in parent_call[1]["rows"])


def test_failed_compact_activation_cleans_only_the_staged_revision(tmp_path: Path) -> None:
    source = tmp_path / "new.txt"
    source.write_text("one two three four five six", encoding="utf-8")
    manifest = CorpusManifest(
        corpus_id("demo"),
        "demo",
        "Demo",
        {"uri": "neo4j+s://example.databases.neo4j.io"},
        embedding_dimension=4,
        retrieval_unit="parent",
        retrieval_vector_index="parent_embedding_v1",
        retrieval_keyword_index="parent_keyword_v1",
    )

    class ActivationFailingStore(FakeStore):
        def __init__(self):
            super().__init__()
            self.failed = []
            self.collected = []

        def activate_compact_revision(self, document_id, revision_id, expected_parents):
            raise RuntimeError("activation failed")

        def fail_revision(self, revision_id, message):
            self.failed.append(revision_id)

        def garbage_collect(self, document_ids=None, *, revision_ids=None):
            self.collected.append(revision_ids)
            return {}

    store = ActivationFailingStore()
    updater = CompactCorpusUpdater(
        store=store,
        embedder=MiniLMEmbedder(EmbeddingConfig(dimension=4), model=FakeModel()),
        chunker=HierarchicalChunker(WordTokenizer()),
        adapters=(TextAdapter(),),
    )

    report = updater.update(
        sources=[source],
        corpus_root=tmp_path,
        manifest=manifest,
        manifest_path=tmp_path / "manifest.json",
    )

    assert report.failed == 1
    revision_id = store.compact_revisions[0]["document"].revision_id
    assert store.failed == [revision_id]
    assert store.collected == [[revision_id]]
