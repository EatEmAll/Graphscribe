from __future__ import annotations

import json
from pathlib import Path

import pytest

from notebooklm_graph_pipe.ingestion.adapters import SourcePackage, SourcePackageAdapter
from notebooklm_graph_pipe.ingestion.chunking import ChunkingConfig, HierarchicalChunker
from notebooklm_graph_pipe.ingestion.compact_sync import CompactCorpusUpdater
from notebooklm_graph_pipe.ingestion.embeddings import EmbeddingConfig, MiniLMEmbedder
from notebooklm_graph_pipe.ingestion.ids import corpus_id
from notebooklm_graph_pipe.ingestion.manifest import CorpusManifest, load_manifest, save_manifest
from notebooklm_graph_pipe.ingestion.neo4j_store import GRAPH_SCHEMA_VERSION, Neo4jCorpusStore
from scripts import bootstrap_compact_corpus
from tests.test_compact_updates import FakeModel, FakeStore, WordTokenizer
from tests.test_typed_ingestion import make_package


class Result:
    def __init__(self, row=None):
        self.row = row

    def single(self):
        return self.row

    def consume(self):
        return None


class Driver:
    def __init__(self, nodes: int):
        self.nodes = nodes
        self.queries: list[tuple[str, dict]] = []

    def session(self, database=None):
        driver = self

        class Session:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def run(self, query, **parameters):
                driver.queries.append((query, parameters))
                return Result({"nodes": driver.nodes})

        return Session()


def _manifest(key: str = "aura-compact", uri: str = "neo4j+s://active.databases.neo4j.io") -> CorpusManifest:
    return CorpusManifest(
        corpus_id(key),
        key,
        "Active",
        {"uri": uri, "database": "active", "password_env": "ACTIVE_PASSWORD"},
        dataset_root="/active/sources",
        embedding_dimension=4,
        retrieval_unit="parent",
        retrieval_vector_index="parent_embedding_v1",
        retrieval_keyword_index="parent_keyword_v1",
        graph={"entity_vocabulary_path": "/vocabulary.json"},
        removed_sources=["old.txt"],
    )


def test_bootstrap_refuses_a_database_that_holds_any_node() -> None:
    driver = Driver(nodes=3)
    store = Neo4jCorpusStore(driver, "neo4j", corpus_id=corpus_id("rebuild"))

    with pytest.raises(ValueError, match="empty database; it holds 3 nodes"):
        store.bootstrap_compact_corpus(
            corpus_key="rebuild", corpus_title="Rebuild", embedding_fingerprint="fp", dimension=4
        )

    assert len(driver.queries) == 1


def test_bootstrap_creates_parent_schema_then_the_corpus_node() -> None:
    driver = Driver(nodes=0)
    store = Neo4jCorpusStore(driver, "neo4j", corpus_id=corpus_id("rebuild"))

    store.bootstrap_compact_corpus(
        corpus_key="rebuild", corpus_title="Rebuild", embedding_fingerprint="fp", dimension=4
    )

    queries = [query for query, _ in driver.queries]
    assert any("parent_keyword_v1" in query for query in queries)
    assert not any("chunk_keyword_v1" in query for query in queries)
    query, parameters = driver.queries[-1]
    assert "CREATE (corpus:Corpus" in query
    assert parameters == {
        "corpus_id": corpus_id("rebuild"),
        "corpus_key": "rebuild",
        "corpus_title": "Rebuild",
        "schema_version": GRAPH_SCHEMA_VERSION,
        "embedding_fingerprint": "fp",
    }


def test_target_manifest_copies_settings_onto_a_new_source_free_identity() -> None:
    template = _manifest()
    template.sources = {"kept.txt": object()}

    target = bootstrap_compact_corpus.target_manifest(
        template, corpus_key="rebuild", title="Rebuild", neo4j={"uri": "bolt://127.0.0.1:7687"}
    )

    assert (target.corpus_id, target.corpus_key, target.title) == (corpus_id("rebuild"), "rebuild", "Rebuild")
    assert target.neo4j == {"uri": "bolt://127.0.0.1:7687"}
    assert target.sources == {} and target.removed_sources == [] and target.dataset_root is None
    assert target.graph["entity_vocabulary_path"] == "/vocabulary.json"
    assert (target.retrieval_unit, target.embedding_dimension) == ("parent", 4)
    assert template.corpus_key == "aura-compact"


def _arguments(tmp_path: Path, template: Path, **overrides: str) -> list[str]:
    package = make_package(tmp_path / "package")
    values = {
        "--template-manifest": str(template),
        "--target-manifest": str(tmp_path / "rebuild" / "manifest.json"),
        "--corpus-key": "rebuild",
        "--neo4j-uri": "bolt://127.0.0.1:7687",
        "--confirm-target": "bolt://127.0.0.1:7687|neo4j",
        "--package": str(package),
        **overrides,
    }
    return [item for pair in values.items() for item in pair]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"--corpus-key": "aura-compact"}, "must differ from the template"),
        (
            {"--neo4j-uri": "neo4j+s://active.databases.neo4j.io", "--neo4j-database": "active"},
            "template corpus's own database",
        ),
        ({"--package": "/nonexistent"}, "Not source package directories"),
    ],
)
def test_bootstrap_command_refuses_before_connecting(tmp_path, capsys, overrides, message) -> None:
    template = tmp_path / "active" / "manifest.json"
    save_manifest(template, _manifest())

    with pytest.raises(SystemExit):
        bootstrap_compact_corpus.main(_arguments(tmp_path, template, **overrides))

    assert message in capsys.readouterr().err
    assert not (tmp_path / "rebuild").exists()


def test_bootstrap_command_refuses_an_existing_target_manifest(tmp_path, capsys) -> None:
    template = tmp_path / "active" / "manifest.json"
    save_manifest(template, _manifest())
    existing = tmp_path / "rebuild" / "manifest.json"
    save_manifest(existing, _manifest("rebuild"))
    before = existing.read_bytes()

    with pytest.raises(SystemExit):
        bootstrap_compact_corpus.main(_arguments(tmp_path, template))

    assert "already exists" in capsys.readouterr().err
    assert existing.read_bytes() == before


def test_compact_update_keys_source_packages_by_ledger_identity(tmp_path: Path) -> None:
    package = make_package(tmp_path / "package")
    manifest = _manifest("rebuild")
    manifest_path = tmp_path / "manifest.json"
    updater = CompactCorpusUpdater(
        store=FakeStore(),
        embedder=MiniLMEmbedder(EmbeddingConfig(dimension=4), model=FakeModel()),
        chunker=HierarchicalChunker(
            WordTokenizer(),
            ChunkingConfig(child_target_tokens=4, child_max_tokens=6, child_overlap_tokens=1, child_min_tokens=1),
        ),
        adapters=(SourcePackageAdapter(),),
    )

    report = updater.update(
        sources=[SourcePackage(package)], corpus_root=tmp_path, manifest=manifest, manifest_path=manifest_path
    )

    assert report.added == 1
    (key,) = json.loads(manifest_path.read_text(encoding="utf-8"))["sources"]
    assert key.startswith("source-package/") and key == report.events[0].source
    assert load_manifest(manifest_path).sources[key].ledger_source_id == key.removeprefix("source-package/")
