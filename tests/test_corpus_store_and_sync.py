from __future__ import annotations

import json
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

        def begin_transaction(self):
            return self

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
    node_query = next(query for query, _ in calls if "MERGE (node:__Entity__" in query)
    assert "ON CREATE SET" in node_query
    assert "SET node.last_seen_revision" in node_query
    assert "SET node:" not in node_query.split("ON CREATE SET", 1)[0]
    assert "ON CREATE SET rel += row.properties" in relationship_query


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


def test_parent_graph_encodes_values_neo4j_cannot_store_in_one_transaction() -> None:
    calls = []
    transactions = []

    class Result:
        def consume(self):
            return None

    class Transaction:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            transactions.append("closed")
            return False

        def run(self, query, **parameters):
            calls.append((query, parameters))
            return Result()

    class Session:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def begin_transaction(self):
            transactions.append("begin")
            return Transaction()

    store = Neo4jCorpusStore(SimpleNamespace(session=lambda **kwargs: Session()))
    period = {"symbol": "A", "start": "2014-01-01", "end": "2014-01-05"}
    node = SimpleNamespace(
        id="window",
        type="Period",
        properties={"period": period, "tickers": ["A", "B"], "mixed": ["A", 1], "rows": [period], "n": 3, "gone": None},
    )
    target = SimpleNamespace(id="target", type="Concept", properties={})
    relationship = SimpleNamespace(source=node, target=target, type="COVERS", properties={"window": period, "weight": 0.5})

    store.persist_parent_graph("parent", [], SimpleNamespace(nodes=[node, target], relationships=[relationship]))

    assert transactions == ["begin", "closed"]
    node_rows = next(parameters["nodes"] for query, parameters in calls if "MERGE (node:__Entity__" in query and parameters["nodes"][0]["id"] == "window")
    properties = node_rows[0]["properties"]
    assert json.loads(properties["period"]) == period
    assert properties["tickers"] == ["A", "B"]
    assert json.loads(properties["mixed"]) == ["A", 1]
    assert json.loads(properties["rows"]) == [period]
    assert properties["n"] == 3 and properties["gone"] is None
    relationship_rows = next(parameters["relationships"] for query, parameters in calls if "MERGE (source)" in query)
    assert json.loads(relationship_rows[0]["properties"]["window"]) == period
    assert relationship_rows[0]["properties"]["weight"] == 0.5


def test_parent_graph_merges_a_node_whose_extracted_id_names_an_existing_entity() -> None:
    """An extracted ``id`` property is the node's identity, never a property that rewrites it."""
    calls = []

    class Result:
        def consume(self):
            return None

    class Session:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def begin_transaction(self):
            return self

        def run(self, query, **parameters):
            calls.append((query, parameters))
            return Result()

    store = Neo4jCorpusStore(SimpleNamespace(session=lambda **kwargs: Session()))
    alias = SimpleNamespace(id="evidence_2026_003", type="Document", properties={"id": "evidence-2026-003", "n": 1})
    named = SimpleNamespace(id="evidence-2026-003", type="Evidence", properties={})
    study = SimpleNamespace(id="study", type="Study", properties={"id": "  "})
    graph = SimpleNamespace(
        nodes=[alias, named, study],
        relationships=[
            SimpleNamespace(source=study, target=alias, type="CITES", properties={}),
            SimpleNamespace(source=alias, target=named, type="SAME_AS", properties={}),
        ],
    )

    store.persist_parent_graph("parent", [], graph, revision_id="revision")

    node_rows = [row for query, parameters in calls if "MERGE (node:__Entity__" in query for row in parameters["nodes"]]
    assert sorted(row["id"] for row in node_rows) == ["evidence-2026-003", "evidence-2026-003", "study"]
    assert all("id" not in row["properties"] for row in node_rows)
    assert next(row for row in node_rows if row["type"] == "Document")["properties"] == {"n": 1}
    relationship_rows = [row for query, parameters in calls if "MERGE (source)" in query for row in parameters["relationships"]]
    assert [(row["source_id"], row["target_id"]) for row in relationship_rows] == [("study", "evidence-2026-003")]
    mention_ids = next(parameters["entity_ids"] for query, parameters in calls if "HAS_ENTITY]->(entity)" in query)
    assert mention_ids == ["evidence-2026-003", "study"]


def test_parent_graph_never_gives_an_extracted_entity_a_corpus_schema_label() -> None:
    calls = []

    class Result:
        def consume(self):
            return None

    class Transaction:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def run(self, query, **parameters):
            calls.append((query, parameters))
            return Result()

    class Session:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def begin_transaction(self):
            return Transaction()

    store = Neo4jCorpusStore(SimpleNamespace(session=lambda **kwargs: Session()))
    nodes = [
        SimpleNamespace(id="parent_chunk", type="ParentChunk", properties={}),
        SimpleNamespace(id="evidence-2026-003", type="Document", properties={}),
        SimpleNamespace(id="deflated sharpe ratio", type="Metric", properties={}),
    ]

    store.persist_parent_graph("parent", [], SimpleNamespace(nodes=nodes, relationships=[]))

    labelled = {
        row["id"]: (query.split("ON CREATE SET node:")[1].split(",")[0], row["type"])
        for query, parameters in calls
        if "MERGE (node:__Entity__" in query
        for row in parameters["nodes"]
    }
    assert labelled == {
        "parent_chunk": ("Entity", "ParentChunk"),
        "evidence-2026-003": ("Entity", "Document"),
        "deflated sharpe ratio": ("Metric", "Metric"),
    }


def test_parent_graph_never_stores_a_node_named_by_its_parent_id() -> None:
    calls = []

    class Session:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def begin_transaction(self):
            return self

        def run(self, query, **parameters):
            calls.append((query, parameters))
            return SimpleNamespace(consume=lambda: None)

    store = Neo4jCorpusStore(SimpleNamespace(session=lambda **kwargs: Session()))
    renamed = SimpleNamespace(id="this chunk", type="Section", properties={"id": "parent"})
    named = SimpleNamespace(id="parent", type="ParentChunk", properties={})
    entity = SimpleNamespace(id="sharpe", type="Metric", properties={})
    graph = SimpleNamespace(
        nodes=[renamed, named, entity],
        relationships=[
            SimpleNamespace(source=renamed, target=entity, type="MENTIONS", properties={}),
            SimpleNamespace(source=entity, target=named, type="PART_OF", properties={}),
        ],
    )

    store.persist_parent_graph("parent", [], graph, revision_id="revision")

    node_ids = [row["id"] for query, parameters in calls if "MERGE (node:__Entity__" in query for row in parameters["nodes"]]
    assert node_ids == ["sharpe"]
    assert [row for query, parameters in calls if "MERGE (source)" in query for row in parameters["relationships"]] == []
    assert next(parameters["entity_ids"] for query, parameters in calls if "HAS_ENTITY]->(entity)" in query) == ["sharpe"]


class _RepairTransaction:
    """Applies the repair's label and entity_type statements to an in-memory entity table."""

    def __init__(self, entities):
        self.entities = entities
        self.queries = []

    def run(self, query, **parameters):
        self.queries.append(query)
        if "WHERE any(label IN labels(entity)" in query:
            return [
                {"id": entity_id, "labels": list(state["labels"]), "entity_type": state["entity_type"]}
                for entity_id, state in sorted(self.entities.items())
                if set(state["labels"]) & set(parameters["schema_labels"])
            ]
        if "OPTIONAL MATCH" in query:
            return [
                {
                    "id": entity_id,
                    "labels": list(self.entities[entity_id]["labels"]) if entity_id in self.entities else None,
                    "entity_type": self.entities.get(entity_id, {}).get("entity_type"),
                }
                for entity_id in parameters["ids"]
            ]
        if "SET entity.entity_type" in query:
            for row in parameters["rows"]:
                self.entities[row["id"]]["entity_type"] = row["entity_type"]
        else:
            operation, label = query.rsplit(" ", 2)[-2], query.rsplit(":", 1)[-1]
            for entity_id in parameters["ids"]:
                labels = self.entities[entity_id]["labels"]
                if operation == "REMOVE":
                    labels.remove(label)
                else:
                    labels.append(label)
        return SimpleNamespace(consume=lambda: None)


def test_schema_label_repair_relabels_entities_and_reverts_from_its_journal() -> None:
    from notebooklm_graph_pipe.ingestion.neo4j_store import plan_schema_label_repair, transition_entity_labels

    original = {
        "tax documents": {"labels": ["__Entity__", "Document"], "entity_type": None},
        "chunk_1": {"labels": ["__Entity__", "Chunk"], "entity_type": "Chunk"},
        "the book": {"labels": ["__Entity__", "Document", "Object", "Book"], "entity_type": None},
        "sharpe": {"labels": ["__Entity__", "Metric"], "entity_type": "Metric"},
    }
    tx = _RepairTransaction({key: {"labels": list(value["labels"]), "entity_type": value["entity_type"]} for key, value in original.items()})

    entries = plan_schema_label_repair(tx)
    transition_entity_labels(tx, entries, "before", "after")

    assert {key: (sorted(value["labels"]), value["entity_type"]) for key, value in tx.entities.items()} == {
        "tax documents": (["Entity", "__Entity__"], "Document"),
        "chunk_1": (["Entity", "__Entity__"], "Chunk"),
        "the book": (["Book", "Object", "__Entity__"], "Document"),
        "sharpe": (["Metric", "__Entity__"], "Metric"),
    }
    assert plan_schema_label_repair(tx) == []

    journal = json.loads(json.dumps(entries))
    transition_entity_labels(tx, journal, "after", "before")

    assert {key: (sorted(value["labels"]), value["entity_type"]) for key, value in tx.entities.items()} == {
        key: (sorted(value["labels"]), value["entity_type"]) for key, value in original.items()
    }


def test_schema_label_repair_refuses_drifted_entities_and_foreign_labels() -> None:
    from notebooklm_graph_pipe.ingestion.neo4j_store import transition_entity_labels

    tx = _RepairTransaction({"paper": {"labels": ["__Entity__", "Document"], "entity_type": None}})
    drifted = [{
        "id": "paper",
        "before": {"labels": ["Document", "__Entity__"], "entity_type": "Document"},
        "after": {"labels": ["Entity", "__Entity__"], "entity_type": "Document"},
    }]
    with pytest.raises(ValueError, match="not in their before state"):
        transition_entity_labels(tx, drifted, "before", "after")

    foreign = [{
        "id": "paper",
        "before": {"labels": ["Document", "__Entity__"], "entity_type": None},
        "after": {"labels": ["Paper", "__Entity__"], "entity_type": None},
    }]
    with pytest.raises(ValueError, match="Refusing to change label 'Paper'"):
        transition_entity_labels(tx, foreign, "before", "after")
    assert [query for query in tx.queries if "REMOVE" in query or "SET entity" in query] == []
