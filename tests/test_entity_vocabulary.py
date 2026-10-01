from __future__ import annotations

import asyncio
import re
from types import SimpleNamespace

import pytest

from notebooklm_graph_pipe.retrieval.entity_vocabulary import EntityVocabulary
from notebooklm_graph_pipe.retrieval.graph_extraction import ExecutorGraphTransformer, GraphExtractionWorker
from notebooklm_graph_pipe.runtime.llm_routing import GRAPH_EXTRACTION_ROLE
from notebooklm_graph_pipe.runtime.model_executor import ModelExecutor, ModelUsage

PAYLOAD = {
    "concepts": [
        {"id": "Walk-Forward Analysis", "type": "Method", "aliases": ["walk-forward", "walk forward optimization"]},
        {"id": "Look-Ahead Bias", "type": "Concept", "aliases": ["lookahead bias"]},
        {"id": "White's Reality Check", "type": "Method", "aliases": ["white's reality check"]},
    ]
}


def node(node_id, node_type="Entity"):
    return SimpleNamespace(id=node_id, type=node_type, properties={})


def test_vocabulary_matches_phrase_variants_on_word_boundaries() -> None:
    vocabulary = EntityVocabulary.from_payload(PAYLOAD)

    assert [c.id for c in vocabulary.mentioned("We ran Walk_Forward tests and a walk-forward optimizations")] == [
        "Walk-Forward Analysis"
    ]
    assert [c.id for c in vocabulary.mentioned("Beware LOOKAHEAD BIASES and White's reality check.")] == [
        "Look-Ahead Bias",
        "White's Reality Check",
    ]
    assert vocabulary.mentioned("a sidewalk forwarding note") == []
    full_match = re.compile(vocabulary.mention_pattern)
    assert full_match.fullmatch("line one\nthen a Walk Forward result\nend")
    assert not full_match.fullmatch("nothing relevant\nhere")


@pytest.mark.parametrize(
    "concepts, message",
    [
        ([{"id": "A", "type": "T", "aliases": ["same"]}, {"id": "B", "type": "T", "aliases": ["Same"]}], "claimed"),
        ([{"id": "A", "type": "T", "aliases": ["bad!"]}], "shape"),
        ([{"id": "A", "type": "", "aliases": []}], "type"),
        ([], "non-empty"),
    ],
)
def test_vocabulary_rejects_ambiguous_or_malformed_payloads(concepts, message) -> None:
    with pytest.raises(ValueError, match=message):
        EntityVocabulary.from_payload({"concepts": concepts})


def test_vocabulary_fingerprint_tracks_content() -> None:
    changed = {"concepts": [*PAYLOAD["concepts"], {"id": "Embargo", "type": "Method", "aliases": []}]}
    assert EntityVocabulary.from_payload(PAYLOAD).fingerprint == EntityVocabulary.from_payload(PAYLOAD).fingerprint
    assert EntityVocabulary.from_payload(PAYLOAD).fingerprint != EntityVocabulary.from_payload(changed).fingerprint


def test_apply_canonicalizes_variants_merges_duplicates_and_adds_literal_mentions() -> None:
    vocabulary = EntityVocabulary.from_payload(PAYLOAD)
    first, second, strategy = node("Walk Forward Optimization"), node("WalkForward"), node("Momentum Strategy")
    graph = SimpleNamespace(
        nodes=[first, second, strategy],
        relationships=[
            SimpleNamespace(source=strategy, target=first, type="VALIDATED_BY", properties={}),
            SimpleNamespace(source=strategy, target=second, type="VALIDATED_BY", properties={}),
            SimpleNamespace(source=first, target=second, type="SAME_AS", properties={}),
        ],
    )

    result = vocabulary.apply(graph, "Momentum validated walk-forward, avoiding look-ahead bias.")

    assert [(n.id, n.type) for n in result.nodes] == [
        ("Walk-Forward Analysis", "Method"),
        ("Momentum Strategy", "Entity"),
        ("Look-Ahead Bias", "Concept"),
    ]
    assert [(r.source.id, r.type, r.target.id) for r in result.relationships] == [
        ("Momentum Strategy", "VALIDATED_BY", "Walk-Forward Analysis")
    ]


class Adapter:
    provider = "test"
    model = "test"

    def __init__(self) -> None:
        self.request = None

    def execute(self, request):
        self.request = request
        return "", {"nodes": [{"id": "lookahead bias", "type": "Risk"}], "relationships": []}, ModelUsage()


def test_executor_transformer_prompts_with_and_applies_the_vocabulary() -> None:
    adapter = Adapter()
    executor = ModelExecutor({"graph": adapter}, {GRAPH_EXTRACTION_ROLE: "graph"})
    vocabulary = EntityVocabulary.from_payload(PAYLOAD)

    graph = asyncio.run(ExecutorGraphTransformer(executor, vocabulary).transform("Lookahead bias.", "p"))

    assert "- Look-Ahead Bias (Concept)" in adapter.request.prompt
    assert "Walk-Forward Analysis" not in adapter.request.prompt
    assert [(n.id, n.type) for n in graph.nodes] == [("Look-Ahead Bias", "Concept")]


class BackfillStore:
    def __init__(self):
        self.selection = None
        self.saved = []

    def vocabulary_graph_parents(self, pattern, fingerprint, limit):
        self.selection = (pattern, fingerprint, limit)
        return [{"revision_id": "r1", "parent_id": "p1", "text": "walk-forward", "child_ids": ["c1"]}]

    def persist_parent_graph(self, parent_id, child_ids, graph_document, revision_id=None, vocabulary_fingerprint=None):
        self.saved.append((parent_id, revision_id, vocabulary_fingerprint))

    def fail_parent_graph(self, parent_id, message):
        raise AssertionError(message)


class EmptyTransformer:
    async def transform(self, text, parent_id):
        return SimpleNamespace(nodes=[], relationships=[])


def test_backfill_selects_by_vocabulary_and_records_its_fingerprint() -> None:
    vocabulary = EntityVocabulary.from_payload(PAYLOAD)
    store = BackfillStore()
    worker = GraphExtractionWorker(store, EmptyTransformer(), vocabulary=vocabulary)

    summary = asyncio.run(worker.run_vocabulary_backfill(limit=7))

    assert summary == {"requested": 1, "completed": 1, "failed": 0}
    assert store.selection == (vocabulary.mention_pattern, vocabulary.fingerprint, 7)
    assert store.saved == [("p1", "r1", vocabulary.fingerprint)]


def test_backfill_requires_a_vocabulary() -> None:
    with pytest.raises(ValueError, match="requires an entity vocabulary"):
        asyncio.run(GraphExtractionWorker(BackfillStore(), EmptyTransformer()).run_vocabulary_backfill())


def test_store_marks_fingerprint_and_scopes_backfill_to_active_revisions() -> None:
    from notebooklm_graph_pipe.ingestion.neo4j_store import Neo4jCorpusStore

    calls = []

    class Result(list):
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

    store = Neo4jCorpusStore(SimpleNamespace(session=lambda **kwargs: Session()), corpus_id="corpus")
    store.persist_parent_graph("p1", [], SimpleNamespace(nodes=[], relationships=[]), "r1", vocabulary_fingerprint="fp")
    status_query, status_parameters = calls[-1]
    assert "parent.graph_vocabulary_fingerprint = $vocabulary_fingerprint" in status_query
    assert status_parameters["vocabulary_fingerprint"] == "fp"

    assert store.vocabulary_graph_parents("(?is).*x.*", "fp", 5) == []
    query, parameters = calls[-1]
    assert "ACTIVE_REVISION" in query and "parent.text =~ $mention_pattern" in query
    assert parameters == {"corpus_id": "corpus", "mention_pattern": "(?is).*x.*", "vocabulary_fingerprint": "fp", "limit": 5}
