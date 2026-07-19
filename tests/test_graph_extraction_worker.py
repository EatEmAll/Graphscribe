from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from notebooklm_graph_pipe.retrieval.graph_extraction import GraphCapacityError, GraphExtractionWorker


class Store:
    def __init__(self):
        self.saved = []
        self.failed = []

    def pending_graph_parents(self, limit):
        return [
            {"revision_id": "r1", "parent_id": "ok", "text": "good", "child_ids": ["c1"]},
            {"revision_id": "r1", "parent_id": "bad", "text": "bad", "child_ids": ["c2"]},
        ][:limit]

    def persist_parent_graph(self, parent_id, child_ids, graph_document, revision_id=None):
        self.saved.append((parent_id, child_ids, graph_document, revision_id))

    def fail_parent_graph(self, parent_id, message):
        self.failed.append((parent_id, message))

    def finalize_graph_revisions(self):
        return 1


class Transformer:
    async def transform(self, text, parent_id):
        if parent_id == "bad":
            raise RuntimeError("bad graph")
        return {"graph": text}


def test_worker_isolates_parent_failures_and_finalizes_ready_revisions() -> None:
    store = Store()
    summary = asyncio.run(GraphExtractionWorker(store, Transformer()).run_batch())
    assert summary == {"requested": 2, "completed": 1, "failed": 1, "revisions_finalized": 1}
    assert store.saved[0][0] == "ok"
    assert store.saved[0][3] == "r1"
    assert store.failed == [("bad", "bad graph")]


def test_worker_stops_without_marking_parent_failed_when_capacity_gate_fails() -> None:
    store = Store()

    class CapacityTransformer:
        async def transform(self, text, parent_id):
            return SimpleNamespace(nodes=[object()], relationships=[])

    def reject_capacity(node_count, relationship_count):
        raise GraphCapacityError("full")

    with pytest.raises(GraphCapacityError, match="full"):
        asyncio.run(GraphExtractionWorker(store, CapacityTransformer(), reject_capacity).run_batch(limit=1))

    assert store.saved == []
    assert store.failed == []
