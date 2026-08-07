from __future__ import annotations

import asyncio

from notebooklm_graph_pipe.retrieval.graph_extraction import ExecutorGraphTransformer
from notebooklm_graph_pipe.runtime.llm_routing import GRAPH_EXTRACTION_ROLE
from notebooklm_graph_pipe.runtime.model_executor import ModelExecutor, ModelUsage


class Adapter:
    provider = "test"
    model = "test"

    def execute(self, request):
        return "", {
            "nodes": [
                {"id": "a", "type": "System"},
                {"id": "b", "type": "Database", "properties": {"name": "B"}},
            ],
            "relationships": [
                {"source_id": "a", "target_id": "b", "type": "USES"},
                {"source_id": "a", "target_id": "missing", "type": "INVALID"},
            ],
        }, ModelUsage()


def test_executor_graph_transformer_drops_relationships_with_unknown_endpoints() -> None:
    executor = ModelExecutor(
        {"graph": Adapter()},
        {GRAPH_EXTRACTION_ROLE: "graph"},
    )

    graph = asyncio.run(ExecutorGraphTransformer(executor).transform("A uses B", "parent"))

    assert [node.id for node in graph.nodes] == ["a", "b"]
    assert len(graph.relationships) == 1
    assert graph.relationships[0].type == "USES"
