from __future__ import annotations

import asyncio

from notebooklm_graph_pipe.retrieval.graph_extraction import ExecutorGraphTransformer
from notebooklm_graph_pipe.runtime.llm_routing import GRAPH_EXTRACTION_ROLE
from notebooklm_graph_pipe.runtime.model_executor import ModelExecutor, ModelUsage


class Adapter:
    provider = "test"
    model = "test"

    def __init__(self) -> None:
        self.request = None

    def execute(self, request):
        self.request = request
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
    adapter = Adapter()
    executor = ModelExecutor(
        {"graph": adapter},
        {GRAPH_EXTRACTION_ROLE: "graph"},
    )

    graph = asyncio.run(ExecutorGraphTransformer(executor).transform("A uses B", "parent"))

    assert [node.id for node in graph.nodes] == ["a", "b"]
    assert len(graph.relationships) == 1
    assert graph.relationships[0].type == "USES"
    assert "source_id and target_id" in adapter.request.prompt


def test_executor_graph_transformer_drops_a_node_named_by_the_parent_id() -> None:
    class SelfNamingAdapter(Adapter):
        def execute(self, request):
            return "", {
                "nodes": [{"id": "parent-1", "type": "ParentChunk"}, {"id": "a", "type": "System"}],
                "relationships": [{"source_id": "parent-1", "target_id": "a", "type": "MENTIONS"}],
            }, ModelUsage()

    executor = ModelExecutor({GRAPH_EXTRACTION_ROLE: SelfNamingAdapter()}, {GRAPH_EXTRACTION_ROLE: GRAPH_EXTRACTION_ROLE})

    graph = asyncio.run(ExecutorGraphTransformer(executor).transform("A text.", "parent-1"))

    assert [node.id for node in graph.nodes] == ["a"]
    assert graph.relationships == []


def test_graph_extraction_defaults_to_deepseek_through_openrouter_json(monkeypatch, tmp_path) -> None:
    from notebooklm_graph_pipe.retrieval import graph_extraction
    from notebooklm_graph_pipe.runtime import llm_json_utils
    from notebooklm_graph_pipe.runtime.llm_json_utils import CliResponse

    captured: list[dict] = []

    class _Responses:
        def create(self, **kwargs):
            captured.append(kwargs)
            return CliResponse(output_text='{"nodes": [], "relationships": []}')

    class _Client:
        responses = _Responses()

    built: list[tuple[str, ...]] = []

    def build(*names: str):
        built.append(names)
        return {name: _Client() for name in names}

    monkeypatch.setattr(graph_extraction, "build_single_prompt_clients", build)
    worker = graph_extraction.GraphExtractionWorker.from_routing_config(
        store=None, config_path=None, cache_path=str(tmp_path / "cache.sqlite3")
    )
    asyncio.run(worker.transformer.transform("Alpha uses Beta.", "parent-1"))

    assert built == [("openrouter_json",)]
    assert worker.transformer.executor.policies[GRAPH_EXTRACTION_ROLE].timeout_seconds == 900.0
    assert captured[0]["model"] == "deepseek/deepseek-v4.1-flash"
    assert captured[0]["text"] == {
        "format": {"type": "json_schema", "name": "response", "schema": graph_extraction.GRAPH_SCHEMA, "strict": False}
    }
    assert captured[0]["extra_body"] == {"provider": llm_json_utils.OPENROUTER_JSON_PROVIDER}
    assert llm_json_utils.OPENROUTER_JSON_PROVIDER == {"data_collection": "deny"}
