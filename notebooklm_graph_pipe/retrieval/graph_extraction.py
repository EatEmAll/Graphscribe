from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable, Protocol

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from notebooklm_graph_pipe.runtime.llm_routing import (
    GRAPH_EXTRACTION_ROLE,
    PromptRoleConfig,
    resolve_prompt_role,
)
from notebooklm_graph_pipe.runtime.llm_json_utils import build_single_prompt_clients
from notebooklm_graph_pipe.runtime.model_adapters import RoutedJsonAdapter
from notebooklm_graph_pipe.runtime.model_executor import ExecutionPolicy, ModelExecutor, ModelRequest

GRAPH_SCHEMA: dict[str, object] = {
    "type": "object",
    "required": ["nodes", "relationships"],
    "properties": {
        "nodes": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "type"],
                "properties": {
                    "id": {"type": "string"},
                    "type": {"type": "string"},
                    "properties": {"type": "object"},
                },
            },
        },
        "relationships": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["source_id", "target_id", "type"],
                "properties": {
                    "source_id": {"type": "string"},
                    "target_id": {"type": "string"},
                    "type": {"type": "string"},
                    "properties": {"type": "object"},
                },
            },
        },
    },
}


class GraphTransformer(Protocol):
    async def transform(self, text: str, parent_id: str) -> Any: ...


class GraphCapacityError(RuntimeError):
    pass


class LangChainGraphTransformer:
    def __init__(self, role: PromptRoleConfig):
        self.role = role
        self._transformer: Any | None = None

    def _llm(self):
        if self.role.client == "genai":
            key = os.environ.get("GOOGLE_API_KEY", "")
            if not key:
                raise RuntimeError("Set GOOGLE_API_KEY for graph extraction.")
            return ChatOpenAI(
                api_key=key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                model=self.role.model,
                temperature=0,
            )
        if self.role.client == "openai":
            key = os.environ.get("OPENAI_API_KEY", "")
            if not key:
                raise RuntimeError("Set OPENAI_API_KEY for graph extraction.")
            return ChatOpenAI(api_key=key, model=self.role.model, temperature=0)
        if self.role.client == "openrouter":
            key = os.environ.get("OPENROUTER_API_KEY", "")
            if not key:
                raise RuntimeError("Set OPENROUTER_API_KEY for graph extraction.")
            return ChatOpenAI(api_key=key, base_url="https://openrouter.ai/api/v1", model=self.role.model, temperature=0)
        raise ValueError(f"Unsupported graph extraction client: {self.role.client}")

    async def transform(self, text: str, parent_id: str) -> Any:
        if self._transformer is None:
            from langchain_experimental.graph_transformers import LLMGraphTransformer

            self._transformer = LLMGraphTransformer(
                llm=self._llm(),
                node_properties=True,
                relationship_properties=True,
                additional_instructions=(
                    "Extract only entities and relationships explicitly supported by the source text. "
                    "Use stable, concise entity names and do not infer outside facts."
                ),
            )
        graph_documents = await self._transformer.aconvert_to_graph_documents(
            [Document(page_content=text, metadata={"parent_id": parent_id})]
        )
        if len(graph_documents) != 1:
            raise RuntimeError(f"Expected one graph document for parent {parent_id}, received {len(graph_documents)}.")
        return graph_documents[0]


class ExecutorGraphTransformer:
    def __init__(self, executor: ModelExecutor):
        self.executor = executor

    async def transform(self, text: str, parent_id: str) -> Any:
        result = await self.executor.aexecute_json(
            ModelRequest(
                role=GRAPH_EXTRACTION_ROLE,
                prompt=(
                    "Extract only entities and relationships explicitly supported by this parent chunk. "
                    "Use stable concise entity IDs. Every relationship endpoint must name an extracted node.\n\n"
                    f"Parent ID: {parent_id}\n\n{text}"
                ),
                system_instruction="Return a source-grounded property graph only.",
                response_schema=GRAPH_SCHEMA,
                max_output_tokens=4096,
                cache_namespace="graph-extraction",
                idempotency_key=parent_id,
            )
        )
        payload = result.payload or {}
        nodes = {
            str(raw["id"]): SimpleNamespace(
                id=str(raw["id"]),
                type=str(raw["type"] or "Entity"),
                properties=dict(raw.get("properties") or {}),
            )
            for raw in payload.get("nodes") or []
            if str(raw.get("id") or "").strip()
        }
        relationships = []
        for raw in payload.get("relationships") or []:
            source_id = str(raw.get("source_id") or "")
            target_id = str(raw.get("target_id") or "")
            if source_id not in nodes or target_id not in nodes or source_id == target_id:
                continue
            relationships.append(
                SimpleNamespace(
                    source=nodes[source_id],
                    target=nodes[target_id],
                    type=str(raw.get("type") or "RELATED_TO"),
                    properties=dict(raw.get("properties") or {}),
                )
            )
        return SimpleNamespace(nodes=list(nodes.values()), relationships=relationships)


@dataclass
class GraphExtractionWorker:
    store: Any
    transformer: GraphTransformer
    capacity_guard: Callable[[int, int], None] | None = None
    max_concurrency: int = 4

    @classmethod
    def from_routing_config(
        cls,
        store: Any,
        config_path: str | None,
        capacity_guard: Callable[[int, int], None] | None = None,
        *,
        cache_path: str | None = None,
        metrics_path: str | None = None,
        max_concurrency: int = 4,
    ) -> "GraphExtractionWorker":
        role = resolve_prompt_role(
            config_path,
            GRAPH_EXTRACTION_ROLE,
            default_client="genai",
            default_model="gemini-2.5-flash",
        )
        client = build_single_prompt_clients(role.client)[role.client]
        executor = ModelExecutor(
            {GRAPH_EXTRACTION_ROLE: RoutedJsonAdapter(role, client)},
            {GRAPH_EXTRACTION_ROLE: GRAPH_EXTRACTION_ROLE},
            policies={
                GRAPH_EXTRACTION_ROLE: ExecutionPolicy(
                    max_concurrency=max_concurrency,
                    max_attempts=2,
                )
            },
            cache_path=cache_path,
            metrics_path=metrics_path,
        )
        return cls(store, ExecutorGraphTransformer(executor), capacity_guard, max_concurrency=max_concurrency)

    async def run_batch(self, limit: int = 100) -> dict[str, int]:
        parents = self.store.pending_graph_parents(limit)
        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def process(parent: dict[str, Any]) -> str:
            async with semaphore:
                try:
                    graph_document = await self.transformer.transform(parent["text"], parent["parent_id"])
                    if self.capacity_guard is not None:
                        self.capacity_guard(len(graph_document.nodes), len(graph_document.relationships))
                    self.store.persist_parent_graph(
                        parent["parent_id"],
                        parent["child_ids"],
                        graph_document,
                        revision_id=parent.get("revision_id"),
                    )
                    return "completed"
                except GraphCapacityError:
                    raise
                except Exception as exc:
                    self.store.fail_parent_graph(parent["parent_id"], str(exc))
                    return "failed"

        if self.max_concurrency <= 0:
            raise ValueError("max_concurrency must be positive.")
        tasks = [asyncio.create_task(process(parent)) for parent in parents]
        try:
            outcomes = await asyncio.gather(*tasks)
        except GraphCapacityError:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        finalized = self.store.finalize_graph_revisions()
        return {
            "requested": len(parents),
            "completed": outcomes.count("completed"),
            "failed": outcomes.count("failed"),
            "revisions_finalized": finalized,
        }
