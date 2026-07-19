from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from notebooklm_graph_pipe.runtime.llm_routing import (
    GRAPH_EXTRACTION_ROLE,
    PromptRoleConfig,
    resolve_prompt_role,
)


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


@dataclass
class GraphExtractionWorker:
    store: Any
    transformer: GraphTransformer
    capacity_guard: Callable[[int, int], None] | None = None

    @classmethod
    def from_routing_config(
        cls,
        store: Any,
        config_path: str | None,
        capacity_guard: Callable[[int, int], None] | None = None,
    ) -> "GraphExtractionWorker":
        role = resolve_prompt_role(
            config_path,
            GRAPH_EXTRACTION_ROLE,
            default_client="genai",
            default_model="gemini-2.5-flash",
        )
        return cls(store, LangChainGraphTransformer(role), capacity_guard)

    async def run_batch(self, limit: int = 100) -> dict[str, int]:
        completed = 0
        failed = 0
        parents = self.store.pending_graph_parents(limit)
        for parent in parents:
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
                completed += 1
            except GraphCapacityError:
                raise
            except Exception as exc:
                self.store.fail_parent_graph(parent["parent_id"], str(exc))
                failed += 1
        finalized = self.store.finalize_graph_revisions()
        return {"requested": len(parents), "completed": completed, "failed": failed, "revisions_finalized": finalized}
