from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from neo4j import GraphDatabase

from notebooklm_graph_pipe.ingestion.chunking import load_minilm_tokenizer
from notebooklm_graph_pipe.ingestion.embeddings import MiniLMEmbedder
from notebooklm_graph_pipe.retrieval.answering import GroundedAnswerer
from notebooklm_graph_pipe.retrieval.community_query import CommunityQueryEngine
from notebooklm_graph_pipe.retrieval.hybrid import CrossEncoderReranker, HybridRetriever
from notebooklm_graph_pipe.retrieval.neo4j_backend import Neo4jRetrievalBackend
from notebooklm_graph_pipe.retrieval.lancedb_store import (
    ExternalVectorCandidateRetriever,
    LanceDBVectorStore,
)
from notebooklm_graph_pipe.runtime.neo4j_connection import (
    ResolvedNeo4jConnection,
    resolve_connection_mapping,
    verify_corpus_connection,
)
from notebooklm_graph_pipe.runtime.llm_json_utils import build_single_prompt_clients
from notebooklm_graph_pipe.runtime.llm_routing import (
    DRIFT_PLANNER_ROLE,
    GLOBAL_MAP_ROLE,
    GLOBAL_REDUCE_ROLE,
    resolve_prompt_role,
)
from notebooklm_graph_pipe.runtime.model_adapters import RoutedJsonAdapter
from notebooklm_graph_pipe.runtime.model_executor import ExecutionPolicy, ModelExecutor

from .registry import CorpusRegistryEntry


@dataclass
class CorpusRuntime:
    driver: Any
    backend: Neo4jRetrievalBackend
    retriever: HybridRetriever
    answerer: GroundedAnswerer | None = None
    community_answerer: CommunityQueryEngine | None = None

    def close(self) -> None:
        self.driver.close()


class RuntimeFactory:
    def __init__(self, llm_routing_config: str | None = None):
        self.llm_routing_config = llm_routing_config
        self._runtimes: dict[str, CorpusRuntime] = {}
        self._signatures: dict[str, tuple[Any, ...]] = {}

    @staticmethod
    def _retrieval_profile(entry: CorpusRegistryEntry) -> tuple[str, str, str]:
        manifest = entry.manifest
        unit = str(getattr(manifest, "retrieval_unit", "chunk"))
        prefix = "parent" if unit == "parent" else "chunk"
        return (
            unit,
            str(getattr(manifest, "retrieval_vector_index", f"{prefix}_embedding_v1")),
            str(getattr(manifest, "retrieval_keyword_index", f"{prefix}_keyword_v1")),
        )

    @staticmethod
    def _signature(entry: CorpusRegistryEntry, connection: ResolvedNeo4jConnection) -> tuple[Any, ...]:
        neo4j = entry.manifest.neo4j
        retrieval_profile = RuntimeFactory._retrieval_profile(entry)
        return (
            entry.manifest.corpus_id,
            connection.uri,
            connection.database,
            connection.username,
            neo4j.get("password_env") or "NEO4J_PASSWORD",
            hashlib.sha256(connection.password.encode("utf-8")).digest(),
            entry.manifest.embedding_provider,
            entry.manifest.embedding_model,
            entry.manifest.embedding_dimension,
            entry.manifest.embedding_normalized,
            *retrieval_profile,
            getattr(entry.manifest, "retrieval_vector_provider", "neo4j"),
            getattr(entry.manifest, "retrieval_vector_location", None),
        )

    def get(self, entry: CorpusRegistryEntry) -> CorpusRuntime:
        connection = resolve_connection_mapping(entry.manifest.neo4j)
        retrieval_unit, vector_index, keyword_index = self._retrieval_profile(entry)
        signature = self._signature(entry, connection)
        if entry.key in self._runtimes and self._signatures.get(entry.key) == signature:
            return self._runtimes[entry.key]
        previous = self._runtimes.pop(entry.key, None)
        self._signatures.pop(entry.key, None)
        if previous is not None:
            previous.close()
        if (
            entry.manifest.embedding_provider != "sentence-transformer"
            or entry.manifest.embedding_model.rsplit("/", 1)[-1] != "all-MiniLM-L6-v2"
            or entry.manifest.embedding_dimension != 384
            or not entry.manifest.embedding_normalized
        ):
            raise ValueError(
                "The corpus embedding metadata is incompatible with the configured retrieval embedder. "
                "Use a blue-green rebuild for embedding changes."
            )
        verify_corpus_connection(
            connection,
            dimension=entry.manifest.embedding_dimension,
            retrieval_unit=retrieval_unit,
            vector_index=vector_index,
            keyword_index=keyword_index,
            require_retrieval_vector=getattr(entry.manifest, "retrieval_vector_provider", "neo4j") == "neo4j",
        )
        driver = GraphDatabase.driver(connection.uri, auth=(connection.username, connection.password))
        backend = Neo4jRetrievalBackend(
            driver,
            connection.database,
            entry.manifest.corpus_id,
            retrieval_unit=retrieval_unit,
            vector_index=vector_index,
            keyword_index=keyword_index,
        )
        embedder = MiniLMEmbedder()
        vector_retriever = None
        if getattr(entry.manifest, "retrieval_vector_provider", "neo4j") == "lancedb":
            if not getattr(entry.manifest, "retrieval_vector_location", None):
                raise ValueError("A LanceDB retrieval provider requires retrieval.vector_location.")
            location = Path(entry.manifest.retrieval_vector_location)
            if not location.is_absolute():
                location = entry.manifest_path.parent / location
            vector_retriever = ExternalVectorCandidateRetriever(
                LanceDBVectorStore(location, dimension=entry.manifest.embedding_dimension),
                entry.manifest.corpus_id,
                embedder.fingerprint,
                backend,
            )
        retriever = HybridRetriever(
            backend,
            embedder,
            CrossEncoderReranker(),
            context_tokenizer=load_minilm_tokenizer(),
            vector_retriever=vector_retriever,
        )
        runtime = CorpusRuntime(driver, backend, retriever)
        self._runtimes[entry.key] = runtime
        self._signatures[entry.key] = signature
        return runtime

    def get_answerer(self, entry: CorpusRegistryEntry) -> GroundedAnswerer:
        runtime = self.get(entry)
        if runtime.answerer is None:
            execution = getattr(entry.manifest, "execution", {}) or {}
            root = entry.manifest_path.parent
            runtime.answerer = GroundedAnswerer.from_routing_config(
                runtime.retriever,
                self.llm_routing_config,
                cache_path=str(root / str(execution.get("cache_path") or ".local/model-cache.sqlite3")),
                metrics_path=str(root / str(execution.get("metrics_path") or ".local/model-metrics.jsonl")),
                max_concurrency=int(execution.get("default_max_concurrency") or 4),
            )
        return runtime.answerer

    def get_community_answerer(self, entry: CorpusRegistryEntry) -> CommunityQueryEngine:
        runtime = self.get(entry)
        if runtime.community_answerer is not None:
            return runtime.community_answerer
        roles = {
            GLOBAL_MAP_ROLE: resolve_prompt_role(
                self.llm_routing_config,
                GLOBAL_MAP_ROLE,
                default_client="genai",
                default_model="gemini-2.5-flash",
            ),
            GLOBAL_REDUCE_ROLE: resolve_prompt_role(
                self.llm_routing_config,
                GLOBAL_REDUCE_ROLE,
                default_client="genai",
                default_model="gemini-2.5-flash",
            ),
            DRIFT_PLANNER_ROLE: resolve_prompt_role(
                self.llm_routing_config,
                DRIFT_PLANNER_ROLE,
                default_client="genai",
                default_model="gemini-2.5-flash",
            ),
        }
        clients = build_single_prompt_clients(*(role.client for role in roles.values()))
        execution = getattr(entry.manifest, "execution", {}) or {}
        default_concurrency = int(execution.get("default_max_concurrency") or 4)
        role_limits = execution.get("role_limits") or {}
        adapters = {
            name: RoutedJsonAdapter(role, clients[role.client])
            for name, role in roles.items()
        }
        executor = ModelExecutor(
            adapters,
            {name: name for name in roles},
            policies={
                name: ExecutionPolicy(max_concurrency=int(role_limits.get(name, default_concurrency)))
                for name in roles
            },
            cache_path=entry.manifest_path.parent / str(execution.get("cache_path") or ".local/model-cache.sqlite3"),
            metrics_path=entry.manifest_path.parent / str(execution.get("metrics_path") or ".local/model-metrics.jsonl"),
        )
        runtime.community_answerer = CommunityQueryEngine(
            runtime.backend,
            runtime.retriever.embedder,
            executor,
            self.get_answerer(entry),
        )
        return runtime.community_answerer

    def close(self) -> None:
        for runtime in self._runtimes.values():
            runtime.close()
        self._runtimes.clear()
        self._signatures.clear()
