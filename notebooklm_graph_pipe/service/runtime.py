from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from neo4j import GraphDatabase

from notebooklm_graph_pipe.ingestion.chunking import load_minilm_tokenizer
from notebooklm_graph_pipe.ingestion.embeddings import MiniLMEmbedder
from notebooklm_graph_pipe.retrieval.answering import GroundedAnswerer
from notebooklm_graph_pipe.retrieval.hybrid import CrossEncoderReranker, HybridRetriever
from notebooklm_graph_pipe.retrieval.neo4j_backend import Neo4jRetrievalBackend
from notebooklm_graph_pipe.runtime.neo4j_connection import (
    ResolvedNeo4jConnection,
    resolve_connection_mapping,
    verify_corpus_connection,
)

from .registry import CorpusRegistryEntry


@dataclass
class CorpusRuntime:
    driver: Any
    backend: Neo4jRetrievalBackend
    retriever: HybridRetriever
    answerer: GroundedAnswerer | None = None

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
        retriever = HybridRetriever(
            backend,
            MiniLMEmbedder(),
            CrossEncoderReranker(),
            context_tokenizer=load_minilm_tokenizer(),
        )
        runtime = CorpusRuntime(driver, backend, retriever)
        self._runtimes[entry.key] = runtime
        self._signatures[entry.key] = signature
        return runtime

    def get_answerer(self, entry: CorpusRegistryEntry) -> GroundedAnswerer:
        runtime = self.get(entry)
        if runtime.answerer is None:
            runtime.answerer = GroundedAnswerer.from_routing_config(runtime.retriever, self.llm_routing_config)
        return runtime.answerer

    def close(self) -> None:
        for runtime in self._runtimes.values():
            runtime.close()
        self._runtimes.clear()
        self._signatures.clear()
