from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from neo4j import GraphDatabase

from notebooklm_graph_pipe.ingestion.chunking import load_minilm_tokenizer
from notebooklm_graph_pipe.ingestion.embeddings import MiniLMEmbedder
from notebooklm_graph_pipe.retrieval.answering import GroundedAnswerer
from notebooklm_graph_pipe.retrieval.hybrid import CrossEncoderReranker, HybridRetriever
from notebooklm_graph_pipe.retrieval.neo4j_backend import Neo4jRetrievalBackend

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
    def _signature(entry: CorpusRegistryEntry) -> tuple[Any, ...]:
        neo4j = entry.manifest.neo4j
        return (
            entry.manifest.corpus_id,
            neo4j.get("uri"),
            neo4j.get("database") or "neo4j",
            neo4j.get("username"),
            neo4j.get("password"),
            entry.manifest.embedding_provider,
            entry.manifest.embedding_model,
            entry.manifest.embedding_dimension,
            entry.manifest.embedding_normalized,
        )

    def get(self, entry: CorpusRegistryEntry) -> CorpusRuntime:
        signature = self._signature(entry)
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
        neo4j = entry.manifest.neo4j
        driver = GraphDatabase.driver(neo4j["uri"], auth=(neo4j["username"], neo4j["password"]))
        backend = Neo4jRetrievalBackend(driver, neo4j.get("database") or "neo4j", entry.manifest.corpus_id)
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
