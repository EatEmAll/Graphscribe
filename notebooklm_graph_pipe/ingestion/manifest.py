from __future__ import annotations

import json
import warnings
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


MANIFEST_VERSION = 4
LEGACY_MANIFEST_VERSIONS = {3}
RETRIEVAL_UNITS = {"chunk", "parent"}

DEFAULT_EXECUTION_CONFIG: dict[str, Any] = {
    "cache_path": ".local/model-cache.sqlite3",
    "metrics_path": ".local/model-metrics.jsonl",
    "default_max_concurrency": 4,
    "role_limits": {},
}
DEFAULT_GRAPH_CONFIG: dict[str, Any] = {
    "extraction_mode": "llm",
    "active_prompt_hash": None,
    "claims_enabled": False,
    "include_provisional_by_default": False,
}
DEFAULT_COMMUNITY_CONFIG: dict[str, Any] = {
    "enabled": False,
    "algorithm": "hierarchical_leiden",
    "relationship_weighting": "active_parent_count",
    "max_cluster_size": 10,
    "seed": 42,
    "report_embedding_enabled": True,
}


@dataclass
class SourceManifestEntry:
    document_id: str
    active_revision_id: str
    checksum: str
    extractor: str
    extractor_version: str
    status: str = "ready"
    warnings: list[str] = field(default_factory=list)


@dataclass
class CorpusManifest:
    corpus_id: str
    corpus_key: str
    title: str
    neo4j: dict[str, Any]
    dataset_root: str | None = None
    sources: dict[str, SourceManifestEntry] = field(default_factory=dict)
    removed_sources: list[str] = field(default_factory=list)
    suppressed_sources: list[str] = field(default_factory=list)
    embedding_provider: str = "sentence-transformer"
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    embedding_normalized: bool = True
    retrieval_unit: str = "chunk"
    retrieval_vector_provider: str = "neo4j"
    retrieval_vector_location: str | None = None
    retrieval_vector_index: str = "chunk_embedding_v1"
    retrieval_keyword_index: str = "chunk_keyword_v1"
    execution: dict[str, Any] = field(default_factory=lambda: deepcopy(DEFAULT_EXECUTION_CONFIG))
    graph: dict[str, Any] = field(default_factory=lambda: deepcopy(DEFAULT_GRAPH_CONFIG))
    community: dict[str, Any] = field(default_factory=lambda: deepcopy(DEFAULT_COMMUNITY_CONFIG))
    version: int = MANIFEST_VERSION

    def __post_init__(self) -> None:
        self.execution = {**deepcopy(DEFAULT_EXECUTION_CONFIG), **dict(self.execution or {})}
        self.graph = {**deepcopy(DEFAULT_GRAPH_CONFIG), **dict(self.graph or {})}
        self.community = {**deepcopy(DEFAULT_COMMUNITY_CONFIG), **dict(self.community or {})}
        self.version = MANIFEST_VERSION
        if self.retrieval_unit not in RETRIEVAL_UNITS:
            raise ValueError(f"Unsupported retrieval unit: {self.retrieval_unit}")
        if self.retrieval_vector_provider not in {"neo4j", "lancedb"}:
            raise ValueError(f"Unsupported vector provider: {self.retrieval_vector_provider}")
        if int(self.execution.get("default_max_concurrency") or 0) <= 0:
            raise ValueError("execution.default_max_concurrency must be positive.")
        if str(self.graph.get("extraction_mode") or "") not in {"llm", "provisional"}:
            raise ValueError("graph.extraction_mode must be 'llm' or 'provisional'.")
        if str(self.community.get("algorithm") or "") != "hierarchical_leiden":
            raise ValueError("community.algorithm must be 'hierarchical_leiden'.")
        if int(self.community.get("max_cluster_size") or 0) <= 0:
            raise ValueError("community.max_cluster_size must be positive.")

    def to_dict(self) -> dict[str, Any]:
        neo4j = {key: value for key, value in self.neo4j.items() if key != "password"}
        neo4j.setdefault("password_env", "NEO4J_PASSWORD")
        neo4j.setdefault("deployment", "managed-local" if neo4j.get("container_name") else "external")
        retrieval = {
            "unit": self.retrieval_unit,
            "vector_index": self.retrieval_vector_index,
            "keyword_index": self.retrieval_keyword_index,
        }
        if self.retrieval_vector_provider != "neo4j" or self.retrieval_vector_location is not None:
            retrieval["vector_provider"] = self.retrieval_vector_provider
            retrieval["vector_location"] = self.retrieval_vector_location
        return {
            "version": self.version,
            "corpus": {"id": self.corpus_id, "key": self.corpus_key, "title": self.title},
            "dataset_root": self.dataset_root,
            "embedding": {
                "provider": self.embedding_provider,
                "model": self.embedding_model,
                "dimension": self.embedding_dimension,
                "normalized": self.embedding_normalized,
            },
            "neo4j": neo4j,
            "retrieval": retrieval,
            "execution": self.execution,
            "graph": self.graph,
            "community": self.community,
            "sources": {key: asdict(value) for key, value in sorted(self.sources.items())},
            "removed_sources": sorted(self.removed_sources),
            "suppressed_sources": sorted(self.suppressed_sources),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CorpusManifest":
        version = int(payload.get("version") or 0)
        if version not in {MANIFEST_VERSION, *LEGACY_MANIFEST_VERSIONS}:
            raise ValueError(
                f"Expected manifest version {MANIFEST_VERSION} or a supported legacy version."
            )
        corpus = payload.get("corpus") or {}
        embedding = payload.get("embedding") or {}
        neo4j = dict(payload.get("neo4j") or {})
        retrieval = dict(payload.get("retrieval") or {})
        retrieval_unit = str(retrieval.get("unit") or "chunk")
        default_prefix = "parent" if retrieval_unit == "parent" else "chunk"
        if "password" in neo4j:
            warnings.warn(
                "Legacy plaintext Neo4j password found in corpus manifest; set password_env and resave the manifest.",
                FutureWarning,
                stacklevel=2,
            )
        neo4j.setdefault("password_env", "NEO4J_PASSWORD")
        neo4j.setdefault("deployment", "managed-local" if neo4j.get("container_name") else "external")
        return cls(
            corpus_id=str(corpus["id"]),
            corpus_key=str(corpus["key"]),
            title=str(corpus["title"]),
            neo4j=neo4j,
            dataset_root=str(payload.get("dataset_root") or "") or None,
            sources={key: SourceManifestEntry(**value) for key, value in (payload.get("sources") or {}).items()},
            removed_sources=list(payload.get("removed_sources") or []),
            suppressed_sources=list(payload.get("suppressed_sources") or []),
            embedding_provider=str(embedding.get("provider") or "sentence-transformer"),
            embedding_model=str(embedding.get("model") or "all-MiniLM-L6-v2"),
            embedding_dimension=int(embedding.get("dimension") or 384),
            embedding_normalized=bool(embedding.get("normalized", True)),
            retrieval_unit=retrieval_unit,
            retrieval_vector_provider=str(retrieval.get("vector_provider") or "neo4j"),
            retrieval_vector_location=str(retrieval.get("vector_location") or "") or None,
            retrieval_vector_index=str(retrieval.get("vector_index") or f"{default_prefix}_embedding_v1"),
            retrieval_keyword_index=str(retrieval.get("keyword_index") or f"{default_prefix}_keyword_v1"),
            execution={**deepcopy(DEFAULT_EXECUTION_CONFIG), **dict(payload.get("execution") or {})},
            graph={**deepcopy(DEFAULT_GRAPH_CONFIG), **dict(payload.get("graph") or {})},
            community={**deepcopy(DEFAULT_COMMUNITY_CONFIG), **dict(payload.get("community") or {})},
            version=MANIFEST_VERSION,
        )


def load_manifest(path: Path) -> CorpusManifest | None:
    if not path.exists():
        return None
    return CorpusManifest.from_dict(json.loads(path.read_text(encoding="utf-8")))


def save_manifest(path: Path, manifest: CorpusManifest) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)
