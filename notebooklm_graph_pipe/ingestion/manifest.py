from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


MANIFEST_VERSION = 3


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
    version: int = MANIFEST_VERSION

    def to_dict(self) -> dict[str, Any]:
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
            "neo4j": self.neo4j,
            "sources": {key: asdict(value) for key, value in sorted(self.sources.items())},
            "removed_sources": sorted(self.removed_sources),
            "suppressed_sources": sorted(self.suppressed_sources),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CorpusManifest":
        if int(payload.get("version") or 0) != MANIFEST_VERSION:
            raise ValueError(f"Expected manifest version {MANIFEST_VERSION}.")
        corpus = payload.get("corpus") or {}
        embedding = payload.get("embedding") or {}
        return cls(
            corpus_id=str(corpus["id"]),
            corpus_key=str(corpus["key"]),
            title=str(corpus["title"]),
            neo4j=dict(payload.get("neo4j") or {}),
            dataset_root=str(payload.get("dataset_root") or "") or None,
            sources={key: SourceManifestEntry(**value) for key, value in (payload.get("sources") or {}).items()},
            removed_sources=list(payload.get("removed_sources") or []),
            suppressed_sources=list(payload.get("suppressed_sources") or []),
            embedding_provider=str(embedding.get("provider") or "sentence-transformer"),
            embedding_model=str(embedding.get("model") or "all-MiniLM-L6-v2"),
            embedding_dimension=int(embedding.get("dimension") or 384),
            embedding_normalized=bool(embedding.get("normalized", True)),
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
