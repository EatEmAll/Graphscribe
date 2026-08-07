from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from notebooklm_graph_pipe.ingestion.models import CanonicalDocument


class SourceReader(Protocol):
    def read(self, source: object, context: object) -> CanonicalDocument: ...


class Chunker(Protocol):
    def chunk(self, document: CanonicalDocument) -> object: ...


class EmbeddingProvider(Protocol):
    @property
    def fingerprint(self) -> str: ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class GraphExtractor(Protocol):
    async def transform(self, text: str, parent_id: str) -> object: ...


class EvidenceStore(Protocol):
    def active_revision_ids(self) -> list[str]: ...


@dataclass(frozen=True)
class VectorRecord:
    record_id: str
    corpus_id: str
    revision_id: str
    document_id: str
    parent_id: str
    chunk_id: str
    unit_type: str
    embedding_fingerprint: str
    vector: tuple[float, ...]
    text: str
    title: str = ""
    source_uri: str = ""
    source_type: str = ""
    language: str = ""


@dataclass(frozen=True)
class VectorQuery:
    corpus_id: str
    active_revision_ids: tuple[str, ...]
    embedding_fingerprint: str
    vector: tuple[float, ...]
    limit: int
    filters: dict[str, Any]


@dataclass(frozen=True)
class VectorHit:
    record_id: str
    score: float
    metadata: dict[str, Any]


class VectorRetriever(Protocol):
    def upsert(self, records: Sequence[VectorRecord]) -> None: ...

    def delete_revisions(self, corpus_id: str, revision_ids: Sequence[str]) -> None: ...

    def query(self, request: VectorQuery) -> list[VectorHit]: ...

    def health(self) -> dict[str, object]: ...


class CommunityBuilder(Protocol):
    def build(self, projection: object, config: object) -> object: ...
