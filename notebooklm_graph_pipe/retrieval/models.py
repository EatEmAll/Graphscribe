from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Candidate:
    chunk_id: str
    document_id: str
    parent_id: str
    text: str
    title: str
    source_uri: str
    page_start: int | None = None
    page_end: int | None = None
    timestamp_start_ms: int | None = None
    timestamp_end_ms: int | None = None
    section_path: tuple[str, ...] = ()
    channels: set[str] = field(default_factory=set)
    channel_ranks: dict[str, int] = field(default_factory=dict)
    rrf_score: float = 0.0
    reranker_score: float | None = None
    graph_paths: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class ContextItem:
    citation_id: str
    parent_id: str
    document_id: str
    title: str
    source_uri: str
    text: str
    page_start: int | None = None
    page_end: int | None = None
    timestamp_start_ms: int | None = None
    timestamp_end_ms: int | None = None
    section_path: tuple[str, ...] = ()
    matched_chunk_ids: tuple[str, ...] = ()

    def citation_payload(self) -> dict[str, Any]:
        return {
            "id": self.citation_id,
            "parent_id": self.parent_id,
            "document_id": self.document_id,
            "title": self.title,
            "source_uri": self.source_uri,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "timestamp_start_ms": self.timestamp_start_ms,
            "timestamp_end_ms": self.timestamp_end_ms,
            "section_path": list(self.section_path),
            "quote_preview": self.text[:280],
        }
