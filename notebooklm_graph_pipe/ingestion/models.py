from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


SCHEMA_VERSION = 1
BLOCK_TYPES = {
    "heading",
    "paragraph",
    "list",
    "table",
    "code",
    "quote",
    "transcript",
    "page_break",
}


@dataclass(frozen=True)
class CanonicalBlock:
    block_id: str
    ordinal: int
    block_type: str
    text: str
    section_path: tuple[str, ...] = ()
    page_number: int | None = None
    timestamp_start_ms: int | None = None
    timestamp_end_ms: int | None = None
    source_offset_start: int | None = None
    source_offset_end: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.block_type not in BLOCK_TYPES:
            raise ValueError(f"Unsupported canonical block type: {self.block_type}")
        if self.ordinal < 0:
            raise ValueError("Block ordinal must be non-negative.")
        if not self.text.strip() and self.block_type != "page_break":
            raise ValueError("Canonical blocks must contain text.")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["section_path"] = list(self.section_path)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CanonicalBlock":
        return cls(
            **{
                **payload,
                "section_path": tuple(payload.get("section_path") or ()),
                "metadata": dict(payload.get("metadata") or {}),
            }
        )


@dataclass(frozen=True)
class CanonicalDocument:
    corpus_id: str
    document_id: str
    revision_id: str
    source_type: str
    source_uri: str
    title: str
    source_checksum: str
    extractor: str
    extractor_version: str
    blocks: tuple[CanonicalBlock, ...]
    relative_path: str | None = None
    language: str | None = None
    extracted_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.blocks:
            raise ValueError("Canonical documents must contain at least one block.")
        ordinals = [block.ordinal for block in self.blocks]
        if ordinals != list(range(len(self.blocks))):
            raise ValueError("Canonical block ordinals must be contiguous and ordered.")

    @property
    def text(self) -> str:
        return "\n\n".join(block.text for block in self.blocks if block.text.strip())

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["blocks"] = [block.to_dict() for block in self.blocks]
        payload["warnings"] = list(self.warnings)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CanonicalDocument":
        return cls(
            **{
                **payload,
                "blocks": tuple(CanonicalBlock.from_dict(item) for item in payload["blocks"]),
                "warnings": tuple(payload.get("warnings") or ()),
                "metadata": dict(payload.get("metadata") or {}),
            }
        )


@dataclass(frozen=True)
class ParentChunk:
    id: str
    revision_id: str
    document_id: str
    position: int
    text: str
    token_count: int
    section_path: tuple[str, ...]
    block_ids: tuple[str, ...]
    page_start: int | None = None
    page_end: int | None = None
    timestamp_start_ms: int | None = None
    timestamp_end_ms: int | None = None


@dataclass(frozen=True)
class ChildChunk:
    id: str
    parent_id: str
    revision_id: str
    document_id: str
    position: int
    text: str
    token_count: int
    section_path: tuple[str, ...]
    block_ids: tuple[str, ...]
    page_start: int | None = None
    page_end: int | None = None
    timestamp_start_ms: int | None = None
    timestamp_end_ms: int | None = None
