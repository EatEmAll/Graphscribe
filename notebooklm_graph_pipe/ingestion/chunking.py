from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Protocol, Sequence

from .ids import child_id, parent_id
from .models import CanonicalBlock, CanonicalDocument, ChildChunk, ParentChunk


class Tokenizer(Protocol):
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]: ...

    def decode(self, token_ids: Sequence[int], *, skip_special_tokens: bool = True) -> str: ...


@dataclass(frozen=True)
class ChunkingConfig:
    child_target_tokens: int = 192
    child_max_tokens: int = 220
    child_overlap_tokens: int = 24
    child_min_tokens: int = 32
    parent_target_tokens: int = 1200
    parent_max_tokens: int = 1600
    chunker_version: str = "hierarchical-v1"

    def __post_init__(self) -> None:
        if not 0 <= self.child_overlap_tokens < self.child_target_tokens <= self.child_max_tokens:
            raise ValueError("Invalid child chunk size configuration.")
        if not 0 < self.parent_target_tokens <= self.parent_max_tokens:
            raise ValueError("Invalid parent chunk size configuration.")


@dataclass(frozen=True)
class ChunkingResult:
    parents: tuple[ParentChunk, ...]
    children: tuple[ChildChunk, ...]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class _Piece:
    text: str
    token_count: int
    block: CanonicalBlock


class HierarchicalChunker:
    def __init__(self, tokenizer: Tokenizer, config: ChunkingConfig | None = None):
        self.tokenizer = tokenizer
        self.config = config or ChunkingConfig()

    def _token_ids(self, text: str) -> list[int]:
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def _split_block(self, block: CanonicalBlock) -> list[_Piece]:
        token_ids = self._token_ids(block.text)
        if len(token_ids) <= self.config.child_max_tokens:
            return [_Piece(block.text.strip(), len(token_ids), block)]

        if block.block_type == "table":
            table_pieces = self._split_table(block)
            if table_pieces:
                return table_pieces

        overlap = 0 if block.block_type in {"table", "code"} else self.config.child_overlap_tokens
        stride = self.config.child_target_tokens - overlap
        pieces: list[_Piece] = []
        for start in range(0, len(token_ids), stride):
            segment_ids = token_ids[start : start + self.config.child_target_tokens]
            if not segment_ids:
                break
            text = self.tokenizer.decode(segment_ids, skip_special_tokens=True).strip()
            if text:
                pieces.append(_Piece(text, len(segment_ids), block))
            if start + self.config.child_target_tokens >= len(token_ids):
                break
        return pieces

    def _split_table(self, block: CanonicalBlock) -> list[_Piece]:
        lines = [line for line in block.text.splitlines() if line.strip()]
        if len(lines) < 3:
            return []
        header = lines[:2]
        header_text = "\n".join(header)
        header_tokens = len(self._token_ids(header_text))
        if header_tokens >= self.config.child_max_tokens:
            return []
        pieces: list[_Piece] = []
        rows: list[str] = []
        for row in lines[2:]:
            candidate = "\n".join([*header, *rows, row])
            if rows and len(self._token_ids(candidate)) > self.config.child_target_tokens:
                text = "\n".join([*header, *rows])
                pieces.append(_Piece(text, len(self._token_ids(text)), block))
                rows = []
            row_candidate = "\n".join([*header, row])
            if len(self._token_ids(row_candidate)) > self.config.child_max_tokens:
                return []
            rows.append(row)
        if rows:
            text = "\n".join([*header, *rows])
            pieces.append(_Piece(text, len(self._token_ids(text)), block))
        return pieces

    def _build_parent_groups(self, blocks: tuple[CanonicalBlock, ...]) -> list[list[CanonicalBlock]]:
        groups: list[list[CanonicalBlock]] = []
        current: list[CanonicalBlock] = []
        current_tokens = 0
        current_section: tuple[str, ...] | None = None
        for block in blocks:
            block_tokens = len(self._token_ids(block.text))
            if block_tokens > self.config.parent_max_tokens:
                if current:
                    groups.append(current)
                    current = []
                    current_tokens = 0
                token_ids = self._token_ids(block.text)
                for start in range(0, len(token_ids), self.config.parent_target_tokens):
                    segment = self.tokenizer.decode(
                        token_ids[start : start + self.config.parent_target_tokens],
                        skip_special_tokens=True,
                    ).strip()
                    if segment:
                        groups.append([replace(block, text=segment)])
                current_section = block.section_path
                continue
            section_changed = current and block.section_path != current_section and block.block_type == "heading"
            exceeds_target = current and current_tokens + block_tokens > self.config.parent_target_tokens
            exceeds_max = current and current_tokens + block_tokens > self.config.parent_max_tokens
            if section_changed or exceeds_max or (exceeds_target and block.block_type in {"heading", "page_break"}):
                groups.append(current)
                current = []
                current_tokens = 0
            current.append(block)
            current_tokens += block_tokens
            current_section = block.section_path
        if current:
            groups.append(current)
        return groups

    def chunk(self, document: CanonicalDocument) -> ChunkingResult:
        parents: list[ParentChunk] = []
        children: list[ChildChunk] = []
        warnings: list[str] = []
        child_position = 0
        for parent_position, blocks in enumerate(self._build_parent_groups(document.blocks)):
            parent_text = "\n\n".join(block.text for block in blocks if block.text.strip()).strip()
            parent_tokens = len(self._token_ids(parent_text))
            if parent_tokens > self.config.parent_max_tokens:
                raise ValueError(f"Parent chunk exceeds configured token maximum: {parent_tokens}")
            parent_identifier = parent_id(
                document.revision_id,
                parent_position,
                parent_text,
                self.config.chunker_version,
            )
            section_path = next((block.section_path for block in reversed(blocks) if block.section_path), ())
            pages = [block.page_number for block in blocks if block.page_number is not None]
            starts = [block.timestamp_start_ms for block in blocks if block.timestamp_start_ms is not None]
            ends = [block.timestamp_end_ms for block in blocks if block.timestamp_end_ms is not None]
            parents.append(
                ParentChunk(
                    id=parent_identifier,
                    revision_id=document.revision_id,
                    document_id=document.document_id,
                    position=parent_position,
                    text=parent_text,
                    token_count=parent_tokens,
                    section_path=section_path,
                    block_ids=tuple(block.block_id for block in blocks),
                    page_start=min(pages) if pages else None,
                    page_end=max(pages) if pages else None,
                    timestamp_start_ms=min(starts) if starts else None,
                    timestamp_end_ms=max(ends) if ends else None,
                )
            )

            pending: list[_Piece] = []
            pending_tokens = 0

            def flush() -> None:
                nonlocal child_position, pending, pending_tokens
                if not pending:
                    return
                text = "\n\n".join(piece.text for piece in pending).strip()
                actual_tokens = len(self._token_ids(text))
                if actual_tokens > self.config.child_max_tokens:
                    raise ValueError(f"Child chunk exceeds embedding token limit: {actual_tokens}")
                block_ids = tuple(dict.fromkeys(piece.block.block_id for piece in pending))
                child_identifier = child_id(
                    parent_identifier,
                    child_position,
                    text,
                    self.config.chunker_version,
                )
                chunk_pages = [piece.block.page_number for piece in pending if piece.block.page_number is not None]
                chunk_starts = [piece.block.timestamp_start_ms for piece in pending if piece.block.timestamp_start_ms is not None]
                chunk_ends = [piece.block.timestamp_end_ms for piece in pending if piece.block.timestamp_end_ms is not None]
                children.append(
                    ChildChunk(
                        id=child_identifier,
                        parent_id=parent_identifier,
                        revision_id=document.revision_id,
                        document_id=document.document_id,
                        position=child_position,
                        text=text,
                        token_count=actual_tokens,
                        section_path=pending[-1].block.section_path,
                        block_ids=block_ids,
                        page_start=min(chunk_pages) if chunk_pages else None,
                        page_end=max(chunk_pages) if chunk_pages else None,
                        timestamp_start_ms=min(chunk_starts) if chunk_starts else None,
                        timestamp_end_ms=max(chunk_ends) if chunk_ends else None,
                    )
                )
                child_position += 1
                pending = []
                pending_tokens = 0

            for block in blocks:
                for piece in self._split_block(block):
                    if pending and pending_tokens + piece.token_count > self.config.child_target_tokens:
                        flush()
                    pending.append(piece)
                    pending_tokens += piece.token_count
                    if pending_tokens >= self.config.child_target_tokens:
                        flush()
            flush()

            if children and children[-1].parent_id == parent_identifier and children[-1].token_count < self.config.child_min_tokens:
                warnings.append(
                    f"Child {children[-1].position} is below the useful token target: {children[-1].token_count}"
                )

        if not children:
            raise ValueError(f"Chunking produced no child chunks for {document.source_uri}")
        return ChunkingResult(tuple(parents), tuple(children), tuple(warnings))


def load_minilm_tokenizer(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> Tokenizer:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_name)
