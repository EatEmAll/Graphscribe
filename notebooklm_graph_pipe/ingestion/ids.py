from __future__ import annotations

import hashlib
import uuid
from pathlib import Path


REPOSITORY_NAMESPACE = uuid.UUID("de164e30-d041-4de2-8886-c1af314c7676")


def normalize_corpus_key(value: str) -> str:
    return "-".join(part for part in value.strip().lower().replace("_", "-").split("-") if part)


def corpus_id(corpus_key: str) -> str:
    normalized = normalize_corpus_key(corpus_key)
    if not normalized:
        raise ValueError("Corpus key cannot be empty.")
    return str(uuid.uuid5(REPOSITORY_NAMESPACE, normalized))


def canonical_file_identity(path: Path, corpus_root: Path) -> str:
    resolved_path = path.resolve()
    resolved_root = corpus_root.resolve()
    try:
        relative = resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"Source path is outside corpus root: {resolved_path}") from exc
    return relative.as_posix()


def canonical_youtube_identity(video_id: str) -> str:
    if len(video_id) != 11:
        raise ValueError(f"Invalid YouTube video id: {video_id}")
    return f"https://www.youtube.com/watch?v={video_id}"


def document_id(corpus_identifier: str, source_identity: str) -> str:
    return str(uuid.uuid5(uuid.UUID(corpus_identifier), source_identity))


def ledger_source_id(corpus_identifier: str, provider: str, provider_source_id: str) -> str:
    identity = f"{provider.strip().lower()}:{provider_source_id.strip()}"
    if not provider.strip() or not provider_source_id.strip():
        raise ValueError("Ledger provider and provider source ID cannot be empty.")
    return str(uuid.uuid5(uuid.UUID(corpus_identifier), identity))


def sha256_text(*parts: object) -> str:
    payload = "\x1f".join(str(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def revision_id(document_identifier: str, checksum: str, extractor_version: str) -> str:
    return sha256_text(document_identifier, checksum, extractor_version)


def block_id(revision_identifier: str, ordinal: int, text: str) -> str:
    return sha256_text(revision_identifier, ordinal, normalize_identity_text(text))


def parent_id(revision_identifier: str, ordinal: int, text: str, chunker_version: str) -> str:
    return sha256_text(revision_identifier, chunker_version, ordinal, normalize_identity_text(text))


def child_id(parent_identifier: str, ordinal: int, text: str, chunker_version: str) -> str:
    return sha256_text(parent_identifier, chunker_version, ordinal, normalize_identity_text(text))


def normalize_identity_text(text: str) -> str:
    return " ".join(text.replace("\r\n", "\n").replace("\r", "\n").split())
