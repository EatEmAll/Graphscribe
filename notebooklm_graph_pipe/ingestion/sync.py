from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .adapters import (
    DEFAULT_ADAPTERS,
    ExtractionContext,
    SourceAdapter,
    YoutubeSource,
    adapter_for,
    file_checksum,
    youtube_video_id,
)
from .chunking import HierarchicalChunker
from .embeddings import MiniLMEmbedder
from .ids import canonical_file_identity, canonical_youtube_identity
from .manifest import CorpusManifest, SourceManifestEntry, save_manifest
from .models import CanonicalDocument
from .neo4j_store import Neo4jCorpusStore


SUPPORTED_SUFFIXES = {".txt", ".md", ".markdown", ".pdf"}
EXCLUDED_DIRS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", "normalized"}


@dataclass
class SyncEvent:
    source: str
    status: str
    message: str = ""


@dataclass
class SyncReport:
    started_at: str
    completed_at: str | None = None
    unchanged: int = 0
    added: int = 0
    updated: int = 0
    removed: int = 0
    failed: int = 0
    parent_chunks: int = 0
    child_chunks: int = 0
    events: list[SyncEvent] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.__dict__,
            "events": [event.__dict__ for event in self.events],
        }


def discover_local_sources(root: Path) -> list[Path]:
    resolved = root.resolve()
    sources: list[Path] = []
    for path in sorted(resolved.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        relative = path.relative_to(resolved)
        if any(part.startswith(".") or part in EXCLUDED_DIRS for part in relative.parts[:-1]):
            continue
        if path.name.startswith(".") or path.name.startswith("~$") or path.stat().st_size == 0:
            continue
        sources.append(path)
    return sources


def load_youtube_sources(root: Path) -> list[YoutubeSource]:
    path = root / "sources.yaml"
    if not path.exists():
        return []
    import yaml

    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if int(payload.get("version") or 0) != 1:
        raise ValueError("sources.yaml must declare version: 1")
    result: list[YoutubeSource] = []
    for item in payload.get("youtube") or []:
        result.append(
            YoutubeSource(
                url=str(item["url"]),
                title=str(item.get("title") or "") or None,
                preferred_languages=tuple(item.get("preferred_languages") or ("en",)),
            )
        )
    return result


class CorpusSynchronizer:
    def __init__(
        self,
        *,
        store: Neo4jCorpusStore,
        embedder: MiniLMEmbedder,
        chunker: HierarchicalChunker,
        adapters: Iterable[SourceAdapter] = DEFAULT_ADAPTERS,
    ):
        self.store = store
        self.embedder = embedder
        self.chunker = chunker
        self.adapters = tuple(adapters)

    def _write_artifacts(self, artifact_root: Path, document: CanonicalDocument) -> None:
        revision_root = artifact_root / document.document_id / document.revision_id
        revision_root.mkdir(parents=True, exist_ok=True)
        (revision_root / "document.json").write_text(
            json.dumps(document.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (revision_root / "document.md").write_text(document.text + "\n", encoding="utf-8")

    def sync(
        self,
        *,
        corpus_root: Path,
        manifest: CorpusManifest,
        manifest_path: Path,
        artifact_root: Path,
    ) -> SyncReport:
        started = datetime.now(timezone.utc).isoformat()
        report = SyncReport(started_at=started)
        context = ExtractionContext(manifest.corpus_id, corpus_root)
        manifest_model = manifest.embedding_model.rsplit("/", 1)[-1]
        configured_model = self.embedder.config.model.rsplit("/", 1)[-1]
        if (
            manifest.embedding_provider != "sentence-transformer"
            or manifest_model != configured_model
            or manifest.embedding_dimension != self.embedder.config.dimension
            or manifest.embedding_normalized != self.embedder.config.normalize
        ):
            raise ValueError(
                "Manifest embedding metadata does not match the configured embedder. "
                "Use a blue-green rebuild for embedding changes."
            )
        self.store.ensure_schema(manifest.embedding_dimension)
        self.store.assert_embedding_fingerprint(manifest.corpus_key, self.embedder.fingerprint)
        sources: list[object] = [*discover_local_sources(corpus_root), *load_youtube_sources(corpus_root)]
        current_keys: set[str] = set()

        for source in sources:
            if isinstance(source, Path):
                key = canonical_file_identity(source, corpus_root)
            else:
                key = canonical_youtube_identity(youtube_video_id(source.url))
            if key in manifest.suppressed_sources:
                report.events.append(SyncEvent(key, "suppressed"))
                continue
            current_keys.add(key)
            previous = manifest.sources.get(key)
            document: CanonicalDocument | None = None
            activated = False
            try:
                adapter = adapter_for(source, self.adapters)
                if (
                    isinstance(source, Path)
                    and previous
                    and previous.checksum == file_checksum(source)
                    and previous.extractor == adapter.name
                    and previous.extractor_version == adapter.version
                ):
                    report.unchanged += 1
                    report.events.append(SyncEvent(key, "unchanged"))
                    continue
                document = adapter.extract(source, context)
                if previous and previous.checksum == document.source_checksum and previous.extractor_version == document.extractor_version:
                    report.unchanged += 1
                    report.events.append(SyncEvent(key, "unchanged"))
                    continue
                chunks = self.chunker.chunk(document)
                vectors = self.embedder.embed_documents([chunk.text for chunk in chunks.children])
                self._write_artifacts(artifact_root, document)
                self.store.begin_revision(
                    corpus_key=manifest.corpus_key,
                    corpus_title=manifest.title,
                    embedding_fingerprint=self.embedder.fingerprint,
                    document=document,
                    chunks=chunks,
                    embeddings=vectors,
                )
                self.store.activate_revision(document.document_id, document.revision_id, len(chunks.children))
                activated = True
                manifest.sources[key] = SourceManifestEntry(
                    document_id=document.document_id,
                    active_revision_id=document.revision_id,
                    checksum=document.source_checksum,
                    extractor=document.extractor,
                    extractor_version=document.extractor_version,
                    warnings=list(document.warnings) + list(chunks.warnings),
                )
                if previous:
                    report.updated += 1
                    status = "updated"
                else:
                    report.added += 1
                    status = "added"
                report.parent_chunks += len(chunks.parents)
                report.child_chunks += len(chunks.children)
                report.events.append(SyncEvent(key, status))
                save_manifest(manifest_path, manifest)
            except Exception as exc:
                if activated and document is not None:
                    try:
                        if previous is not None:
                            self.store.restore_revision(document.document_id, previous.active_revision_id)
                        else:
                            self.store.deactivate_document(document.document_id)
                    except Exception as rollback_exc:
                        report.events.append(SyncEvent(key, "rollback_failed", str(rollback_exc)))
                    finally:
                        if previous is not None:
                            manifest.sources[key] = previous
                        else:
                            manifest.sources.pop(key, None)
                if document is not None:
                    try:
                        self.store.fail_revision(document.revision_id, str(exc))
                    except Exception as cleanup_exc:
                        report.events.append(SyncEvent(key, "cleanup_failed", str(cleanup_exc)))
                report.failed += 1
                report.events.append(SyncEvent(key, "failed", str(exc)))

        removed = sorted(set(manifest.sources) - current_keys)
        for key in removed:
            entry = manifest.sources[key]
            self.store.deactivate_document(entry.document_id)
            del manifest.sources[key]
            report.removed += 1
            report.events.append(SyncEvent(key, "removed"))
        manifest.removed_sources = removed
        if report.failed == 0:
            self.store.garbage_collect()
        save_manifest(manifest_path, manifest)
        report.completed_at = datetime.now(timezone.utc).isoformat()
        return report
