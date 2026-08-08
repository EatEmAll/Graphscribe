from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Iterable

from .adapters import DEFAULT_ADAPTERS, ExtractionContext, SourceAdapter, adapter_for
from .chunking import HierarchicalChunker
from .embeddings import MiniLMEmbedder, weighted_parent_embedding
from .ids import block_id, canonical_file_identity, revision_id
from .manifest import CorpusManifest, SourceManifestEntry, save_manifest
from .neo4j_store import Neo4jCorpusStore
from .source_ledger import SourceIdentity, identity_from_document


@dataclass
class CompactUpdateEvent:
    source: str
    status: str
    revision_id: str | None = None
    previous_revision_id: str | None = None
    parent_chunks: int = 0
    transient_child_chunks: int = 0
    message: str = ""


@dataclass
class CompactUpdateReport:
    added: int = 0
    updated: int = 0
    unchanged: int = 0
    conflicted: int = 0
    legacy_only: int = 0
    failed: int = 0
    revision_ids: list[str] = field(default_factory=list)
    events: list[CompactUpdateEvent] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "added": self.added,
            "updated": self.updated,
            "unchanged": self.unchanged,
            "conflicted": self.conflicted,
            "legacy_only": self.legacy_only,
            "failed": self.failed,
            "revision_ids": self.revision_ids,
            "events": [asdict(event) for event in self.events],
        }


class CompactCorpusUpdater:
    """Add explicit sources to a parent-retrieval corpus without deleting absent sources."""

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

    @staticmethod
    def _validate_profile(manifest: CorpusManifest) -> None:
        expected = ("parent", "parent_embedding_v1", "parent_keyword_v1")
        actual = (
            manifest.retrieval_unit,
            manifest.retrieval_vector_index,
            manifest.retrieval_keyword_index,
        )
        if actual != expected:
            raise ValueError(f"Compact updates require retrieval profile {expected}; received {actual}.")

    def update(
        self,
        *,
        sources: Iterable[object],
        corpus_root: Path,
        manifest: CorpusManifest,
        manifest_path: Path,
        capacity_guard: Callable[[int], None] | None = None,
        force_refresh: bool = False,
    ) -> CompactUpdateReport:
        self._validate_profile(manifest)
        manifest_model = manifest.embedding_model.rsplit("/", 1)[-1]
        configured_model = self.embedder.config.model.rsplit("/", 1)[-1]
        if (
            manifest.embedding_provider != "sentence-transformer"
            or manifest_model != configured_model
            or manifest.embedding_dimension != self.embedder.config.dimension
            or manifest.embedding_normalized != self.embedder.config.normalize
        ):
            raise ValueError("Manifest embedding metadata does not match the configured embedder.")
        self.store.assert_existing_corpus(manifest.corpus_id, manifest.corpus_key)
        self.store.assert_embedding_fingerprint(manifest.corpus_key, self.embedder.fingerprint)
        context = ExtractionContext(manifest.corpus_id, corpus_root.resolve())
        report = CompactUpdateReport()

        for source in sources:
            path = source.resolve() if isinstance(source, Path) else None
            key = canonical_file_identity(path, corpus_root) if path else str(getattr(source, "url", source))
            document = None
            activated = False
            revision_started = False
            manifest_changed = False
            previous_revision_id: str | None = None
            previous_entry = manifest.sources.get(key)
            try:
                adapter = adapter_for(path or source, self.adapters)
                document = adapter.extract(path or source, context)
                identity = identity_from_document(document)
                ledger_match = self.store.resolve_ledger_source(identity)
                if ledger_match:
                    identity = replace(identity, ledger_id=str(ledger_match["ledger_source_id"]))
                if ledger_match and ledger_match.get("retrieval_status") == "LEGACY_ONLY" and not force_refresh:
                    report.legacy_only += 1
                    report.events.append(CompactUpdateEvent(key, "legacy_only", message="Use --force-refresh to materialize this historical source."))
                    continue
                if ledger_match and ledger_match.get("document_id"):
                    document = _reidentify_document(document, str(ledger_match["document_id"]))
                active = self.store.active_revision_for_document(document.document_id)
                if (
                    not force_refresh
                    and
                    active
                    and (
                        bool(
                            ledger_match
                            and ledger_match.get("content_checksum") == identity.content_checksum
                        )
                        or (
                            active.get("checksum") == document.source_checksum
                            and active.get("extractor") == document.extractor
                            and active.get("extractor_version") == document.extractor_version
                        )
                    )
                ):
                    if (
                        ledger_match
                        and int(ledger_match.get("duplicate_match_count") or 1) == 1
                        and (
                            ledger_match.get("provider") != identity.provider
                            or ledger_match.get("provider_source_id") != identity.provider_source_id
                        )
                    ):
                        self.store.promote_ledger_identity(
                            str(ledger_match["ledger_source_id"]), identity
                        )
                    manifest.sources[key] = SourceManifestEntry(
                        document_id=document.document_id,
                        active_revision_id=str(active["revision_id"]),
                        checksum=str(active["checksum"]),
                        extractor=str(active["extractor"]),
                        extractor_version=str(active["extractor_version"]),
                        warnings=list(previous_entry.warnings) if previous_entry else [],
                        ledger_source_id=str(ledger_match["ledger_source_id"]) if ledger_match else identity.id,
                    )
                    manifest_changed = True
                    save_manifest(manifest_path, manifest)
                    report.unchanged += 1
                    report.events.append(
                        CompactUpdateEvent(
                            key,
                            "unchanged",
                            str(active["revision_id"]),
                            str(active["revision_id"]),
                        )
                    )
                    continue
                previous_revision_id = str(active["revision_id"]) if active else None
                chunks = self.chunker.chunk(document)
                if capacity_guard is not None:
                    capacity_guard(len(chunks.parents))
                child_vectors = self.embedder.embed_documents([child.text for child in chunks.children])
                children_by_parent: dict[str, list[dict[str, Any]]] = {}
                for child, vector in zip(chunks.children, child_vectors, strict=True):
                    children_by_parent.setdefault(child.parent_id, []).append(
                        {"text": child.text, "token_count": child.token_count, "embedding": vector}
                    )
                parent_vectors = [weighted_parent_embedding(children_by_parent[parent.id]) for parent in chunks.parents]
                revision_started = True
                self.store.begin_compact_revision(
                    corpus_key=manifest.corpus_key,
                    corpus_title=manifest.title,
                    embedding_fingerprint=self.embedder.fingerprint,
                    document=document,
                    chunks=chunks,
                    parent_embeddings=parent_vectors,
                )
                self.store.activate_compact_revision(
                    document.document_id, document.revision_id, len(chunks.parents), ledger=identity
                )
                activated = True
                manifest.sources[key] = SourceManifestEntry(
                    document_id=document.document_id,
                    active_revision_id=document.revision_id,
                    checksum=document.source_checksum,
                    extractor=document.extractor,
                    extractor_version=document.extractor_version,
                    warnings=[*document.warnings, *chunks.warnings],
                    ledger_source_id=identity.id,
                )
                manifest_changed = True
                save_manifest(manifest_path, manifest)
                status = "updated" if previous_revision_id else "added"
                setattr(report, status, getattr(report, status) + 1)
                report.revision_ids.append(document.revision_id)
                report.events.append(
                    CompactUpdateEvent(
                        key,
                        status,
                        document.revision_id,
                        previous_revision_id,
                        len(chunks.parents),
                        len(chunks.children),
                    )
                )
            except Exception as exc:
                errors = [str(exc)]
                cleanup_failed = False
                if activated and document is not None:
                    try:
                        if previous_revision_id:
                            self.store.restore_revision(document.document_id, previous_revision_id)
                        else:
                            self.store.deactivate_document(document.document_id)
                    except Exception as rollback_exc:
                        errors.append(f"database rollback failed: {rollback_exc}")
                        cleanup_failed = True
                if manifest_changed:
                    if previous_entry is None:
                        manifest.sources.pop(key, None)
                    else:
                        manifest.sources[key] = previous_entry
                    try:
                        save_manifest(manifest_path, manifest)
                    except Exception as rollback_exc:
                        errors.append(f"manifest rollback failed: {rollback_exc}")
                        cleanup_failed = True
                if revision_started and document is not None:
                    try:
                        self.store.fail_revision(document.revision_id, "; ".join(errors))
                        self.store.garbage_collect(revision_ids=[document.revision_id])
                        if activated and previous_revision_id is None:
                            self.store.deactivate_document(document.document_id)
                    except Exception as cleanup_exc:
                        errors.append(f"revision cleanup failed: {cleanup_exc}")
                        cleanup_failed = True
                report.failed += 1
                if "conflict" in str(exc).lower():
                    report.conflicted += 1
                report.events.append(
                    CompactUpdateEvent(
                        key,
                        "failed",
                        getattr(document, "revision_id", None),
                        previous_revision_id,
                        message="; ".join(errors),
                    )
                )
                if cleanup_failed:
                    break

        return report


def _reidentify_document(document: Any, target_document_id: str):
    if document.document_id == target_document_id:
        return document
    target_revision_id = revision_id(
        target_document_id, document.source_checksum, f"{document.extractor}:{document.extractor_version}"
    )
    blocks = tuple(
        replace(block, block_id=block_id(target_revision_id, block.ordinal, block.text))
        for block in document.blocks
    )
    return replace(document, document_id=target_document_id, revision_id=target_revision_id, blocks=blocks)
