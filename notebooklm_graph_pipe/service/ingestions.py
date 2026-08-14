from __future__ import annotations

import hashlib
import json
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from filelock import FileLock, Timeout

from notebooklm_graph_pipe.ingestion.adapters import ExtractionContext, SourcePackage, SourcePackageAdapter
from notebooklm_graph_pipe.ingestion.chunking import HierarchicalChunker, load_minilm_tokenizer
from notebooklm_graph_pipe.ingestion.compact_sync import _reidentify_document
from notebooklm_graph_pipe.ingestion.embeddings import MiniLMEmbedder, weighted_parent_embedding
from notebooklm_graph_pipe.ingestion.manifest import SourceManifestEntry, load_manifest, save_manifest
from notebooklm_graph_pipe.ingestion.neo4j_store import Neo4jCorpusStore
from notebooklm_graph_pipe.ingestion.source_ledger import SourceIdentity, identity_from_document

from .registry import CorpusRegistry, CorpusRegistryEntry
from .runtime import RuntimeFactory


TERMINAL = frozenset({"duplicate", "accepted", "rolled_back", "failed", "quarantined"})


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def package_digest(root: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        raise ValueError("Source package is empty.")
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


@dataclass
class IngestionRecord:
    id: str
    corpus_key: str
    idempotency_key: str
    package_sha256: str
    package_path: str
    status: str
    created_at: str
    updated_at: str
    document_id: str | None = None
    revision_id: str | None = None
    previous_revision_id: str | None = None
    source_key: str | None = None
    expected_parents: int = 0
    source_checksum: str | None = None
    ledger: dict[str, Any] | None = None
    ledger_match: dict[str, Any] | None = None
    evaluation: dict[str, Any] = field(default_factory=dict)
    error: str = ""


class CorpusIngestionManager:
    def __init__(
        self,
        registry: CorpusRegistry,
        runtimes: RuntimeFactory,
        ingestion_root: Path,
        *,
        store_factory: Callable[..., Neo4jCorpusStore] = Neo4jCorpusStore,
        embedder_factory: Callable[[], MiniLMEmbedder] = MiniLMEmbedder,
        chunker_factory: Callable[[], HierarchicalChunker] | None = None,
    ):
        self.registry = registry
        self.runtimes = runtimes
        self.ingestion_root = ingestion_root.resolve()
        self.store_factory = store_factory
        self.embedder_factory = embedder_factory
        self.chunker_factory = chunker_factory or (lambda: HierarchicalChunker(load_minilm_tokenizer()))
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="corpus-ingestion")
        self._records: dict[str, IngestionRecord] = {}
        self._active_corpora: set[str] = set()
        self._lock = threading.Lock()
        self._load_records()
        self._recover_interrupted()

    def _record_path(self, entry: CorpusRegistryEntry, record_id: str) -> Path:
        path = entry.manifest_path.parent / "ingestions" / f"{record_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _save(self, entry: CorpusRegistryEntry, record: IngestionRecord) -> None:
        record.updated_at = utc_now()
        path = self._record_path(entry, record.id)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(asdict(record), indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(path)

    def _load_records(self) -> None:
        for path in self.registry.root.glob("*/ingestions/*.json"):
            record = IngestionRecord(**json.loads(path.read_text(encoding="utf-8")))
            self._records[record.id] = record

    def _recover_interrupted(self) -> None:
        for record in list(self._records.values()):
            if record.status not in {"queued", "staging"}:
                continue
            entry = self.registry.get(record.corpus_key)
            record.status = "queued"
            self._active_corpora.add(record.corpus_key)
            self._save(entry, record)
            self._executor.submit(self._stage, entry, record)

    def _package_path(self, relative_path: str) -> Path:
        candidate = Path(relative_path)
        if candidate.is_absolute():
            raise ValueError("Source package path must be relative to the configured ingestion root.")
        resolved = (self.ingestion_root / candidate).resolve()
        if not resolved.is_relative_to(self.ingestion_root) or not resolved.is_dir():
            raise ValueError("Source package path escapes the configured ingestion root or is missing.")
        return resolved

    def submit(
        self,
        corpus_key: str,
        *,
        idempotency_key: str,
        package_sha256: str,
        package_path: str,
    ) -> IngestionRecord:
        if not idempotency_key.strip() or len(idempotency_key) > 200:
            raise ValueError("A bounded idempotency key is required.")
        if len(package_sha256) != 64 or any(char not in "0123456789abcdef" for char in package_sha256):
            raise ValueError("package_sha256 must be a lowercase SHA-256 digest.")
        root = self._package_path(package_path)
        actual = package_digest(root)
        if actual != package_sha256:
            raise ValueError("Source package digest mismatch.")
        entry = self.registry.get(corpus_key)
        with self._lock:
            existing = next(
                (
                    item
                    for item in self._records.values()
                    if item.corpus_key == corpus_key and item.idempotency_key == idempotency_key
                ),
                None,
            )
            if existing:
                if existing.package_sha256 != package_sha256:
                    raise ValueError("Idempotency key is already bound to a different package.")
                return existing
            if corpus_key in self._active_corpora:
                raise RuntimeError(f"A mutating job is already running for corpus {corpus_key}.")
            record = IngestionRecord(
                id=str(uuid.uuid4()),
                corpus_key=corpus_key,
                idempotency_key=idempotency_key,
                package_sha256=package_sha256,
                package_path=package_path,
                status="queued",
                created_at=utc_now(),
                updated_at=utc_now(),
            )
            self._records[record.id] = record
            self._active_corpora.add(corpus_key)
            self._save(entry, record)
            self._executor.submit(self._stage, entry, record)
            return record

    def _stage(self, entry: CorpusRegistryEntry, record: IngestionRecord) -> None:
        record.status = "staging"
        self._save(entry, record)
        store: Neo4jCorpusStore | None = None
        corpus_lock: FileLock | None = None
        try:
            corpus_lock = FileLock(str(entry.manifest_path.parent / "sync.lock"))
            corpus_lock.acquire(timeout=0)
            root = self._package_path(record.package_path)
            if package_digest(root) != record.package_sha256:
                raise ValueError("Source package changed after submission.")
            runtime = self.runtimes.get(entry)
            store = self.store_factory(runtime.driver, entry.manifest.neo4j.get("database") or "neo4j", corpus_id=entry.manifest.corpus_id)
            document = SourcePackageAdapter().extract(
                SourcePackage(root), ExtractionContext(entry.manifest.corpus_id, self.ingestion_root)
            )
            identity = identity_from_document(document)
            match = store.resolve_ledger_source(identity)
            if match:
                identity = replace(identity, ledger_id=str(match["ledger_source_id"]))
                if match.get("document_id"):
                    document = _reidentify_document(document, str(match["document_id"]))
            active = store.active_revision_for_document(document.document_id)
            if match and match.get("content_checksum") == identity.content_checksum and active:
                record.status = "duplicate"
                record.document_id = document.document_id
                record.revision_id = str(active["revision_id"])
                record.ledger = identity.to_row()
                record.ledger_match = match
                self._save(entry, record)
                return
            chunker = self.chunker_factory()
            embedder = self.embedder_factory()
            expected_profile = ("parent", "parent_embedding_v1", "parent_keyword_v1")
            actual_profile = (
                entry.manifest.retrieval_unit,
                entry.manifest.retrieval_vector_index,
                entry.manifest.retrieval_keyword_index,
            )
            if actual_profile != expected_profile:
                raise ValueError(
                    f"Typed source-package ingestion requires retrieval profile {expected_profile}; "
                    f"received {actual_profile}."
                )
            store.assert_existing_corpus(entry.manifest.corpus_id, entry.manifest.corpus_key)
            store.assert_embedding_fingerprint(entry.manifest.corpus_key, embedder.fingerprint)
            chunks = chunker.chunk(document)
            if not chunks.parents or not chunks.children:
                raise ValueError("Source package produced no retrievable chunks.")
            child_vectors = embedder.embed_documents([child.text for child in chunks.children])
            children_by_parent: dict[str, list[dict[str, Any]]] = {}
            for child, vector in zip(chunks.children, child_vectors, strict=True):
                children_by_parent.setdefault(child.parent_id, []).append(
                    {"text": child.text, "token_count": child.token_count, "embedding": vector}
                )
            parent_vectors = [weighted_parent_embedding(children_by_parent[parent.id]) for parent in chunks.parents]
            record.document_id = document.document_id
            record.revision_id = document.revision_id
            self._save(entry, record)
            store.begin_compact_revision(
                corpus_key=entry.manifest.corpus_key,
                corpus_title=entry.manifest.title,
                embedding_fingerprint=embedder.fingerprint,
                document=document,
                chunks=chunks,
                parent_embeddings=parent_vectors,
            )
            store.stage_compact_revision(document.document_id, document.revision_id, len(chunks.parents))
            source_key = next(
                (
                    key
                    for key, source in entry.manifest.sources.items()
                    if source.document_id == document.document_id
                ),
                f"source-package/{identity.id}",
            )
            record.previous_revision_id = str(active["revision_id"]) if active else None
            record.source_key = source_key
            record.expected_parents = len(chunks.parents)
            record.source_checksum = document.source_checksum
            record.ledger = identity.to_row()
            record.ledger_match = match
            record.status = "staged"
            self._save(entry, record)
        except Exception as exc:
            record.status = "quarantined"
            record.error = str(exc)[:2000]
            if store is not None and record.revision_id:
                try:
                    store.fail_revision(record.revision_id, record.error)
                except Exception:
                    pass
            self._save(entry, record)
        finally:
            if store is not None:
                store.close()
            if corpus_lock is not None and corpus_lock.is_locked:
                corpus_lock.release()
            with self._lock:
                self._active_corpora.discard(entry.key)

    def get(self, record_id: str) -> IngestionRecord:
        try:
            return self._records[record_id]
        except KeyError as exc:
            raise KeyError(f"Ingestion not found: {record_id}") from exc

    def evaluate(self, record_id: str, metrics: dict[str, Any]) -> IngestionRecord:
        import math

        record = self.get(record_id)
        if record.status not in {"staged", "evaluated"} or not record.document_id or not record.revision_id:
            raise RuntimeError("Only a staged ingestion can be evaluated.")
        entry = self.registry.get(record.corpus_key)
        runtime = self.runtimes.get(entry)
        store = self.store_factory(runtime.driver, entry.manifest.neo4j.get("database") or "neo4j", corpus_id=entry.manifest.corpus_id)
        try:
            state = store.staged_revision_state(record.document_id, record.revision_id)
        finally:
            store.close()
        if not state or state.get("is_active") or state.get("status") != "STAGED":
            raise RuntimeError("Revision is not safely staged.")
        required = {
            "baseline_quality_ratio": 0.95,
            "effective_citation_ratio": 1.0,
            "graph_expansion_ratio": 0.90,
            "capacity_headroom_ratio": 0.25,
        }
        numeric = [*required, "unsupported_claim_delta"]
        if any(not math.isfinite(float(metrics.get(key, float("nan")))) for key in numeric):
            raise ValueError("Evaluation metrics must be finite.")
        failures = [key for key, threshold in required.items() if float(metrics.get(key, -1)) < threshold]
        if float(metrics.get("unsupported_claim_delta", 1)) > 0:
            failures.append("unsupported_claim_delta")
        if not bool(metrics.get("source_canary_retrieved")):
            failures.append("source_canary_retrieved")
        if not state.get("graph_ready") or int(state.get("completed_parents") or 0) != record.expected_parents:
            failures.append("graph_ready")
        record.evaluation = {"passed": not failures, "failures": sorted(set(failures)), "metrics": metrics, "state": state}
        record.status = "evaluated"
        self._save(entry, record)
        return record

    def accept(self, record_id: str) -> IngestionRecord:
        record = self.get(record_id)
        if record.status == "accepted":
            return record
        if record.status != "evaluated" or not record.evaluation.get("passed"):
            raise RuntimeError("Ingestion must pass evaluation before acceptance.")
        if not all((record.document_id, record.revision_id, record.source_key, record.ledger, record.source_checksum)):
            raise RuntimeError("Ingestion record is incomplete.")
        entry_hint = self.registry.get(record.corpus_key)
        try:
            with FileLock(str(entry_hint.manifest_path.parent / "sync.lock"), timeout=0):
                entry = self.registry.get(record.corpus_key)
                runtime = self.runtimes.get(entry)
                store = self.store_factory(runtime.driver, entry.manifest.neo4j.get("database") or "neo4j", corpus_id=entry.manifest.corpus_id)
                ledger_payload = dict(record.ledger)
                ledger_payload["ledger_id"] = ledger_payload.pop("id")
                ledger_payload["notebook_ids"] = tuple(ledger_payload.get("notebook_ids") or ())
                identity = SourceIdentity(**ledger_payload)
                store.activate_compact_revision(
                    record.document_id,
                    record.revision_id,
                    record.expected_parents,
                    ledger=identity,
                    require_staged=True,
                )
                try:
                    entry.manifest.sources[record.source_key] = SourceManifestEntry(
                        document_id=record.document_id,
                        active_revision_id=record.revision_id,
                        checksum=record.source_checksum,
                        extractor="source-package",
                        extractor_version="1",
                        ledger_source_id=identity.id,
                    )
                    save_manifest(entry.manifest_path, entry.manifest)
                except Exception:
                    store.rollback_failed_accept(
                        record.document_id,
                        record.revision_id,
                        record.previous_revision_id,
                        identity.id,
                        remove_new_ledger=record.ledger_match is None,
                    )
                    raise
                finally:
                    store.close()
        except Timeout as exc:
            raise RuntimeError(f"A mutating job is already running for corpus {record.corpus_key}.") from exc
        record.status = "accepted"
        self._save(self.registry.get(record.corpus_key), record)
        return record

    def rollback(self, record_id: str) -> IngestionRecord:
        record = self.get(record_id)
        if record.status == "rolled_back":
            return record
        if record.status == "accepted":
            raise RuntimeError("Accepted ingestions require the existing document rollback workflow.")
        if not record.revision_id:
            raise RuntimeError("Ingestion has no staged revision to roll back.")
        entry = self.registry.get(record.corpus_key)
        runtime = self.runtimes.get(entry)
        store = self.store_factory(runtime.driver, entry.manifest.neo4j.get("database") or "neo4j", corpus_id=entry.manifest.corpus_id)
        try:
            store.fail_revision(record.revision_id, "Rolled back before acceptance.")
            store.garbage_collect(revision_ids=[record.revision_id])
        finally:
            store.close()
        record.status = "rolled_back"
        self._save(entry, record)
        return record

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=False)
