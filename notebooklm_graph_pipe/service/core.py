from __future__ import annotations

from dataclasses import asdict
from typing import Any

from filelock import FileLock, Timeout

from notebooklm_graph_pipe.ingestion.manifest import save_manifest
from notebooklm_graph_pipe.ingestion.neo4j_store import Neo4jCorpusStore
from notebooklm_graph_pipe.ingestion.source_ledger import SourceIdentity, SourceIdentityConflict
from notebooklm_graph_pipe.retrieval.hybrid import SearchRequest

from .jobs import CorpusJobManager
from .conversation import ConversationStore, contextualize_question
from .registry import CorpusRegistry
from .runtime import RuntimeFactory
from .ingestions import CorpusIngestionManager


class CorpusService:
    def __init__(
        self,
        registry: CorpusRegistry,
        runtimes: RuntimeFactory,
        jobs: CorpusJobManager,
        conversations: ConversationStore | None = None,
        ingestions: CorpusIngestionManager | None = None,
    ):
        self.registry = registry
        self.runtimes = runtimes
        self.jobs = jobs
        self.conversations = conversations
        self.ingestions = ingestions

    def resolve_sources(self, key: str, probes: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not 1 <= len(probes) <= 100:
            raise ValueError("Source resolution accepts between 1 and 100 probes.")
        entry = self.registry.get(key)
        runtime = self.runtimes.get(entry)
        store = Neo4jCorpusStore(
            runtime.driver,
            entry.manifest.neo4j.get("database") or "neo4j",
            corpus_id=entry.manifest.corpus_id,
        )
        results: list[dict[str, Any]] = []
        try:
            for index, probe in enumerate(probes):
                identity = SourceIdentity(
                    corpus_id=entry.manifest.corpus_id,
                    provider=str(probe.get("provider") or ""),
                    provider_source_id=str(probe.get("provider_source_id") or ""),
                    title=str(probe.get("title") or "discovery probe"),
                    source_type=str(probe.get("source_type") or "document"),
                    canonical_uri=str(probe["canonical_uri"]) if probe.get("canonical_uri") else None,
                    content_checksum=str(probe.get("content_checksum") or ""),
                    notebooklm_source_id=(
                        str(probe["notebooklm_source_id"])
                        if probe.get("notebooklm_source_id")
                        else None
                    ),
                )
                if not any(
                    (
                        identity.provider and identity.provider_source_id,
                        identity.canonical_uri,
                        identity.content_checksum,
                        identity.notebooklm_source_id,
                    )
                ):
                    raise ValueError(f"Source probe {index} has no exact identity field.")
                try:
                    match = store.resolve_ledger_source(identity)
                except SourceIdentityConflict as exc:
                    results.append(
                        {
                            "index": index,
                            "status": "conflict",
                            "ledger_source_ids": sorted(
                                str(item["ledger_source_id"]) for item in exc.matches
                            ),
                        }
                    )
                    continue
                results.append(
                    {"index": index, "status": "matched" if match else "novel", "match": match}
                )
        finally:
            store.close()
        return results

    def submit_ingestion(self, key: str, payload: dict[str, Any]) -> dict[str, Any]:
        if self.ingestions is None:
            raise RuntimeError("Typed ingestion is not configured.")
        return asdict(
            self.ingestions.submit(
                key,
                idempotency_key=str(payload["idempotency_key"]),
                package_sha256=str(payload["package_sha256"]),
                package_path=str(payload["package_path"]),
            )
        )

    def get_ingestion(self, ingestion_id: str) -> dict[str, Any]:
        if self.ingestions is None:
            raise RuntimeError("Typed ingestion is not configured.")
        return asdict(self.ingestions.get(ingestion_id))

    def evaluate_ingestion(self, ingestion_id: str, metrics: dict[str, Any]) -> dict[str, Any]:
        if self.ingestions is None:
            raise RuntimeError("Typed ingestion is not configured.")
        return asdict(self.ingestions.evaluate(ingestion_id, metrics))

    def accept_ingestion(self, ingestion_id: str) -> dict[str, Any]:
        if self.ingestions is None:
            raise RuntimeError("Typed ingestion is not configured.")
        return asdict(self.ingestions.accept(ingestion_id))

    def rollback_ingestion(self, ingestion_id: str) -> dict[str, Any]:
        if self.ingestions is None:
            raise RuntimeError("Typed ingestion is not configured.")
        return asdict(self.ingestions.rollback(ingestion_id))

    def list_corpora(self) -> list[dict[str, Any]]:
        return [
            {
                "key": entry.key,
                "id": entry.manifest.corpus_id,
                "title": entry.manifest.title,
                "source_count": len(entry.manifest.sources),
            }
            for entry in self.registry.entries().values()
        ]

    def get_corpus(self, key: str) -> dict[str, Any]:
        entry = self.registry.get(key)
        payload = entry.manifest.to_dict()
        payload["neo4j"] = {
            field: value
            for field, value in payload.get("neo4j", {}).items()
            if field != "password"
        }
        return payload

    def search(self, key: str, payload: dict[str, Any]) -> dict[str, Any]:
        entry = self.registry.get(key)
        request = SearchRequest(
            query=str(payload["query"]),
            mode=str(payload.get("mode") or "graph_hybrid"),
            top_k=int(payload.get("top_k") or 12),
            graph_hops=int(payload.get("graph_hops") or 1),
            include_diagnostics=bool(payload.get("include_diagnostics", False)),
            filters=dict(payload.get("filters") or {}),
        )
        result = self.runtimes.get(entry).retriever.search(request)
        return {
            "results": [
                {
                    **asdict(candidate),
                    "channels": sorted(candidate.channels),
                    "section_path": list(candidate.section_path),
                }
                for candidate in result.candidates
            ],
            "contexts": [asdict(context) for context in result.contexts],
            "diagnostics": result.diagnostics,
        }

    def answer(self, key: str, payload: dict[str, Any]) -> dict[str, Any]:
        entry = self.registry.get(key)
        question = str(payload["question"])
        conversation_id = str(payload.get("conversation_id") or "").strip() or None
        history = (
            self.conversations.history(conversation_id, entry.manifest.corpus_id)
            if conversation_id and self.conversations
            else []
        )
        contextualized = contextualize_question(question, history)
        mode = str(payload.get("mode") or "graph_hybrid")
        if mode == "global":
            result = self.runtimes.get_community_answerer(entry).global_answer(contextualized)
        elif mode == "drift":
            result = self.runtimes.get_community_answerer(entry).drift_answer(
                contextualized,
                graph_hops=int(payload.get("graph_hops") or 1),
            )
        else:
            result = self.runtimes.get_answerer(entry).answer(
                contextualized,
                mode=mode,
                graph_hops=int(payload.get("graph_hops") or 1),
            )
        if conversation_id and self.conversations:
            self.conversations.append_exchange(
                conversation_id,
                entry.manifest.corpus_id,
                question,
                str(result.get("answer") or ""),
                list(result.get("citations") or []),
            )
            result["conversation_id"] = conversation_id
            result["conversation_turns_used"] = len(history)
        return result

    def answer_stream(self, key: str, payload: dict[str, Any]):
        yield {"event": "started", "data": {"mode": str(payload.get("mode") or "graph_hybrid")}}
        result = self.answer(key, payload)
        for chunk in _text_chunks(str(result.get("answer") or "")):
            yield {"event": "answer_delta", "data": {"text": chunk}}
        yield {"event": "citations", "data": result.get("citations") or []}
        yield {
            "event": "diagnostics",
            "data": {
                "retrieval": result.get("retrieval") or {},
                "warnings": result.get("warnings") or [],
                "conversation_id": result.get("conversation_id"),
                "cost_usd": result.get("cost_usd"),
                "cancellation": "cooperative_between_sse_events",
            },
        }
        yield {"event": "done", "data": {"cancelled": False}}

    def list_documents(self, key: str) -> list[dict[str, Any]]:
        entry = self.registry.get(key)
        return [
            {
                "source_key": source_key,
                **asdict(source),
            }
            for source_key, source in sorted(entry.manifest.sources.items())
        ]

    def get_document(self, key: str, document_id: str) -> dict[str, Any]:
        documents = self.list_documents(key)
        for document in documents:
            if document["document_id"] == document_id:
                return document
        raise KeyError(f"Document not found: {document_id}")

    def delete_document(self, key: str, document_id: str) -> dict[str, Any]:
        entry_hint = self.registry.get(key)
        try:
            with FileLock(str(entry_hint.manifest_path.parent / "sync.lock"), timeout=0):
                entry = self.registry.get(key)
                source_key = next(
                    (source_key for source_key, source in entry.manifest.sources.items() if source.document_id == document_id),
                    None,
                )
                if source_key is None:
                    raise KeyError(f"Document not found: {document_id}")
                source = entry.manifest.sources[source_key]
                runtime = entry.manifest.neo4j
                store = Neo4jCorpusStore(
                    self.runtimes.get(entry).driver,
                    runtime.get("database") or "neo4j",
                    corpus_id=entry.manifest.corpus_id,
                )
                store.deactivate_document(document_id)
                try:
                    del entry.manifest.sources[source_key]
                    entry.manifest.removed_sources = sorted(set(entry.manifest.removed_sources) | {source_key})
                    entry.manifest.suppressed_sources = sorted(set(entry.manifest.suppressed_sources) | {source_key})
                    save_manifest(entry.manifest_path, entry.manifest)
                except Exception:
                    store.restore_revision(document_id, source.active_revision_id)
                    raise
                result = {"document_id": document_id, "status": "deleted"}
                try:
                    store.garbage_collect()
                except Exception as exc:
                    result["warning"] = f"Document was deleted, but garbage collection must be retried: {exc}"
                return result
        except Timeout as exc:
            raise RuntimeError(f"A mutating job is already running for corpus {key}.") from exc

    def submit_sync(self, key: str) -> dict[str, Any]:
        return asdict(self.jobs.submit_sync(self.registry.get(key)))

    def get_job(self, job_id: str) -> dict[str, Any]:
        return asdict(self.jobs.get(job_id))

    def graph_neighbors(self, key: str, entity_id: str, hops: int = 1, limit: int = 50) -> list[dict[str, Any]]:
        entry = self.registry.get(key)
        return self.runtimes.get(entry).backend.graph_neighbors(entity_id, hops, limit)

    def close(self) -> None:
        self.jobs.close()
        if self.ingestions is not None:
            self.ingestions.close()
        self.runtimes.close()


def _text_chunks(text: str, size: int = 80):
    for start in range(0, len(text), size):
        yield text[start : start + size]
