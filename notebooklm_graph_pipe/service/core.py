from __future__ import annotations

from dataclasses import asdict
from typing import Any

from filelock import FileLock, Timeout

from notebooklm_graph_pipe.ingestion.manifest import save_manifest
from notebooklm_graph_pipe.ingestion.neo4j_store import Neo4jCorpusStore
from notebooklm_graph_pipe.retrieval.hybrid import SearchRequest

from .jobs import CorpusJobManager
from .registry import CorpusRegistry
from .runtime import RuntimeFactory


class CorpusService:
    def __init__(self, registry: CorpusRegistry, runtimes: RuntimeFactory, jobs: CorpusJobManager):
        self.registry = registry
        self.runtimes = runtimes
        self.jobs = jobs

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
        return self.runtimes.get_answerer(entry).answer(
            str(payload["question"]),
            mode=str(payload.get("mode") or "graph_hybrid"),
            graph_hops=int(payload.get("graph_hops") or 1),
        )

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
        self.runtimes.close()
