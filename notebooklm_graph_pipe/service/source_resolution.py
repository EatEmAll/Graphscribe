"""Lightweight exact source resolution without retrieval or LLM dependencies."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from neo4j import GraphDatabase

from notebooklm_graph_pipe.ingestion.neo4j_store import Neo4jCorpusStore
from notebooklm_graph_pipe.ingestion.source_ledger import (
    SourceIdentity,
    SourceIdentityConflict,
)
from notebooklm_graph_pipe.runtime.neo4j_connection import resolve_connection_mapping

from .registry import CorpusRegistry, CorpusRegistryEntry


def resolve_source_probes(
    registry: CorpusRegistry,
    driver_provider: Callable[[CorpusRegistryEntry], Any],
    key: str,
    probes: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    if not 1 <= len(probes) <= 100:
        raise ValueError("Source resolution accepts between 1 and 100 probes.")
    entry = registry.get(key)
    store = Neo4jCorpusStore(
        driver_provider(entry),
        entry.manifest.neo4j.get("database") or "neo4j",
        corpus_id=entry.manifest.corpus_id,
    )
    results: list[dict[str, Any]] = []
    for index, probe in enumerate(probes):
        identity = SourceIdentity(
            corpus_id=entry.manifest.corpus_id,
            provider=str(probe.get("connector_id") or probe.get("provider") or ""),
            provider_source_id=str(
                probe.get("provider_id") or probe.get("provider_source_id") or ""
            ),
            title=str(probe.get("title") or "discovery probe"),
            source_type=str(probe.get("source_type") or "document"),
            canonical_uri=str(probe["canonical_uri"]) if probe.get("canonical_uri") else None,
            content_checksum=str(probe.get("content_checksum") or ""),
            notebooklm_source_id=(
                str(probe["notebooklm_source_id"])
                if probe.get("notebooklm_source_id") else None
            ),
        )
        if not any((
            identity.provider and identity.provider_source_id,
            identity.canonical_uri,
            identity.content_checksum,
            identity.notebooklm_source_id,
        )):
            raise ValueError(f"Source probe {index} has no exact identity field.")
        try:
            match = store.resolve_ledger_source(identity)
        except SourceIdentityConflict as exc:
            results.append({
                "index": index,
                "classification": "conflict",
                "source_id": None,
                "match_reason": "conflicting-exact-identities",
                "ledger_source_ids": sorted(
                    str(item["ledger_source_id"]) for item in exc.matches
                ),
            })
            continue
        results.append({
            "index": index,
            "classification": "existing-unchanged" if match else "new",
            "source_id": str(match["ledger_source_id"]) if match else None,
            "match_reason": "exact-ledger-identity" if match else "no-exact-match",
        })
    return {"results": results}


class SourceResolutionService:
    """Own and reuse one Neo4j driver per registered corpus."""

    def __init__(self, registry: CorpusRegistry):
        self.registry = registry
        self._drivers: dict[str, Any] = {}

    def _driver(self, entry: CorpusRegistryEntry):
        driver = self._drivers.get(entry.key)
        if driver is None:
            connection = resolve_connection_mapping(entry.manifest.neo4j)
            driver = GraphDatabase.driver(
                connection.uri, auth=(connection.username, connection.password)
            )
            driver.verify_connectivity()
            self._drivers[entry.key] = driver
        return driver

    def resolve_sources(self, key: str, probes: list[dict[str, Any]]) -> dict:
        return resolve_source_probes(self.registry, self._driver, key, probes)

    def close(self) -> None:
        for driver in self._drivers.values():
            driver.close()
        self._drivers.clear()
