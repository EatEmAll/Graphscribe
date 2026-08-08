from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .ids import ledger_source_id
from .source_ledger import LEDGER_VERSION, canonical_uri, content_fingerprint


def load_notebooklm_sources(notebook_id: str) -> list[dict[str, Any]]:
    result = subprocess.run(
        ["nlm", "source", "list", notebook_id, "--json"], capture_output=True, text=True, check=False
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "NotebookLM source inventory failed.")
    payload = json.loads(result.stdout)
    if not isinstance(payload, list):
        raise ValueError("NotebookLM source inventory is not a JSON list.")
    return payload


def reconcile_inventory(
    *, corpus_id: str, notebook_id: str, inventory_path: Path,
    live_sources: list[dict[str, Any]], canonical_documents: list[dict[str, Any]],
    legacy_documents: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    historical = {str(row["source_id"]): row for row in inventory["sources"]}
    live = {str(row["id"]): row for row in live_sources}
    if set(historical) != set(live):
        missing = sorted(set(historical) - set(live))
        extra = sorted(set(live) - set(historical))
        raise ValueError(f"NotebookLM source IDs differ from historical inventory: missing={missing}, extra={extra}")
    if len(historical) != 244:
        raise ValueError(f"Expected exactly 244 historical sources; found {len(historical)}.")
    canonical = _unique_by_source_id(canonical_documents, "canonical")
    legacy = _group_by_source_id(legacy_documents)
    rows = []
    for source_id, old in sorted(historical.items()):
        current = live[source_id]
        if str(current.get("title")) != str(old.get("title")) or str(current.get("type")) != str(old.get("source_type")):
            raise ValueError(f"Live NotebookLM metadata differs for source {source_id}.")
        content_backed = bool(old.get("relative_path"))
        materialized = canonical.get(source_id)
        evidence = legacy.get(source_id, [])
        if content_backed:
            if not materialized:
                raise ValueError(f"No canonical Aura document for content source {source_id}.")
            if materialized.get("title") != old.get("title") or materialized.get("checksum") != old.get("sha256"):
                raise ValueError(f"Canonical title/checksum mismatch for source {source_id}.")
            if not all(materialized.get(name) is True for name in ("vector_ready", "graph_ready")):
                raise ValueError(f"Canonical document is not retrieval-ready for source {source_id}.")
        elif materialized:
            raise ValueError(f"Empty historical source unexpectedly has canonical content: {source_id}.")
        if not evidence:
            raise ValueError(f"No legacy evidence document for source {source_id}.")
        url = old.get("url") if not content_backed else None
        normalized_url = canonical_uri(url)
        fingerprint = _inventory_content_fingerprint(inventory_path, old) if content_backed else None
        if old["source_type"] == "youtube":
            provider = "youtube"
            provider_source_id = (
                normalized_url.rsplit("=", 1)[-1]
                if normalized_url and "youtube.com/watch?v=" in normalized_url
                else f"content:{fingerprint}:notebooklm:{source_id}"
            )
        else:
            provider = "document"
            provider_source_id = f"content:{fingerprint}:notebooklm:{source_id}"
        rows.append({
            "id": ledger_source_id(corpus_id, provider, provider_source_id),
            "corpus_id": corpus_id, "provider": provider, "provider_source_id": provider_source_id,
            "notebooklm_source_id": source_id,
            "title": old["title"], "source_type": old["source_type"],
            "canonical_uri": normalized_url, "content_checksum": fingerprint,
            "acquisition_method": old.get("acquisition"), "notebook_ids": [notebook_id],
            "ledger_version": LEDGER_VERSION, "ingestion_status": "INGESTED",
            "retrieval_status": "ACTIVE" if content_backed else "LEGACY_ONLY",
            "document_id": materialized.get("document_id") if materialized else None,
            "legacy_document_ids": sorted({str(item["document_id"]) for item in evidence}),
        })
    return rows


def backfill_run_id(rows: list[dict[str, Any]]) -> str:
    stable = sorted(
        ({key: row[key] for key in sorted(row) if key not in {"backfill_run_id", "verified_at"}} for row in rows),
        key=lambda row: row["id"],
    )
    return hashlib.sha256(json.dumps(stable, sort_keys=True).encode()).hexdigest()


def apply_backfill(session: Any, rows: list[dict[str, Any]], run_id: str) -> dict[str, int]:
    existing = list(session.run(
        """
        MATCH (s:CorpusSource {corpus_id: $corpus_id})
        OPTIONAL MATCH (s)-[:MATERIALIZED_AS]->(document:Document)
        OPTIONAL MATCH (s)-[:LEGACY_EVIDENCE]->(legacy:Document)
        RETURN s.id AS id, s.provider AS provider, s.provider_source_id AS provider_source_id,
               s.notebooklm_source_id AS notebooklm_source_id,
               s.title AS title, s.source_type AS source_type, s.canonical_uri AS canonical_uri,
               s.content_checksum AS content_checksum, s.acquisition_method AS acquisition_method,
               s.notebook_ids AS notebook_ids, s.ledger_version AS ledger_version,
               s.ingestion_status AS ingestion_status, s.retrieval_status AS retrieval_status,
               document.id AS document_id, collect(DISTINCT elementId(legacy)) AS legacy_document_ids,
               s.backfill_run_id AS backfill_run_id
        """,
        corpus_id=rows[0]["corpus_id"],
    ))
    if len(existing) == len(rows):
        expected = {row["notebooklm_source_id"]: row for row in rows}
        compare_fields = {
            "provider", "provider_source_id", "title", "source_type", "canonical_uri",
            "content_checksum", "acquisition_method", "notebook_ids", "ledger_version",
            "ingestion_status", "retrieval_status", "document_id", "legacy_document_ids", "notebooklm_source_id",
        }
        migration_needed = False
        for record in existing:
            actual = dict(record)
            alias = actual.get("notebooklm_source_id") or (
                actual.get("provider_source_id") if actual.get("provider") == "notebooklm" else None
            )
            if alias not in expected:
                raise ValueError(f"Existing ledger has an unknown NotebookLM identity: {alias}")
            wanted = expected[alias]
            for field in compare_fields:
                left = sorted(actual[field] or []) if field in {"notebook_ids", "legacy_document_ids"} else actual[field]
                right = sorted(wanted[field] or []) if field in {"notebook_ids", "legacy_document_ids"} else wanted[field]
                if left != right:
                    migration_needed = True
            if actual["id"] != wanted["id"] or actual["backfill_run_id"] != run_id:
                migration_needed = True
        if migration_needed:
            changed = session.run(
                """
                UNWIND $rows AS row
                MATCH (source:CorpusSource {corpus_id: row.corpus_id})
                WHERE source.notebooklm_source_id = row.notebooklm_source_id
                   OR (source.provider = 'notebooklm' AND source.provider_source_id = row.notebooklm_source_id)
                SET source.id = row.id, source.provider = row.provider,
                    source.provider_source_id = row.provider_source_id,
                    source.notebooklm_source_id = row.notebooklm_source_id,
                    source.source_type = row.source_type, source.canonical_uri = row.canonical_uri,
                    source.content_checksum = row.content_checksum,
                    source.ledger_version = row.ledger_version,
                    source.backfill_run_id = $run_id, source.last_verified_at = datetime()
                RETURN count(source) AS changed
                """, rows=rows, run_id=run_id,
            ).single()["changed"]
            if int(changed) != len(rows):
                raise RuntimeError(f"Canonical identity migration changed {changed} of {len(rows)} sources.")
            return {"created": 0, "changed": int(changed), "unchanged": 0}
        return {"created": 0, "changed": 0, "unchanged": len(rows)}
    if existing:
        raise ValueError("Partial source ledger exists; audit and resolve it before backfill.")
    verified_at = datetime.now(timezone.utc).isoformat()
    result = session.run(
        """
        MATCH (corpus:Corpus {id: $corpus_id})
        UNWIND $rows AS row
        CREATE (source:CorpusSource)
        SET source.id = row.id, source.corpus_id = row.corpus_id,
            source.provider = row.provider, source.provider_source_id = row.provider_source_id,
            source.notebooklm_source_id = row.notebooklm_source_id,
            source.title = row.title, source.source_type = row.source_type,
            source.canonical_uri = row.canonical_uri, source.content_checksum = row.content_checksum,
            source.acquisition_method = row.acquisition_method, source.notebook_ids = row.notebook_ids,
            source.ledger_version = row.ledger_version, source.ingestion_status = row.ingestion_status,
            source.retrieval_status = row.retrieval_status, source.backfill_run_id = $run_id,
            source.first_seen_at = datetime($verified_at), source.last_verified_at = datetime($verified_at)
        CREATE (corpus)-[:HAS_SOURCE]->(source)
        WITH source, row
        OPTIONAL MATCH (document:Document {id: row.document_id})
        FOREACH (_ IN CASE WHEN document IS NULL THEN [] ELSE [1] END |
            CREATE (source)-[:MATERIALIZED_AS]->(document))
        WITH source, row
        UNWIND row.legacy_document_ids AS legacy_id
        MATCH (legacy:Document) WHERE elementId(legacy) = legacy_id
        CREATE (source)-[:LEGACY_EVIDENCE]->(legacy)
        RETURN count(DISTINCT source) AS created
        """, corpus_id=rows[0]["corpus_id"], rows=rows, run_id=run_id, verified_at=verified_at
    ).single()
    created = int(result["created"]) if result else 0
    if created != len(rows):
        raise RuntimeError(f"Atomic backfill created {created} of {len(rows)} ledger records.")
    return {"created": created, "changed": 0, "unchanged": 0}


def _unique_by_source_id(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    grouped = _group_by_source_id(rows)
    ambiguous = [key for key, values in grouped.items() if len(values) != 1]
    if ambiguous:
        raise ValueError(f"Ambiguous {label} document matches: {ambiguous}")
    return {key: values[0] for key, values in grouped.items()}


def _group_by_source_id(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["source_id"]), []).append(row)
    return grouped


def _inventory_content_fingerprint(inventory_path: Path, source: dict[str, Any]) -> str:
    path = inventory_path.parent / str(source["relative_path"])
    text = path.read_text(encoding="utf-8")
    if text.startswith("---\n"):
        end = text.find("\n---\n", 4)
        if end >= 0:
            text = text[end + 5 :]
    return content_fingerprint(text)
