#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

from neo4j import GraphDatabase

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from notebooklm_graph_pipe.ingestion.ledger_backfill import (
    apply_backfill, backfill_run_id, load_notebooklm_sources, reconcile_inventory,
)
from notebooklm_graph_pipe.ingestion.manifest import load_manifest, save_manifest
from notebooklm_graph_pipe.ingestion.neo4j_store import Neo4jCorpusStore
from notebooklm_graph_pipe.runtime.neo4j_connection import resolve_connection_mapping, verify_connection


def _documents(session, corpus_id: str):
    canonical = [dict(row) for row in session.run(
        """
        MATCH (:Corpus {id: $corpus_id})-[:HAS_DOCUMENT]->(d:Document)-[:ACTIVE_REVISION]->(r:DocumentRevision)
        WHERE d.source_uri STARTS WITH 'documents/'
        RETURN replace(replace(d.source_uri, 'documents/', ''), '.md', '') AS source_id,
               d.id AS document_id, d.title AS title, r.checksum AS checksum,
               r.vector_ready AS vector_ready, r.graph_ready AS graph_ready
        """, corpus_id=corpus_id
    )]
    legacy_raw = [dict(row) for row in session.run(
        """
        MATCH (d:Document) WHERE d.fileName IS NOT NULL
        RETURN d.fileName AS file_name, elementId(d) AS document_id
        """
    )]
    legacy = []
    for row in legacy_raw:
        match = re.search(
            r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
            row["file_name"], re.I,
        )
        if match:
            legacy.append({"source_id": match.group(0).lower(), "document_id": row["document_id"]})
    return canonical, legacy


def _audit(session, corpus_id: str) -> dict[str, int]:
    row = session.run(
        """
        MATCH (:Corpus {id: $corpus_id})-[:HAS_SOURCE]->(s:CorpusSource)
        OPTIONAL MATCH (s)-[:MATERIALIZED_AS]->(d:Document)-[:ACTIVE_REVISION]->(r:DocumentRevision)
        OPTIONAL MATCH (s)-[:LEGACY_EVIDENCE]->(legacy:Document)
        RETURN count(DISTINCT s) AS total,
               count(DISTINCT CASE WHEN s.retrieval_status='ACTIVE' THEN s END) AS active,
               count(DISTINCT CASE WHEN s.retrieval_status='LEGACY_ONLY' THEN s END) AS legacy_only,
               count(DISTINCT CASE WHEN d IS NOT NULL AND r.vector_ready AND r.graph_ready THEN s END) AS ready,
               count(DISTINCT CASE WHEN legacy IS NOT NULL THEN s END) AS with_legacy
        """, corpus_id=corpus_id
    ).single()
    return dict(row) if row else {"total": 0, "active": 0, "legacy_only": 0, "ready": 0, "with_legacy": 0}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Audit or bootstrap the Aura corpus source ledger.")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("audit", "backfill-notebooklm"):
        command = sub.add_parser(name)
        command.add_argument("--manifest-path", required=True)
        command.add_argument("--confirm-target", required=True)
        if name == "backfill-notebooklm":
            command.add_argument("--notebook-id", required=True)
            command.add_argument("--inventory", required=True)
            command.add_argument("--backup-file")
            command.add_argument("--execute", action="store_true")
            command.add_argument("--max-nodes", type=int, default=150_000)
            command.add_argument("--max-relationships", type=int, default=300_000)
            command.add_argument("--capacity-headroom", type=float, default=0.25)
    args = parser.parse_args(argv)
    manifest_path = Path(args.manifest_path).resolve()
    manifest = load_manifest(manifest_path)
    if manifest is None:
        parser.error("Manifest not found.")
    runtime = resolve_connection_mapping(manifest.neo4j)
    target = f"{runtime.uri}|{runtime.database}"
    if args.confirm_target != target:
        parser.error(f"Target confirmation mismatch; expected exactly: {target}")
    if args.command == "backfill-notebooklm" and args.execute:
        backup = Path(args.backup_file or "")
        if not backup.is_file() or backup.stat().st_size == 0:
            parser.error("--execute requires a non-empty verified --backup-file.")
        if not 0 <= args.capacity_headroom < 1:
            parser.error("--capacity-headroom must be in [0, 1).")
    verify_connection(runtime, require_write=args.command == "backfill-notebooklm" and args.execute)
    driver = GraphDatabase.driver(runtime.uri, auth=(runtime.username, runtime.password))
    try:
        with driver.session(database=runtime.database) as session:
            if args.command == "audit":
                payload = {"target": target, "ledger": _audit(session, manifest.corpus_id)}
            else:
                canonical, legacy = _documents(session, manifest.corpus_id)
                rows = reconcile_inventory(
                    corpus_id=manifest.corpus_id, notebook_id=args.notebook_id,
                    inventory_path=Path(args.inventory), live_sources=load_notebooklm_sources(args.notebook_id),
                    canonical_documents=canonical, legacy_documents=legacy,
                )
                run_id = backfill_run_id(rows)
                result = {"created": 0, "changed": 0, "unchanged": 0}
                if args.execute:
                    inventory = session.run(
                        "MATCH (n) WITH count(n) AS nodes MATCH ()-[r]->() "
                        "RETURN nodes, count(r) AS relationships"
                    ).single()
                    ledger_count = session.run(
                        "MATCH (s:CorpusSource {corpus_id: $corpus_id}) RETURN count(s) AS count",
                        corpus_id=manifest.corpus_id,
                    ).single()["count"]
                    add_rows = 0 if int(ledger_count) == len(rows) else len(rows)
                    projected = {
                        "nodes": int(inventory["nodes"]) + add_rows,
                        "relationships": int(inventory["relationships"])
                        + (sum(1 + bool(row["document_id"]) + len(row["legacy_document_ids"]) for row in rows) if add_rows else 0),
                    }
                    limits = {
                        "nodes": int(args.max_nodes * (1 - args.capacity_headroom)),
                        "relationships": int(args.max_relationships * (1 - args.capacity_headroom)),
                    }
                    if any(projected[key] > limits[key] for key in limits):
                        raise RuntimeError(f"Aura capacity headroom gate failed: projected={projected}, guarded_limits={limits}")
                    store = Neo4jCorpusStore(driver, runtime.database, corpus_id=manifest.corpus_id)
                    store.ensure_parent_retrieval_schema(manifest.embedding_dimension)
                    result = session.execute_write(lambda tx: apply_backfill(tx, rows, run_id))
                    session.run(
                        "MATCH (c:Corpus {id: $corpus_id}) SET c.schema_version = 5",
                        corpus_id=manifest.corpus_id,
                    ).consume()
                    ledger_by_document = {
                        row["document_id"]: row["id"] for row in rows if row["document_id"]
                    }
                    manifest_changed = False
                    for entry in manifest.sources.values():
                        ledger_id = ledger_by_document.get(entry.document_id)
                        if ledger_id and entry.ledger_source_id != ledger_id:
                            entry.ledger_source_id = ledger_id
                            manifest_changed = True
                    if manifest_changed or manifest.version != 5:
                        save_manifest(manifest_path, manifest)
                payload = {"target": target, "dry_run": not args.execute, "records": len(rows),
                           "active": sum(r["retrieval_status"] == "ACTIVE" for r in rows),
                           "legacy_only": sum(r["retrieval_status"] == "LEGACY_ONLY" for r in rows),
                           "run_id": run_id, "backup_sha256": hashlib.sha256(Path(args.backup_file).read_bytes()).hexdigest() if args.execute else None,
                           **result, "ledger": _audit(session, manifest.corpus_id)}
    finally:
        driver.close()
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
