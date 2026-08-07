#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from neo4j import GraphDatabase

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from notebooklm_graph_pipe.consolidation import taxonomy_cleanup, tier1_lemmatize, tier2_relabel, tier3_semantic
from notebooklm_graph_pipe.ingestion.manifest import load_manifest
from notebooklm_graph_pipe.runtime.neo4j_connection import resolve_connection_mapping, verify_corpus_connection


def _preflight(
    driver,
    database: str,
    corpus_id: str,
    revision_ids: list[str],
    *,
    require_apoc: bool,
) -> dict[str, int]:
    with driver.session(database=database) as session:
        rows = list(
            session.run(
                """
                UNWIND $revision_ids AS revision_id
                OPTIONAL MATCH (corpus:Corpus {id: $corpus_id})-[:HAS_DOCUMENT]->
                               (document:Document)-[:HAS_REVISION]->
                               (revision:DocumentRevision {id: revision_id})
                OPTIONAL MATCH (revision)-[:HAS_PARENT]->(parent:ParentChunk)
                OPTIONAL MATCH (parent)-[:HAS_ENTITY]->(entity:__Entity__)
                RETURN revision_id, revision.status AS status, revision.graph_ready AS graph_ready,
                       EXISTS { MATCH (document)-[:ACTIVE_REVISION]->(revision) } AS is_active,
                       count(DISTINCT parent) AS parents, count(DISTINCT entity) AS entities
                """,
                corpus_id=corpus_id,
                revision_ids=revision_ids,
            )
        )
        missing = [row["revision_id"] for row in rows if row["status"] is None]
        not_ready = [
            row["revision_id"]
            for row in rows
            if row["status"] != "ACTIVE" or not row["is_active"] or not row["graph_ready"]
        ]
        if missing:
            raise RuntimeError(f"Unknown revision ids: {missing}")
        if not_ready:
            raise RuntimeError(f"Delta consolidation requires ACTIVE, graph-ready revisions: {not_ready}")
        if require_apoc:
            procedure = session.run(
                "SHOW PROCEDURES YIELD name WHERE name = 'apoc.refactor.mergeNodes' RETURN count(*) AS count"
            ).single()
            if not procedure or int(procedure["count"]) != 1:
                raise RuntimeError("APOC Core procedure apoc.refactor.mergeNodes is required for consolidation.")
        return {
            "revisions": len(rows),
            "parents": sum(int(row["parents"] or 0) for row in rows),
            "grounded_entities": sum(int(row["entities"] or 0) for row in rows),
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Consolidate only entities grounded in selected corpus revisions.")
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--revision-id", action="append", required=True)
    parser.add_argument("--confirm-target", required=True, help="Must equal <neo4j-uri>|<database>.")
    parser.add_argument("--execute", action="store_true", help="Write changes; default is dry-run.")
    parser.add_argument("--backup-file", help="Exported pre-consolidation Aura .backup; required with --execute.")
    parser.add_argument("--llm-routing-config")
    parser.add_argument("--max-nodes", type=int, default=500)
    parser.add_argument("--max-candidates", type=int, default=600)
    parser.add_argument("--max-merges", type=int, default=200)
    args = parser.parse_args(argv)
    backup_file = Path(args.backup_file).resolve() if args.backup_file else None
    if args.execute and (backup_file is None or not backup_file.is_file()):
        parser.error("--execute requires an existing --backup-file exported from Aura before consolidation.")

    manifest = load_manifest(Path(args.manifest_path).resolve())
    if manifest is None:
        parser.error("Corpus manifest was not found.")
    runtime = resolve_connection_mapping(manifest.neo4j)
    expected_confirmation = f"{runtime.uri}|{runtime.database}"
    if args.confirm_target != expected_confirmation:
        parser.error(f"Target confirmation mismatch; expected exactly: {expected_confirmation}")
    verify_corpus_connection(
        runtime,
        dimension=manifest.embedding_dimension,
        require_write=args.execute,
        retrieval_unit=manifest.retrieval_unit,
        vector_index=manifest.retrieval_vector_index,
        keyword_index=manifest.retrieval_keyword_index,
        require_retrieval_vector=manifest.retrieval_vector_provider == "neo4j",
    )
    revisions = list(dict.fromkeys(args.revision_id))
    driver = GraphDatabase.driver(runtime.uri, auth=(runtime.username, runtime.password))
    try:
        scope = _preflight(
            driver,
            runtime.database,
            manifest.corpus_id,
            revisions,
            require_apoc=args.execute,
        )
    finally:
        driver.close()
    if scope["grounded_entities"] == 0:
        raise RuntimeError("Selected revisions have no grounded entities; run graph extraction first.")

    connection = {
        "neo4j_uri": runtime.uri,
        "neo4j_user": runtime.username,
        "neo4j_password": runtime.password,
        "neo4j_database": runtime.database,
    }
    dry_run = not args.execute
    tier1_lemmatize.run(dry_run=dry_run, scope_revision_ids=revisions, **connection)
    tier2 = tier2_relabel.run(
        dry_run=dry_run,
        batch_size=50,
        sleep_seconds=0,
        max_nodes=args.max_nodes,
        labels_json=None,
        decisions_jsonl=None,
        llm_routing_config=args.llm_routing_config,
        scope_revision_ids=revisions,
        **connection,
    )
    taxonomy = taxonomy_cleanup.run(
        dry_run=dry_run,
        max_nodes=args.max_nodes,
        candidate_limit=8,
        embedding_threshold=0.84,
        labels_json=None,
        tier2_decisions_jsonl=None,
        prior_taxonomy_decisions_jsonl=None,
        prior_review_json=None,
        summary_json=None,
        decisions_jsonl=None,
        llm_routing_config=args.llm_routing_config,
        scope_revision_ids=revisions,
        **connection,
    )
    tier3 = tier3_semantic.run(
        dry_run=dry_run,
        threshold=0.85,
        max_candidates=args.max_candidates,
        max_merges=args.max_merges,
        sleep_seconds=0,
        llm_routing_config=args.llm_routing_config,
        scope_revision_ids=revisions,
        **connection,
    )
    print(
        json.dumps(
            {
                "target": expected_confirmation,
                "dry_run": dry_run,
                "backup_file": str(backup_file) if backup_file else None,
                "scope": scope,
                "tier2": tier2,
                "taxonomy": taxonomy,
                "tier3": tier3,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
