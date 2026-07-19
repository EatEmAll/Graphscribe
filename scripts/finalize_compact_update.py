#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from neo4j import GraphDatabase

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from notebooklm_graph_pipe.ingestion.manifest import SourceManifestEntry, load_manifest, save_manifest
from notebooklm_graph_pipe.ingestion.neo4j_store import Neo4jCorpusStore
from notebooklm_graph_pipe.runtime.neo4j_connection import resolve_connection_mapping, verify_corpus_connection


def _revision(driver, database: str, corpus_id: str, revision_id: str) -> dict | None:
    with driver.session(database=database) as session:
        row = session.run(
            """
            MATCH (:Corpus {id: $corpus_id})-[:HAS_DOCUMENT]->
                  (document:Document)-[:HAS_REVISION]->(revision:DocumentRevision {id: $revision_id})
            RETURN document.id AS document_id, revision.id AS revision_id,
                   revision.status AS status, revision.vector_ready AS vector_ready,
                   revision.graph_ready AS graph_ready, revision.checksum AS checksum,
                   revision.extractor AS extractor, revision.extractor_version AS extractor_version,
                   EXISTS { MATCH (document)-[:ACTIVE_REVISION]->(revision) } AS is_active
            """,
            corpus_id=corpus_id,
            revision_id=revision_id,
        ).single()
        return dict(row) if row else None


def _previous_revision(
    driver,
    database: str,
    corpus_id: str,
    document_id: str,
    active_revision_id: str,
) -> dict | None:
    with driver.session(database=database) as session:
        row = session.run(
            """
            MATCH (:Corpus {id: $corpus_id})-[:HAS_DOCUMENT]->
                  (document:Document {id: $document_id})-[:HAS_REVISION]->(revision:DocumentRevision)
            WHERE revision.id <> $active_revision_id AND revision.status = 'INACTIVE'
            RETURN document.id AS document_id, revision.id AS revision_id,
                   revision.checksum AS checksum, revision.extractor AS extractor,
                   revision.extractor_version AS extractor_version
            ORDER BY coalesce(revision.deactivated_at, revision.created_at) DESC
            LIMIT 1
            """,
            corpus_id=corpus_id,
            document_id=document_id,
            active_revision_id=active_revision_id,
        ).single()
        return dict(row) if row else None


def _active_revision(driver, database: str, corpus_id: str, document_id: str) -> dict | None:
    with driver.session(database=database) as session:
        row = session.run(
            """
            MATCH (:Corpus {id: $corpus_id})-[:HAS_DOCUMENT]->
                  (document:Document {id: $document_id})-[:ACTIVE_REVISION]->(revision:DocumentRevision)
            RETURN document.id AS document_id, revision.id AS revision_id,
                   revision.checksum AS checksum, revision.extractor AS extractor,
                   revision.extractor_version AS extractor_version
            """,
            corpus_id=corpus_id,
            document_id=document_id,
        ).single()
        return dict(row) if row else None


def _manifest_entry(row: dict) -> SourceManifestEntry:
    return SourceManifestEntry(
        document_id=row["document_id"],
        active_revision_id=row["revision_id"],
        checksum=str(row["checksum"] or ""),
        extractor=str(row["extractor"] or ""),
        extractor_version=str(row["extractor_version"] or ""),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Accept or roll back a compact corpus revision update.")
    parser.add_argument("action", choices=("accept", "rollback"))
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--revision-id", required=True, help="One revision per finalize operation.")
    parser.add_argument("--confirm-target", required=True, help="Must equal <neo4j-uri>|<database>.")
    args = parser.parse_args(argv)

    manifest_path = Path(args.manifest_path).resolve()
    manifest = load_manifest(manifest_path)
    if manifest is None:
        parser.error("Corpus manifest was not found.")
    runtime = resolve_connection_mapping(manifest.neo4j)
    expected_confirmation = f"{runtime.uri}|{runtime.database}"
    if args.confirm_target != expected_confirmation:
        parser.error(f"Target confirmation mismatch; expected exactly: {expected_confirmation}")
    verify_corpus_connection(
        runtime,
        dimension=manifest.embedding_dimension,
        require_write=True,
        retrieval_unit=manifest.retrieval_unit,
        vector_index=manifest.retrieval_vector_index,
        keyword_index=manifest.retrieval_keyword_index,
    )

    driver = GraphDatabase.driver(runtime.uri, auth=(runtime.username, runtime.password))
    store = Neo4jCorpusStore(driver, runtime.database, corpus_id=manifest.corpus_id)
    try:
        revisions = [_revision(driver, runtime.database, manifest.corpus_id, args.revision_id)]
        if revisions[0] is None:
            raise RuntimeError(f"Unknown revision id for corpus {manifest.corpus_key}: {args.revision_id}")
        resolved = [revision for revision in revisions if revision is not None]
        document_ids = [row["document_id"] for row in resolved]
        if len(document_ids) != len(set(document_ids)):
            raise RuntimeError("Pass at most one revision per document in a finalize operation.")
        if args.action == "accept":
            invalid = [
                row["revision_id"]
                for row in resolved
                if row["status"] != "ACTIVE"
                or not row["is_active"]
                or not row["vector_ready"]
                or not row["graph_ready"]
            ]
            if invalid:
                raise RuntimeError(f"Only ACTIVE, vector-ready, graph-ready revisions can be accepted: {invalid}")
            garbage_collected = store.garbage_collect(document_ids)
            result = {"accepted": [args.revision_id], "garbage_collected": garbage_collected}
        else:
            recovered = False
            for row in resolved:
                matching_keys = [
                    key for key, entry in manifest.sources.items() if entry.document_id == row["document_id"]
                ]
                if len(matching_keys) > 1:
                    raise RuntimeError(f"Manifest does not map exactly one source to document {row['document_id']}.")
                if row["status"] != "ACTIVE" or not row["is_active"]:
                    active = _active_revision(
                        driver,
                        runtime.database,
                        manifest.corpus_id,
                        row["document_id"],
                    )
                    if active:
                        if len(matching_keys) != 1:
                            raise RuntimeError(
                                f"Cannot reconcile manifest source for active document {row['document_id']}."
                            )
                        manifest.sources[matching_keys[0]] = _manifest_entry(active)
                    elif matching_keys:
                        del manifest.sources[matching_keys[0]]
                    recovered = True
                    continue
                if len(matching_keys) != 1:
                    raise RuntimeError(f"Manifest does not map exactly one source to document {row['document_id']}.")
                previous = _previous_revision(
                    driver,
                    runtime.database,
                    manifest.corpus_id,
                    row["document_id"],
                    row["revision_id"],
                )
                if previous:
                    store.restore_revision(row["document_id"], previous["revision_id"])
                    manifest.sources[matching_keys[0]] = _manifest_entry(previous)
                    store.fail_revision(row["revision_id"], "Rolled back by compact update operator.")
                else:
                    del manifest.sources[matching_keys[0]]
                    store.fail_revision(row["revision_id"], "Rolled back by compact update operator.")
                    store.deactivate_document(row["document_id"])
            save_manifest(manifest_path, manifest)
            garbage_collected = store.garbage_collect(revision_ids=[args.revision_id])
            result = {
                "rolled_back": [args.revision_id],
                "recovered_manifest": recovered,
                "garbage_collected": garbage_collected,
            }
    finally:
        store.close()
    print(json.dumps({"target": expected_confirmation, **result}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
