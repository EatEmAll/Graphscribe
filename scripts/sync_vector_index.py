from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from neo4j import GraphDatabase

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from notebooklm_graph_pipe.ingestion.embeddings import EmbeddingConfig, MiniLMEmbedder
from notebooklm_graph_pipe.ingestion.manifest import load_manifest
from notebooklm_graph_pipe.retrieval.lancedb_store import LanceDBVectorStore
from notebooklm_graph_pipe.runtime.contracts import VectorRecord
from notebooklm_graph_pipe.runtime.neo4j_connection import resolve_connection_mapping


def active_records(driver, database: str, corpus_id: str, unit_type: str, fingerprint: str):
    if unit_type == "parent":
        match = (
            "MATCH (:Corpus {id: $corpus_id})-[:HAS_DOCUMENT]->(document:Document)"
            "-[:ACTIVE_REVISION]->(revision:DocumentRevision)-[:HAS_PARENT]->(unit:ParentChunk)"
        )
        parent = "unit.id"
    else:
        match = (
            "MATCH (:Corpus {id: $corpus_id})-[:HAS_DOCUMENT]->(document:Document)"
            "-[:ACTIVE_REVISION]->(revision:DocumentRevision)<-[:IN_REVISION]-(unit:Chunk)"
        )
        parent = "unit.parent_id"
    with driver.session(database=database) as session:
        rows = session.run(
            f"""
            {match}
            WHERE revision.vector_ready = true AND unit.embedding IS NOT NULL
            RETURN unit.id AS record_id, revision.id AS revision_id,
                   document.id AS document_id, {parent} AS parent_id,
                   unit.id AS chunk_id, unit.embedding AS vector, unit.text AS text,
                   document.title AS title, document.source_uri AS source_uri,
                   document.source_type AS source_type, document.language AS language
            ORDER BY record_id
            """,
            corpus_id=corpus_id,
        )
        for row in rows:
            yield VectorRecord(
                str(row["record_id"]),
                corpus_id,
                str(row["revision_id"]),
                str(row["document_id"]),
                str(row["parent_id"]),
                str(row["chunk_id"]),
                unit_type,
                fingerprint,
                tuple(float(value) for value in row["vector"]),
                str(row.get("text") or ""),
                str(row.get("title") or ""),
                str(row.get("source_uri") or ""),
                str(row.get("source_type") or ""),
                str(row.get("language") or ""),
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Synchronize active Neo4j vectors into configured LanceDB.")
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1000)
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive.")
    manifest_path = args.manifest_path.resolve()
    manifest = load_manifest(manifest_path)
    if manifest is None:
        raise ValueError(f"Manifest does not exist: {manifest_path}")
    if manifest.retrieval_vector_provider != "lancedb" or not manifest.retrieval_vector_location:
        raise ValueError("Manifest must configure retrieval.vector_provider=lancedb and vector_location.")
    location = Path(manifest.retrieval_vector_location)
    if not location.is_absolute():
        location = manifest_path.parent / location
    embedder = MiniLMEmbedder(
        EmbeddingConfig(
            model=manifest.embedding_model,
            dimension=manifest.embedding_dimension,
            normalize=manifest.embedding_normalized,
        )
    )
    vector_store = LanceDBVectorStore(location, dimension=manifest.embedding_dimension)
    previous_revisions = vector_store.revisions(manifest.corpus_id)
    connection = resolve_connection_mapping(manifest.neo4j)
    active_revisions = set()
    processed = 0
    batch = []
    with GraphDatabase.driver(connection.uri, auth=(connection.username, connection.password)) as driver:
        for record in active_records(
            driver,
            connection.database,
            manifest.corpus_id,
            manifest.retrieval_unit,
            embedder.fingerprint,
        ):
            active_revisions.add(record.revision_id)
            batch.append(record)
            if len(batch) >= args.batch_size:
                vector_store.upsert(batch)
                processed += len(batch)
                batch.clear()
        vector_store.upsert(batch)
        processed += len(batch)
    stale = sorted(previous_revisions - active_revisions)
    vector_store.delete_revisions(manifest.corpus_id, stale)
    print(
        json.dumps(
            {
                "records_upserted": processed,
                "active_revisions": len(active_revisions),
                "stale_revisions_deleted": stale,
                "health": vector_store.health(),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
