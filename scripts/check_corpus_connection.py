#!/usr/bin/env python3
"""Read-only connection smoke test for a corpus manifest.

Resolves the Neo4j connection exactly as the REST, MCP, extraction, and sync
paths do (manifest metadata plus the manifest's `password_env`), then reports
server identity, retrieval index readiness, and node/relationship counts.
Performs no writes.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from neo4j import GraphDatabase

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from notebooklm_graph_pipe.ingestion.manifest import load_manifest
from notebooklm_graph_pipe.runtime.neo4j_connection import (
    Neo4jConnectionError,
    resolve_connection_mapping,
    verify_connection,
)

COUNT_LABELS = (
    "Corpus",
    "Document",
    "DocumentRevision",
    "CorpusSource",
    "ParentChunk",
    "Chunk",
    "__Entity__",
)


def _label_counts(session) -> dict[str, int]:
    counts: dict[str, int] = {}
    for label in COUNT_LABELS:
        record = session.run(f"MATCH (n:`{label}`) RETURN count(n) AS total").single()
        counts[label] = int(record["total"]) if record else 0
    return counts


def _relationship_counts(session, limit: int) -> list[dict[str, object]]:
    rows = session.run(
        "MATCH ()-[r]->() RETURN type(r) AS type, count(r) AS total "
        "ORDER BY total DESC LIMIT $limit",
        limit=limit,
    )
    return [{"type": row["type"], "count": int(row["total"])} for row in rows]


def _retrieval_indexes(session, expected: dict[str, str]) -> list[dict[str, object]]:
    actual = {
        str(row["name"]): row
        for row in session.run(
            "SHOW INDEXES YIELD name, type, state, populationPercent "
            "RETURN name, type, state, populationPercent"
        )
    }
    report = []
    for role, name in expected.items():
        row = actual.get(name)
        report.append(
            {
                "role": role,
                "name": name,
                "present": row is not None,
                "type": str(row["type"]) if row else None,
                "state": str(row["state"]) if row else None,
                "population_percent": float(row["populationPercent"]) if row else None,
                "online": bool(row) and str(row["state"]) == "ONLINE",
            }
        )
    return report


def _corpus_present(session, corpus_id: str) -> bool:
    record = session.run(
        "MATCH (c:Corpus {id: $corpus_id}) RETURN count(c) AS total", corpus_id=corpus_id
    ).single()
    return bool(record and int(record["total"]) > 0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify a corpus manifest's Neo4j connection without writing.",
    )
    parser.add_argument("--manifest-path", type=Path, required=True, help="Path to the corpus manifest.json")
    parser.add_argument("--json", action="store_true", help="Emit the report as JSON instead of text")
    parser.add_argument(
        "--relationship-limit", type=int, default=10, help="Number of relationship types to report"
    )
    return parser


def collect_report(manifest_path: Path, relationship_limit: int) -> dict[str, object]:
    manifest = load_manifest(manifest_path)
    if manifest is None:
        raise ValueError(f"Manifest does not exist: {manifest_path}")
    connection = resolve_connection_mapping(manifest.neo4j)
    started = time.monotonic()
    server = verify_connection(connection)
    connect_seconds = time.monotonic() - started
    expected_indexes = {
        "vector": manifest.retrieval_vector_index,
        "keyword": manifest.retrieval_keyword_index,
    }
    with GraphDatabase.driver(connection.uri, auth=(connection.username, connection.password)) as driver:
        with driver.session(database=connection.database) as session:
            report: dict[str, object] = {
                "manifest_path": str(manifest_path),
                "corpus": {
                    "key": manifest.corpus_key,
                    "id": manifest.corpus_id,
                    "title": manifest.title,
                    "manifest_sources": len(manifest.sources),
                    "present_in_database": _corpus_present(session, manifest.corpus_id),
                },
                "connection": {
                    "uri": connection.uri,
                    "username": connection.username,
                    "database": connection.database,
                    "deployment": connection.deployment,
                    "password_env": str(manifest.neo4j.get("password_env") or "NEO4J_PASSWORD"),
                    "connect_seconds": round(connect_seconds, 2),
                    "server_address": server["address"],
                    "server_agent": server["agent"],
                },
                "retrieval": {
                    "unit": manifest.retrieval_unit,
                    "vector_provider": manifest.retrieval_vector_provider,
                    "embedding_model": manifest.embedding_model,
                    "embedding_dimension": manifest.embedding_dimension,
                    "indexes": _retrieval_indexes(session, expected_indexes),
                },
                "counts": {
                    "labels": _label_counts(session),
                    "relationships": _relationship_counts(session, relationship_limit),
                },
            }
    indexes = report["retrieval"]["indexes"]
    report["ok"] = report["corpus"]["present_in_database"] and all(entry["online"] for entry in indexes)
    return report


def render_text(report: dict[str, object]) -> str:
    connection = report["connection"]
    corpus = report["corpus"]
    retrieval = report["retrieval"]
    lines = [
        f"corpus     {corpus['key']} ({corpus['id']})",
        f"           {corpus['manifest_sources']} manifest sources; "
        f"Corpus node {'found' if corpus['present_in_database'] else 'MISSING'}",
        f"connection {connection['uri']} db={connection['database']} user={connection['username']}",
        f"           password_env={connection['password_env']} deployment={connection['deployment']}",
        f"           connected in {connection['connect_seconds']}s to {connection['server_address']} "
        f"({connection['server_agent']})",
        f"retrieval  unit={retrieval['unit']} provider={retrieval['vector_provider']} "
        f"model={retrieval['embedding_model']} dim={retrieval['embedding_dimension']}",
    ]
    for entry in retrieval["indexes"]:
        if entry["present"]:
            status = f"{entry['type']} {entry['state']} {entry['population_percent']}%"
        else:
            status = "MISSING"
        lines.append(f"           {entry['role']:<8} {entry['name']} -> {status}")
    lines.append("counts")
    for label, total in report["counts"]["labels"].items():
        lines.append(f"           {label:<18} {total}")
    for entry in report["counts"]["relationships"]:
        lines.append(f"           {entry['type']:<18} {entry['count']}")
    lines.append("ok" if report["ok"] else "NOT READY")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = collect_report(args.manifest_path.resolve(), args.relationship_limit)
    except (Neo4jConnectionError, ValueError) as error:
        print(f"connection check failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2) if args.json else render_text(report))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
