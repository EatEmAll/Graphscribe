#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from neo4j import GraphDatabase

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from notebooklm_graph_pipe.ingestion.manifest import load_manifest
from notebooklm_graph_pipe.ingestion.neo4j_store import Neo4jCorpusStore
from notebooklm_graph_pipe.retrieval.graph_extraction import GraphCapacityError, GraphExtractionWorker
from notebooklm_graph_pipe.runtime.neo4j_connection import resolve_connection_mapping, verify_corpus_connection


def main() -> int:
    parser = argparse.ArgumentParser(description="Process pending parent-chunk graph extraction jobs.")
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--llm-routing-config")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--confirm-target", help="When supplied, must equal <neo4j-uri>|<database>.")
    parser.add_argument("--max-nodes", type=int)
    parser.add_argument("--max-relationships", type=int)
    parser.add_argument("--capacity-headroom", type=float, default=0.25)
    args = parser.parse_args()
    manifest = load_manifest(Path(args.manifest_path))
    if manifest is None:
        parser.error("Corpus manifest was not found.")
    runtime = resolve_connection_mapping(manifest.neo4j)
    expected_confirmation = f"{runtime.uri}|{runtime.database}"
    if args.confirm_target is not None and args.confirm_target != expected_confirmation:
        parser.error(f"Target confirmation mismatch; expected exactly: {expected_confirmation}")
    if (args.max_nodes is None) != (args.max_relationships is None):
        parser.error("--max-nodes and --max-relationships must be supplied together.")
    if not 0 <= args.capacity_headroom < 1:
        parser.error("--capacity-headroom must be in [0, 1).")
    verify_corpus_connection(
        runtime,
        dimension=manifest.embedding_dimension,
        require_write=True,
        retrieval_unit=manifest.retrieval_unit,
        vector_index=manifest.retrieval_vector_index,
        keyword_index=manifest.retrieval_keyword_index,
        require_retrieval_vector=manifest.retrieval_vector_provider == "neo4j",
    )
    driver = GraphDatabase.driver(runtime.uri, auth=(runtime.username, runtime.password))
    store = Neo4jCorpusStore(
        driver,
        runtime.database,
        corpus_id=manifest.corpus_id,
    )
    try:
        capacity_guard = None
        if args.max_nodes is not None:
            guarded_nodes = int(args.max_nodes * (1 - args.capacity_headroom))
            guarded_relationships = int(args.max_relationships * (1 - args.capacity_headroom))

            def capacity_guard(node_count: int, relationship_count: int) -> None:
                with driver.session(database=runtime.database) as session:
                    row = session.run(
                        "MATCH (n) WITH count(n) AS nodes MATCH ()-[r]->() "
                        "RETURN nodes, count(r) AS relationships"
                    ).single()
                projected_nodes = int(row["nodes"]) + node_count
                projected_relationships = int(row["relationships"]) + relationship_count + node_count
                if projected_nodes >= guarded_nodes or projected_relationships >= guarded_relationships:
                    raise GraphCapacityError(
                        "Projected graph extraction exceeds capacity headroom: "
                        f"nodes={projected_nodes}/{guarded_nodes}, "
                        f"relationships={projected_relationships}/{guarded_relationships}"
                    )

        worker = GraphExtractionWorker.from_routing_config(
            store,
            args.llm_routing_config,
            capacity_guard=capacity_guard,
            cache_path=str(Path(args.manifest_path).resolve().parent / str(manifest.execution["cache_path"])),
            metrics_path=str(Path(args.manifest_path).resolve().parent / str(manifest.execution["metrics_path"])),
            max_concurrency=int(manifest.execution["default_max_concurrency"]),
        )
        summary = asyncio.run(worker.run_batch(args.limit))
    finally:
        store.close()
    print(json.dumps(summary, indent=2))
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
