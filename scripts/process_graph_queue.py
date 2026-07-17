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
from notebooklm_graph_pipe.retrieval.graph_extraction import GraphExtractionWorker
from notebooklm_graph_pipe.runtime.neo4j_connection import resolve_connection_mapping, verify_corpus_connection


def main() -> int:
    parser = argparse.ArgumentParser(description="Process pending parent-chunk graph extraction jobs.")
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--llm-routing-config")
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()
    manifest = load_manifest(Path(args.manifest_path))
    if manifest is None:
        parser.error("Corpus manifest was not found.")
    runtime = resolve_connection_mapping(manifest.neo4j)
    verify_corpus_connection(
        runtime,
        dimension=manifest.embedding_dimension,
        require_write=True,
    )
    driver = GraphDatabase.driver(runtime.uri, auth=(runtime.username, runtime.password))
    store = Neo4jCorpusStore(
        driver,
        runtime.database,
        corpus_id=manifest.corpus_id,
    )
    try:
        worker = GraphExtractionWorker.from_routing_config(store, args.llm_routing_config)
        summary = asyncio.run(worker.run_batch(args.limit))
    finally:
        store.close()
    print(json.dumps(summary, indent=2))
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
