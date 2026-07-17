#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from notebooklm_graph_pipe.ingestion.manifest import load_manifest
from notebooklm_graph_pipe.retrieval.hybrid import SearchRequest
from notebooklm_graph_pipe.service.registry import CorpusRegistryEntry
from notebooklm_graph_pipe.service.runtime import RuntimeFactory


def percentile(values: Sequence[float], percentage: float) -> float:
    if not values:
        raise ValueError("Cannot calculate a percentile from no values.")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentage / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def load_queries(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw = payload.get("questions") if isinstance(payload, dict) else payload
    if not isinstance(raw, list) or not raw:
        raise ValueError("Query file must contain a non-empty JSON list or questions array.")
    queries = [str(item.get("text") if isinstance(item, dict) else item).strip() for item in raw]
    if any(not query for query in queries):
        raise ValueError("Benchmark queries cannot be empty.")
    return queries


def corpus_counts(driver: Any, database: str, corpus_id: str) -> dict[str, int]:
    query = """
    MATCH (corpus:Corpus {id: $corpus_id})-[:HAS_DOCUMENT]->(document:Document)
    OPTIONAL MATCH (document)-[:HAS_REVISION]->(revision:DocumentRevision {status: 'ACTIVE'})
    OPTIONAL MATCH (revision)-[:HAS_PARENT]->(parent:ParentChunk)
    OPTIONAL MATCH (parent)-[:HAS_CHILD]->(chunk:Chunk)
    RETURN count(DISTINCT document) AS documents,
           count(DISTINCT revision) AS active_revisions,
           count(DISTINCT parent) AS parent_chunks,
           count(DISTINCT chunk) AS child_chunks
    """
    with driver.session(database=database) as session:
        row = session.run(query, corpus_id=corpus_id).single()
    if row is None:
        raise RuntimeError(f"Corpus not found in Neo4j: {corpus_id}")
    return {name: int(row[name]) for name in ("documents", "active_revisions", "parent_chunks", "child_chunks")}


def run_benchmark(
    runtime: Any,
    queries: Sequence[str],
    *,
    modes: Sequence[str],
    warmups: int,
    iterations: int,
    top_k: int,
) -> dict[str, Any]:
    measurements: dict[str, Any] = {}
    for mode in modes:
        for index in range(warmups):
            runtime.retriever.search(SearchRequest(queries[index % len(queries)], mode=mode, top_k=top_k))
        latencies: list[float] = []
        started = time.perf_counter()
        for index in range(iterations):
            query = queries[index % len(queries)]
            call_started = time.perf_counter()
            runtime.retriever.search(SearchRequest(query, mode=mode, top_k=top_k))
            latencies.append((time.perf_counter() - call_started) * 1000)
        elapsed = time.perf_counter() - started
        measurements[mode] = {
            "iterations": iterations,
            "elapsed_seconds": elapsed,
            "queries_per_second": iterations / elapsed if elapsed else None,
            "latency_ms": {
                "min": min(latencies),
                "p50": percentile(latencies, 50),
                "p95": percentile(latencies, 95),
                "p99": percentile(latencies, 99),
                "max": max(latencies),
            },
        }
    return measurements


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark retrieval latency against a staged Neo4j corpus.")
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--queries-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("100k", "1m", "5m", "custom"), default="custom")
    parser.add_argument("--modes", nargs="+", choices=("vector", "lexical", "hybrid", "graph_hybrid"), default=("hybrid", "graph_hybrid"))
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=12)
    args = parser.parse_args()
    if args.warmups < 0 or args.iterations < 1:
        parser.error("--warmups must be non-negative and --iterations must be positive")

    manifest_path = args.manifest_path.resolve()
    manifest = load_manifest(manifest_path)
    if manifest is None:
        parser.error(f"Manifest not found: {manifest_path}")
    entry = CorpusRegistryEntry(manifest.corpus_key, manifest_path, manifest)
    factory = RuntimeFactory()
    try:
        runtime = factory.get(entry)
        counts = corpus_counts(runtime.driver, manifest.neo4j.get("database") or "neo4j", manifest.corpus_id)
        queries = load_queries(args.queries_file)
        report = {
            "stage": args.stage,
            "corpus_key": manifest.corpus_key,
            "manifest_path": str(manifest_path),
            "counts": counts,
            "query_count": len(queries),
        }
        report["measurements"] = run_benchmark(
            runtime,
            queries,
            modes=args.modes,
            warmups=args.warmups,
            iterations=args.iterations,
            top_k=args.top_k,
        )
    finally:
        factory.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
