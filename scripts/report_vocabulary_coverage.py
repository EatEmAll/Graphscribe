#!/usr/bin/env python3
"""Read-only report of canonical-vocabulary entity coverage and graph-hybrid retrieval.

Coverage counts, per vocabulary concept, the active parents whose text mentions it and the active
parents linked to its canonical entity. The optional retrieval section runs graph-hybrid search for
each fixed question and records what graph expansion contributed to the selected results.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT_PATH = Path(__file__).resolve().parents[1]
if str(REPO_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_PATH))

from neo4j import GraphDatabase

from notebooklm_graph_pipe.ingestion.manifest import load_manifest
from notebooklm_graph_pipe.ingestion.neo4j_store import Neo4jCorpusStore
from notebooklm_graph_pipe.retrieval.entity_vocabulary import EntityVocabulary
from notebooklm_graph_pipe.runtime.neo4j_connection import resolve_connection_mapping


def coverage_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    mentioning = sum(row["mentioning_parents"] for row in rows)
    covered = sum(row["covered_parents"] for row in rows)
    return {
        "concepts": len(rows),
        "concepts_present": sum(1 for row in rows if row["mentioning_parents"]),
        "concepts_covered": sum(1 for row in rows if row["mentioning_parents"] and row["covered_parents"]),
        "mentioning_parent_links": mentioning,
        "covered_parent_links": covered,
        "coverage_ratio": covered / mentioning if mentioning else 0.0,
    }


def retrieval_row(question: dict[str, Any], result: dict[str, Any], vocabulary: EntityVocabulary) -> dict[str, Any]:
    concepts = vocabulary.mentioned(str(question["text"]))
    results = result.get("results") or []
    graph_results = [item for item in results if "graph" in (item.get("channels") or [])]
    return {
        "question_id": question["question_id"],
        "concepts": [concept.id for concept in concepts],
        "graph_candidates": int((result.get("diagnostics") or {}).get("graph_candidates") or 0),
        "results": len(results),
        "graph_channel_results": len(graph_results),
        "graph_only_results": sum(1 for item in graph_results if item.get("channels") == ["graph"]),
        "concept_results": sum(
            1 for item in results if any(concept in vocabulary.mentioned(str(item.get("text") or "")) for concept in concepts)
        ),
        "parent_ids": [item.get("parent_id") for item in results],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--vocabulary", required=True)
    parser.add_argument("--questions-file", help="Fixed questions whose graph-hybrid retrieval is recorded.")
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest_path = Path(args.manifest_path).resolve()
    manifest = load_manifest(manifest_path)
    if manifest is None:
        parser.error("Corpus manifest was not found.")
    vocabulary = EntityVocabulary.from_path(args.vocabulary)
    runtime = resolve_connection_mapping(manifest.neo4j)
    driver = GraphDatabase.driver(runtime.uri, auth=(runtime.username, runtime.password))
    store = Neo4jCorpusStore(driver, runtime.database, corpus_id=manifest.corpus_id)
    try:
        coverage = store.vocabulary_coverage(vocabulary)
    finally:
        store.close()
    report: dict[str, Any] = {
        "corpus_key": manifest.corpus_key,
        "vocabulary_fingerprint": vocabulary.fingerprint,
        "coverage": coverage,
        "coverage_summary": coverage_summary(coverage),
    }
    if args.questions_file:
        from notebooklm_graph_pipe.paths import REPO_ROOT
        from notebooklm_graph_pipe.service.core import CorpusService
        from notebooklm_graph_pipe.service.jobs import CorpusJobManager
        from notebooklm_graph_pipe.service.registry import CorpusRegistry
        from notebooklm_graph_pipe.service.runtime import RuntimeFactory

        questions = json.loads(Path(args.questions_file).read_text(encoding="utf-8"))["questions"]
        registry = CorpusRegistry(manifest_path.parent.parent)
        service = CorpusService(registry, RuntimeFactory(None), CorpusJobManager(registry, REPO_ROOT))
        try:
            report["retrieval"] = [
                retrieval_row(
                    question,
                    service.search(
                        manifest.corpus_key,
                        {
                            "query": question["text"],
                            "mode": "graph_hybrid",
                            "top_k": args.top_k,
                            "graph_hops": 1,
                            "include_diagnostics": True,
                        },
                    ),
                    vocabulary,
                )
                for question in questions
            ]
        finally:
            service.close()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["coverage_summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
