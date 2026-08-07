#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from filelock import FileLock, Timeout
from neo4j import GraphDatabase

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from notebooklm_graph_pipe.ingestion.chunking import HierarchicalChunker, load_minilm_tokenizer
from notebooklm_graph_pipe.ingestion.compact_sync import CompactCorpusUpdater
from notebooklm_graph_pipe.ingestion.embeddings import MiniLMEmbedder
from notebooklm_graph_pipe.ingestion.manifest import load_manifest
from notebooklm_graph_pipe.ingestion.neo4j_store import Neo4jCorpusStore
from notebooklm_graph_pipe.runtime.neo4j_connection import resolve_connection_mapping, verify_corpus_connection


def _inventory(driver, database: str) -> dict[str, int]:
    with driver.session(database=database) as session:
        row = session.run(
            "MATCH (n) WITH count(n) AS nodes MATCH ()-[r]->() RETURN nodes, count(r) AS relationships"
        ).single()
        return {"nodes": int(row["nodes"]), "relationships": int(row["relationships"])}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Add explicit sources to an Aura-authoritative compact parent corpus."
    )
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--source", action="append", required=True)
    parser.add_argument("--confirm-target", required=True, help="Must equal <neo4j-uri>|<database>.")
    parser.add_argument("--max-nodes", type=int, default=150_000)
    parser.add_argument("--max-relationships", type=int, default=300_000)
    parser.add_argument("--capacity-headroom", type=float, default=0.25)
    args = parser.parse_args(argv)

    manifest_path = Path(args.manifest_path).resolve()
    manifest = load_manifest(manifest_path)
    if manifest is None:
        parser.error(f"Corpus manifest not found: {manifest_path}")
    source_root = Path(args.source_root).resolve()
    if not source_root.is_dir():
        parser.error(f"Source root not found: {source_root}")
    sources = [Path(value).resolve() for value in args.source]
    missing = [str(path) for path in sources if not path.is_file()]
    if missing:
        parser.error(f"Source files not found: {missing}")
    if not 0 <= args.capacity_headroom < 1:
        parser.error("--capacity-headroom must be in [0, 1).")

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
        require_retrieval_vector=manifest.retrieval_vector_provider == "neo4j",
    )

    driver = GraphDatabase.driver(runtime.uri, auth=(runtime.username, runtime.password))
    store = Neo4jCorpusStore(driver, runtime.database, corpus_id=manifest.corpus_id)
    try:
        before = _inventory(driver, runtime.database)
        limits = {
            "nodes": int(args.max_nodes * (1 - args.capacity_headroom)),
            "relationships": int(args.max_relationships * (1 - args.capacity_headroom)),
        }
        if any(before[key] >= limits[key] for key in limits):
            raise RuntimeError(f"Aura capacity headroom gate failed: inventory={before}, guarded_limits={limits}")
        updater = CompactCorpusUpdater(
            store=store,
            embedder=MiniLMEmbedder(),
            chunker=HierarchicalChunker(load_minilm_tokenizer()),
        )

        def capacity_guard(additional_parents: int) -> None:
            current = _inventory(driver, runtime.database)
            projected = {
                "nodes": current["nodes"] + additional_parents + 2,
                "relationships": current["relationships"] + additional_parents + 2,
            }
            if any(projected[key] >= limits[key] for key in limits):
                raise RuntimeError(
                    f"Projected compact revision exceeds Aura headroom: projected={projected}, "
                    f"guarded_limits={limits}"
                )

        try:
            with FileLock(str(manifest_path.with_suffix(".update.lock")), timeout=0):
                report = updater.update(
                    sources=sources,
                    corpus_root=source_root,
                    manifest=manifest,
                    manifest_path=manifest_path,
                    capacity_guard=capacity_guard,
                )
        except Timeout as exc:
            raise RuntimeError("Another compact corpus update is already running.") from exc
        after = _inventory(driver, runtime.database)
    finally:
        store.close()

    payload = {"target": expected_confirmation, "before": before, "after": after, **report.to_dict()}
    print(json.dumps(payload, indent=2))
    return 1 if report.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
