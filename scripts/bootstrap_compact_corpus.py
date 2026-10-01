#!/usr/bin/env python3
"""Seed a new compact parent-retrieval corpus on an empty Neo4j database from source packages.

The staged-ingestion gate compares a preview with an active corpus, so it cannot seed an empty
one. This command builds the new corpus manifest from a template manifest's embedding, retrieval,
execution, graph, and community settings, refuses the template's own database and any database
that already holds a node, creates the parent schema and the Corpus node, and adds each
content-addressed source package. Graph extraction stays with ``process_graph_queue.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from filelock import FileLock, Timeout
from neo4j import GraphDatabase

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from notebooklm_graph_pipe.ingestion.adapters import SourcePackage
from notebooklm_graph_pipe.ingestion.chunking import HierarchicalChunker, load_minilm_tokenizer
from notebooklm_graph_pipe.ingestion.compact_sync import CompactCorpusUpdater
from notebooklm_graph_pipe.ingestion.embeddings import MiniLMEmbedder
from notebooklm_graph_pipe.ingestion.ids import corpus_id
from notebooklm_graph_pipe.ingestion.manifest import CorpusManifest, load_manifest, save_manifest
from notebooklm_graph_pipe.ingestion.neo4j_store import Neo4jCorpusStore
from notebooklm_graph_pipe.runtime.neo4j_connection import resolve_connection_mapping, verify_corpus_connection


def target_manifest(
    template: CorpusManifest, *, corpus_key: str, title: str, neo4j: dict[str, str]
) -> CorpusManifest:
    """Copy the template's settings onto a new, source-free corpus identity and target."""
    return replace(
        template,
        corpus_id=corpus_id(corpus_key),
        corpus_key=corpus_key,
        title=title,
        neo4j=neo4j,
        dataset_root=None,
        sources={},
        removed_sources=[],
        suppressed_sources=[],
    )


def _target(neo4j: dict) -> str:
    return f"{str(neo4j.get('uri') or '').strip().rstrip('/').lower()}|{neo4j.get('database') or 'neo4j'}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--template-manifest", required=True)
    parser.add_argument("--target-manifest", required=True, help="Must not exist yet.")
    parser.add_argument("--corpus-key", required=True)
    parser.add_argument("--title")
    parser.add_argument("--neo4j-uri", required=True)
    parser.add_argument("--neo4j-user", default="neo4j")
    parser.add_argument("--neo4j-database", default="neo4j")
    parser.add_argument("--neo4j-password-env", default="NEO4J_PASSWORD")
    parser.add_argument("--confirm-target", required=True, help="Must equal <neo4j-uri>|<database>.")
    parser.add_argument("--package", action="append", required=True, help="Source package directory.")
    args = parser.parse_args(argv)

    template = load_manifest(Path(args.template_manifest).resolve())
    if template is None:
        parser.error(f"Template manifest not found: {args.template_manifest}")
    manifest_path = Path(args.target_manifest).resolve()
    if manifest_path.exists():
        parser.error(f"Target manifest already exists: {manifest_path}")
    if args.corpus_key == template.corpus_key:
        parser.error("The new corpus key must differ from the template's.")
    packages = [Path(value).resolve() for value in args.package]
    missing = [str(path) for path in packages if not (path / "source.json").is_file()]
    if missing:
        parser.error(f"Not source package directories: {missing}")
    manifest = target_manifest(
        template,
        corpus_key=args.corpus_key,
        title=args.title or args.corpus_key,
        neo4j={
            "uri": args.neo4j_uri,
            "username": args.neo4j_user,
            "database": args.neo4j_database,
            "deployment": "external",
            "password_env": args.neo4j_password_env,
        },
    )
    try:
        CompactCorpusUpdater._validate_profile(manifest)
    except ValueError as exc:
        parser.error(str(exc))
    if _target(manifest.neo4j) == _target(template.neo4j):
        parser.error("The target database is the template corpus's own database.")
    runtime = resolve_connection_mapping(manifest.neo4j)
    expected_confirmation = f"{runtime.uri}|{runtime.database}"
    if args.confirm_target != expected_confirmation:
        parser.error(f"Target confirmation mismatch; expected exactly: {expected_confirmation}")

    embedder = MiniLMEmbedder()
    driver = GraphDatabase.driver(runtime.uri, auth=(runtime.username, runtime.password))
    store = Neo4jCorpusStore(driver, runtime.database, corpus_id=manifest.corpus_id)
    try:
        store.bootstrap_compact_corpus(
            corpus_key=manifest.corpus_key,
            corpus_title=manifest.title,
            embedding_fingerprint=embedder.fingerprint,
            dimension=manifest.embedding_dimension,
        )
        verify_corpus_connection(
            runtime,
            dimension=manifest.embedding_dimension,
            require_write=True,
            retrieval_unit=manifest.retrieval_unit,
            vector_index=manifest.retrieval_vector_index,
            keyword_index=manifest.retrieval_keyword_index,
        )
        save_manifest(manifest_path, manifest)
        updater = CompactCorpusUpdater(
            store=store, embedder=embedder, chunker=HierarchicalChunker(load_minilm_tokenizer())
        )
        try:
            with FileLock(str(manifest_path.with_suffix(".update.lock")), timeout=0):
                report = updater.update(
                    sources=[SourcePackage(path) for path in packages],
                    corpus_root=manifest_path.parent,
                    manifest=manifest,
                    manifest_path=manifest_path,
                )
        except Timeout as exc:
            raise RuntimeError("Another compact corpus update is already running.") from exc
    finally:
        store.close()

    payload = {
        "target": expected_confirmation,
        "manifest_path": str(manifest_path),
        "corpus_id": manifest.corpus_id,
        "corpus_key": manifest.corpus_key,
        "embedding_fingerprint": embedder.fingerprint,
        **report.to_dict(),
    }
    print(json.dumps(payload, indent=2))
    return 1 if report.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
