#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

from filelock import FileLock, Timeout
from neo4j import GraphDatabase

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from notebooklm_graph_pipe.ingestion.chunking import HierarchicalChunker, load_minilm_tokenizer
from notebooklm_graph_pipe.ingestion.embeddings import MiniLMEmbedder
from notebooklm_graph_pipe.ingestion.ids import corpus_id, normalize_corpus_key
from notebooklm_graph_pipe.ingestion.manifest import CorpusManifest, load_manifest
from notebooklm_graph_pipe.ingestion.neo4j_store import Neo4jCorpusStore
from notebooklm_graph_pipe.ingestion.sync import CorpusSynchronizer
from notebooklm_graph_pipe.runtime.neo4j_connection import (
    Neo4jConnectionSpec,
    connection_spec_to_mapping,
    resolve_connection,
    resolve_connection_mapping,
    verify_corpus_connection,
)
from scripts.sync_notebook_graph import DockerNeo4jProvisioner, Neo4jRuntime, notebook_title_hash


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Synchronize local sources into Neo4j vector and graph storage.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("create", "update"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--dataset-dir", required=True)
        sub.add_argument("--corpus-title")
        sub.add_argument("--corpus-key")
        sub.add_argument("--export-dir")
        sub.add_argument("--neo4j-uri")
        sub.add_argument("--neo4j-user")
        sub.add_argument("--neo4j-password")
        sub.add_argument("--neo4j-password-env", default="NEO4J_PASSWORD")
        sub.add_argument("--neo4j-database")
    return parser


def _runtime(args: argparse.Namespace, key: str, title: str, existing: CorpusManifest | None) -> Neo4jRuntime:
    password_env = str(getattr(args, "neo4j_password_env", None) or "NEO4J_PASSWORD")
    explicit_uri = args.neo4j_uri or os.environ.get("NEO4J_URI")
    external_requested = bool(explicit_uri or args.neo4j_user or args.neo4j_password or args.neo4j_database)
    if external_requested:
        if not explicit_uri:
            raise ValueError("External Neo4j runtime requires --neo4j-uri or NEO4J_URI.")
        connection = resolve_connection(
            Neo4jConnectionSpec(
                uri=explicit_uri,
                username=args.neo4j_user or os.environ.get("NEO4J_USERNAME", "neo4j"),
                database=args.neo4j_database or os.environ.get("NEO4J_DATABASE", "neo4j"),
                password_env=password_env,
            ),
            password=args.neo4j_password,
            allow_standard_env_overrides=False,
        )
        return Neo4jRuntime(
            connection.uri,
            connection.username,
            connection.password,
            connection.database,
            password_env=password_env,
            deployment="external",
        )
    if existing and existing.neo4j and existing.neo4j.get("deployment") != "managed-local":
        connection = resolve_connection_mapping(existing.neo4j)
        return Neo4jRuntime(
            connection.uri,
            connection.username,
            connection.password,
            connection.database,
            password_env=str(existing.neo4j.get("password_env") or "NEO4J_PASSWORD"),
            deployment=connection.deployment,
        )
    prior = None
    if existing and existing.neo4j:
        prior = Neo4jRuntime(
            uri=str(existing.neo4j.get("uri") or ""),
            username=str(existing.neo4j.get("username") or "neo4j"),
            password=str(existing.neo4j.get("password") or ""),
            database=str(existing.neo4j.get("database") or "neo4j"),
            password_env=password_env,
            deployment="managed-local",
            container_name=existing.neo4j.get("container_name"),
            container_id=existing.neo4j.get("container_id"),
            bolt_port=existing.neo4j.get("bolt_port"),
            http_port=existing.neo4j.get("http_port"),
            image=existing.neo4j.get("image"),
        )
    provisioner = DockerNeo4jProvisioner()
    provisioner.ensure_available()
    return provisioner.ensure_runtime(key, notebook_title_hash(title), prior)


def _runtime_manifest(runtime: Neo4jRuntime) -> dict[str, object]:
    return connection_spec_to_mapping(
        runtime,
        password_env=runtime.password_env,
        extra=asdict(runtime),
    )


def _run_sync(args: argparse.Namespace, dataset_dir: Path, key: str, export_dir: Path) -> int:
    manifest_path = export_dir / "manifest.json"
    existing = load_manifest(manifest_path)
    if existing is not None and existing.corpus_key != key:
        raise ValueError(
            f"Corpus key {key!r} does not match the existing manifest key {existing.corpus_key!r}."
        )
    title = args.corpus_title or (existing.title if existing else dataset_dir.name)
    if args.command == "update" and existing is None:
        raise ValueError(f"Corpus manifest not found: {manifest_path}")
    runtime = _runtime(args, key, title, existing)
    runtime_manifest = _runtime_manifest(runtime)
    if existing is not None:
        fields = ("uri", "username", "database")
        changed = [field for field in fields if existing.neo4j.get(field) != runtime_manifest.get(field)]
        if changed:
            raise ValueError(
                "Changing the Neo4j runtime during an in-place update is unsafe. "
                "Create a separate corpus/export directory for a blue-green migration."
            )
    manifest = existing or CorpusManifest(
        corpus_id=corpus_id(key),
        corpus_key=key,
        title=title,
        neo4j=runtime_manifest,
        dataset_root=str(dataset_dir),
    )
    manifest.dataset_root = str(dataset_dir)
    manifest.neo4j = runtime_manifest
    verify_corpus_connection(
        resolve_connection_mapping({**runtime_manifest, "password": runtime.password}),
        dimension=manifest.embedding_dimension,
        initialize_schema=True,
        require_write=True,
    )
    driver = GraphDatabase.driver(runtime.uri, auth=(runtime.username, runtime.password))
    store = Neo4jCorpusStore(driver, runtime.database, corpus_id=manifest.corpus_id)
    try:
        synchronizer = CorpusSynchronizer(
            store=store,
            embedder=MiniLMEmbedder(),
            chunker=HierarchicalChunker(load_minilm_tokenizer()),
        )
        report = synchronizer.sync(
            corpus_root=dataset_dir,
            manifest=manifest,
            manifest_path=manifest_path,
            artifact_root=export_dir / "normalized",
        )
    finally:
        store.close()
    report_path = export_dir / "last_sync.json"
    report_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    print(json.dumps(report.to_dict(), indent=2))
    return 1 if report.failed else 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dataset_dir = Path(args.dataset_dir).resolve()
    if not dataset_dir.is_dir():
        raise ValueError(f"Dataset directory not found: {dataset_dir}")
    key = normalize_corpus_key(args.corpus_key or args.corpus_title or dataset_dir.name)
    export_dir = Path(args.export_dir).resolve() if args.export_dir else REPO_ROOT / "data" / "corpora" / key
    export_dir.mkdir(parents=True, exist_ok=True)
    try:
        with FileLock(str(export_dir / "sync.lock"), timeout=0):
            return _run_sync(args, dataset_dir, key, export_dir)
    except Timeout as exc:
        raise RuntimeError(f"Another mutating process is already running for corpus {key}.") from exc


if __name__ == "__main__":
    raise SystemExit(main())
