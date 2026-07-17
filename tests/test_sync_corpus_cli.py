from __future__ import annotations

import argparse
from types import SimpleNamespace

import pytest

from notebooklm_graph_pipe.ingestion.ids import corpus_id
from notebooklm_graph_pipe.ingestion.manifest import CorpusManifest, save_manifest
import scripts.sync_notebook_graph as legacy
import scripts.sync_corpus_graph as corpus_cli


def test_legacy_main_delegates_without_notebooklm(monkeypatch, tmp_path) -> None:
    captured = []
    args = argparse.Namespace(
        command="create",
        dataset_dir=str(tmp_path),
        dataset_key=None,
        registry_path=None,
        notebook_title="Demo",
        notebook_id=None,
        export_dir=str(tmp_path / "out"),
        neo4j_uri="bolt://test",
        neo4j_user="neo4j",
        neo4j_password="pw",
        neo4j_password_env="NEO4J_PASSWORD",
        neo4j_database="neo4j",
        model="ignored",
        parallel=1,
        poll_interval=1,
        min_file_size=1,
        token_chunk_size=1,
        chunk_overlap=0,
        chunks_to_combine=1,
        embedding_provider=None,
        embedding_model=None,
        llm_routing_config=None,
        skip_build=False,
        skip_postprocess=False,
    )
    monkeypatch.setattr(legacy, "build_parser", lambda: type("Parser", (), {"parse_args": lambda self: args})())
    import scripts.sync_corpus_graph as corpus_cli

    monkeypatch.setattr(corpus_cli, "main", lambda delegated: captured.extend(delegated) or 0)
    assert legacy.main() == 0
    assert captured[:3] == ["create", "--dataset-dir", str(tmp_path)]
    assert "--corpus-title" in captured
    assert "Demo" in captured


def _corpus_args(command: str = "update") -> argparse.Namespace:
    return argparse.Namespace(
        command=command,
        corpus_title=None,
        corpus_key="demo",
        neo4j_uri=None,
        neo4j_user=None,
        neo4j_password=None,
        neo4j_password_env="NEO4J_PASSWORD",
        neo4j_database=None,
    )


def test_update_rejects_manifest_key_mismatch(tmp_path) -> None:
    export_dir = tmp_path / "export"
    save_manifest(
        export_dir / "manifest.json",
        CorpusManifest(corpus_id("other"), "other", "Other", {"uri": "bolt://old"}),
    )

    with pytest.raises(ValueError, match="does not match"):
        corpus_cli._run_sync(_corpus_args(), tmp_path, "demo", export_dir)


def test_update_rejects_in_place_runtime_cutover(monkeypatch, tmp_path) -> None:
    export_dir = tmp_path / "export"
    save_manifest(
        export_dir / "manifest.json",
        CorpusManifest(
            corpus_id("demo"),
            "demo",
            "Demo",
            {"uri": "bolt://blue", "username": "neo4j", "password": "blue", "database": "neo4j"},
        ),
    )
    monkeypatch.setattr(
        corpus_cli,
        "_runtime",
        lambda *args: legacy.Neo4jRuntime(
            uri="bolt://green", username="neo4j", password="green", database="neo4j"
        ),
    )

    with pytest.raises(ValueError, match="blue-green"):
        corpus_cli._run_sync(_corpus_args(), tmp_path, "demo", export_dir)


def test_existing_hosted_runtime_resolves_password_from_manifest_environment(monkeypatch) -> None:
    manifest = CorpusManifest(
        corpus_id("demo"),
        "demo",
        "Demo",
        {
            "deployment": "external",
            "uri": "neo4j+s://hosted.example.com",
            "username": "neo4j",
            "database": "neo4j",
            "password_env": "DEMO_NEO4J_PASSWORD",
        },
    )
    monkeypatch.setenv("DEMO_NEO4J_PASSWORD", "secret")

    runtime = corpus_cli._runtime(_corpus_args(), "demo", "Demo", manifest)

    assert runtime.uri == "neo4j+s://hosted.example.com"
    assert runtime.password == "secret"
    assert "password" not in corpus_cli._runtime_manifest(runtime)
