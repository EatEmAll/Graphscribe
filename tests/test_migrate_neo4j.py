from __future__ import annotations

import json
from pathlib import Path

import pytest

from notebooklm_graph_pipe.cli import migrate_neo4j as migration
from notebooklm_graph_pipe.runtime.neo4j_connection import ResolvedNeo4jConnection


def connection(uri: str = "neo4j+s://target.databases.neo4j.io") -> ResolvedNeo4jConnection:
    return ResolvedNeo4jConnection(uri, "neo4j", "secret", "neo4j")


def test_quote_token_escapes_backticks() -> None:
    assert migration.quote_token("Odd`Label") == "`Odd``Label`"


def test_portable_schema_statement_removes_version_specific_provider() -> None:
    assert migration.portable_schema_statement(
        "CREATE INDEX `x` FOR (n:`N`) ON (n.`p`) OPTIONS {indexProvider: 'range-1.0'}"
    ) == "CREATE INDEX `x` FOR (n:`N`) ON (n.`p`)"
    assert "indexProvider" not in migration.portable_schema_statement(
        "CREATE VECTOR INDEX `v` FOR (n:`N`) ON (n.`p`) "
        "OPTIONS {indexConfig: {`vector.dimensions`: 3}, indexProvider: 'vector-2.0'}"
    )


def test_portable_schema_statement_removes_aura_vector_tuning_options() -> None:
    statement = (
        "CREATE VECTOR INDEX `vector` FOR (n:`Chunk`) ON (n.`embedding`) OPTIONS {indexConfig: {"
        "`vector.default_search_expansion_factor`: 1.5,`vector.dimensions`: 384,"
        "`vector.hnsw.ef_construction`: 100,`vector.hnsw.m`: 16,"
        "`vector.quantization.type`: 'SCALAR',`vector.similarity_function`: 'COSINE'}}"
    )

    assert migration.portable_schema_statement(statement) == (
        "CREATE VECTOR INDEX `vector` FOR (n:`Chunk`) ON (n.`embedding`) "
        "OPTIONS {indexConfig: {`vector.dimensions`: 384, `vector.similarity_function`: 'COSINE'}}"
    )


def test_batched_preserves_all_rows() -> None:
    assert list(migration.batched(({"id": value} for value in range(5)), 2)) == [
        [{"id": 0}, {"id": 1}],
        [{"id": 2}, {"id": 3}],
        [{"id": 4}],
    ]


def test_fingerprint_is_order_independent_and_property_sensitive() -> None:
    rows = [
        {"source_id": "1", "labels": ["B", "A"], "properties": {"name": "one"}},
        {"source_id": "2", "labels": [], "properties": {"values": [1, 2]}},
    ]
    reordered = [
        {"source_id": "2", "labels": [], "properties": {"values": [1, 2]}},
        {"source_id": "1", "labels": ["A", "B"], "properties": {"name": "one"}},
    ]
    changed = [*reordered[:-1], {**reordered[-1], "properties": {"name": "different"}}]

    assert migration._fingerprint_rows(rows) == migration._fingerprint_rows(reordered)
    assert migration._fingerprint_rows(rows) != migration._fingerprint_rows(changed)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("neo4j 5.26.7", (5, 26, 7)), ("2026.05.1", (2026, 5, 1)), ("5.28", (5, 28, 0))],
)
def test_parse_neo4j_version(raw: str, expected: tuple[int, ...]) -> None:
    assert migration.parse_neo4j_version(raw) == expected


def test_verify_inventory_rejects_label_mismatch() -> None:
    source = migration.GraphInventory(1, 0, {"Document": 1}, {})
    target = migration.GraphInventory(1, 0, {"Chunk": 1}, {})
    with pytest.raises(migration.MigrationError, match="verification failed"):
        migration.verify_inventory(source, target)


def test_verify_corpus_inventory_rejects_missing_active_embedding() -> None:
    inventory = migration.CorpusInventory(1, 1, 1, 2, 1, 2, 1)
    with pytest.raises(migration.MigrationError, match="active chunks are missing embeddings"):
        migration.verify_corpus_inventory(inventory, inventory)


def test_complete_staging_resume_skips_recopy_only_for_exact_staged_state() -> None:
    source = migration.GraphInventory(2, 1, {"Document": 2}, {"LINKS": 1})
    target = migration.GraphInventory(2, 1, {"Document": 2, "__LGP_MIGRATION__": 2}, {"LINKS": 1})
    schema = {"vector": {"type": "VECTOR"}}

    assert migration.complete_staging_resume_ready(
        resume=True,
        staging_constraint=True,
        source_inventory=source,
        target_inventory=target,
        total_target_nodes=2,
        staged_target_nodes=2,
        unstaged_target_relationships=0,
        source_schema=schema,
        target_schema=schema,
    )
    assert not migration.complete_staging_resume_ready(
        resume=True,
        staging_constraint=True,
        source_inventory=source,
        target_inventory=target,
        total_target_nodes=2,
        staged_target_nodes=2,
        unstaged_target_relationships=0,
        source_schema=schema,
        target_schema={},
    )


def test_activate_manifest_writes_non_secret_target(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps({"version": 2, "neo4j": {"password": "old-secret", "container_name": "local"}}),
        encoding="utf-8",
    )

    migration.activate_manifest(path, connection(), "NEO4J_TARGET_PASSWORD")

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["version"] == 3
    assert payload["neo4j"] == {
        "deployment": "external",
        "uri": "neo4j+s://target.databases.neo4j.io",
        "username": "neo4j",
        "database": "neo4j",
        "password_env": "NEO4J_TARGET_PASSWORD",
    }
    assert "old-secret" not in path.read_text(encoding="utf-8")


def test_activate_corpus_manifest_preserves_corpus_and_embedding_metadata(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    payload = {
        "version": 3,
        "corpus": {"id": "corpus-id", "key": "demo", "title": "Demo"},
        "embedding": {"provider": "sentence-transformer", "model": "all-MiniLM-L6-v2", "dimension": 384},
        "sources": {"a.txt": {"document_id": "doc"}},
        "neo4j": {"uri": "bolt://local", "password": "old-secret"},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    migration.activate_manifest(path, connection(), "DEMO_NEO4J_PASSWORD")

    updated = json.loads(path.read_text(encoding="utf-8"))
    assert updated["corpus"] == payload["corpus"]
    assert updated["embedding"] == payload["embedding"]
    assert updated["sources"] == payload["sources"]
    assert updated["neo4j"]["password_env"] == "DEMO_NEO4J_PASSWORD"
    assert "old-secret" not in path.read_text(encoding="utf-8")


def test_portable_dry_run_rejects_non_empty_target_without_overwrite(monkeypatch) -> None:
    class Driver:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def verify_connectivity(self):
            return None

    drivers = iter([Driver(), Driver()])
    monkeypatch.setattr(migration.GraphDatabase, "driver", lambda *args, **kwargs: next(drivers))
    inventories = iter(
        [
            migration.GraphInventory(2, 1, {"Document": 2}, {"LINKS": 1}),
            migration.GraphInventory(1, 0, {"Existing": 1}, {}),
        ]
    )
    monkeypatch.setattr(migration, "read_inventory", lambda *args: next(inventories))
    monkeypatch.setattr(migration, "read_corpus_inventory", lambda *args: migration.CorpusInventory(0, 0, 0, 0, 0, 0, 0))
    monkeypatch.setattr(migration, "read_schema", lambda *args: [])
    monkeypatch.setattr(migration, "read_schema_signature", lambda *args: {})
    monkeypatch.setattr(migration, "read_staging_state", lambda *args: (1, 0, 0))
    monkeypatch.setattr(migration, "has_staging_constraint", lambda *args: False)
    monkeypatch.setattr(migration, "ensure_migration_names_available", lambda *args: None)

    with pytest.raises(migration.MigrationError, match="not empty"):
        migration.portable_migrate(connection("bolt://source:7687"), connection())


def test_main_requires_exact_overwrite_confirmation(monkeypatch) -> None:
    monkeypatch.setenv("NEO4J_TARGET_PASSWORD", "secret")
    monkeypatch.setenv("NEO4J_SOURCE_PASSWORD", "secret")
    with pytest.raises(migration.MigrationError, match="must exactly match"):
        migration.main(
            [
                "--source-uri",
                "bolt://localhost:7687",
                "--target-uri",
                "neo4j+s://target.databases.neo4j.io",
                "--overwrite-target",
                "--confirm-target",
                "wrong",
            ]
        )


def test_aura_upload_requires_explicit_overwrite(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "neo4j.dump").write_bytes(b"dump")
    monkeypatch.setattr(migration.shutil, "which", lambda value: "neo4j-admin")
    monkeypatch.setattr(
        migration.subprocess,
        "run",
        lambda *args, **kwargs: type("Result", (), {"returncode": 0, "stdout": "5.26.7", "stderr": ""})(),
    )
    with pytest.raises(migration.MigrationError, match="requires --overwrite-target"):
        migration.aura_upload(
            connection(),
            database="neo4j",
            dump_dir=tmp_path,
            neo4j_admin_bin="neo4j-admin",
            execute=False,
            overwrite_target=False,
        )


def test_aura_dry_run_allows_dump_planned_from_container(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(migration.shutil, "which", lambda value: "neo4j-admin")
    monkeypatch.setattr(
        migration.subprocess,
        "run",
        lambda *args, **kwargs: type("Result", (), {"returncode": 0, "stdout": "5.26.7", "stderr": ""})(),
    )

    summary = migration.aura_upload(
        connection(),
        database="neo4j",
        dump_dir=tmp_path,
        neo4j_admin_bin="neo4j-admin",
        execute=False,
        overwrite_target=True,
        dump_will_be_created=True,
    )

    assert summary["executed"] is False


def test_aura_dry_run_accepts_neo4j_5_26_lts(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "neo4j.dump").write_bytes(b"dump")
    monkeypatch.setattr(migration.shutil, "which", lambda value: "neo4j-admin")
    monkeypatch.setattr(
        migration.subprocess,
        "run",
        lambda *args, **kwargs: type("Result", (), {"returncode": 0, "stdout": "5.26.0", "stderr": ""})(),
    )

    summary = migration.aura_upload(
        connection(),
        database="neo4j",
        dump_dir=tmp_path,
        neo4j_admin_bin="neo4j-admin",
        execute=False,
        overwrite_target=True,
    )

    assert summary["executed"] is False


def test_aura_container_upload_keeps_password_out_of_command(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "neo4j.dump").write_bytes(b"dump")
    captured: dict[str, object] = {}

    def fake_run(command, **kwargs):
        if command[:2] == ["docker", "inspect"]:
            payload = [{"Config": {"Image": "neo4j:5.26.7"}}]
            return type("Result", (), {"returncode": 0, "stdout": json.dumps(payload), "stderr": ""})()
        if command[:3] == ["docker", "exec", "managed-neo4j"]:
            return type("Result", (), {"returncode": 0, "stdout": "5.26.7", "stderr": ""})()
        captured["command"] = command
        captured["env"] = kwargs["env"]
        return type("Result", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(migration.subprocess, "run", fake_run)
    monkeypatch.setattr(migration, "verify_connection", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        migration,
        "inspect_corpus_connection",
        lambda *args, **kwargs: (migration.CorpusInventory(0, 0, 0, 0, 0, 0, 0), 384, False),
    )

    summary = migration.aura_upload(
        connection(),
        database="neo4j",
        dump_dir=tmp_path,
        neo4j_admin_bin="neo4j-admin",
        execute=True,
        overwrite_target=True,
        neo4j_admin_container="managed-neo4j",
    )

    assert summary["verified"] is True
    assert "secret" not in " ".join(captured["command"])
    assert captured["env"]["NEO4J_PASSWORD"] == "secret"


def test_aura_upload_rejects_expected_corpus_without_vector_schema(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "neo4j.dump").write_bytes(b"dump")
    expected = migration.CorpusInventory(1, 1, 1, 1, 1, 1, 1)
    monkeypatch.setattr(migration.shutil, "which", lambda value: "neo4j-admin")
    monkeypatch.setattr(
        migration.subprocess,
        "run",
        lambda *args, **kwargs: type("Result", (), {"returncode": 0, "stdout": "5.26.0", "stderr": ""})(),
    )
    monkeypatch.setattr(migration, "verify_connection", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        migration,
        "inspect_corpus_connection",
        lambda *args, **kwargs: (expected, 384, False),
    )

    with pytest.raises(migration.MigrationError, match="without its required vector index"):
        migration.aura_upload(
            connection(),
            database="neo4j",
            dump_dir=tmp_path,
            neo4j_admin_bin="neo4j-admin",
            execute=True,
            overwrite_target=True,
            expected_corpus=expected,
        )


def test_portable_resume_rejects_non_staging_target(monkeypatch) -> None:
    class Driver:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def verify_connectivity(self):
            return None

    drivers = iter([Driver(), Driver()])
    monkeypatch.setattr(migration.GraphDatabase, "driver", lambda *args, **kwargs: next(drivers))
    monkeypatch.setattr(
        migration,
        "read_inventory",
        lambda *args: migration.GraphInventory(1, 0, {"Document": 1}, {}),
    )
    monkeypatch.setattr(migration, "read_corpus_inventory", lambda *args: migration.CorpusInventory(0, 0, 0, 0, 0, 0, 0))
    monkeypatch.setattr(migration, "read_schema", lambda *args: [])
    monkeypatch.setattr(migration, "read_schema_signature", lambda *args: {})
    monkeypatch.setattr(migration, "ensure_migration_names_available", lambda *args: None)
    monkeypatch.setattr(migration, "read_staging_state", lambda *args: (1, 0, 0))
    monkeypatch.setattr(migration, "has_staging_constraint", lambda *args: False)

    with pytest.raises(migration.MigrationError, match="only nodes from an interrupted"):
        migration.portable_migrate(
            connection("bolt://source:7687"),
            connection(),
            resume=True,
        )


def test_portable_failure_reports_phase_counts_and_resume_hint(monkeypatch) -> None:
    def fail(source, target, progress, **kwargs):
        progress.update(
            {
                "phase": "copy-relationships",
                "staging_started": True,
                "source": {"nodes": 3, "relationships": 2},
                "target_before": {"nodes": 0, "relationships": 0},
            }
        )
        raise RuntimeError("network interrupted")

    monkeypatch.setattr(migration, "_portable_migrate_impl", fail)

    with pytest.raises(migration.MigrationError) as captured:
        migration.portable_migrate(connection("bolt://source:7687"), connection(), execute=True)

    message = str(captured.value)
    assert "copy-relationships" in message
    assert '"nodes": 3' in message
    assert "--resume --execute" in message


def test_container_dump_restarts_database_after_dump_failure(tmp_path: Path, monkeypatch) -> None:
    commands: list[list[str]] = []

    def fake_run(command, **kwargs):
        commands.append(command)
        if command[:3] == ["docker", "exec", "managed-neo4j"]:
            return type("Result", (), {"returncode": 0, "stdout": "5.26.7", "stderr": ""})()
        if command[:2] == ["docker", "inspect"]:
            payload = [{"State": {"Running": True}, "Config": {"Image": "neo4j:5.26.7"}}]
            return type("Result", (), {"returncode": 0, "stdout": json.dumps(payload), "stderr": ""})()
        if command[:2] == ["docker", "run"]:
            return type("Result", (), {"returncode": 1, "stdout": "", "stderr": "boom"})()
        return type("Result", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(migration.subprocess, "run", fake_run)
    monkeypatch.setattr(migration, "verify_connection", lambda connection: {})

    with pytest.raises(migration.MigrationError, match="Command failed"):
        migration.create_container_dump(
            connection("bolt://localhost:7687"),
            container_name="managed-neo4j",
            dump_dir=tmp_path,
            execute=True,
        )

    assert ["docker", "stop", "managed-neo4j"] in commands
    assert ["docker", "start", "managed-neo4j"] in commands
