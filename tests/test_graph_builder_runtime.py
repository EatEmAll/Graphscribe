from __future__ import annotations

from pathlib import Path

import build_graph as bg
import graph_builder_runtime as runtime
import scripts.sync_notebook_graph as sng


class FakeAccess:
    def __init__(self) -> None:
        self.created = []

    def create_source_node(self, node) -> None:
        self.created.append(node)


def test_upload_file_registers_document_node(monkeypatch, tmp_path: Path) -> None:
    source_path = tmp_path / "paper.txt"
    source_path.write_text("content", encoding="utf-8")
    access = FakeAccess()

    monkeypatch.setattr(runtime, "create_graph_database_connection", lambda credentials: object())
    monkeypatch.setattr(runtime, "graphDBdataAccess", lambda graph: access)
    monkeypatch.setattr(runtime, "close_db_connection", lambda graph, api_name: None)

    api = runtime.GraphBuilderAPI(
        neo4j_uri="bolt://host.docker.internal:17687",
        neo4j_user="neo4j",
        neo4j_password="pw-123",
        neo4j_database="neo4j",
        sources_dir=tmp_path,
    )
    result = api.upload_file(source_path, "google_flash")

    assert result["status"] == "Success"
    assert len(access.created) == 1
    assert access.created[0].file_name == "paper.txt"
    assert access.created[0].file_source == "local file"


def test_retry_processing_updates_document_state(monkeypatch) -> None:
    captured = []

    monkeypatch.setattr(runtime, "create_graph_database_connection", lambda credentials: object())
    monkeypatch.setattr(runtime, "set_status_retry", lambda graph, file_name, retry_condition: captured.append((file_name, retry_condition)))
    monkeypatch.setattr(runtime, "close_db_connection", lambda graph, api_name: None)

    api = runtime.GraphBuilderAPI(
        neo4j_uri="bolt://host.docker.internal:17687",
        neo4j_user="neo4j",
        neo4j_password="pw-123",
        neo4j_database="neo4j",
    )
    result = api.retry_processing("paper.txt", sng.RETRY_CONDITION)

    assert result["status"] == "Success"
    assert captured == [("paper.txt", sng.RETRY_CONDITION)]


def test_post_processing_calls_index_creation(monkeypatch) -> None:
    calls = []

    monkeypatch.setattr(runtime, "create_vector_fulltext_indexes", lambda credentials, provider, model: calls.append((provider, model)))

    api = runtime.GraphBuilderAPI(
        neo4j_uri="bolt://host.docker.internal:17687",
        neo4j_user="neo4j",
        neo4j_password="pw-123",
        neo4j_database="neo4j",
    )
    result = api.post_processing(["enable_hybrid_search_and_fulltext_search_in_bloom"], "sentence-transformer", "all-MiniLM-L6-v2")

    assert result["status"] == "Success"
    assert calls == [("sentence-transformer", "all-MiniLM-L6-v2")]


def test_build_graph_command_omits_backend_url(tmp_path: Path) -> None:
    args = sng.build_parser().parse_args(
        [
            "create",
            "--dataset-dir",
            str(tmp_path),
            "--notebook-title",
            "demo",
            "--export-dir",
            str(tmp_path / "export"),
        ]
    )
    runtime_info = sng.Neo4jRuntime(
        uri="bolt://host.docker.internal:17687",
        username="neo4j",
        password="pw-123",
        database="neo4j",
    )

    command = sng.build_graph_command(args, tmp_path / "sources", runtime_info)

    assert "--backend-url" not in command
    assert command[:2] == [sng.sys.executable, str(sng.REPO_ROOT / "build_graph.py")]


class FakeBuildAPI:
    def __init__(self, retry_condition: str | None = None) -> None:
        self.retry_condition = retry_condition
        self.extract_calls: list[dict[str, object]] = []

    def sources_list(self) -> list[dict[str, str]]:
        row = {"fileName": "paper.txt", "status": "Ready to Reprocess"}
        if self.retry_condition:
            row["retry_condition"] = self.retry_condition
        return [row]

    def upload_file(self, file_path: Path, model: str) -> dict[str, object]:
        return {"status": "Success"}

    def extract(self, file_name: str, model: str, **kwargs) -> dict[str, object]:
        self.extract_calls.append({"file_name": file_name, "model": model, **kwargs})
        return {
            "status": "Success",
            "data": {"status": "Completed", "nodeCount": 1, "relationshipCount": 2},
        }


def test_phase_upload_and_extract_propagates_retry_condition(tmp_path: Path) -> None:
    source_path = tmp_path / "paper.txt"
    source_path.write_text("content", encoding="utf-8")
    api = FakeBuildAPI(sng.RETRY_CONDITION)

    completed, skipped, failed = bg.phase_upload_and_extract(
        api,
        tmp_path,
        "google_flash",
        min_file_size=1,
        parallel=1,
        poll_interval=0,
        token_chunk_size=2000,
        chunk_overlap=200,
        chunks_to_combine=1,
    )

    assert (completed, skipped, failed) == (1, 0, 0)
    assert api.extract_calls == [
        {
            "file_name": "paper.txt",
            "model": "google_flash",
            "retry_condition": sng.RETRY_CONDITION,
            "token_chunk_size": 2000,
            "chunk_overlap": 200,
            "chunks_to_combine": 1,
        }
    ]
