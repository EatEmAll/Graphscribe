from __future__ import annotations

from pathlib import Path

import notebooklm_graph_pipe.cli.build_graph as bg
import notebooklm_graph_pipe.runtime.dataset_registry as dr
import notebooklm_graph_pipe.runtime.graph_builder_runtime as runtime
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


def test_extract_passes_embedding_provider_and_model_to_processing(monkeypatch, tmp_path: Path) -> None:
    source_path = tmp_path / "paper.txt"
    source_path.write_text("content", encoding="utf-8")
    captured: dict[str, object] = {}

    async def fake_processing_source(credentials, params, pages, merged_file_path=None, is_uploaded_from_local=None):
        captured["params"] = params
        captured["pages"] = pages
        return None, {"status": "Completed", "nodeCount": 1, "relationshipCount": 2}

    monkeypatch.setattr(runtime, "get_documents_from_file_by_path", lambda file_path, file_name: (file_name, ["page"], None))
    monkeypatch.setattr(runtime, "processing_source", fake_processing_source)
    monkeypatch.setattr(runtime, "create_graph_database_connection", lambda credentials: object())
    monkeypatch.setattr(runtime, "close_db_connection", lambda graph, api_name: None)

    api = runtime.GraphBuilderAPI(
        neo4j_uri="bolt://host.docker.internal:17687",
        neo4j_user="neo4j",
        neo4j_password="pw-123",
        neo4j_database="neo4j",
        sources_dir=tmp_path,
    )
    result = api.extract(
        "paper.txt",
        "google_flash",
        embedding_provider="openrouter",
        embedding_model="text-embedding-3-small",
    )

    assert result["status"] == "Success"
    params = captured["params"]
    assert params.embedding_provider == "openrouter"
    assert params.embedding_model == "text-embedding-3-small"


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
    assert command[:3] == [sng.sys.executable, "-m", "notebooklm_graph_pipe.cli.build_graph"]


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
        "openrouter",
        "text-embedding-3-small",
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
            "embedding_provider": "openrouter",
            "embedding_model": "text-embedding-3-small",
            "retry_condition": sng.RETRY_CONDITION,
            "token_chunk_size": 2000,
            "chunk_overlap": 200,
            "chunks_to_combine": 1,
        }
    ]


def test_build_graph_parse_args_accepts_dataset_registry_defaults(monkeypatch) -> None:
    entry = dr.DatasetRegistryEntry(
        key="bench-imdb-scifi",
        notebook=dr.RegistryNotebook(id="nb-1", title="bench-imdb-scifi"),
        neo4j=dr.RegistryNeo4j(
            uri="bolt://127.0.0.1:61706",
            username="neo4j",
            password="pw-123",
            database="neo4j",
        ),
    )
    monkeypatch.setattr(bg, "load_dataset_entry", lambda dataset_key, registry_path=None: entry)
    monkeypatch.setattr(bg, "default_sources_dir", lambda dataset_key: Path("C:/tmp/bench-imdb-scifi/sources"))
    monkeypatch.setattr(bg.sys, "argv", ["build_graph.py", "--dataset-key", "bench-imdb-scifi"])

    args = bg.parse_args()

    assert args.neo4j_uri == "bolt://127.0.0.1:61706"
    assert args.neo4j_user == "neo4j"
    assert args.neo4j_password == "pw-123"
    assert args.neo4j_database == "neo4j"
    assert args.sources_dir == "C:\\tmp\\bench-imdb-scifi\\sources"


def test_dataset_registry_defaults_use_config_dir() -> None:
    assert dr.default_registry_path() == dr.CONFIG_DIR / "benchmark_dataset_registry.json"
