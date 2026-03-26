from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

import notebooklm_graph_pipe.runtime.dataset_registry as dr
import scripts.postprocess_graph as ppg


class FakeGraphAPI:
    def __init__(self, *, healthy: bool = True, connect_status: str = "Success", postprocess_status: str = "Success"):
        self.healthy = healthy
        self.connect_status = connect_status
        self.postprocess_status = postprocess_status
        self.connect_calls: list[tuple[str, str]] = []
        self.postprocess_calls: list[tuple[list[str], str, str]] = []

    def health_check(self) -> bool:
        return self.healthy

    def connect(self, embedding_provider: str = "", embedding_model: str = "") -> dict:
        self.connect_calls.append((embedding_provider, embedding_model))
        return {"status": self.connect_status}

    def post_processing(self, tasks: list[str], embedding_provider: str = "", embedding_model: str = "") -> dict:
        self.postprocess_calls.append((list(tasks), embedding_provider, embedding_model))
        return {"status": self.postprocess_status}


class FakeCommandRunner:
    def __init__(self, returncode: int = 0):
        self.returncode = returncode
        self.calls: list[tuple[list[str], Path, dict[str, str]]] = []

    def run(self, args: list[str], *, cwd: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
        self.calls.append((list(args), cwd, dict(env)))
        return subprocess.CompletedProcess(args=args, returncode=self.returncode, stdout="", stderr="")


def test_run_backend_postprocess_executes_index_workflow() -> None:
    api = FakeGraphAPI()

    ppg.run_backend_postprocess(
        api,
        embedding_provider="sentence-transformer",
        embedding_model="all-MiniLM-L6-v2",
    )

    assert api.connect_calls == [("sentence-transformer", "all-MiniLM-L6-v2")]
    assert api.postprocess_calls == [
        (
            ["enable_hybrid_search_and_fulltext_search_in_bloom"],
            "sentence-transformer",
            "all-MiniLM-L6-v2",
        )
    ]


def test_run_backend_postprocess_fails_when_backend_unhealthy() -> None:
    api = FakeGraphAPI(healthy=False)

    with pytest.raises(ppg.WorkflowError, match="Local graph runtime health check failed"):
        ppg.run_backend_postprocess(
            api,
            embedding_provider="sentence-transformer",
            embedding_model="all-MiniLM-L6-v2",
        )


def test_preflight_consolidation_validates_requirements(monkeypatch) -> None:
    config = ppg.ConsolidationConfig(
        max_iterations=5,
        run_dir=None,
        codex_bin="codex",
        dry_run=False,
        resume=False,
        required_consecutive_passes=0,
    )
    monkeypatch.setenv("GOOGLE_API_KEY", "secret")
    monkeypatch.setattr(ppg, "resolve_codex_executable", lambda _: "C:\\tools\\codex.cmd")

    codex_executable = ppg.preflight_consolidation(config)

    assert codex_executable == "C:\\tools\\codex.cmd"


def test_build_consolidation_command_uses_dry_run_defaults() -> None:
    config = ppg.ConsolidationConfig(
        max_iterations=7,
        run_dir="C:\\runs\\demo",
        codex_bin="codex",
        dry_run=True,
        resume=False,
        required_consecutive_passes=0,
    )

    command = ppg.build_consolidation_command(config, "C:\\tools\\codex.cmd")

    assert command == [
        ppg.sys.executable,
        "-m",
        "notebooklm_graph_pipe.consolidation.self_improving",
        "--max-iterations",
        "7",
        "--required-consecutive-passes",
        "2",
        "--codex-bin",
        "C:\\tools\\codex.cmd",
        "--run-dir",
        "C:\\runs\\demo",
        "--dry-run",
    ]


def test_run_consolidation_passes_runtime_env() -> None:
    runner = FakeCommandRunner()
    config = ppg.ConsolidationConfig(
        max_iterations=5,
        run_dir="C:\\runs\\demo",
        codex_bin="codex",
        dry_run=False,
        resume=True,
        required_consecutive_passes=0,
    )

    ppg.run_consolidation(
        config=config,
        codex_executable="C:\\tools\\codex.cmd",
        neo4j_uri="bolt://127.0.0.1:17687",
        neo4j_user="neo4j",
        neo4j_password="pw-123",
        neo4j_database="neo4j",
        runner=runner,
    )

    assert len(runner.calls) == 1
    command, cwd, env = runner.calls[0]
    assert cwd == ppg.REPO_ROOT
    assert command == [
        ppg.sys.executable,
        "-m",
        "notebooklm_graph_pipe.consolidation.self_improving",
        "--max-iterations",
        "5",
        "--required-consecutive-passes",
        "1",
        "--codex-bin",
        "C:\\tools\\codex.cmd",
        "--run-dir",
        "C:\\runs\\demo",
        "--resume",
    ]
    assert env["NEO4J_URI"] == "bolt://127.0.0.1:17687"
    assert env["NEO4J_USERNAME"] == "neo4j"
    assert env["NEO4J_PASSWORD"] == "pw-123"
    assert env["NEO4J_DATABASE"] == "neo4j"
    assert env["PYTHONUNBUFFERED"] == "1"


def test_preflight_consolidation_requires_google_api_key(monkeypatch) -> None:
    config = ppg.ConsolidationConfig(
        max_iterations=5,
        run_dir=None,
        codex_bin="codex",
        dry_run=False,
        resume=False,
        required_consecutive_passes=0,
    )
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    with pytest.raises(ppg.WorkflowError, match="GOOGLE_API_KEY is not set"):
        ppg.preflight_consolidation(config)


def test_preflight_consolidation_uses_routed_agent_executables(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "llm-routing.json"
    config_path.write_text(
        """
        {
          "agents": {
            "review": {"client": "claude", "model": "claude-sonnet-4", "executable": "claude-custom"},
            "taxonomy_tail": {"client": "opencode", "model": "gpt-5.4-mini", "executable": "opencode-custom"}
          },
          "single_prompt": {
            "tier2_primary": {"client": "openai", "model": "gpt-5.4-mini"},
            "tier2_secondary": {"client": "openai", "model": "gpt-5.4"},
            "taxonomy_primary": {"client": "openai", "model": "gpt-5.4-mini"},
            "taxonomy_secondary": {"client": "openai", "model": "gpt-5.4"},
            "tier3_judge_primary": {"client": "openai", "model": "gpt-5.4-mini"},
            "tier3_judge_secondary": {"client": "openai", "model": "gpt-5.4"}
          },
          "embeddings": {
            "tier3": {"client": "openai", "model": "text-embedding-3-small"}
          }
        }
        """,
        encoding="utf-8",
    )
    config = ppg.ConsolidationConfig(
        max_iterations=5,
        run_dir=None,
        codex_bin="codex",
        dry_run=False,
        resume=False,
        required_consecutive_passes=0,
        llm_routing_config=str(config_path),
    )
    calls: list[str] = []
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    monkeypatch.setattr(ppg, "resolve_cli_executable", lambda executable: calls.append(executable) or executable)

    codex_executable = ppg.preflight_consolidation(config)

    assert codex_executable is None
    assert calls == ["claude-custom", "opencode-custom"]


def test_parse_args_accepts_dataset_registry_defaults(monkeypatch) -> None:
    entry = dr.DatasetRegistryEntry(
        key="bench-openalex-rag",
        notebook=dr.RegistryNotebook(id="nb-1", title="bench-openalex-rag"),
        neo4j=dr.RegistryNeo4j(
            uri="bolt://127.0.0.1:17687",
            username="neo4j",
            password="pw-123",
            database="neo4j",
        ),
    )
    class FakeNow:
        def strftime(self, fmt: str) -> str:
            return "20260325_1015"

    monkeypatch.setattr(ppg, "load_dataset_entry", lambda dataset_key, registry_path=None: entry)
    monkeypatch.setattr(ppg, "datetime", type("FakeDateTime", (), {"now": staticmethod(lambda: FakeNow())}))
    monkeypatch.setattr(
        ppg.sys,
        "argv",
        ["postprocess_graph.py", "--dataset-key", "bench-openalex-rag"],
    )

    args = ppg.parse_args()

    assert args.neo4j_uri == "bolt://127.0.0.1:17687"
    assert args.neo4j_user == "neo4j"
    assert args.neo4j_password == "pw-123"
    assert args.neo4j_database == "neo4j"
    assert args.run_dir.endswith("runs\\bench-openalex-rag\\postprocess_20260325_1015")
