from __future__ import annotations

import json
import subprocess
from pathlib import Path

import dataset_registry as dr
import scripts.sync_notebook_graph as sng


class FakeRunner:
    def __init__(self, responses: dict[tuple[str, ...], object]):
        self.responses = responses
        self.calls: list[list[str]] = []

    def run(self, args: list[str]) -> subprocess.CompletedProcess[str]:
        self.calls.append(args)
        key = tuple(args)
        try:
            response = self.responses[key]
        except KeyError as exc:
            raise AssertionError(f"Unexpected command: {args}") from exc
        if isinstance(response, list):
            if not response:
                raise AssertionError(f"No responses left for command: {args}")
            next_response = response.pop(0)
            if not isinstance(next_response, subprocess.CompletedProcess):
                raise AssertionError(f"Invalid queued response for command: {args}")
            return next_response
        if not isinstance(response, subprocess.CompletedProcess):
            raise AssertionError(f"Invalid response for command: {args}")
        return response


class FakeNotebookCLI:
    def __init__(
        self,
        notebook_to_create: sng.NotebookRef,
        known_notebooks: list[sng.NotebookRef] | None = None,
        sources: list[sng.NotebookSource] | None = None,
        delete_error: str | None = None,
        add_error_after: int | None = None,
    ):
        self.notebook_to_create = notebook_to_create
        self.known_notebooks = list(known_notebooks or [])
        self.sources = list(sources or [])
        self.deleted: list[str] = []
        self.added: list[Path] = []
        self.create_calls: list[str] = []
        self.delete_error = delete_error
        self.add_error_after = add_error_after

    def ensure_available(self) -> None:
        return None

    def ensure_authenticated(self) -> list[sng.NotebookRef]:
        return list(self.known_notebooks)

    def create_notebook(self, title: str) -> sng.NotebookRef:
        self.create_calls.append(title)
        if self.notebook_to_create not in self.known_notebooks:
            self.known_notebooks.append(self.notebook_to_create)
        return self.notebook_to_create

    def list_sources(self, notebook_id: str) -> list[sng.NotebookSource]:
        return list(self.sources)

    def delete_source(self, source_id: str) -> None:
        if self.delete_error:
            raise sng.SyncError(self.delete_error)
        self.deleted.append(source_id)
        self.sources = [source for source in self.sources if source.source_id != source_id]

    def add_file_source(self, notebook_id: str, file_path: Path) -> sng.NotebookSource:
        if self.add_error_after is not None and len(self.added) >= self.add_error_after:
            raise sng.SyncError("add failed")
        self.added.append(file_path)
        source = sng.NotebookSource(source_id=f"src-{len(self.added)}", title=file_path.name)
        self.sources.append(source)
        return source

    def get_source_content(self, source_id: str) -> str:
        return f"content for {source_id}"


class FakeProvisioner:
    def __init__(self, runtime: sng.Neo4jRuntime):
        self.runtime = runtime
        self.ensure_available_calls = 0
        self.ensure_runtime_calls: list[tuple[str, str, sng.Neo4jRuntime | None]] = []

    def ensure_available(self) -> None:
        self.ensure_available_calls += 1

    def ensure_runtime(
        self,
        project_slug: str,
        project_title_hash: str,
        existing_runtime: sng.Neo4jRuntime | None = None,
    ) -> sng.Neo4jRuntime:
        self.ensure_runtime_calls.append((project_slug, project_title_hash, existing_runtime))
        return self.runtime


class FakeGraphAPI:
    def __init__(self, *args, **kwargs):
        self.retry_calls: list[tuple[str, str]] = []
        self.source_rows: list[dict[str, str]] = []

    def health_check(self) -> bool:
        return True

    def connect(self, *args, **kwargs) -> dict:
        return {"status": "Success"}

    def sources_list(self) -> list[dict[str, str]]:
        return list(self.source_rows)

    def retry_processing(self, file_name: str, retry_condition: str) -> dict:
        self.retry_calls.append((file_name, retry_condition))
        return {"status": "Success"}


def make_completed(args: list[str], stdout: str = "", stderr: str = "", returncode: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=args, returncode=returncode, stdout=stdout, stderr=stderr)


def managed_inspect(
    container_name: str,
    project_slug: str,
    *,
    status: str = "running",
    health: str = "healthy",
    managed: bool = True,
    bolt_port: int = 17687,
    http_port: int = 17474,
    password: str = "pw-123",
    image: str = sng.DEFAULT_NEO4J_IMAGE,
    title_hash: str | None = None,
) -> str:
    labels = {sng.PROJECT_SLUG_LABEL: project_slug}
    if managed:
        labels[sng.MANAGED_LABEL] = "true"
    if title_hash:
        labels[sng.PROJECT_TITLE_HASH_LABEL] = title_hash
    payload = [
        {
            "Id": "container-id",
            "Name": f"/{container_name}",
            "Config": {
                "Image": image,
                "Labels": labels,
                "Env": [f"NEO4J_AUTH=neo4j/{password}"],
            },
            "State": {
                "Status": status,
                "Health": {"Status": health},
            },
            "NetworkSettings": {
                "Ports": {
                    "7687/tcp": [{"HostPort": str(bolt_port)}],
                    "7474/tcp": [{"HostPort": str(http_port)}],
                }
            },
        }
    ]
    return json.dumps(payload)


def default_runtime() -> sng.Neo4jRuntime:
    return sng.Neo4jRuntime(
        uri="bolt://127.0.0.1:17687",
        username="neo4j",
        password="pw-123",
        database="neo4j",
        container_name=sng.managed_container_name("demo"),
        container_id="container-id",
        bolt_port=17687,
        http_port=17474,
        image=sng.DEFAULT_NEO4J_IMAGE,
    )


def test_discover_dataset_files_skips_hidden_temp_and_zero_byte(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / "paper.pdf").write_bytes(b"pdf")
    (dataset_dir / ".hidden.txt").write_text("hidden", encoding="utf-8")
    (dataset_dir / "~$draft.docx").write_text("temp", encoding="utf-8")
    (dataset_dir / "empty.txt").write_text("", encoding="utf-8")
    nested = dataset_dir / "notes"
    nested.mkdir()
    (nested / "summary.md").write_text("summary", encoding="utf-8")

    files = sng.discover_dataset_files(dataset_dir)

    assert [item.relative_path.as_posix() for item in files] == ["notes/summary.md", "paper.pdf"]


def test_managed_names_and_free_ports_are_project_scoped() -> None:
    assert sng.managed_container_name("alpha") == "llm-graph-builder-neo4j-alpha"
    assert sng.managed_volume_name("alpha", "data") == "llm-graph-builder-neo4j-alpha-data"
    assert len(sng.notebook_title_hash("Alpha")) == 10
    port_a = sng.find_free_port()
    port_b = sng.find_free_port({port_a})
    assert port_a != port_b


def test_manifest_v2_round_trip(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    runtime = default_runtime()
    notebook = sng.NotebookRef("nb-1", "Demo")
    entries = {"paper.pdf": sng.ManifestEntry("paper.pdf", "hash", "src-1", "paper.txt", "exported")}

    sng.save_manifest(manifest_path, "demo", notebook, runtime, entries, ["removed.pdf"])

    state = sng.load_manifest_state(manifest_path)

    assert state.version == sng.MANIFEST_VERSION
    assert state.project_slug == "demo"
    assert state.notebook_id == "nb-1"
    assert state.neo4j is not None
    assert state.neo4j.uri == runtime.uri
    assert state.entries["paper.pdf"].staged_txt_name == "paper.txt"


def test_manifest_loader_accepts_legacy_entry_keys(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    runtime = default_runtime()
    manifest_path.write_text(
        json.dumps(
            {
                "version": 2,
                "project_slug": "demo",
                "notebook": {"id": "nb-1", "title": "Demo"},
                "neo4j": {
                    "uri": runtime.uri,
                    "username": runtime.username,
                    "password": runtime.password,
                    "database": runtime.database,
                    "container_name": runtime.container_name,
                    "container_id": runtime.container_id,
                    "bolt_port": runtime.bolt_port,
                    "http_port": runtime.http_port,
                    "image": runtime.image,
                },
                "entries": {
                    "paper.pdf": {
                        "relative_path": "paper.pdf",
                        "content_hash": "hash",
                        "notebook_source_id": "src-legacy",
                        "staged_txt_name": "paper.txt",
                        "last_sync_status": "exported",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    state = sng.load_manifest_state(manifest_path)

    assert state.entries["paper.pdf"].source_id == "src-legacy"
    assert state.entries["paper.pdf"].status == "exported"


def test_notebooklm_cli_adapter_supports_current_cli_shapes(tmp_path: Path) -> None:
    file_path = tmp_path / "paper.pdf"
    file_path.write_bytes(b"pdf")
    runner = FakeRunner(
        {
            ("nlm", "notebook", "list", "--json"): make_completed(
                ["nlm", "notebook", "list", "--json"],
                '[{"id": "nb-1", "title": "Alpha"}]',
            ),
            ("nlm", "notebook", "create", "Alpha"): make_completed(
                ["nlm", "notebook", "create", "Alpha"],
                "✓ Created notebook: Alpha\n  ID: nb-1\n",
            ),
            ("nlm", "source", "list", "nb-1", "--json"): make_completed(
                ["nlm", "source", "list", "nb-1", "--json"],
                '[{"id": "src-1", "title": "paper.pdf"}]',
            ),
            (
                "nlm",
                "source",
                "add",
                "nb-1",
                "--file",
                str(file_path),
                "--wait",
            ): make_completed(
                [
                    "nlm",
                    "source",
                    "add",
                    "nb-1",
                    "--file",
                    str(file_path),
                    "--wait",
                ],
                "Uploading paper.pdf and waiting for processing...\n✓ Added source: paper.pdf (ready)\nSource ID: src-2\n",
            ),
            ("nlm", "source", "content", "src-2", "--json"): make_completed(
                ["nlm", "source", "content", "src-2", "--json"],
                '{"value": {"content": "normalized text"}}',
            ),
            ("nlm", "source", "delete", "src-2", "--confirm"): make_completed(
                ["nlm", "source", "delete", "src-2", "--confirm"],
                "✓ Deleted source: src-2\n",
            ),
        }
    )
    cli = sng.NotebookLMCliAdapter(runner=runner, executable="nlm")

    assert cli.list_notebooks() == [sng.NotebookRef("nb-1", "Alpha")]
    assert cli.create_notebook("Alpha") == sng.NotebookRef("nb-1", "Alpha")
    assert cli.list_sources("nb-1") == [sng.NotebookSource("src-1", "paper.pdf")]
    assert cli.add_file_source("nb-1", file_path) == sng.NotebookSource("src-2", "paper.pdf")
    assert cli.get_source_content("src-2") == "normalized text"
    cli.delete_source("src-2")


def test_notebooklm_cli_adapter_still_accepts_legacy_json_shapes(tmp_path: Path) -> None:
    file_path = tmp_path / "paper.pdf"
    file_path.write_bytes(b"pdf")
    runner = FakeRunner(
        {
            ("nlm", "notebook", "list", "--json"): make_completed(
                ["nlm", "notebook", "list", "--json"],
                '{"notebooks": [{"id": "nb-1", "title": "Alpha"}]}',
            ),
            ("nlm", "source", "list", "nb-1", "--json"): make_completed(
                ["nlm", "source", "list", "nb-1", "--json"],
                '{"sources": [{"id": "src-1", "title": "paper.pdf"}]}',
            ),
            ("nlm", "source", "content", "src-1", "--json"): make_completed(
                ["nlm", "source", "content", "src-1", "--json"],
                '{"content": "normalized text"}',
            ),
        }
    )
    cli = sng.NotebookLMCliAdapter(runner=runner, executable="nlm")

    assert cli.list_notebooks() == [sng.NotebookRef("nb-1", "Alpha")]
    assert cli.list_sources("nb-1") == [sng.NotebookSource("src-1", "paper.pdf")]
    assert cli.get_source_content("src-1") == "normalized text"


def test_provisioner_creates_new_container(monkeypatch) -> None:
    container_name = sng.managed_container_name("demo")
    runner = FakeRunner(
        {
            ("docker", "inspect", container_name): [
                make_completed(["docker", "inspect", container_name], stderr="Error: No such object", returncode=1),
                make_completed(
                    ["docker", "inspect", container_name],
                    stdout=managed_inspect(container_name, "demo", title_hash=sng.notebook_title_hash("Demo")),
                ),
                make_completed(
                    ["docker", "inspect", container_name],
                    stdout=managed_inspect(container_name, "demo", title_hash=sng.notebook_title_hash("Demo")),
                ),
                make_completed(
                    ["docker", "inspect", container_name],
                    stdout=managed_inspect(container_name, "demo", title_hash=sng.notebook_title_hash("Demo")),
                ),
            ],
            (
                "docker",
                "run",
                "-d",
                "--name",
                container_name,
                "--label",
                f"{sng.MANAGED_LABEL}=true",
                "--label",
                f"{sng.PROJECT_SLUG_LABEL}=demo",
                "--label",
                f"{sng.PROJECT_TITLE_HASH_LABEL}={sng.notebook_title_hash('Demo')}",
                "--publish",
                "17474:7474",
                "--publish",
                "17687:7687",
                "--volume",
                f"{sng.managed_volume_name('demo', 'data')}:/data",
                "--volume",
                f"{sng.managed_volume_name('demo', 'logs')}:/logs",
                "--volume",
                f"{sng.managed_volume_name('demo', 'plugins')}:/plugins",
                "--env",
                "NEO4J_AUTH=neo4j/pw-123",
                "--env",
                'NEO4J_PLUGINS=["apoc"]',
                "--env",
                "NEO4J_dbms_security_procedures_unrestricted=apoc.*",
                "--env",
                "NEO4J_dbms_security_procedures_allowlist=apoc.*",
                "--env",
                "NEO4J_apoc_export_file_enabled=true",
                "--env",
                "NEO4J_apoc_import_file_enabled=true",
                '--health-cmd=cypher-shell -u neo4j -p pw-123 "RETURN 1" || exit 1',
                "--health-interval=10s",
                "--health-timeout=10s",
                "--health-retries=12",
                "--health-start-period=30s",
                sng.DEFAULT_NEO4J_IMAGE,
            ): make_completed(["docker", "run"]),
        }
    )
    provisioner = sng.DockerNeo4jProvisioner(runner=runner)
    ports = iter([17687, 17474])
    monkeypatch.setattr(sng, "generate_neo4j_password", lambda: "pw-123")
    monkeypatch.setattr(sng, "find_free_port", lambda exclude=None: next(ports))
    monkeypatch.setattr(sng.time, "sleep", lambda _: None)

    runtime = provisioner.ensure_runtime("demo", sng.notebook_title_hash("Demo"))

    assert runtime.uri == "bolt://127.0.0.1:17687"
    assert runtime.container_name == container_name
    assert any(call[:3] == ["docker", "run", "-d"] for call in runner.calls)


def test_provisioner_reuses_existing_healthy_container(monkeypatch) -> None:
    container_name = sng.managed_container_name("demo")
    runner = FakeRunner(
        {
            ("docker", "inspect", container_name): [
                make_completed(
                    ["docker", "inspect", container_name],
                    stdout=managed_inspect(container_name, "demo", title_hash=sng.notebook_title_hash("Demo")),
                ),
                make_completed(
                    ["docker", "inspect", container_name],
                    stdout=managed_inspect(container_name, "demo", title_hash=sng.notebook_title_hash("Demo")),
                ),
                make_completed(
                    ["docker", "inspect", container_name],
                    stdout=managed_inspect(container_name, "demo", title_hash=sng.notebook_title_hash("Demo")),
                ),
            ],
        }
    )
    provisioner = sng.DockerNeo4jProvisioner(runner=runner)
    monkeypatch.setattr(sng.time, "sleep", lambda _: None)

    runtime = provisioner.ensure_runtime("demo", sng.notebook_title_hash("Demo"), default_runtime())

    assert runtime.container_name == container_name
    assert not any(call[:2] == ["docker", "run"] for call in runner.calls)
    assert not any(call[:2] == ["docker", "start"] for call in runner.calls)


def test_provisioner_starts_existing_stopped_container(monkeypatch) -> None:
    container_name = sng.managed_container_name("demo")
    runner = FakeRunner(
        {
            ("docker", "inspect", container_name): [
                make_completed(
                    ["docker", "inspect", container_name],
                    stdout=managed_inspect(
                        container_name,
                        "demo",
                        status="exited",
                        health="starting",
                        title_hash=sng.notebook_title_hash("Demo"),
                    ),
                ),
                make_completed(
                    ["docker", "inspect", container_name],
                    stdout=managed_inspect(container_name, "demo", title_hash=sng.notebook_title_hash("Demo")),
                ),
                make_completed(
                    ["docker", "inspect", container_name],
                    stdout=managed_inspect(container_name, "demo", title_hash=sng.notebook_title_hash("Demo")),
                ),
                make_completed(
                    ["docker", "inspect", container_name],
                    stdout=managed_inspect(container_name, "demo", title_hash=sng.notebook_title_hash("Demo")),
                ),
            ],
            ("docker", "start", container_name): make_completed(["docker", "start", container_name]),
        }
    )
    provisioner = sng.DockerNeo4jProvisioner(runner=runner)
    monkeypatch.setattr(sng.time, "sleep", lambda _: None)

    runtime = provisioner.ensure_runtime("demo", sng.notebook_title_hash("Demo"), default_runtime())

    assert runtime.container_name == container_name
    assert ["docker", "start", container_name] in runner.calls


def test_provisioner_rejects_unmanaged_conflict() -> None:
    container_name = sng.managed_container_name("demo")
    runner = FakeRunner(
        {
            ("docker", "inspect", container_name): make_completed(
                ["docker", "inspect", container_name],
                stdout=managed_inspect(container_name, "demo", managed=False),
            ),
        }
    )
    provisioner = sng.DockerNeo4jProvisioner(runner=runner)

    try:
        provisioner.ensure_runtime("demo", sng.notebook_title_hash("Demo"), default_runtime())
    except sng.SyncError as exc:
        assert "is not managed by this repo" in str(exc)
    else:
        raise AssertionError("Expected SyncError")


def test_provisioner_rejects_title_hash_conflict() -> None:
    container_name = sng.managed_container_name("demo")
    runner = FakeRunner(
        {
            ("docker", "inspect", container_name): make_completed(
                ["docker", "inspect", container_name],
                stdout=managed_inspect(
                    container_name,
                    "demo",
                    title_hash=sng.notebook_title_hash("Different Title"),
                ),
            ),
        }
    )
    provisioner = sng.DockerNeo4jProvisioner(runner=runner)

    try:
        provisioner.ensure_runtime("demo", sng.notebook_title_hash("Demo"), default_runtime())
    except sng.SyncError as exc:
        assert "different notebook title fingerprint" in str(exc)
    else:
        raise AssertionError("Expected SyncError")


def test_create_provisions_notebook_and_runtime_then_builds(monkeypatch, tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / "paper.pdf").write_bytes(b"pdf")

    notebook = sng.NotebookRef("nb-1", "demo")
    fake_cli = FakeNotebookCLI(notebook, known_notebooks=[])
    fake_provisioner = FakeProvisioner(default_runtime())
    fake_graph = FakeGraphAPI()
    build_calls: list[tuple[Path, sng.Neo4jRuntime]] = []

    monkeypatch.setattr(sng, "NotebookLMCliAdapter", lambda: fake_cli)
    monkeypatch.setattr(sng, "DockerNeo4jProvisioner", lambda: fake_provisioner)
    monkeypatch.setattr(sng, "GraphBuilderAPI", lambda *args, **kwargs: fake_graph)
    monkeypatch.setattr(sng, "run_build_graph", lambda args, sources_dir, runtime: build_calls.append((sources_dir, runtime)))

    args = sng.build_parser().parse_args(
        [
            "create",
            "--dataset-dir",
            str(dataset_dir),
            "--notebook-title",
            "demo",
            "--export-dir",
            str(tmp_path / "export"),
        ]
    )

    exit_code = sng.sync_dataset(args)

    assert exit_code == 0
    assert fake_cli.create_calls == ["demo"]
    assert fake_provisioner.ensure_runtime_calls == [("demo", sng.notebook_title_hash("demo"), None)]
    assert build_calls == [(Path(args.export_dir) / "sources", default_runtime())]
    manifest = sng.load_manifest_state(Path(args.export_dir) / "manifest.json")
    assert manifest.project_slug == "demo"
    assert manifest.notebook_id == "nb-1"
    assert manifest.neo4j is not None
    assert manifest.neo4j.uri == default_runtime().uri


def test_create_reuses_single_existing_notebook_and_manifest_runtime(monkeypatch, tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / "paper.pdf").write_bytes(b"pdf")
    export_dir = tmp_path / "export"
    export_dir.mkdir()

    notebook = sng.NotebookRef("nb-1", "demo")
    runtime = default_runtime()
    sng.save_manifest(export_dir / "manifest.json", "demo", notebook, runtime, {}, [])

    fake_cli = FakeNotebookCLI(notebook, known_notebooks=[notebook])
    fake_provisioner = FakeProvisioner(runtime)
    fake_graph = FakeGraphAPI()

    monkeypatch.setattr(sng, "NotebookLMCliAdapter", lambda: fake_cli)
    monkeypatch.setattr(sng, "DockerNeo4jProvisioner", lambda: fake_provisioner)
    monkeypatch.setattr(sng, "GraphBuilderAPI", lambda *args, **kwargs: fake_graph)
    monkeypatch.setattr(sng, "run_build_graph", lambda args, sources_dir, runtime: None)

    args = sng.build_parser().parse_args(
        [
            "create",
            "--dataset-dir",
            str(dataset_dir),
            "--notebook-title",
            "demo",
            "--export-dir",
            str(export_dir),
        ]
    )

    exit_code = sng.sync_dataset(args)

    assert exit_code == 0
    assert fake_cli.create_calls == []
    assert fake_provisioner.ensure_runtime_calls == [("demo", sng.notebook_title_hash("demo"), runtime)]


def test_create_is_idempotent_for_unchanged_dataset(monkeypatch, tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    source_path = dataset_dir / "paper.pdf"
    source_path.write_bytes(b"pdf")
    export_dir = tmp_path / "export"
    export_dir.mkdir()

    notebook = sng.NotebookRef("nb-1", "demo")
    runtime = default_runtime()
    staged_name = sng.staged_txt_name_for(Path("paper.pdf"))
    old_entry = sng.ManifestEntry("paper.pdf", sng.file_sha256(source_path), "src-1", staged_name, "exported")
    sng.save_manifest(export_dir / "manifest.json", "demo", notebook, runtime, {"paper.pdf": old_entry}, [])
    (export_dir / "sources").mkdir()
    (export_dir / "sources" / staged_name).write_text("already exported", encoding="utf-8")

    fake_cli = FakeNotebookCLI(notebook, known_notebooks=[notebook], sources=[sng.NotebookSource("src-1", "paper.pdf")])
    fake_provisioner = FakeProvisioner(runtime)
    fake_graph = FakeGraphAPI()
    build_calls: list[tuple[Path, sng.Neo4jRuntime]] = []

    monkeypatch.setattr(sng, "NotebookLMCliAdapter", lambda: fake_cli)
    monkeypatch.setattr(sng, "DockerNeo4jProvisioner", lambda: fake_provisioner)
    monkeypatch.setattr(sng, "GraphBuilderAPI", lambda *args, **kwargs: fake_graph)
    monkeypatch.setattr(sng, "run_build_graph", lambda args, sources_dir, runtime: build_calls.append((sources_dir, runtime)))

    args = sng.build_parser().parse_args(
        [
            "create",
            "--dataset-dir",
            str(dataset_dir),
            "--notebook-title",
            "demo",
            "--export-dir",
            str(export_dir),
        ]
    )

    exit_code = sng.sync_dataset(args)

    assert exit_code == 0
    assert fake_provisioner.ensure_runtime_calls == [("demo", sng.notebook_title_hash("demo"), runtime)]
    assert fake_cli.create_calls == []
    assert fake_cli.deleted == []
    assert fake_cli.added == []
    assert fake_graph.retry_calls == []
    assert build_calls == [(export_dir / "sources", runtime)]


def test_create_persists_partial_manifest_progress_on_mid_run_failure(monkeypatch, tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    first = dataset_dir / "alpha.pdf"
    second = dataset_dir / "beta.pdf"
    first.write_bytes(b"alpha")
    second.write_bytes(b"beta")
    export_dir = tmp_path / "export"

    notebook = sng.NotebookRef("nb-1", "demo")
    runtime = default_runtime()
    fake_cli = FakeNotebookCLI(notebook, known_notebooks=[notebook], add_error_after=1)
    fake_provisioner = FakeProvisioner(runtime)
    fake_graph = FakeGraphAPI()

    monkeypatch.setattr(sng, "NotebookLMCliAdapter", lambda: fake_cli)
    monkeypatch.setattr(sng, "DockerNeo4jProvisioner", lambda: fake_provisioner)
    monkeypatch.setattr(sng, "GraphBuilderAPI", lambda *args, **kwargs: fake_graph)
    monkeypatch.setattr(sng, "run_build_graph", lambda args, sources_dir, runtime: None)

    args = sng.build_parser().parse_args(
        [
            "create",
            "--dataset-dir",
            str(dataset_dir),
            "--notebook-title",
            "demo",
            "--export-dir",
            str(export_dir),
        ]
    )

    try:
        sng.sync_dataset(args)
    except sng.SyncError as exc:
        assert "add failed" in str(exc)
    else:
        raise AssertionError("Expected SyncError")

    manifest = sng.load_manifest_state(export_dir / "manifest.json")
    assert manifest.notebook_id == "nb-1"
    assert manifest.neo4j is not None
    assert set(manifest.entries) == {"alpha.pdf"}
    assert manifest.entries["alpha.pdf"].source_id == "src-1"


def test_create_fails_on_ambiguous_notebook_titles_without_manifest_pin(monkeypatch, tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / "paper.pdf").write_bytes(b"pdf")
    duplicates = [sng.NotebookRef("nb-1", "demo"), sng.NotebookRef("nb-2", "demo")]

    fake_cli = FakeNotebookCLI(duplicates[0], known_notebooks=duplicates)
    fake_provisioner = FakeProvisioner(default_runtime())

    monkeypatch.setattr(sng, "NotebookLMCliAdapter", lambda: fake_cli)
    monkeypatch.setattr(sng, "DockerNeo4jProvisioner", lambda: fake_provisioner)

    args = sng.build_parser().parse_args(
        ["create", "--dataset-dir", str(dataset_dir), "--notebook-title", "demo"]
    )

    try:
        sng.sync_dataset(args)
    except sng.SyncError as exc:
        assert "Multiple NotebookLM notebooks found" in str(exc)
    else:
        raise AssertionError("Expected SyncError")


def test_create_uses_manifest_pinned_notebook_when_titles_are_duplicated(monkeypatch, tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / "paper.pdf").write_bytes(b"pdf")
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    duplicates = [sng.NotebookRef("nb-1", "demo"), sng.NotebookRef("nb-2", "demo")]
    runtime = default_runtime()
    sng.save_manifest(export_dir / "manifest.json", "demo", duplicates[1], runtime, {}, [])

    fake_cli = FakeNotebookCLI(duplicates[0], known_notebooks=duplicates)
    fake_provisioner = FakeProvisioner(runtime)
    fake_graph = FakeGraphAPI()

    monkeypatch.setattr(sng, "NotebookLMCliAdapter", lambda: fake_cli)
    monkeypatch.setattr(sng, "DockerNeo4jProvisioner", lambda: fake_provisioner)
    monkeypatch.setattr(sng, "GraphBuilderAPI", lambda *args, **kwargs: fake_graph)
    monkeypatch.setattr(sng, "run_build_graph", lambda args, sources_dir, runtime: None)

    args = sng.build_parser().parse_args(
        [
            "create",
            "--dataset-dir",
            str(dataset_dir),
            "--notebook-title",
            "demo",
            "--export-dir",
            str(export_dir),
        ]
    )

    exit_code = sng.sync_dataset(args)

    assert exit_code == 0
    assert fake_cli.create_calls == []
    assert fake_provisioner.ensure_runtime_calls == [("demo", sng.notebook_title_hash("demo"), runtime)]


def test_update_uses_manifest_runtime_and_retries_changed_sources(monkeypatch, tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    source_path = dataset_dir / "paper.pdf"
    source_path.write_bytes(b"new pdf")

    export_dir = tmp_path / "export"
    export_dir.mkdir()
    runtime = default_runtime()
    notebook = sng.NotebookRef("nb-1", "demo")
    old_source = sng.NotebookSource("src-old", "paper.pdf")
    staged_name = sng.staged_txt_name_for(Path("paper.pdf"))
    old_entry = sng.ManifestEntry("paper.pdf", "old-hash", "src-old", staged_name, "exported")
    sng.save_manifest(export_dir / "manifest.json", "demo", notebook, runtime, {"paper.pdf": old_entry}, [])

    fake_cli = FakeNotebookCLI(notebook, known_notebooks=[notebook], sources=[old_source])
    fake_provisioner = FakeProvisioner(runtime)
    fake_graph = FakeGraphAPI()
    fake_graph.source_rows = [{"fileName": staged_name}]
    build_calls: list[tuple[Path, sng.Neo4jRuntime]] = []

    monkeypatch.setattr(sng, "NotebookLMCliAdapter", lambda: fake_cli)
    monkeypatch.setattr(sng, "DockerNeo4jProvisioner", lambda: fake_provisioner)
    monkeypatch.setattr(sng, "GraphBuilderAPI", lambda *args, **kwargs: fake_graph)
    monkeypatch.setattr(sng, "run_build_graph", lambda args, sources_dir, runtime: build_calls.append((sources_dir, runtime)))

    args = sng.build_parser().parse_args(
        [
            "update",
            "--dataset-dir",
            str(dataset_dir),
            "--notebook-title",
            "demo",
            "--export-dir",
            str(export_dir),
        ]
    )

    exit_code = sng.sync_dataset(args)

    assert exit_code == 0
    assert fake_provisioner.ensure_runtime_calls == [("demo", sng.notebook_title_hash("demo"), runtime)]
    assert fake_cli.deleted == ["src-old"]
    assert fake_graph.retry_calls == [(staged_name, sng.RETRY_CONDITION)]
    assert build_calls == [(export_dir / "sources", runtime)]


def test_update_is_idempotent_for_unchanged_dataset(monkeypatch, tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    source_path = dataset_dir / "paper.pdf"
    source_path.write_bytes(b"pdf")
    export_dir = tmp_path / "export"
    export_dir.mkdir()

    runtime = default_runtime()
    notebook = sng.NotebookRef("nb-1", "demo")
    staged_name = sng.staged_txt_name_for(Path("paper.pdf"))
    old_entry = sng.ManifestEntry("paper.pdf", sng.file_sha256(source_path), "src-1", staged_name, "exported")
    sng.save_manifest(export_dir / "manifest.json", "demo", notebook, runtime, {"paper.pdf": old_entry}, [])
    (export_dir / "sources").mkdir()
    (export_dir / "sources" / staged_name).write_text("already exported", encoding="utf-8")

    fake_cli = FakeNotebookCLI(notebook, known_notebooks=[notebook], sources=[sng.NotebookSource("src-1", "paper.pdf")])
    fake_provisioner = FakeProvisioner(runtime)
    fake_graph = FakeGraphAPI()
    build_calls: list[tuple[Path, sng.Neo4jRuntime]] = []

    monkeypatch.setattr(sng, "NotebookLMCliAdapter", lambda: fake_cli)
    monkeypatch.setattr(sng, "DockerNeo4jProvisioner", lambda: fake_provisioner)
    monkeypatch.setattr(sng, "GraphBuilderAPI", lambda *args, **kwargs: fake_graph)
    monkeypatch.setattr(sng, "run_build_graph", lambda args, sources_dir, runtime: build_calls.append((sources_dir, runtime)))

    args = sng.build_parser().parse_args(
        [
            "update",
            "--dataset-dir",
            str(dataset_dir),
            "--notebook-title",
            "demo",
            "--export-dir",
            str(export_dir),
        ]
    )

    exit_code = sng.sync_dataset(args)

    assert exit_code == 0
    assert fake_provisioner.ensure_runtime_calls == [("demo", sng.notebook_title_hash("demo"), runtime)]
    assert fake_cli.deleted == []
    assert fake_cli.added == []
    assert fake_graph.retry_calls == []
    assert build_calls == [(export_dir / "sources", runtime)]


def test_update_aborts_if_existing_source_cannot_be_deleted(monkeypatch, tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / "paper.pdf").write_bytes(b"new pdf")
    export_dir = tmp_path / "export"
    export_dir.mkdir()

    runtime = default_runtime()
    notebook = sng.NotebookRef("nb-1", "demo")
    old_source = sng.NotebookSource("src-old", "paper.pdf")
    staged_name = sng.staged_txt_name_for(Path("paper.pdf"))
    old_entry = sng.ManifestEntry("paper.pdf", "old-hash", "src-old", staged_name, "exported")
    sng.save_manifest(export_dir / "manifest.json", "demo", notebook, runtime, {"paper.pdf": old_entry}, [])

    fake_cli = FakeNotebookCLI(notebook, known_notebooks=[notebook], sources=[old_source], delete_error="delete failed")
    fake_provisioner = FakeProvisioner(runtime)
    fake_graph = FakeGraphAPI()

    monkeypatch.setattr(sng, "NotebookLMCliAdapter", lambda: fake_cli)
    monkeypatch.setattr(sng, "DockerNeo4jProvisioner", lambda: fake_provisioner)
    monkeypatch.setattr(sng, "GraphBuilderAPI", lambda *args, **kwargs: fake_graph)

    args = sng.build_parser().parse_args(
        [
            "update",
            "--dataset-dir",
            str(dataset_dir),
            "--notebook-title",
            "demo",
            "--export-dir",
            str(export_dir),
        ]
    )

    try:
        sng.sync_dataset(args)
    except sng.SyncError as exc:
        assert "Failed to replace existing NotebookLM source" in str(exc)
    else:
        raise AssertionError("Expected SyncError")


def test_update_legacy_manifest_requires_explicit_runtime(monkeypatch, tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / "paper.pdf").write_bytes(b"pdf")
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    (export_dir / "manifest.json").write_text(
        json.dumps(
            {
                "version": 1,
                "notebook": {"id": "nb-1", "title": "demo"},
                "entries": {},
            }
        ),
        encoding="utf-8",
    )

    notebook = sng.NotebookRef("nb-1", "demo")
    fake_cli = FakeNotebookCLI(notebook, known_notebooks=[notebook])
    fake_provisioner = FakeProvisioner(default_runtime())

    monkeypatch.setattr(sng, "NotebookLMCliAdapter", lambda: fake_cli)
    monkeypatch.setattr(sng, "DockerNeo4jProvisioner", lambda: fake_provisioner)

    args = sng.build_parser().parse_args(
        [
            "update",
            "--dataset-dir",
            str(dataset_dir),
            "--notebook-title",
            "demo",
            "--export-dir",
            str(export_dir),
        ]
    )

    try:
        sng.sync_dataset(args)
    except sng.SyncError as exc:
        assert "Legacy manifests without Neo4j runtime metadata require explicit" in str(exc)
    else:
        raise AssertionError("Expected SyncError")


def test_update_legacy_manifest_accepts_explicit_runtime(monkeypatch, tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / "paper.pdf").write_bytes(b"pdf")
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    (export_dir / "manifest.json").write_text(
        json.dumps(
            {
                "version": 1,
                "notebook": {"id": "nb-1", "title": "demo"},
                "entries": {},
            }
        ),
        encoding="utf-8",
    )

    notebook = sng.NotebookRef("nb-1", "demo")
    fake_cli = FakeNotebookCLI(notebook, known_notebooks=[notebook])
    fake_provisioner = FakeProvisioner(default_runtime())
    fake_graph = FakeGraphAPI()

    monkeypatch.setattr(sng, "NotebookLMCliAdapter", lambda: fake_cli)
    monkeypatch.setattr(sng, "DockerNeo4jProvisioner", lambda: fake_provisioner)
    monkeypatch.setattr(sng, "GraphBuilderAPI", lambda *args, **kwargs: fake_graph)
    monkeypatch.setattr(sng, "run_build_graph", lambda args, sources_dir, runtime: None)

    args = sng.build_parser().parse_args(
        [
            "update",
            "--dataset-dir",
            str(dataset_dir),
            "--notebook-title",
            "demo",
            "--export-dir",
            str(export_dir),
            "--neo4j-uri",
            "bolt://127.0.0.1:17687",
            "--neo4j-user",
            "neo4j",
            "--neo4j-password",
            "pw-123",
            "--neo4j-database",
            "neo4j",
        ]
    )

    exit_code = sng.sync_dataset(args)

    assert exit_code == 0
    manifest = sng.load_manifest_state(export_dir / "manifest.json")
    assert manifest.neo4j is not None
    assert manifest.neo4j.uri == "bolt://127.0.0.1:17687"


def test_update_legacy_manifest_with_explicit_runtime_does_not_require_docker(monkeypatch, tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / "paper.pdf").write_bytes(b"pdf")
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    (export_dir / "manifest.json").write_text(
        json.dumps(
            {
                "version": 1,
                "notebook": {"id": "nb-1", "title": "demo"},
                "entries": {},
            }
        ),
        encoding="utf-8",
    )

    notebook = sng.NotebookRef("nb-1", "demo")
    fake_cli = FakeNotebookCLI(notebook, known_notebooks=[notebook])
    fake_graph = FakeGraphAPI()

    class FailProvisioner:
        def ensure_available(self) -> None:
            raise AssertionError("docker should not be checked")

        def ensure_runtime(self, *args, **kwargs):
            raise AssertionError("docker runtime should not be resolved")

    monkeypatch.setattr(sng, "NotebookLMCliAdapter", lambda: fake_cli)
    monkeypatch.setattr(sng, "DockerNeo4jProvisioner", lambda: FailProvisioner())
    monkeypatch.setattr(sng, "GraphBuilderAPI", lambda *args, **kwargs: fake_graph)
    monkeypatch.setattr(sng, "run_build_graph", lambda args, sources_dir, runtime: None)

    args = sng.build_parser().parse_args(
        [
            "update",
            "--dataset-dir",
            str(dataset_dir),
            "--notebook-id",
            "nb-1",
            "--export-dir",
            str(export_dir),
            "--neo4j-uri",
            "bolt://127.0.0.1:17687",
            "--neo4j-user",
            "neo4j",
            "--neo4j-password",
            "pw-123",
            "--neo4j-database",
            "neo4j",
        ]
    )

    assert sng.sync_dataset(args) == 0


def test_update_allows_notebook_id_without_title(monkeypatch, tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / "paper.pdf").write_bytes(b"pdf")
    export_dir = tmp_path / "export"
    export_dir.mkdir()

    notebook = sng.NotebookRef("nb-1", "demo")
    runtime = default_runtime()
    sng.save_manifest(export_dir / "manifest.json", "demo", notebook, runtime, {}, [])

    fake_cli = FakeNotebookCLI(notebook, known_notebooks=[notebook])
    fake_provisioner = FakeProvisioner(runtime)
    fake_graph = FakeGraphAPI()

    monkeypatch.setattr(sng, "NotebookLMCliAdapter", lambda: fake_cli)
    monkeypatch.setattr(sng, "DockerNeo4jProvisioner", lambda: fake_provisioner)
    monkeypatch.setattr(sng, "GraphBuilderAPI", lambda *args, **kwargs: fake_graph)
    monkeypatch.setattr(sng, "run_build_graph", lambda args, sources_dir, runtime: None)

    args = sng.build_parser().parse_args(
        [
            "update",
            "--dataset-dir",
            str(dataset_dir),
            "--notebook-id",
            "nb-1",
            "--export-dir",
            str(export_dir),
        ]
    )

    assert sng.sync_dataset(args) == 0


def test_update_restages_file_when_manifest_source_is_missing(monkeypatch, tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    source_path = dataset_dir / "paper.pdf"
    source_path.write_bytes(b"pdf")
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    runtime = default_runtime()
    notebook = sng.NotebookRef("nb-1", "demo")
    staged_name = sng.staged_txt_name_for(Path("paper.pdf"))
    old_entry = sng.ManifestEntry("paper.pdf", sng.file_sha256(source_path), "src-old", staged_name, "exported")
    sng.save_manifest(export_dir / "manifest.json", "demo", notebook, runtime, {"paper.pdf": old_entry}, [])

    fake_cli = FakeNotebookCLI(notebook, known_notebooks=[notebook], sources=[])
    fake_provisioner = FakeProvisioner(runtime)
    fake_graph = FakeGraphAPI()

    monkeypatch.setattr(sng, "NotebookLMCliAdapter", lambda: fake_cli)
    monkeypatch.setattr(sng, "DockerNeo4jProvisioner", lambda: fake_provisioner)
    monkeypatch.setattr(sng, "GraphBuilderAPI", lambda *args, **kwargs: fake_graph)
    monkeypatch.setattr(sng, "run_build_graph", lambda args, sources_dir, runtime: None)

    args = sng.build_parser().parse_args(
        [
            "update",
            "--dataset-dir",
            str(dataset_dir),
            "--notebook-title",
            "demo",
            "--export-dir",
            str(export_dir),
        ]
    )

    assert sng.sync_dataset(args) == 0
    assert fake_cli.deleted == []
    assert fake_cli.added == [source_path]


def test_update_rejects_export_dir_from_different_notebook(monkeypatch, tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / "paper.pdf").write_bytes(b"pdf")
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    sng.save_manifest(export_dir / "manifest.json", "old", sng.NotebookRef("nb-old", "old"), default_runtime(), {}, [])

    fake_cli = FakeNotebookCLI(sng.NotebookRef("nb-new", "demo"), known_notebooks=[sng.NotebookRef("nb-new", "demo")])
    fake_provisioner = FakeProvisioner(default_runtime())

    monkeypatch.setattr(sng, "NotebookLMCliAdapter", lambda: fake_cli)
    monkeypatch.setattr(sng, "DockerNeo4jProvisioner", lambda: fake_provisioner)

    args = sng.build_parser().parse_args(
        [
            "update",
            "--dataset-dir",
            str(dataset_dir),
            "--notebook-title",
            "demo",
            "--export-dir",
            str(export_dir),
        ]
    )

    try:
        sng.sync_dataset(args)
    except sng.SyncError as exc:
        assert "belongs to notebook nb-old" in str(exc)
    else:
        raise AssertionError("Expected SyncError")


def test_sync_dataset_uses_dataset_registry_defaults(monkeypatch, tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    source_path = dataset_dir / "paper.pdf"
    source_path.write_bytes(b"pdf")
    export_dir = tmp_path / "registry-export"
    runtime = default_runtime()
    entry = dr.DatasetRegistryEntry(
        key="bench-imdb-scifi",
        notebook=dr.RegistryNotebook(id="nb-1", title="bench-imdb-scifi"),
        neo4j=dr.RegistryNeo4j(
            uri=runtime.uri,
            username=runtime.username,
            password=runtime.password,
            database=runtime.database,
        ),
    )

    fake_cli = FakeNotebookCLI(sng.NotebookRef("nb-1", "bench-imdb-scifi"), known_notebooks=[sng.NotebookRef("nb-1", "bench-imdb-scifi")])
    fake_graph = FakeGraphAPI()
    build_calls: list[tuple[Path, sng.Neo4jRuntime]] = []

    class FailProvisioner:
        def ensure_available(self) -> None:
            raise AssertionError("docker should not be checked when registry fills explicit runtime")

        def ensure_runtime(self, *args, **kwargs):
            raise AssertionError("docker runtime should not be resolved")

    monkeypatch.setattr(sng, "load_dataset_entry", lambda dataset_key, registry_path=None: entry)
    monkeypatch.setattr(sng, "default_export_dir", lambda dataset_key: export_dir)
    monkeypatch.setattr(sng, "NotebookLMCliAdapter", lambda: fake_cli)
    monkeypatch.setattr(sng, "DockerNeo4jProvisioner", lambda: FailProvisioner())
    monkeypatch.setattr(sng, "GraphBuilderAPI", lambda *args, **kwargs: fake_graph)
    monkeypatch.setattr(sng, "run_build_graph", lambda args, sources_dir, runtime: build_calls.append((sources_dir, runtime)))

    args = sng.build_parser().parse_args(
        [
            "create",
            "--dataset-dir",
            str(dataset_dir),
            "--dataset-key",
            "bench-imdb-scifi",
        ]
    )

    assert sng.sync_dataset(args) == 0
    manifest = sng.load_manifest_state(export_dir / "manifest.json")
    assert manifest.notebook_id == "nb-1"
    assert manifest.notebook_title == "bench-imdb-scifi"
    assert manifest.neo4j is not None
    assert manifest.neo4j.uri == runtime.uri
    assert build_calls == [(export_dir / "sources", manifest.neo4j)]
