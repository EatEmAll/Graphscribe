#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import secrets
import socket
import os
import subprocess
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from notebooklm_graph_pipe.runtime.dataset_registry import default_export_dir, load_dataset_entry
from notebooklm_graph_pipe.runtime.graph_builder_runtime import GraphBuilderAPI

DEFAULT_NEO4J_IMAGE = "neo4j:5.26.7"
DEFAULT_DIRECT_NEO4J_URI = "bolt://127.0.0.1:7687"
DEFAULT_NEO4J_USER = "neo4j"
DEFAULT_NEO4J_PASSWORD = "password123"
DEFAULT_NEO4J_DATABASE = "neo4j"
MANAGED_CONTAINER_PREFIX = "llm-graph-builder-neo4j"
MANAGED_LABEL = "io.llm_graph_builder.managed"
PROJECT_SLUG_LABEL = "io.llm_graph_builder.project_slug"
PROJECT_TITLE_HASH_LABEL = "io.llm_graph_builder.project_title_hash"
MANIFEST_VERSION = 3
RETRY_CONDITION = "delete_entities_and_start_from_beginning"


class SyncError(RuntimeError):
    pass


@dataclass(frozen=True)
class NotebookRef:
    notebook_id: str
    title: str


@dataclass(frozen=True)
class NotebookSource:
    source_id: str
    title: str


@dataclass(frozen=True)
class DatasetFile:
    relative_path: Path
    absolute_path: Path
    content_hash: str


@dataclass(frozen=True)
class ManifestEntry:
    relative_path: str
    content_hash: str
    source_id: str
    staged_txt_name: str
    status: str


@dataclass(frozen=True)
class Neo4jRuntime:
    uri: str
    username: str
    password: str
    database: str
    password_env: str = "NEO4J_PASSWORD"
    deployment: str = "external"
    container_name: str | None = None
    container_id: str | None = None
    bolt_port: int | None = None
    http_port: int | None = None
    image: str | None = None


@dataclass(frozen=True)
class ManifestState:
    version: int
    project_slug: str
    notebook_id: str | None
    notebook_title: str | None
    neo4j: Neo4jRuntime | None
    entries: dict[str, ManifestEntry]
    removed_files: list[str]


class SubprocessRunner:
    def run(self, args: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(args, capture_output=True, text=True, check=False)


class NotebookLMCliAdapter:
    def __init__(self, *, runner: SubprocessRunner | None = None, executable: str = "nlm"):
        self.runner = runner or SubprocessRunner()
        self.executable = executable

    def _run_checked(self, args: list[str]) -> subprocess.CompletedProcess[str]:
        result = self.runner.run(args)
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip() or "command failed"
            raise SyncError(detail)
        return result

    def _run_json(self, args: list[str]) -> Any:
        result = self._run_checked(args)
        try:
            return json.loads(result.stdout or "{}")
        except json.JSONDecodeError as exc:
            raise SyncError(f"Invalid JSON from {' '.join(args)}: {exc}") from exc

    def _normalize_collection_payload(self, payload: Any, key: str) -> list[dict[str, Any]]:
        if isinstance(payload, dict):
            if key in payload:
                items = payload.get(key) or []
                if isinstance(items, list):
                    return [item for item in items if isinstance(item, dict)]
                if isinstance(items, dict):
                    return [items]
                return []
            return [payload]
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        return []

    def _parse_notebook_create_output(self, stdout: str, title: str) -> NotebookRef:
        notebook_id: str | None = None
        parsed_title = title
        for raw_line in stdout.splitlines():
            line = raw_line.strip()
            if line.startswith("ID:"):
                notebook_id = line.partition(":")[2].strip()
            if "Created notebook:" in line:
                parsed_title = line.partition("Created notebook:")[2].strip() or title
        if not notebook_id:
            raise SyncError(f"Unable to parse notebook id from create output: {stdout.strip() or '<empty>'}")
        return NotebookRef(notebook_id=notebook_id, title=parsed_title)

    def _parse_source_add_output(self, stdout: str, file_path: Path) -> NotebookSource:
        source_id: str | None = None
        parsed_title = file_path.name
        for raw_line in stdout.splitlines():
            line = raw_line.strip()
            if line.startswith("Source ID:"):
                source_id = line.partition(":")[2].strip()
            if "Added source:" in line:
                parsed_title = line.partition("Added source:")[2].strip()
                parsed_title = re.sub(r"\s+\(ready\)$", "", parsed_title)
        if not source_id:
            raise SyncError(f"Unable to parse source id from add output: {stdout.strip() or '<empty>'}")
        return NotebookSource(source_id=source_id, title=parsed_title)

    def ensure_available(self) -> None:
        result = self.runner.run([self.executable, "--help"])
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip() or "nlm not available"
            raise SyncError(detail)

    def list_notebooks(self) -> list[NotebookRef]:
        payload = self._run_json([self.executable, "notebook", "list", "--json"])
        return [
            NotebookRef(notebook_id=item["id"], title=item.get("title", ""))
            for item in self._normalize_collection_payload(payload, "notebooks")
        ]

    def ensure_authenticated(self) -> list[NotebookRef]:
        return self.list_notebooks()

    def create_notebook(self, title: str) -> NotebookRef:
        result = self._run_checked([self.executable, "notebook", "create", title])
        return self._parse_notebook_create_output(result.stdout, title)

    def list_sources(self, notebook_id: str) -> list[NotebookSource]:
        payload = self._run_json([self.executable, "source", "list", notebook_id, "--json"])
        return [
            NotebookSource(source_id=item["id"], title=item.get("title", ""))
            for item in self._normalize_collection_payload(payload, "sources")
        ]

    def delete_source(self, source_id: str) -> None:
        self._run_checked([self.executable, "source", "delete", source_id, "--confirm"])

    def add_file_source(self, notebook_id: str, file_path: Path) -> NotebookSource:
        result = self._run_checked(
            [
                self.executable,
                "source",
                "add",
                notebook_id,
                "--file",
                str(file_path),
                "--wait",
            ]
        )
        return self._parse_source_add_output(result.stdout, file_path)

    def get_source_content(self, source_id: str) -> str:
        payload = self._run_json([self.executable, "source", "content", source_id, "--json"])
        if isinstance(payload, dict):
            if "value" in payload and isinstance(payload["value"], dict):
                return str(payload["value"].get("content", ""))
            return str(payload.get("content", ""))
        return ""


def log(message: str) -> None:
    print(f"[sync_notebook_graph] {message}")


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "notebook"


def notebook_title_hash(title: str) -> str:
    return hashlib.sha1(title.encode("utf-8")).hexdigest()[:10]


def managed_container_name(project_slug: str) -> str:
    return f"{MANAGED_CONTAINER_PREFIX}-{project_slug}"


def managed_volume_name(project_slug: str, kind: str) -> str:
    return f"{managed_container_name(project_slug)}-{kind}"


def generate_neo4j_password() -> str:
    return secrets.token_urlsafe(18)


def find_free_port(exclude: set[int] | None = None) -> int:
    excluded = set(exclude or set())
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        if port not in excluded:
            return port


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_dataset_files(dataset_dir: Path) -> list[DatasetFile]:
    files: list[DatasetFile] = []
    for path in sorted(dataset_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.name.startswith(".") or path.name.startswith("~$"):
            continue
        if path.stat().st_size == 0:
            continue
        rel = path.relative_to(dataset_dir)
        files.append(DatasetFile(relative_path=rel, absolute_path=path, content_hash=file_sha256(path)))
    return files


def staged_txt_name_for(relative_path: Path) -> str:
    rel = relative_path.as_posix()
    stem = re.sub(r"[^A-Za-z0-9]+", "_", rel).strip("_").lower() or "source"
    suffix = hashlib.sha1(rel.encode("utf-8")).hexdigest()[:10]
    return f"{stem}__{suffix}.txt"


def save_manifest(
    manifest_path: Path,
    project_slug: str,
    notebook: NotebookRef,
    neo4j_runtime: Neo4jRuntime | None,
    entries: dict[str, ManifestEntry],
    removed_files: list[str],
) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "version": MANIFEST_VERSION,
        "project_slug": project_slug,
        "notebook": {"id": notebook.notebook_id, "title": notebook.title},
        "entries": {name: asdict(entry) for name, entry in sorted(entries.items())},
        "removed_files": sorted(removed_files),
    }
    if neo4j_runtime is not None:
        runtime_payload = asdict(neo4j_runtime)
        runtime_payload.pop("password", None)
        runtime_payload["deployment"] = "managed-local" if neo4j_runtime.container_name else "external"
        payload["neo4j"] = runtime_payload
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_manifest_state(manifest_path: Path) -> ManifestState:
    if not manifest_path.exists():
        return ManifestState(MANIFEST_VERSION, "", None, None, None, {}, [])
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    version = int(data.get("version") or 1)
    notebook = data.get("notebook") or {}
    project_slug = data.get("project_slug") or slugify(str(notebook.get("title") or ""))
    neo4j_data = data.get("neo4j")
    neo4j = None
    if neo4j_data:
        if "password" in neo4j_data:
            warnings.warn(
                f"Legacy plaintext Neo4j password found in {manifest_path}; set password_env and resave manifest v3.",
                FutureWarning,
                stacklevel=2,
            )
        neo4j = Neo4jRuntime(
            uri=str(neo4j_data["uri"]),
            username=str(neo4j_data.get("username") or neo4j_data.get("user") or DEFAULT_NEO4J_USER),
            password=str(neo4j_data.get("password") or ""),
            database=str(neo4j_data.get("database") or DEFAULT_NEO4J_DATABASE),
            password_env=str(neo4j_data.get("password_env") or "NEO4J_PASSWORD"),
            deployment=str(
                neo4j_data.get("deployment")
                or ("managed-local" if neo4j_data.get("container_name") else "external")
            ),
            container_name=neo4j_data.get("container_name"),
            container_id=neo4j_data.get("container_id"),
            bolt_port=neo4j_data.get("bolt_port"),
            http_port=neo4j_data.get("http_port"),
            image=neo4j_data.get("image"),
        )
    entries = {
        name: ManifestEntry(
            relative_path=str(entry.get("relative_path") or name),
            content_hash=str(entry.get("content_hash") or ""),
            source_id=str(entry.get("source_id") or entry.get("notebook_source_id") or ""),
            staged_txt_name=str(entry.get("staged_txt_name") or staged_txt_name_for(Path(name))),
            status=str(entry.get("status") or entry.get("last_sync_status") or "unknown"),
        )
        for name, entry in (data.get("entries") or {}).items()
    }
    removed_files = [str(item) for item in data.get("removed_files") or []]
    return ManifestState(
        version=version,
        project_slug=project_slug,
        notebook_id=notebook.get("id"),
        notebook_title=notebook.get("title"),
        neo4j=neo4j,
        entries=entries,
        removed_files=removed_files,
    )


class DockerNeo4jProvisioner:
    def __init__(self, *, runner: SubprocessRunner | None = None):
        self.runner = runner or SubprocessRunner()

    def ensure_available(self) -> None:
        result = self.runner.run(["docker", "version", "--format", "{{.Server.Version}}"])
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip() or "docker unavailable"
            raise SyncError(detail)

    def inspect_container(self, container_name: str) -> dict[str, Any] | None:
        result = self.runner.run(["docker", "inspect", container_name])
        if result.returncode != 0:
            detail = f"{result.stderr}\n{result.stdout}".strip()
            if "No such object" in detail:
                return None
            raise SyncError(detail or f"docker inspect failed for {container_name}")
        payload = json.loads(result.stdout or "[]")
        if not payload:
            return None
        return payload[0]

    def container_status(self, inspect_payload: dict[str, Any]) -> str:
        return str((inspect_payload.get("State") or {}).get("Status") or "")

    def container_labels(self, inspect_payload: dict[str, Any]) -> dict[str, str]:
        return dict((inspect_payload.get("Config") or {}).get("Labels") or {})

    def container_env(self, inspect_payload: dict[str, Any]) -> dict[str, str]:
        env_map: dict[str, str] = {}
        for item in (inspect_payload.get("Config") or {}).get("Env") or []:
            if "=" in item:
                key, value = item.split("=", 1)
                env_map[key] = value
        return env_map

    def host_port(self, inspect_payload: dict[str, Any], container_port: str) -> int:
        ports = ((inspect_payload.get("NetworkSettings") or {}).get("Ports") or {}).get(container_port) or []
        if not ports:
            raise SyncError(f"Container port mapping missing for {container_port}")
        return int(ports[0]["HostPort"])

    def runtime_from_inspect(self, inspect_payload: dict[str, Any]) -> Neo4jRuntime:
        env = self.container_env(inspect_payload)
        auth_value = env.get("NEO4J_AUTH", f"neo4j/{DEFAULT_NEO4J_PASSWORD}")
        password = auth_value.split("/", 1)[1] if "/" in auth_value else DEFAULT_NEO4J_PASSWORD
        bolt_port = self.host_port(inspect_payload, "7687/tcp")
        http_port = self.host_port(inspect_payload, "7474/tcp")
        return Neo4jRuntime(
            uri=f"bolt://127.0.0.1:{bolt_port}",
            username="neo4j",
            password=password,
            database="neo4j",
            deployment="managed-local",
            container_name=str(inspect_payload.get("Name", "")).lstrip("/"),
            container_id=str(inspect_payload.get("Id") or ""),
            bolt_port=bolt_port,
            http_port=http_port,
            image=str((inspect_payload.get("Config") or {}).get("Image") or DEFAULT_NEO4J_IMAGE),
        )

    def validate_managed_container(self, inspect_payload: dict[str, Any], project_slug: str, project_title_hash: str) -> None:
        labels = self.container_labels(inspect_payload)
        if labels.get(MANAGED_LABEL) != "true":
            raise SyncError(f"Managed Neo4j container {managed_container_name(project_slug)} is not managed by this repo.")
        existing_hash = labels.get(PROJECT_TITLE_HASH_LABEL)
        if existing_hash and existing_hash != project_title_hash:
            raise SyncError(
                f"Managed Neo4j container {managed_container_name(project_slug)} belongs to a different notebook title fingerprint."
            )

    def create_container(
        self,
        *,
        project_slug: str,
        project_title_hash: str,
        container_name: str,
        password: str,
        bolt_port: int,
        http_port: int,
        image: str,
    ) -> None:
        command = [
            "docker",
            "run",
            "-d",
            "--name",
            container_name,
            "--label",
            f"{MANAGED_LABEL}=true",
            "--label",
            f"{PROJECT_SLUG_LABEL}={project_slug}",
            "--label",
            f"{PROJECT_TITLE_HASH_LABEL}={project_title_hash}",
            "--publish",
            f"{http_port}:7474",
            "--publish",
            f"{bolt_port}:7687",
            "--volume",
            f"{managed_volume_name(project_slug, 'data')}:/data",
            "--volume",
            f"{managed_volume_name(project_slug, 'logs')}:/logs",
            "--volume",
            f"{managed_volume_name(project_slug, 'plugins')}:/plugins",
            "--env",
            f"NEO4J_AUTH=neo4j/{password}",
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
            f'--health-cmd=cypher-shell -u neo4j -p {password} "RETURN 1" || exit 1',
            "--health-interval=10s",
            "--health-timeout=10s",
            "--health-retries=12",
            "--health-start-period=30s",
            image,
        ]
        result = self.runner.run(command)
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip() or "docker run failed"
            raise SyncError(detail)

    def start_container(self, container_name: str) -> None:
        result = self.runner.run(["docker", "start", container_name])
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip() or "docker start failed"
            raise SyncError(detail)

    def wait_for_healthy(self, container_name: str, timeout_seconds: int = 180) -> dict[str, Any]:
        deadline = time.time() + timeout_seconds
        last_status = "unknown"
        while time.time() < deadline:
            inspect_payload = self.inspect_container(container_name)
            if inspect_payload is None:
                raise SyncError(f"Managed Neo4j container {container_name} disappeared during startup.")
            state = inspect_payload.get("State") or {}
            last_status = str(state.get("Status") or last_status)
            health = str((state.get("Health") or {}).get("Status") or "")
            if last_status == "running" and health in {"", "healthy"}:
                return inspect_payload
            time.sleep(5)
        raise SyncError(f"Managed Neo4j container {container_name} did not become healthy (last status: {last_status}).")

    def ensure_runtime(
        self,
        project_slug: str,
        project_title_hash: str,
        existing_runtime: Neo4jRuntime | None = None,
    ) -> Neo4jRuntime:
        if existing_runtime and not existing_runtime.container_name:
            return existing_runtime
        container_name = (existing_runtime.container_name if existing_runtime and existing_runtime.container_name else managed_container_name(project_slug))
        inspect_payload = self.inspect_container(container_name)
        if inspect_payload is None:
            password = existing_runtime.password if existing_runtime and existing_runtime.password else generate_neo4j_password()
            bolt_port = existing_runtime.bolt_port if existing_runtime and existing_runtime.bolt_port else find_free_port()
            http_port = existing_runtime.http_port if existing_runtime and existing_runtime.http_port else find_free_port({bolt_port})
            image = existing_runtime.image if existing_runtime and existing_runtime.image else DEFAULT_NEO4J_IMAGE
            self.create_container(
                project_slug=project_slug,
                project_title_hash=project_title_hash,
                container_name=container_name,
                password=password,
                bolt_port=bolt_port,
                http_port=http_port,
                image=image,
            )
            inspect_payload = self.wait_for_healthy(container_name)
            return self.runtime_from_inspect(inspect_payload)

        self.validate_managed_container(inspect_payload, project_slug, project_title_hash)
        if self.container_status(inspect_payload) != "running":
            self.start_container(container_name)
        inspect_payload = self.wait_for_healthy(container_name)
        return self.runtime_from_inspect(inspect_payload)


def explicit_runtime_from_args(args: argparse.Namespace) -> Neo4jRuntime | None:
    password_env = str(getattr(args, "neo4j_password_env", None) or "NEO4J_PASSWORD")
    cli_values = [args.neo4j_uri, args.neo4j_user, args.neo4j_password, args.neo4j_database]
    if not any(cli_values) and not os.environ.get("NEO4J_URI"):
        return None
    uri = args.neo4j_uri or os.environ.get("NEO4J_URI")
    user = args.neo4j_user or os.environ.get("NEO4J_USERNAME")
    password = args.neo4j_password or os.environ.get(password_env)
    database = args.neo4j_database or os.environ.get("NEO4J_DATABASE")
    values = [uri, user, password, database]
    if not all(values):
        raise SyncError(
            "External Neo4j runtime requires URI, username, password, and database via flags or NEO4J_* environment variables."
        )
    return Neo4jRuntime(
        uri=str(uri),
        username=str(user),
        password=str(password),
        database=str(database),
        password_env=password_env,
        deployment="external",
    )


def build_graph_command(args: argparse.Namespace, sources_dir: Path, runtime: Neo4jRuntime) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "notebooklm_graph_pipe.cli.build_graph",
        "--neo4j-uri",
        runtime.uri,
        "--neo4j-user",
        runtime.username,
        "--neo4j-database",
        runtime.database,
        "--sources-dir",
        str(sources_dir),
        "--model",
        args.model,
        "--parallel",
        str(args.parallel),
        "--poll-interval",
        str(args.poll_interval),
        "--min-file-size",
        str(args.min_file_size),
        "--token-chunk-size",
        str(args.token_chunk_size),
        "--chunk-overlap",
        str(args.chunk_overlap),
        "--chunks-to-combine",
        str(args.chunks_to_combine),
    ]
    if args.embedding_provider:
        command.extend(["--embedding-provider", args.embedding_provider])
    if args.embedding_model:
        command.extend(["--embedding-model", args.embedding_model])
    if args.llm_routing_config:
        command.extend(["--llm-routing-config", args.llm_routing_config])
    if args.skip_postprocess:
        command.append("--skip-postprocess")
    return command


def run_build_graph(args: argparse.Namespace, sources_dir: Path, runtime: Neo4jRuntime) -> None:
    command = build_graph_command(args, sources_dir, runtime)
    log(f"Running graph build: {' '.join(command)}")
    env = os.environ.copy()
    env.update(
        {
            "NEO4J_URI": runtime.uri,
            "NEO4J_USERNAME": runtime.username,
            "NEO4J_PASSWORD": runtime.password,
            "NEO4J_DATABASE": runtime.database,
        }
    )
    result = subprocess.run(command, text=True, check=False, env=env)
    if result.returncode != 0:
        raise SyncError(f"notebooklm_graph_pipe.cli.build_graph failed with exit code {result.returncode}")


def resolve_export_dir(args: argparse.Namespace, project_slug: str) -> Path:
    if args.export_dir:
        return Path(args.export_dir).resolve()
    return (REPO_ROOT / "data" / "notebooklm_exports" / project_slug).resolve()


def apply_dataset_registry_defaults(args: argparse.Namespace) -> None:
    if not getattr(args, "dataset_key", None):
        return
    entry = load_dataset_entry(args.dataset_key, getattr(args, "registry_path", None))
    if not args.notebook_id:
        args.notebook_id = entry.notebook.id
    if not args.notebook_title:
        args.notebook_title = entry.notebook.title
    if not args.export_dir:
        args.export_dir = str(default_export_dir(args.dataset_key))
    password_env = str(getattr(args, "neo4j_password_env", None) or "NEO4J_PASSWORD")
    if not args.neo4j_uri and not os.environ.get("NEO4J_URI"):
        args.neo4j_uri = entry.neo4j.uri
    if not args.neo4j_user and not os.environ.get("NEO4J_USERNAME"):
        args.neo4j_user = entry.neo4j.username
    if not args.neo4j_password and not os.environ.get(password_env):
        args.neo4j_password = entry.neo4j.password
    if not args.neo4j_database and not os.environ.get("NEO4J_DATABASE"):
        args.neo4j_database = entry.neo4j.database


def resolve_notebook(cli: NotebookLMCliAdapter, args: argparse.Namespace, manifest_state: ManifestState) -> NotebookRef:
    notebooks = cli.ensure_authenticated()
    if args.notebook_id:
        matches = [item for item in notebooks if item.notebook_id == args.notebook_id]
        if not matches:
            raise SyncError(f"NotebookLM notebook {args.notebook_id} was not found.")
        return matches[0]
    if manifest_state.notebook_id:
        matches = [item for item in notebooks if item.notebook_id == manifest_state.notebook_id]
        if matches:
            return matches[0]
    matches = [item for item in notebooks if item.title == args.notebook_title]
    if len(matches) > 1:
        raise SyncError(f"Multiple NotebookLM notebooks found with title '{args.notebook_title}'.")
    if len(matches) == 1:
        return matches[0]
    if args.command == "create":
        return cli.create_notebook(args.notebook_title)
    raise SyncError(f"NotebookLM notebook '{args.notebook_title}' was not found.")


def resolve_runtime_for_create(
    args: argparse.Namespace,
    manifest_state: ManifestState,
    provisioner: DockerNeo4jProvisioner,
    project_slug: str,
    project_title_hash: str,
) -> Neo4jRuntime:
    explicit_runtime = explicit_runtime_from_args(args)
    if explicit_runtime is not None:
        return explicit_runtime
    if manifest_state.neo4j is not None and not manifest_state.neo4j.container_name:
        password = os.environ.get(manifest_state.neo4j.password_env, "")
        if not password:
            raise SyncError(
                f"Set {manifest_state.neo4j.password_env} before using or upgrading the hosted Neo4j runtime recorded in the manifest."
            )
        return Neo4jRuntime(**{**asdict(manifest_state.neo4j), "password": password})
    return provisioner.ensure_runtime(project_slug, project_title_hash, manifest_state.neo4j)


def resolve_runtime_for_update(
    args: argparse.Namespace,
    manifest_state: ManifestState,
    provisioner: DockerNeo4jProvisioner,
    project_slug: str,
    project_title_hash: str,
) -> Neo4jRuntime:
    explicit_runtime = explicit_runtime_from_args(args)
    if explicit_runtime is not None:
        return explicit_runtime
    if manifest_state.neo4j is not None:
        if not manifest_state.neo4j.container_name:
            password = os.environ.get(manifest_state.neo4j.password_env, "")
            if not password:
                raise SyncError(
                    f"Set {manifest_state.neo4j.password_env} before using or upgrading the hosted Neo4j runtime recorded in the manifest."
                )
            return Neo4jRuntime(**{**asdict(manifest_state.neo4j), "password": password})
        return provisioner.ensure_runtime(project_slug, project_title_hash, manifest_state.neo4j)
    raise SyncError(
        "Legacy manifests without Neo4j runtime metadata require explicit --neo4j-uri/--neo4j-user/--neo4j-password/--neo4j-database."
    )


def export_source_text(target_path: Path, content: str) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(content, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sync a local dataset into NotebookLM and build/update its Neo4j graph.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(subparser: argparse.ArgumentParser, *, require_title: bool) -> None:
        subparser.add_argument("--dataset-dir", required=True, help="Directory containing the source dataset")
        subparser.add_argument("--dataset-key", help="Optional dataset key from a local registry JSON")
        subparser.add_argument("--registry-path", help="Path to an optional local dataset registry JSON")
        subparser.add_argument("--notebook-title", required=False, help="NotebookLM notebook title")
        subparser.add_argument("--notebook-id", help="NotebookLM notebook id override")
        subparser.add_argument("--export-dir", help="Export directory for manifest and staged sources")
        subparser.add_argument("--neo4j-uri", help="Explicit Neo4j Bolt URI")
        subparser.add_argument("--neo4j-user", help="Explicit Neo4j username")
        subparser.add_argument("--neo4j-password", help="Explicit Neo4j password")
        subparser.add_argument("--neo4j-password-env", default="NEO4J_PASSWORD", help="Environment variable containing the Neo4j password")
        subparser.add_argument("--neo4j-database", help="Explicit Neo4j database")
        subparser.add_argument("--model", default="google_flash", help="Graph extraction model name")
        subparser.add_argument("--parallel", type=int, default=1, help="Concurrent extractions")
        subparser.add_argument("--poll-interval", type=int, default=15, help="Extraction polling interval")
        subparser.add_argument("--min-file-size", type=int, default=10, help="Skip staged files smaller than N bytes")
        subparser.add_argument("--token-chunk-size", type=int, default=2000, help="Token chunk size")
        subparser.add_argument("--chunk-overlap", type=int, default=200, help="Chunk overlap")
        subparser.add_argument("--chunks-to-combine", type=int, default=1, help="Chunks to combine")
        subparser.add_argument("--embedding-provider", default=None, help="Embedding provider override")
        subparser.add_argument("--embedding-model", default=None, help="Embedding model override")
        subparser.add_argument("--llm-routing-config", default=None, help="Optional JSON config for role-based LLM routing")
        subparser.add_argument("--skip-build", action="store_true", help="Do not run scripts/build_graph.py")
        subparser.add_argument("--skip-postprocess", action="store_true", help="Pass --skip-postprocess to scripts/build_graph.py")

    add_common(subparsers.add_parser("create", help="Create or reuse notebook/runtime and sync files"), require_title=True)
    add_common(subparsers.add_parser("update", help="Update an existing notebook/runtime and sync files"), require_title=False)
    return parser


def sync_dataset(args: argparse.Namespace) -> int:
    apply_dataset_registry_defaults(args)
    dataset_dir = Path(args.dataset_dir).resolve()
    if not dataset_dir.exists():
        raise SyncError(f"Dataset directory not found: {dataset_dir}")
    if args.command == "create" and not args.notebook_title:
        raise SyncError("Create requires --notebook-title.")
    if args.command == "update" and not args.notebook_title and not args.notebook_id:
        raise SyncError("Update requires --notebook-title or --notebook-id.")

    cli = NotebookLMCliAdapter()
    cli.ensure_available()
    explicit_runtime = explicit_runtime_from_args(args)
    provisioner = DockerNeo4jProvisioner()
    if explicit_runtime is None:
        provisioner.ensure_available()

    export_dir_hint = resolve_export_dir(args, slugify(args.notebook_title or args.notebook_id or "notebook"))
    manifest_path = export_dir_hint / "manifest.json"
    manifest_state = load_manifest_state(manifest_path)
    notebook = resolve_notebook(cli, args, manifest_state)
    if manifest_state.notebook_id and manifest_state.notebook_id != notebook.notebook_id:
        raise SyncError(f"Export dir {export_dir_hint} belongs to notebook {manifest_state.notebook_id}.")

    project_slug = manifest_state.project_slug or slugify(notebook.title)
    export_dir = resolve_export_dir(args, project_slug)
    manifest_path = export_dir / "manifest.json"
    manifest_state = load_manifest_state(manifest_path)
    if manifest_state.notebook_id and manifest_state.notebook_id != notebook.notebook_id:
        raise SyncError(f"Export dir {export_dir} belongs to notebook {manifest_state.notebook_id}.")

    title_hash = notebook_title_hash(notebook.title)
    if args.command == "create":
        runtime = resolve_runtime_for_create(args, manifest_state, provisioner, project_slug, title_hash)
    else:
        runtime = resolve_runtime_for_update(args, manifest_state, provisioner, project_slug, title_hash)

    graph_api = GraphBuilderAPI(
        neo4j_uri=runtime.uri,
        neo4j_user=runtime.username,
        neo4j_password=runtime.password,
        neo4j_database=runtime.database,
        sources_dir=export_dir / "sources",
    )
    if hasattr(graph_api, "preflight_capabilities"):
        graph_api.preflight_capabilities()
    graph_sources = {
        row.get("fileName") or row.get("name") or row.get("file_name")
        for row in graph_api.sources_list()
    }

    dataset_files = discover_dataset_files(dataset_dir)
    existing_sources = {source.source_id: source for source in cli.list_sources(notebook.notebook_id)}
    entries: dict[str, ManifestEntry] = dict(manifest_state.entries)
    current_paths = {item.relative_path.as_posix() for item in dataset_files}
    save_manifest(manifest_path, project_slug, notebook, runtime, entries, manifest_state.removed_files)

    for dataset_file in dataset_files:
        relative_name = dataset_file.relative_path.as_posix()
        previous = entries.get(relative_name)
        staged_txt_name = previous.staged_txt_name if previous else staged_txt_name_for(dataset_file.relative_path)
        staged_path = export_dir / "sources" / staged_txt_name
        has_live_source = bool(previous and previous.source_id and previous.source_id in existing_sources)
        needs_refresh = (
            previous is None
            or previous.content_hash != dataset_file.content_hash
            or previous.status != "exported"
            or not staged_path.exists()
            or not has_live_source
        )
        if not needs_refresh:
            continue
        if has_live_source:
            try:
                cli.delete_source(previous.source_id)
            except SyncError as exc:
                raise SyncError(f"Failed to replace existing NotebookLM source for {relative_name}: {exc}") from exc
        source = cli.add_file_source(notebook.notebook_id, dataset_file.absolute_path)
        content = cli.get_source_content(source.source_id)
        export_source_text(staged_path, content)
        entries[relative_name] = ManifestEntry(
            relative_path=relative_name,
            content_hash=dataset_file.content_hash,
            source_id=source.source_id,
            staged_txt_name=staged_txt_name,
            status="exported",
        )
        if previous and staged_txt_name in graph_sources:
            graph_api.retry_processing(staged_txt_name, RETRY_CONDITION)
        save_manifest(
            manifest_path,
            project_slug,
            notebook,
            runtime,
            entries,
            sorted(set(entries) - current_paths),
        )

    removed_files = sorted(set(entries) - current_paths)
    save_manifest(manifest_path, project_slug, notebook, runtime, entries, removed_files)

    if not args.skip_build:
        run_build_graph(args, export_dir / "sources", runtime)
    return 0


def main() -> int:
    args = build_parser().parse_args()
    return sync_dataset(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SyncError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
