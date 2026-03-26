from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_REGISTRY_PATH = REPO_ROOT / "benchmark_dataset_registry.json"


@dataclass(frozen=True)
class RegistryNotebook:
    id: str
    title: str


@dataclass(frozen=True)
class RegistryNeo4j:
    uri: str
    username: str
    password: str
    database: str
    container_name: str | None = None
    container_id: str | None = None
    bolt_port: int | None = None
    http_port: int | None = None
    image: str | None = None


@dataclass(frozen=True)
class DatasetRegistryEntry:
    key: str
    notebook: RegistryNotebook
    neo4j: RegistryNeo4j
    runs_dir: str | None = None


def _to_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def default_registry_path() -> Path:
    return DEFAULT_REGISTRY_PATH


def default_export_dir(dataset_key: str) -> Path:
    return (REPO_ROOT / "data" / "notebooklm_exports" / dataset_key).resolve()


def default_sources_dir(dataset_key: str) -> Path:
    return default_export_dir(dataset_key) / "sources"


def load_dataset_registry(registry_path: str | Path | None = None) -> dict[str, DatasetRegistryEntry]:
    path = Path(registry_path).resolve() if registry_path else default_registry_path()
    if not path.exists():
        raise ValueError(f"Dataset registry not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    datasets = payload.get("datasets")
    if not isinstance(datasets, dict):
        raise ValueError(f"Invalid dataset registry format: {path}")

    entries: dict[str, DatasetRegistryEntry] = {}
    for key, raw_entry in datasets.items():
        if not isinstance(raw_entry, dict):
            continue
        notebook = raw_entry.get("notebook") or {}
        neo4j = raw_entry.get("neo4j") or {}
        entries[str(key)] = DatasetRegistryEntry(
            key=str(key),
            notebook=RegistryNotebook(
                id=str(notebook.get("id") or ""),
                title=str(notebook.get("title") or ""),
            ),
            neo4j=RegistryNeo4j(
                uri=str(neo4j.get("uri") or ""),
                username=str(neo4j.get("username") or neo4j.get("user") or ""),
                password=str(neo4j.get("password") or ""),
                database=str(neo4j.get("database") or ""),
                container_name=neo4j.get("container_name"),
                container_id=neo4j.get("container_id"),
                bolt_port=_to_int(neo4j.get("bolt_port")),
                http_port=_to_int(neo4j.get("http_port")),
                image=neo4j.get("image"),
            ),
            runs_dir=str(raw_entry.get("runs_dir") or "") or None,
        )
    return entries


def load_dataset_entry(dataset_key: str, registry_path: str | Path | None = None) -> DatasetRegistryEntry:
    entries = load_dataset_registry(registry_path)
    try:
        return entries[dataset_key]
    except KeyError as exc:
        raise ValueError(f"Dataset '{dataset_key}' not found in registry.") from exc
