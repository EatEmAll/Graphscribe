#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.parse import urlparse

import yaml
from neo4j import GraphDatabase


UUID_PATTERN = re.compile(
    r"(?<![0-9a-f])([0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12})(?![0-9a-f])",
    re.IGNORECASE,
)
MARKER_NAME = ".notebooklm-corpus-export.json"
SUPPORTED_ORIGINAL_SUFFIXES = {".txt", ".md", ".markdown", ".pdf"}


class ExportError(RuntimeError):
    pass


@dataclass(frozen=True)
class Notebook:
    id: str
    title: str


@dataclass(frozen=True)
class Source:
    id: str
    title: str
    source_type: str
    url: str | None = None


class NlmClient:
    def __init__(
        self,
        *,
        executable: str = "nlm",
        profile: str | None = None,
        runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    ):
        self.executable = executable
        self.profile = profile
        self.runner = runner

    def _json(self, *arguments: str) -> Any:
        command = [self.executable, *arguments, "--json"]
        if self.profile:
            command.extend(["--profile", self.profile])
        result = self.runner(command, capture_output=True, text=True, check=False, timeout=180)
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip() or "unknown nlm failure"
            raise ExportError(f"NotebookLM command failed: {detail}")
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise ExportError("NotebookLM returned invalid JSON.") from exc

    def notebooks(self) -> list[Notebook]:
        payload = self._json("list", "notebooks")
        rows = payload.get("notebooks") if isinstance(payload, dict) else payload
        if not isinstance(rows, list):
            raise ExportError("NotebookLM notebook list has an unexpected shape.")
        return [Notebook(str(row["id"]), str(row.get("title") or row["id"])) for row in rows]

    def sources(self, notebook_id: str) -> list[Source]:
        payload = self._json("list", "sources", notebook_id)
        rows = payload.get("sources") if isinstance(payload, dict) else payload
        if not isinstance(rows, list):
            raise ExportError(f"NotebookLM source list for {notebook_id} has an unexpected shape.")
        return [
            Source(
                id=str(row["id"]),
                title=str(row.get("title") or row["id"]),
                source_type=str(row.get("type") or row.get("source_type") or "unknown"),
                url=str(row.get("url") or "") or None,
            )
            for row in rows
        ]

    def source_content(self, source: Source) -> dict[str, Any]:
        payload = self._json("source", "get", source.id)
        value = payload.get("value") if isinstance(payload, dict) else None
        if not isinstance(value, dict):
            raise ExportError(f"NotebookLM source {source.id} has an unexpected content response.")
        return value


def extract_source_ids(file_names: Iterable[str]) -> set[str]:
    ids: set[str] = set()
    for name in file_names:
        ids.update(match.lower() for match in UUID_PATTERN.findall(str(name or "")))
    return ids


def read_legacy_source_ids(uri: str, username: str, password: str, database: str) -> set[str]:
    if not uri or not password:
        raise ExportError("Legacy Neo4j URI and password are required.")
    with GraphDatabase.driver(uri, auth=(username, password)) as driver:
        driver.verify_connectivity()
        with driver.session(database=database) as session:
            names = [
                str(row["file_name"])
                for row in session.run(
                    "MATCH (d:Document) WHERE d.fileName IS NOT NULL RETURN d.fileName AS file_name"
                )
            ]
    ids = extract_source_ids(names)
    if not ids:
        raise ExportError("No NotebookLM source UUIDs were found in legacy Document.fileName values.")
    return ids


def matched_notebooks(
    client: NlmClient,
    legacy_ids: set[str],
    selected_notebook_ids: set[str] | None = None,
) -> tuple[list[Notebook], dict[str, list[Source]]]:
    notebooks = client.notebooks()
    if selected_notebook_ids:
        unknown = selected_notebook_ids - {notebook.id for notebook in notebooks}
        if unknown:
            raise ExportError(f"Notebook IDs not found: {', '.join(sorted(unknown))}")
        notebooks = [notebook for notebook in notebooks if notebook.id in selected_notebook_ids]
    source_map = {notebook.id: client.sources(notebook.id) for notebook in notebooks}
    matches = [
        notebook
        for notebook in notebooks
        if legacy_ids.intersection(source.id.lower() for source in source_map[notebook.id])
    ]
    if selected_notebook_ids:
        unmatched = selected_notebook_ids - {notebook.id for notebook in matches}
        if unmatched:
            raise ExportError(f"Selected notebooks contain no legacy source UUIDs: {', '.join(sorted(unmatched))}")
    if not matches:
        raise ExportError("No NotebookLM notebook contains a legacy source UUID.")
    return matches, source_map


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_youtube_url(value: str | None) -> bool:
    if not value:
        return False
    host = (urlparse(value).hostname or "").lower()
    return host in {"youtube.com", "www.youtube.com", "youtu.be", "www.youtu.be"}


def _source_url(detail: dict[str, Any], source: Source) -> str | None:
    url = str(detail.get("url") or source.url or "").strip()
    if url:
        return url
    title = str(detail.get("title") or source.title or "").strip()
    return title if _is_youtube_url(title) else None


def index_originals(root: Path | None) -> dict[str, list[Path]]:
    if root is None:
        return {}
    if not root.is_dir():
        raise ExportError(f"Originals root not found: {root}")
    result: dict[str, list[Path]] = {}
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_ORIGINAL_SUFFIXES:
            result.setdefault(path.name.casefold(), []).append(path)
    return result


def prepare_output(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        marker = path / MARKER_NAME
        if not overwrite:
            raise ExportError(f"Output directory is not empty: {path}")
        try:
            marker_payload = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            marker_payload = {}
        if marker_payload != {"format": "notebooklm-corpus-export", "version": 1}:
            raise ExportError(f"Refusing to overwrite an unmarked directory: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    (path / MARKER_NAME).write_text(
        json.dumps({"format": "notebooklm-corpus-export", "version": 1}, indent=2),
        encoding="utf-8",
    )


def _write_markdown(path: Path, metadata: dict[str, Any], content: str) -> None:
    frontmatter = yaml.safe_dump(metadata, allow_unicode=True, sort_keys=True).strip()
    path.write_text(f"---\n{frontmatter}\n---\n\n{content.strip()}\n", encoding="utf-8")


def _original_target_name(title: str, source_id: str, suffix: str) -> str:
    stem = Path(title).stem
    safe_stem = re.sub(r"[^0-9A-Za-z._-]+", "-", stem).strip("-._") or "source"
    return f"{safe_stem[:80]}__{source_id}{suffix.lower()}"


def export_sources(
    client: NlmClient,
    notebooks: list[Notebook],
    source_map: dict[str, list[Source]],
    legacy_ids: set[str],
    output_dir: Path,
    *,
    originals_root: Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    prepare_output(output_dir, overwrite)
    document_dir = output_dir / "documents"
    document_dir.mkdir()
    originals = index_originals(originals_root)
    notebook_by_id = {notebook.id: notebook for notebook in notebooks}
    unique_sources: dict[str, Source] = {}
    source_notebooks: dict[str, list[str]] = {}
    for notebook in notebooks:
        for source in source_map[notebook.id]:
            unique_sources.setdefault(source.id, source)
            source_notebooks.setdefault(source.id, []).append(notebook.id)

    details = {source_id: client.source_content(source) for source_id, source in unique_sources.items()}
    youtube_url_counts: dict[str, int] = {}
    for source_id, source in unique_sources.items():
        detail = details[source_id]
        url = _source_url(detail, source)
        if _is_youtube_url(url):
            youtube_url_counts[url] = youtube_url_counts.get(url, 0) + 1

    records: list[dict[str, Any]] = []
    youtube: list[dict[str, Any]] = []
    for source_id in sorted(unique_sources):
        source = unique_sources[source_id]
        detail = details[source_id]
        title = str(detail.get("title") or source.title).strip() or source.id
        source_type = str(detail.get("source_type") or source.source_type or "unknown")
        url = _source_url(detail, source)
        content = str(detail.get("content") or "").strip()
        matches = originals.get(title.casefold(), [])
        warnings: list[str] = []
        if len(matches) == 1:
            original = matches[0]
            target = document_dir / _original_target_name(title, source.id, original.suffix)
            shutil.copy2(original, target)
            acquisition = "original_file"
            relative_path = target.relative_to(output_dir).as_posix()
            checksum = _sha256(target)
        elif _is_youtube_url(url) and youtube_url_counts[url] == 1:
            youtube.append({"url": url, "title": title, "preferred_languages": ["en"]})
            acquisition = "youtube_url"
            relative_path = None
            checksum = hashlib.sha256(url.encode("utf-8")).hexdigest()
        else:
            if not content:
                raise ExportError(f"NotebookLM source {source.id} ({title}) has no recoverable content.")
            target = document_dir / f"{source.id}.md"
            _write_markdown(
                target,
                {
                    "title": title,
                    "notebooklm": {
                        "source_id": source.id,
                        "notebook_ids": sorted(source_notebooks[source.id]),
                        "source_type": source_type,
                        "original_url": url,
                        "acquisition": "notebooklm_text_fallback",
                    },
                },
                content,
            )
            acquisition = "notebooklm_text_fallback"
            relative_path = target.relative_to(output_dir).as_posix()
            checksum = _sha256(target)
            if len(matches) > 1:
                warnings.append("Multiple original files matched the title; NotebookLM text was used.")
            if _is_youtube_url(url) and youtube_url_counts[url] > 1:
                warnings.append("The YouTube URL is shared by multiple source IDs; NotebookLM text preserved their identity.")
        records.append(
            {
                "source_id": source.id,
                "title": title,
                "source_type": source_type,
                "url": url,
                "notebooks": [
                    {"id": notebook_id, "title": notebook_by_id[notebook_id].title}
                    for notebook_id in sorted(source_notebooks[source.id])
                ],
                "classification": "legacy_matched" if source.id.lower() in legacy_ids else "current_new",
                "acquisition": acquisition,
                "relative_path": relative_path,
                "sha256": checksum,
                "warnings": warnings,
            }
        )

    if youtube:
        (output_dir / "sources.yaml").write_text(
            yaml.safe_dump({"version": 1, "youtube": youtube}, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
    current_ids = {source_id.lower() for source_id in unique_sources}
    inventory = {
        "version": 1,
        "notebooks": [{"id": notebook.id, "title": notebook.title} for notebook in notebooks],
        "summary": {
            "legacy_source_ids": len(legacy_ids),
            "current_sources": len(records),
            "legacy_matched": sum(row["classification"] == "legacy_matched" for row in records),
            "current_new": sum(row["classification"] == "current_new" for row in records),
            "legacy_missing": len(legacy_ids - current_ids),
        },
        "legacy_missing_source_ids": sorted(legacy_ids - current_ids),
        "sources": records,
    }
    (output_dir / "source_inventory.json").write_text(
        json.dumps(inventory, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return inventory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export legacy-matched NotebookLM notebooks as a v3 corpus dataset.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--originals-root", type=Path)
    parser.add_argument("--notebook-id", action="append", default=[])
    parser.add_argument("--profile")
    parser.add_argument("--nlm-bin", default="nlm")
    parser.add_argument("--legacy-uri", default=os.environ.get("NEO4J_URI"))
    parser.add_argument("--legacy-user", default=os.environ.get("NEO4J_USERNAME", "neo4j"))
    parser.add_argument("--legacy-database", default=os.environ.get("NEO4J_DATABASE", "neo4j"))
    parser.add_argument("--legacy-password-env", default="NEO4J_PASSWORD")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    password = os.environ.get(args.legacy_password_env, "")
    legacy_ids = read_legacy_source_ids(args.legacy_uri or "", args.legacy_user, password, args.legacy_database)
    client = NlmClient(executable=args.nlm_bin, profile=args.profile)
    notebooks, source_map = matched_notebooks(client, legacy_ids, set(args.notebook_id) or None)
    inventory = export_sources(
        client,
        notebooks,
        source_map,
        legacy_ids,
        args.output_dir.resolve(),
        originals_root=args.originals_root.resolve() if args.originals_root else None,
        overwrite=args.overwrite,
    )
    print(json.dumps(inventory["summary"], indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ExportError as exc:
        print(f"ERROR: {exc}", file=os.sys.stderr)
        raise SystemExit(1)
