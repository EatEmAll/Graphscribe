#!/usr/bin/env python3
"""Prepare bounded benchmark corpora for the local NotebookLM + Neo4j workflow.

The pipeline creates one canonical `sources/` directory per dataset and writes
manifests that plug directly into this repo's `scripts/sync_notebook_graph.py`
for notebook creation/update plus graph rebuilds.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable
from urllib import error, parse, request


REPO_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_ROOT = REPO_ROOT / "benchmark-datasets"
GRAPH_BUILDER_ROOT = REPO_ROOT
SYNC_NOTEBOOK_GRAPH = GRAPH_BUILDER_ROOT / "scripts" / "sync_notebook_graph.py"

USER_AGENT = "benchmark-corpus-pipeline/1.0"
DEFAULT_TIMEOUT = 60
OPEN_TARGETS_GRAPHQL_URL = "https://api.platform.opentargets.org/api/v4/graphql"
OPENALEX_WORKS_URL = "https://api.openalex.org/works"
IMDB_DOWNLOADS = {
    "title.basics.tsv.gz": "https://datasets.imdbws.com/title.basics.tsv.gz",
    "title.ratings.tsv.gz": "https://datasets.imdbws.com/title.ratings.tsv.gz",
    "title.principals.tsv.gz": "https://datasets.imdbws.com/title.principals.tsv.gz",
    "name.basics.tsv.gz": "https://datasets.imdbws.com/name.basics.tsv.gz",
}


@dataclass(frozen=True)
class DatasetConfig:
    folder_slug: str
    notebook_title: str
    neo4j_database: str
    target_size: int
    record_type: str
    description: str


DATASETS: dict[str, DatasetConfig] = {
    "openalex-rag": DatasetConfig(
        folder_slug="openalex-rag",
        notebook_title="bench-openalex-rag",
        neo4j_database="bench_openalex",
        target_size=100,
        record_type="scholarly_work",
        description="OpenAlex works matching retrieval augmented generation (2023-2025).",
    ),
    "imdb-scifi": DatasetConfig(
        folder_slug="imdb-scifi",
        notebook_title="bench-imdb-scifi",
        neo4j_database="bench_imdb",
        target_size=100,
        record_type="movie",
        description="IMDb sci-fi movies (2000-2024) sorted by vote count.",
    ),
    "opentargets-alzheimers": DatasetConfig(
        folder_slug="opentargets-alzheimers",
        notebook_title="bench-opentargets-alzheimers",
        neo4j_database="bench_opentargets",
        target_size=100,
        record_type="target_disease_association",
        description="Open Targets Alzheimer's disease target associations sorted by score.",
    ),
}


def utc_now() -> str:
    """Return an ISO8601 UTC timestamp."""
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def sanitize_filename(value: str) -> str:
    """Return a stable ASCII-ish file stem."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("._")
    return cleaned or "record"


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fetch_json(url: str, params: dict[str, Any] | None = None, timeout: int = DEFAULT_TIMEOUT) -> Any:
    """Fetch JSON over HTTP using stdlib only."""
    full_url = url
    if params:
        query_string = parse.urlencode(params, doseq=True)
        full_url = f"{url}?{query_string}"
    req = request.Request(full_url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    with request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def post_json(url: str, payload: dict[str, Any], timeout: int = DEFAULT_TIMEOUT) -> Any:
    """POST a JSON payload and decode a JSON response."""
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=body,
        headers={"User-Agent": USER_AGENT, "Accept": "application/json", "Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def download_file(url: str, destination: Path, overwrite: bool = False, timeout: int = DEFAULT_TIMEOUT) -> Path:
    """Download a remote file to disk."""
    if destination.exists() and not overwrite:
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    req = request.Request(url, headers={"User-Agent": USER_AGENT})
    with request.urlopen(req, timeout=timeout) as response:
        destination.write_bytes(response.read())
    return destination


def reconstruct_openalex_abstract(inverted_index: dict[str, list[int]] | None) -> str:
    """Reconstruct OpenAlex abstract text from the inverted index."""
    if not inverted_index:
        return ""
    size = 0
    for positions in inverted_index.values():
        if positions:
            size = max(size, max(positions) + 1)
    words = [""] * size
    for token, positions in inverted_index.items():
        for idx in positions:
            if 0 <= idx < size:
                words[idx] = token
    return " ".join(part for part in words if part).strip()


def format_bullets(values: Iterable[str], empty_text: str = "None listed") -> str:
    """Format a flat bullet list for the source documents."""
    items = [value.strip() for value in values if value and value.strip()]
    if not items:
        return f"- {empty_text}"
    return "\n".join(f"- {item}" for item in items)


def join_text(values: Iterable[str], empty_text: str = "Unknown") -> str:
    """Join non-empty values with commas."""
    items = [value.strip() for value in values if value and value.strip()]
    return ", ".join(items) if items else empty_text


def ensure_dataset_layout(dataset: DatasetConfig) -> dict[str, Path]:
    """Create the standard raw/sources/manifest folder structure."""
    root = BENCHMARK_ROOT / dataset.folder_slug
    paths = {
        "root": root,
        "raw": root / "raw",
        "sources": root / "sources",
        "manifest": root / "manifest",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def load_json_if_present(path: Path) -> Any | None:
    """Read a JSON file if it exists."""
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def update_dataset_metadata(paths: dict[str, Path], dataset: DatasetConfig, notebook_id: str | None = None) -> None:
    """Write the dataset metadata manifest."""
    metadata_path = paths["manifest"] / "dataset_config.json"
    existing = load_json_if_present(metadata_path) or {}
    metadata = {
        "description": dataset.description,
        "folder_slug": dataset.folder_slug,
        "last_updated_utc": utc_now(),
        "neo4j_database": dataset.neo4j_database,
        "notebook_id": notebook_id if notebook_id is not None else existing.get("notebook_id"),
        "notebook_title": dataset.notebook_title,
        "record_type": dataset.record_type,
        "target_size": dataset.target_size,
    }
    write_json(metadata_path, metadata)


def default_export_dir(dataset: DatasetConfig) -> Path:
    """Return the default local export root used by the sync workflow."""
    return GRAPH_BUILDER_ROOT / "data" / "notebooklm_exports" / dataset.notebook_title


def build_sync_command(
    dataset: DatasetConfig,
    paths: dict[str, Path],
    *,
    mode: str,
    notebook_id: str | None = None,
) -> str:
    """Build a local sync_notebook_graph.py command string."""
    command = [
        "python",
        f"\"{SYNC_NOTEBOOK_GRAPH}\"",
        mode,
        "--dataset-dir",
        f"\"{paths['sources']}\"",
        "--export-dir",
        f"\"{default_export_dir(dataset)}\"",
        "--neo4j-database",
        f"\"{dataset.neo4j_database}\"",
        "--parallel",
        "1",
    ]
    if mode == "create":
        command.extend(["--notebook-title", f"\"{dataset.notebook_title}\""])
    elif notebook_id:
        command.extend(["--notebook-id", f"\"{notebook_id}\""])
    else:
        command.extend(["--notebook-title", f"\"{dataset.notebook_title}\""])
    return " ".join(command)


def write_source_manifests(
    dataset: DatasetConfig,
    paths: dict[str, Path],
    source_rows: list[dict[str, Any]],
    acquisition: dict[str, Any],
    notebook_id: str | None = None,
) -> None:
    """Write all manifests shared by NotebookLM and Neo4j ingestion."""
    update_dataset_metadata(paths, dataset, notebook_id=notebook_id)
    write_json(paths["manifest"] / "acquisition.json", acquisition)
    write_json(paths["manifest"] / "source_files.json", {"generated_at_utc": utc_now(), "sources": source_rows})
    write_csv(
        paths["manifest"] / "source_files.csv",
        ["external_id", "filename", "title", "source_path", "record_type"],
        source_rows,
    )

    notebook_manifest = {
        "create_command": build_sync_command(dataset, paths, mode="create"),
        "dataset": dataset.folder_slug,
        "dataset_dir": str(paths["sources"]),
        "export_dir": str(default_export_dir(dataset)),
        "generated_at_utc": utc_now(),
        "import_mode": "sync_notebook_graph",
        "manual_fallback": [
            "Preferred flow: run sync_notebook_graph.py create once, then use update for subsequent dataset refreshes.",
            f"If needed, the generated corpus lives at '{paths['sources']}'.",
            "The graph rebuild is already part of the sync command unless you pass --skip-build.",
        ],
        "mcp_recommended_sequence": [
            "mcp__notebooklm-mcp__notebook_create",
            "mcp__notebooklm-mcp__source_add",
            "mcp__notebooklm-mcp__notebook_get",
        ],
        "notebook_id": notebook_id,
        "notebook_title": dataset.notebook_title,
        "source_count": len(source_rows),
        "source_dir": str(paths["sources"]),
        "source_files_csv": str(paths["manifest"] / "source_files.csv"),
        "update_command": build_sync_command(dataset, paths, mode="update", notebook_id=notebook_id),
    }
    write_json(paths["manifest"] / "notebooklm_import.json", notebook_manifest)

    sync_manifest = {
        "create_command": build_sync_command(dataset, paths, mode="create"),
        "build_graph_script": str(GRAPH_BUILDER_ROOT / "scripts" / "build_graph.py"),
        "dataset": dataset.folder_slug,
        "dataset_dir": str(paths["sources"]),
        "export_dir": str(default_export_dir(dataset)),
        "generated_at_utc": utc_now(),
        "neo4j_database": dataset.neo4j_database,
        "parallel": 1,
        "postprocess": True,
        "run_consolidation": False,
        "sync_script": str(SYNC_NOTEBOOK_GRAPH),
        "update_command": build_sync_command(dataset, paths, mode="update", notebook_id=notebook_id),
    }
    write_json(paths["manifest"] / "sync_workflow.json", sync_manifest)


def source_row(filename: str, external_id: str, title: str, record_type: str, source_path: Path) -> dict[str, Any]:
    """Build a source manifest row."""
    return {
        "external_id": external_id,
        "filename": filename,
        "record_type": record_type,
        "source_path": str(source_path),
        "title": title,
    }


def prepare_openalex(
    paths: dict[str, Path],
    dataset: DatasetConfig,
    *,
    skip_download: bool,
    target_size: int,
    notebook_id: str | None,
) -> None:
    """Download and transform the OpenAlex RAG slice."""
    raw_path = paths["raw"] / "openalex_works.json"
    if not skip_download:
        params = {
            "search": "retrieval augmented generation",
            "filter": "publication_year:2023-2025,has_abstract:true,language:en",
            "per-page": target_size,
            "sort": "cited_by_count:desc",
        }
        raw_data = fetch_json(OPENALEX_WORKS_URL, params=params)
        write_json(raw_path, raw_data)
    else:
        raw_data = load_json_if_present(raw_path)
        if raw_data is None:
            raise FileNotFoundError(
                f"OpenAlex raw file not found at {raw_path}. "
                "Place a saved API response there or rerun without --skip-download."
            )

    results = raw_data.get("results", [])[:target_size]
    generated_rows: list[dict[str, Any]] = []
    for item in results:
        work_id = item.get("id") or item.get("ids", {}).get("openalex")
        if not work_id:
            continue
        title = item.get("title") or "Untitled work"
        abstract = reconstruct_openalex_abstract(item.get("abstract_inverted_index"))
        authorships = item.get("authorships") or []
        author_lines = []
        institution_names: list[str] = []
        for authorship in authorships:
            author_name = (authorship.get("author") or {}).get("display_name") or "Unknown author"
            institutions = [inst.get("display_name", "").strip() for inst in authorship.get("institutions") or []]
            institution_names.extend(institutions)
            if institutions:
                author_lines.append(f"{author_name} ({'; '.join(inst for inst in institutions if inst)})")
            else:
                author_lines.append(author_name)

        topic_names = [topic.get("display_name", "").strip() for topic in item.get("topics") or []]
        venue = ((item.get("primary_location") or {}).get("source") or {}).get("display_name") or "Unknown venue"
        doi = item.get("doi") or "None"
        referenced_works = item.get("referenced_works") or []
        filename = f"work_{sanitize_filename(work_id.rsplit('/', 1)[-1])}.txt"
        source_path = paths["sources"] / filename
        content = "\n".join(
            [
                f"Dataset: {dataset.notebook_title}",
                "Record Type: scholarly_work",
                f"External ID: {work_id}",
                f"Source URL: {work_id}",
                f"Generated At: {utc_now()}",
                "---",
                f"Title: {title}",
                f"Publication Year: {item.get('publication_year', 'Unknown')}",
                f"Venue: {venue}",
                f"DOI: {doi}",
                f"Cited By Count: {item.get('cited_by_count', 'Unknown')}",
                "Abstract:",
                abstract or "Abstract unavailable",
                "Authors:",
                format_bullets(author_lines),
                "Institutions:",
                format_bullets(sorted(set(name for name in institution_names if name))),
                "Topics:",
                format_bullets(topic_names),
                "Referenced Works:",
                format_bullets(referenced_works, empty_text="None listed in the slice"),
            ]
        )
        write_text(source_path, content)
        generated_rows.append(source_row(filename, work_id, title, dataset.record_type, source_path))

    acquisition = {
        "api": OPENALEX_WORKS_URL,
        "dataset": dataset.folder_slug,
        "filters": {
            "has_abstract": True,
            "language": "en",
            "publication_year": "2023-2025",
            "search": "retrieval augmented generation",
            "sort": "cited_by_count:desc",
        },
        "generated_source_count": len(generated_rows),
        "manual_fallback": [
            f"Save a paginated OpenAlex API response to '{raw_path}'.",
            "Then rerun `prepare --dataset openalex-rag --skip-download`.",
        ],
        "raw_files": [str(raw_path)],
        "retrieved_at_utc": utc_now(),
        "target_size": target_size,
    }
    write_source_manifests(dataset, paths, generated_rows, acquisition, notebook_id=notebook_id)


def open_tsv_gz(path: Path) -> Iterable[dict[str, str]]:
    """Yield rows from a gzipped TSV file."""
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        yield from reader

def prepare_imdb(
    paths: dict[str, Path],
    dataset: DatasetConfig,
    *,
    overwrite: bool,
    skip_download: bool,
    target_size: int,
    notebook_id: str | None,
) -> None:
    """Download and transform the IMDb sci-fi slice."""
    raw_paths = {name: paths["raw"] / name for name in IMDB_DOWNLOADS}
    if not skip_download:
        for name, url in IMDB_DOWNLOADS.items():
            download_file(url, raw_paths[name], overwrite=overwrite)
    else:
        missing = [str(path) for path in raw_paths.values() if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "IMDb raw files are missing. Expected: "
                + ", ".join(missing)
                + ". Place the official TSV.GZ files there or rerun without --skip-download."
            )

    ratings_by_title: dict[str, dict[str, str]] = {}
    for row in open_tsv_gz(raw_paths["title.ratings.tsv.gz"]):
        ratings_by_title[row["tconst"]] = row

    candidate_titles: list[dict[str, str]] = []
    for row in open_tsv_gz(raw_paths["title.basics.tsv.gz"]):
        genres = row.get("genres", "")
        year = row.get("startYear", "\\N")
        if row.get("titleType") != "movie":
            continue
        if "Sci-Fi" not in genres.split(","):
            continue
        if year == "\\N":
            continue
        year_int = int(year)
        if year_int < 2000 or year_int > 2024:
            continue
        ratings = ratings_by_title.get(row["tconst"])
        if not ratings:
            continue
        row = dict(row)
        row["averageRating"] = ratings.get("averageRating", "\\N")
        row["numVotes"] = ratings.get("numVotes", "0")
        candidate_titles.append(row)

    candidate_titles.sort(key=lambda item: int(item.get("numVotes", "0")), reverse=True)
    selected_titles = candidate_titles[:target_size]
    selected_title_ids = {row["tconst"] for row in selected_titles}

    principals_by_title: dict[str, list[dict[str, str]]] = {title_id: [] for title_id in selected_title_ids}
    name_ids: set[str] = set()
    for row in open_tsv_gz(raw_paths["title.principals.tsv.gz"]):
        title_id = row.get("tconst")
        if title_id not in selected_title_ids:
            continue
        principals_by_title[title_id].append(row)
        if row.get("nconst"):
            name_ids.add(row["nconst"])

    names_by_id: dict[str, dict[str, str]] = {}
    for row in open_tsv_gz(raw_paths["name.basics.tsv.gz"]):
        person_id = row.get("nconst")
        if person_id in name_ids:
            names_by_id[person_id] = row

    generated_rows: list[dict[str, Any]] = []
    for item in selected_titles:
        title_id = item["tconst"]
        principals = principals_by_title.get(title_id, [])
        directors: list[str] = []
        writers: list[str] = []
        cast: list[str] = []
        known_for: list[str] = []
        for principal in principals:
            person = names_by_id.get(principal.get("nconst", ""))
            person_name = (person or {}).get("primaryName") or principal.get("nconst") or "Unknown person"
            category = principal.get("category", "")
            if category == "director":
                directors.append(person_name)
            elif category == "writer":
                writers.append(person_name)
            elif category in {"actor", "actress", "self"} and len(cast) < 10:
                cast.append(person_name)
            if person and person.get("knownForTitles") and person["knownForTitles"] != "\\N":
                known_for.append(f"{person_name}: {person['knownForTitles']}")

        title = item.get("primaryTitle") or "Untitled movie"
        filename = f"title_{sanitize_filename(title_id)}.txt"
        source_path = paths["sources"] / filename
        content = "\n".join(
            [
                f"Dataset: {dataset.notebook_title}",
                "Record Type: movie",
                f"External ID: {title_id}",
                f"Source URL: https://www.imdb.com/title/{title_id}/",
                f"Generated At: {utc_now()}",
                "---",
                f"Primary Title: {title}",
                f"Original Title: {item.get('originalTitle') or title}",
                f"Release Year: {item.get('startYear', 'Unknown')}",
                f"Genres: {item.get('genres', 'Unknown')}",
                f"Average Rating: {item.get('averageRating', 'Unknown')}",
                f"Vote Count: {item.get('numVotes', 'Unknown')}",
                "Directors:",
                format_bullets(directors),
                "Writers:",
                format_bullets(writers),
                "Principal Cast:",
                format_bullets(cast),
                "Known For Title IDs:",
                format_bullets(known_for, empty_text="None captured for principals"),
            ]
        )
        write_text(source_path, content)
        generated_rows.append(source_row(filename, title_id, title, dataset.record_type, source_path))

    acquisition = {
        "dataset": dataset.folder_slug,
        "downloads": IMDB_DOWNLOADS,
        "filters": {
            "genres": "Sci-Fi",
            "start_year": "2000-2024",
            "title_type": "movie",
            "sort": "numVotes desc",
        },
        "generated_source_count": len(generated_rows),
        "manual_fallback": [
            f"Download the official IMDb TSV.GZ files into '{paths['raw']}'.",
            "Then rerun `prepare --dataset imdb-scifi --skip-download`.",
        ],
        "raw_files": [str(path) for path in raw_paths.values()],
        "retrieved_at_utc": utc_now(),
        "target_size": target_size,
    }
    write_source_manifests(dataset, paths, generated_rows, acquisition, notebook_id=notebook_id)


def run_open_targets_query(query: str, variables: dict[str, Any]) -> Any:
    """Execute a GraphQL query against Open Targets."""
    return post_json(OPEN_TARGETS_GRAPHQL_URL, {"query": query, "variables": variables}, timeout=DEFAULT_TIMEOUT)


def resolve_alzheimers_disease() -> tuple[str, dict[str, Any]]:
    """Resolve Alzheimer's disease to an Open Targets disease identifier."""
    query = """
    query ResolveDisease($queryString: String!) {
      search(queryString: $queryString, entityNames: [\"disease\"], page: { index: 0, size: 10 }) {
        hits {
          id
          name
          entity
        }
      }
    }
    """
    response = run_open_targets_query(query, {"queryString": "Alzheimer's disease"})
    hits = (((response or {}).get("data") or {}).get("search") or {}).get("hits") or []
    for hit in hits:
        disease_id = hit.get("id")
        if disease_id:
            return disease_id, response
    fallback_id = "EFO_0000249"
    return fallback_id, response


def fetch_open_targets_associations(disease_id: str, target_size: int) -> dict[str, Any]:
    """Fetch the top disease-target associations for a disease id."""
    query = """
    query DiseaseTargets($diseaseId: String!, $size: Int!) {
      disease(efoId: $diseaseId) {
        id
        name
        associatedTargets(page: { index: 0, size: $size }) {
          count
          rows {
            score
            datatypeScores {
              id
              score
            }
            target {
              id
              approvedSymbol
              approvedName
              biotype
            }
          }
        }
      }
    }
    """
    return run_open_targets_query(query, {"diseaseId": disease_id, "size": target_size})

def prepare_open_targets(
    paths: dict[str, Path],
    dataset: DatasetConfig,
    *,
    skip_download: bool,
    target_size: int,
    notebook_id: str | None,
) -> None:
    """Download and transform the Open Targets Alzheimer's slice."""
    search_raw = paths["raw"] / "disease_search.json"
    association_raw = paths["raw"] / "associations.json"
    if not skip_download:
        disease_id, search_response = resolve_alzheimers_disease()
        write_json(search_raw, search_response)
        association_response = fetch_open_targets_associations(disease_id, target_size)
        write_json(association_raw, association_response)
        raw_data = association_response
    else:
        raw_data = load_json_if_present(association_raw)
        if raw_data is None:
            raise FileNotFoundError(
                f"Open Targets raw file not found at {association_raw}. "
                "Place an association export there or rerun without --skip-download."
            )
        search_response = load_json_if_present(search_raw) or {}

    disease_block = (((raw_data or {}).get("data") or {}).get("disease")) or {}
    disease_id = disease_block.get("id") or "EFO_0000249"
    disease_name = disease_block.get("name") or "Alzheimer's disease"
    rows = ((disease_block.get("associatedTargets") or {}).get("rows")) or []

    generated_rows: list[dict[str, Any]] = []
    for row in rows[:target_size]:
        target = row.get("target") or {}
        target_id = target.get("id")
        if not target_id:
            continue
        symbol = target.get("approvedSymbol") or target_id
        filename = f"association_{sanitize_filename(disease_id)}_{sanitize_filename(target_id)}.txt"
        source_path = paths["sources"] / filename
        datatype_scores = row.get("datatypeScores") or []
        evidence_lines = [f"{score.get('id', 'unknown')}: {score.get('score', 'unknown')}" for score in datatype_scores]
        content = "\n".join(
            [
                f"Dataset: {dataset.notebook_title}",
                "Record Type: target_disease_association",
                f"External ID: {disease_id}::{target_id}",
                "Source URL: https://platform.opentargets.org/",
                f"Generated At: {utc_now()}",
                "---",
                f"Disease ID: {disease_id}",
                f"Disease Name: {disease_name}",
                f"Target ID: {target_id}",
                f"Target Symbol: {symbol}",
                f"Target Name: {target.get('approvedName') or 'Unknown target name'}",
                f"Target Biotype: {target.get('biotype') or 'Unknown'}",
                f"Association Score: {row.get('score', 'Unknown')}",
                "Evidence Type Scores:",
                format_bullets(evidence_lines),
                "Linked Drugs:",
                "- Not captured in the lightweight API slice",
                "Pathways Or Mechanism Notes:",
                "- Not captured in the lightweight API slice",
                "Provenance:",
                format_bullets(
                    [
                        "Source system: Open Targets Platform GraphQL API",
                        f"Association row keys: {join_text(sorted(row.keys()), empty_text='Unknown')}",
                    ]
                ),
            ]
        )
        write_text(source_path, content)
        generated_rows.append(
            source_row(
                filename,
                f"{disease_id}::{target_id}",
                f"{disease_name} -> {symbol}",
                dataset.record_type,
                source_path,
            )
        )

    acquisition = {
        "api": OPEN_TARGETS_GRAPHQL_URL,
        "dataset": dataset.folder_slug,
        "disease_id": disease_id,
        "disease_name": disease_name,
        "generated_source_count": len(generated_rows),
        "manual_fallback": [
            f"Place an Open Targets association export at '{association_raw}'.",
            "Optional: also save the disease search response to 'disease_search.json'.",
            "Then rerun `prepare --dataset opentargets-alzheimers --skip-download`.",
        ],
        "raw_files": [str(search_raw), str(association_raw)],
        "resolution_preview": search_response,
        "retrieved_at_utc": utc_now(),
        "target_size": target_size,
    }
    write_source_manifests(dataset, paths, generated_rows, acquisition, notebook_id=notebook_id)


def init_dataset(dataset: DatasetConfig) -> None:
    """Create folders and metadata for a dataset without downloading anything."""
    paths = ensure_dataset_layout(dataset)
    update_dataset_metadata(paths, dataset)
    placeholder = {
        "dataset": dataset.folder_slug,
        "generated_at_utc": utc_now(),
        "manual_fallback": [
            f"Put raw dataset files into '{paths['raw']}'.",
            "Run the `prepare` command after raw files are present.",
        ],
        "raw_files": [],
        "target_size": dataset.target_size,
    }
    write_json(paths["manifest"] / "acquisition.json", placeholder)
    write_json(
        paths["manifest"] / "sync_workflow.json",
        {
            "create_command": build_sync_command(dataset, paths, mode="create"),
            "dataset": dataset.folder_slug,
            "dataset_dir": str(paths["sources"]),
            "export_dir": str(default_export_dir(dataset)),
            "generated_at_utc": utc_now(),
            "neo4j_database": dataset.neo4j_database,
            "sync_script": str(SYNC_NOTEBOOK_GRAPH),
            "update_command": build_sync_command(dataset, paths, mode="update"),
        },
    )


def prepare_dataset(
    dataset: DatasetConfig,
    *,
    overwrite: bool,
    skip_download: bool,
    target_size: int | None,
    notebook_id: str | None,
) -> None:
    """Prepare a dataset end-to-end."""
    paths = ensure_dataset_layout(dataset)
    effective_target_size = target_size or dataset.target_size
    if overwrite:
        for file_path in paths["sources"].glob("*.txt"):
            file_path.unlink()

    if dataset.folder_slug == "openalex-rag":
        prepare_openalex(paths, dataset, skip_download=skip_download, target_size=effective_target_size, notebook_id=notebook_id)
    elif dataset.folder_slug == "imdb-scifi":
        prepare_imdb(
            paths,
            dataset,
            overwrite=overwrite,
            skip_download=skip_download,
            target_size=effective_target_size,
            notebook_id=notebook_id,
        )
    elif dataset.folder_slug == "opentargets-alzheimers":
        prepare_open_targets(paths, dataset, skip_download=skip_download, target_size=effective_target_size, notebook_id=notebook_id)
    else:
        raise ValueError(f"Unsupported dataset: {dataset.folder_slug}")


def register_notebook_id(dataset: DatasetConfig, notebook_id: str) -> None:
    """Record a NotebookLM notebook id into the dataset manifests."""
    paths = ensure_dataset_layout(dataset)
    update_dataset_metadata(paths, dataset, notebook_id=notebook_id)
    import_manifest_path = paths["manifest"] / "notebooklm_import.json"
    import_manifest = load_json_if_present(import_manifest_path) or {}
    import_manifest["dataset"] = dataset.folder_slug
    import_manifest["generated_at_utc"] = utc_now()
    import_manifest["notebook_id"] = notebook_id
    import_manifest["notebook_title"] = dataset.notebook_title
    import_manifest.setdefault("source_dir", str(paths["sources"]))
    import_manifest["create_command"] = build_sync_command(dataset, paths, mode="create")
    import_manifest["update_command"] = build_sync_command(dataset, paths, mode="update", notebook_id=notebook_id)
    write_json(import_manifest_path, import_manifest)

    sync_manifest_path = paths["manifest"] / "sync_workflow.json"
    sync_manifest = load_json_if_present(sync_manifest_path) or {}
    sync_manifest["dataset"] = dataset.folder_slug
    sync_manifest["dataset_dir"] = str(paths["sources"])
    sync_manifest["export_dir"] = str(default_export_dir(dataset))
    sync_manifest["generated_at_utc"] = utc_now()
    sync_manifest["neo4j_database"] = dataset.neo4j_database
    sync_manifest["sync_script"] = str(SYNC_NOTEBOOK_GRAPH)
    sync_manifest["create_command"] = build_sync_command(dataset, paths, mode="create")
    sync_manifest["update_command"] = build_sync_command(dataset, paths, mode="update", notebook_id=notebook_id)
    write_json(sync_manifest_path, sync_manifest)


def print_status(dataset: DatasetConfig) -> None:
    """Print a short human-readable status summary for one dataset."""
    paths = ensure_dataset_layout(dataset)
    metadata = load_json_if_present(paths["manifest"] / "dataset_config.json") or {}
    source_files = sorted(paths["sources"].glob("*.txt"))
    print(f"[{dataset.folder_slug}]")
    print(f"  notebook_title: {dataset.notebook_title}")
    print(f"  notebook_id: {metadata.get('notebook_id') or 'unset'}")
    print(f"  neo4j_database: {dataset.neo4j_database}")
    print(f"  source_count: {len(source_files)}")
    print(f"  raw_dir: {paths['raw']}")
    print(f"  sources_dir: {paths['sources']}")
    print(f"  manifest_dir: {paths['manifest']}")

def parse_args(argv: list[str]) -> argparse.Namespace:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Create dataset folders and metadata.")
    init_parser.add_argument("--dataset", choices=["all", *DATASETS.keys()], default="all")

    prepare_parser = subparsers.add_parser("prepare", help="Download raw data and build canonical sources.")
    prepare_parser.add_argument("--dataset", choices=["all", *DATASETS.keys()], default="all")
    prepare_parser.add_argument("--skip-download", action="store_true", help="Use existing raw files only.")
    prepare_parser.add_argument("--overwrite", action="store_true", help="Regenerate source files from scratch.")
    prepare_parser.add_argument("--target-size", type=int, help="Override the dataset default target size.")
    prepare_parser.add_argument("--notebook-id", help="Persist a NotebookLM notebook id into manifests.")

    notebook_parser = subparsers.add_parser("register-notebook", help="Save a NotebookLM notebook id.")
    notebook_parser.add_argument("--dataset", choices=list(DATASETS.keys()), required=True)
    notebook_parser.add_argument("--notebook-id", required=True)

    status_parser = subparsers.add_parser("status", help="Show local benchmark dataset status.")
    status_parser.add_argument("--dataset", choices=["all", *DATASETS.keys()], default="all")

    return parser.parse_args(argv)


def iter_datasets(dataset_name: str) -> list[DatasetConfig]:
    """Resolve `all` into the ordered dataset list."""
    if dataset_name == "all":
        return [DATASETS[name] for name in DATASETS]
    return [DATASETS[dataset_name]]


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    args = parse_args(argv or sys.argv[1:])
    try:
        if args.command == "init":
            for dataset in iter_datasets(args.dataset):
                init_dataset(dataset)
                print(f"Initialized {dataset.folder_slug}")
            return 0

        if args.command == "prepare":
            for dataset in iter_datasets(args.dataset):
                prepare_dataset(
                    dataset,
                    overwrite=args.overwrite,
                    skip_download=args.skip_download,
                    target_size=args.target_size,
                    notebook_id=args.notebook_id,
                )
                print(f"Prepared {dataset.folder_slug}")
            return 0

        if args.command == "register-notebook":
            dataset = DATASETS[args.dataset]
            register_notebook_id(dataset, args.notebook_id)
            print(f"Registered NotebookLM id for {dataset.folder_slug}")
            return 0

        if args.command == "status":
            for dataset in iter_datasets(args.dataset):
                print_status(dataset)
            return 0
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except error.URLError as exc:
        print(
            "ERROR: network access failed while retrieving dataset files. "
            "Use the raw/ folder manual fallback documented in the manifest.",
            file=sys.stderr,
        )
        print(str(exc), file=sys.stderr)
        return 1
    except Exception as exc:  # pragma: no cover - CLI guardrail
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print("ERROR: unhandled command", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
