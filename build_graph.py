#!/usr/bin/env python3
"""
Idempotent local batch upload, extract, and post-process pipeline.

This entrypoint reads staged NotebookLM-exported `.txt` files directly from the
filesystem and drives graph extraction in-process, without the FastAPI backend.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path

from dataset_registry import default_sources_dir, load_dataset_entry
from graph_builder_runtime import GraphBuilderAPI
from llm_routing import resolve_graph_build_embedding

DEFAULT_TOKEN_CHUNK_SIZE = 2000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_CHUNKS_TO_COMBINE = 1
TERMINAL_STATUSES = {"Completed", "Failed", "Cancelled"}
DEFAULT_NEO4J_URI = "bolt://127.0.0.1:7687"
DEFAULT_NEO4J_USER = "neo4j"
DEFAULT_NEO4J_PASSWORD = "password123"
DEFAULT_NEO4J_DATABASE = "neo4j"
DEFAULT_SOURCES_DIR = "data/notebooklm_exports/default/sources"


class C:
    """ANSI colour helpers (disabled on non-TTY or Windows without VT)."""

    _ok = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
    RESET = "\033[0m" if _ok else ""
    BOLD = "\033[1m" if _ok else ""
    GREEN = "\033[32m" if _ok else ""
    YELLOW = "\033[33m" if _ok else ""
    RED = "\033[31m" if _ok else ""
    CYAN = "\033[36m" if _ok else ""
    DIM = "\033[2m" if _ok else ""


def log(msg: str, level: str = "INFO") -> None:
    ts = time.strftime("%H:%M:%S")
    colour = {
        "INFO": C.CYAN,
        "OK": C.GREEN,
        "WARN": C.YELLOW,
        "ERROR": C.RED,
        "SKIP": C.DIM,
    }.get(level, "")
    print(f"{C.DIM}[{ts}]{C.RESET} {colour}{level:5s}{C.RESET}  {msg}")


def _existing_source_map(api: GraphBuilderAPI) -> dict[str, dict[str, str | None]]:
    source_map: dict[str, dict[str, str | None]] = {}
    for src in api.sources_list():
        name = src.get("fileName") or src.get("name") or src.get("file_name") or ""
        if not name:
            continue
        retry_condition = src.get("retry_condition") or src.get("retryCondition")
        source_map[str(name)] = {
            "status": str(src.get("status") or src.get("Status") or ""),
            "retry_condition": str(retry_condition) if retry_condition else None,
        }
    return source_map


def phase_upload(api: GraphBuilderAPI, sources_dir: Path, model: str, min_file_size: int) -> tuple[int, int, int]:
    log(f"{C.BOLD}Phase 1: Register Sources{C.RESET}", "INFO")
    local_files = sorted(f for f in sources_dir.glob("*.txt") if f.stat().st_size >= min_file_size)
    if not local_files:
        log(f"No .txt files found in {sources_dir}", "ERROR")
        return 0, 0, 0

    source_map = _existing_source_map(api)
    completed_names = {name for name, info in source_map.items() if info.get("status") == "Completed"}
    log(f"Found {len(local_files)} source files (>={min_file_size} bytes)")
    log(f"Completed (skip): {len(completed_names)}, will register: {len(local_files) - len(completed_names)}")

    uploaded = 0
    skipped = 0
    failed = 0
    for index, file_path in enumerate(local_files, 1):
        prefix = f"[{index}/{len(local_files)}]"
        if file_path.name in completed_names:
            log(f"{prefix} {file_path.name} - Completed, skip", "SKIP")
            skipped += 1
            continue
        resp = api.upload_file(file_path, model)
        if resp.get("status") == "Success":
            log(f"{prefix} {file_path.name} - registered", "OK")
            uploaded += 1
        else:
            log(f"{prefix} {file_path.name} - register failed: {resp.get('message', 'unknown')}", "ERROR")
            failed += 1
    log(f"Registration complete: {uploaded} registered, {skipped} skipped, {failed} failed")
    return uploaded, skipped, failed


def _wait_for_extraction(api: GraphBuilderAPI, file_name: str, poll_interval: int, max_wait: int = 3600) -> str:
    start = time.time()
    while time.time() - start < max_wait:
        info = api.document_status(file_name)
        if info:
            status = str(info.get("status") or "Unknown")
            processed = info.get("processed_chunk", "?")
            total = info.get("total_chunks", "?")
            if status in TERMINAL_STATUSES:
                return status
            log(f"  -> {file_name}: {status} ({processed}/{total} chunks)", "INFO")
        time.sleep(poll_interval)
    return "Timeout"


def _extract_response_status(resp: dict, file_name: str, poll_interval: int, api: GraphBuilderAPI) -> str:
    if resp.get("status") == "Failed":
        msg = str(resp.get("message") or resp.get("error") or "Unknown")[:200]
        return f"Failed: {msg}"
    data = resp.get("data", {})
    if isinstance(data, dict):
        doc_status = data.get("status", "")
        if doc_status == "Completed":
            nodes = data.get("nodeCount", "?")
            rels = data.get("relationshipCount", "?")
            return f"Completed ({nodes} nodes, {rels} rels)"
        if doc_status in {"Failed", "Cancelled"}:
            err = str(data.get("errorMessage") or data.get("message") or "")[:200]
            return f"{doc_status}: {err}"
    return _wait_for_extraction(api, file_name, poll_interval)


def _upload_and_extract_one(
    api: GraphBuilderAPI,
    file_path: Path,
    model: str,
    embedding_provider: str,
    embedding_model: str,
    poll_interval: int,
    token_chunk_size: int,
    chunk_overlap: int,
    chunks_to_combine: int,
    retry_condition: str | None = None,
) -> tuple[str, str]:
    up_resp = api.upload_file(file_path, model)
    if up_resp.get("status") != "Success":
        return file_path.name, f"Upload failed: {up_resp.get('message', 'upload failed')[:200]}"
    resp = api.extract(
        file_path.name,
        model,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        retry_condition=retry_condition,
        token_chunk_size=token_chunk_size,
        chunk_overlap=chunk_overlap,
        chunks_to_combine=chunks_to_combine,
    )
    return file_path.name, _extract_response_status(resp, file_path.name, poll_interval, api)


def _extract_existing_one(
    api: GraphBuilderAPI,
    file_name: str,
    model: str,
    embedding_provider: str,
    embedding_model: str,
    poll_interval: int,
    token_chunk_size: int,
    chunk_overlap: int,
    chunks_to_combine: int,
    retry_condition: str | None = None,
) -> tuple[str, str]:
    resp = api.extract(
        file_name,
        model,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        retry_condition=retry_condition,
        token_chunk_size=token_chunk_size,
        chunk_overlap=chunk_overlap,
        chunks_to_combine=chunks_to_combine,
    )
    return file_name, _extract_response_status(resp, file_name, poll_interval, api)


def _run_parallel(items, fn, parallel: int):
    if parallel <= 1:
        for item in items:
            yield fn(item)
        return
    with ThreadPoolExecutor(max_workers=parallel) as pool:
        futures = {pool.submit(fn, item): item for item in items}
        for future in as_completed(futures):
            yield future.result()


@contextmanager
def _temporary_embedding_dimension_override(dimension: int | None):
    original = os.environ.get("EMBEDDING_DIMENSION_OVERRIDE")
    try:
        if dimension is None:
            os.environ.pop("EMBEDDING_DIMENSION_OVERRIDE", None)
        else:
            os.environ["EMBEDDING_DIMENSION_OVERRIDE"] = str(dimension)
        yield
    finally:
        if original is None:
            os.environ.pop("EMBEDDING_DIMENSION_OVERRIDE", None)
        else:
            os.environ["EMBEDDING_DIMENSION_OVERRIDE"] = original


def _summarize_results(results: list[tuple[str, str]], skipped: int) -> tuple[int, int, int]:
    completed = sum(1 for _, status in results if "Completed" in status)
    failed = len(results) - completed
    return completed, skipped, failed


def phase_upload_and_extract(
    api: GraphBuilderAPI,
    sources_dir: Path,
    model: str,
    embedding_provider: str,
    embedding_model: str,
    min_file_size: int,
    parallel: int,
    poll_interval: int,
    token_chunk_size: int,
    chunk_overlap: int,
    chunks_to_combine: int,
) -> tuple[int, int, int]:
    log(f"{C.BOLD}Phase 1+2: Register & Extract{C.RESET}", "INFO")
    local_files = sorted(f for f in sources_dir.glob("*.txt") if f.stat().st_size >= min_file_size)
    if not local_files:
        log(f"No .txt files found in {sources_dir}", "ERROR")
        return 0, 0, 0
    source_map = _existing_source_map(api)
    completed_names = {name for name, info in source_map.items() if info.get("status") == "Completed"}
    to_process = [file_path for file_path in local_files if file_path.name not in completed_names]
    skipped = len(local_files) - len(to_process)
    log(f"Found {len(local_files)} source files (>={min_file_size} bytes)")
    log(f"Completed (skip): {skipped}, will process: {len(to_process)}")
    if not to_process:
        log("All sources already Completed - nothing to do!", "OK")
        return 0, skipped, 0

    def process_one(file_path: Path) -> tuple[str, str]:
        retry_condition = source_map.get(file_path.name, {}).get("retry_condition")
        return _upload_and_extract_one(
            api,
            file_path,
            model,
            embedding_provider,
            embedding_model,
            poll_interval,
            token_chunk_size,
            chunk_overlap,
            chunks_to_combine,
            retry_condition=retry_condition,
        )

    results = list(_run_parallel(to_process, process_one, parallel))
    for index, (name, status) in enumerate(results, 1):
        level = "OK" if "Completed" in status else "ERROR"
        log(f"[{index}/{len(results)}] {name} - {status}", level)
    completed, skipped_count, failed = _summarize_results(results, skipped)
    log(f"Done: {completed} completed, {skipped_count} skipped, {failed} failed")
    return completed, skipped_count, failed


def phase_extract_existing(
    api: GraphBuilderAPI,
    sources_dir: Path,
    model: str,
    embedding_provider: str,
    embedding_model: str,
    min_file_size: int,
    parallel: int,
    poll_interval: int,
    token_chunk_size: int,
    chunk_overlap: int,
    chunks_to_combine: int,
) -> tuple[int, int, int]:
    log(f"{C.BOLD}Phase 2: Extract Existing Sources{C.RESET}", "INFO")
    local_files = sorted(f for f in sources_dir.glob("*.txt") if f.stat().st_size >= min_file_size)
    if not local_files:
        log(f"No .txt files found in {sources_dir}", "ERROR")
        return 0, 0, 0
    source_map = _existing_source_map(api)
    completed_names = {name for name, info in source_map.items() if info.get("status") == "Completed"}
    to_process = [file_path.name for file_path in local_files if file_path.name not in completed_names]
    skipped = len(local_files) - len(to_process)
    if not to_process:
        log("All sources already Completed - nothing to do!", "OK")
        return 0, skipped, 0

    def process_one(file_name: str) -> tuple[str, str]:
        retry_condition = source_map.get(file_name, {}).get("retry_condition")
        return _extract_existing_one(
            api,
            file_name,
            model,
            embedding_provider,
            embedding_model,
            poll_interval,
            token_chunk_size,
            chunk_overlap,
            chunks_to_combine,
            retry_condition=retry_condition,
        )

    results = list(_run_parallel(to_process, process_one, parallel))
    for index, (name, status) in enumerate(results, 1):
        level = "OK" if "Completed" in status else "ERROR"
        log(f"[{index}/{len(results)}] {name} - {status}", level)
    completed, skipped_count, failed = _summarize_results(results, skipped)
    log(f"Extraction complete: {completed} completed, {skipped_count} skipped, {failed} failed")
    return completed, skipped_count, failed


def phase_postprocess(api: GraphBuilderAPI, embedding_provider: str, embedding_model: str) -> bool:
    log(f"{C.BOLD}Phase 3: Post-Processing{C.RESET}", "INFO")
    tasks = ["enable_hybrid_search_and_fulltext_search_in_bloom"]
    log(f"Running tasks: {', '.join(tasks)}")
    try:
        resp = api.post_processing(tasks, embedding_provider, embedding_model)
        if resp.get("status") == "Success":
            log("Post-processing completed successfully", "OK")
            return True
        log(f"Post-processing returned: {resp.get('status', 'Unknown')} - {resp.get('message', 'Unknown error')}", "ERROR")
        return False
    except Exception as exc:
        log(f"Post-processing error: {exc}", "ERROR")
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Idempotent local register, extract, and post-process pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python build_graph.py                           # run all phases
  python build_graph.py --parallel 3              # 3 concurrent extractions
  python build_graph.py --skip-upload             # extract + post-process only
  python build_graph.py --skip-upload --skip-extract  # post-process only
        """,
    )
    parser.add_argument("--dataset-key", help="Dataset key from benchmark_dataset_registry.json")
    parser.add_argument("--registry-path", help="Path to benchmark dataset registry JSON")
    parser.add_argument("--neo4j-uri", default=DEFAULT_NEO4J_URI, help="Neo4j Bolt URI")
    parser.add_argument("--neo4j-user", default=DEFAULT_NEO4J_USER, help="Neo4j username")
    parser.add_argument("--neo4j-password", default=DEFAULT_NEO4J_PASSWORD, help="Neo4j password")
    parser.add_argument("--neo4j-database", default=DEFAULT_NEO4J_DATABASE, help="Neo4j database name")
    parser.add_argument("--model", default="google_flash", help="LLM model name for extraction")
    parser.add_argument("--sources-dir", default=DEFAULT_SOURCES_DIR, help="Directory containing exported .txt source files")
    parser.add_argument("--parallel", type=int, default=1, help="Concurrent extractions")
    parser.add_argument("--poll-interval", type=int, default=15, help="Seconds between extraction status polls")
    parser.add_argument("--min-file-size", type=int, default=10, help="Skip files smaller than N bytes")
    parser.add_argument("--token-chunk-size", type=int, default=DEFAULT_TOKEN_CHUNK_SIZE, help=f"Token chunk size for extraction (default: {DEFAULT_TOKEN_CHUNK_SIZE})")
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP, help=f"Chunk overlap for extraction (default: {DEFAULT_CHUNK_OVERLAP})")
    parser.add_argument("--chunks-to-combine", type=int, default=DEFAULT_CHUNKS_TO_COMBINE, help=f"Number of chunks to combine during extraction (default: {DEFAULT_CHUNKS_TO_COMBINE})")
    parser.add_argument("--embedding-provider", default=None, help="Embedding provider override")
    parser.add_argument("--embedding-model", default=None, help="Embedding model override")
    parser.add_argument("--llm-routing-config", default=None, help="Optional JSON config for role-based LLM routing")
    parser.add_argument("--skip-upload", action="store_true", help="Skip source registration")
    parser.add_argument("--skip-extract", action="store_true", help="Skip extraction")
    parser.add_argument("--skip-postprocess", action="store_true", help="Skip the post-processing phase")
    args = parser.parse_args()
    if args.dataset_key:
        try:
            entry = load_dataset_entry(args.dataset_key, args.registry_path)
        except ValueError as exc:
            parser.error(str(exc))
        if args.neo4j_uri == DEFAULT_NEO4J_URI:
            args.neo4j_uri = entry.neo4j.uri
        if args.neo4j_user == DEFAULT_NEO4J_USER:
            args.neo4j_user = entry.neo4j.username
        if args.neo4j_password == DEFAULT_NEO4J_PASSWORD:
            args.neo4j_password = entry.neo4j.password
        if args.neo4j_database == DEFAULT_NEO4J_DATABASE:
            args.neo4j_database = entry.neo4j.database
        if args.sources_dir == DEFAULT_SOURCES_DIR:
            args.sources_dir = str(default_sources_dir(args.dataset_key))
    return args


def main() -> None:
    args = parse_args()
    sources_dir = Path(args.sources_dir).resolve()
    graph_build_embedding = resolve_graph_build_embedding(
        config_path=args.llm_routing_config,
        embedding_provider=args.embedding_provider,
        embedding_model=args.embedding_model,
        default_provider="sentence-transformer",
        default_model="all-MiniLM-L6-v2",
    )

    print(f"\n{C.BOLD}Local Graph Builder Pipeline{C.RESET}\n")
    log(f"Neo4j:       {args.neo4j_uri}")
    log(f"Model:       {args.model}")
    log(f"Sources:     {sources_dir}")
    log(f"Parallel:    {args.parallel}")
    log(f"Embedding:   {graph_build_embedding.client}/{graph_build_embedding.model}")
    log(f"Poll every:  {args.poll_interval}s")
    log(f"Chunking:    size={args.token_chunk_size}, overlap={args.chunk_overlap}, combine={args.chunks_to_combine}")
    print()

    api = GraphBuilderAPI(
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_password=args.neo4j_password,
        neo4j_database=args.neo4j_database,
        sources_dir=sources_dir,
    )

    log("Checking local graph runtime...", "INFO")
    if not api.health_check():
        log("Neo4j/runtime check failed", "ERROR")
        sys.exit(1)
    log("Local runtime is healthy", "OK")

    with _temporary_embedding_dimension_override(graph_build_embedding.dimension):
        log("Connecting to Neo4j...", "INFO")
        resp = api.connect(graph_build_embedding.client, graph_build_embedding.model)
        if resp.get("status") != "Success":
            log(f"Neo4j connection failed: {resp.get('message', 'unknown')}", "ERROR")
            sys.exit(1)
        log("Neo4j connection established", "OK")

        total_start = time.time()
        results: dict[str, dict[str, int | bool]] = {}

        if not sources_dir.exists():
            log(f"Sources directory not found: {sources_dir}", "ERROR")
            sys.exit(1)

        if not args.skip_upload and not args.skip_extract:
            c, s, f = phase_upload_and_extract(
                api,
                sources_dir,
                args.model,
                graph_build_embedding.client,
                graph_build_embedding.model,
                args.min_file_size,
                args.parallel,
                args.poll_interval,
                args.token_chunk_size,
                args.chunk_overlap,
                args.chunks_to_combine,
            )
            results["upload_and_extract"] = {"completed": c, "skipped": s, "failed": f}
        elif not args.skip_upload and args.skip_extract:
            u, s, f = phase_upload(api, sources_dir, args.model, args.min_file_size)
            results["upload"] = {"uploaded": u, "skipped": s, "failed": f}
        elif args.skip_upload and not args.skip_extract:
            c, s, f = phase_extract_existing(
                api,
                sources_dir,
                args.model,
                graph_build_embedding.client,
                graph_build_embedding.model,
                args.min_file_size,
                args.parallel,
                args.poll_interval,
                args.token_chunk_size,
                args.chunk_overlap,
                args.chunks_to_combine,
            )
            results["extract"] = {"completed": c, "skipped": s, "failed": f}
        else:
            log("Skipping upload and extraction phases", "SKIP")

        if not args.skip_postprocess:
            ok = phase_postprocess(api, graph_build_embedding.client, graph_build_embedding.model)
            results["postprocess"] = {"success": ok}
        else:
            log("Skipping post-processing phase (--skip-postprocess)", "SKIP")

    total_time = time.time() - total_start
    print(f"{C.BOLD}{'=' * 56}{C.RESET}")
    log(f"Pipeline finished in {total_time:.1f}s", "OK")
    for phase, data in results.items():
        log(f"  {phase}: {data}")
    print()


if __name__ == "__main__":
    main()
