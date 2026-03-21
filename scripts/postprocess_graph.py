#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from build_graph import GraphBuilderAPI

DEFAULT_NEO4J_URI = "bolt://127.0.0.1:7687"
DEFAULT_NEO4J_USER = "neo4j"
DEFAULT_NEO4J_PASSWORD = "password123"
DEFAULT_NEO4J_DATABASE = "neo4j"
POSTPROCESS_TASKS = ["enable_hybrid_search_and_fulltext_search_in_bloom"]


class WorkflowError(RuntimeError):
    pass


@dataclass(frozen=True)
class ConsolidationConfig:
    max_iterations: int
    run_dir: str | None
    codex_bin: str
    dry_run: bool
    resume: bool
    required_consecutive_passes: int


class CommandRunner:
    def run(
        self,
        args: list[str],
        *,
        cwd: Path,
        env: dict[str, str],
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(args, cwd=str(cwd), env=env, text=True, check=False)


def log(message: str) -> None:
    print(f"[postprocess_graph] {message}")


def effective_required_consecutive_passes(config: ConsolidationConfig) -> int:
    if config.required_consecutive_passes > 0:
        return config.required_consecutive_passes
    return 2 if config.dry_run else 1


def resolve_codex_executable(codex_bin: str) -> str:
    candidate = Path(codex_bin)
    if candidate.exists():
        return str(candidate.resolve())
    resolved = shutil.which(codex_bin)
    if resolved:
        return resolved
    raise WorkflowError(f"codex CLI '{codex_bin}' is not available on PATH.")


def ensure_google_api_key() -> None:
    if not os.environ.get("GOOGLE_API_KEY"):
        raise WorkflowError("GOOGLE_API_KEY is not set.")


def preflight_consolidation(config: ConsolidationConfig) -> str:
    ensure_google_api_key()
    return resolve_codex_executable(config.codex_bin)


def run_backend_postprocess(
    api: GraphBuilderAPI,
    *,
    embedding_provider: str,
    embedding_model: str,
) -> None:
    if not api.health_check():
        raise WorkflowError("Local graph runtime health check failed.")

    response = api.connect(embedding_provider, embedding_model)
    if response.get("status") != "Success":
        raise WorkflowError(f"Neo4j connection failed: {response.get('message', response)}")

    log(f"Running graph index tasks: {', '.join(POSTPROCESS_TASKS)}")
    response = api.post_processing(POSTPROCESS_TASKS, embedding_provider, embedding_model)
    if response.get("status") != "Success":
        raise WorkflowError(f"Graph index setup failed: {response.get('message', response)}")


def build_consolidation_command(config: ConsolidationConfig, codex_executable: str) -> list[str]:
    command = [
        sys.executable,
        str(REPO_ROOT / "consolidate_self_improving.py"),
        "--max-iterations",
        str(config.max_iterations),
        "--required-consecutive-passes",
        str(effective_required_consecutive_passes(config)),
        "--codex-bin",
        codex_executable,
    ]
    if config.run_dir:
        command.extend(["--run-dir", config.run_dir])
    if config.dry_run:
        command.append("--dry-run")
    if config.resume:
        command.append("--resume")
    return command


def build_consolidation_env(
    *,
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    neo4j_database: str,
) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "NEO4J_URI": neo4j_uri,
            "NEO4J_USERNAME": neo4j_user,
            "NEO4J_PASSWORD": neo4j_password,
            "NEO4J_DATABASE": neo4j_database,
        }
    )
    return env


def run_consolidation(
    *,
    config: ConsolidationConfig,
    codex_executable: str,
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    neo4j_database: str,
    runner: CommandRunner | None = None,
) -> None:
    command = build_consolidation_command(config, codex_executable)
    env = build_consolidation_env(
        neo4j_uri=neo4j_uri,
        neo4j_user=neo4j_user,
        neo4j_password=neo4j_password,
        neo4j_database=neo4j_database,
    )

    log(f"Running consolidation: {' '.join(command)}")
    result = (runner or CommandRunner()).run(command, cwd=REPO_ROOT, env=env)
    if result.returncode != 0:
        raise WorkflowError(f"consolidate_self_improving.py failed with exit code {result.returncode}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run graph indexes followed by self-improving consolidation.")
    parser.add_argument("--neo4j-uri", default=DEFAULT_NEO4J_URI, help="Neo4j Bolt URI")
    parser.add_argument("--neo4j-user", default=DEFAULT_NEO4J_USER, help="Neo4j username")
    parser.add_argument("--neo4j-password", default=DEFAULT_NEO4J_PASSWORD, help="Neo4j password")
    parser.add_argument("--neo4j-database", default=DEFAULT_NEO4J_DATABASE, help="Neo4j database name")
    parser.add_argument("--embedding-provider", default="sentence-transformer", help="Embedding provider")
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2", help="Embedding model")
    parser.add_argument("--max-iterations", type=int, default=5, help="Max consolidation iterations")
    parser.add_argument("--run-dir", help="Consolidation run directory")
    parser.add_argument("--codex-bin", default="codex", help="Codex CLI executable")
    parser.add_argument("--dry-run", action="store_true", help="Run consolidation in dry-run mode")
    parser.add_argument("--resume", action="store_true", help="Resume an existing consolidation run-dir")
    parser.add_argument(
        "--required-consecutive-passes",
        type=int,
        default=0,
        help="Override consolidation stop gate passes (default: 1, or 2 with --dry-run)",
    )
    args = parser.parse_args()

    if args.max_iterations <= 0:
        parser.error("--max-iterations must be > 0")
    if args.required_consecutive_passes < 0:
        parser.error("--required-consecutive-passes must be >= 0")
    if args.resume and not args.run_dir:
        parser.error("--resume requires --run-dir")
    return args


def main() -> int:
    args = parse_args()
    consolidation_config = ConsolidationConfig(
        max_iterations=args.max_iterations,
        run_dir=args.run_dir,
        codex_bin=args.codex_bin,
        dry_run=args.dry_run,
        resume=args.resume,
        required_consecutive_passes=args.required_consecutive_passes,
    )
    codex_executable = preflight_consolidation(consolidation_config)
    api = GraphBuilderAPI(
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_password=args.neo4j_password,
        neo4j_database=args.neo4j_database,
    )
    run_backend_postprocess(
        api,
        embedding_provider=args.embedding_provider,
        embedding_model=args.embedding_model,
    )
    run_consolidation(
        config=consolidation_config,
        codex_executable=codex_executable,
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_password=args.neo4j_password,
        neo4j_database=args.neo4j_database,
    )
    log("Post-processing workflow completed successfully.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except WorkflowError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
