#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from notebooklm_graph_pipe.paths import RUNS_DIR
from notebooklm_graph_pipe.runtime.graph_builder_runtime import GraphBuilderAPI
from notebooklm_graph_pipe.runtime.dataset_registry import load_dataset_entry
from notebooklm_graph_pipe.runtime.llm_routing import (
    AGENT_REVIEW_ROLE,
    AGENT_TAXONOMY_TAIL_ROLE,
    GRAPH_BUILD_EMBEDDING_ROLE,
    TAXONOMY_PRIMARY_ROLE,
    TAXONOMY_SECONDARY_ROLE,
    TIER2_PRIMARY_ROLE,
    TIER2_SECONDARY_ROLE,
    TIER3_EMBEDDING_ROLE,
    TIER3_JUDGE_PRIMARY_ROLE,
    TIER3_JUDGE_SECONDARY_ROLE,
    missing_required_env_vars,
    resolve_agent_role,
    resolve_embedding_role,
    resolve_graph_build_embedding,
    resolve_prompt_role,
)

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
    llm_routing_config: str | None = None


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


def effective_required_consecutive_passes(config: ConsolidationConfig) -> int:
    if config.required_consecutive_passes > 0:
        return config.required_consecutive_passes
    return 2 if config.dry_run else 1


def resolve_cli_executable(executable: str) -> str:
    candidate = Path(executable)
    if candidate.exists():
        return str(candidate.resolve())
    resolved = shutil.which(executable)
    if resolved:
        return resolved
    raise WorkflowError(f"CLI executable '{executable}' is not available on PATH.")


def resolve_codex_executable(codex_bin: str) -> str:
    return resolve_cli_executable(codex_bin)


def _required_consolidation_api_envs(config: ConsolidationConfig) -> list[str]:
    prompt_roles = [
        resolve_prompt_role(config.llm_routing_config, TIER2_PRIMARY_ROLE, default_client="genai", default_model="gemini-3.1-flash-lite-preview"),
        resolve_prompt_role(config.llm_routing_config, TIER2_SECONDARY_ROLE, default_client="genai", default_model="gemini-3-flash-preview"),
        resolve_prompt_role(config.llm_routing_config, TAXONOMY_PRIMARY_ROLE, default_client="genai", default_model="gemini-3.1-flash-lite-preview"),
        resolve_prompt_role(config.llm_routing_config, TAXONOMY_SECONDARY_ROLE, default_client="genai", default_model="gemini-3.1-pro-preview"),
        resolve_prompt_role(config.llm_routing_config, TIER3_JUDGE_PRIMARY_ROLE, default_client="genai", default_model="gemini-3.1-flash-lite-preview"),
        resolve_prompt_role(config.llm_routing_config, TIER3_JUDGE_SECONDARY_ROLE, default_client="genai", default_model="gemini-3-flash-preview"),
    ]
    embedding_role = resolve_embedding_role(
        config.llm_routing_config,
        TIER3_EMBEDDING_ROLE,
        default_client="genai",
        default_model="gemini-embedding-001",
    )
    return missing_required_env_vars(*(role.client for role in [*prompt_roles, embedding_role]))


def preflight_consolidation(config: ConsolidationConfig) -> str | None:
    missing_envs = _required_consolidation_api_envs(config)
    if missing_envs:
        if missing_envs == ["GOOGLE_API_KEY"]:
            raise WorkflowError("GOOGLE_API_KEY is not set")
        raise WorkflowError(f"Missing required environment variables: {', '.join(missing_envs)}")

    review_role = resolve_agent_role(
        config.llm_routing_config,
        AGENT_REVIEW_ROLE,
        default_client="codex",
        default_model=None,
        default_executable=config.codex_bin,
    )
    tail_role = resolve_agent_role(
        config.llm_routing_config,
        AGENT_TAXONOMY_TAIL_ROLE,
        default_client="codex",
        default_model=None,
        default_executable=config.codex_bin,
    )
    resolve_cli_executable(review_role.executable or config.codex_bin)
    resolve_cli_executable(tail_role.executable or config.codex_bin)
    if not config.llm_routing_config:
        return resolve_codex_executable(config.codex_bin)
    return None


def run_backend_postprocess(
    api: GraphBuilderAPI,
    *,
    embedding_provider: str,
    embedding_model: str,
) -> None:
    if not api.health_check():
        raise WorkflowError("Neo4j graph runtime health check failed.")

    response = api.connect(embedding_provider, embedding_model)
    if response.get("status") != "Success":
        raise WorkflowError(f"Neo4j connection failed: {response.get('message', response)}")

    log(f"Running graph index tasks: {', '.join(POSTPROCESS_TASKS)}")
    response = api.post_processing(POSTPROCESS_TASKS, embedding_provider, embedding_model)
    if response.get("status") != "Success":
        raise WorkflowError(f"Graph index setup failed: {response.get('message', response)}")


def build_consolidation_command(config: ConsolidationConfig, codex_executable: str | None = None) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "notebooklm_graph_pipe.consolidation.self_improving",
        "--max-iterations",
        str(config.max_iterations),
        "--required-consecutive-passes",
        str(effective_required_consecutive_passes(config)),
    ]
    if config.llm_routing_config:
        command.extend(["--llm-routing-config", config.llm_routing_config])
    else:
        command.extend(["--codex-bin", codex_executable or config.codex_bin])
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
    codex_executable: str | None = None,
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
        raise WorkflowError(f"notebooklm_graph_pipe.consolidation.self_improving failed with exit code {result.returncode}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run graph indexes followed by self-improving consolidation.")
    parser.add_argument("--dataset-key", help="Optional dataset key from a local registry JSON")
    parser.add_argument("--registry-path", help="Path to an optional local dataset registry JSON")
    parser.add_argument("--neo4j-uri", default=os.environ.get("NEO4J_URI", DEFAULT_NEO4J_URI), help="Neo4j Bolt URI")
    parser.add_argument("--neo4j-user", default=os.environ.get("NEO4J_USERNAME", DEFAULT_NEO4J_USER), help="Neo4j username")
    parser.add_argument("--neo4j-password", default=os.environ.get("NEO4J_PASSWORD", DEFAULT_NEO4J_PASSWORD), help="Neo4j password (prefer NEO4J_PASSWORD)")
    parser.add_argument("--neo4j-database", default=os.environ.get("NEO4J_DATABASE", DEFAULT_NEO4J_DATABASE), help="Neo4j database name")
    parser.add_argument("--embedding-provider", default=None, help="Embedding provider override")
    parser.add_argument("--embedding-model", default=None, help="Embedding model override")
    parser.add_argument("--llm-routing-config", default=None, help="Optional JSON config for role-based LLM routing")
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
        if not args.run_dir:
            stamp = datetime.now().strftime("%Y%m%d_%H%M")
            args.run_dir = str(RUNS_DIR / args.dataset_key / f"postprocess_{stamp}")
    return args


def main() -> int:
    args = parse_args()
    consolidation_config = ConsolidationConfig(
        max_iterations=args.max_iterations,
        run_dir=args.run_dir,
        codex_bin=args.codex_bin,
        llm_routing_config=args.llm_routing_config,
        dry_run=args.dry_run,
        resume=args.resume,
        required_consecutive_passes=args.required_consecutive_passes,
    )
    preflight_consolidation(consolidation_config)
    graph_build_embedding = resolve_graph_build_embedding(
        config_path=args.llm_routing_config,
        embedding_provider=args.embedding_provider,
        embedding_model=args.embedding_model,
        default_provider="sentence-transformer",
        default_model="all-MiniLM-L6-v2",
    )
    missing_graph_build_envs = missing_required_env_vars(graph_build_embedding.client)
    if missing_graph_build_envs:
        raise WorkflowError(f"Missing required environment variables: {', '.join(missing_graph_build_envs)}")
    api = GraphBuilderAPI(
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_password=args.neo4j_password,
        neo4j_database=args.neo4j_database,
    )
    api.preflight_capabilities()
    with _temporary_embedding_dimension_override(graph_build_embedding.dimension):
        run_backend_postprocess(
            api,
            embedding_provider=graph_build_embedding.client,
            embedding_model=graph_build_embedding.model,
        )
    run_consolidation(
        config=consolidation_config,
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
