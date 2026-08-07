from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

from neo4j import GraphDatabase

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from notebooklm_graph_pipe.community import ModelCommunityReporter, NativeCommunityBuilder, Neo4jCommunityStore
from notebooklm_graph_pipe.community.models import CommunityConfig
from notebooklm_graph_pipe.ingestion.embeddings import EmbeddingConfig, MiniLMEmbedder
from notebooklm_graph_pipe.ingestion.manifest import load_manifest
from notebooklm_graph_pipe.runtime.llm_json_utils import build_single_prompt_clients
from notebooklm_graph_pipe.runtime.llm_routing import COMMUNITY_REPORT_ROLE, resolve_prompt_role
from notebooklm_graph_pipe.runtime.model_adapters import RoutedJsonAdapter
from notebooklm_graph_pipe.runtime.model_executor import ExecutionPolicy, ModelExecutor
from notebooklm_graph_pipe.runtime.neo4j_connection import resolve_connection_mapping
from notebooklm_graph_pipe.workflows import CommunityBuildStep, WorkflowContext, WorkflowDefinition, WorkflowEngine
from notebooklm_graph_pipe.workflows.migration import MigrationDelegateStep, UpgradeInPlaceStep


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a resumable GraphScribe corpus workflow.")
    parser.add_argument("--manifest-path", type=Path, required=True)
    subparsers = parser.add_subparsers(dest="workflow", required=True)
    upgrade = subparsers.add_parser("neo4j-schema-upgrade")
    upgrade.add_argument("--backup-file", type=Path)
    upgrade.add_argument("--execute", action="store_true")
    upgrade.add_argument("--confirm-source")
    upgrade.add_argument("--run-id")
    migrate = subparsers.add_parser(
        "neo4j-migrate",
        help="Run portable, corpus-merge, Aura, or in-place migration; pass migration flags after --.",
    )
    migrate.add_argument("--run-id")
    community = subparsers.add_parser("community-build")
    community.add_argument("--run-id")
    community.add_argument("--llm-routing-config")
    return parser


def main(argv: list[str] | None = None) -> int:
    args, remaining = build_parser().parse_known_args(argv)
    manifest_path = args.manifest_path.resolve()
    manifest = load_manifest(manifest_path)
    if manifest is None:
        raise ValueError(f"Manifest does not exist: {manifest_path}")
    run_id = args.run_id or str(uuid.uuid4())
    resources = []
    if args.workflow == "neo4j-schema-upgrade":
        if remaining:
            raise ValueError(f"Unrecognized arguments: {' '.join(remaining)}")
        connection = resolve_connection_mapping(manifest.neo4j)
        step = UpgradeInPlaceStep(
            connection,
            manifest_path,
            args.backup_file,
            args.execute,
            args.confirm_source,
        )
        definition = WorkflowDefinition("neo4j-schema-upgrade", "1", (step,))
        configuration = {
            "execute": args.execute,
            "source": f"{connection.uri}|{connection.database}",
        }
    elif args.workflow == "neo4j-migrate":
        migration_args = list(remaining)
        if migration_args and migration_args[0] == "--":
            migration_args.pop(0)
        if "--manifest-path" not in migration_args:
            migration_args.extend(("--manifest-path", str(manifest_path)))
        step = MigrationDelegateStep(tuple(migration_args))
        definition = WorkflowDefinition("neo4j-migrate", "1", (step,))
        configuration = {"arguments": migration_args}
    else:
        if remaining:
            raise ValueError(f"Unrecognized arguments: {' '.join(remaining)}")
        connection = resolve_connection_mapping(manifest.neo4j)
        driver = GraphDatabase.driver(connection.uri, auth=(connection.username, connection.password))
        resources.append(driver)
        store = Neo4jCommunityStore(driver, connection.database, manifest.corpus_id)
        projection = store.active_projection()
        role = resolve_prompt_role(
            args.llm_routing_config,
            COMMUNITY_REPORT_ROLE,
            default_client="genai",
            default_model="gemini-2.5-flash",
        )
        client = build_single_prompt_clients(role.client)[role.client]
        execution = manifest.execution
        executor = ModelExecutor(
            {"community-report": RoutedJsonAdapter(role, client)},
            {COMMUNITY_REPORT_ROLE: "community-report"},
            policies={
                COMMUNITY_REPORT_ROLE: ExecutionPolicy(
                    max_concurrency=int(
                        (execution.get("role_limits") or {}).get(
                            COMMUNITY_REPORT_ROLE,
                            execution["default_max_concurrency"],
                        )
                    )
                )
            },
            cache_path=manifest_path.parent / str(execution["cache_path"]),
            metrics_path=manifest_path.parent / str(execution["metrics_path"]),
        )
        embedder = MiniLMEmbedder(
            EmbeddingConfig(
                model=manifest.embedding_model,
                dimension=manifest.embedding_dimension,
                normalize=manifest.embedding_normalized,
            )
        )
        raw = manifest.community
        config = CommunityConfig(
            max_cluster_size=int(raw["max_cluster_size"]),
            seed=int(raw["seed"]),
            relationship_weighting=str(raw["relationship_weighting"]),
            algorithm=str(raw["algorithm"]),
            prompt_hash=str(raw.get("prompt_hash") or "default"),
            report_model_fingerprint=f"{role.client}:{role.model}:{role.reasoning_effort or 'default'}",
            embedding_fingerprint=embedder.fingerprint,
        )
        step = CommunityBuildStep(
            store,
            NativeCommunityBuilder(),
            ModelCommunityReporter(executor),
            embedder,
            config,
        )
        definition = WorkflowDefinition("community-build", "1", (step,))
        configuration = {
            "community": raw,
            "report_model": config.report_model_fingerprint,
            "embedding": config.embedding_fingerprint,
        }
    root = manifest_path.parent / "workflows"
    resume = bool(args.run_id and (root / run_id / "run.json").exists())
    inputs = {}
    if args.workflow == "community-build":
        inputs = {
            "active_revision_hash": projection.active_revision_hash,
            "projection_fingerprint": projection.fingerprint,
        }
    try:
        result = WorkflowEngine(root).run(
            definition,
            WorkflowContext(
                manifest.corpus_id,
                root / run_id,
                configuration,
                inputs=inputs,
            ),
            run_id=run_id,
            resume=resume,
        )
    finally:
        for resource in resources:
            resource.close()
    print(json.dumps({"run_id": result.run_id, "status": result.status}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
