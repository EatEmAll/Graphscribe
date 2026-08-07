from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from neo4j import GraphDatabase

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from notebooklm_graph_pipe.community.store import Neo4jCommunityStore
from notebooklm_graph_pipe.enrichment.claims import ClaimExtractor, claim_rows
from notebooklm_graph_pipe.enrichment.prompt_tuning import PromptTuner, activate_proposal, save_proposal
from notebooklm_graph_pipe.enrichment.provisional import ProvisionalExtractionWorker, ProvisionalGraphExtractor
from notebooklm_graph_pipe.ingestion.manifest import load_manifest
from notebooklm_graph_pipe.ingestion.neo4j_store import Neo4jCorpusStore
from notebooklm_graph_pipe.runtime.llm_json_utils import build_single_prompt_clients
from notebooklm_graph_pipe.runtime.llm_routing import (
    CLAIM_EXTRACTION_ROLE,
    PROMPT_TUNING_ROLE,
    resolve_prompt_role,
)
from notebooklm_graph_pipe.runtime.model_adapters import RoutedJsonAdapter
from notebooklm_graph_pipe.runtime.model_executor import ExecutionPolicy, ModelExecutor
from notebooklm_graph_pipe.runtime.neo4j_connection import resolve_connection_mapping


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Run opt-in GraphScribe graph enrichment workflows.")
    result.add_argument("--manifest-path", type=Path, required=True)
    result.add_argument("--llm-routing-config")
    subparsers = result.add_subparsers(dest="command", required=True)
    prompt = subparsers.add_parser("prompt-propose")
    prompt.add_argument("--output", type=Path, required=True)
    prompt.add_argument("--sample-size", type=int, default=24)
    activation = subparsers.add_parser("prompt-activate")
    activation.add_argument("--proposal", type=Path, required=True)
    activation.add_argument("--confirm-proposal-id", required=True)
    provisional = subparsers.add_parser("provisional-build")
    provisional.add_argument("--limit", type=int, default=100)
    claims = subparsers.add_parser("claims-build")
    claims.add_argument("--limit", type=int, default=100)
    return result


def model_executor(manifest_path: Path, manifest, config_path: str | None, role_name: str) -> ModelExecutor:
    defaults = {
        PROMPT_TUNING_ROLE: ("genai", "gemini-2.5-flash"),
        CLAIM_EXTRACTION_ROLE: ("genai", "gemini-2.5-flash"),
    }
    default_client, default_model = defaults[role_name]
    role = resolve_prompt_role(
        config_path,
        role_name,
        default_client=default_client,
        default_model=default_model,
    )
    client = build_single_prompt_clients(role.client)[role.client]
    execution = manifest.execution
    concurrency = int(
        (execution.get("role_limits") or {}).get(role_name, execution["default_max_concurrency"])
    )
    return ModelExecutor(
        {role_name: RoutedJsonAdapter(role, client)},
        {role_name: role_name},
        policies={role_name: ExecutionPolicy(max_concurrency=concurrency)},
        cache_path=manifest_path.parent / str(execution["cache_path"]),
        metrics_path=manifest_path.parent / str(execution["metrics_path"]),
    )


def representative_samples(parents, size: int) -> list[dict[str, str]]:
    if size <= 0:
        raise ValueError("sample-size must be positive.")
    ordered = sorted(parents, key=lambda item: item.id)
    if len(ordered) <= size:
        selected = ordered
    elif size == 1:
        selected = [ordered[len(ordered) // 2]]
    else:
        selected = [ordered[round(index * (len(ordered) - 1) / (size - 1))] for index in range(size)]
    return [{"parent_id": item.id, "text": item.text} for item in selected]


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    manifest_path = args.manifest_path.resolve()
    manifest = load_manifest(manifest_path)
    if manifest is None:
        raise ValueError(f"Manifest does not exist: {manifest_path}")
    if args.command == "prompt-activate":
        active = activate_proposal(
            args.proposal.resolve(),
            manifest_path,
            manifest,
            expected_proposal_id=args.confirm_proposal_id,
        )
        print(json.dumps({"proposal_id": active.id, "status": active.status}, indent=2))
        return 0

    connection = resolve_connection_mapping(manifest.neo4j)
    with GraphDatabase.driver(connection.uri, auth=(connection.username, connection.password)) as driver:
        if args.command == "provisional-build":
            worker = ProvisionalExtractionWorker(
                Neo4jCorpusStore(driver, connection.database, corpus_id=manifest.corpus_id),
                ProvisionalGraphExtractor(),
            )
            print(json.dumps(worker.run_batch(args.limit), indent=2, sort_keys=True))
            return 0

        if args.command == "claims-build":
            extractor = ClaimExtractor(
                model_executor(manifest_path, manifest, args.llm_routing_config, CLAIM_EXTRACTION_ROLE)
            )
            store = Neo4jCorpusStore(driver, connection.database, corpus_id=manifest.corpus_id)
            pending = store.pending_claim_parents(args.limit, extractor.fingerprint)
            processed = 0
            claims = 0
            failed = 0
            for parent in pending:
                try:
                    extracted = extractor.extract(str(parent["parent_id"]), str(parent.get("text") or ""))
                    store.persist_parent_claims(
                        str(parent["parent_id"]),
                        claim_rows(extracted),
                        extraction_fingerprint=extractor.fingerprint,
                    )
                    processed += 1
                    claims += len(extracted)
                except Exception as exc:
                    store.fail_parent_claims(str(parent["parent_id"]), str(exc))
                    failed += 1
            print(
                json.dumps(
                    {"parents_processed": processed, "claims": claims, "failed": failed},
                    indent=2,
                    sort_keys=True,
                )
            )
            return 1 if failed else 0

        projection = Neo4jCommunityStore(driver, connection.database, manifest.corpus_id).active_projection()
        if args.command == "prompt-propose":
            samples = representative_samples(projection.parents, args.sample_size)
            if not samples:
                raise RuntimeError("The active corpus does not contain parent chunks to sample.")
            proposal = PromptTuner(
                model_executor(manifest_path, manifest, args.llm_routing_config, PROMPT_TUNING_ROLE)
            ).propose(manifest.corpus_id, samples, manifest.graph.get("entity_types") or [])
            save_proposal(args.output.resolve(), proposal)
            print(json.dumps({"proposal_id": proposal.id, "status": proposal.status, "path": str(args.output.resolve())}, indent=2))
            return 0

        raise ValueError(f"Unsupported enrichment command: {args.command}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
