from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import notebooklm_graph_pipe.consolidation.self_improving as csi


DEFAULT_MANIFEST = REPO_ROOT / "data" / "notebooklm_exports" / "imdb-scifi-test" / "manifest.json"
DEFAULT_MCP_CONFIG = REPO_ROOT / "runs" / "imdb-scifi-test" / "smoke-configs" / "claude-neo4j-mcp.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "runs" / "imdb-scifi-test" / "debug" / "claude"


def _load_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _build_full_review_prompt(manifest: dict) -> str:
    neo4j = manifest["neo4j"]
    return csi._build_codex_prompt(
        review_client="claude",
        target_concept_ratio=0.05,
        target_duplicate_rate=0.015,
        target_concept_without_taxonomy_ratio=0.60,
        current_tier2=csi.Tier2Params(
            neo4j_uri=neo4j["uri"],
            neo4j_user=neo4j["username"],
            neo4j_password=neo4j["password"],
            neo4j_database=neo4j["database"],
        ),
        current_tier3=csi.Tier3Params(
            neo4j_uri=neo4j["uri"],
            neo4j_user=neo4j["username"],
            neo4j_password=neo4j["password"],
            neo4j_database=neo4j["database"],
        ),
        current_catalog=csi._default_tier2_catalog(),
    )


def _run_case(
    *,
    name: str,
    command: list[str],
    stdin_text: str,
    output_dir: Path,
    timeout_seconds: int,
) -> dict:
    case_dir = output_dir / name
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "prompt.txt").write_text(stdin_text, encoding="utf-8")
    started = time.time()
    try:
        result = subprocess.run(
            command,
            text=True,
            input=stdin_text,
            capture_output=True,
            cwd=REPO_ROOT,
            check=False,
            timeout=timeout_seconds,
        )
        duration = time.time() - started
        stdout = result.stdout
        stderr = result.stderr
        returncode = result.returncode
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        duration = time.time() - started
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        returncode = None
        timed_out = True
    (case_dir / "stdout.txt").write_text(stdout, encoding="utf-8")
    (case_dir / "stderr.txt").write_text(stderr, encoding="utf-8")
    summary = {
        "name": name,
        "returncode": returncode,
        "timed_out": timed_out,
        "duration_seconds": round(duration, 3),
        "command": command,
        "stdin_length": len(stdin_text),
        "stdout_length": len(stdout),
        "stderr_length": len(stderr),
    }
    (case_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Isolate Claude review-agent behavior with direct CLI repros.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--mcp-config", type=Path, default=DEFAULT_MCP_CONFIG)
    parser.add_argument("--model", default="claude-haiku-4-5")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    args = parser.parse_args()

    manifest = _load_manifest(args.manifest)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    count_prompt = 'Use the neo4j MCP tool to run MATCH (n:__Entity__) RETURN count(n) AS c. Return exactly {"count": <int>} and nothing else.'
    full_prompt = _build_full_review_prompt(manifest)
    schema_file = output_dir / "review.schema.json"
    schema_file.write_text(json.dumps(csi._review_json_schema(), indent=2), encoding="utf-8")
    short_review_prompt = (
        "Use the neo4j MCP tool to run MATCH (n:__Entity__) RETURN count(n) AS entity_count. "
        "Return strict JSON matching the provided schema with the measured entity_count, "
        "set concept_only_count, duplicate_anchor_count, subclass_rel_count, "
        "concept_only_without_taxonomy_count, concept_only_degree_le_2_count, "
        "concept_only_degree_le_3_count, concept_only_with_similarity_or_alias_count to 0, "
        'focus_examples arrays to ["example"], proposed_tier2 to {"batch_size":50,"sleep_seconds":1.0,"max_nodes":1000}, '
        'proposed_tier3 to {"threshold":0.85,"max_candidates":600,"max_merges":200}, '
        'proposed_tier2_catalog to {"labels":["Concept"],"add":[],"remove":[],"rename_map":{},"guidance":{},"rationale":"test"}, '
        'diagnosis to "balanced", rationale to "test", confidence to 0.5, and is_consolidated to false.'
    )
    executable = csi._resolve_cli_executable("claude")

    common_prefix = [
        executable,
        "--mcp-config",
        str(args.mcp_config),
        "--strict-mcp-config",
        "-p",
        "--permission-mode",
        "bypassPermissions",
        "--model",
        args.model,
    ]

    summaries = [
        _run_case(
            name="simple_count",
            command=[*common_prefix, "--output-format", "json"],
            stdin_text=count_prompt,
            output_dir=output_dir,
            timeout_seconds=args.timeout_seconds,
        ),
        _run_case(
            name="short_review_schema",
            command=[*common_prefix, "--json-schema", str(schema_file)],
            stdin_text=short_review_prompt,
            output_dir=output_dir,
            timeout_seconds=args.timeout_seconds,
        ),
        _run_case(
            name="full_review_no_schema",
            command=[*common_prefix, "--output-format", "json"],
            stdin_text=full_prompt,
            output_dir=output_dir,
            timeout_seconds=args.timeout_seconds,
        ),
        _run_case(
            name="full_review",
            command=[*common_prefix, "--json-schema", str(schema_file)],
            stdin_text=full_prompt,
            output_dir=output_dir,
            timeout_seconds=args.timeout_seconds,
        ),
    ]
    (output_dir / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
