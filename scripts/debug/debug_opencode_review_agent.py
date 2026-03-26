from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import notebooklm_graph_pipe.consolidation.self_improving as csi


DEFAULT_MANIFEST = REPO_ROOT / "data" / "notebooklm_exports" / "imdb-scifi-test" / "manifest.json"
DEFAULT_XDG_CONFIG_HOME = REPO_ROOT / "runs" / "imdb-scifi-test" / "xdg"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "runs" / "imdb-scifi-test" / "debug" / "opencode"


def _load_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _build_full_review_prompt(manifest: dict) -> str:
    neo4j = manifest["neo4j"]
    return csi._build_codex_prompt(
        review_client="opencode",
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


def _parse_opencode_events(raw_text: str) -> dict:
    text_parts: list[str] = []
    tool_uses: list[dict] = []
    event_count = 0
    for line in raw_text.splitlines():
        candidate = line.strip()
        if not candidate:
            continue
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        event_count += 1
        if payload.get("type") == "text":
            part = payload.get("part") or {}
            text = part.get("text")
            if isinstance(text, str):
                text_parts.append(text)
        if payload.get("type") == "tool_use":
            tool_uses.append(payload)
    return {
        "event_count": event_count,
        "text": "\n".join(text_parts),
        "tool_uses": tool_uses,
    }


def _run_case(
    *,
    name: str,
    command: list[str],
    env: dict[str, str],
    output_dir: Path,
    timeout_seconds: int,
) -> dict:
    case_dir = output_dir / name
    case_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    try:
        result = subprocess.run(
            command,
            text=True,
            capture_output=True,
            cwd=REPO_ROOT,
            env=env,
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
    (case_dir / "stdout.jsonl").write_text(stdout, encoding="utf-8")
    (case_dir / "stderr.txt").write_text(stderr, encoding="utf-8")
    parsed = _parse_opencode_events(stdout)
    (case_dir / "normalized_text.txt").write_text(parsed["text"], encoding="utf-8")
    (case_dir / "tool_uses.json").write_text(json.dumps(parsed["tool_uses"], indent=2), encoding="utf-8")
    summary = {
        "name": name,
        "returncode": returncode,
        "timed_out": timed_out,
        "duration_seconds": round(duration, 3),
        "command": command,
        "event_count": parsed["event_count"],
        "tool_use_count": len(parsed["tool_uses"]),
        "normalized_text_length": len(parsed["text"]),
    }
    (case_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Isolate OpenCode review-agent behavior with direct CLI repros.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--xdg-config-home", type=Path, default=DEFAULT_XDG_CONFIG_HOME)
    parser.add_argument("--model", default="openrouter/moonshotai/kimi-k2.5")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    args = parser.parse_args()

    manifest = _load_manifest(args.manifest)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env["XDG_CONFIG_HOME"] = str(args.xdg_config_home)
    count_prompt = 'Use the neo4j MCP tool to run MATCH (n:__Entity__) RETURN count(n) AS c. Return exactly {"count": <int>} and nothing else.'
    full_prompt = _build_full_review_prompt(manifest)
    executable = csi._resolve_cli_executable("opencode")

    summaries = [
        _run_case(
            name="simple_count",
            command=[executable, "run", "--format", "json", "--model", args.model, count_prompt],
            env=env,
            output_dir=output_dir,
            timeout_seconds=args.timeout_seconds,
        ),
        _run_case(
            name="full_review",
            command=[executable, "run", "--format", "json", "--model", args.model, full_prompt],
            env=env,
            output_dir=output_dir,
            timeout_seconds=args.timeout_seconds,
        ),
    ]
    (output_dir / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
