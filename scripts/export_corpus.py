from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from neo4j import GraphDatabase

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from notebooklm_graph_pipe.exports import export_portable_projection
from notebooklm_graph_pipe.ingestion.manifest import load_manifest
from notebooklm_graph_pipe.runtime.neo4j_connection import resolve_connection_mapping


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export a versioned Neo4j corpus projection to Parquet/GraphML.")
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--graphml", action="store_true")
    args = parser.parse_args(argv)
    manifest = load_manifest(args.manifest_path.resolve())
    if manifest is None:
        raise ValueError(f"Manifest does not exist: {args.manifest_path.resolve()}")
    connection = resolve_connection_mapping(manifest.neo4j)
    with GraphDatabase.driver(connection.uri, auth=(connection.username, connection.password)) as driver:
        result = export_portable_projection(
            driver,
            connection.database,
            manifest.corpus_id,
            args.output_root,
            graphml=args.graphml,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
