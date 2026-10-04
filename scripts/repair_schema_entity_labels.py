#!/usr/bin/env python3
"""Take corpus-schema labels off extracted entities, reversibly.

An extracted entity typed like a corpus-schema label (Document, Claim, Chunk, ...) used to get that
label, so it posed as a schema node. Without --apply or --revert this only reads and prints the
repair plan. --apply relabels every such entity in one transaction, as persist_parent_graph now
labels new ones, and writes the before and after state of each to --journal. --revert restores the
before state recorded in a journal. Both refuse to run while the corpus sync lock is held, and
both change only an entity that is still in the state the journal expects.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from filelock import FileLock, Timeout
from neo4j import READ_ACCESS, GraphDatabase

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from notebooklm_graph_pipe.ingestion.manifest import load_manifest
from notebooklm_graph_pipe.ingestion.neo4j_store import plan_schema_label_repair, transition_entity_labels
from notebooklm_graph_pipe.runtime.neo4j_connection import resolve_connection_mapping

OPERATION = "repair-schema-entity-labels"


def _summary(entries: list[dict]) -> dict[str, object]:
    return {
        "entities": len(entries),
        "by_schema_label": dict(
            sorted(Counter(label for entry in entries for label in set(entry["before"]["labels"]) - set(entry["after"]["labels"])).items())
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Take corpus-schema labels off extracted entities, reversibly.")
    parser.add_argument("--manifest-path", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true", help="Relabel the entities and write --journal.")
    mode.add_argument("--revert", action="store_true", help="Restore the before state recorded in --journal.")
    parser.add_argument("--journal", type=Path, help="Journal written by --apply and read by --revert.")
    parser.add_argument("--confirm-target", help="Required to write; must equal <neo4j-uri>|<database>.")
    args = parser.parse_args(argv)

    manifest_path = args.manifest_path.resolve()
    manifest = load_manifest(manifest_path)
    if manifest is None:
        parser.error(f"Corpus manifest not found: {manifest_path}")
    runtime = resolve_connection_mapping(manifest.neo4j)
    writing = args.apply or args.revert
    if writing:
        if args.journal is None:
            parser.error("--apply and --revert need --journal.")
        if args.apply and args.journal.exists():
            parser.error(f"Journal already exists: {args.journal}")
        if args.confirm_target != f"{runtime.uri}|{runtime.database}":
            parser.error("--confirm-target does not equal the manifest's <neo4j-uri>|<database>.")
        if args.revert:
            journal = json.loads(args.journal.read_text(encoding="utf-8"))
            if journal.get("operation") != OPERATION or journal.get("database") != runtime.database:
                parser.error("The journal does not record this operation on this database.")
            entries = journal["entries"]

    with GraphDatabase.driver(runtime.uri, auth=(runtime.username, runtime.password)) as driver:
        if not writing:
            with driver.session(database=runtime.database, default_access_mode=READ_ACCESS) as session:
                entries = session.execute_read(plan_schema_label_repair)
            print(json.dumps({"operation": OPERATION, "mode": "plan", **_summary(entries), "entries": entries}, indent=2))
            return 0
        try:
            with FileLock(str(manifest_path.parent / "sync.lock"), timeout=0):
                with driver.session(database=runtime.database) as session, session.begin_transaction() as tx:
                    if args.apply:
                        entries = plan_schema_label_repair(tx)
                        transition_entity_labels(tx, entries, "before", "after")
                        journal = {"operation": OPERATION, "corpus_id": manifest.corpus_id, "database": runtime.database, "entries": entries}
                        args.journal.parent.mkdir(parents=True, exist_ok=True)
                        args.journal.write_text(json.dumps(journal, indent=2) + "\n", encoding="utf-8")
                    else:
                        transition_entity_labels(tx, entries, "after", "before")
        except Timeout:
            print("The corpus sync lock is held; retry when no ingestion is running.", file=sys.stderr)
            return 1
    print(json.dumps({"operation": OPERATION, "mode": "apply" if args.apply else "revert", **_summary(entries)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
