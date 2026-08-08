"""
Tier 1 - Lemmatized Node Merge
==============================
Finds entity nodes that differ only by plural/suffix form (e.g. Trade/Trades,
Model/Models) and merges them into a single canonical node using APOC refactor.

Usage:
    python scripts/consolidation/consolidate_tier1_lemmatize.py [--dry-run]
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from typing import Any

import inflect
from neo4j import GraphDatabase

DEFAULT_NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
DEFAULT_NEO4J_USER = os.environ.get("NEO4J_USERNAME", "neo4j")
DEFAULT_NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "password123")
DEFAULT_NEO4J_DATABASE = os.environ.get("NEO4J_DATABASE", "neo4j")

p = inflect.engine()


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = [str(item).strip() for item in value if str(item).strip()]
        return " | ".join(parts)
    return str(value).strip()


def canonical(name: str) -> str:
    """Return a normalized canonical form: lowercase, singular."""
    name = _coerce_text(name).lower()
    words = name.split()
    if words:
        singular = p.singular_noun(words[-1])
        if singular:
            words[-1] = singular
    return " ".join(words)


def fetch_entities(session, scope_revision_ids: list[str] | None = None) -> list[dict[str, Any]]:
    result = session.run(
        """
        MATCH (n:__Entity__)
        WHERE NOT n:CorpusSource
          AND NOT EXISTS { (n)-[:HAS_SOURCE|MATERIALIZED_AS|LEGACY_EVIDENCE]-() }
        RETURN elementId(n) AS eid, n.id AS name, labels(n) AS labels,
               CASE WHEN $scope_revision_ids IS NULL THEN true ELSE EXISTS {
                   MATCH (n)<-[:HAS_ENTITY]-(:ParentChunk)<-[:HAS_PARENT]-(revision:DocumentRevision)
                   WHERE revision.id IN $scope_revision_ids
               } END AS in_scope
        """,
        scope_revision_ids=scope_revision_ids,
    )
    return [
        {"eid": row["eid"], "name": row["name"], "labels": row["labels"], "in_scope": bool(row["in_scope"])}
        for row in result
    ]


def merge_group(session, eids: list[str], canonical_name: str, dry_run: bool) -> None:
    """Merge all nodes in the group into the first one (canonical)."""
    if len(eids) < 2 or dry_run:
        return

    session.run(
        """
        MATCH (n:__Entity__)
        WHERE elementId(n) IN $eids
          AND NOT n:CorpusSource
          AND NOT EXISTS { (n)-[:HAS_SOURCE|MATERIALIZED_AS|LEGACY_EVIDENCE]-() }
        WITH collect(n) AS nodes
        WHERE size(nodes) = size($eids) AND size(nodes) > 1
        CALL apoc.refactor.mergeNodes(nodes, {
            properties: 'combine',
            mergeRels: true
        })
        YIELD node
        SET node.id = $canonical_name
        RETURN node
        """,
        eids=eids,
        canonical_name=canonical_name,
    )


def run(
    *,
    dry_run: bool,
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    neo4j_database: str,
    scope_revision_ids: list[str] | None = None,
) -> None:
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    try:
        with driver.session(database=neo4j_database) as session:
            print("Fetching all __Entity__ nodes...")
            entities = (
                fetch_entities(session, scope_revision_ids)
                if scope_revision_ids is not None
                else fetch_entities(session)
            )
            print(f"  Total entities: {len(entities)}")

            groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for entity in entities:
                name = _coerce_text(entity["name"])
                if not name:
                    continue
                key = canonical(name)
                groups[key].append(entity)

            merge_candidates = {
                key: sorted(members, key=lambda member: member["in_scope"])
                for key, members in groups.items()
                if len(members) > 1 and any(member["in_scope"] for member in members)
            }
            print(f"  Merge candidate groups: {len(merge_candidates)}")

            if not merge_candidates:
                print("Nothing to merge.")
                return

            total_merged = 0
            for canon, members in sorted(merge_candidates.items(), key=lambda item: -len(item[1])):
                names = [_coerce_text(member["name"]) for member in members]
                eids = [member["eid"] for member in members]
                print(f"\n  [{len(members)} nodes] canonical='{canon}'")
                for name in names:
                    print(f"    - '{name}'")

                if not dry_run:
                    merge_group(session, eids, canon, dry_run=False)
                    total_merged += len(members) - 1

            if dry_run:
                would_merge = sum(len(values) - 1 for values in merge_candidates.values())
                print(f"\n[DRY RUN] Would merge {would_merge} nodes across {len(merge_candidates)} groups.")
            else:
                print(f"\nMerged {total_merged} duplicate nodes into {len(merge_candidates)} canonical nodes.")
    finally:
        driver.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Tier 1: Lemmatized entity merge")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be merged without writing")
    parser.add_argument("--neo4j-uri", type=str, default=DEFAULT_NEO4J_URI)
    parser.add_argument("--neo4j-user", type=str, default=DEFAULT_NEO4J_USER)
    parser.add_argument("--neo4j-password", type=str, default=DEFAULT_NEO4J_PASSWORD)
    parser.add_argument("--neo4j-database", type=str, default=DEFAULT_NEO4J_DATABASE)
    args = parser.parse_args()

    run(
        dry_run=args.dry_run,
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_password=args.neo4j_password,
        neo4j_database=args.neo4j_database,
    )


if __name__ == "__main__":
    main()
