from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import networkx as nx
import pyarrow as pa
import pyarrow.parquet as pq


EXPORT_SCHEMA_VERSION = 1

QUERIES = {
    "documents": """
        MATCH (:Corpus {id: $corpus_id})-[:HAS_DOCUMENT]->(document:Document)
              -[:ACTIVE_REVISION]->(revision:DocumentRevision)
        RETURN document.id AS document_id, document.title AS title,
               document.source_uri AS source_uri, document.source_type AS source_type,
               revision.id AS active_revision_id
        ORDER BY document_id
    """,
    "revisions": """
        MATCH (:Corpus {id: $corpus_id})-[:HAS_DOCUMENT]->(document:Document)
              -[:ACTIVE_REVISION]->(revision:DocumentRevision)
        RETURN revision.id AS revision_id, document.id AS document_id,
               revision.source_checksum AS source_checksum, revision.vector_ready AS vector_ready,
               revision.graph_ready AS graph_ready
        ORDER BY revision_id
    """,
    "parents": """
        MATCH (:Corpus {id: $corpus_id})-[:HAS_DOCUMENT]->(document:Document)
              -[:ACTIVE_REVISION]->(revision:DocumentRevision)-[:HAS_PARENT]->(parent:ParentChunk)
        RETURN parent.id AS parent_id, revision.id AS revision_id, document.id AS document_id,
               parent.position AS position, parent.text AS text, parent.section_path AS section_path,
               parent.page_start AS page_start, parent.page_end AS page_end,
               parent.timestamp_start_ms AS timestamp_start_ms,
               parent.timestamp_end_ms AS timestamp_end_ms
        ORDER BY parent_id
    """,
    "children": """
        MATCH (:Corpus {id: $corpus_id})-[:HAS_DOCUMENT]->(document:Document)
              -[:ACTIVE_REVISION]->(revision:DocumentRevision)-[:HAS_PARENT]->(parent:ParentChunk)
              -[:HAS_CHILD]->(child:Chunk)
        RETURN child.id AS chunk_id, parent.id AS parent_id, revision.id AS revision_id,
               document.id AS document_id, child.position AS position, child.text AS text
        ORDER BY chunk_id
    """,
    "entities": """
        MATCH (:Corpus {id: $corpus_id})-[:HAS_DOCUMENT]->(:Document)
              -[:ACTIVE_REVISION]->(:DocumentRevision)-[:HAS_PARENT]->(parent:ParentChunk)
              -[mention:HAS_ENTITY]->(entity:__Entity__)
        RETURN entity.id AS entity_id, coalesce(entity.entity_type, labels(entity)[0]) AS entity_type,
               entity.description AS description, collect(DISTINCT parent.id) AS source_parent_ids,
               collect(DISTINCT coalesce(mention.extraction_state, 'VERIFIED')) AS extraction_states
        ORDER BY entity_id
    """,
    "relationships": """
        MATCH (:Corpus {id: $corpus_id})-[:HAS_DOCUMENT]->(:Document)
              -[:ACTIVE_REVISION]->(:DocumentRevision)-[:HAS_PARENT]->(parent:ParentChunk)
        WITH collect(DISTINCT parent.id) AS active_parent_ids
        MATCH (source:__Entity__)-[relationship]->(target:__Entity__)
        WHERE NOT type(relationship) IN $excluded_relationships
        WITH source, relationship, target,
             [id IN coalesce(relationship.source_parent_ids, []) WHERE id IN active_parent_ids] AS parent_ids
        WHERE size(parent_ids) > 0
        RETURN source.id AS source_id, target.id AS target_id, type(relationship) AS relationship_type,
               relationship.description AS description, parent_ids AS source_parent_ids,
               [id IN parent_ids WHERE id IN coalesce(relationship.provisional_parent_ids, [])]
                   AS provisional_parent_ids
        ORDER BY source_id, target_id, relationship_type
    """,
    "claims": """
        MATCH (:Corpus {id: $corpus_id})-[:HAS_DOCUMENT]->(:Document)
              -[:ACTIVE_REVISION]->(:DocumentRevision)-[:HAS_PARENT]->(parent:ParentChunk)
              -[evidence:SUPPORTS|CONTRADICTS]->(claim:Claim)
        RETURN claim.id AS claim_id, claim.subject AS subject, claim.predicate AS predicate,
               claim.object AS object, claim.valid_from AS valid_from, claim.valid_to AS valid_to,
               type(evidence) AS stance, evidence.extraction_confidence AS extraction_confidence,
               parent.id AS source_parent_id
        ORDER BY claim_id, source_parent_id
    """,
    "communities": """
        MATCH (:Corpus {id: $corpus_id})-[:ACTIVE_COMMUNITY_BUILD]->(build:CommunityBuild)
              -[:HAS_COMMUNITY]->(community:Community)-[:HAS_REPORT]->(report:CommunityReport)
        OPTIONAL MATCH (report)-[:HAS_FINDING]->(finding:CommunityFinding)
              -[:GROUNDED_IN]->(parent:ParentChunk)
        RETURN build.id AS build_id, community.id AS community_id, community.level AS level,
               community.parent_id AS parent_community_id, community.member_ids AS member_ids,
               report.id AS report_id, report.title AS title, report.summary AS summary,
               report.full_content AS full_content,
               collect(DISTINCT parent.id) AS source_parent_ids
        ORDER BY level, community_id
    """,
}


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return str(value)


def _checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def export_portable_projection(
    driver: Any,
    database: str,
    corpus_id: str,
    output_root: Path,
    *,
    graphml: bool = False,
) -> dict[str, Any]:
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    export_id = f"graphscribe-v{EXPORT_SCHEMA_VERSION}-{timestamp}"
    target = output_root / export_id
    if target.exists():
        raise FileExistsError(f"Export already exists: {target}")
    with tempfile.TemporaryDirectory(prefix=f".{export_id}-", dir=output_root) as temporary:
        staging = Path(temporary)
        files = {}
        rows_by_name = {}
        with driver.session(database=database) as session:
            for name, query in QUERIES.items():
                parameters = {"corpus_id": corpus_id}
                if name == "relationships":
                    parameters["excluded_relationships"] = [
                        "HAS_ENTITY", "PART_OF", "NEXT_CHUNK", "IN_REVISION", "HAS_CHILD",
                        "MEMBER_OF", "CHILD_OF", "HAS_REPORT", "GROUNDED_IN",
                    ]
                rows = [{key: _json_safe(value) for key, value in dict(row).items()} for row in session.run(query, **parameters)]
                rows_by_name[name] = rows
                path = staging / f"{name}.parquet"
                table = pa.Table.from_pylist(rows) if rows else pa.table({"_empty": pa.array([], type=pa.bool_())})
                pq.write_table(table, path, compression="zstd")
                files[path.name] = {"rows": len(rows), "sha256": _checksum(path)}
        if graphml:
            graph = nx.MultiDiGraph(corpus_id=corpus_id, export_schema_version=EXPORT_SCHEMA_VERSION)
            for entity in rows_by_name["entities"]:
                graph.add_node(entity["entity_id"], entity_type=entity.get("entity_type") or "Entity")
            for relationship in rows_by_name["relationships"]:
                graph.add_edge(
                    relationship["source_id"],
                    relationship["target_id"],
                    relationship_type=relationship["relationship_type"],
                    source_parent_ids=json.dumps(relationship.get("source_parent_ids") or []),
                )
            path = staging / "graph.graphml"
            nx.write_graphml(graph, path)
            files[path.name] = {"rows": graph.number_of_edges(), "sha256": _checksum(path)}
        manifest = {
            "export_schema_version": EXPORT_SCHEMA_VERSION,
            "export_id": export_id,
            "corpus_id": corpus_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "authority": "neo4j",
            "files": files,
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(staging, target)
    return {**manifest, "path": str(target)}
