from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from neo4j import GraphDatabase

from notebooklm_graph_pipe.ingestion.embeddings import EmbeddingError, weighted_parent_embedding as _weighted_parent_embedding
from notebooklm_graph_pipe.ingestion.manifest import CorpusManifest, load_manifest, save_manifest
from notebooklm_graph_pipe.ingestion.neo4j_store import GRAPH_SCHEMA_VERSION
from notebooklm_graph_pipe.runtime.neo4j_connection import (
    Neo4jConnectionSpec,
    ResolvedNeo4jConnection,
    resolve_connection,
    verify_corpus_connection,
)


UUID_PATTERN = re.compile(
    r"(?P<uuid>[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})",
    re.IGNORECASE,
)
STRUCTURAL_SCHEMA = (
    "CREATE CONSTRAINT corpus_id_unique IF NOT EXISTS FOR (n:Corpus) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT corpus_key_unique IF NOT EXISTS FOR (n:Corpus) REQUIRE n.key IS UNIQUE",
    "CREATE CONSTRAINT document_id_unique IF NOT EXISTS FOR (n:Document) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT revision_id_unique IF NOT EXISTS FOR (n:DocumentRevision) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT parent_chunk_id_unique IF NOT EXISTS FOR (n:ParentChunk) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS FOR (n:Chunk) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT community_build_id_unique IF NOT EXISTS FOR (n:CommunityBuild) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT community_id_unique IF NOT EXISTS FOR (n:Community) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT community_report_id_unique IF NOT EXISTS FOR (n:CommunityReport) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT community_finding_id_unique IF NOT EXISTS FOR (n:CommunityFinding) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT claim_id_unique IF NOT EXISTS FOR (n:Claim) REQUIRE n.id IS UNIQUE",
    "CREATE INDEX document_status IF NOT EXISTS FOR (n:Document) ON (n.status)",
    "CREATE INDEX revision_ready IF NOT EXISTS FOR (n:DocumentRevision) ON (n.vector_ready, n.graph_ready)",
    "CREATE FULLTEXT INDEX entities IF NOT EXISTS FOR (n:__Entity__) ON EACH [n.id, n.description]",
    "CREATE FULLTEXT INDEX community_report_keyword_v1 IF NOT EXISTS "
    "FOR (n:CommunityReport) ON EACH [n.summary, n.full_content]",
    "CREATE FULLTEXT INDEX claim_keyword_v1 IF NOT EXISTS "
    "FOR (n:Claim) ON EACH [n.subject, n.predicate, n.object]",
)


class CompactCorpusError(RuntimeError):
    pass


class CompactCorpusGateError(CompactCorpusError):
    def __init__(self, message: str, report: dict[str, Any]):
        super().__init__(message)
        self.report = report


def weighted_parent_embedding(children: Iterable[Mapping[str, Any]]) -> list[float]:
    try:
        return _weighted_parent_embedding(children)
    except EmbeddingError as exc:
        raise CompactCorpusError(str(exc)) from exc


@dataclass(frozen=True)
class TextMatch:
    parent_id: str
    method: str
    coverage: float
    similarity: float


@dataclass(frozen=True)
class EntityMention:
    parent_id: str
    entity_id: str
    source_chunk_ids: tuple[str, ...]


def normalize_text(value: str) -> str:
    return " ".join(re.findall(r"\w+", value.casefold(), flags=re.UNICODE))


def token_shingles(value: str, size: int = 3) -> set[tuple[str, ...]]:
    tokens = normalize_text(value).split()
    if not tokens:
        return set()
    if len(tokens) < size:
        return {tuple(tokens)}
    return {tuple(tokens[index : index + size]) for index in range(len(tokens) - size + 1)}


def ordered_token_coverage(old_text: str, candidate_text: str) -> float:
    old_tokens = normalize_text(old_text).split()
    candidate_tokens = normalize_text(candidate_text).split()
    if not old_tokens or not candidate_tokens:
        return 0.0
    matched = sum(
        block.size
        for block in difflib.SequenceMatcher(
            None,
            old_tokens,
            candidate_tokens,
            autojunk=False,
        ).get_matching_blocks()
    )
    return matched / len(old_tokens)


def direct_entity_mentions(
    parents: Sequence[Mapping[str, Any]],
    evidence_chunks: Sequence[Mapping[str, Any]],
) -> list[EntityMention]:
    entity_chunks: dict[str, set[str]] = defaultdict(set)
    for chunk in evidence_chunks:
        for entity_id in chunk.get("entity_ids") or []:
            entity_chunks[str(entity_id)].add(str(chunk["chunk_id"]))
    mentions: list[EntityMention] = []
    for parent in parents:
        parent_text = f" {normalize_text(str(parent.get('text') or ''))} "
        for entity_id, chunk_ids in entity_chunks.items():
            normalized_entity = normalize_text(entity_id)
            if normalized_entity and f" {normalized_entity} " in parent_text:
                mentions.append(
                    EntityMention(
                        str(parent["id"]),
                        entity_id,
                        tuple(sorted(chunk_ids)),
                    )
                )
    return mentions


def match_parent_text(
    old_text: str,
    parents: Sequence[Mapping[str, Any]],
    *,
    minimum_coverage: float = 0.75,
    minimum_similarity: float = 0.60,
) -> TextMatch | None:
    old_normalized = normalize_text(old_text)
    if not old_normalized:
        return None
    old_shingles = token_shingles(old_text)
    candidates: list[TextMatch] = []
    for parent in parents:
        parent_text = str(parent.get("text") or "")
        parent_normalized = normalize_text(parent_text)
        if not parent_normalized:
            continue
        if old_normalized in parent_normalized:
            coverage = 1.0
            similarity = 1.0
            candidates.append(TextMatch(str(parent["id"]), "exact", coverage, similarity))
            continue
        parent_shingles = token_shingles(parent_text)
        if not old_shingles or not parent_shingles:
            continue
        overlap = len(old_shingles & parent_shingles)
        similarity = overlap / min(len(old_shingles), len(parent_shingles))
        if similarity < minimum_similarity:
            continue
        coverage = ordered_token_coverage(old_text, parent_text)
        if coverage >= minimum_coverage and similarity >= minimum_similarity:
            candidates.append(TextMatch(str(parent["id"]), "shingle", coverage, similarity))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item.coverage, item.similarity, item.parent_id), reverse=True)
    best = candidates[0]
    if len(candidates) > 1:
        runner_up = candidates[1]
        if (best.coverage, best.similarity) == (runner_up.coverage, runner_up.similarity):
            return None
    return best


def _batches(values: Sequence[Any], size: int) -> Iterator[Sequence[Any]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _connection_from_args(args: argparse.Namespace, prefix: str) -> ResolvedNeo4jConnection:
    password_env = str(getattr(args, f"{prefix}_password_env"))
    spec = Neo4jConnectionSpec(
        uri=str(getattr(args, f"{prefix}_uri") or ""),
        username=str(getattr(args, f"{prefix}_user") or "neo4j"),
        database=str(getattr(args, f"{prefix}_database") or "neo4j"),
        password_env=password_env,
    )
    return resolve_connection(spec, password=os.environ.get(password_env, ""))


def _inventory(driver: Any, database: str) -> dict[str, int]:
    with driver.session(database=database) as session:
        row = session.run(
            "MATCH (n) WITH count(n) AS nodes MATCH ()-[r]->() RETURN nodes, count(r) AS relationships"
        ).single(strict=True)
    return {"nodes": int(row["nodes"]), "relationships": int(row["relationships"])}


def _source_rows(source: Any, database: str, corpus_id: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with source.session(database=database) as session:
        corpus = session.run(
            "MATCH (n:Corpus {id: $corpus_id}) RETURN properties(n) AS properties",
            corpus_id=corpus_id,
        ).single()
        if corpus is None:
            raise CompactCorpusError(f"Source corpus not found: {corpus_id}")
        rows = [
            row.data()
            for row in session.run(
                """
                MATCH (:Corpus {id: $corpus_id})-[:HAS_DOCUMENT]->(document:Document)
                      -[:ACTIVE_REVISION]->(revision:DocumentRevision)-[:HAS_PARENT]->(parent:ParentChunk)
                RETURN document.id AS document_id, properties(document) AS document,
                       revision.id AS revision_id, properties(revision) AS revision,
                       parent.id AS parent_id, properties(parent) AS parent
                ORDER BY document.id, parent.position, parent.id
                """,
                corpus_id=corpus_id,
            )
        ]
    return dict(corpus["properties"]), rows


def _parent_embeddings(
    source: Any,
    database: str,
    parent_ids: Sequence[str],
    *,
    batch_size: int,
) -> dict[str, list[float]]:
    result: dict[str, list[float]] = {}
    for identifiers in _batches(parent_ids, batch_size):
        children: dict[str, list[dict[str, Any]]] = defaultdict(list)
        with source.session(database=database, fetch_size=1000) as session:
            for row in session.run(
                """
                UNWIND $parent_ids AS parent_id
                MATCH (parent:ParentChunk {id: parent_id})-[:HAS_CHILD]->(child:Chunk)
                RETURN parent.id AS parent_id, child.text AS text,
                       child.token_count AS token_count, child.embedding AS embedding
                ORDER BY parent.id, child.position, child.id
                """,
                parent_ids=list(identifiers),
            ):
                children[str(row["parent_id"])].append(row.data())
        for parent_id in identifiers:
            result[str(parent_id)] = weighted_parent_embedding(children.get(str(parent_id), []))
    return result


def _embedding_coverage(source: Any, database: str, corpus_id: str) -> dict[str, int]:
    with source.session(database=database) as session:
        row = session.run(
            """
            MATCH (:Corpus {id: $corpus_id})-[:HAS_DOCUMENT]->(:Document)-[:ACTIVE_REVISION]->
                  (:DocumentRevision)-[:HAS_PARENT]->(parent:ParentChunk)-[:HAS_CHILD]->(child:Chunk)
            WITH parent, count(child) AS children, count(child.embedding) AS embedded_children
            RETURN count(parent) AS parents,
                   count(CASE WHEN children > 0 AND children = embedded_children THEN 1 END) AS complete_parents,
                   sum(children) AS children, sum(embedded_children) AS embedded_children
            """,
            corpus_id=corpus_id,
        ).single(strict=True)
    return {key: int(row[key]) for key in ("parents", "complete_parents", "children", "embedded_children")}


def _write_compact_structure(
    target: Any,
    database: str,
    corpus_properties: Mapping[str, Any],
    source_rows: Sequence[Mapping[str, Any]],
    embeddings: Mapping[str, Sequence[float]],
    *,
    batch_size: int,
) -> None:
    documents: dict[str, dict[str, Any]] = {}
    revisions: dict[str, dict[str, Any]] = {}
    revision_documents: dict[str, str] = {}
    parents: list[dict[str, Any]] = []
    for row in source_rows:
        documents[str(row["document_id"])] = dict(row["document"])
        revision = dict(row["revision"])
        revision["graph_ready"] = False
        revisions[str(row["revision_id"])] = revision
        revision_documents[str(row["revision_id"])] = str(row["document_id"])
        parent = dict(row["parent"])
        parent["embedding"] = list(embeddings[str(row["parent_id"])])
        parents.append(
            {
                "id": str(row["parent_id"]),
                "document_id": str(row["document_id"]),
                "revision_id": str(row["revision_id"]),
                "properties": parent,
            }
        )
    target.execute_query(
        "CREATE (corpus:Corpus) SET corpus = $properties",
        properties=dict(corpus_properties),
        database_=database,
    )
    for values in _batches(list(documents.items()), batch_size):
        target.execute_query(
            """
            MATCH (corpus:Corpus {id: $corpus_id})
            UNWIND $rows AS row
            MERGE (document:Document {id: row.id}) SET document = row.properties
            MERGE (corpus)-[:HAS_DOCUMENT]->(document)
            """,
            corpus_id=str(corpus_properties["id"]),
            rows=[{"id": key, "properties": properties} for key, properties in values],
            database_=database,
        )
    for values in _batches(list(revisions.items()), batch_size):
        target.execute_query(
            """
            UNWIND $rows AS row
            MATCH (document:Document {id: row.document_id})
            MERGE (revision:DocumentRevision {id: row.id}) SET revision = row.properties
            MERGE (document)-[:HAS_REVISION]->(revision)
            MERGE (document)-[:ACTIVE_REVISION]->(revision)
            """,
            rows=[
                {
                    "id": key,
                    "document_id": revision_documents[key],
                    "properties": properties,
                }
                for key, properties in values
            ],
            database_=database,
        )
    for values in _batches(parents, batch_size):
        target.execute_query(
            """
            UNWIND $rows AS row
            MATCH (revision:DocumentRevision {id: row.revision_id})
            MERGE (parent:ParentChunk {id: row.id}) SET parent = row.properties
            MERGE (revision)-[:HAS_PARENT]->(parent)
            """,
            rows=list(values),
            database_=database,
        )


def _source_uuid(value: str) -> str | None:
    match = UUID_PATTERN.search(value)
    return match.group("uuid").lower() if match else None


def _old_chunk_evidence(target: Any, database: str) -> list[dict[str, Any]]:
    with target.session(database=database, fetch_size=1000) as session:
        return [
            row.data()
            for row in session.run(
                """
                MATCH (chunk:Chunk)-[:HAS_ENTITY]->(entity:__Entity__)
                RETURN chunk.id AS chunk_id, chunk.fileName AS file_name, chunk.text AS text,
                       collect(DISTINCT entity.id) AS entity_ids
                ORDER BY chunk.id
                """
            )
        ]


def _build_bridges(
    target: Any,
    database: str,
    source_rows: Sequence[Mapping[str, Any]],
    *,
    minimum_coverage: float,
    minimum_similarity: float,
    batch_size: int,
    write: bool,
) -> dict[str, Any]:
    parents_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    document_ids_by_source: dict[str, set[str]] = defaultdict(set)
    for row in source_rows:
        source_uuid = _source_uuid(str(row["document"].get("source_uri") or ""))
        if source_uuid:
            document_ids_by_source[source_uuid].add(str(row["document_id"]))
            parents_by_source[source_uuid].append(
                {
                    "id": str(row["parent_id"]),
                    "text": str(row["parent"].get("text") or ""),
                    "normalized": normalize_text(str(row["parent"].get("text") or "")),
                }
            )
    evidence = _old_chunk_evidence(target, database)
    evidence_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for chunk in evidence:
        source_uuid = _source_uuid(str(chunk.get("file_name") or ""))
        if source_uuid:
            evidence_by_source[source_uuid].append(chunk)
    matched: list[tuple[dict[str, Any], TextMatch]] = []
    evidence_without_source = [
        chunk for chunk in evidence if _source_uuid(str(chunk.get("file_name") or "")) is None
    ]
    unmapped: list[str] = [str(chunk["chunk_id"]) for chunk in evidence_without_source]
    evidence_without_source_uuid = len(evidence_without_source)
    missing_source_uuids = sorted(set(evidence_by_source) - set(parents_by_source))
    matched_source_uuids: set[str] = set()
    ambiguous_source_uuids = sorted(
        source_uuid for source_uuid, document_ids in document_ids_by_source.items() if len(document_ids) != 1
    )
    text_threshold_or_ambiguity = 0
    for source_uuid, chunks in evidence_by_source.items():
        if source_uuid not in parents_by_source or source_uuid in ambiguous_source_uuids:
            unmapped.extend(str(chunk["chunk_id"]) for chunk in chunks)
            continue
        for chunk in chunks:
            old_text = str(chunk.get("text") or "")
            match = match_parent_text(
                old_text,
                parents_by_source.get(source_uuid, []),
                minimum_coverage=minimum_coverage,
                minimum_similarity=minimum_similarity,
            )
            if match is None:
                unmapped.append(str(chunk["chunk_id"]))
                text_threshold_or_ambiguity += 1
            else:
                matched.append((chunk, match))
                matched_source_uuids.add(source_uuid)
    coverage = len(matched) / len(evidence) if evidence else 0.0
    bridges: dict[tuple[str, str], dict[str, Any]] = {}
    entity_parents: dict[str, set[str]] = defaultdict(set)
    for chunk, match in matched:
        for entity_id in chunk["entity_ids"]:
            key = (match.parent_id, str(entity_id))
            bridge = bridges.setdefault(
                key,
                {
                    "parent_id": key[0],
                    "entity_id": key[1],
                    "source_chunk_ids": [],
                    "mapping_methods": [],
                    "mapping_scores": [],
                    "mapping_score": 0.0,
                },
            )
            bridge["source_chunk_ids"].append(str(chunk["chunk_id"]))
            bridge["mapping_methods"].append(match.method)
            bridge["mapping_scores"].append(match.similarity)
            bridge["mapping_score"] = max(float(bridge["mapping_score"]), match.similarity)
            entity_parents[str(entity_id)].add(match.parent_id)
    direct_mentions: list[EntityMention] = []
    directly_grounded_chunk_ids: set[str] = set()
    for source_uuid, chunks in evidence_by_source.items():
        mentions = direct_entity_mentions(parents_by_source.get(source_uuid, []), chunks)
        direct_mentions.extend(mentions)
        for mention in mentions:
            key = (mention.parent_id, mention.entity_id)
            bridge = bridges.setdefault(
                key,
                {
                    "parent_id": key[0],
                    "entity_id": key[1],
                    "source_chunk_ids": [],
                    "mapping_methods": [],
                    "mapping_scores": [],
                    "mapping_score": 0.0,
                },
            )
            for chunk_id in mention.source_chunk_ids:
                bridge["source_chunk_ids"].append(chunk_id)
                bridge["mapping_methods"].append("entity_exact_mention")
                bridge["mapping_scores"].append(1.0)
                directly_grounded_chunk_ids.add(chunk_id)
            bridge["mapping_score"] = 1.0
            entity_parents[mention.entity_id].add(mention.parent_id)
    text_mapped_chunk_ids = {str(chunk["chunk_id"]) for chunk, _ in matched}
    grounded_chunk_ids = text_mapped_chunk_ids | directly_grounded_chunk_ids
    grounding_coverage = len(grounded_chunk_ids) / len(evidence) if evidence else 0.0
    if write:
        for values in _batches(list(bridges.values()), batch_size):
            target.execute_query(
                """
                UNWIND $rows AS row
                MATCH (parent:ParentChunk {id: row.parent_id}), (entity:__Entity__ {id: row.entity_id})
                MERGE (parent)-[relationship:HAS_ENTITY]->(entity)
                SET relationship.source_chunk_ids = row.source_chunk_ids,
                    relationship.mapping_methods = row.mapping_methods,
                    relationship.mapping_scores = row.mapping_scores,
                    relationship.mapping_score = row.mapping_score
                """,
                rows=list(values),
                database_=database,
            )
    with target.session(database=database, fetch_size=1000) as session:
        relationship_rows = [
            row.data()
            for row in session.run(
                """
                MATCH (left:__Entity__)-[relationship]->(right:__Entity__)
                WHERE type(relationship) <> 'HAS_ENTITY'
                RETURN elementId(relationship) AS element_id, left.id AS left_id, right.id AS right_id
                """
            )
        ]
    grounded_relationships = []
    for row in relationship_rows:
        parents = sorted(entity_parents.get(str(row["left_id"]), set()) & entity_parents.get(str(row["right_id"]), set()))
        if parents:
            grounded_relationships.append({"element_id": row["element_id"], "source_parent_ids": parents})
    if write:
        for values in _batches(grounded_relationships, batch_size):
            target.execute_query(
                """
                UNWIND $rows AS row
                MATCH ()-[relationship]->() WHERE elementId(relationship) = row.element_id
                SET relationship.source_parent_ids = row.source_parent_ids
                """,
                rows=list(values),
                database_=database,
            )
    return {
        "evidence_chunks": len(evidence),
        "mapped_evidence_chunks": len(matched),
        "coverage": coverage,
        "text_mapping_coverage": coverage,
        "direct_entity_mentions": len(direct_mentions),
        "directly_grounded_evidence_chunks": len(directly_grounded_chunk_ids),
        "grounded_evidence_chunks": len(grounded_chunk_ids),
        "grounding_coverage": grounding_coverage,
        "unmapped_chunk_ids": unmapped,
        "source_coverage": {
            "evidence_source_uuids": len(evidence_by_source),
            "matched_source_uuids": len(matched_source_uuids),
            "missing_source_uuids": missing_source_uuids,
            "ambiguous_source_uuids": ambiguous_source_uuids,
            "evidence_without_source_uuid": evidence_without_source_uuid,
        },
        "unmapped_reason_counts": {
            "without_source_uuid": evidence_without_source_uuid,
            "missing_or_ambiguous_source": len(unmapped) - text_threshold_or_ambiguity,
            "text_threshold_or_ambiguous_tie": text_threshold_or_ambiguity,
        },
        "parent_entity_bridges": len(bridges),
        "grounded_entity_relationships": len(grounded_relationships),
    }


def _ensure_schema(target: Any, database: str, dimension: int) -> None:
    with target.session(database=database) as session:
        for query in STRUCTURAL_SCHEMA:
            session.run(query).consume()
        session.run(
            "CREATE FULLTEXT INDEX parent_keyword_v1 IF NOT EXISTS FOR (n:ParentChunk) ON EACH [n.text]"
        ).consume()
        session.run(
            f"""
            CREATE VECTOR INDEX parent_embedding_v1 IF NOT EXISTS
            FOR (n:ParentChunk) ON (n.embedding)
            OPTIONS {{indexConfig: {{`vector.dimensions`: {dimension}, `vector.similarity_function`: 'cosine'}}}}
            """
        ).consume()
        session.run(
            f"""
            CREATE VECTOR INDEX community_report_embedding_v1 IF NOT EXISTS
            FOR (n:CommunityReport) ON n.embedding
            OPTIONS {{indexConfig: {{
                `vector.dimensions`: {dimension},
                `vector.similarity_function`: 'cosine'
            }}}}
            """
        ).consume()
        session.run("DROP INDEX community_keyword IF EXISTS").consume()
        session.run("CALL db.awaitIndexes(1800)").consume()


def _fingerprint(rows: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda item: str(item["parent_id"])):
        digest.update(str(row["parent_id"]).encode())
        digest.update(json.dumps(row["parent"], sort_keys=True, default=str).encode())
    return digest.hexdigest()


def _projection_fingerprint(rows: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda item: str(item["parent_id"])):
        payload = {
            "document_id": str(row["document_id"]),
            "revision_id": str(row["revision_id"]),
            "parent_id": str(row["parent_id"]),
            "text": str(row["parent"].get("text") or ""),
            "position": row["parent"].get("position"),
        }
        digest.update(json.dumps(payload, sort_keys=True, default=str).encode())
    return digest.hexdigest()


def _verify_existing_candidate(
    target: Any,
    target_connection: ResolvedNeo4jConnection,
    manifest: CorpusManifest,
    source_rows: Sequence[Mapping[str, Any]],
    target_before: Mapping[str, int],
) -> dict[str, Any]:
    corpus, target_rows = _source_rows(target, target_connection.database, manifest.corpus_id)
    expected_key = f"{manifest.corpus_key}-aura-compact"
    if str(corpus.get("key") or "") != expected_key:
        raise CompactCorpusError(
            f"Existing target corpus key is {corpus.get('key')!r}, expected {expected_key!r}."
        )
    source_fingerprint = _projection_fingerprint(source_rows)
    target_fingerprint = _projection_fingerprint(target_rows)
    if source_fingerprint != target_fingerprint:
        raise CompactCorpusError("Existing compact candidate projection fingerprint does not match the source.")
    with target.session(database=target_connection.database) as session:
        counts = session.run(
            """
            MATCH (:Corpus {id: $corpus_id})-[:HAS_DOCUMENT]->(document:Document)
                  -[:ACTIVE_REVISION]->(revision:DocumentRevision)-[:HAS_PARENT]->(parent:ParentChunk)
            WITH count(DISTINCT document) AS documents, count(DISTINCT revision) AS revisions,
                 count(DISTINCT parent) AS parents,
                 count(DISTINCT CASE WHEN parent.embedding IS NOT NULL THEN parent END) AS embedded_parents
            OPTIONAL MATCH (:Corpus {id: $corpus_id})-[:HAS_DOCUMENT]->(:Document)
                  -[:ACTIVE_REVISION]->(:DocumentRevision)<-[:IN_REVISION]-(child:Chunk)
            RETURN documents, revisions, parents, embedded_parents, count(DISTINCT child) AS corpus_chunks
            """,
            corpus_id=manifest.corpus_id,
        ).single(strict=True)
        grounding = session.run(
            """
            MATCH (:Corpus {id: $corpus_id})-[:HAS_DOCUMENT]->(:Document)-[:ACTIVE_REVISION]->
                  (:DocumentRevision)-[:HAS_PARENT]->(parent:ParentChunk)
            OPTIONAL MATCH (parent)-[bridge:HAS_ENTITY]->(:__Entity__)
            WITH count(DISTINCT bridge) AS parent_entity_bridges
            MATCH (:__Entity__)-[relationship]->(:__Entity__)
            WHERE relationship.source_parent_ids IS NOT NULL
            RETURN parent_entity_bridges, count(relationship) AS grounded_entity_relationships
            """,
            corpus_id=manifest.corpus_id,
        ).single(strict=True)
    actual = {key: int(counts[key]) for key in counts.keys()}
    expected = {
        "documents": len({str(row["document_id"]) for row in source_rows}),
        "revisions": len({str(row["revision_id"]) for row in source_rows}),
        "parents": len(source_rows),
        "embedded_parents": len(source_rows),
        "corpus_chunks": 0,
    }
    if actual != expected:
        raise CompactCorpusError(f"Existing compact candidate counts are invalid: expected={expected}, actual={actual}.")
    verify_corpus_connection(
        target_connection,
        dimension=manifest.embedding_dimension,
        retrieval_unit="parent",
        vector_index="parent_embedding_v1",
        keyword_index="parent_keyword_v1",
    )
    return {
        "source_corpus_id": manifest.corpus_id,
        "source_corpus_key": manifest.corpus_key,
        "target_corpus_key": expected_key,
        "source_documents": expected["documents"],
        "source_parents": expected["parents"],
        "source_projection_fingerprint": source_fingerprint,
        "target_projection_fingerprint": target_fingerprint,
        "target_after": dict(target_before),
        "candidate_counts": actual,
        "grounding": {key: int(grounding[key]) for key in grounding.keys()},
        "existing_verified": True,
        "verified": True,
        "execute": True,
    }


def _duplicate_chunk_stats(source: Any, database: str, corpus_id: str) -> dict[str, int]:
    with source.session(database=database) as session:
        row = session.run(
            """
            MATCH (:Corpus {id: $corpus_id})-[:HAS_DOCUMENT]->(:Document)-[:ACTIVE_REVISION]->
                  (:DocumentRevision)<-[:IN_REVISION]-(chunk:Chunk)
            WITH chunk.text AS text, count(*) AS occurrences
            WHERE occurrences > 1
            RETURN count(*) AS groups, coalesce(sum(occurrences), 0) AS chunks,
                   coalesce(sum(occurrences - 1), 0) AS removable
            """,
            corpus_id=corpus_id,
        ).single(strict=True)
    return {key: int(row[key]) for key in ("groups", "chunks", "removable")}


def compact_corpus(
    source_connection: ResolvedNeo4jConnection,
    target_connection: ResolvedNeo4jConnection,
    manifest: CorpusManifest,
    *,
    execute: bool,
    batch_size: int = 200,
    minimum_bridge_coverage: float = 0.95,
    minimum_grounding_coverage: float = 0.90,
    minimum_text_coverage: float = 0.75,
    minimum_text_similarity: float = 0.60,
    maximum_nodes: int = 200_000,
    maximum_relationships: int = 400_000,
    capacity_headroom: float = 0.25,
) -> dict[str, Any]:
    if maximum_nodes <= 0 or maximum_relationships <= 0:
        raise CompactCorpusError("Capacity limits must be greater than zero.")
    if not 0.0 <= capacity_headroom < 1.0:
        raise CompactCorpusError("Capacity headroom must be in [0, 1).")
    if (source_connection.uri, source_connection.database) == (target_connection.uri, target_connection.database):
        raise CompactCorpusError("Source and target must be different databases.")
    started = time.monotonic()
    with GraphDatabase.driver(source_connection.uri, auth=(source_connection.username, source_connection.password)) as source:
        with GraphDatabase.driver(target_connection.uri, auth=(target_connection.username, target_connection.password)) as target:
            source.verify_connectivity()
            target.verify_connectivity()
            target_before = _inventory(target, target_connection.database)
            corpus_properties, source_rows = _source_rows(source, source_connection.database, manifest.corpus_id)
            with target.session(database=target_connection.database) as session:
                target_has_corpus = bool(session.run(
                    "MATCH (corpus:Corpus {id: $corpus_id}) RETURN count(corpus) AS count",
                    corpus_id=manifest.corpus_id,
                ).single(strict=True)["count"])
            if target_has_corpus:
                if not execute:
                    raise CompactCorpusError(
                        "Target already contains the source corpus; use --execute with exact target confirmation "
                        "to verify the existing candidate."
                    )
                return _verify_existing_candidate(
                    target,
                    target_connection,
                    manifest,
                    source_rows,
                    target_before,
                )
            target_corpus_key = f"{manifest.corpus_key}-aura-compact"
            target_corpus_properties = {
                **corpus_properties,
                "key": target_corpus_key,
                "title": f"{manifest.title} Aura Compact",
                "projection": "parent",
                "projection_of_corpus_id": manifest.corpus_id,
                "projection_of_corpus_key": manifest.corpus_key,
                "schema_version": GRAPH_SCHEMA_VERSION,
            }
            document_ids = {str(row["document_id"]) for row in source_rows}
            parent_ids = [str(row["parent_id"]) for row in source_rows]
            ready_sources = sum(1 for source_entry in manifest.sources.values() if source_entry.status.lower() == "ready")
            if len(document_ids) != ready_sources:
                raise CompactCorpusError(
                    f"Active document count {len(document_ids)} does not match {ready_sources} ready manifest sources."
                )
            bridge_preflight = _build_bridges(
                target,
                target_connection.database,
                source_rows,
                minimum_coverage=minimum_text_coverage,
                minimum_similarity=minimum_text_similarity,
                batch_size=batch_size,
                write=False,
            )
            embedding_coverage = _embedding_coverage(
                source, source_connection.database, manifest.corpus_id
            )
            embeddings_complete = (
                embedding_coverage["parents"] == len(parent_ids)
                and embedding_coverage["complete_parents"] == len(parent_ids)
                and embedding_coverage["children"] == embedding_coverage["embedded_children"]
            )
            projected_target = {
                "nodes": target_before["nodes"] + 1 + len(document_ids) * 2 + len(parent_ids),
                "relationships": (
                    target_before["relationships"]
                    + len(document_ids) * 3
                    + len(parent_ids)
                    + int(bridge_preflight["parent_entity_bridges"])
                ),
            }
            capacity_limits = {
                "nodes": math.floor(maximum_nodes * (1.0 - capacity_headroom)),
                "relationships": math.floor(maximum_relationships * (1.0 - capacity_headroom)),
            }
            capacity_gate_pass = all(
                projected_target[key] <= capacity_limits[key] for key in ("nodes", "relationships")
            )
            summary: dict[str, Any] = {
                "source_corpus_id": manifest.corpus_id,
                "source_corpus_key": manifest.corpus_key,
                "target_corpus_key": target_corpus_key,
                "source_documents": len(document_ids),
                "source_parents": len(parent_ids),
                "source_fingerprint": _fingerprint(source_rows),
                "exact_duplicate_chunks": _duplicate_chunk_stats(
                    source, source_connection.database, manifest.corpus_id
                ),
                "bridge_preflight": bridge_preflight,
                "text_mapping_gate_pass": bridge_preflight["coverage"] >= minimum_bridge_coverage,
                "bridge_gate_pass": bridge_preflight["grounding_coverage"] >= minimum_grounding_coverage,
                "embedding_coverage": embedding_coverage,
                "embedding_gate_pass": embeddings_complete,
                "gate_thresholds": {
                    "minimum_bridge_coverage": minimum_bridge_coverage,
                    "minimum_grounding_coverage": minimum_grounding_coverage,
                    "minimum_text_coverage": minimum_text_coverage,
                    "minimum_text_similarity": minimum_text_similarity,
                    "capacity_headroom": capacity_headroom,
                },
                "projected_target": projected_target,
                "capacity_limits_with_headroom": capacity_limits,
                "capacity_gate_pass": capacity_gate_pass,
                "estimated_parent_text_bytes": sum(
                    len(str(row["parent"].get("text") or "").encode("utf-8")) for row in source_rows
                ),
                "estimated_embedding_bytes": len(parent_ids) * manifest.embedding_dimension * 4,
                "target_before": target_before,
                "execute": execute,
            }
            if not execute:
                return summary
            if not summary["bridge_gate_pass"]:
                message = (
                    f"Combined grounding coverage {bridge_preflight['grounding_coverage']:.2%} is below "
                    f"the required {minimum_grounding_coverage:.2%}; target was not modified."
                )
                summary["blocked_reason"] = message
                raise CompactCorpusGateError(message, summary)
            if not embeddings_complete:
                message = f"Source child embedding coverage is incomplete: {embedding_coverage}; target was not modified."
                summary["blocked_reason"] = message
                raise CompactCorpusGateError(message, summary)
            if not capacity_gate_pass:
                message = (
                    f"Projected target {projected_target} exceeds capacity limits with headroom {capacity_limits}; "
                    "target was not modified."
                )
                summary["blocked_reason"] = message
                raise CompactCorpusGateError(message, summary)
            embeddings = _parent_embeddings(
                source,
                source_connection.database,
                parent_ids,
                batch_size=batch_size,
            )
            if len(embeddings) != len(parent_ids):
                raise CompactCorpusError("Not every active parent received an embedding.")
            _ensure_schema(target, target_connection.database, manifest.embedding_dimension)
            _write_compact_structure(
                target,
                target_connection.database,
                target_corpus_properties,
                source_rows,
                embeddings,
                batch_size=batch_size,
            )
            bridge_report = _build_bridges(
                target,
                target_connection.database,
                source_rows,
                minimum_coverage=minimum_text_coverage,
                minimum_similarity=minimum_text_similarity,
                batch_size=batch_size,
                write=True,
            )
            if bridge_report["grounding_coverage"] < minimum_grounding_coverage:
                raise CompactCorpusError(
                    f"Combined grounding coverage {bridge_report['grounding_coverage']:.2%} is below "
                    f"the required {minimum_grounding_coverage:.2%}."
                )
            target.execute_query(
                "MATCH (:Corpus {id: $corpus_id})-[:HAS_DOCUMENT]->(:Document)-[:ACTIVE_REVISION]->"
                "(revision:DocumentRevision) SET revision.graph_ready = true",
                corpus_id=manifest.corpus_id,
                database_=target_connection.database,
            )
            verify_corpus_connection(
                target_connection,
                dimension=manifest.embedding_dimension,
                retrieval_unit="parent",
                vector_index="parent_embedding_v1",
                keyword_index="parent_keyword_v1",
            )
            target_after = _inventory(target, target_connection.database)
            if target_after != projected_target:
                raise CompactCorpusError(
                    f"Compact projection count mismatch: projected={projected_target}, actual={target_after}."
                )
            summary.update(
                {
                    "parent_embeddings": len(embeddings),
                    "bridge": bridge_report,
                    "target_after": target_after,
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                    "verified": True,
                }
            )
            return summary


def compact_manifest(
    source: CorpusManifest,
    target: ResolvedNeo4jConnection,
    *,
    password_env: str = "NEO4J_PASSWORD",
) -> CorpusManifest:
    return CorpusManifest(
        corpus_id=source.corpus_id,
        corpus_key=f"{source.corpus_key}-aura-compact",
        title=f"{source.title} Aura Compact",
        neo4j={
            "uri": target.uri,
            "username": target.username,
            "database": target.database,
            "deployment": target.deployment,
            "password_env": password_env,
        },
        dataset_root=source.dataset_root,
        sources=source.sources,
        removed_sources=source.removed_sources,
        suppressed_sources=source.suppressed_sources,
        embedding_provider=source.embedding_provider,
        embedding_model=source.embedding_model,
        embedding_dimension=source.embedding_dimension,
        embedding_normalized=source.embedding_normalized,
        retrieval_unit="parent",
        retrieval_vector_index="parent_embedding_v1",
        retrieval_keyword_index="parent_keyword_v1",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build an isolated parent-level Neo4j corpus projection.")
    for prefix in ("source", "target"):
        parser.add_argument(f"--{prefix}-uri", required=True)
        parser.add_argument(f"--{prefix}-user", default="neo4j")
        parser.add_argument(f"--{prefix}-database", default="neo4j")
        parser.add_argument(f"--{prefix}-password-env", default=f"NEO4J_{prefix.upper()}_PASSWORD")
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--target-manifest", type=Path)
    parser.add_argument("--report-path", type=Path)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--minimum-bridge-coverage", type=float, default=0.95)
    parser.add_argument("--minimum-grounding-coverage", type=float, default=0.90)
    parser.add_argument("--minimum-text-coverage", type=float, default=0.75)
    parser.add_argument("--minimum-text-similarity", type=float, default=0.60)
    parser.add_argument("--maximum-nodes", type=int, default=200_000)
    parser.add_argument("--maximum-relationships", type=int, default=400_000)
    parser.add_argument("--capacity-headroom", type=float, default=0.25)
    parser.add_argument("--confirm-target")
    parser.add_argument("--execute", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.batch_size <= 0:
        raise CompactCorpusError("--batch-size must be greater than zero.")
    source = _connection_from_args(args, "source")
    target = _connection_from_args(args, "target")
    expected_confirmation = f"{target.uri}|{target.database}"
    if args.execute and args.confirm_target != expected_confirmation:
        raise CompactCorpusError(f"--confirm-target must exactly match: {expected_confirmation}")
    manifest = load_manifest(args.source_manifest.resolve())
    if manifest is None:
        raise CompactCorpusError(f"Source manifest not found: {args.source_manifest}")
    gate_error: CompactCorpusGateError | None = None
    try:
        summary = compact_corpus(
            source,
            target,
            manifest,
            execute=args.execute,
            batch_size=args.batch_size,
            minimum_bridge_coverage=args.minimum_bridge_coverage,
            minimum_grounding_coverage=args.minimum_grounding_coverage,
            minimum_text_coverage=args.minimum_text_coverage,
            minimum_text_similarity=args.minimum_text_similarity,
            maximum_nodes=args.maximum_nodes,
            maximum_relationships=args.maximum_relationships,
            capacity_headroom=args.capacity_headroom,
        )
    except CompactCorpusGateError as exc:
        gate_error = exc
        summary = exc.report
    if args.execute and args.target_manifest:
        if gate_error is None:
            compact = compact_manifest(manifest, target, password_env=args.target_password_env)
            save_manifest(args.target_manifest.resolve(), compact)
    if args.report_path:
        args.report_path.resolve().parent.mkdir(parents=True, exist_ok=True)
        args.report_path.resolve().write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if gate_error is not None:
        print(str(gate_error), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
