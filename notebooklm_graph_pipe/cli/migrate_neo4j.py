from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator
from urllib.parse import urlsplit, urlunsplit

from neo4j import GraphDatabase

from notebooklm_graph_pipe.runtime.neo4j_connection import (
    Neo4jConnectionSpec,
    ResolvedNeo4jConnection,
    resolve_connection,
    validate_neo4j_uri,
    verify_connection,
    verify_corpus_connection,
)

MIGRATION_LABEL = "__LGP_MIGRATION__"
MIGRATION_ID = "_lgp_migration_id"
MIGRATION_CONSTRAINT = "__lgp_migration_id_unique"
DEFAULT_BATCH_SIZE = 1000
CORPUS_NODE_QUERIES = {
    "Corpus": "MATCH (n:Corpus {id: $corpus_id}) RETURN n.id AS id, properties(n) AS properties",
    "Document": (
        "MATCH (:Corpus {id: $corpus_id})-[:HAS_DOCUMENT]->(n:Document) "
        "RETURN n.id AS id, properties(n) AS properties"
    ),
    "DocumentRevision": (
        "MATCH (:Corpus {id: $corpus_id})-[:HAS_DOCUMENT]->(:Document)-[:HAS_REVISION]->(n:DocumentRevision) "
        "RETURN n.id AS id, properties(n) AS properties"
    ),
    "ParentChunk": (
        "MATCH (:Corpus {id: $corpus_id})-[:HAS_DOCUMENT]->(:Document)-[:HAS_REVISION]->"
        "(:DocumentRevision)-[:HAS_PARENT]->(n:ParentChunk) "
        "RETURN n.id AS id, properties(n) AS properties"
    ),
    "Chunk": (
        "MATCH (:Corpus {id: $corpus_id})-[:HAS_DOCUMENT]->(:Document)-[:HAS_REVISION]->"
        "(:DocumentRevision)<-[:IN_REVISION]-(n:Chunk) "
        "RETURN n.id AS id, properties(n) AS properties"
    ),
}
CORPUS_RELATIONSHIP_PATTERNS = (
    ("Corpus", "HAS_DOCUMENT", "Document"),
    ("Document", "HAS_REVISION", "DocumentRevision"),
    ("Document", "ACTIVE_REVISION", "DocumentRevision"),
    ("DocumentRevision", "HAS_PARENT", "ParentChunk"),
    ("ParentChunk", "HAS_CHILD", "Chunk"),
    ("Chunk", "IN_REVISION", "DocumentRevision"),
    ("Chunk", "PART_OF", "Document"),
    ("Chunk", "NEXT_CHUNK", "Chunk"),
)
CORPUS_RELATIONSHIP_QUERIES = {
    ("Corpus", "HAS_DOCUMENT", "Document"): (
        "MATCH (a:Corpus {id: $corpus_id})-[r:HAS_DOCUMENT]->(b:Document)"
    ),
    ("Document", "HAS_REVISION", "DocumentRevision"): (
        "MATCH (:Corpus {id: $corpus_id})-[:HAS_DOCUMENT]->(a:Document)-[r:HAS_REVISION]->(b:DocumentRevision)"
    ),
    ("Document", "ACTIVE_REVISION", "DocumentRevision"): (
        "MATCH (:Corpus {id: $corpus_id})-[:HAS_DOCUMENT]->(a:Document)-[r:ACTIVE_REVISION]->(b:DocumentRevision)"
    ),
    ("DocumentRevision", "HAS_PARENT", "ParentChunk"): (
        "MATCH (:Corpus {id: $corpus_id})-[:HAS_DOCUMENT]->(:Document)-[:HAS_REVISION]->"
        "(a:DocumentRevision)-[r:HAS_PARENT]->(b:ParentChunk)"
    ),
    ("ParentChunk", "HAS_CHILD", "Chunk"): (
        "MATCH (:Corpus {id: $corpus_id})-[:HAS_DOCUMENT]->(:Document)-[:HAS_REVISION]->"
        "(:DocumentRevision)-[:HAS_PARENT]->(a:ParentChunk)-[r:HAS_CHILD]->(b:Chunk)"
    ),
    ("Chunk", "IN_REVISION", "DocumentRevision"): (
        "MATCH (:Corpus {id: $corpus_id})-[:HAS_DOCUMENT]->(:Document)-[:HAS_REVISION]->"
        "(b:DocumentRevision)<-[r:IN_REVISION]-(a:Chunk)"
    ),
    ("Chunk", "PART_OF", "Document"): (
        "MATCH (:Corpus {id: $corpus_id})-[:HAS_DOCUMENT]->(b:Document)<-[r:PART_OF]-(a:Chunk)"
    ),
    ("Chunk", "NEXT_CHUNK", "Chunk"): (
        "MATCH (:Corpus {id: $corpus_id})-[:HAS_DOCUMENT]->(:Document)-[:HAS_REVISION]->"
        "(:DocumentRevision)<-[:IN_REVISION]-(a:Chunk)-[r:NEXT_CHUNK]->(b:Chunk)"
    ),
}
CORPUS_SCHEMA_NAMES = {
    "chunk_embedding_v1",
    "chunk_id_unique",
    "chunk_keyword_v1",
    "chunk_parent_id",
    "corpus_id_unique",
    "corpus_key_unique",
    "document_id_unique",
    "document_status",
    "entities",
    "community_keyword",
    "parent_chunk_id_unique",
    "revision_id_unique",
    "revision_ready",
}
CORPUS_SEMANTIC_INDEXES = {"chunk_embedding_v1", "chunk_keyword_v1", "entities", "community_keyword"}
LEGACY_CHUNK_INDEXES = {"vector", "keyword"}


class MigrationError(RuntimeError):
    pass


@dataclass(frozen=True)
class GraphInventory:
    nodes: int
    relationships: int
    labels: dict[str, int]
    relationship_types: dict[str, int]
    node_property_keys: dict[str, int] = field(default_factory=dict)
    relationship_property_keys: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class SchemaObject:
    kind: str
    name: str
    create_statement: str


@dataclass(frozen=True)
class CorpusInventory:
    corpora: int
    documents: int
    active_revisions: int
    chunks: int
    embedded_chunks: int
    active_chunks: int
    active_embedded_chunks: int
    parents: int = 0
    embedded_parents: int = 0
    active_parents: int = 0
    active_embedded_parents: int = 0

    @property
    def present(self) -> bool:
        return self.corpora > 0

    def to_dict(self) -> dict[str, int]:
        return {
            "corpora": self.corpora,
            "documents": self.documents,
            "active_revisions": self.active_revisions,
            "chunks": self.chunks,
            "embedded_chunks": self.embedded_chunks,
            "active_chunks": self.active_chunks,
            "active_embedded_chunks": self.active_embedded_chunks,
            "parents": self.parents,
            "embedded_parents": self.embedded_parents,
            "active_parents": self.active_parents,
            "active_embedded_parents": self.active_embedded_parents,
        }


@dataclass(frozen=True)
class CorpusMergeInventory:
    nodes: dict[str, int]
    relationships: dict[str, int]
    embedded_chunks: int

    @property
    def total_nodes(self) -> int:
        return sum(self.nodes.values())

    @property
    def total_relationships(self) -> int:
        return sum(self.relationships.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": self.nodes,
            "relationships": self.relationships,
            "embedded_chunks": self.embedded_chunks,
            "total_nodes": self.total_nodes,
            "total_relationships": self.total_relationships,
        }


def quote_token(value: str) -> str:
    return f"`{value.replace('`', '``')}`"


def batched(rows: Iterable[dict[str, Any]], size: int) -> Iterator[list[dict[str, Any]]]:
    batch: list[dict[str, Any]] = []
    for row in rows:
        batch.append(row)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def read_inventory(driver: Any, database: str) -> GraphInventory:
    with driver.session(database=database) as session:
        counts = session.run(
            "MATCH (n) WITH count(n) AS nodes OPTIONAL MATCH ()-[r]->() RETURN nodes, count(r) AS relationships"
        ).single(strict=True)
        labels = {
            str(row["label"]): int(row["count"])
            for row in session.run("MATCH (n) UNWIND labels(n) AS label RETURN label, count(*) AS count ORDER BY label")
        }
        relationship_types = {
            str(row["type"]): int(row["count"])
            for row in session.run("MATCH ()-[r]->() RETURN type(r) AS type, count(*) AS count ORDER BY type")
        }
        node_property_keys = {
            str(row["key"]): int(row["count"])
            for row in session.run("MATCH (n) UNWIND keys(n) AS key RETURN key, count(*) AS count ORDER BY key")
        }
        relationship_property_keys = {
            str(row["key"]): int(row["count"])
            for row in session.run(
                "MATCH ()-[r]->() UNWIND keys(r) AS key RETURN key, count(*) AS count ORDER BY key"
            )
        }
    return GraphInventory(
        int(counts["nodes"]),
        int(counts["relationships"]),
        labels,
        relationship_types,
        node_property_keys,
        relationship_property_keys,
    )


def read_corpus_inventory(driver: Any, database: str) -> CorpusInventory:
    queries = {
        "corpora": "MATCH (n:Corpus) RETURN count(n) AS count",
        "documents": "MATCH (n:Document) RETURN count(n) AS count",
        "active_revisions": "MATCH (:Document)-[:ACTIVE_REVISION]->(n:DocumentRevision) RETURN count(n) AS count",
        "chunks": "MATCH (n:Chunk) RETURN count(n) AS count",
        "embedded_chunks": "MATCH (n:Chunk) WHERE n.embedding IS NOT NULL RETURN count(n) AS count",
        "active_chunks": (
            "MATCH (:Document)-[:ACTIVE_REVISION]->(:DocumentRevision)<-[:IN_REVISION]-(n:Chunk) "
            "RETURN count(n) AS count"
        ),
        "active_embedded_chunks": (
            "MATCH (:Document)-[:ACTIVE_REVISION]->(:DocumentRevision)<-[:IN_REVISION]-(n:Chunk) "
            "WHERE n.embedding IS NOT NULL RETURN count(n) AS count"
        ),
        "parents": "MATCH (n:ParentChunk) RETURN count(n) AS count",
        "embedded_parents": "MATCH (n:ParentChunk) WHERE n.embedding IS NOT NULL RETURN count(n) AS count",
        "active_parents": (
            "MATCH (:Document)-[:ACTIVE_REVISION]->(:DocumentRevision)-[:HAS_PARENT]->(n:ParentChunk) "
            "RETURN count(n) AS count"
        ),
        "active_embedded_parents": (
            "MATCH (:Document)-[:ACTIVE_REVISION]->(:DocumentRevision)-[:HAS_PARENT]->(n:ParentChunk) "
            "WHERE n.embedding IS NOT NULL RETURN count(n) AS count"
        ),
    }
    with driver.session(database=database) as session:
        counts = {
            key: int(session.run(query).single(strict=True)["count"])
            for key, query in queries.items()
        }
    return CorpusInventory(**counts)


def read_corpus_merge_inventory(driver: Any, database: str, corpus_id: str) -> CorpusMergeInventory:
    node_counts: dict[str, int] = {}
    relationship_counts: dict[str, int] = {}
    with driver.session(database=database) as session:
        for label, query in CORPUS_NODE_QUERIES.items():
            match_query = query.rsplit(" RETURN ", 1)[0]
            node_counts[label] = int(
                session.run(
                    f"{match_query} RETURN count(n) AS count",
                    corpus_id=corpus_id,
                ).single(strict=True)["count"]
            )
        chunk_match = CORPUS_NODE_QUERIES["Chunk"].rsplit(" RETURN ", 1)[0]
        embedded_chunks = int(
            session.run(
                f"{chunk_match} RETURN count(CASE WHEN n.embedding IS NOT NULL THEN 1 END) AS count",
                corpus_id=corpus_id,
            ).single(strict=True)["count"]
        )
        for start_label, relationship_type, end_label in CORPUS_RELATIONSHIP_PATTERNS:
            key = f"{start_label}:{relationship_type}:{end_label}"
            relationship_counts[key] = int(
                session.run(
                    CORPUS_RELATIONSHIP_QUERIES[(start_label, relationship_type, end_label)]
                    + " RETURN count(r) AS count",
                    corpus_id=corpus_id,
                ).single(strict=True)["count"]
            )
    return CorpusMergeInventory(node_counts, relationship_counts, embedded_chunks)


def verify_corpus_inventory(
    expected: CorpusInventory,
    actual: CorpusInventory,
    *,
    retrieval_unit: str = "chunk",
) -> None:
    if expected != actual:
        raise MigrationError(
            "Corpus migration verification failed: "
            f"source={expected.to_dict()}, target={actual.to_dict()}."
        )
    if retrieval_unit not in {"chunk", "parent"}:
        raise MigrationError(f"Unsupported retrieval unit: {retrieval_unit}")
    active_units = actual.active_parents if retrieval_unit == "parent" else actual.active_chunks
    active_embedded_units = (
        actual.active_embedded_parents if retrieval_unit == "parent" else actual.active_embedded_chunks
    )
    if actual.present and active_units != active_embedded_units:
        unit_label = "parents" if retrieval_unit == "parent" else "chunks"
        raise MigrationError(
            f"Corpus migration verification failed: one or more active {unit_label} are missing embeddings."
        )


def inspect_corpus_connection(
    connection: ResolvedNeo4jConnection,
    *,
    retrieval_unit: str = "chunk",
    vector_index: str = "chunk_embedding_v1",
) -> tuple[CorpusInventory, int, bool]:
    with GraphDatabase.driver(connection.uri, auth=(connection.username, connection.password)) as driver:
        driver.verify_connectivity()
        inventory = read_corpus_inventory(driver, connection.database)
        schema_signature = read_schema_signature(driver, connection.database)
    dimension = corpus_vector_dimension(schema_signature, vector_index=vector_index)
    return inventory, dimension, inventory.present and vector_index in schema_signature


def read_schema_signature(driver: Any, database: str) -> dict[str, dict[str, Any]]:
    with driver.session(database=database) as session:
        constraints = {
            str(row["name"]): {
                "kind": "constraint",
                # Neo4j Community and Aura expose the same property-uniqueness
                # constraint with different type names.
                "type": (
                    "UNIQUENESS"
                    if str(row["type"]) == "NODE_PROPERTY_UNIQUENESS"
                    else str(row["type"])
                ),
                "entity_type": str(row["entityType"]),
                "labels_or_types": sorted(str(item) for item in row["labelsOrTypes"]),
                "properties": [str(item) for item in row["properties"]],
            }
            for row in session.run(
                "SHOW CONSTRAINTS YIELD name, type, entityType, labelsOrTypes, properties "
                "RETURN name, type, entityType, labelsOrTypes, properties"
            )
            if str(row["name"]) != MIGRATION_CONSTRAINT
        }
        indexes: dict[str, dict[str, Any]] = {}
        for row in session.run(
            "SHOW INDEXES YIELD name, type, entityType, labelsOrTypes, properties, options, owningConstraint "
            "WHERE owningConstraint IS NULL AND type <> 'LOOKUP' "
            "RETURN name, type, entityType, labelsOrTypes, properties, options"
        ):
            options = dict(row["options"] or {})
            index_config = dict(options.get("indexConfig") or {})
            if str(row["type"]) == "VECTOR":
                relevant_config = {
                    key: index_config[key]
                    for key in ("vector.dimensions", "vector.similarity_function")
                    if key in index_config
                }
            else:
                relevant_config = _normalized_value(index_config)
            indexes[str(row["name"])] = {
                "kind": "index",
                "type": str(row["type"]),
                "entity_type": str(row["entityType"]),
                "labels_or_types": sorted(str(item) for item in row["labelsOrTypes"]),
                "properties": [str(item) for item in row["properties"]],
                "index_config": relevant_config,
            }
    return {**constraints, **indexes}


def read_schema(driver: Any, database: str) -> list[SchemaObject]:
    with driver.session(database=database) as session:
        constraints = [
            SchemaObject("constraint", str(row["name"]), str(row["createStatement"]))
            for row in session.run(
                "SHOW CONSTRAINTS YIELD name, createStatement RETURN name, createStatement ORDER BY name"
            )
            if row["createStatement"]
        ]
        indexes = [
            SchemaObject("index", str(row["name"]), str(row["createStatement"]))
            for row in session.run(
                "SHOW INDEXES YIELD name, type, owningConstraint, createStatement "
                "WHERE owningConstraint IS NULL AND type <> 'LOOKUP' "
                "RETURN name, createStatement ORDER BY name"
            )
            if row["createStatement"]
        ]
    return constraints + indexes


def corpus_vector_dimension(
    schema_signature: Mapping[str, dict[str, Any]],
    *,
    vector_index: str = "chunk_embedding_v1",
) -> int:
    index = schema_signature.get(vector_index) or {}
    config = index.get("index_config") or {}
    return int(config.get("vector.dimensions") or 384)


def portable_schema_statement(statement: str) -> str:
    if re.match(r"\s*CREATE\s+VECTOR\s+INDEX\b", statement, flags=re.IGNORECASE):
        dimension = re.search(r"`?vector\.dimensions`?\s*:\s*(\d+)", statement, flags=re.IGNORECASE)
        similarity = re.search(
            r"`?vector\.similarity_function`?\s*:\s*['\"]([^'\"]+)['\"]",
            statement,
            flags=re.IGNORECASE,
        )
        if dimension and similarity:
            prefix = re.split(r"\s+OPTIONS\s+", statement, maxsplit=1, flags=re.IGNORECASE)[0]
            return (
                f"{prefix} OPTIONS {{indexConfig: {{`vector.dimensions`: {dimension.group(1)}, "
                f"`vector.similarity_function`: '{similarity.group(1)}'}}}}"
            )
    without_provider_only = re.sub(
        r"\s+OPTIONS\s+\{\s*indexProvider\s*:\s*'[^']+'\s*\}\s*$",
        "",
        statement,
        flags=re.IGNORECASE,
    )
    return re.sub(
        r",\s*indexProvider\s*:\s*'[^']+'(?=\s*\})",
        "",
        without_provider_only,
        flags=re.IGNORECASE,
    )


def ensure_migration_names_available(driver: Any, database: str) -> None:
    with driver.session(database=database) as session:
        collision = session.run(
            f"MATCH (n) WHERE $migration_label IN labels(n) OR n.{quote_token(MIGRATION_ID)} IS NOT NULL "
            "RETURN count(n) AS count",
            migration_label=MIGRATION_LABEL,
        ).single(strict=True)
        rel_collision = session.run(
            f"MATCH ()-[r]->() WHERE r.{quote_token(MIGRATION_ID)} IS NOT NULL RETURN count(r) AS count"
        ).single(strict=True)
    if int(collision["count"]) or int(rel_collision["count"]):
        raise MigrationError("Source graph already uses reserved migration label/property names.")


def read_staging_state(driver: Any, database: str) -> tuple[int, int, int]:
    with driver.session(database=database) as session:
        row = session.run(
            f"MATCH (n) WITH count(n) AS total_nodes, "
            f"count(CASE WHEN n:{quote_token(MIGRATION_LABEL)} THEN 1 END) AS staged_nodes "
            f"OPTIONAL MATCH ()-[r]->() RETURN total_nodes, staged_nodes, "
            f"count(CASE WHEN r IS NOT NULL AND r.{quote_token(MIGRATION_ID)} IS NULL THEN 1 END) "
            "AS unstaged_relationships"
        ).single(strict=True)
    return int(row["total_nodes"]), int(row["staged_nodes"]), int(row["unstaged_relationships"])


def has_staging_constraint(driver: Any, database: str) -> bool:
    with driver.session(database=database) as session:
        row = session.run(
            "SHOW CONSTRAINTS YIELD name WHERE name = $name RETURN count(*) AS count",
            name=MIGRATION_CONSTRAINT,
        ).single(strict=True)
    return bool(row["count"])


def clear_target(driver: Any, database: str) -> None:
    with driver.session(database=database) as session:
        constraints = [str(row["name"]) for row in session.run("SHOW CONSTRAINTS YIELD name RETURN name")]
        indexes = [
            str(row["name"])
            for row in session.run(
                "SHOW INDEXES YIELD name, type, owningConstraint "
                "WHERE owningConstraint IS NULL AND type <> 'LOOKUP' RETURN name"
            )
        ]
        for name in constraints:
            session.run(f"DROP CONSTRAINT {quote_token(name)} IF EXISTS").consume()
        for name in indexes:
            session.run(f"DROP INDEX {quote_token(name)} IF EXISTS").consume()
        session.run("MATCH (n) CALL (n) { DETACH DELETE n } IN TRANSACTIONS OF 1000 ROWS").consume()


def create_staging_constraint(driver: Any, database: str) -> None:
    statement = (
        f"CREATE CONSTRAINT {quote_token(MIGRATION_CONSTRAINT)} IF NOT EXISTS "
        f"FOR (n:{quote_token(MIGRATION_LABEL)}) REQUIRE n.{quote_token(MIGRATION_ID)} IS UNIQUE"
    )
    driver.execute_query(statement, database_=database)


def _node_rows(driver: Any, database: str) -> Iterator[dict[str, Any]]:
    with driver.session(database=database, fetch_size=DEFAULT_BATCH_SIZE) as session:
        for row in session.run(
            "MATCH (n) RETURN elementId(n) AS source_id, labels(n) AS labels, properties(n) AS properties"
        ):
            yield row.data()


def _relationship_rows(driver: Any, database: str) -> Iterator[dict[str, Any]]:
    with driver.session(database=database, fetch_size=DEFAULT_BATCH_SIZE) as session:
        for row in session.run(
            "MATCH (a)-[r]->(b) RETURN elementId(r) AS relationship_id, elementId(a) AS start_id, "
            "elementId(b) AS end_id, type(r) AS type, properties(r) AS properties"
        ):
            yield row.data()


def _corpus_node_rows(driver: Any, database: str, corpus_id: str, label: str) -> Iterator[dict[str, Any]]:
    with driver.session(database=database, fetch_size=DEFAULT_BATCH_SIZE) as session:
        for row in session.run(CORPUS_NODE_QUERIES[label], corpus_id=corpus_id):
            yield row.data()


def _corpus_node_id_rows(driver: Any, database: str, corpus_id: str, label: str) -> Iterator[dict[str, Any]]:
    query = CORPUS_NODE_QUERIES[label].rsplit(" RETURN ", 1)[0] + " RETURN n.id AS id"
    with driver.session(database=database, fetch_size=DEFAULT_BATCH_SIZE) as session:
        for row in session.run(query, corpus_id=corpus_id):
            yield row.data()


def _corpus_relationship_rows(
    driver: Any,
    database: str,
    corpus_id: str,
    start_label: str,
    relationship_type: str,
    end_label: str,
) -> Iterator[dict[str, Any]]:
    query = CORPUS_RELATIONSHIP_QUERIES[(start_label, relationship_type, end_label)]
    query += " RETURN a.id AS start_id, b.id AS end_id, properties(r) AS properties"
    with driver.session(database=database, fetch_size=DEFAULT_BATCH_SIZE) as session:
        for row in session.run(query, corpus_id=corpus_id):
            yield row.data()


def corpus_content_fingerprint(driver: Any, database: str, corpus_id: str) -> dict[str, tuple[int, str]]:
    result: dict[str, tuple[int, str]] = {}
    for label in CORPUS_NODE_QUERIES:
        if label == "Chunk":
            match_query = CORPUS_NODE_QUERIES[label].rsplit(" RETURN ", 1)[0]
            query = (
                f"{match_query} RETURN n.id AS id, n{{.*, embedding: null}} AS properties, "
                "size(n.embedding) AS embedding_dimension, n.embedding[0] AS embedding_first, "
                "n.embedding[-1] AS embedding_last, "
                "reduce(total = 0.0, value IN n.embedding | total + value) AS embedding_sum, "
                "reduce(total = 0.0, index IN range(0, size(n.embedding) - 1) | "
                "total + (index + 1) * n.embedding[index]) AS embedding_weighted_sum"
            )

            def chunk_rows() -> Iterator[dict[str, Any]]:
                with driver.session(database=database, fetch_size=DEFAULT_BATCH_SIZE) as session:
                    for row in session.run(query, corpus_id=corpus_id):
                        yield {"kind": f"node:{label}", **row.data()}

            rows = chunk_rows()
        else:
            rows = (
                {"kind": f"node:{label}", **row}
                for row in _corpus_node_rows(driver, database, corpus_id, label)
            )
        result[f"node:{label}"] = _fingerprint_rows(rows)
    for start_label, relationship_type, end_label in CORPUS_RELATIONSHIP_PATTERNS:
        key = f"relationship:{start_label}:{relationship_type}:{end_label}"
        rows = (
            {"kind": key, **row}
            for row in _corpus_relationship_rows(
                driver,
                database,
                corpus_id,
                start_label,
                relationship_type,
                end_label,
            )
        )
        result[key] = _fingerprint_rows(rows)
    return result


def base_graph_content_fingerprint(driver: Any, database: str, corpus_id: str) -> dict[str, tuple[int, str]]:
    with driver.session(database=database) as session:
        corpus_exists = int(
            session.run(
                "MATCH (n:Corpus {id: $corpus_id}) RETURN count(n) AS count",
                corpus_id=corpus_id,
            ).single(strict=True)["count"]
        )
    if not corpus_exists:
        return graph_content_fingerprint(driver, database, staged=False)
    def donor_predicate(alias: str) -> str:
        return (
            f"({alias}:Corpus AND {alias}.id = $corpus_id) OR "
            f"({alias}:Document AND EXISTS {{ MATCH (:Corpus {{id: $corpus_id}})-[:HAS_DOCUMENT]->({alias}) }}) OR "
            f"{alias}:DocumentRevision OR {alias}:ParentChunk OR "
            f"({alias}:Chunk AND {alias}.revision_id IS NOT NULL)"
        )

    def node_rows() -> Iterator[dict[str, Any]]:
        with driver.session(database=database, fetch_size=DEFAULT_BATCH_SIZE) as session:
            for row in session.run(
                f"MATCH (candidate) WHERE NOT ({donor_predicate('candidate')}) "
                "RETURN elementId(candidate) AS source_id, labels(candidate) AS labels, "
                "properties(candidate) AS properties",
                corpus_id=corpus_id,
            ):
                yield row.data()

    def relationship_rows() -> Iterator[dict[str, Any]]:
        with driver.session(database=database, fetch_size=DEFAULT_BATCH_SIZE) as session:
            for row in session.run(
                f"MATCH (a)-[r]->(b) WHERE NOT ({donor_predicate('a')}) "
                f"AND NOT ({donor_predicate('b')}) "
                "RETURN labels(a) AS start_labels, properties(a) AS start_properties, "
                "labels(b) AS end_labels, properties(b) AS end_properties, "
                "type(r) AS type, properties(r) AS properties",
                corpus_id=corpus_id,
            ):
                yield row.data()

    return {
        "nodes": _fingerprint_rows(node_rows()),
        "relationships": _fingerprint_rows(relationship_rows()),
    }


def _normalized_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _normalized_value(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [_normalized_value(item) for item in value]
    if isinstance(value, bytes):
        return {"__bytes__": value.hex()}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    iso_format = getattr(value, "iso_format", None)
    if callable(iso_format):
        return {"__type__": type(value).__name__, "value": iso_format()}
    return {"__type__": type(value).__name__, "value": str(value)}


def _fingerprint_rows(rows: Iterable[dict[str, Any]]) -> tuple[int, str]:
    aggregate = 0
    count = 0
    for row in rows:
        normalized_row = dict(row)
        if isinstance(normalized_row.get("labels"), list):
            normalized_row["labels"] = sorted(str(label) for label in normalized_row["labels"])
        payload = json.dumps(
            _normalized_value(normalized_row), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        aggregate ^= int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest(), "big")
        count += 1
    return count, f"{aggregate:064x}"


def graph_content_fingerprint(driver: Any, database: str, *, staged: bool) -> dict[str, tuple[int, str]]:
    if not staged:
        return {
            "nodes": _fingerprint_rows(_node_rows(driver, database)),
            "relationships": _fingerprint_rows(_relationship_rows(driver, database)),
        }

    def staged_nodes() -> Iterator[dict[str, Any]]:
        with driver.session(database=database, fetch_size=DEFAULT_BATCH_SIZE) as session:
            for row in session.run(
                f"MATCH (n:{quote_token(MIGRATION_LABEL)}) RETURN "
                f"n.{quote_token(MIGRATION_ID)} AS source_id, "
                f"[label IN labels(n) WHERE label <> $migration_label] AS labels, "
                f"properties(n) AS properties",
                migration_label=MIGRATION_LABEL,
            ):
                data = row.data()
                data["properties"].pop(MIGRATION_ID, None)
                yield data

    def staged_relationships() -> Iterator[dict[str, Any]]:
        with driver.session(database=database, fetch_size=DEFAULT_BATCH_SIZE) as session:
            for row in session.run(
                f"MATCH (a:{quote_token(MIGRATION_LABEL)})-[r]->(b:{quote_token(MIGRATION_LABEL)}) RETURN "
                f"r.{quote_token(MIGRATION_ID)} AS relationship_id, "
                f"a.{quote_token(MIGRATION_ID)} AS start_id, b.{quote_token(MIGRATION_ID)} AS end_id, "
                "type(r) AS type, properties(r) AS properties"
            ):
                data = row.data()
                data["properties"].pop(MIGRATION_ID, None)
                yield data

    return {
        "nodes": _fingerprint_rows(staged_nodes()),
        "relationships": _fingerprint_rows(staged_relationships()),
    }


def copy_nodes(source: Any, target: Any, source_database: str, target_database: str, batch_size: int) -> int:
    copied = 0
    for raw_batch in batched(_node_rows(source, source_database), batch_size):
        grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
        for row in raw_batch:
            grouped[tuple(sorted(str(label) for label in row["labels"]))].append(row)
        for labels, rows in grouped.items():
            label_clause = "".join(f":{quote_token(label)}" for label in labels)
            query = (
                f"UNWIND $rows AS row MERGE (n:{quote_token(MIGRATION_LABEL)} "
                f"{{{quote_token(MIGRATION_ID)}: row.source_id}}) "
                f"SET n += row.properties"
            )
            if label_clause:
                query += f" SET n{label_clause}"
            target.execute_query(query, rows=rows, database_=target_database)
            copied += len(rows)
    return copied


def copy_relationships(source: Any, target: Any, source_database: str, target_database: str, batch_size: int) -> int:
    copied = 0
    for raw_batch in batched(_relationship_rows(source, source_database), batch_size):
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in raw_batch:
            grouped[str(row["type"])].append(row)
        for relationship_type, rows in grouped.items():
            query = (
                f"UNWIND $rows AS row MATCH (a:{quote_token(MIGRATION_LABEL)} "
                f"{{{quote_token(MIGRATION_ID)}: row.start_id}}), "
                f"(b:{quote_token(MIGRATION_LABEL)} {{{quote_token(MIGRATION_ID)}: row.end_id}}) "
                f"MERGE (a)-[r:{quote_token(relationship_type)} "
                f"{{{quote_token(MIGRATION_ID)}: row.relationship_id}}]->(b) SET r += row.properties"
            )
            target.execute_query(query, rows=rows, database_=target_database)
            copied += len(rows)
    return copied


def recreate_schema(driver: Any, database: str, schema: list[SchemaObject]) -> list[str]:
    created: list[str] = []
    existing = set(read_schema_signature(driver, database))
    for item in sorted(schema, key=lambda value: value.kind != "constraint"):
        if item.name in existing:
            created.append(item.name)
            continue
        try:
            driver.execute_query(portable_schema_statement(item.create_statement), database_=database)
        except Exception as exc:
            raise MigrationError(f"Failed to recreate {item.kind} '{item.name}': {exc}") from exc
        created.append(item.name)
    if any(item.kind == "index" for item in schema):
        driver.execute_query("CALL db.awaitIndexes(300)", database_=database)
    return created


def _corpus_schema(driver: Any, database: str) -> list[SchemaObject]:
    return [item for item in read_schema(driver, database) if item.name in CORPUS_SCHEMA_NAMES]


def _assert_corpus_node_compatible(
    target: Any,
    database: str,
    label: str,
    rows: list[dict[str, Any]],
) -> None:
    if not rows:
        return
    with target.session(database=database) as session:
        existing = {
            str(row["id"]): dict(row["properties"])
            for row in session.run(
                f"UNWIND $ids AS id MATCH (n:{quote_token(label)} {{id: id}}) "
                "RETURN n.id AS id, properties(n) AS properties",
                ids=[row["id"] for row in rows],
            )
        }
    for row in rows:
        current = existing.get(str(row["id"]))
        if current is not None and _normalized_value(current) != _normalized_value(dict(row["properties"])):
            raise MigrationError(
                f"Corpus merge identifier collision for {label}.id={row['id']!r}; "
                "the target node has different properties."
            )


def _assert_no_corpus_id_collisions(
    source: Any,
    target: Any,
    source_database: str,
    target_database: str,
    corpus_id: str,
    batch_size: int,
) -> None:
    with target.session(database=target_database) as session:
        target_labels = {str(row["label"]) for row in session.run("CALL db.labels() YIELD label RETURN label")}
        for label in CORPUS_NODE_QUERIES:
            if label not in target_labels:
                continue
            target_ids = {
                str(row["id"])
                for row in session.run(
                    f"MATCH (n:{quote_token(label)}) WHERE n.id IS NOT NULL RETURN n.id AS id"
                )
            }
            for rows in batched(_corpus_node_id_rows(source, source_database, corpus_id, label), batch_size):
                collision = next((str(row["id"]) for row in rows if str(row["id"]) in target_ids), None)
                if collision:
                    raise MigrationError(
                        f"Corpus merge identifier collision for {label}.id={collision!r}; "
                        "the target must not already contain donor identifiers on a fresh merge."
                    )


def _copy_corpus_nodes(
    source: Any,
    target: Any,
    source_database: str,
    target_database: str,
    corpus_id: str,
    label: str,
    batch_size: int,
) -> int:
    copied = 0
    for rows in batched(_corpus_node_rows(source, source_database, corpus_id, label), batch_size):
        _assert_corpus_node_compatible(target, target_database, label, rows)
        target.execute_query(
            f"UNWIND $rows AS row MERGE (n:{quote_token(label)} {{id: row.id}}) SET n = row.properties",
            rows=rows,
            database_=target_database,
        )
        copied += len(rows)
    return copied


def _copy_corpus_relationships(
    source: Any,
    target: Any,
    source_database: str,
    target_database: str,
    corpus_id: str,
    start_label: str,
    relationship_type: str,
    end_label: str,
    batch_size: int,
) -> int:
    copied = 0
    rows_iter = _corpus_relationship_rows(
        source,
        source_database,
        corpus_id,
        start_label,
        relationship_type,
        end_label,
    )
    for rows in batched(rows_iter, batch_size):
        result = target.execute_query(
            f"UNWIND $rows AS row MATCH (a:{quote_token(start_label)} {{id: row.start_id}}), "
            f"(b:{quote_token(end_label)} {{id: row.end_id}}) "
            f"MERGE (a)-[r:{quote_token(relationship_type)}]->(b) SET r = row.properties "
            "RETURN count(r) AS count",
            rows=rows,
            database_=target_database,
        )
        records = result.records if hasattr(result, "records") else result[0]
        matched = int(records[0]["count"]) if records else 0
        if matched != len(rows):
            raise MigrationError(
                f"Corpus merge could only match {matched} of {len(rows)} {relationship_type} relationships."
            )
        copied += len(rows)
    return copied


def _prepare_corpus_schema(target: Any, database: str, schema: list[SchemaObject]) -> list[str]:
    structural = [item for item in schema if item.name not in CORPUS_SEMANTIC_INDEXES]
    return recreate_schema(target, database, structural)


def _activate_corpus_search_indexes(target: Any, database: str, schema: list[SchemaObject]) -> list[str]:
    existing = read_schema_signature(target, database)
    expected = {item.name: item for item in schema if item.name in CORPUS_SEMANTIC_INDEXES}
    with target.session(database=database) as session:
        for legacy_name in LEGACY_CHUNK_INDEXES:
            if legacy_name in existing:
                session.run(f"DROP INDEX {quote_token(legacy_name)} IF EXISTS").consume()
        if "entities" in existing:
            session.run(f"DROP INDEX {quote_token('entities')} IF EXISTS").consume()
    created = recreate_schema(
        target,
        database,
        list(expected.values()),
    )
    target.execute_query("CALL db.awaitIndexes(1800)", database_=database)
    return created


def _read_merge_state(
    path: Path,
    *,
    corpus_id: str,
    source_uri: str,
    source_database: str,
    target_uri: str,
    target_database: str,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "corpus_id": corpus_id,
        "source_uri": source_uri,
        "source_database": source_database,
        "target_uri": target_uri,
        "target_database": target_database,
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise MigrationError("Corpus merge state does not match the requested source, target, and corpus.")
    payload.setdefault("completed_stages", [])
    return payload


def _write_merge_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def corpus_merge(
    source_connection: ResolvedNeo4jConnection,
    target_connection: ResolvedNeo4jConnection,
    *,
    corpus_id: str,
    state_path: Path,
    batch_size: int = DEFAULT_BATCH_SIZE,
    execute: bool = False,
    resume: bool = False,
) -> dict[str, Any]:
    if source_connection.uri == target_connection.uri and source_connection.database == target_connection.database:
        raise MigrationError("Source and target resolve to the same Neo4j database.")
    with GraphDatabase.driver(source_connection.uri, auth=(source_connection.username, source_connection.password)) as source:
        with GraphDatabase.driver(target_connection.uri, auth=(target_connection.username, target_connection.password)) as target:
            source.verify_connectivity()
            target.verify_connectivity()
            source_inventory = read_corpus_merge_inventory(source, source_connection.database, corpus_id)
            if source_inventory.nodes.get("Corpus") != 1:
                raise MigrationError(f"Expected exactly one donor Corpus with id {corpus_id!r}.")
            if source_inventory.nodes.get("Chunk") != source_inventory.embedded_chunks:
                raise MigrationError("Donor corpus contains one or more chunks without embeddings.")
            target_before = read_inventory(target, target_connection.database)
            schema = _corpus_schema(source, source_connection.database)
            missing_schema = CORPUS_SCHEMA_NAMES - {item.name for item in schema}
            if missing_schema:
                raise MigrationError(f"Donor corpus is missing required schema: {sorted(missing_schema)}")
            summary: dict[str, Any] = {
                "mode": "corpus-merge",
                "executed": execute,
                "resume": resume,
                "corpus_id": corpus_id,
                "source_corpus": source_inventory.to_dict(),
                "target_before": {
                    "nodes": target_before.nodes,
                    "relationships": target_before.relationships,
                },
                "projected_target": {
                    "nodes": target_before.nodes + source_inventory.total_nodes,
                    "relationships": target_before.relationships + source_inventory.total_relationships,
                },
                "state_path": str(state_path),
            }
            if not execute:
                _assert_no_corpus_id_collisions(
                    source,
                    target,
                    source_connection.database,
                    target_connection.database,
                    corpus_id,
                    batch_size,
                )
                summary["identifier_collisions"] = 0
                return summary

            state = {
                "corpus_id": corpus_id,
                "source_uri": source_connection.uri,
                "source_database": source_connection.database,
                "target_uri": target_connection.uri,
                "target_database": target_connection.database,
                "target_initial": {
                    "nodes": target_before.nodes,
                    "relationships": target_before.relationships,
                },
                "completed_stages": [],
                "status": "running",
            }
            fresh_merge = not resume and not state_path.exists()
            if resume:
                if not state_path.exists():
                    raise MigrationError("--resume requires an existing corpus merge state file.")
                state = _read_merge_state(
                    state_path,
                    corpus_id=corpus_id,
                    source_uri=source_connection.uri,
                    source_database=source_connection.database,
                    target_uri=target_connection.uri,
                    target_database=target_connection.database,
                )
            elif state_path.exists():
                existing_state = _read_merge_state(
                    state_path,
                    corpus_id=corpus_id,
                    source_uri=source_connection.uri,
                    source_database=source_connection.database,
                    target_uri=target_connection.uri,
                    target_database=target_connection.database,
                )
                if existing_state.get("status") != "completed":
                    raise MigrationError("An incomplete corpus merge exists; rerun with --resume.")
                state = existing_state
            initial = state["target_initial"]
            summary["projected_target"] = {
                "nodes": int(initial["nodes"]) + source_inventory.total_nodes,
                "relationships": int(initial["relationships"]) + source_inventory.total_relationships,
            }
            if fresh_merge:
                _assert_no_corpus_id_collisions(
                    source,
                    target,
                    source_connection.database,
                    target_connection.database,
                    corpus_id,
                    batch_size,
                )
                state["source_fingerprint"] = corpus_content_fingerprint(
                    source, source_connection.database, corpus_id
                )
                state["base_fingerprint"] = base_graph_content_fingerprint(
                    target, target_connection.database, corpus_id
                )
            _write_merge_state(state_path, state)

            completed = set(state["completed_stages"])

            def complete(stage: str) -> None:
                completed.add(stage)
                state["completed_stages"] = sorted(completed)
                _write_merge_state(state_path, state)

            if "schema-structural" not in completed:
                summary["schema_created"] = _prepare_corpus_schema(target, target_connection.database, schema)
                complete("schema-structural")

            copied_nodes: dict[str, int] = {}
            for label in CORPUS_NODE_QUERIES:
                stage = f"nodes:{label}"
                if stage not in completed:
                    copied_nodes[label] = _copy_corpus_nodes(
                        source,
                        target,
                        source_connection.database,
                        target_connection.database,
                        corpus_id,
                        label,
                        batch_size,
                    )
                    complete(stage)
            summary["nodes_processed"] = copied_nodes

            copied_relationships: dict[str, int] = {}
            for start_label, relationship_type, end_label in CORPUS_RELATIONSHIP_PATTERNS:
                key = f"{start_label}:{relationship_type}:{end_label}"
                stage = f"relationships:{key}"
                if stage not in completed:
                    copied_relationships[key] = _copy_corpus_relationships(
                        source,
                        target,
                        source_connection.database,
                        target_connection.database,
                        corpus_id,
                        start_label,
                        relationship_type,
                        end_label,
                        batch_size,
                    )
                    complete(stage)
            summary["relationships_processed"] = copied_relationships

            if "schema-search" not in completed:
                summary["search_indexes_created"] = _activate_corpus_search_indexes(
                    target, target_connection.database, schema
                )
                complete("schema-search")

            target_corpus = read_corpus_merge_inventory(target, target_connection.database, corpus_id)
            if target_corpus != source_inventory:
                raise MigrationError(
                    "Corpus merge verification failed: "
                    f"source={source_inventory.to_dict()}, target={target_corpus.to_dict()}."
                )
            target_fingerprint = corpus_content_fingerprint(target, target_connection.database, corpus_id)
            if _normalized_value(target_fingerprint) != _normalized_value(state.get("source_fingerprint")):
                raise MigrationError("Corpus merge content fingerprint does not match the donor corpus.")
            base_fingerprint = base_graph_content_fingerprint(target, target_connection.database, corpus_id)
            if _normalized_value(base_fingerprint) != _normalized_value(state.get("base_fingerprint")):
                raise MigrationError("Corpus merge modified pre-existing graph content.")
            target_after = read_inventory(target, target_connection.database)
            initial = state["target_initial"]
            expected_nodes = int(initial["nodes"]) + source_inventory.total_nodes
            expected_relationships = int(initial["relationships"]) + source_inventory.total_relationships
            if target_after.nodes != expected_nodes or target_after.relationships != expected_relationships:
                raise MigrationError(
                    "Corpus merge changed an unexpected number of graph elements: "
                    f"expected {expected_nodes}/{expected_relationships}, "
                    f"got {target_after.nodes}/{target_after.relationships}."
                )
            verify_corpus_connection(target_connection, dimension=corpus_vector_dimension(read_schema_signature(target, target_connection.database)))
            state["status"] = "completed"
            _write_merge_state(state_path, state)
            summary["target_corpus"] = target_corpus.to_dict()
            summary["content_fingerprint"] = target_fingerprint
            summary["base_graph_preserved"] = True
            summary["target_after"] = {
                "nodes": target_after.nodes,
                "relationships": target_after.relationships,
            }
            summary["verified"] = True
            return summary


def cleanup_staging(driver: Any, database: str) -> None:
    with driver.session(database=database) as session:
        session.run(
            f"MATCH ()-[r]->() WHERE r.{quote_token(MIGRATION_ID)} IS NOT NULL "
            f"CALL (r) {{ REMOVE r.{quote_token(MIGRATION_ID)} }} IN TRANSACTIONS OF 1000 ROWS"
        ).consume()
        session.run(
            f"MATCH (n:{quote_token(MIGRATION_LABEL)}) CALL (n) {{ "
            f"REMOVE n.{quote_token(MIGRATION_ID)}, n:{quote_token(MIGRATION_LABEL)} "
            "} IN TRANSACTIONS OF 1000 ROWS"
        ).consume()
    driver.execute_query(
        f"DROP CONSTRAINT {quote_token(MIGRATION_CONSTRAINT)} IF EXISTS",
        database_=database,
    )


def verify_inventory(expected: GraphInventory, actual: GraphInventory) -> None:
    if expected != actual:
        raise MigrationError(
            "Migration verification failed: "
            f"source={expected.nodes} nodes/{expected.relationships} relationships, "
            f"target={actual.nodes} nodes/{actual.relationships} relationships."
        )


def verify_entity_counts(expected: GraphInventory, actual: GraphInventory) -> None:
    if expected.nodes != actual.nodes or expected.relationships != actual.relationships:
        raise MigrationError(
            "Migration verification failed before staging cleanup: "
            f"source={expected.nodes} nodes/{expected.relationships} relationships, "
            f"target={actual.nodes} nodes/{actual.relationships} relationships."
        )


def complete_staging_resume_ready(
    *,
    resume: bool,
    staging_constraint: bool,
    source_inventory: GraphInventory,
    target_inventory: GraphInventory,
    total_target_nodes: int,
    staged_target_nodes: int,
    unstaged_target_relationships: int,
    source_schema: Mapping[str, dict[str, Any]],
    target_schema: Mapping[str, dict[str, Any]],
) -> bool:
    return bool(
        resume
        and staging_constraint
        and total_target_nodes == staged_target_nodes == source_inventory.nodes
        and target_inventory.relationships == source_inventory.relationships
        and unstaged_target_relationships == 0
        and source_schema == target_schema
    )


def _portable_migrate_impl(
    source_connection: ResolvedNeo4jConnection,
    target_connection: ResolvedNeo4jConnection,
    progress: dict[str, Any],
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    execute: bool = False,
    overwrite_target: bool = False,
    resume: bool = False,
) -> dict[str, Any]:
    progress["phase"] = "connect"
    with GraphDatabase.driver(source_connection.uri, auth=(source_connection.username, source_connection.password)) as source:
        with GraphDatabase.driver(target_connection.uri, auth=(target_connection.username, target_connection.password)) as target:
            source.verify_connectivity()
            target.verify_connectivity()
            source_server = source.get_server_info() if hasattr(source, "get_server_info") else None
            target_server = target.get_server_info() if hasattr(target, "get_server_info") else None
            if (
                source_server is not None
                and target_server is not None
                and str(source_server.address) == str(target_server.address)
                and source_connection.database == target_connection.database
            ):
                raise MigrationError("Source and target resolve to the same Neo4j server and database.")
            progress["phase"] = "inventory"
            inventory = read_inventory(source, source_connection.database)
            corpus_inventory = read_corpus_inventory(source, source_connection.database)
            target_inventory = read_inventory(target, target_connection.database)
            progress["source"] = {"nodes": inventory.nodes, "relationships": inventory.relationships}
            progress["target_before"] = {
                "nodes": target_inventory.nodes,
                "relationships": target_inventory.relationships,
            }
            schema = read_schema(source, source_connection.database)
            source_schema_signature = read_schema_signature(source, source_connection.database)
            corpus_dimension = corpus_vector_dimension(source_schema_signature)
            is_v3_corpus = corpus_inventory.present and "chunk_embedding_v1" in source_schema_signature
            target_schema = read_schema(target, target_connection.database)
            target_schema_signature = read_schema_signature(target, target_connection.database)
            ensure_migration_names_available(source, source_connection.database)
            total_target_nodes, staged_target_nodes, unstaged_target_relationships = read_staging_state(
                target, target_connection.database
            )
            staging_constraint = has_staging_constraint(target, target_connection.database)
            has_staging_state = staged_target_nodes > 0
            cleanup_only_resume = bool(
                resume
                and staging_constraint
                and (staged_target_nodes != total_target_nodes or unstaged_target_relationships)
            )
            verification_only_resume = complete_staging_resume_ready(
                resume=resume,
                staging_constraint=staging_constraint,
                source_inventory=inventory,
                target_inventory=target_inventory,
                total_target_nodes=total_target_nodes,
                staged_target_nodes=staged_target_nodes,
                unstaged_target_relationships=unstaged_target_relationships,
                source_schema=source_schema_signature,
                target_schema=target_schema_signature,
            )
            if resume:
                if not cleanup_only_resume and (
                    not has_staging_state
                    or total_target_nodes != staged_target_nodes
                    or unstaged_target_relationships
                ):
                    raise MigrationError(
                        "--resume requires a target containing only nodes from an interrupted portable migration."
                    )
            elif target_inventory.nodes or target_inventory.relationships or target_schema:
                if not overwrite_target:
                    raise MigrationError(
                        "Target database is not empty; it contains data or user schema. "
                        "Use explicit overwrite confirmation to replace it."
                    )
            summary = {
                "mode": "portable",
                "executed": execute,
                "source": {"nodes": inventory.nodes, "relationships": inventory.relationships},
                "target_before": {"nodes": target_inventory.nodes, "relationships": target_inventory.relationships},
                "schema_objects": len(schema),
                "resume": resume,
            }
            if is_v3_corpus:
                summary["source_corpus"] = corpus_inventory.to_dict()
                if target_connection.uri.startswith("neo4j+s://") and corpus_inventory.embedded_chunks:
                    summary.setdefault("warnings", []).append(
                        "Portable mode streams and fingerprints every stored embedding; prefer Aura dump/upload "
                        "for large v3 vector corpora."
                    )
            if not execute:
                summary["cleanup_only_resume"] = cleanup_only_resume
                return summary
            if cleanup_only_resume:
                progress["staging_started"] = True
                progress["phase"] = "cleanup-staging"
                cleanup_staging(target, target_connection.database)
                progress["staging_started"] = False
                progress["phase"] = "final-verification"
                final_inventory = read_inventory(target, target_connection.database)
                verify_inventory(inventory, final_inventory)
                target_schema_signature = read_schema_signature(target, target_connection.database)
                if source_schema_signature != target_schema_signature:
                    raise MigrationError("Migration schema verification failed after resumed cleanup.")
                if is_v3_corpus:
                    target_corpus_inventory = read_corpus_inventory(target, target_connection.database)
                    verify_corpus_inventory(corpus_inventory, target_corpus_inventory)
                    verify_corpus_connection(target_connection, dimension=corpus_dimension)
                    summary["target_corpus"] = target_corpus_inventory.to_dict()
                summary["cleanup_only_resume"] = True
                summary["target_after"] = {
                    "nodes": final_inventory.nodes,
                    "relationships": final_inventory.relationships,
                }
                summary["verified"] = True
                return summary
            if not resume and (target_inventory.nodes or target_inventory.relationships or target_schema):
                progress["phase"] = "clear-target"
                clear_target(target, target_connection.database)
            if verification_only_resume:
                progress["staging_started"] = True
                summary["verification_only_resume"] = True
            else:
                progress["phase"] = "create-staging"
                create_staging_constraint(target, target_connection.database)
                progress["staging_started"] = True
                progress["phase"] = "copy-nodes"
                summary["nodes_copied"] = copy_nodes(
                    source, target, source_connection.database, target_connection.database, batch_size
                )
                progress["phase"] = "copy-relationships"
                summary["relationships_copied"] = copy_relationships(
                    source, target, source_connection.database, target_connection.database, batch_size
                )
                progress["phase"] = "recreate-schema"
                summary["schema_created"] = recreate_schema(target, target_connection.database, schema)
            staged_inventory = read_inventory(target, target_connection.database)
            verify_entity_counts(inventory, staged_inventory)
            source_fingerprint = graph_content_fingerprint(source, source_connection.database, staged=False)
            target_fingerprint = graph_content_fingerprint(target, target_connection.database, staged=True)
            if source_fingerprint != target_fingerprint:
                raise MigrationError("Migration content verification failed before staging cleanup.")
            summary["content_fingerprint"] = source_fingerprint
            progress["phase"] = "cleanup-staging"
            cleanup_staging(target, target_connection.database)
            progress["staging_started"] = False
            progress["phase"] = "final-verification"
            final_inventory = read_inventory(target, target_connection.database)
            verify_inventory(inventory, final_inventory)
            target_schema_signature = read_schema_signature(target, target_connection.database)
            if source_schema_signature != target_schema_signature:
                raise MigrationError(
                    "Migration schema verification failed after copy. "
                    f"source objects={sorted(source_schema_signature)}, target objects={sorted(target_schema_signature)}"
                )
            if is_v3_corpus:
                target_corpus_inventory = read_corpus_inventory(target, target_connection.database)
                verify_corpus_inventory(corpus_inventory, target_corpus_inventory)
                verify_corpus_connection(target_connection, dimension=corpus_dimension)
                summary["target_corpus"] = target_corpus_inventory.to_dict()
            summary["target_after"] = {
                "nodes": final_inventory.nodes,
                "relationships": final_inventory.relationships,
            }
            summary["verified"] = True
            return summary


def portable_migrate(
    source_connection: ResolvedNeo4jConnection,
    target_connection: ResolvedNeo4jConnection,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    execute: bool = False,
    overwrite_target: bool = False,
    resume: bool = False,
) -> dict[str, Any]:
    progress: dict[str, Any] = {"phase": "preflight", "staging_started": False}
    try:
        return _portable_migrate_impl(
            source_connection,
            target_connection,
            progress,
            batch_size=batch_size,
            execute=execute,
            overwrite_target=overwrite_target,
            resume=resume,
        )
    except MigrationError as exc:
        if progress["phase"] in {"preflight", "connect", "inventory"} and not progress["staging_started"]:
            raise
        hint = (
            "Rerun with --resume --execute using the same source and target."
            if progress["staging_started"]
            else "Inspect the target and rerun the original command."
        )
        raise MigrationError(
            f"Portable migration failed during {progress['phase']}: {exc}. {hint} "
            f"Progress: {json.dumps({key: value for key, value in progress.items() if key != 'staging_started'}, sort_keys=True)}"
        ) from exc
    except Exception as exc:
        hint = (
            "Rerun with --resume --execute using the same source and target."
            if progress["staging_started"]
            else "Inspect the target and rerun the original command."
        )
        raise MigrationError(
            f"Portable migration failed during {progress['phase']}: {exc}. {hint} "
            f"Progress: {json.dumps({key: value for key, value in progress.items() if key != 'staging_started'}, sort_keys=True)}"
        ) from exc


def _run_checked(command: list[str], *, env: dict[str, str] | None = None) -> None:
    result = subprocess.run(command, env=env, text=True, check=False)
    if result.returncode != 0:
        raise MigrationError(f"Command failed with exit code {result.returncode}: {command[0]}")


def parse_neo4j_version(raw: str) -> tuple[int, ...]:
    match = re.search(r"(\d+)\.(\d+)(?:\.(\d+))?", raw)
    if not match:
        raise MigrationError(f"Unable to parse Neo4j version from: {raw}")
    return tuple(int(value or 0) for value in match.groups())


def aura_upload(
    target: ResolvedNeo4jConnection,
    *,
    database: str,
    dump_dir: Path,
    neo4j_admin_bin: str,
    execute: bool,
    overwrite_target: bool,
    dump_will_be_created: bool = False,
    neo4j_admin_container: str | None = None,
    expected_corpus: CorpusInventory | None = None,
    corpus_dimension: int = 384,
    retrieval_unit: str = "chunk",
    vector_index: str = "chunk_embedding_v1",
    keyword_index: str = "chunk_keyword_v1",
) -> dict[str, Any]:
    if not target.uri.startswith("neo4j+s://") or not target.uri.endswith(".databases.neo4j.io"):
        raise MigrationError("Aura upload requires an Aura neo4j+s://*.databases.neo4j.io target URI.")
    image = ""
    if neo4j_admin_container:
        version_command = ["docker", "exec", neo4j_admin_container, "neo4j-admin", "--version"]
        inspect = subprocess.run(
            ["docker", "inspect", neo4j_admin_container], capture_output=True, text=True, check=False
        )
        inspect_payload = json.loads(inspect.stdout or "[]") if inspect.returncode == 0 else []
        image = str((inspect_payload[0].get("Config") or {}).get("Image") or "") if inspect_payload else ""
        if not image:
            raise MigrationError(f"Unable to determine image for neo4j-admin container '{neo4j_admin_container}'.")
        executable = "neo4j-admin"
    else:
        executable = shutil.which(neo4j_admin_bin) or (
            str(Path(neo4j_admin_bin).resolve()) if Path(neo4j_admin_bin).exists() else ""
        )
        if not executable:
            raise MigrationError(f"neo4j-admin executable not found: {neo4j_admin_bin}")
        version_command = [executable, "--version"]
    version = subprocess.run(version_command, capture_output=True, text=True, check=False)
    if version.returncode != 0 or parse_neo4j_version(version.stdout + version.stderr) < (5, 26, 0):
        raise MigrationError("Aura upload requires neo4j-admin and a dump from Neo4j 5.26 LTS or later.")
    dump_file = dump_dir / f"{database}.dump"
    if not dump_file.exists() and not (not execute and dump_will_be_created):
        raise MigrationError(f"Database dump not found: {dump_file}")
    if not overwrite_target:
        raise MigrationError("Aura upload replaces the destination and requires --overwrite-target.")
    summary = {
        "mode": "aura-upload",
        "executed": execute,
        "dump": str(dump_file),
        "target": target.uri,
        "warnings": ["Aura capacity is not exposed over Bolt; confirm the target tier can hold the source dump."],
    }
    if expected_corpus is not None:
        summary["source_corpus"] = expected_corpus.to_dict()
    if not execute:
        return summary
    env = os.environ.copy()
    env.update({"NEO4J_USERNAME": target.username, "NEO4J_PASSWORD": target.password})
    upload_args = [
        "neo4j-admin" if neo4j_admin_container else executable,
        "database",
        "upload",
        database,
        f"--from-path={'/backups' if neo4j_admin_container else dump_dir}",
        f"--to-uri={target.uri}",
        "--overwrite-destination=true",
    ]
    command = upload_args
    if neo4j_admin_container:
        command = [
            "docker",
            "run",
            "--rm",
            "--entrypoint",
            "neo4j-admin",
            "--volume",
            f"{dump_dir}:/backups:ro",
            "--env",
            "NEO4J_USERNAME",
            "--env",
            "NEO4J_PASSWORD",
            image,
            *upload_args[1:],
        ]
    _run_checked(command, env=env)
    verify_connection(target, require_write=True)
    target_corpus, target_dimension, is_v3_corpus = inspect_corpus_connection(
        target,
        retrieval_unit=retrieval_unit,
        vector_index=vector_index,
    )
    if expected_corpus is not None:
        verify_corpus_inventory(expected_corpus, target_corpus, retrieval_unit=retrieval_unit)
        if not is_v3_corpus:
            raise MigrationError("Aura upload copied v3 corpus data without its required vector index.")
    if (
        expected_corpus is None
        and target_corpus.present
        and (target_corpus.active_revisions or target_corpus.embedded_chunks)
        and not is_v3_corpus
    ):
        raise MigrationError("Aura upload contains v3-like corpus data without its required vector index.")
    if is_v3_corpus:
        expected_dimension = corpus_dimension if expected_corpus is not None else target_dimension
        verify_corpus_connection(
            target,
            dimension=expected_dimension,
            retrieval_unit=retrieval_unit,
            vector_index=vector_index,
            keyword_index=keyword_index,
        )
        summary["target_corpus"] = target_corpus.to_dict()
    summary["verified"] = True
    return summary


def create_container_dump(
    source: ResolvedNeo4jConnection,
    *,
    container_name: str,
    dump_dir: Path,
    execute: bool,
) -> Path:
    version = subprocess.run(
        ["docker", "exec", container_name, "neo4j-admin", "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    if version.returncode != 0:
        raise MigrationError(f"Unable to run neo4j-admin in source container '{container_name}'.")
    if parse_neo4j_version(version.stdout + version.stderr) < (5, 26, 0):
        raise MigrationError(
            "Aura upload requires a source container running Neo4j 5.26 LTS or later; use portable mode for older containers."
        )
    dump_path = dump_dir / f"{source.database}.dump"
    if not execute:
        return dump_path
    dump_dir.mkdir(parents=True, exist_ok=True)
    inspect = subprocess.run(
        ["docker", "inspect", container_name],
        capture_output=True,
        text=True,
        check=False,
    )
    if inspect.returncode != 0:
        raise MigrationError(f"Unable to inspect source container '{container_name}'.")
    payload = json.loads(inspect.stdout or "[]")
    if not payload or not bool((payload[0].get("State") or {}).get("Running")):
        raise MigrationError(f"Source container '{container_name}' must be running before dump creation.")
    image = str((payload[0].get("Config") or {}).get("Image") or "")
    if not image:
        raise MigrationError(f"Unable to determine the image for source container '{container_name}'.")
    stopped = False
    try:
        _run_checked(["docker", "stop", container_name])
        stopped = True
        _run_checked(
            [
                "docker",
                "run",
                "--rm",
                "--volumes-from",
                container_name,
                "--volume",
                f"{dump_dir}:/backups",
                image,
                "neo4j-admin",
                "database",
                "dump",
                source.database,
                "--to-path=/backups",
                "--overwrite-destination=true",
            ]
        )
    finally:
        if stopped:
            _run_checked(["docker", "start", container_name])
            deadline = time.monotonic() + 120
            last_error: Exception | None = None
            while time.monotonic() < deadline:
                try:
                    current = subprocess.run(
                        ["docker", "inspect", container_name],
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    current_payload = json.loads(current.stdout or "[]") if current.returncode == 0 else []
                    ports = (
                        (((current_payload[0].get("NetworkSettings") or {}).get("Ports") or {}).get("7687/tcp") or [])
                        if current_payload
                        else []
                    )
                    verification_connection = source
                    if ports and ports[0].get("HostPort"):
                        parsed = urlsplit(source.uri)
                        hostname = parsed.hostname or "127.0.0.1"
                        host = f"[{hostname}]" if ":" in hostname and not hostname.startswith("[") else hostname
                        current_uri = urlunsplit(
                            (parsed.scheme, f"{host}:{ports[0]['HostPort']}", parsed.path, parsed.query, parsed.fragment)
                        )
                        verification_connection = ResolvedNeo4jConnection(
                            current_uri,
                            source.username,
                            source.password,
                            source.database,
                            source.deployment,
                        )
                    verify_connection(verification_connection)
                    break
                except Exception as exc:
                    last_error = exc
                    time.sleep(2)
            else:
                raise MigrationError(
                    f"Source container '{container_name}' restarted but Neo4j did not become reachable: {last_error}"
                )
    if not dump_path.exists():
        raise MigrationError(f"neo4j-admin completed without creating expected dump: {dump_path}")
    return dump_path


def activate_manifest(path: Path, target: ResolvedNeo4jConnection, password_env: str) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["version"] = max(int(payload.get("version") or 1), 3)
    payload["neo4j"] = {
        "deployment": "external",
        "uri": target.uri,
        "username": target.username,
        "database": target.database,
        "password_env": password_env,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def manifest_corpus_id(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    corpus_id = str((payload.get("corpus") or {}).get("id") or "").strip()
    if not corpus_id:
        raise MigrationError(f"Manifest does not define corpus.id: {path}")
    return corpus_id


def manifest_retrieval_profile(path: Path) -> tuple[str, str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    retrieval = dict(payload.get("retrieval") or {})
    unit = str(retrieval.get("unit") or "chunk")
    if unit not in {"chunk", "parent"}:
        raise MigrationError(f"Manifest defines unsupported retrieval unit '{unit}': {path}")
    prefix = "parent" if unit == "parent" else "chunk"
    return (
        unit,
        str(retrieval.get("vector_index") or f"{prefix}_embedding_v1"),
        str(retrieval.get("keyword_index") or f"{prefix}_keyword_v1"),
    )


def _connection_from_args(args: argparse.Namespace, prefix: str) -> ResolvedNeo4jConnection:
    upper = prefix.upper()
    uri = getattr(args, f"{prefix}_uri") or os.environ.get(f"NEO4J_{upper}_URI", "")
    username = getattr(args, f"{prefix}_user") or os.environ.get(f"NEO4J_{upper}_USERNAME", "neo4j")
    database = getattr(args, f"{prefix}_database") or os.environ.get(f"NEO4J_{upper}_DATABASE", "neo4j")
    password_env = getattr(args, f"{prefix}_password_env")
    validate_neo4j_uri(uri)
    return resolve_connection(
        Neo4jConnectionSpec(uri, username, database, password_env),
        environ={},
        password=os.environ.get(password_env, ""),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Migrate a local Neo4j database to a hosted Neo4j database.")
    parser.add_argument("--mode", choices=("portable", "aura-upload", "corpus-merge"), default="portable")
    for prefix in ("source", "target"):
        parser.add_argument(f"--{prefix}-uri")
        parser.add_argument(f"--{prefix}-user")
        parser.add_argument(f"--{prefix}-database")
        parser.add_argument(f"--{prefix}-password-env", default=f"NEO4J_{prefix.upper()}_PASSWORD")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Resume an interrupted portable or corpus merge")
    parser.add_argument("--overwrite-target", action="store_true")
    parser.add_argument("--confirm-target", help="Exact '<target-uri>|<target-database>' overwrite confirmation")
    parser.add_argument("--manifest-path", type=Path)
    parser.add_argument("--state-path", type=Path, help="Checkpoint file for --mode corpus-merge")
    parser.add_argument("--activate-target", action="store_true")
    parser.add_argument("--dump-dir", type=Path)
    parser.add_argument("--source-container", help="Managed local Neo4j container to dump before Aura upload")
    parser.add_argument("--keep-dump", action="store_true", help="Keep a dump generated from --source-container")
    parser.add_argument("--neo4j-admin-bin", default="neo4j-admin")
    return parser


def main(argv: list[str] | None = None) -> int:
    started = time.monotonic()
    args = build_parser().parse_args(argv)
    if args.batch_size <= 0:
        raise MigrationError("--batch-size must be greater than zero.")
    target = _connection_from_args(args, "target")
    expected_confirmation = f"{target.uri}|{target.database}"
    if (args.overwrite_target or (args.mode == "corpus-merge" and args.execute)) and args.confirm_target != expected_confirmation:
        raise MigrationError(f"--confirm-target must exactly match: {expected_confirmation}")
    if args.activate_target and (not args.execute or not args.manifest_path):
        raise MigrationError("--activate-target requires --execute and --manifest-path.")
    if args.resume and args.mode not in {"portable", "corpus-merge"}:
        raise MigrationError("--resume is only supported in portable and corpus-merge modes.")
    if args.resume and args.overwrite_target:
        raise MigrationError("--resume and --overwrite-target are mutually exclusive.")
    if args.mode in {"portable", "corpus-merge"}:
        source = _connection_from_args(args, "source")
        if source.uri == target.uri and source.database == target.database:
            raise MigrationError("Source and target resolve to the same Neo4j database.")
        if args.mode == "portable":
            summary = portable_migrate(
                source,
                target,
                batch_size=args.batch_size,
                execute=args.execute,
                overwrite_target=args.overwrite_target,
                resume=args.resume,
            )
        else:
            if not args.manifest_path:
                raise MigrationError("--mode corpus-merge requires --manifest-path.")
            manifest_path = args.manifest_path.resolve()
            state_path = (
                args.state_path.resolve()
                if args.state_path
                else manifest_path.with_name(f"{manifest_path.stem}.corpus-merge-state.json")
            )
            summary = corpus_merge(
                source,
                target,
                corpus_id=manifest_corpus_id(manifest_path),
                state_path=state_path,
                batch_size=args.batch_size,
                execute=args.execute,
                resume=args.resume,
            )
    else:
        if not args.dump_dir:
            raise MigrationError("Aura upload requires --dump-dir containing <source-database>.dump.")
        database = args.source_database or os.environ.get("NEO4J_SOURCE_DATABASE", "neo4j")
        generated_dump: Path | None = None
        expected_corpus: CorpusInventory | None = None
        expected_corpus_dimension = 384
        retrieval_unit = "chunk"
        vector_index = "chunk_embedding_v1"
        keyword_index = "chunk_keyword_v1"
        if args.manifest_path:
            retrieval_unit, vector_index, keyword_index = manifest_retrieval_profile(
                args.manifest_path.resolve()
            )
        if args.source_container:
            source = _connection_from_args(args, "source")
            source_corpus, source_dimension, source_is_v3 = inspect_corpus_connection(
                source,
                retrieval_unit=retrieval_unit,
                vector_index=vector_index,
            )
            if source_is_v3:
                expected_corpus = source_corpus
                expected_corpus_dimension = source_dimension
            generated_dump = create_container_dump(
                source,
                container_name=args.source_container,
                dump_dir=args.dump_dir.resolve(),
                execute=args.execute,
            )
        summary = aura_upload(
            target,
            database=database,
            dump_dir=args.dump_dir.resolve(),
            neo4j_admin_bin=args.neo4j_admin_bin,
            execute=args.execute,
            overwrite_target=args.overwrite_target,
            dump_will_be_created=bool(args.source_container),
            neo4j_admin_container=args.source_container,
            expected_corpus=expected_corpus,
            corpus_dimension=expected_corpus_dimension,
            retrieval_unit=retrieval_unit,
            vector_index=vector_index,
            keyword_index=keyword_index,
        )
        if generated_dump is not None and args.execute and not args.keep_dump:
            generated_dump.unlink(missing_ok=True)
    if args.activate_target:
        activate_manifest(args.manifest_path.resolve(), target, args.target_password_env)
    summary["elapsed_seconds"] = round(time.monotonic() - started, 3)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except MigrationError as exc:
        print(f"ERROR: {exc}", file=os.sys.stderr)
        raise SystemExit(1)
