from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Mapping
from urllib.parse import urlparse

from neo4j import GraphDatabase


SUPPORTED_NEO4J_SCHEMES = {
    "bolt",
    "bolt+s",
    "bolt+ssc",
    "neo4j",
    "neo4j+s",
    "neo4j+ssc",
}


class Neo4jConnectionError(ValueError):
    pass


@dataclass(frozen=True)
class Neo4jConnectionSpec:
    uri: str
    username: str
    database: str
    password_env: str = "NEO4J_PASSWORD"
    deployment: str = "external"


@dataclass(frozen=True)
class ResolvedNeo4jConnection:
    uri: str
    username: str
    password: str
    database: str
    deployment: str = "external"

    @property
    def spec(self) -> Neo4jConnectionSpec:
        return Neo4jConnectionSpec(
            uri=self.uri,
            username=self.username,
            database=self.database,
            deployment=self.deployment,
        )


def validate_neo4j_uri(uri: str) -> str:
    value = uri.strip()
    parsed = urlparse(value)
    if parsed.scheme not in SUPPORTED_NEO4J_SCHEMES:
        schemes = ", ".join(sorted(SUPPORTED_NEO4J_SCHEMES))
        raise Neo4jConnectionError(f"Unsupported Neo4j URI scheme in '{value}'. Expected one of: {schemes}.")
    if not parsed.hostname:
        raise Neo4jConnectionError(f"Neo4j URI is missing a hostname: {value}")
    return value


def connection_spec_from_mapping(
    value: Mapping[str, object],
    *,
    default_password_env: str = "NEO4J_PASSWORD",
) -> Neo4jConnectionSpec:
    return Neo4jConnectionSpec(
        uri=str(value.get("uri") or "").strip(),
        username=str(value.get("username") or value.get("user") or "").strip(),
        database=str(value.get("database") or "neo4j").strip(),
        password_env=str(value.get("password_env") or default_password_env).strip(),
        deployment=str(value.get("deployment") or ("managed-local" if value.get("container_name") else "external")),
    )


def connection_spec_to_mapping(
    connection: Neo4jConnectionSpec | ResolvedNeo4jConnection,
    *,
    password_env: str = "NEO4J_PASSWORD",
    extra: Mapping[str, object] | None = None,
) -> dict[str, object]:
    deployment = connection.deployment
    payload: dict[str, object] = {
        "deployment": deployment,
        "uri": connection.uri,
        "username": connection.username,
        "database": connection.database,
        "password_env": password_env,
    }
    if extra:
        payload.update({key: value for key, value in extra.items() if value is not None and key != "password"})
    return payload


def resolve_connection(
    spec: Neo4jConnectionSpec,
    *,
    environ: Mapping[str, str] | None = None,
    password: str | None = None,
    allow_standard_env_overrides: bool = True,
) -> ResolvedNeo4jConnection:
    env = environ if environ is not None else os.environ
    uri = validate_neo4j_uri(env.get("NEO4J_URI", spec.uri) if allow_standard_env_overrides else spec.uri)
    username = (env.get("NEO4J_USERNAME", spec.username) if allow_standard_env_overrides else spec.username).strip()
    database = (env.get("NEO4J_DATABASE", spec.database) if allow_standard_env_overrides else spec.database).strip()
    secret = password or env.get(spec.password_env, "")
    if not username:
        raise Neo4jConnectionError("Neo4j username is required.")
    if not database:
        raise Neo4jConnectionError("Neo4j database is required.")
    if not secret:
        raise Neo4jConnectionError(
            f"Neo4j password is required. Set environment variable '{spec.password_env}'."
        )
    return ResolvedNeo4jConnection(uri, username, secret, database, spec.deployment)


def resolve_connection_mapping(
    value: Mapping[str, object],
    *,
    environ: Mapping[str, str] | None = None,
) -> ResolvedNeo4jConnection:
    """Resolve a persisted connection without allowing global URI/user/database overrides."""
    spec = connection_spec_from_mapping(value)
    env = environ if environ is not None else os.environ
    legacy_password = None if env.get(spec.password_env) else (str(value.get("password") or "") or None)
    return resolve_connection(
        spec,
        environ=env,
        password=legacy_password,
        allow_standard_env_overrides=False,
    )


def verify_connection(connection: ResolvedNeo4jConnection, *, require_write: bool = False) -> dict[str, str]:
    with GraphDatabase.driver(connection.uri, auth=(connection.username, connection.password)) as driver:
        driver.verify_connectivity()
        server = driver.get_server_info()
        with driver.session(database=connection.database) as session:
            session.run("RETURN 1 AS ok").consume()
            if require_write:
                session.run(
                    "CREATE (n:`__Neo4jConnectionProbe`) DELETE n RETURN 1 AS ok"
                ).consume()
    return {"address": str(server.address), "agent": str(server.agent)}


def required_apoc_capabilities(connection: ResolvedNeo4jConnection) -> tuple[set[str], set[str]]:
    required_procedures = {"apoc.merge.node", "apoc.refactor.mergeNodes"}
    required_functions = {
        "apoc.any.properties",
        "apoc.coll.flatten",
        "apoc.coll.removeAll",
        "apoc.coll.subtract",
        "apoc.coll.toSet",
        "apoc.text.distance",
    }
    with GraphDatabase.driver(connection.uri, auth=(connection.username, connection.password)) as driver:
        with driver.session(database=connection.database) as session:
            procedures = {
                str(row["name"])
                for row in session.run("SHOW PROCEDURES YIELD name WHERE name STARTS WITH 'apoc.' RETURN name")
            }
            functions = {
                str(row["name"])
                for row in session.run("SHOW FUNCTIONS YIELD name WHERE name STARTS WITH 'apoc.' RETURN name")
            }
    return required_procedures - procedures, required_functions - functions


def verify_workflow_connection(connection: ResolvedNeo4jConnection) -> dict[str, str]:
    server = verify_connection(connection, require_write=True)
    missing_procedures, missing_functions = required_apoc_capabilities(connection)
    missing = sorted(missing_procedures | missing_functions)
    if missing:
        raise Neo4jConnectionError(
            "Neo4j is missing APOC capabilities required by the graph workflow: " + ", ".join(missing)
        )
    return server


def _neo4j_version(server_agent: str) -> tuple[int, int, int]:
    match = re.search(r"(?:Neo4j/)?(\d+)\.(\d+)(?:\.(\d+))?", server_agent)
    if not match:
        raise Neo4jConnectionError(f"Unable to determine Neo4j version from server agent: {server_agent}")
    return tuple(int(value or 0) for value in match.groups())


def _index_schema_compatible(
    name: str,
    actual: tuple[str, str, list[str], list[str]],
    expected: tuple[str, str, list[str], list[str]],
) -> bool:
    if name != "entities":
        return actual == expected
    return (
        actual[:2] == expected[:2]
        and set(expected[2]).issubset(actual[2])
        and actual[3] == expected[3]
    )


def _validate_corpus_indexes(
    session: Any,
    dimension: int,
    retrieval_unit: str = "chunk",
    vector_index: str = "chunk_embedding_v1",
    keyword_index: str = "chunk_keyword_v1",
) -> None:
    if retrieval_unit not in {"chunk", "parent"}:
        raise Neo4jConnectionError(f"Unsupported retrieval unit: {retrieval_unit}")
    if any(not re.fullmatch(r"[A-Za-z0-9_]+", name) for name in (vector_index, keyword_index)):
        raise Neo4jConnectionError("Retrieval index names may contain only letters, digits, and underscores.")
    retrieval_label = "ParentChunk" if retrieval_unit == "parent" else "Chunk"
    required = {
        "document_status": ("RANGE", "NODE", ["Document"], ["status"]),
        "revision_ready": ("RANGE", "NODE", ["DocumentRevision"], ["vector_ready", "graph_ready"]),
        keyword_index: ("FULLTEXT", "NODE", [retrieval_label], ["text"]),
        "entities": ("FULLTEXT", "NODE", ["__Entity__"], ["id", "description"]),
        "community_keyword": ("FULLTEXT", "NODE", ["__Community__"], ["summary"]),
        vector_index: ("VECTOR", "NODE", [retrieval_label], ["embedding"]),
    }
    if retrieval_unit == "chunk":
        required["chunk_parent_id"] = ("RANGE", "NODE", ["Chunk"], ["parent_id"])
    rows = {
        str(row["name"]): dict(row)
        for row in session.run(
                "SHOW INDEXES YIELD name, type, entityType, labelsOrTypes, properties, options, state, "
                "populationPercent RETURN name, type, entityType, labelsOrTypes, properties, options, state, "
                "populationPercent"
        )
    }
    for name, expected in required.items():
        row = rows.get(name)
        if row is None:
            raise Neo4jConnectionError(
                f"Required corpus index '{name}' was not created. A conflicting index may already use its schema. "
                "Use a fresh blue-green database or remove the incompatible index explicitly."
            )
        actual = (
            str(row.get("type")),
            str(row.get("entityType")),
            sorted(str(value) for value in row.get("labelsOrTypes") or []),
            [str(value) for value in row.get("properties") or []],
        )
        normalized_expected = (expected[0], expected[1], sorted(expected[2]), expected[3])
        if not _index_schema_compatible(name, actual, normalized_expected):
            raise Neo4jConnectionError(
                f"Corpus index '{name}' has incompatible schema: expected={normalized_expected}, actual={actual}."
            )
        if str(row.get("state")) != "ONLINE":
            raise Neo4jConnectionError(f"Corpus index '{name}' is not ONLINE (state={row.get('state')}).")
        if float(row.get("populationPercent") or 0.0) != 100.0:
            raise Neo4jConnectionError(
                f"Corpus index '{name}' is not fully populated "
                f"(populationPercent={row.get('populationPercent')})."
            )
        if name == vector_index:
            config = dict((dict(row.get("options") or {}).get("indexConfig") or {}))
            actual_dimension = int(config.get("vector.dimensions") or 0)
            similarity = str(config.get("vector.similarity_function") or "").lower()
            if actual_dimension != dimension or similarity != "cosine":
                raise Neo4jConnectionError(
                    "Corpus vector index has incompatible configuration: "
                    f"expected dimension={dimension}/cosine, actual dimension={actual_dimension}/{similarity or 'unknown'}."
                )


def _validate_corpus_constraints(session: Any) -> None:
    required = {
        "corpus_id_unique": (["Corpus"], ["id"]),
        "corpus_key_unique": (["Corpus"], ["key"]),
        "document_id_unique": (["Document"], ["id"]),
        "revision_id_unique": (["DocumentRevision"], ["id"]),
        "parent_chunk_id_unique": (["ParentChunk"], ["id"]),
        "chunk_id_unique": (["Chunk"], ["id"]),
    }
    rows = {
        str(row["name"]): dict(row)
        for row in session.run(
            "SHOW CONSTRAINTS YIELD name, type, entityType, labelsOrTypes, properties "
            "RETURN name, type, entityType, labelsOrTypes, properties"
        )
    }
    for name, (labels, properties) in required.items():
        row = rows.get(name)
        if row is None:
            raise Neo4jConnectionError(f"Required corpus constraint '{name}' was not created.")
        constraint_type = str(row.get("type"))
        if constraint_type == "NODE_PROPERTY_UNIQUENESS":
            constraint_type = "UNIQUENESS"
        actual = (
            constraint_type,
            str(row.get("entityType")),
            sorted(str(value) for value in row.get("labelsOrTypes") or []),
            [str(value) for value in row.get("properties") or []],
        )
        expected = ("UNIQUENESS", "NODE", sorted(labels), properties)
        if actual != expected:
            raise Neo4jConnectionError(
                f"Corpus constraint '{name}' has incompatible schema: expected={expected}, actual={actual}."
            )


def verify_corpus_connection(
    connection: ResolvedNeo4jConnection,
    *,
    dimension: int = 384,
    initialize_schema: bool = False,
    require_write: bool = False,
    retrieval_unit: str = "chunk",
    vector_index: str = "chunk_embedding_v1",
    keyword_index: str = "chunk_keyword_v1",
) -> dict[str, str]:
    server = verify_connection(connection, require_write=require_write or initialize_schema)
    if _neo4j_version(server["agent"]) < (5, 23, 0):
        raise Neo4jConnectionError("The corpus workflow requires Neo4j 5.23 or later.")
    with GraphDatabase.driver(connection.uri, auth=(connection.username, connection.password)) as driver:
        if initialize_schema:
            from notebooklm_graph_pipe.ingestion.neo4j_store import Neo4jCorpusStore

            Neo4jCorpusStore(driver, connection.database).ensure_schema(dimension)
        with driver.session(database=connection.database) as session:
            if initialize_schema:
                session.run("CALL db.awaitIndexes(300)").consume()
            _validate_corpus_indexes(
                session,
                dimension,
                retrieval_unit=retrieval_unit,
                vector_index=vector_index,
                keyword_index=keyword_index,
            )
            _validate_corpus_constraints(session)
            query_vector = [0.0] * dimension
            query_vector[0] = 1.0
            session.run(
                f"CALL db.index.vector.queryNodes('{vector_index}', 1, $embedding) "
                "YIELD node RETURN node LIMIT 1",
                embedding=query_vector,
            ).consume()
            session.run(
                f"CALL db.index.fulltext.queryNodes('{keyword_index}', $search_text) "
                "YIELD node RETURN node LIMIT 1",
                search_text="__lgp_preflight_no_match__",
            ).consume()
    return server


def redacted_connection(connection: Neo4jConnectionSpec | ResolvedNeo4jConnection) -> str:
    return f"{connection.uri} (user={connection.username}, database={connection.database})"
