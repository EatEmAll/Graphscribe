from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping
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


def resolve_connection(
    spec: Neo4jConnectionSpec,
    *,
    environ: Mapping[str, str] | None = None,
    password: str | None = None,
) -> ResolvedNeo4jConnection:
    env = environ if environ is not None else os.environ
    uri = validate_neo4j_uri(env.get("NEO4J_URI", spec.uri))
    username = env.get("NEO4J_USERNAME", spec.username).strip()
    database = env.get("NEO4J_DATABASE", spec.database).strip()
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


def redacted_connection(connection: Neo4jConnectionSpec | ResolvedNeo4jConnection) -> str:
    return f"{connection.uri} (user={connection.username}, database={connection.database})"
