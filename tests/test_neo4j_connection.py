from __future__ import annotations

import pytest

import notebooklm_graph_pipe.runtime.neo4j_connection as connection_module
from notebooklm_graph_pipe.runtime.neo4j_connection import (
    Neo4jConnectionError,
    Neo4jConnectionSpec,
    redacted_connection,
    resolve_connection,
    resolve_connection_mapping,
    validate_neo4j_uri,
    verify_workflow_connection,
)


@pytest.mark.parametrize(
    "scheme",
    ["bolt", "bolt+s", "bolt+ssc", "neo4j", "neo4j+s", "neo4j+ssc"],
)
def test_validate_neo4j_uri_accepts_driver_schemes(scheme: str) -> None:
    assert validate_neo4j_uri(f"{scheme}://graph.example.com:7687") == f"{scheme}://graph.example.com:7687"


def test_validate_neo4j_uri_rejects_http() -> None:
    with pytest.raises(Neo4jConnectionError, match="Unsupported Neo4j URI scheme"):
        validate_neo4j_uri("https://graph.example.com")


def test_resolve_connection_uses_environment_overrides() -> None:
    resolved = resolve_connection(
        Neo4jConnectionSpec("bolt://localhost:7687", "neo4j", "neo4j", "GRAPH_PASSWORD"),
        environ={
            "NEO4J_URI": "neo4j+s://hosted.example.com",
            "NEO4J_USERNAME": "hosted-user",
            "NEO4J_DATABASE": "hosted-db",
            "GRAPH_PASSWORD": "secret-value",
        },
    )

    assert resolved.uri == "neo4j+s://hosted.example.com"
    assert resolved.username == "hosted-user"
    assert resolved.database == "hosted-db"
    assert resolved.password == "secret-value"


def test_resolve_connection_reports_named_missing_secret() -> None:
    spec = Neo4jConnectionSpec("bolt://localhost:7687", "neo4j", "neo4j", "GRAPH_PASSWORD")
    with pytest.raises(Neo4jConnectionError, match="GRAPH_PASSWORD"):
        resolve_connection(spec, environ={})


def test_manifest_connection_uses_only_its_named_secret() -> None:
    resolved = resolve_connection_mapping(
        {
            "uri": "neo4j+s://corpus.example.com",
            "username": "corpus-user",
            "database": "corpus-db",
            "password_env": "CORPUS_PASSWORD",
        },
        environ={
            "NEO4J_URI": "bolt://wrong.example.com",
            "NEO4J_USERNAME": "wrong-user",
            "NEO4J_DATABASE": "wrong-db",
            "CORPUS_PASSWORD": "secret",
        },
    )

    assert resolved.uri == "neo4j+s://corpus.example.com"
    assert resolved.username == "corpus-user"
    assert resolved.database == "corpus-db"
    assert resolved.password == "secret"


def test_named_environment_secret_overrides_legacy_manifest_password() -> None:
    resolved = resolve_connection_mapping(
        {
            "uri": "bolt://localhost:7687",
            "username": "neo4j",
            "database": "neo4j",
            "password": "legacy",
            "password_env": "CORPUS_PASSWORD",
        },
        environ={"CORPUS_PASSWORD": "rotated"},
    )

    assert resolved.password == "rotated"


def test_redacted_connection_never_contains_password() -> None:
    resolved = resolve_connection(
        Neo4jConnectionSpec("bolt://localhost:7687", "neo4j", "neo4j"),
        environ={"NEO4J_PASSWORD": "do-not-print"},
    )
    assert "do-not-print" not in redacted_connection(resolved)


def test_verify_workflow_connection_rejects_missing_apoc(monkeypatch) -> None:
    resolved = resolve_connection(
        Neo4jConnectionSpec("bolt://localhost:7687", "neo4j", "neo4j"),
        environ={"NEO4J_PASSWORD": "secret"},
    )
    monkeypatch.setattr(connection_module, "verify_connection", lambda *args, **kwargs: {"agent": "Neo4j/5"})
    monkeypatch.setattr(
        connection_module,
        "required_apoc_capabilities",
        lambda connection: ({"apoc.merge.node"}, {"apoc.text.distance"}),
    )

    with pytest.raises(Neo4jConnectionError, match="apoc.merge.node.*apoc.text.distance"):
        verify_workflow_connection(resolved)


def test_corpus_index_validation_rejects_wrong_vector_dimension() -> None:
    rows = [
        {
            "name": "document_status",
            "type": "RANGE",
            "entityType": "NODE",
            "labelsOrTypes": ["Document"],
            "properties": ["status"],
            "options": {},
            "state": "ONLINE",
        },
        {
            "name": "revision_ready",
            "type": "RANGE",
            "entityType": "NODE",
            "labelsOrTypes": ["DocumentRevision"],
            "properties": ["vector_ready", "graph_ready"],
            "options": {},
            "state": "ONLINE",
        },
        {
            "name": "chunk_parent_id",
            "type": "RANGE",
            "entityType": "NODE",
            "labelsOrTypes": ["Chunk"],
            "properties": ["parent_id"],
            "options": {},
            "state": "ONLINE",
        },
        {
            "name": "chunk_keyword_v1",
            "type": "FULLTEXT",
            "entityType": "NODE",
            "labelsOrTypes": ["Chunk"],
            "properties": ["text"],
            "options": {},
            "state": "ONLINE",
        },
        {
            "name": "entities",
            "type": "FULLTEXT",
            "entityType": "NODE",
            "labelsOrTypes": ["__Entity__"],
            "properties": ["id", "description"],
            "options": {},
            "state": "ONLINE",
        },
        {
            "name": "community_keyword",
            "type": "FULLTEXT",
            "entityType": "NODE",
            "labelsOrTypes": ["__Community__"],
            "properties": ["summary"],
            "options": {},
            "state": "ONLINE",
        },
        {
            "name": "chunk_embedding_v1",
            "type": "VECTOR",
            "entityType": "NODE",
            "labelsOrTypes": ["Chunk"],
            "properties": ["embedding"],
            "options": {
                "indexConfig": {"vector.dimensions": 1536, "vector.similarity_function": "cosine"}
            },
            "state": "ONLINE",
        },
    ]

    class Session:
        def run(self, query):
            return rows

    with pytest.raises(Neo4jConnectionError, match="dimension=384"):
        connection_module._validate_corpus_indexes(Session(), 384)
