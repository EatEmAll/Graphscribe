from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

DELETE_ENTITIES_AND_START_FROM_BEGINNING = "delete_entities_and_start_from_beginning"
START_FROM_LAST_PROCESSED_POSITION = "start_from_last_processed_position"
SUPPORTED_RETRY_CONDITIONS = {
    DELETE_ENTITIES_AND_START_FROM_BEGINNING,
    START_FROM_LAST_PROCESSED_POSITION,
}


@dataclass
class Neo4jCredentials:
    uri: str
    userName: str
    password: str
    database: str
    email: str | None = None


@dataclass
class SourceScanExtractParams:
    model: str
    file_name: str
    source_type: str
    retry_condition: str | None = None
    token_chunk_size: int | None = None
    chunk_overlap: int | None = None
    chunks_to_combine: int | None = None
    allowedNodes: str | None = None
    allowedRelationship: str | None = None
    additional_instructions: str | None = None
    embedding_provider: str | None = None
    embedding_model: str | None = None


class LLMGraphBuilderException(RuntimeError):
    pass


def create_graph_database_connection(credentials):
    from src.shared.common_fn import create_graph_database_connection as impl

    return impl(credentials)


def close_db_connection(graph, api_name: str) -> None:
    from src.shared.common_fn import close_db_connection as impl

    impl(graph, api_name)


def graphDBdataAccess(graph):
    from src.graphDB_dataAccess import graphDBdataAccess as impl

    return impl(graph)


def connection_check_and_get_vector_dimensions(graph, database, email, uri):
    from src.main import connection_check_and_get_vector_dimensions as impl

    return impl(graph, database, email, uri)


async def processing_source(credentials, params, pages, merged_file_path=None, is_uploaded_from_local=None):
    from src.main import processing_source as impl

    return await impl(credentials, params, pages, merged_file_path=merged_file_path, is_uploaded_from_local=is_uploaded_from_local)


def set_status_retry(graph, file_name: str, retry_condition: str) -> None:
    from src.main import set_status_retry as impl

    impl(graph, file_name, retry_condition)


def create_vector_fulltext_indexes(credentials, embedding_provider: str, embedding_model: str) -> None:
    from src.post_processing import create_vector_fulltext_indexes as impl

    impl(credentials, embedding_provider, embedding_model)


def graph_schema_consolidation(graph) -> None:
    from src.post_processing import graph_schema_consolidation as impl

    impl(graph)


def get_documents_from_file_by_path(file_path: str, file_name: str):
    from src.document_sources.local_file import get_documents_from_file_by_path as impl

    return impl(file_path, file_name)


def make_source_node():
    from src.entities.source_node import sourceNode

    return sourceNode()


def update_exception_db(graph, file_name: str, message: str, retry_condition: str | None) -> None:
    access = graphDBdataAccess(graph)
    access.update_exception_db(file_name, message, retry_condition)


class GraphBuilderAPI:
    """Local runtime wrapper that preserves the old orchestration surface."""

    def __init__(
        self,
        neo4j_uri: str,
        neo4j_user: str,
        neo4j_password: str,
        neo4j_database: str,
        *,
        sources_dir: Path | None = None,
        base_url: str | None = None,
        timeout: int = 600,
    ):
        self.credentials = Neo4jCredentials(
            uri=neo4j_uri,
            userName=neo4j_user,
            password=neo4j_password,
            database=neo4j_database,
            email=None,
        )
        self.sources_dir = Path(sources_dir).resolve() if sources_dir else None
        self.base_url = base_url
        self.timeout = timeout

    def _graph(self):
        return create_graph_database_connection(self.credentials)

    def preflight_capabilities(self) -> dict[str, str]:
        from notebooklm_graph_pipe.runtime.neo4j_connection import (
            ResolvedNeo4jConnection,
            verify_workflow_connection,
        )

        return verify_workflow_connection(
            ResolvedNeo4jConnection(
                uri=self.credentials.uri,
                username=self.credentials.userName,
                password=self.credentials.password,
                database=self.credentials.database,
            )
        )

    def _resolve_source_path(self, file_name: str) -> Path:
        if self.sources_dir is None:
            raise FileNotFoundError(f"No sources_dir configured for runtime; cannot resolve '{file_name}'")
        path = (self.sources_dir / file_name).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Source file not found: {path}")
        return path

    def health_check(self) -> bool:
        try:
            graph = self._graph()
            try:
                connection_check_and_get_vector_dimensions(
                    graph,
                    self.credentials.database,
                    self.credentials.email,
                    self.credentials.uri,
                )
            finally:
                close_db_connection(graph, "health_check")
            return True
        except Exception:
            return False

    def connect(self, embedding_provider: str = "", embedding_model: str = "") -> dict[str, Any]:
        graph = self._graph()
        try:
            result = connection_check_and_get_vector_dimensions(
                graph,
                self.credentials.database,
                self.credentials.email,
                self.credentials.uri,
            )
            return {"status": "Success", "data": result}
        except Exception as exc:
            return {"status": "Failed", "message": str(exc)}
        finally:
            close_db_connection(graph, "connect")

    def sources_list(self) -> list[dict[str, Any]]:
        graph = self._graph()
        try:
            access = graphDBdataAccess(graph)
            return access.get_source_list()
        finally:
            close_db_connection(graph, "sources_list")

    def retry_processing(self, file_name: str, retry_condition: str) -> dict[str, Any]:
        graph = self._graph()
        try:
            set_status_retry(graph, file_name, retry_condition)
            return {"status": "Success", "message": f"Status set to Ready to Reprocess for filename : {file_name}"}
        finally:
            close_db_connection(graph, "retry_processing")

    def upload_file(self, file_path: Path, model: str) -> dict[str, Any]:
        file_path = file_path.resolve()
        obj_source_node = make_source_node()
        obj_source_node.file_name = file_path.name
        obj_source_node.file_type = file_path.suffix.lstrip(".")
        obj_source_node.file_size = file_path.stat().st_size
        obj_source_node.file_source = "local file"
        obj_source_node.model = model
        obj_source_node.created_at = datetime.now()
        obj_source_node.chunkNodeCount = 0
        obj_source_node.chunkRelCount = 0
        obj_source_node.entityNodeCount = 0
        obj_source_node.entityEntityRelCount = 0
        obj_source_node.communityNodeCount = 0
        obj_source_node.communityRelCount = 0

        graph = self._graph()
        try:
            access = graphDBdataAccess(graph)
            access.create_source_node(obj_source_node)
            return {
                "status": "Success",
                "data": {
                    "file_name": file_path.name,
                    "file_size": obj_source_node.file_size,
                    "file_extension": obj_source_node.file_type,
                },
            }
        except Exception as exc:
            return {"status": "Failed", "message": str(exc), "error": str(exc), "file_name": file_path.name}
        finally:
            close_db_connection(graph, "upload_file")

    def extract(
        self,
        file_name: str,
        model: str,
        source_type: str = "local file",
        embedding_provider: str | None = None,
        embedding_model: str | None = None,
        retry_condition: str | None = None,
        token_chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        chunks_to_combine: int | None = None,
    ) -> dict[str, Any]:
        if source_type != "local file":
            return {"status": "Failed", "message": f"Unsupported source_type: {source_type}", "file_name": file_name}

        params = SourceScanExtractParams(
            model=model,
            file_name=file_name,
            source_type=source_type,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            retry_condition=retry_condition,
            token_chunk_size=token_chunk_size,
            chunk_overlap=chunk_overlap,
            chunks_to_combine=chunks_to_combine,
        )
        try:
            if retry_condition in SUPPORTED_RETRY_CONDITIONS:
                pages = []
            else:
                file_path = self._resolve_source_path(file_name)
                resolved_name, pages, _ = get_documents_from_file_by_path(str(file_path), file_name)
                if pages is None or len(pages) == 0:
                    raise LLMGraphBuilderException(f"File content is not available for file : {resolved_name}")
                params.file_name = resolved_name

            _, result = asyncio.run(
                processing_source(
                    self.credentials,
                    params,
                    pages,
                    merged_file_path=None,
                    is_uploaded_from_local=False,
                )
            )
            return {"status": "Success", "data": result, "file_name": params.file_name}
        except Exception as exc:
            error_message = str(exc)
            graph = self._graph()
            try:
                update_exception_db(graph, params.file_name, error_message, retry_condition)
            finally:
                close_db_connection(graph, "extract_failure")
            return {
                "status": "Failed",
                "message": error_message,
                "error": error_message,
                "file_name": params.file_name,
            }

    def document_status(self, file_name: str) -> dict[str, Any] | None:
        graph = self._graph()
        try:
            access = graphDBdataAccess(graph)
            result = access.get_current_status_document_node(file_name)
            if not result:
                return None
            row = result[0]
            return {
                "fileName": file_name,
                "status": row.get("Status"),
                "processingTime": row.get("processingTime"),
                "nodeCount": row.get("nodeCount"),
                "relationshipCount": row.get("relationshipCount"),
                "total_chunks": row.get("total_chunks"),
                "processed_chunk": row.get("processed_chunk"),
                "fileSource": row.get("fileSource"),
            }
        finally:
            close_db_connection(graph, "document_status")

    def post_processing(self, tasks: list[str], embedding_provider: str = "", embedding_model: str = "") -> dict[str, Any]:
        task_set = {task.strip() for task in tasks}
        executed: list[str] = []
        if "enable_hybrid_search_and_fulltext_search_in_bloom" in task_set:
            create_vector_fulltext_indexes(self.credentials, embedding_provider, embedding_model)
            executed.append("enable_hybrid_search_and_fulltext_search_in_bloom")
        if "graph_schema_consolidation" in task_set:
            graph = self._graph()
            try:
                graph_schema_consolidation(graph)
            finally:
                close_db_connection(graph, "graph_schema_consolidation")
            executed.append("graph_schema_consolidation")
        unsupported = sorted(task for task in task_set if task not in set(executed))
        if unsupported:
            return {"status": "Failed", "message": f"Unsupported post-processing tasks: {', '.join(unsupported)}"}
        return {"status": "Success", "data": {"tasks": executed}}
