from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Sequence

import pyarrow as pa

from notebooklm_graph_pipe.runtime.contracts import VectorHit, VectorQuery, VectorRecord


def _literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


class LanceDBVectorStore:
    def __init__(self, location: str | Path, *, dimension: int, table_name: str = "graphrag_vectors"):
        if dimension <= 0:
            raise ValueError("Vector dimension must be positive.")
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", table_name):
            raise ValueError("Invalid LanceDB table name.")
        import lancedb

        self.location = Path(location).resolve()
        self.location.mkdir(parents=True, exist_ok=True)
        self.dimension = dimension
        self.table_name = table_name
        self.database = lancedb.connect(str(self.location))
        schema = pa.schema(
            [
                pa.field("record_id", pa.string(), nullable=False),
                pa.field("corpus_id", pa.string(), nullable=False),
                pa.field("revision_id", pa.string(), nullable=False),
                pa.field("document_id", pa.string(), nullable=False),
                pa.field("parent_id", pa.string(), nullable=False),
                pa.field("chunk_id", pa.string(), nullable=False),
                pa.field("unit_type", pa.string(), nullable=False),
                pa.field("embedding_fingerprint", pa.string(), nullable=False),
                pa.field("vector", pa.list_(pa.float32(), dimension), nullable=False),
                pa.field("text", pa.string(), nullable=False),
                pa.field("title", pa.string(), nullable=False),
                pa.field("source_uri", pa.string(), nullable=False),
                pa.field("source_type", pa.string(), nullable=False),
                pa.field("language", pa.string(), nullable=False),
            ]
        )
        self.table = self.database.create_table(table_name, schema=schema, exist_ok=True)
        if self.table.schema != schema:
            raise ValueError("Existing LanceDB table schema is incompatible; rebuild the derived vector index.")

    def upsert(self, records: Sequence[VectorRecord]) -> None:
        if not records:
            return
        rows = []
        for record in records:
            if len(record.vector) != self.dimension:
                raise ValueError(
                    f"Vector {record.record_id} has dimension {len(record.vector)}; expected {self.dimension}."
                )
            rows.append({**record.__dict__, "vector": list(record.vector)})
        (
            self.table.merge_insert("record_id")
            .when_matched_update_all()
            .when_not_matched_insert_all()
            .execute(rows)
        )

    def delete_revisions(self, corpus_id: str, revision_ids: Sequence[str]) -> None:
        if not revision_ids:
            return
        revisions = ", ".join(_literal(value) for value in revision_ids)
        self.table.delete(f"corpus_id = {_literal(corpus_id)} AND revision_id IN ({revisions})")

    def revisions(self, corpus_id: str) -> set[str]:
        rows = self.table.search().where(f"corpus_id = {_literal(corpus_id)}").select(["revision_id"]).to_list()
        return {str(row["revision_id"]) for row in rows}

    def query(self, request: VectorQuery) -> list[VectorHit]:
        if len(request.vector) != self.dimension:
            raise ValueError("Query vector dimension does not match the LanceDB table.")
        if not request.active_revision_ids:
            return []
        revisions = ", ".join(_literal(value) for value in request.active_revision_ids)
        predicates = [
            f"corpus_id = {_literal(request.corpus_id)}",
            f"revision_id IN ({revisions})",
            f"embedding_fingerprint = {_literal(request.embedding_fingerprint)}",
        ]
        document_ids = list(request.filters.get("document_ids") or [])
        if document_ids:
            predicates.append(f"document_id IN ({', '.join(_literal(str(value)) for value in document_ids)})")
        source_types = list(request.filters.get("source_types") or [])
        if source_types:
            predicates.append(f"source_type IN ({', '.join(_literal(str(value)) for value in source_types)})")
        language = request.filters.get("language")
        if language:
            predicates.append(f"language = {_literal(str(language))}")
        rows = (
            self.table.search(list(request.vector), vector_column_name="vector")
            .where(" AND ".join(predicates))
            .limit(request.limit)
            .to_list()
        )
        return [
            VectorHit(
                str(row["record_id"]),
                1.0 / (1.0 + float(row.get("_distance") or 0.0)),
                {key: value for key, value in row.items() if key not in {"vector", "_distance"}},
            )
            for row in rows
        ]

    def health(self) -> dict[str, object]:
        return {
            "provider": "lancedb",
            "location": str(self.location),
            "table": self.table_name,
            "dimension": self.dimension,
            "rows": self.table.count_rows(),
        }


class ExternalVectorCandidateRetriever:
    def __init__(self, store: LanceDBVectorStore, corpus_id: str, embedding_fingerprint: str, backend: Any):
        self.store = store
        self.corpus_id = corpus_id
        self.embedding_fingerprint = embedding_fingerprint
        self.backend = backend

    def vector_search(self, embedding, limit, filters=None):
        from .models import Candidate

        hits = self.store.query(
            VectorQuery(
                self.corpus_id,
                tuple(self.backend.active_revision_ids()),
                self.embedding_fingerprint,
                tuple(float(value) for value in embedding),
                limit,
                dict(filters or {}),
            )
        )
        return [
            Candidate(
                chunk_id=str(hit.metadata["chunk_id"]),
                document_id=str(hit.metadata["document_id"]),
                parent_id=str(hit.metadata["parent_id"]),
                text=str(hit.metadata.get("text") or ""),
                title=str(hit.metadata.get("title") or ""),
                source_uri=str(hit.metadata.get("source_uri") or ""),
            )
            for hit in hits
        ]
