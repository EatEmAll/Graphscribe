from __future__ import annotations

from typing import Any, Sequence

from .models import Candidate


class Neo4jRetrievalBackend:
    def __init__(self, driver: Any, database: str, corpus_id: str):
        self.driver = driver
        self.database = database
        self.corpus_id = corpus_id

    def _session(self):
        return self.driver.session(database=self.database)

    @staticmethod
    def _candidate(row: dict[str, Any]) -> Candidate:
        return Candidate(
            chunk_id=str(row["chunk_id"]),
            document_id=str(row["document_id"]),
            parent_id=str(row["parent_id"]),
            text=str(row.get("text") or ""),
            title=str(row.get("title") or ""),
            source_uri=str(row.get("source_uri") or ""),
            page_start=row.get("page_start"),
            page_end=row.get("page_end"),
            timestamp_start_ms=row.get("timestamp_start_ms"),
            timestamp_end_ms=row.get("timestamp_end_ms"),
            section_path=tuple(row.get("section_path") or ()),
        )

    @staticmethod
    def _filters(filters: dict[str, Any] | None) -> dict[str, Any]:
        value = filters or {}
        return {
            "document_ids": list(value.get("document_ids") or []),
            "source_types": list(value.get("source_types") or []),
            "language": value.get("language"),
        }

    def vector_search(
        self,
        embedding: Sequence[float],
        limit: int,
        filters: dict[str, Any] | None = None,
    ) -> list[Candidate]:
        filter_values = self._filters(filters)
        with self._session() as session:
            rows = session.run(
                """
                CALL db.index.vector.queryNodes('chunk_embedding_v1', $query_limit, $embedding)
                YIELD node AS chunk, score
                MATCH (corpus:Corpus {id: $corpus_id})-[:HAS_DOCUMENT]->(document:Document)-[:ACTIVE_REVISION]->(revision:DocumentRevision)<-[:IN_REVISION]-(chunk)
                WHERE revision.vector_ready = true AND document.status = 'READY'
                  AND ($document_ids = [] OR document.id IN $document_ids)
                  AND ($source_types = [] OR document.source_type IN $source_types)
                  AND ($language IS NULL OR document.language = $language)
                RETURN chunk.id AS chunk_id, chunk.parent_id AS parent_id, chunk.text AS text,
                       document.id AS document_id, document.title AS title, document.source_uri AS source_uri,
                       chunk.page_start AS page_start, chunk.page_end AS page_end,
                       chunk.timestamp_start_ms AS timestamp_start_ms,
                       chunk.timestamp_end_ms AS timestamp_end_ms,
                       chunk.section_path AS section_path, score
                ORDER BY score DESC
                LIMIT $limit
                """,
                query_limit=max(limit * 3, limit),
                embedding=list(embedding),
                limit=limit,
                corpus_id=self.corpus_id,
                **filter_values,
            )
            return [self._candidate(dict(row)) for row in rows]

    def lexical_search(self, query: str, limit: int, filters: dict[str, Any] | None = None) -> list[Candidate]:
        filter_values = self._filters(filters)
        with self._session() as session:
            rows = session.run(
                """
                CALL db.index.fulltext.queryNodes('chunk_keyword_v1', $search_text, {limit: $query_limit})
                YIELD node AS chunk, score
                MATCH (corpus:Corpus {id: $corpus_id})-[:HAS_DOCUMENT]->(document:Document)-[:ACTIVE_REVISION]->(revision:DocumentRevision)<-[:IN_REVISION]-(chunk)
                WHERE revision.vector_ready = true AND document.status = 'READY'
                  AND ($document_ids = [] OR document.id IN $document_ids)
                  AND ($source_types = [] OR document.source_type IN $source_types)
                  AND ($language IS NULL OR document.language = $language)
                RETURN chunk.id AS chunk_id, chunk.parent_id AS parent_id, chunk.text AS text,
                       document.id AS document_id, document.title AS title, document.source_uri AS source_uri,
                       chunk.page_start AS page_start, chunk.page_end AS page_end,
                       chunk.timestamp_start_ms AS timestamp_start_ms,
                       chunk.timestamp_end_ms AS timestamp_end_ms,
                       chunk.section_path AS section_path, score
                ORDER BY score DESC
                LIMIT $limit
                """,
                search_text=query,
                query_limit=max(limit * 3, limit),
                limit=limit,
                corpus_id=self.corpus_id,
                **filter_values,
            )
            return [self._candidate(dict(row)) for row in rows]

    def graph_expand(
        self,
        seed_ids: Sequence[str],
        hops: int,
        limit: int,
        filters: dict[str, Any] | None = None,
    ) -> list[Candidate]:
        if hops not in {1, 2}:
            raise ValueError("Graph expansion supports one or two hops.")
        max_path = hops
        query = f"""
            UNWIND $seed_ids AS seed_id
            MATCH (corpus:Corpus {{id: $corpus_id}})-[:HAS_DOCUMENT]->(seed_document:Document)-[:ACTIVE_REVISION]->(seed_revision:DocumentRevision)<-[:IN_REVISION]-(seed:Chunk {{id: seed_id}})
            MATCH (seed:Chunk {{id: seed_id}})<-[:HAS_CHILD]-(seed_parent:ParentChunk)-[:HAS_ENTITY]->(origin:__Entity__)
            WHERE COUNT {{ (origin)<-[:HAS_ENTITY]-(:ParentChunk) }} <= $max_entity_degree
            MATCH path=(origin)-[*0..{max_path}]-(reached:__Entity__)
            MATCH (reached)<-[:HAS_ENTITY]-(evidence_parent:ParentChunk)-[:HAS_CHILD]->(chunk:Chunk)
            MATCH (corpus)-[:HAS_DOCUMENT]->(document:Document)-[:ACTIVE_REVISION]->(revision:DocumentRevision)<-[:IN_REVISION]-(chunk)
            WHERE revision.graph_ready = true AND document.status = 'READY' AND NOT chunk.id IN $seed_ids
              AND ($document_ids = [] OR document.id IN $document_ids)
              AND ($source_types = [] OR document.source_type IN $source_types)
              AND ($language IS NULL OR document.language = $language)
              AND COUNT {{ (reached)<-[:HAS_ENTITY]-(:ParentChunk) }} <= $max_entity_degree
              AND all(rel IN relationships(path)
                  WHERE NOT type(rel) IN $excluded_relationships
                    AND any(parent_id IN coalesce(rel.source_parent_ids, []) WHERE EXISTS {{
                        MATCH (source_parent:ParentChunk {{id: parent_id}})<-[:HAS_PARENT]-(source_revision:DocumentRevision)<-[:ACTIVE_REVISION]-(:Document)<-[:HAS_DOCUMENT]-(corpus)
                    }}))
            RETURN DISTINCT chunk.id AS chunk_id, chunk.parent_id AS parent_id, chunk.text AS text,
                   document.id AS document_id, document.title AS title, document.source_uri AS source_uri,
                   chunk.page_start AS page_start, chunk.page_end AS page_end,
                   chunk.timestamp_start_ms AS timestamp_start_ms,
                   chunk.timestamp_end_ms AS timestamp_end_ms,
                   chunk.section_path AS section_path, origin.id AS origin_entity,
                   reached.id AS reached_entity, length(path) AS hops
            ORDER BY hops ASC
            LIMIT $limit
        """
        with self._session() as session:
            rows = list(
                session.run(
                    query,
                    seed_ids=list(seed_ids),
                    corpus_id=self.corpus_id,
                    limit=limit * 5,
                    max_entity_degree=1000,
                    excluded_relationships=["HAS_ENTITY", "PART_OF", "NEXT_CHUNK", "IN_REVISION", "HAS_CHILD"],
                    **self._filters(filters),
                )
            )
        candidates: dict[str, Candidate] = {}
        per_entity: dict[str, int] = {}
        per_document: dict[str, int] = {}
        for raw in rows:
            row = dict(raw)
            entity = str(row.get("reached_entity") or "")
            document = str(row["document_id"])
            if per_entity.get(entity, 0) >= 5 or per_document.get(document, 0) >= 10:
                continue
            candidate = candidates.get(str(row["chunk_id"]))
            if candidate is None:
                candidate = self._candidate(row)
                candidates[candidate.chunk_id] = candidate
                per_entity[entity] = per_entity.get(entity, 0) + 1
                per_document[document] = per_document.get(document, 0) + 1
            candidate.graph_paths.append(
                {
                    "origin_entity": row.get("origin_entity"),
                    "reached_entity": row.get("reached_entity"),
                    "hops": row.get("hops"),
                }
            )
            if len(candidates) >= limit:
                break
        return list(candidates.values())

    def parent_contexts(self, parent_ids: Sequence[str]) -> dict[str, dict[str, Any]]:
        with self._session() as session:
            rows = session.run(
                """
                UNWIND $parent_ids AS parent_id
                MATCH (corpus:Corpus {id: $corpus_id})-[:HAS_DOCUMENT]->(document:Document)-[:ACTIVE_REVISION]->(revision:DocumentRevision)-[:HAS_PARENT]->(parent:ParentChunk {id: parent_id})
                WHERE revision.vector_ready = true AND document.status = 'READY'
                RETURN parent.id AS parent_id, parent.text AS text,
                       parent.page_start AS page_start, parent.page_end AS page_end,
                       parent.timestamp_start_ms AS timestamp_start_ms,
                       parent.timestamp_end_ms AS timestamp_end_ms,
                       parent.section_path AS section_path,
                       document.id AS document_id, document.title AS title, document.source_uri AS source_uri
                """,
                parent_ids=list(parent_ids),
                corpus_id=self.corpus_id,
            )
            return {str(row["parent_id"]): dict(row) for row in rows}

    def graph_neighbors(self, entity_id: str, hops: int = 1, limit: int = 50) -> list[dict[str, Any]]:
        if hops not in {1, 2}:
            raise ValueError("Graph neighbor exploration supports one or two hops.")
        if not 1 <= limit <= 200:
            raise ValueError("Graph neighbor limit must be between 1 and 200.")
        query = f"""
            MATCH (corpus:Corpus {{id: $corpus_id}})-[:HAS_DOCUMENT]->(:Document)-[:ACTIVE_REVISION]->(:DocumentRevision)-[:HAS_PARENT]->(:ParentChunk)-[:HAS_ENTITY]->(origin:__Entity__ {{id: $entity_id}})
            MATCH path=(origin)-[*1..{hops}]-(neighbor:__Entity__)
            WHERE EXISTS {{
                MATCH (corpus)-[:HAS_DOCUMENT]->(:Document)-[:ACTIVE_REVISION]->(:DocumentRevision)-[:HAS_PARENT]->(:ParentChunk)-[:HAS_ENTITY]->(neighbor)
            }}
              AND all(rel IN relationships(path)
                  WHERE NOT type(rel) IN $excluded_relationships
                    AND any(parent_id IN coalesce(rel.source_parent_ids, []) WHERE EXISTS {{
                        MATCH (source_parent:ParentChunk {{id: parent_id}})<-[:HAS_PARENT]-(source_revision:DocumentRevision)<-[:ACTIVE_REVISION]-(:Document)<-[:HAS_DOCUMENT]-(corpus)
                    }}))
            RETURN DISTINCT neighbor.id AS entity_id, labels(neighbor) AS labels,
                   neighbor.description AS description, length(path) AS hops,
                   [relationship IN relationships(path) | type(relationship)] AS relationships
            ORDER BY hops, entity_id
            LIMIT $limit
        """
        with self._session() as session:
            return [
                dict(row)
                for row in session.run(
                    query,
                    entity_id=entity_id,
                    corpus_id=self.corpus_id,
                    limit=limit,
                    excluded_relationships=["HAS_ENTITY", "PART_OF", "NEXT_CHUNK", "IN_REVISION", "HAS_CHILD"],
                )
            ]
