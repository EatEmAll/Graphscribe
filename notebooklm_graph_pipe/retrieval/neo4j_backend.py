from __future__ import annotations

import re
from typing import Any, Sequence

from .models import Candidate


class Neo4jRetrievalBackend:
    def __init__(
        self,
        driver: Any,
        database: str,
        corpus_id: str,
        *,
        retrieval_unit: str = "chunk",
        vector_index: str = "chunk_embedding_v1",
        keyword_index: str = "chunk_keyword_v1",
    ):
        if retrieval_unit not in {"chunk", "parent"}:
            raise ValueError(f"Unsupported retrieval unit: {retrieval_unit}")
        if any(not re.fullmatch(r"[A-Za-z0-9_]+", name) for name in (vector_index, keyword_index)):
            raise ValueError("Retrieval index names may contain only letters, digits, and underscores.")
        self.driver = driver
        self.database = database
        self.corpus_id = corpus_id
        self.retrieval_unit = retrieval_unit
        self.vector_index = vector_index
        self.keyword_index = keyword_index

    def _retrieval_match(self) -> str:
        if self.retrieval_unit == "parent":
            return (
                "MATCH (corpus:Corpus {id: $corpus_id})-[:HAS_DOCUMENT]->(document:Document)"
                "-[:ACTIVE_REVISION]->(revision:DocumentRevision)-[:HAS_PARENT]->(unit:ParentChunk)"
            )
        return (
            "MATCH (corpus:Corpus {id: $corpus_id})-[:HAS_DOCUMENT]->(document:Document)"
            "-[:ACTIVE_REVISION]->(revision:DocumentRevision)<-[:IN_REVISION]-(unit:Chunk)"
        )

    def _candidate_projection(self) -> str:
        if self.retrieval_unit == "parent":
            return (
                "unit.id AS chunk_id, unit.id AS parent_id, unit.text AS text, "
                "document.id AS document_id, document.title AS title, document.source_uri AS source_uri, "
                "unit.page_start AS page_start, unit.page_end AS page_end, "
                "unit.timestamp_start_ms AS timestamp_start_ms, "
                "unit.timestamp_end_ms AS timestamp_end_ms, unit.section_path AS section_path"
            )
        return (
            "unit.id AS chunk_id, unit.parent_id AS parent_id, unit.text AS text, "
            "document.id AS document_id, document.title AS title, document.source_uri AS source_uri, "
            "unit.page_start AS page_start, unit.page_end AS page_end, "
            "unit.timestamp_start_ms AS timestamp_start_ms, "
            "unit.timestamp_end_ms AS timestamp_end_ms, unit.section_path AS section_path"
        )

    def _session(self):
        return self.driver.session(database=self.database)

    def active_revision_ids(self) -> list[str]:
        with self._session() as session:
            return [
                str(row["revision_id"])
                for row in session.run(
                    """
                    MATCH (:Corpus {id: $corpus_id})-[:HAS_DOCUMENT]->(:Document)
                          -[:ACTIVE_REVISION]->(revision:DocumentRevision)
                    RETURN DISTINCT revision.id AS revision_id ORDER BY revision_id
                    """,
                    corpus_id=self.corpus_id,
                )
            ]

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
        retrieval_match = self._retrieval_match()
        projection = self._candidate_projection()
        with self._session() as session:
            query_limit = max(limit * 3, limit)
            total_indexed: int | None = None
            while True:
                rows = list(
                    session.run(
                        f"""
                        CALL db.index.vector.queryNodes('{self.vector_index}', $query_limit, $embedding)
                        YIELD node AS unit, score
                        {retrieval_match}
                        WHERE revision.vector_ready = true AND document.status = 'READY'
                          AND ($document_ids = [] OR document.id IN $document_ids)
                          AND ($source_types = [] OR document.source_type IN $source_types)
                          AND ($language IS NULL OR document.language = $language)
                        RETURN {projection}, score
                        ORDER BY score DESC
                        LIMIT $limit
                        """,
                        query_limit=query_limit,
                        embedding=list(embedding),
                        limit=limit,
                        corpus_id=self.corpus_id,
                        **filter_values,
                    )
                )
                candidates = [self._candidate(dict(row)) for row in rows]
                if len(candidates) >= limit:
                    return candidates
                if total_indexed is None:
                    count_row = session.run(
                        f"MATCH (n:{'ParentChunk' if self.retrieval_unit == 'parent' else 'Chunk'}) "
                        "WHERE n.embedding IS NOT NULL RETURN count(n) AS count"
                    ).single()
                    if count_row is None:
                        return candidates
                    total_indexed = int(count_row["count"])
                if query_limit >= total_indexed:
                    return candidates
                query_limit = min(total_indexed, query_limit * 4)

    def lexical_search(self, query: str, limit: int, filters: dict[str, Any] | None = None) -> list[Candidate]:
        filter_values = self._filters(filters)
        retrieval_match = self._retrieval_match()
        projection = self._candidate_projection()
        with self._session() as session:
            query_limit = max(limit * 3, limit)
            total_indexed: int | None = None
            while True:
                rows = list(
                    session.run(
                        f"""
                        CALL db.index.fulltext.queryNodes('{self.keyword_index}', $search_text, {{limit: $query_limit}})
                        YIELD node AS unit, score
                        {retrieval_match}
                        WHERE revision.vector_ready = true AND document.status = 'READY'
                          AND ($document_ids = [] OR document.id IN $document_ids)
                          AND ($source_types = [] OR document.source_type IN $source_types)
                          AND ($language IS NULL OR document.language = $language)
                        RETURN {projection}, score
                        ORDER BY score DESC
                        LIMIT $limit
                        """,
                        search_text=query,
                        query_limit=query_limit,
                        limit=limit,
                        corpus_id=self.corpus_id,
                        **filter_values,
                    )
                )
                candidates = [self._candidate(dict(row)) for row in rows]
                if len(candidates) >= limit:
                    return candidates
                if total_indexed is None:
                    count_row = session.run(
                        f"MATCH (n:{'ParentChunk' if self.retrieval_unit == 'parent' else 'Chunk'}) "
                        "WHERE n.text IS NOT NULL RETURN count(n) AS count"
                    ).single()
                    if count_row is None:
                        return candidates
                    total_indexed = int(count_row["count"])
                if query_limit >= total_indexed:
                    return candidates
                query_limit = min(total_indexed, query_limit * 4)

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
        if self.retrieval_unit == "parent":
            seed_match = (
                "MATCH (corpus:Corpus {id: $corpus_id})-[:HAS_DOCUMENT]->(seed_document:Document)"
                "-[:ACTIVE_REVISION]->(seed_revision:DocumentRevision)-[:HAS_PARENT]->"
                "(seed_parent:ParentChunk {id: seed_id})"
            )
            evidence_match = (
                "MATCH (reached)<-[evidence_mention:HAS_ENTITY]-(unit:ParentChunk)"
                "<-[:HAS_PARENT]-(revision:DocumentRevision)<-[:ACTIVE_REVISION]-(document:Document)"
                "<-[:HAS_DOCUMENT]-(corpus)"
            )
            candidate_projection = self._candidate_projection().replace("unit.", "unit.")
            seed_exclusion = "NOT unit.id IN $seed_ids"
        else:
            seed_match = (
                "MATCH (corpus:Corpus {id: $corpus_id})-[:HAS_DOCUMENT]->(seed_document:Document)"
                "-[:ACTIVE_REVISION]->(seed_revision:DocumentRevision)<-[:IN_REVISION]-"
                "(seed:Chunk {id: seed_id})\n"
                "MATCH (seed)<-[:HAS_CHILD]-(seed_parent:ParentChunk)"
            )
            evidence_match = (
                "MATCH (reached)<-[evidence_mention:HAS_ENTITY]-(evidence_parent:ParentChunk)-[:HAS_CHILD]->(unit:Chunk)\n"
                "MATCH (corpus)-[:HAS_DOCUMENT]->(document:Document)-[:ACTIVE_REVISION]->"
                "(revision:DocumentRevision)<-[:IN_REVISION]-(unit)"
            )
            candidate_projection = self._candidate_projection()
            seed_exclusion = "NOT unit.id IN $seed_ids"
        query = f"""
            UNWIND $seed_ids AS seed_id
            {seed_match}
            MATCH (seed_parent)-[seed_mention:HAS_ENTITY]->(origin:__Entity__)
            WHERE coalesce(seed_mention.extraction_state, 'VERIFIED') = 'VERIFIED'
              AND COUNT {{ (origin)<-[:HAS_ENTITY]-(:ParentChunk) }} <= $max_entity_degree
            MATCH path=(origin)-[*0..{max_path}]-(reached:__Entity__)
            {evidence_match}
            WHERE revision.graph_ready = true AND document.status = 'READY' AND {seed_exclusion}
              AND coalesce(evidence_mention.extraction_state, 'VERIFIED') = 'VERIFIED'
              AND ($document_ids = [] OR document.id IN $document_ids)
              AND ($source_types = [] OR document.source_type IN $source_types)
              AND ($language IS NULL OR document.language = $language)
              AND COUNT {{ (reached)<-[:HAS_ENTITY]-(:ParentChunk) }} <= $max_entity_degree
              AND all(rel IN relationships(path)
                  WHERE NOT type(rel) IN $excluded_relationships
                    AND any(parent_id IN coalesce(rel.source_parent_ids, [])
                            WHERE NOT parent_id IN coalesce(rel.provisional_parent_ids, []) AND EXISTS {{
                        MATCH (source_parent:ParentChunk {{id: parent_id}})<-[:HAS_PARENT]-(source_revision:DocumentRevision)<-[:ACTIVE_REVISION]-(:Document)<-[:HAS_DOCUMENT]-(corpus)
                    }}))
            RETURN DISTINCT {candidate_projection}, origin.id AS origin_entity,
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

    def community_reports(self, limit: int = 200) -> list[dict[str, Any]]:
        if not 1 <= limit <= 1000:
            raise ValueError("Community report limit must be between 1 and 1000.")
        with self._session() as session:
            rows = session.run(
                """
                MATCH (:Corpus {id: $corpus_id})-[:ACTIVE_COMMUNITY_BUILD]->(build:CommunityBuild)
                      -[:HAS_COMMUNITY]->(community:Community)-[:HAS_REPORT]->(report:CommunityReport)
                WITH build, min(community.level) AS broadest_level
                MATCH (build)-[:HAS_COMMUNITY]->(community:Community {level: broadest_level})
                      -[:HAS_REPORT]->(report:CommunityReport)
                OPTIONAL MATCH (report)-[:HAS_FINDING]->(finding:CommunityFinding)
                      -[:GROUNDED_IN]->(parent:ParentChunk)
                RETURN report.id AS report_id, community.id AS community_id,
                       report.title AS title, report.summary AS summary,
                       report.full_content AS full_content, report.rank AS rank,
                       collect(DISTINCT {summary: finding.summary,
                                         explanation: finding.explanation,
                                         confidence: finding.confidence,
                                         parent_id: parent.id}) AS findings
                ORDER BY rank DESC, report_id
                LIMIT $limit
                """,
                corpus_id=self.corpus_id,
                limit=limit,
            )
            return [dict(row) for row in rows]

    def search_community_reports(
        self,
        query: str,
        embedding: Sequence[float],
        limit: int = 12,
    ) -> list[dict[str, Any]]:
        if not 1 <= limit <= 50:
            raise ValueError("Community report search limit must be between 1 and 50.")
        with self._session() as session:
            overfetch = max(limit * 4, limit)
            total_indexed: int | None = None
            while True:
                rows = list(
                    session.run(
                        """
                        CALL {
                            CALL db.index.vector.queryNodes(
                                'community_report_embedding_v1', $overfetch, $embedding
                            ) YIELD node AS report, score
                            RETURN report, score, 'vector' AS channel
                            UNION ALL
                            CALL db.index.fulltext.queryNodes(
                                'community_report_keyword_v1', $query, {limit: $overfetch}
                            ) YIELD node AS report, score
                            RETURN report, score, 'lexical' AS channel
                        }
                        MATCH (:Corpus {id: $corpus_id})-[:ACTIVE_COMMUNITY_BUILD]->(:CommunityBuild)
                              -[:HAS_COMMUNITY]->(community:Community)-[:HAS_REPORT]->(report)
                        WITH report, community, collect(DISTINCT channel) AS channels, max(score) AS score
                        OPTIONAL MATCH (report)-[:HAS_FINDING]->(finding:CommunityFinding)
                              -[:GROUNDED_IN]->(parent:ParentChunk)
                        RETURN report.id AS report_id, community.id AS community_id,
                               report.title AS title, report.summary AS summary,
                               report.full_content AS full_content, report.rank AS rank,
                               channels, score,
                               collect(DISTINCT {summary: finding.summary,
                                                 explanation: finding.explanation,
                                                 confidence: finding.confidence,
                                                 parent_id: parent.id}) AS findings
                        ORDER BY score DESC, rank DESC, report_id
                        LIMIT $limit
                        """,
                        corpus_id=self.corpus_id,
                        query=query,
                        embedding=list(embedding),
                        overfetch=overfetch,
                        limit=limit,
                    )
                )
                reports = [dict(row) for row in rows]
                if len(reports) >= limit:
                    return reports
                if total_indexed is None:
                    count_row = session.run(
                        "MATCH (report:CommunityReport) RETURN count(report) AS count"
                    ).single()
                    total_indexed = int(count_row["count"]) if count_row else 0
                if overfetch >= total_indexed:
                    return reports
                overfetch = min(total_indexed, overfetch * 4)

    def graph_neighbors(self, entity_id: str, hops: int = 1, limit: int = 50) -> list[dict[str, Any]]:
        if hops not in {1, 2}:
            raise ValueError("Graph neighbor exploration supports one or two hops.")
        if not 1 <= limit <= 200:
            raise ValueError("Graph neighbor limit must be between 1 and 200.")
        query = f"""
            MATCH (corpus:Corpus {{id: $corpus_id}})-[:HAS_DOCUMENT]->(:Document)-[:ACTIVE_REVISION]->(:DocumentRevision)-[:HAS_PARENT]->(:ParentChunk)-[origin_mention:HAS_ENTITY]->(origin:__Entity__ {{id: $entity_id}})
            WHERE coalesce(origin_mention.extraction_state, 'VERIFIED') = 'VERIFIED'
            MATCH path=(origin)-[*1..{hops}]-(neighbor:__Entity__)
            WHERE EXISTS {{
                MATCH (corpus)-[:HAS_DOCUMENT]->(:Document)-[:ACTIVE_REVISION]->(:DocumentRevision)-[:HAS_PARENT]->(:ParentChunk)-[neighbor_mention:HAS_ENTITY]->(neighbor)
                WHERE coalesce(neighbor_mention.extraction_state, 'VERIFIED') = 'VERIFIED'
            }}
              AND all(rel IN relationships(path)
                  WHERE NOT type(rel) IN $excluded_relationships
                    AND any(parent_id IN coalesce(rel.source_parent_ids, [])
                            WHERE NOT parent_id IN coalesce(rel.provisional_parent_ids, []) AND EXISTS {{
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
