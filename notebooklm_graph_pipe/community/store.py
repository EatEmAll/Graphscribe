from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any, Sequence

from .models import (
    CommunityBuildResult,
    CommunityProjection,
    EntityRecord,
    EvidenceParent,
    RelationshipRecord,
    stable_hash,
)


class Neo4jCommunityStore:
    def __init__(self, driver: Any, database: str, corpus_id: str, *, batch_size: int = 500):
        self.driver = driver
        self.database = database
        self.corpus_id = corpus_id
        self.batch_size = batch_size

    def _session(self):
        return self.driver.session(database=self.database)

    def active_projection(self) -> CommunityProjection:
        with self._session() as session:
            revision_rows = list(
                session.run(
                    """
                    MATCH (:Corpus {id: $corpus_id})-[:HAS_DOCUMENT]->(document:Document)
                          -[:ACTIVE_REVISION]->(revision:DocumentRevision)
                    RETURN document.id AS document_id, revision.id AS revision_id
                    ORDER BY document_id, revision_id
                    """,
                    corpus_id=self.corpus_id,
                )
            )
            parent_rows = list(
                session.run(
                    """
                    MATCH (:Corpus {id: $corpus_id})-[:HAS_DOCUMENT]->(document:Document)
                          -[:ACTIVE_REVISION]->(revision:DocumentRevision)-[:HAS_PARENT]->(parent:ParentChunk)
                    RETURN parent.id AS id, document.id AS document_id, revision.id AS revision_id,
                           parent.text AS text, document.title AS title, document.source_uri AS source_uri
                    ORDER BY id
                    """,
                    corpus_id=self.corpus_id,
                )
            )
            entity_rows = list(
                session.run(
                    """
                    MATCH (:Corpus {id: $corpus_id})-[:HAS_DOCUMENT]->(:Document)
                          -[:ACTIVE_REVISION]->(:DocumentRevision)-[:HAS_PARENT]->(parent:ParentChunk)
                          -[mention:HAS_ENTITY]->(entity:__Entity__)
                    WHERE coalesce(mention.extraction_state, 'VERIFIED') = 'VERIFIED'
                    RETURN entity.id AS id, coalesce(entity.title, entity.id) AS title,
                           coalesce(entity.description, '') AS description,
                           collect(DISTINCT parent.id) AS parent_ids
                    ORDER BY id
                    """,
                    corpus_id=self.corpus_id,
                )
            )
            relationship_rows = list(
                session.run(
                    """
                    MATCH (:Corpus {id: $corpus_id})-[:HAS_DOCUMENT]->(:Document)
                          -[:ACTIVE_REVISION]->(:DocumentRevision)-[:HAS_PARENT]->(active_parent:ParentChunk)
                    WITH collect(DISTINCT active_parent.id) AS active_parent_ids
                    MATCH (source:__Entity__)-[relationship]->(target:__Entity__)
                    WHERE NOT type(relationship) IN $excluded_relationships
                      AND coalesce(relationship.legacy_unscoped, false) = false
                    WITH source, relationship, target,
                         [parent_id IN coalesce(relationship.source_parent_ids, [])
                          WHERE parent_id IN active_parent_ids
                            AND NOT parent_id IN coalesce(relationship.provisional_parent_ids, [])] AS parent_ids
                    WHERE size(parent_ids) > 0
                    RETURN source.id AS source_id, target.id AS target_id,
                           type(relationship) AS relationship_type,
                           coalesce(relationship.description, '') AS description,
                           parent_ids
                    ORDER BY source_id, target_id, relationship_type
                    """,
                    corpus_id=self.corpus_id,
                    excluded_relationships=[
                        "HAS_ENTITY",
                        "PART_OF",
                        "NEXT_CHUNK",
                        "IN_REVISION",
                        "HAS_CHILD",
                        "MEMBER_OF",
                    ],
                )
            )

        parents = tuple(
            EvidenceParent(
                str(row["id"]),
                str(row["document_id"]),
                str(row["revision_id"]),
                str(row.get("text") or ""),
                str(row.get("title") or ""),
                str(row.get("source_uri") or ""),
            )
            for row in parent_rows
        )
        entities = tuple(
            EntityRecord(
                str(row["id"]),
                str(row.get("title") or row["id"]),
                str(row.get("description") or ""),
                tuple(sorted(str(value) for value in row.get("parent_ids") or [])),
            )
            for row in entity_rows
        )
        entity_ids = {entity.id for entity in entities}
        relationships: list[RelationshipRecord] = []
        for row in relationship_rows:
            source_id = str(row["source_id"])
            target_id = str(row["target_id"])
            parent_ids = tuple(sorted(set(str(value) for value in row.get("parent_ids") or [])))
            if source_id == target_id or source_id not in entity_ids or target_id not in entity_ids or not parent_ids:
                continue
            relationships.append(
                RelationshipRecord(
                    stable_hash(
                        {
                            "source": source_id,
                            "target": target_id,
                            "type": str(row["relationship_type"]),
                        }
                    ),
                    source_id,
                    target_id,
                    str(row.get("description") or row["relationship_type"]),
                    float(len(parent_ids)),
                    parent_ids,
                )
            )
        return CommunityProjection(
            self.corpus_id,
            tuple(str(row["revision_id"]) for row in revision_rows),
            parents,
            entities,
            tuple(relationships),
        )

    def stage_build(self, build: CommunityBuildResult, metadata: dict[str, Any]) -> None:
        if build.corpus_id != self.corpus_id:
            raise ValueError("Community build belongs to a different corpus.")
        community_rows = [
            {
                "id": item.id,
                "level": item.level,
                "source_cluster": item.source_cluster,
                "parent_id": item.parent_id,
                "member_ids": list(item.member_ids),
                "rank": item.rank,
            }
            for item in build.communities
        ]
        report_rows = []
        finding_rows = []
        for report in build.reports:
            report_rows.append(
                {
                    "id": report.id,
                    "community_id": report.community_id,
                    "title": report.title,
                    "summary": report.summary,
                    "full_content": report.full_content,
                    "rank": report.rank,
                    "rating_explanation": report.rating_explanation,
                    "raw_json": json.dumps(report.raw, ensure_ascii=True, sort_keys=True),
                    "embedding": list(report.embedding),
                }
            )
            finding_rows.extend(
                {
                    "id": finding.id,
                    "report_id": report.id,
                    "position": finding.position,
                    "summary": finding.summary,
                    "explanation": finding.explanation,
                    "confidence": finding.confidence,
                    "parent_ids": list(finding.parent_ids),
                }
                for finding in report.findings
            )
        with self._session() as session:
            active = session.run(
                """
                MATCH (corpus:Corpus {id: $corpus_id})-[:HAS_COMMUNITY_BUILD]->
                      (build:CommunityBuild {id: $build_id})
                WHERE build.status = 'ACTIVE'
                   OR (corpus)-[:ACTIVE_COMMUNITY_BUILD]->(build)
                RETURN build.id AS id
                """,
                corpus_id=self.corpus_id,
                build_id=build.id,
            ).single()
            if active:
                raise RuntimeError("The active community build cannot be restaged.")
            session.run(
                """
                MATCH (:Corpus {id: $corpus_id})-[:HAS_COMMUNITY_BUILD]->
                      (build:CommunityBuild {id: $build_id})
                OPTIONAL MATCH (build)-[:HAS_COMMUNITY]->(community:Community)
                OPTIONAL MATCH (community)-[:HAS_REPORT]->(report:CommunityReport)
                OPTIONAL MATCH (report)-[:HAS_FINDING]->(finding:CommunityFinding)
                WITH collect(DISTINCT finding) AS findings, collect(DISTINCT report) AS reports,
                     collect(DISTINCT community) AS communities, build
                FOREACH (node IN findings | DETACH DELETE node)
                FOREACH (node IN reports | DETACH DELETE node)
                FOREACH (node IN communities | DETACH DELETE node)
                DETACH DELETE build
                """,
                corpus_id=self.corpus_id,
                build_id=build.id,
            ).consume()
            session.run(
                """
                MATCH (corpus:Corpus {id: $corpus_id})
                MERGE (build:CommunityBuild {id: $build_id})
                ON CREATE SET build.created_at = datetime()
                SET build.status = 'BUILDING', build.corpus_id = $corpus_id,
                    build.active_revision_hash = $active_revision_hash,
                    build.active_revision_ids = $active_revision_ids,
                    build.projection_fingerprint = $projection_fingerprint,
                    build.configuration_fingerprint = $configuration_fingerprint,
                    build.metadata_json = $metadata_json, build.error = null
                MERGE (corpus)-[:HAS_COMMUNITY_BUILD]->(build)
                """,
                corpus_id=self.corpus_id,
                build_id=build.id,
                active_revision_hash=build.active_revision_hash,
                active_revision_ids=list(build.active_revision_ids),
                projection_fingerprint=build.projection_fingerprint,
                configuration_fingerprint=build.configuration_fingerprint,
                metadata_json=json.dumps(metadata, ensure_ascii=True, sort_keys=True),
            ).consume()
            for rows in self._batches(community_rows):
                session.run(
                    """
                    MATCH (build:CommunityBuild {id: $build_id})
                    UNWIND $rows AS row
                    MERGE (community:Community {id: row.id})
                    SET community.build_id = $build_id, community.level = row.level,
                        community.source_cluster = row.source_cluster,
                        community.member_ids = row.member_ids, community.rank = row.rank,
                        community.parent_id = row.parent_id
                    MERGE (build)-[:HAS_COMMUNITY]->(community)
                    WITH row, community
                    UNWIND row.member_ids AS member_id
                    MATCH (:Corpus {id: $corpus_id})-[:HAS_DOCUMENT]->(:Document)
                          -[:ACTIVE_REVISION]->(:DocumentRevision)-[:HAS_PARENT]->(:ParentChunk)
                          -[mention:HAS_ENTITY]->(entity:__Entity__ {id: member_id})
                    WHERE coalesce(mention.extraction_state, 'VERIFIED') = 'VERIFIED'
                    MERGE (entity)-[:MEMBER_OF {build_id: $build_id}]->(community)
                    """,
                    corpus_id=self.corpus_id,
                    build_id=build.id,
                    rows=rows,
                ).consume()
            session.run(
                """
                MATCH (build:CommunityBuild {id: $build_id})-[:HAS_COMMUNITY]->(child:Community)
                WHERE child.id IN $child_ids
                WITH build, child
                MATCH (build)-[:HAS_COMMUNITY]->(parent:Community {id: child.parent_id})
                MERGE (child)-[:CHILD_OF]->(parent)
                """,
                build_id=build.id,
                child_ids=[row["id"] for row in community_rows if row["parent_id"]],
            ).consume()
            for rows in self._batches(report_rows):
                session.run(
                    """
                    MATCH (build:CommunityBuild {id: $build_id})-[:HAS_COMMUNITY]->(community:Community)
                    UNWIND $rows AS row
                    WITH build, community, row WHERE community.id = row.community_id
                    MERGE (report:CommunityReport {id: row.id})
                    SET report += row, report.build_id = $build_id
                    REMOVE report.community_id
                    MERGE (community)-[:HAS_REPORT]->(report)
                    """,
                    build_id=build.id,
                    rows=rows,
                ).consume()
            for rows in self._batches(finding_rows):
                session.run(
                    """
                    UNWIND $rows AS row
                    MATCH (report:CommunityReport {id: row.report_id, build_id: $build_id})
                    MERGE (finding:CommunityFinding {id: row.id})
                    SET finding.position = row.position, finding.summary = row.summary,
                        finding.explanation = row.explanation, finding.confidence = row.confidence,
                        finding.build_id = $build_id
                    MERGE (report)-[:HAS_FINDING]->(finding)
                    WITH finding, row
                    UNWIND row.parent_ids AS parent_id
                    MATCH (:Corpus {id: $corpus_id})-[:HAS_DOCUMENT]->(:Document)
                          -[:ACTIVE_REVISION]->(:DocumentRevision)-[:HAS_PARENT]->
                          (parent:ParentChunk {id: parent_id})
                    MERGE (finding)-[:GROUNDED_IN]->(parent)
                    """,
                    corpus_id=self.corpus_id,
                    build_id=build.id,
                    rows=rows,
                ).consume()

    def activate_build(self, build_id: str, expected_active_revision_hash: str) -> None:
        projection = self.active_projection()
        if projection.active_revision_hash != expected_active_revision_hash:
            raise RuntimeError("Active corpus revisions changed while the community build was running.")
        with self._session() as session:
            result = session.run(
                """
                MATCH (corpus:Corpus {id: $corpus_id})-[:HAS_COMMUNITY_BUILD]->(build:CommunityBuild {id: $build_id})
                WHERE build.status = 'BUILDING' AND build.active_revision_hash = $active_revision_hash
                MATCH (corpus)-[:HAS_DOCUMENT]->(:Document)-[:ACTIVE_REVISION]->
                      (active_revision:DocumentRevision)
                WITH corpus, build, collect(DISTINCT active_revision.id) AS current_revision_ids
                WHERE size(current_revision_ids) = size(build.active_revision_ids)
                  AND all(revision_id IN current_revision_ids
                          WHERE revision_id IN build.active_revision_ids)
                MATCH (build)-[:HAS_COMMUNITY]->(community:Community)-[:HAS_REPORT]->(report:CommunityReport)
                MATCH (report)-[:HAS_FINDING]->(finding:CommunityFinding)-[:GROUNDED_IN]->(:ParentChunk)
                WITH corpus, build, count(DISTINCT community) AS reported_communities,
                     count(DISTINCT finding) AS grounded_findings
                MATCH (build)-[:HAS_COMMUNITY]->(all_community:Community)
                WITH corpus, build, reported_communities, grounded_findings,
                     count(DISTINCT all_community) AS all_communities
                WHERE all_communities > 0 AND reported_communities = all_communities AND grounded_findings > 0
                OPTIONAL MATCH (corpus)-[active:ACTIVE_COMMUNITY_BUILD]->(previous:CommunityBuild)
                DELETE active
                FOREACH (_ IN CASE WHEN previous IS NULL THEN [] ELSE [1] END |
                    SET previous.status = 'INACTIVE', previous.deactivated_at = datetime())
                MERGE (corpus)-[:ACTIVE_COMMUNITY_BUILD]->(build)
                SET build.status = 'ACTIVE', build.activated_at = datetime()
                RETURN build.id AS id
                """,
                corpus_id=self.corpus_id,
                build_id=build_id,
                active_revision_hash=expected_active_revision_hash,
            ).single()
            if not result:
                raise RuntimeError("Community build activation validation failed.")

    def rollback_build(self, build_id: str) -> None:
        with self._session() as session:
            result = session.run(
                """
                MATCH (corpus:Corpus {id: $corpus_id})-[:HAS_COMMUNITY_BUILD]->(build:CommunityBuild {id: $build_id})
                WHERE build.status IN ['ACTIVE', 'INACTIVE']
                MATCH (corpus)-[:HAS_DOCUMENT]->(:Document)-[:ACTIVE_REVISION]->
                      (active_revision:DocumentRevision)
                WITH corpus, build, collect(DISTINCT active_revision.id) AS current_revision_ids
                WHERE size(current_revision_ids) = size(build.active_revision_ids)
                  AND all(revision_id IN current_revision_ids
                          WHERE revision_id IN build.active_revision_ids)
                OPTIONAL MATCH (corpus)-[active:ACTIVE_COMMUNITY_BUILD]->(current:CommunityBuild)
                DELETE active
                FOREACH (_ IN CASE WHEN current IS NULL OR current = build THEN [] ELSE [1] END |
                    SET current.status = 'INACTIVE', current.deactivated_at = datetime())
                MERGE (corpus)-[:ACTIVE_COMMUNITY_BUILD]->(build)
                SET build.status = 'ACTIVE', build.activated_at = datetime()
                RETURN build.id AS id
                """,
                corpus_id=self.corpus_id,
                build_id=build_id,
            ).single()
            if not result:
                raise RuntimeError(f"Community build cannot be restored: {build_id}")

    def active_build(self) -> dict[str, Any] | None:
        with self._session() as session:
            row = session.run(
                """
                MATCH (:Corpus {id: $corpus_id})-[:ACTIVE_COMMUNITY_BUILD]->(build:CommunityBuild)
                RETURN properties(build) AS build
                """,
                corpus_id=self.corpus_id,
            ).single()
        return dict(row["build"]) if row else None

    def garbage_collect_builds(self, build_ids: Sequence[str]) -> dict[str, int]:
        if not build_ids:
            return {"builds": 0, "communities": 0, "reports": 0, "findings": 0}
        with self._session() as session:
            row = session.run(
                """
                MATCH (corpus:Corpus {id: $corpus_id})-[:HAS_COMMUNITY_BUILD]->(build:CommunityBuild)
                WHERE build.id IN $build_ids
                  AND NOT (corpus)-[:ACTIVE_COMMUNITY_BUILD]->(build)
                  AND build.status IN ['INACTIVE', 'FAILED', 'STALE']
                OPTIONAL MATCH (build)-[:HAS_COMMUNITY]->(community:Community)
                OPTIONAL MATCH (community)-[:HAS_REPORT]->(report:CommunityReport)
                OPTIONAL MATCH (report)-[:HAS_FINDING]->(finding:CommunityFinding)
                WITH collect(DISTINCT build) AS builds, collect(DISTINCT community) AS communities,
                     collect(DISTINCT report) AS reports, collect(DISTINCT finding) AS findings
                FOREACH (node IN findings | DETACH DELETE node)
                FOREACH (node IN reports | DETACH DELETE node)
                FOREACH (node IN communities | DETACH DELETE node)
                FOREACH (node IN builds | DETACH DELETE node)
                RETURN size(builds) AS builds, size(communities) AS communities,
                       size(reports) AS reports, size(findings) AS findings
                """,
                corpus_id=self.corpus_id,
                build_ids=list(build_ids),
            ).single()
        return dict(row) if row else {"builds": 0, "communities": 0, "reports": 0, "findings": 0}

    def _batches(self, rows: Sequence[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        return [list(rows[start : start + self.batch_size]) for start in range(0, len(rows), self.batch_size)]
