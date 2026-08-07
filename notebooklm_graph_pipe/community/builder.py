from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict
from typing import Any, Awaitable, Callable, Protocol

from notebooklm_graph_pipe.runtime.llm_routing import COMMUNITY_REPORT_ROLE
from notebooklm_graph_pipe.runtime.model_executor import ModelExecutor, ModelRequest

from .models import (
    CommunityBuildResult,
    CommunityConfig,
    CommunityFinding,
    CommunityProjection,
    CommunityRecord,
    CommunityReport,
    stable_hash,
)

REPORT_SCHEMA: dict[str, object] = {
    "type": "object",
    "required": ["title", "summary", "rank", "rating_explanation", "findings"],
    "properties": {
        "title": {"type": "string"},
        "summary": {"type": "string"},
        "rank": {"type": "number"},
        "rating_explanation": {"type": "string"},
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["summary", "explanation", "source_parent_ids"],
                "properties": {
                    "summary": {"type": "string"},
                    "explanation": {"type": "string"},
                    "source_parent_ids": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                    "confidence": {"type": "number"},
                },
            },
        },
    },
}


class CommunityReporter(Protocol):
    async def report(self, community: CommunityRecord, context: dict[str, Any]) -> dict[str, Any]: ...


class ModelCommunityReporter:
    def __init__(self, executor: ModelExecutor):
        self.executor = executor

    async def report(self, community: CommunityRecord, context: dict[str, Any]) -> dict[str, Any]:
        request = ModelRequest(
            role=COMMUNITY_REPORT_ROLE,
            prompt=(
                "Create a source-grounded community report from the supplied JSON context. "
                "Every finding must cite one or more source_parent_ids present in the context.\n\n"
                + json.dumps(context, ensure_ascii=False, sort_keys=True)
            ),
            system_instruction="Return only the requested structured community report.",
            response_schema=REPORT_SCHEMA,
            max_output_tokens=4096,
            cache_namespace="community-report",
            idempotency_key=community.id,
        )
        result = await self.executor.aexecute_json(request)
        if result.payload is None:
            raise RuntimeError(f"Community report generation returned no payload for {community.id}.")
        return result.payload


def _default_clusterer(edges: list[tuple[str, str, float]], config: CommunityConfig) -> list[Any]:
    try:
        import graspologic_native
    except ImportError as exc:
        raise RuntimeError("Install graspologic-native>=1.2,<1.3 for community builds.") from exc
    return list(
        graspologic_native.hierarchical_leiden(
            edges=edges,
            max_cluster_size=config.max_cluster_size,
            seed=config.seed,
            starting_communities=None,
            resolution=1.0,
            randomness=0.001,
            use_modularity=True,
            iterations=1,
        )
    )


class NativeCommunityBuilder:
    def __init__(
        self,
        clusterer: Callable[[list[tuple[str, str, float]], CommunityConfig], list[Any]] | None = None,
    ):
        self.clusterer = clusterer or _default_clusterer

    def build(self, projection: CommunityProjection, config: CommunityConfig) -> CommunityBuildResult:
        entity_ids = {entity.id for entity in projection.entities}
        edges = sorted(
            (
                relationship.source_id,
                relationship.target_id,
                float(relationship.weight),
            )
            for relationship in projection.relationships
            if relationship.source_id in entity_ids
            and relationship.target_id in entity_ids
            and relationship.source_id != relationship.target_id
            and relationship.weight > 0
        )
        config_fingerprint = stable_hash(asdict(config))
        build_id = stable_hash(
            {
                "corpus_id": projection.corpus_id,
                "active_revision_hash": projection.active_revision_hash,
                "projection": projection.fingerprint,
                "configuration": config_fingerprint,
            }
        )
        grouped: dict[tuple[int, int], set[str]] = defaultdict(set)
        parents: dict[tuple[int, int], int | None] = {}
        for entry in self.clusterer(edges, config) if edges else []:
            key = (int(entry.level), int(entry.cluster))
            grouped[key].add(str(entry.node))
            parents[key] = int(entry.parent_cluster) if entry.parent_cluster is not None else None

        clustered = {member for members in grouped.values() for member in members}
        next_singleton = -1
        for entity_id in sorted(entity_ids - clustered):
            grouped[(0, next_singleton)].add(entity_id)
            parents[(0, next_singleton)] = None
            next_singleton -= 1

        ids = {
            key: stable_hash(
                {"build_id": build_id, "level": key[0], "members": sorted(members)}
            )
            for key, members in grouped.items()
        }
        communities: list[CommunityRecord] = []
        for key, members in sorted(grouped.items()):
            parent_cluster = parents.get(key)
            parent_key = None
            if parent_cluster is not None:
                candidates = [
                    candidate
                    for candidate in grouped
                    if candidate[1] == parent_cluster and candidate[0] < key[0]
                ]
                if candidates:
                    parent_key = max(candidates)
            communities.append(
                CommunityRecord(
                    id=ids[key],
                    build_id=build_id,
                    level=key[0],
                    source_cluster=key[1],
                    parent_id=ids.get(parent_key),
                    member_ids=tuple(sorted(members)),
                    rank=float(len(members)),
                )
            )
        self._validate_hierarchy(communities, entity_ids)
        return CommunityBuildResult(
            build_id,
            projection.corpus_id,
            tuple(sorted(projection.active_revision_ids)),
            projection.active_revision_hash,
            projection.fingerprint,
            config_fingerprint,
            tuple(communities),
        )

    @staticmethod
    def _validate_hierarchy(communities: list[CommunityRecord], entity_ids: set[str]) -> None:
        ids = {community.id for community in communities}
        seen_by_level: dict[int, set[str]] = defaultdict(set)
        for community in communities:
            if not community.member_ids:
                raise ValueError("Community hierarchy contains an empty community.")
            if community.parent_id is not None and community.parent_id not in ids:
                raise ValueError("Community hierarchy contains a missing parent.")
            overlap = seen_by_level[community.level].intersection(community.member_ids)
            if overlap:
                raise ValueError(f"Entities belong to multiple communities at level {community.level}: {sorted(overlap)}")
            seen_by_level[community.level].update(community.member_ids)
        if communities and not entity_ids.issubset(set().union(*seen_by_level.values())):
            raise ValueError("Community hierarchy omitted projected entities.")
        parent_of = {community.id: community.parent_id for community in communities}
        for community_id in parent_of:
            visited: set[str] = set()
            current: str | None = community_id
            while current is not None:
                if current in visited:
                    raise ValueError("Community hierarchy contains a cycle.")
                visited.add(current)
                current = parent_of.get(current)

    async def build_reports(
        self,
        build: CommunityBuildResult,
        projection: CommunityProjection,
        reporter: CommunityReporter,
        embed_documents: Callable[[list[str]], list[list[float]]],
    ) -> CommunityBuildResult:
        entities = {entity.id: entity for entity in projection.entities}
        parents = {parent.id: parent for parent in projection.parents}
        relationships = tuple(projection.relationships)
        reports: dict[str, CommunityReport] = {}
        for community in sorted(build.communities, key=lambda item: (-item.level, item.id)):
            member_ids = set(community.member_ids)
            evidence_ids = sorted(
                {
                    parent_id
                    for entity_id in member_ids
                    for parent_id in entities[entity_id].parent_ids
                    if parent_id in parents
                }
                | {
                    parent_id
                    for relationship in relationships
                    if relationship.source_id in member_ids and relationship.target_id in member_ids
                    for parent_id in relationship.parent_ids
                    if parent_id in parents
                }
            )
            child_reports = [
                reports[item.id]
                for item in build.communities
                if item.parent_id == community.id and item.id in reports
            ]
            context = {
                "community_id": community.id,
                "entities": [asdict(entities[item]) for item in community.member_ids],
                "relationships": [
                    asdict(item)
                    for item in relationships
                    if item.source_id in member_ids and item.target_id in member_ids
                ],
                "parents": [asdict(parents[item]) for item in evidence_ids],
                "child_findings": [asdict(finding) for report in child_reports for finding in report.findings],
            }
            payload = await reporter.report(community, context)
            report_id = stable_hash({"build_id": build.id, "community_id": community.id})
            findings: list[CommunityFinding] = []
            allowed = set(evidence_ids)
            for position, raw in enumerate(payload.get("findings") or []):
                source_ids = tuple(dict.fromkeys(str(value) for value in raw.get("source_parent_ids") or []))
                if not source_ids or not set(source_ids).issubset(allowed):
                    raise ValueError(f"Community finding {position} contains invalid source parent IDs.")
                findings.append(
                    CommunityFinding(
                        id=stable_hash({"report_id": report_id, "position": position}),
                        report_id=report_id,
                        position=position,
                        summary=str(raw.get("summary") or "").strip(),
                        explanation=str(raw.get("explanation") or "").strip(),
                        parent_ids=source_ids,
                        confidence=float(raw.get("confidence", 1.0)),
                    )
                )
            if not findings:
                raise ValueError(f"Community report {report_id} did not contain grounded findings.")
            full_content = "\n\n".join(
                (str(payload.get("summary") or "").strip(), *(finding.explanation for finding in findings))
            ).strip()
            reports[community.id] = CommunityReport(
                id=report_id,
                build_id=build.id,
                community_id=community.id,
                title=str(payload.get("title") or "").strip(),
                summary=str(payload.get("summary") or "").strip(),
                full_content=full_content,
                rank=float(payload.get("rank") or community.rank),
                rating_explanation=str(payload.get("rating_explanation") or "").strip(),
                findings=tuple(findings),
                raw=dict(payload),
            )
        ordered = [reports[community.id] for community in build.communities]
        vectors = embed_documents([report.full_content for report in ordered])
        if len(vectors) != len(ordered):
            raise ValueError("Every community report must receive exactly one embedding.")
        embedded = tuple(
            CommunityReport(**{**asdict(report), "findings": report.findings, "embedding": tuple(vector)})
            for report, vector in zip(ordered, vectors, strict=True)
        )
        return build.with_reports(embedded)
