from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from notebooklm_graph_pipe.community.builder import NativeCommunityBuilder
from notebooklm_graph_pipe.community.models import (
    CommunityConfig,
    CommunityProjection,
    EntityRecord,
    EvidenceParent,
    RelationshipRecord,
)


@dataclass(frozen=True)
class Cluster:
    node: str
    cluster: int
    parent_cluster: int | None
    level: int


def projection() -> CommunityProjection:
    return CommunityProjection(
        corpus_id="corpus-1",
        active_revision_ids=("revision-1",),
        parents=(
            EvidenceParent("parent-a", "doc", "revision-1", "Alpha evidence"),
            EvidenceParent("parent-b", "doc", "revision-1", "Beta evidence"),
        ),
        entities=(
            EntityRecord("alpha", "Alpha", "", ("parent-a",)),
            EntityRecord("beta", "Beta", "", ("parent-b",)),
            EntityRecord("isolated", "Isolated", "", ("parent-a",)),
        ),
        relationships=(
            RelationshipRecord("r", "alpha", "beta", "related", 2.0, ("parent-a", "parent-b")),
        ),
    )


def test_build_is_deterministic_and_preserves_isolated_entities() -> None:
    entries = [Cluster("alpha", 1, None, 0), Cluster("beta", 1, None, 0)]
    builder = NativeCommunityBuilder(lambda edges, config: entries)

    first = builder.build(projection(), CommunityConfig())
    second = builder.build(projection(), CommunityConfig())

    assert first == second
    assert {member for community in first.communities for member in community.member_ids} == {
        "alpha",
        "beta",
        "isolated",
    }
    assert any(community.member_ids == ("isolated",) for community in first.communities)


def test_reports_are_embedded_and_grounded_in_projected_parents() -> None:
    builder = NativeCommunityBuilder(
        lambda edges, config: [Cluster("alpha", 1, None, 0), Cluster("beta", 1, None, 0)]
    )
    build = builder.build(projection(), CommunityConfig())

    class Reporter:
        async def report(self, community, context):
            source_id = context["parents"][0]["id"]
            return {
                "title": "Report",
                "summary": "Summary",
                "rank": 1,
                "rating_explanation": "Grounded",
                "findings": [
                    {
                        "summary": "Finding",
                        "explanation": "Explanation",
                        "source_parent_ids": [source_id],
                    }
                ],
            }

    result = asyncio.run(
        builder.build_reports(build, projection(), Reporter(), lambda texts: [[0.1, 0.2] for _ in texts])
    )

    assert len(result.reports) == len(result.communities)
    assert all(report.embedding == (0.1, 0.2) for report in result.reports)
    assert all(report.findings[0].parent_ids for report in result.reports)


def test_report_rejects_parent_outside_projection() -> None:
    builder = NativeCommunityBuilder(
        lambda edges, config: [Cluster("alpha", 1, None, 0), Cluster("beta", 1, None, 0)]
    )
    build = builder.build(projection(), CommunityConfig())

    class Reporter:
        async def report(self, community, context):
            return {
                "title": "Report",
                "summary": "Summary",
                "rank": 1,
                "rating_explanation": "Invalid",
                "findings": [
                    {
                        "summary": "Finding",
                        "explanation": "Explanation",
                        "source_parent_ids": ["foreign-parent"],
                    }
                ],
            }

    with pytest.raises(ValueError, match="invalid source parent IDs"):
        asyncio.run(builder.build_reports(build, projection(), Reporter(), lambda texts: [[0.1] for _ in texts]))
