from __future__ import annotations

from types import SimpleNamespace

from notebooklm_graph_pipe.community.models import (
    CommunityBuildResult,
    CommunityFinding,
    CommunityRecord,
    CommunityReport,
)
from notebooklm_graph_pipe.community.store import Neo4jCommunityStore


class Result:
    def __init__(self, row=None):
        self.row = row

    def consume(self):
        return None

    def single(self):
        return self.row


class Session:
    def __init__(self, calls):
        self.calls = calls

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def run(self, query, **parameters):
        self.calls.append((query, parameters))
        return Result()


class Driver:
    def __init__(self):
        self.calls = []

    def session(self, *, database):
        return Session(self.calls)


def test_stage_build_scopes_members_and_grounding_to_active_corpus() -> None:
    driver = Driver()
    store = Neo4jCommunityStore(driver, "neo4j", "corpus-1")
    finding = CommunityFinding("finding", "report", 0, "Finding", "Evidence", ("parent",))
    report = CommunityReport(
        "report",
        "build",
        "community",
        "Title",
        "Summary",
        "Content",
        1.0,
        "Reason",
        (finding,),
        (0.1,),
    )
    build = CommunityBuildResult(
        "build",
        "corpus-1",
        ("revision-1",),
        "revision-hash",
        "projection-hash",
        "config-hash",
        (CommunityRecord("community", "build", 0, 1, None, ("entity",), 1.0),),
        (report,),
    )

    store.stage_build(build, {})

    assert "active community build" not in " ".join(query.lower() for query, _ in driver.calls)
    cleanup_query = next(query for query, _ in driver.calls if "DETACH DELETE build" in query)
    member_query = next(query for query, _ in driver.calls if "MEMBER_OF" in query)
    grounding_query = next(query for query, _ in driver.calls if "GROUNDED_IN" in query)
    assert "Corpus {id: $corpus_id}" in member_query
    assert "ACTIVE_REVISION" in member_query
    assert "Corpus {id: $corpus_id}" in grounding_query
    assert "ACTIVE_REVISION" in grounding_query
    assert "HAS_COMMUNITY_BUILD" in cleanup_query
    assert all(parameters.get("corpus_id", "corpus-1") == "corpus-1" for _, parameters in driver.calls)


def test_garbage_collection_explicitly_excludes_active_build() -> None:
    driver = Driver()
    store = Neo4jCommunityStore(driver, "neo4j", "corpus-1")

    store.garbage_collect_builds(["old-build"])

    query, parameters = driver.calls[0]
    assert "NOT (corpus)-[:ACTIVE_COMMUNITY_BUILD]->(build)" in query
    assert parameters["build_ids"] == ["old-build"]


def test_activation_rechecks_exact_active_revision_set_inside_cypher() -> None:
    class ActivationSession(Session):
        def run(self, query, **parameters):
            self.calls.append((query, parameters))
            return Result({"id": "build"})

    class ActivationDriver(Driver):
        def session(self, *, database):
            return ActivationSession(self.calls)

    class Store(Neo4jCommunityStore):
        def active_projection(self):
            return SimpleNamespace(active_revision_hash="revision-hash")

    driver = ActivationDriver()
    Store(driver, "neo4j", "corpus-1").activate_build("build", "revision-hash")

    query, _ = driver.calls[0]
    assert "collect(DISTINCT active_revision.id) AS current_revision_ids" in query
    assert "size(current_revision_ids) = size(build.active_revision_ids)" in query
    assert "revision_id IN build.active_revision_ids" in query


def test_rollback_rejects_build_for_a_different_active_revision_set() -> None:
    driver = Driver()
    store = Neo4jCommunityStore(driver, "neo4j", "corpus-1")

    try:
        store.rollback_build("stale-build")
    except RuntimeError:
        pass
    else:
        raise AssertionError("Rollback should fail when the Cypher validation returns no build.")

    query, _ = driver.calls[0]
    assert "collect(DISTINCT active_revision.id) AS current_revision_ids" in query
    assert "size(current_revision_ids) = size(build.active_revision_ids)" in query
