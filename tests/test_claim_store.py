from __future__ import annotations

from notebooklm_graph_pipe.ingestion.neo4j_store import Neo4jCorpusStore


class Result(list):
    def consume(self):
        return None

    def single(self):
        return self[0] if self else None


class Session:
    def __init__(self, calls):
        self.calls = calls

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def run(self, query, **parameters):
        self.calls.append((query, parameters))
        if "RETURN parent.id AS parent_id" in query:
            return Result([{"parent_id": "p1", "text": "text", "revision_id": "r1"}])
        if "RETURN persisted" in query:
            return Result([{"persisted": len(parameters["rows"])}])
        return Result()


class Driver:
    def __init__(self):
        self.calls = []

    def session(self, *, database):
        return Session(self.calls)


def test_claim_batch_selects_only_active_stale_or_failed_parents() -> None:
    driver = Driver()
    store = Neo4jCorpusStore(driver, "neo4j", corpus_id="corpus")

    rows = store.pending_claim_parents(10, "fingerprint")

    query, parameters = driver.calls[0]
    assert rows[0]["parent_id"] == "p1"
    assert "ACTIVE_REVISION" in query
    assert "claim_status = 'FAILED'" in query
    assert "coalesce(parent.claim_extraction_fingerprint, '') <> $extraction_fingerprint" in query
    assert parameters["extraction_fingerprint"] == "fingerprint"


def test_empty_claim_result_still_atomically_checkpoints_parent() -> None:
    driver = Driver()
    store = Neo4jCorpusStore(driver, "neo4j", corpus_id="corpus")

    store.persist_parent_claims("p1", [], extraction_fingerprint="fingerprint")

    query, parameters = driver.calls[0]
    assert "CALL {" in query
    assert "RETURN count(row) AS persisted" in query
    assert "parent.claim_status = 'COMPLETED'" in query
    assert parameters["rows"] == []
    assert parameters["extraction_fingerprint"] == "fingerprint"
