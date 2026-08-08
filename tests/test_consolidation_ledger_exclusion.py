import pytest

from notebooklm_graph_pipe.consolidation import self_improving
from notebooklm_graph_pipe.consolidation import taxonomy_cleanup
from notebooklm_graph_pipe.consolidation import tier1_lemmatize
from notebooklm_graph_pipe.consolidation import tier2_relabel
from notebooklm_graph_pipe.consolidation import tier3_semantic


LEDGER_RELATION_PATTERN = "HAS_SOURCE|MATERIALIZED_AS|LEGACY_EVIDENCE"


class _EmptySession:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def run(self, query: str, **_kwargs):
        self.queries.append(query)
        return []


class _SingleResult:
    def __init__(self, row):
        self.row = row

    def single(self):
        return self.row


class _MissingEndpointSession:
    def run(self, query: str, **_kwargs):
        if "reverse_count" in query:
            return _SingleResult({"reverse_count": 0})
        if "would_cycle" in query:
            return _SingleResult(None)
        raise AssertionError(f"Unexpected query: {query}")


@pytest.mark.parametrize(
    "fetch",
    [
        lambda session: tier1_lemmatize.fetch_entities(session),
        lambda session: tier2_relabel.fetch_concept_only_nodes(session, None),
        lambda session: tier3_semantic.fetch_entities(session),
        lambda session: taxonomy_cleanup.fetch_taxonomy_candidates(
            session, seed_eids=[], max_nodes=10
        ),
        lambda session: taxonomy_cleanup.fetch_candidate_pool(session),
    ],
)
def test_consolidation_candidate_queries_exclude_ledger_graph(fetch) -> None:
    session = _EmptySession()

    assert fetch(session) == []

    query = session.queries[-1]
    assert "NOT n:CorpusSource" in query
    assert LEDGER_RELATION_PATTERN in query


@pytest.mark.parametrize(
    "mutate",
    [
        lambda session: tier1_lemmatize.merge_group(session, ["a", "b"], "name", False),
        lambda session: tier2_relabel.apply_label(session, "a", "Method"),
        lambda session: tier3_semantic.merge_pair(session, "a", "b", "name"),
        lambda session: taxonomy_cleanup.apply_label(
            session, source_eid="a", old_labels=["Concept"], new_label="Method"
        ),
        lambda session: taxonomy_cleanup.add_relation(
            session, source_eid="a", relation="TYPE_OF", target_eid="b"
        ),
        lambda session: self_improving._apply_taxonomy_label(
            session, source_eid="a", old_labels=["Concept"], new_label="Method"
        ),
        lambda session: self_improving._add_taxonomy_relation(
            session, source_eid="a", relation="TYPE_OF", target_eid="b"
        ),
    ],
)
def test_consolidation_mutations_exclude_ledger_graph(mutate) -> None:
    session = _EmptySession()

    mutate(session)

    query = session.queries[-1]
    assert ":__Entity__" in query
    assert "CorpusSource" in query
    assert LEDGER_RELATION_PATTERN in query


@pytest.mark.parametrize(
    "validate",
    [taxonomy_cleanup.can_apply_relation, self_improving._can_apply_taxonomy_relation],
)
def test_taxonomy_validation_rejects_non_consolidatable_endpoint(validate) -> None:
    allowed, reason = validate(
        _MissingEndpointSession(),
        source_eid="ledger-node",
        relation="TYPE_OF",
        target_eid="entity-node",
    )

    assert allowed is False
    assert reason == "Source or target is not a consolidatable entity"


def test_tier1_merge_requires_every_requested_entity() -> None:
    session = _EmptySession()

    tier1_lemmatize.merge_group(session, ["a", "b"], "name", False)

    assert "size(nodes) = size($eids)" in session.queries[-1]
