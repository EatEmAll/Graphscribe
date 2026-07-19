from __future__ import annotations

import math

import pytest

from notebooklm_graph_pipe.cli.compact_corpus import (
    CompactCorpusError,
    compact_manifest,
    direct_entity_mentions,
    match_parent_text,
    ordered_token_coverage,
    weighted_parent_embedding,
    _projection_fingerprint,
)
from notebooklm_graph_pipe.ingestion.manifest import CorpusManifest
from notebooklm_graph_pipe.runtime.neo4j_connection import ResolvedNeo4jConnection


def test_weighted_parent_embedding_deduplicates_text_and_normalizes() -> None:
    result = weighted_parent_embedding(
        [
            {"text": "alpha", "token_count": 2, "embedding": [1.0, 0.0]},
            {"text": "ALPHA!", "token_count": 50, "embedding": [0.0, 1.0]},
            {"text": "beta", "token_count": 1, "embedding": [0.0, 1.0]},
        ]
    )

    assert math.isclose(math.sqrt(sum(value * value for value in result)), 1.0)
    assert result[0] > result[1]


def test_weighted_parent_embedding_rejects_inconsistent_dimensions() -> None:
    with pytest.raises(CompactCorpusError, match="dimensions"):
        weighted_parent_embedding(
            [
                {"text": "alpha", "embedding": [1.0, 0.0]},
                {"text": "beta", "embedding": [1.0]},
            ]
        )


def test_parent_text_match_prefers_exact_containment() -> None:
    result = match_parent_text(
        "alpha beta gamma",
        [
            {"id": "wrong", "text": "unrelated text"},
            {"id": "right", "text": "prefix alpha beta gamma suffix"},
        ],
    )

    assert result is not None
    assert result.parent_id == "right"
    assert result.method == "exact"


def test_parent_text_match_rejects_ambiguous_tie() -> None:
    assert (
        match_parent_text(
            "alpha beta gamma",
            [
                {"id": "one", "text": "alpha beta gamma"},
                {"id": "two", "text": "alpha beta gamma"},
            ],
        )
        is None
    )


def test_parent_text_match_uses_overlap_similarity_for_boundary_variation() -> None:
    result = match_parent_text(
        "a b c d e f g h i j k l m n o p q r s t",
        [{"id": "parent", "text": "x b c d e f g h i j k l m n o p q r s y"}],
    )

    assert result is not None
    assert result.method == "shingle"


def test_ordered_token_coverage_tolerates_insertions_but_preserves_order() -> None:
    assert ordered_token_coverage("a b c d", "a b inserted c d") == 1.0
    assert ordered_token_coverage("a b c d", "d c b a") < 0.75


def test_direct_entity_mentions_are_source_local_and_token_bounded() -> None:
    mentions = direct_entity_mentions(
        [
            {"id": "p1", "text": "A Hidden Markov Model and AI."},
            {"id": "p2", "text": "The chair is unrelated."},
        ],
        [
            {"chunk_id": "c1", "entity_ids": ["Hidden Markov Model", "AI"]},
            {"chunk_id": "c2", "entity_ids": ["AI"]},
        ],
    )

    assert {(item.parent_id, item.entity_id, item.source_chunk_ids) for item in mentions} == {
        ("p1", "Hidden Markov Model", ("c1",)),
        ("p1", "AI", ("c1", "c2")),
    }


def test_projection_fingerprint_ignores_embedding_but_detects_text_changes() -> None:
    row = {
        "document_id": "d1",
        "revision_id": "r1",
        "parent_id": "p1",
        "parent": {"text": "alpha", "position": 0, "embedding": [1.0, 0.0]},
    }

    assert _projection_fingerprint([row]) == _projection_fingerprint(
        [{**row, "parent": {**row["parent"], "embedding": [0.0, 1.0]}}]
    )
    assert _projection_fingerprint([row]) != _projection_fingerprint(
        [{**row, "parent": {**row["parent"], "text": "beta"}}]
    )


def test_compact_manifest_preserves_requested_password_environment() -> None:
    source = CorpusManifest(corpus_id="corpus", corpus_key="source", title="Source", neo4j={})
    target = ResolvedNeo4jConnection("neo4j+s://example.databases.neo4j.io", "neo4j", "secret", "neo4j")

    result = compact_manifest(source, target, password_env="NEO4J_AURA_COMPACT_PASSWORD")

    assert result.retrieval_unit == "parent"
    assert result.corpus_key == "source-aura-compact"
    assert result.neo4j["password_env"] == "NEO4J_AURA_COMPACT_PASSWORD"
