from __future__ import annotations

import pytest

from notebooklm_graph_pipe.retrieval.answering import GroundedAnswerer
from notebooklm_graph_pipe.retrieval.hybrid import HybridRetriever, SearchRequest
from notebooklm_graph_pipe.retrieval.models import Candidate
from notebooklm_graph_pipe.retrieval.neo4j_backend import Neo4jRetrievalBackend
from notebooklm_graph_pipe.runtime.llm_routing import PromptRoleConfig


class Tokenizer:
    def encode(self, text, *, add_special_tokens=False):
        return list(range(len(text.split())))

    def decode(self, tokens, *, skip_special_tokens=True):
        return " ".join(f"word{token}" for token in tokens)


def candidate(identifier, document, parent, text=None):
    return Candidate(
        chunk_id=identifier,
        document_id=document,
        parent_id=parent,
        text=text or identifier,
        title=f"Title {document}",
        source_uri=f"{document}.md",
    )


class Backend:
    def vector_search(self, embedding, limit, filters=None):
        return [candidate("shared", "d1", "p1"), candidate("vector", "d1", "p1")]

    def lexical_search(self, query, limit, filters=None):
        return [candidate("shared", "d1", "p1"), candidate("lexical", "d2", "p2")]

    def graph_expand(self, seed_ids, hops, limit, filters=None):
        return [candidate("graph", "d3", "p3")]

    def parent_contexts(self, parent_ids):
        return {
            parent_id: {
                "parent_id": parent_id,
                "document_id": f"d{index}",
                "title": f"Parent {index}",
                "source_uri": f"source-{index}.md",
                "text": "context text " * 5,
            }
            for index, parent_id in enumerate(parent_ids, 1)
        }


class Embedder:
    def embed_query(self, text):
        return [1.0, 0.0]


class Reranker:
    def rerank(self, query, candidates):
        for index, item in enumerate(candidates):
            item.reranker_score = float(len(candidates) - index)
        return sorted(candidates, key=lambda item: -item.reranker_score)


def retriever():
    return HybridRetriever(Backend(), Embedder(), Reranker(), context_tokenizer=Tokenizer())


def test_graph_hybrid_fuses_channels_and_builds_citations() -> None:
    result = retriever().search(SearchRequest("question", include_diagnostics=True))
    shared = next(item for item in result.candidates if item.chunk_id == "shared")
    assert shared.channels == {"vector", "lexical"}
    assert any(item.chunk_id == "graph" for item in result.candidates)
    assert [item.citation_id for item in result.contexts] == ["S1", "S2", "S3"]
    assert result.diagnostics["graph_candidates"] == 1


def test_grounded_answerer_removes_invalid_citations() -> None:
    def generator(client, **kwargs):
        return {"answer": "Supported [S1], invented [S99].", "citation_ids": ["S1", "S99"]}, ""

    answerer = GroundedAnswerer(
        retriever(),
        PromptRoleConfig("genai", "test"),
        object(),
        generator,
    )
    answer = answerer.answer("question")
    assert [citation["id"] for citation in answer["citations"]] == ["S1"]
    assert "[S99]" not in answer["answer"]
    assert "invalid model citations" in answer["warnings"][0]


def test_no_context_returns_insufficient_evidence() -> None:
    backend = Backend()
    backend.vector_search = lambda embedding, limit, filters=None: []
    backend.lexical_search = lambda query, limit, filters=None: []
    backend.graph_expand = lambda seed_ids, hops, limit, filters=None: []
    empty = HybridRetriever(backend, Embedder(), Reranker(), context_tokenizer=Tokenizer())
    answerer = GroundedAnswerer(empty, PromptRoleConfig("genai", "test"), object())
    answer = answerer.answer("question")
    assert "Insufficient evidence" in answer["answer"]
    assert answer["citations"] == []


def test_missing_first_parent_does_not_discard_later_context() -> None:
    backend = Backend()
    original = backend.parent_contexts
    backend.parent_contexts = lambda parent_ids: {key: value for key, value in original(parent_ids).items() if key != parent_ids[0]}
    result = HybridRetriever(backend, Embedder(), Reranker(), context_tokenizer=Tokenizer()).search(
        SearchRequest("question", mode="hybrid")
    )
    assert result.contexts
    assert all(item.parent_id != "p1" for item in result.contexts)


def test_lexical_backend_does_not_collide_with_driver_query_argument() -> None:
    calls = []

    class Result(list):
        def single(self):
            return None

    class Session:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def run(self, query, **parameters):
            calls.append((query, parameters))
            return Result()

    class Driver:
        def session(self, **kwargs):
            return Session()

    assert Neo4jRetrievalBackend(Driver(), "neo4j", "corpus-id").lexical_search("alpha", 5) == []
    assert calls[0][1]["search_text"] == "alpha"
    assert calls[0][1]["corpus_id"] == "corpus-id"
    assert "query" not in calls[0][1]


def test_vector_backend_widens_global_query_until_corpus_results_are_found() -> None:
    query_limits = []

    class Result(list):
        def __init__(self, values=(), single_value=None):
            super().__init__(values)
            self.single_value = single_value

        def single(self):
            return self.single_value

    class Session:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def run(self, query, **parameters):
            if "RETURN count(n) AS count" in query:
                return Result(single_value={"count": 1000})
            query_limits.append(parameters["query_limit"])
            if parameters["query_limit"] < 1000:
                return Result()
            return Result(
                [
                    {
                        "chunk_id": "target",
                        "document_id": "doc",
                        "parent_id": "parent",
                        "text": "target result",
                        "title": "Target",
                        "source_uri": "target.md",
                    }
                ]
            )

    class Driver:
        def session(self, **kwargs):
            return Session()

    results = Neo4jRetrievalBackend(Driver(), "neo4j", "corpus-id").vector_search([1.0, 0.0], 1)

    assert [result.chunk_id for result in results] == ["target"]
    assert query_limits == [3, 12, 48, 192, 768, 1000]


def test_community_backend_widens_global_indexes_until_active_corpus_results_are_found() -> None:
    overfetch_values = []

    class Result(list):
        def __init__(self, values=(), single_value=None):
            super().__init__(values)
            self.single_value = single_value

        def single(self):
            return self.single_value

    class Session:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def run(self, query_text, **parameters):
            if "RETURN count(report) AS count" in query_text:
                return Result(single_value={"count": 500})
            overfetch_values.append(parameters["overfetch"])
            if parameters["overfetch"] < 500:
                return Result()
            return Result([{"report_id": "target", "findings": []}])

    class Driver:
        def session(self, **kwargs):
            return Session()

    reports = Neo4jRetrievalBackend(Driver(), "neo4j", "corpus-id").search_community_reports(
        "question", [1.0, 0.0], limit=1
    )

    assert reports[0]["report_id"] == "target"
    assert overfetch_values == [4, 16, 64, 256, 500]


def test_graph_backend_uses_neo4j_5_compatible_variable_paths() -> None:
    calls = []

    class Session:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def run(self, query, **parameters):
            calls.append((query, parameters))
            return []

    class Driver:
        def session(self, **kwargs):
            return Session()

    backend = Neo4jRetrievalBackend(Driver(), "neo4j", "corpus-id")
    assert backend.graph_expand(["seed"], 2, 5) == []
    assert backend.graph_neighbors("entity", 2) == []
    assert all(":!HAS_ENTITY" not in query for query, _ in calls)
    assert all("all(rel IN relationships(path)" in query for query, _ in calls)
    assert all(parameters["corpus_id"] == "corpus-id" for _, parameters in calls)


def test_parent_backend_uses_declared_indexes_and_parent_paths() -> None:
    calls = []

    class Result(list):
        def single(self):
            return {"count": 1}

    class Session:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def run(self, query, **parameters):
            calls.append((query, parameters))
            if "queryNodes" in query:
                return Result(
                    [
                        {
                            "chunk_id": "parent-1",
                            "parent_id": "parent-1",
                            "document_id": "doc-1",
                            "text": "parent evidence",
                            "title": "Document",
                            "source_uri": "doc.md",
                        }
                    ]
                )
            return Result()

    class Driver:
        def session(self, **kwargs):
            return Session()

    backend = Neo4jRetrievalBackend(
        Driver(),
        "neo4j",
        "corpus-id",
        retrieval_unit="parent",
        vector_index="parent_embedding_v1",
        keyword_index="parent_keyword_v1",
    )
    result = backend.vector_search([1.0, 0.0], 1)
    backend.graph_expand(["parent-1"], 1, 5)

    assert result[0].chunk_id == result[0].parent_id == "parent-1"
    assert "queryNodes('parent_embedding_v1'" in calls[0][0]
    assert "-[:HAS_PARENT]->(unit:ParentChunk)" in calls[0][0]
    graph_query = calls[1][0]
    assert "(seed_parent:ParentChunk {id: seed_id})" in graph_query
    assert "coalesce(rel.source_parent_ids, [])" in graph_query
    assert graph_query.count("<-[:ACTIVE_REVISION]-(:Document)") >= 3


def test_backend_rejects_unsafe_index_names() -> None:
    with pytest.raises(ValueError, match="index names"):
        Neo4jRetrievalBackend(object(), "neo4j", "corpus", vector_index="bad-name")
