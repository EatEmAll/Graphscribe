from __future__ import annotations

from notebooklm_graph_pipe.retrieval.community_query import CommunityQueryEngine
from notebooklm_graph_pipe.runtime.llm_routing import (
    DRIFT_PLANNER_ROLE,
    GLOBAL_MAP_ROLE,
    GLOBAL_REDUCE_ROLE,
)
from notebooklm_graph_pipe.runtime.model_executor import ModelExecutor, ModelUsage


class Adapter:
    provider = "test"
    model = "test"

    def execute(self, request):
        if request.role == GLOBAL_MAP_ROLE:
            payload = {
                "points": [
                    {"text": "Grounded theme", "score": 0.9, "source_parent_ids": ["parent-1"]},
                    {"text": "Bad", "score": 1.0, "source_parent_ids": ["foreign"]},
                ]
            }
        elif request.role == DRIFT_PLANNER_ROLE:
            payload = {"subqueries": ["focused question"]}
        else:
            payload = {"answer": "Synthesized answer", "citation_parent_ids": ["parent-1"]}
        return "", payload, ModelUsage()


class Backend:
    def community_reports(self):
        return [
            {
                "report_id": "report",
                "summary": "Theme",
                "findings": [{"summary": "Finding", "parent_id": "parent-1"}],
            }
        ]

    def search_community_reports(self, query, embedding):
        return self.community_reports()

    def parent_contexts(self, parent_ids):
        return {
            "parent-1": {
                "parent_id": "parent-1",
                "document_id": "doc",
                "title": "Document",
                "source_uri": "source",
                "text": "Original evidence",
            }
        }


class Embedder:
    def embed_query(self, query):
        return [0.1]


class LocalAnswerer:
    def answer(self, question, mode, graph_hops):
        return {
            "answer": "Local evidence",
            "citations": [{"parent_id": "parent-1"}],
            "warnings": [],
        }


def engine() -> CommunityQueryEngine:
    adapter = Adapter()
    executor = ModelExecutor(
        {"test": adapter},
        {GLOBAL_MAP_ROLE: "test", GLOBAL_REDUCE_ROLE: "test", DRIFT_PLANNER_ROLE: "test"},
    )
    return CommunityQueryEngine(Backend(), Embedder(), executor, LocalAnswerer())


def test_global_search_filters_ungrounded_map_points() -> None:
    result = engine().global_answer("What are the themes?")

    assert result["answer"] == "Synthesized answer"
    assert result["citations"][0]["parent_id"] == "parent-1"
    assert result["retrieval"]["mode"] == "global"
    assert result["retrieval"]["invalid_points"] == 1


def test_drift_uses_community_primer_and_graph_hybrid_followup() -> None:
    result = engine().drift_answer("Explain the connections")

    assert result["answer"] == "Synthesized answer"
    assert result["retrieval"]["subqueries"] == ["focused question"]
    assert result["citations"][0]["parent_id"] == "parent-1"
