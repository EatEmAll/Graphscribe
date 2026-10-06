from __future__ import annotations

import asyncio
import json

import pytest

from notebooklm_graph_pipe.service.mcp_server import create_mcp_server

# A 25K-token tool-result limit, at a conservative 2.5 characters per token of identifier-heavy JSON.
TOOL_RESULT_CHAR_LIMIT = int(25_000 * 2.5)


def _passage(index: int, size: int) -> str:
    return (f"passage {index} " + "evidence " * size)[:size]


class SearchService:
    def __init__(self, passage_chars: int = 9000, count: int = 12):
        self.passage_chars = passage_chars
        self.count = count
        self.payloads = []

    def search(self, key, payload):
        self.payloads.append((key, payload))
        results = []
        contexts = []
        for index in range(min(self.count, payload["top_k"])):
            text = _passage(index, self.passage_chars)
            results.append({
                "chunk_id": f"p{index}", "document_id": f"d{index}", "parent_id": f"p{index}",
                "text": text, "title": f"Title {index}", "source_uri": f"https://example.org/{index}",
                "page_start": None, "page_end": None, "timestamp_start_ms": None,
                "timestamp_end_ms": None, "section_path": ["Intro"], "channels": ["lexical", "vector"],
                "channel_ranks": {"vector": index + 1}, "rrf_score": 0.03, "reranker_score": 1.5,
                "graph_paths": [],
            })
            contexts.append({
                "citation_id": f"S{index + 1}", "parent_id": f"p{index}", "document_id": f"d{index}",
                "title": f"Title {index}", "source_uri": f"https://example.org/{index}", "text": text,
                "page_start": None, "page_end": None, "timestamp_start_ms": None,
                "timestamp_end_ms": None, "section_path": ["Intro"], "matched_chunk_ids": [f"p{index}"],
            })
        return {"results": results, "contexts": contexts, "diagnostics": {}}

    def list_sources(self, key, *, query=None, offset=0, limit=50):
        self.payloads.append((key, {"query": query, "offset": offset, "limit": limit}))
        return {"total": 0, "offset": offset, "limit": limit, "sources": []}


def _call(server, name, arguments):
    result = asyncio.run(server.call_tool(name, arguments))
    content = result[0] if isinstance(result, tuple) else result
    text = content[0].text
    return json.loads(text), text


def test_corpus_search_returns_each_passage_once_within_the_default_size_bound() -> None:
    service = SearchService()
    payload, text = _call(create_mcp_server(service), "corpus_search", {"corpus_key": "demo", "query": "q"})

    assert service.payloads[0][1]["top_k"] == 12
    assert len(payload["results"]) == 12
    assert all("text" not in context for context in payload["contexts"])
    assert [context["citation_id"] for context in payload["contexts"]][:2] == ["S1", "S2"]
    first = payload["results"][0]
    assert first["text"] == _passage(0, 9000)[: len(first["text"])]
    assert first["text_truncated"] is True and first["text_chars"] == 9000
    assert len(text) < TOOL_RESULT_CHAR_LIMIT


def test_corpus_search_honors_explicit_passage_and_total_bounds() -> None:
    service = SearchService(passage_chars=500)
    payload, _ = _call(
        create_mcp_server(service),
        "corpus_search",
        {"corpus_key": "demo", "query": "q", "max_passage_chars": 200, "max_total_chars": 700},
    )

    texts = [result.get("text", "") for result in payload["results"]]
    assert [len(text) for text in texts[:4]] == [200, 200, 200, 100]
    assert len(payload["results"]) == 4
    assert payload["omitted_results"] == 8
    assert sum(len(text) for text in texts) == 700


def test_corpus_search_leaves_short_passages_untouched() -> None:
    service = SearchService(passage_chars=100, count=3)
    payload, _ = _call(create_mcp_server(service), "corpus_search", {"corpus_key": "demo", "query": "q"})

    assert [result["text"] for result in payload["results"]] == [_passage(i, 100) for i in range(3)]
    assert all(result["text_truncated"] is False for result in payload["results"])
    assert payload["omitted_results"] == 0


def test_source_list_tool_forwards_filter_and_paging() -> None:
    service = SearchService()
    payload, _ = _call(
        create_mcp_server(service),
        "source_list",
        {"corpus_key": "demo", "query": "momentum", "offset": 50, "limit": 10},
    )

    assert payload == {"total": 0, "offset": 50, "limit": 10, "sources": []}
    assert service.payloads == [("demo", {"query": "momentum", "offset": 50, "limit": 10})]


@pytest.mark.parametrize("arguments", [{"max_passage_chars": 0}, {"max_total_chars": 0}])
def test_corpus_search_rejects_non_positive_bounds(arguments) -> None:
    service = SearchService()
    with pytest.raises(Exception, match="must be between"):
        _call(create_mcp_server(service), "corpus_search", {"corpus_key": "demo", "query": "q", **arguments})
    assert service.payloads == []


def test_tool_descriptions_state_payload_size_and_call_pattern() -> None:
    tools = {tool.name: tool.description for tool in asyncio.run(create_mcp_server(SearchService()).list_tools())}

    assert "characters" in tools["corpus_search"] and "corpus_answer" in tools["corpus_search"]
    assert "source_list" in tools["source_get"] and "query" in tools["source_list"]
    for name in ("corpus_search", "corpus_answer", "source_list", "graph_neighbors"):
        assert "Payload" in tools[name], name
