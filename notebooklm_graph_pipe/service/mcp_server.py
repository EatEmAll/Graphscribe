from __future__ import annotations

from typing import Any

from .core import CorpusService

# Measured on the hosted corpus, a default corpus_search returns about 40K characters of tool-result
# text (roughly 15K tokens), inside a 25K-token tool-result limit.
DEFAULT_MAX_PASSAGE_CHARS = 2000
DEFAULT_MAX_TOTAL_CHARS = 24000
MAX_TEXT_BOUND = 100_000


def compact_search_response(
    response: dict[str, Any],
    *,
    max_passage_chars: int,
    max_total_chars: int,
) -> dict[str, Any]:
    """Return a search response with each passage's text once, bounded per passage and in total.

    ``results[].text`` is the only passage text; ``contexts`` keep citation metadata without
    repeating it. Results past the total budget are dropped and counted in ``omitted_results``.
    """
    results: list[dict[str, Any]] = []
    remaining = max_total_chars
    source_results = list(response.get("results") or [])
    for result in source_results:
        if remaining <= 0:
            break
        text = str(result.get("text") or "")
        kept = text[: min(max_passage_chars, remaining)]
        remaining -= len(kept)
        results.append({**result, "text": kept, "text_chars": len(text), "text_truncated": len(kept) < len(text)})
    return {
        "results": results,
        "omitted_results": len(source_results) - len(results),
        "contexts": [
            {field: value for field, value in context.items() if field != "text"}
            for context in response.get("contexts") or []
        ],
        "diagnostics": response.get("diagnostics") or {},
    }


def create_mcp_server(service: CorpusService):
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise RuntimeError("Install the 'mcp[cli]' dependency to run the corpus MCP server.") from exc

    server = FastMCP("neo4j-corpus-research")

    @server.tool()
    def corpus_list() -> list[dict[str, Any]]:
        """List locally registered research corpora. Payload: one short row per corpus."""
        return service.list_corpora()

    @server.tool()
    def corpus_get(corpus_key: str) -> dict[str, Any]:
        """Get corpus metadata and configured sources.

        Payload: the whole manifest, including every source entry, so it grows with the corpus
        (hundreds of KB for a few hundred sources). Use source_list to look up sources instead.
        """
        return service.get_corpus(corpus_key)

    @server.tool()
    def corpus_search(
        corpus_key: str,
        query: str,
        mode: str = "graph_hybrid",
        top_k: int = 12,
        graph_hops: int = 1,
        max_passage_chars: int = DEFAULT_MAX_PASSAGE_CHARS,
        max_total_chars: int = DEFAULT_MAX_TOTAL_CHARS,
    ) -> dict[str, Any]:
        """Search a corpus with vector, lexical, or graph-hybrid retrieval and return ranked passages.

        Any natural-language query is accepted; query syntax characters are matched literally.
        Payload: each passage's text appears once, in results[].text, cut to max_passage_chars
        characters (default 2000) and max_total_chars characters across all results (default
        24000); text_truncated and text_chars report each cut, omitted_results counts results
        dropped by the total bound. A default call returns about 40K characters (roughly 15K
        tokens) whatever top_k is; contexts carry citation metadata without text. Recommended pattern: ask corpus_answer for a cited answer
        first, then use corpus_search with a specific query to read supporting passages; raise
        max_passage_chars with a smaller top_k to read fewer passages in full.
        """
        for name, value in (("max_passage_chars", max_passage_chars), ("max_total_chars", max_total_chars)):
            if not 1 <= value <= MAX_TEXT_BOUND:
                raise ValueError(f"{name} must be between 1 and {MAX_TEXT_BOUND}.")
        response = service.search(
            corpus_key,
            {"query": query, "mode": mode, "top_k": top_k, "graph_hops": graph_hops},
        )
        return compact_search_response(
            response, max_passage_chars=max_passage_chars, max_total_chars=max_total_chars
        )

    @server.tool()
    def corpus_answer(
        corpus_key: str,
        question: str,
        mode: str = "graph_hybrid",
        graph_hops: int = 1,
        conversation_id: str | None = None,
    ) -> dict[str, Any]:
        """Answer a corpus-grounded question with validated source citations.

        Any natural-language question is accepted. Payload: the answer plus citations with
        280-character quote previews, typically a few thousand tokens. Recommended pattern: the
        first call for a research question; follow up with corpus_search for full passages and
        graph_neighbors for related entities.
        """
        return service.answer(
            corpus_key,
            {
                "question": question,
                "mode": mode,
                "graph_hops": graph_hops,
                "conversation_id": conversation_id,
            },
        )

    @server.tool()
    def source_list(
        corpus_key: str,
        query: str | None = None,
        offset: int = 0,
        limit: int = 50,
    ) -> dict[str, Any]:
        """List the active source documents in a corpus with their titles, one page at a time.

        query keeps sources whose title, source URI, source key, or document id contains it,
        ignoring case. Payload: total matching sources and up to limit (default 50, at most 100)
        rows of source_key, document_id, title, source_uri, and status, about 300 characters each
        (a default page is about 16K characters).
        Recommended pattern: pass a distinctive title fragment as query to check whether a source
        is present; page with offset only to enumerate the corpus.
        """
        return service.list_sources(corpus_key, query=query, offset=offset, limit=limit)

    @server.tool()
    def source_get(corpus_key: str, document_id: str) -> dict[str, Any]:
        """Get one active source document's manifest metadata.

        Payload: one small record. Find the document_id with source_list.
        """
        return service.get_document(corpus_key, document_id)

    @server.tool()
    def sync_status(job_id: str) -> dict[str, Any]:
        """Read the status of a corpus synchronization job. Payload: one small record."""
        return service.get_job(job_id)

    @server.tool()
    def graph_neighbors(
        corpus_key: str,
        entity_id: str,
        hops: int = 1,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Explore one or two graph hops around an entity in a corpus.

        Payload: up to limit (default 50, at most 200) neighbor rows with entity descriptions.
        Recommended pattern: start with one hop and a small limit; a relationship is a lead to
        revalidate with corpus_answer, not evidence by itself.
        """
        return service.graph_neighbors(corpus_key, entity_id, hops, limit)

    return server
