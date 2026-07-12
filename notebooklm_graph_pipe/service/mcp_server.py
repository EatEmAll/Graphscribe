from __future__ import annotations

from typing import Any

from .core import CorpusService


def create_mcp_server(service: CorpusService):
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise RuntimeError("Install the 'mcp[cli]' dependency to run the corpus MCP server.") from exc

    server = FastMCP("neo4j-corpus-research")

    @server.tool()
    def corpus_list() -> list[dict[str, Any]]:
        """List locally registered research corpora."""
        return service.list_corpora()

    @server.tool()
    def corpus_get(corpus_key: str) -> dict[str, Any]:
        """Get corpus metadata and configured sources."""
        return service.get_corpus(corpus_key)

    @server.tool()
    def corpus_search(
        corpus_key: str,
        query: str,
        mode: str = "graph_hybrid",
        top_k: int = 12,
        graph_hops: int = 1,
    ) -> dict[str, Any]:
        """Search a corpus with vector, lexical, or graph-hybrid retrieval."""
        return service.search(
            corpus_key,
            {"query": query, "mode": mode, "top_k": top_k, "graph_hops": graph_hops},
        )

    @server.tool()
    def corpus_answer(
        corpus_key: str,
        question: str,
        mode: str = "graph_hybrid",
        graph_hops: int = 1,
    ) -> dict[str, Any]:
        """Answer a corpus-grounded question with validated source citations."""
        return service.answer(
            corpus_key,
            {"question": question, "mode": mode, "graph_hops": graph_hops},
        )

    @server.tool()
    def source_list(corpus_key: str) -> list[dict[str, Any]]:
        """List the active source documents in a corpus."""
        return service.list_documents(corpus_key)

    @server.tool()
    def source_get(corpus_key: str, document_id: str) -> dict[str, Any]:
        """Get one active source document's metadata."""
        return service.get_document(corpus_key, document_id)

    @server.tool()
    def sync_status(job_id: str) -> dict[str, Any]:
        """Read the status of a corpus synchronization job."""
        return service.get_job(job_id)

    @server.tool()
    def graph_neighbors(
        corpus_key: str,
        entity_id: str,
        hops: int = 1,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Explore one or two graph hops around an entity in a corpus."""
        return service.graph_neighbors(corpus_key, entity_id, hops, limit)

    return server
