from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

from .models import Candidate, ContextItem


MODES = {"vector", "lexical", "hybrid", "graph_hybrid"}


class RetrievalBackend(Protocol):
    def vector_search(self, embedding: Sequence[float], limit: int, filters: dict[str, Any] | None = None) -> list[Candidate]: ...

    def lexical_search(self, query: str, limit: int, filters: dict[str, Any] | None = None) -> list[Candidate]: ...

    def graph_expand(
        self,
        seed_ids: Sequence[str],
        hops: int,
        limit: int,
        filters: dict[str, Any] | None = None,
    ) -> list[Candidate]: ...

    def parent_contexts(self, parent_ids: Sequence[str]) -> dict[str, dict[str, Any]]: ...


class QueryEmbedder(Protocol):
    def embed_query(self, text: str) -> list[float]: ...


class Reranker(Protocol):
    def rerank(self, query: str, candidates: Sequence[Candidate]) -> list[Candidate]: ...


@dataclass(frozen=True)
class SearchRequest:
    query: str
    mode: str = "graph_hybrid"
    top_k: int = 12
    graph_hops: int = 1
    include_diagnostics: bool = False
    filters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"Unsupported retrieval mode: {self.mode}")
        if not self.query.strip():
            raise ValueError("Search query cannot be empty.")
        if not 1 <= self.top_k <= 50:
            raise ValueError("top_k must be between 1 and 50.")
        if self.graph_hops not in {1, 2}:
            raise ValueError("graph_hops must be one or two.")


@dataclass
class SearchResult:
    candidates: list[Candidate]
    contexts: list[ContextItem]
    diagnostics: dict[str, Any] = field(default_factory=dict)


def reciprocal_rank_fusion(channels: dict[str, Sequence[Candidate]], k: int = 60) -> list[Candidate]:
    merged: dict[str, Candidate] = {}
    for channel, candidates in channels.items():
        for rank, candidate in enumerate(candidates, 1):
            current = merged.get(candidate.chunk_id)
            if current is None:
                current = candidate
                merged[candidate.chunk_id] = current
            current.channels.add(channel)
            current.channel_ranks[channel] = rank
            current.rrf_score += 1.0 / (k + rank)
    return sorted(merged.values(), key=lambda item: (-item.rrf_score, item.chunk_id))


class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", model: Any | None = None):
        self.model_name = model_name
        self._model = model

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, query: str, candidates: Sequence[Candidate]) -> list[Candidate]:
        if not candidates:
            return []
        scores = self.model.predict([(query, candidate.text) for candidate in candidates])
        for candidate, score in zip(candidates, scores, strict=True):
            candidate.reranker_score = float(score)
        return sorted(candidates, key=lambda item: (-(item.reranker_score or 0.0), -item.rrf_score, item.chunk_id))


class HybridRetriever:
    def __init__(
        self,
        backend: RetrievalBackend,
        embedder: QueryEmbedder,
        reranker: Reranker,
        *,
        context_tokenizer: Any,
        context_budget: int = 12000,
        max_parents: int = 8,
        vector_retriever: Any | None = None,
    ):
        self.backend = backend
        self.embedder = embedder
        self.reranker = reranker
        self.context_tokenizer = context_tokenizer
        self.context_budget = context_budget
        self.max_parents = max_parents
        self.vector_retriever = vector_retriever or backend

    def search(self, request: SearchRequest) -> SearchResult:
        channels: dict[str, Sequence[Candidate]] = {}
        if request.mode in {"vector", "hybrid", "graph_hybrid"}:
            channels["vector"] = self.vector_retriever.vector_search(
                self.embedder.embed_query(request.query), 50, request.filters
            )
        if request.mode in {"lexical", "hybrid", "graph_hybrid"}:
            channels["lexical"] = self.backend.lexical_search(request.query, 50, request.filters)
        fused = reciprocal_rank_fusion(channels)[:50]
        graph_candidates: list[Candidate] = []
        if request.mode == "graph_hybrid":
            graph_candidates = self.backend.graph_expand(
                [candidate.chunk_id for candidate in fused[:15]],
                request.graph_hops,
                25,
                request.filters,
            )
            for rank, candidate in enumerate(graph_candidates, 1):
                candidate.channels.add("graph")
                candidate.channel_ranks["graph"] = rank
                candidate.rrf_score += 1.0 / (60 + rank)
        union = {candidate.chunk_id: candidate for candidate in fused}
        for candidate in graph_candidates:
            existing = union.get(candidate.chunk_id)
            if existing is None:
                union[candidate.chunk_id] = candidate
            else:
                existing.channels.update(candidate.channels)
                existing.channel_ranks.update(candidate.channel_ranks)
                existing.graph_paths.extend(candidate.graph_paths)
                existing.rrf_score += candidate.rrf_score
        reranked = self.reranker.rerank(request.query, list(union.values())[:75])
        selected = self._select_candidates(reranked, request.top_k)
        contexts = self._build_contexts(selected)
        diagnostics = {
            "mode": request.mode,
            "vector_candidates": len(channels.get("vector", ())),
            "lexical_candidates": len(channels.get("lexical", ())),
            "graph_candidates": len(graph_candidates),
            "candidate_count": len(union),
            "reranked_count": len(reranked),
            "parents_used": len(contexts),
        }
        return SearchResult(selected, contexts, diagnostics if request.include_diagnostics else {})

    @staticmethod
    def _select_candidates(candidates: Sequence[Candidate], top_k: int) -> list[Candidate]:
        selected: list[Candidate] = []
        per_document: dict[str, int] = {}
        distinct_documents = len({candidate.document_id for candidate in candidates})
        document_cap = 4 if distinct_documents >= 3 else top_k
        for candidate in candidates:
            if per_document.get(candidate.document_id, 0) >= document_cap:
                continue
            selected.append(candidate)
            per_document[candidate.document_id] = per_document.get(candidate.document_id, 0) + 1
            if len(selected) >= top_k:
                break
        return selected

    def _build_contexts(self, candidates: Sequence[Candidate]) -> list[ContextItem]:
        ordered_parent_ids = list(dict.fromkeys(candidate.parent_id for candidate in candidates))
        parents = self.backend.parent_contexts(ordered_parent_ids)
        contexts: list[ContextItem] = []
        used_tokens = 0
        for parent_id in ordered_parent_ids:
            if len(contexts) >= self.max_parents:
                break
            if parent_id not in parents:
                continue
            parent = parents[parent_id]
            text = str(parent.get("text") or "")
            token_ids = list(self.context_tokenizer.encode(text, add_special_tokens=False))
            remaining = self.context_budget - used_tokens
            if remaining <= 0:
                break
            if len(token_ids) > remaining:
                matching = next((item.text for item in candidates if item.parent_id == parent_id), "")
                character_offset = text.find(matching) if matching else -1
                prefix_tokens = (
                    len(self.context_tokenizer.encode(text[:character_offset], add_special_tokens=False))
                    if character_offset > 0
                    else 0
                )
                start = max(0, prefix_tokens - remaining // 3)
                token_ids = token_ids[start : start + remaining]
                text = self.context_tokenizer.decode(token_ids, skip_special_tokens=True).strip()
            matches = tuple(candidate.chunk_id for candidate in candidates if candidate.parent_id == parent_id)
            contexts.append(
                ContextItem(
                    citation_id=f"S{len(contexts) + 1}",
                    parent_id=parent_id,
                    document_id=str(parent["document_id"]),
                    title=str(parent.get("title") or ""),
                    source_uri=str(parent.get("source_uri") or ""),
                    text=text,
                    page_start=parent.get("page_start"),
                    page_end=parent.get("page_end"),
                    timestamp_start_ms=parent.get("timestamp_start_ms"),
                    timestamp_end_ms=parent.get("timestamp_end_ms"),
                    section_path=tuple(parent.get("section_path") or ()),
                    matched_chunk_ids=matches,
                )
            )
            used_tokens += len(token_ids)
        return contexts
