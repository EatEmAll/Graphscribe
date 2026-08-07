from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from itertools import combinations
from types import SimpleNamespace
from typing import Any, Sequence


DEFAULT_STOP_WORDS = {
    "about", "after", "again", "also", "among", "because", "before", "being",
    "between", "could", "first", "from", "have", "into", "more", "other", "should",
    "such", "than", "that", "their", "there", "these", "they", "this", "through",
    "using", "were", "which", "while", "with", "would",
}


@dataclass(frozen=True)
class ProvisionalConfig:
    minimum_token_length: int = 5
    max_terms_per_parent: int = 40
    max_document_frequency_ratio: float = 0.6
    stop_words: frozenset[str] = field(default_factory=lambda: frozenset(DEFAULT_STOP_WORDS))

    def __post_init__(self) -> None:
        if self.minimum_token_length < 2 or self.max_terms_per_parent <= 1:
            raise ValueError("Invalid provisional extraction limits.")
        if not 0 < self.max_document_frequency_ratio <= 1:
            raise ValueError("max_document_frequency_ratio must be in (0, 1].")


class ProvisionalGraphExtractor:
    def __init__(self, config: ProvisionalConfig | None = None):
        self.config = config or ProvisionalConfig()

    def candidates(self, text: str) -> tuple[str, ...]:
        capitalized = re.findall(r"\b(?:[A-Z][\w-]+(?:\s+[A-Z][\w-]+){0,3})\b", text)
        tokens = re.findall(r"\b[\w-]+\b", text.casefold(), flags=re.UNICODE)
        repeated = [
            token
            for token, count in Counter(tokens).most_common()
            if count >= 2
            and len(token) >= self.config.minimum_token_length
            and token not in self.config.stop_words
            and not token.isdigit()
        ]
        normalized = {
            " ".join(value.split()).strip("-_ ")
            for value in [*capitalized, *repeated]
            if value.strip("-_ ")
        }
        return tuple(sorted(normalized, key=lambda value: (value.casefold(), value))[: self.config.max_terms_per_parent])

    def extract_batch(self, parents: Sequence[dict[str, Any]]) -> dict[str, Any]:
        terms = {str(parent["parent_id"]): self.candidates(str(parent.get("text") or "")) for parent in parents}
        document_frequency = Counter(term.casefold() for values in terms.values() for term in set(values))
        maximum = max(1, int(len(parents) * self.config.max_document_frequency_ratio))
        result = {}
        for parent in parents:
            parent_id = str(parent["parent_id"])
            retained = [term for term in terms[parent_id] if document_frequency[term.casefold()] <= maximum]
            nodes = [SimpleNamespace(id=term.casefold(), type="ProvisionalEntity", properties={"title": term}) for term in retained]
            nodes_by_id = {node.id: node for node in nodes}
            co_occurrence: Counter[tuple[str, str]] = Counter()
            for sentence in re.split(r"(?<=[.!?])\s+|\n+", str(parent.get("text") or "")):
                normalized_sentence = sentence.casefold()
                present = sorted(node_id for node_id in nodes_by_id if node_id in normalized_sentence)
                co_occurrence.update(combinations(present, 2))
            relationships = [
                    SimpleNamespace(
                        source=nodes_by_id[left],
                        target=nodes_by_id[right],
                        type="CO_OCCURS",
                        properties={"provisional": True, "weight": float(weight)},
                    )
                    for (left, right), weight in sorted(co_occurrence.items())
                ]
            result[parent_id] = SimpleNamespace(nodes=nodes, relationships=relationships)
        return result


@dataclass
class ProvisionalExtractionWorker:
    store: Any
    extractor: ProvisionalGraphExtractor

    def run_batch(self, limit: int = 100) -> dict[str, int]:
        parents = self.store.pending_graph_parents(limit)
        documents = self.extractor.extract_batch(parents)
        completed = 0
        for parent in parents:
            graph = documents[str(parent["parent_id"])]
            self.store.persist_parent_graph(
                str(parent["parent_id"]),
                parent.get("child_ids") or [],
                graph,
                revision_id=parent.get("revision_id"),
                extraction_state="PROVISIONAL",
            )
            completed += 1
        return {"requested": len(parents), "completed": completed, "failed": 0}
