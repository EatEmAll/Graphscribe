from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace
from typing import Any, Callable, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class EmbeddingConfig:
    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    dimension: int = 384
    batch_size: int = 128
    normalize: bool = True
    max_retries: int = 3


class EmbeddingError(RuntimeError):
    pass


def weighted_parent_embedding(children: Iterable[Mapping[str, Any]]) -> list[float]:
    """Token-weighted, text-deduplicated, L2-normalized child embedding mean."""
    totals: list[float] | None = None
    total_weight = 0.0
    seen_text: set[str] = set()
    for child in children:
        normalized = " ".join(
            re.findall(r"\w+", str(child.get("text") or "").casefold(), flags=re.UNICODE)
        )
        if normalized in seen_text:
            continue
        seen_text.add(normalized)
        embedding = [float(value) for value in child.get("embedding") or []]
        if not embedding:
            continue
        if totals is None:
            totals = [0.0] * len(embedding)
        if len(embedding) != len(totals):
            raise EmbeddingError("Child embeddings within one parent have inconsistent dimensions.")
        weight = max(1.0, float(child.get("token_count") or 1.0))
        for index, value in enumerate(embedding):
            totals[index] += value * weight
        total_weight += weight
    if totals is None or not total_weight:
        raise EmbeddingError("Cannot create a parent embedding without embedded child text.")
    averaged = [value / total_weight for value in totals]
    norm = math.sqrt(sum(value * value for value in averaged))
    if not norm:
        raise EmbeddingError("Cannot normalize a zero parent embedding.")
    return [value / norm for value in averaged]


class MiniLMEmbedder:
    def __init__(self, config: EmbeddingConfig | None = None, model: Any | None = None):
        resolved = config or EmbeddingConfig()
        if resolved.model == "all-MiniLM-L6-v2":
            resolved = replace(resolved, model="sentence-transformers/all-MiniLM-L6-v2")
        self.config = resolved
        self._model = model

    @property
    def model(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.config.model)
        return self._model

    @property
    def fingerprint(self) -> str:
        return f"sentence-transformer:{self.config.model}:{self.config.dimension}:normalized={str(self.config.normalize).lower()}"

    def _validate(self, vector: Sequence[float]) -> list[float]:
        result = [float(value) for value in vector]
        if len(result) != self.config.dimension:
            raise EmbeddingError(f"Expected {self.config.dimension} embedding dimensions, received {len(result)}.")
        if not all(math.isfinite(value) for value in result):
            raise EmbeddingError("Embedding contains a non-finite value.")
        return result

    def _encode(self, texts: list[str]) -> list[list[float]]:
        values = self.model.encode(
            texts,
            batch_size=min(self.config.batch_size, len(texts)),
            normalize_embeddings=self.config.normalize,
            show_progress_bar=False,
        )
        return [self._validate(vector) for vector in values]

    def _encode_resilient(self, texts: list[str], attempt: int = 0) -> list[list[float]]:
        try:
            return self._encode(texts)
        except Exception as exc:
            if attempt + 1 < self.config.max_retries:
                return self._encode_resilient(texts, attempt + 1)
            if len(texts) == 1:
                raise EmbeddingError(f"Failed to embed one chunk: {exc}") from exc
            midpoint = len(texts) // 2
            return self._encode_resilient(texts[:midpoint]) + self._encode_resilient(texts[midpoint:])

    def embed_documents(
        self,
        texts: Sequence[str],
        *,
        on_batch: Callable[[int, int], None] | None = None,
    ) -> list[list[float]]:
        all_vectors: list[list[float]] = []
        total = len(texts)
        for start in range(0, total, self.config.batch_size):
            batch = list(texts[start : start + self.config.batch_size])
            all_vectors.extend(self._encode_resilient(batch))
            if on_batch:
                on_batch(min(start + len(batch), total), total)
        return all_vectors

    def embed_query(self, text: str) -> list[float]:
        return self._encode_resilient([text])[0]
