from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Sequence


@dataclass(frozen=True)
class EmbeddingConfig:
    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    dimension: int = 384
    batch_size: int = 128
    normalize: bool = True
    max_retries: int = 3


class EmbeddingError(RuntimeError):
    pass


class MiniLMEmbedder:
    def __init__(self, config: EmbeddingConfig | None = None, model: Any | None = None):
        self.config = config or EmbeddingConfig()
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
