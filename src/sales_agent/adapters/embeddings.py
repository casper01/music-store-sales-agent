"""Embedding providers.

The vector index depends on the small :class:`Embedder` protocol rather than on a vendor SDK, so
the RAG pipeline can be tested offline with a deterministic fake embedder.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

import openai

from sales_agent.errors import EmbeddingError


class Embedder(Protocol):
    model_name: str

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class OpenAICompatibleEmbedder:
    """Embeds text through an OpenAI-compatible embeddings endpoint (Gemini by default)."""

    def __init__(
        self,
        api_key: str,
        model_name: str,
        *,
        base_url: str | None = None,
        timeout_seconds: float = 60.0,
        max_retries: int = 6,
        batch_size: int = 64,
        client: Any | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.model_name = model_name
        self.batch_size = batch_size
        self._client = client or openai.OpenAI(
            api_key=api_key, base_url=base_url, timeout=timeout_seconds, max_retries=max_retries
        )

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            try:
                response = self._client.embeddings.create(model=self.model_name, input=batch)
            except openai.OpenAIError as exc:
                raise EmbeddingError(f"Embedding request failed: {exc}") from exc
            ordered = list(response.data)
            # Some OpenAI-compatible APIs (e.g. Gemini) leave `index` empty; keep API order then.
            if all(item.index is not None for item in ordered):
                ordered.sort(key=lambda item: item.index)
            if len(ordered) != len(batch):
                raise EmbeddingError("Embedding API returned an unexpected number of vectors")
            vectors.extend(list(item.embedding) for item in ordered)
        return vectors
