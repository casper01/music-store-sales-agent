"""Local persistent vector index backed by ChromaDB.

Embeddings are computed by an injected :class:`Embedder` and passed to Chroma explicitly; Chroma's
own embedding-function integration is not used. This keeps the index provider-agnostic and lets
the tests run fully offline. The embedding model name is stored in the collection metadata and
checked on every search, because mixing vectors from different models silently produces nonsense.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import suppress
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.errors import ChromaError, NotFoundError

from sales_agent.adapters.embeddings import Embedder
from sales_agent.errors import KnowledgeIndexError, ToolInputError

DEFAULT_COLLECTION = "business_knowledge"
MAX_TOP_K = 20


@dataclass(frozen=True, slots=True)
class IndexEntry:
    id: str
    text: str
    document: str
    heading: str
    chunk: int


@dataclass(frozen=True, slots=True)
class SearchHit:
    document: str
    heading: str
    chunk: int
    text: str
    similarity: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ChromaVectorIndex:
    def __init__(
        self,
        path: Path,
        embedder: Embedder,
        collection_name: str = DEFAULT_COLLECTION,
    ) -> None:
        self.embedder = embedder
        self.collection_name = collection_name
        try:
            self._client = chromadb.PersistentClient(
                path=str(path), settings=ChromaSettings(anonymized_telemetry=False)
            )
        except ChromaError as exc:
            raise KnowledgeIndexError(f"Could not open vector store at {path}: {exc}") from exc

    def rebuild(self, entries: Sequence[IndexEntry]) -> int:
        """Replace the collection with ``entries``. Returns the number of indexed chunks.

        Embeddings are computed before the old collection is dropped, so a failing embedding API
        call leaves the previous index intact.
        """
        ids = [entry.id for entry in entries]
        if len(set(ids)) != len(ids):
            raise ValueError("Index entry ids must be unique")
        vectors = self.embedder.embed([entry.text for entry in entries]) if entries else []

        with suppress(NotFoundError):
            self._client.delete_collection(self.collection_name)
        collection = self._client.create_collection(
            name=self.collection_name,
            configuration={"hnsw": {"space": "cosine"}},
            metadata={"embedding_model": self.embedder.model_name},
            embedding_function=None,
        )
        if entries:
            collection.add(
                ids=ids,
                documents=[entry.text for entry in entries],
                embeddings=vectors,
                metadatas=[
                    {"document": entry.document, "heading": entry.heading, "chunk": entry.chunk}
                    for entry in entries
                ],
            )
        return len(entries)

    def search(self, query: str, top_k: int) -> list[SearchHit]:
        if not isinstance(query, str) or not query.strip():
            raise ToolInputError("query must be a non-empty string")
        if not 1 <= top_k <= MAX_TOP_K:
            raise ToolInputError(f"top_k must be between 1 and {MAX_TOP_K}")

        try:
            collection = self._client.get_collection(self.collection_name, embedding_function=None)
        except NotFoundError as exc:
            raise KnowledgeIndexError(
                "Knowledge index is missing; run `sales-agent index` first"
            ) from exc

        indexed_model = (collection.metadata or {}).get("embedding_model")
        if indexed_model != self.embedder.model_name:
            raise KnowledgeIndexError(
                f"Index was built with embedding model {indexed_model!r} but "
                f"{self.embedder.model_name!r} is configured; run `sales-agent index` again"
            )
        count = collection.count()
        if count == 0:
            return []

        query_vector = self.embedder.embed([query.strip()])[0]
        try:
            result = collection.query(
                query_embeddings=[query_vector],
                n_results=min(top_k, count),
                include=["documents", "metadatas", "distances"],
            )
        except ChromaError as exc:
            raise KnowledgeIndexError(f"Vector search failed: {exc}") from exc

        documents = result["documents"][0] if result["documents"] else []
        metadatas = result["metadatas"][0] if result["metadatas"] else []
        distances = result["distances"][0] if result["distances"] else []
        return [
            SearchHit(
                document=str(metadata["document"]),
                heading=str(metadata.get("heading", "")),
                chunk=int(metadata["chunk"]),
                text=document,
                similarity=round(1.0 - float(distance), 4),
            )
            for document, metadata, distance in zip(documents, metadatas, distances, strict=True)
        ]
