"""Embedding contracts and local implementations for dense retrieval."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from dynamic_hybrid_retrieval.tokenizer import Tokenizer

FloatMatrix = npt.NDArray[np.float32]


@runtime_checkable
class EmbeddingModel(Protocol):
    """Stable boundary between retrieval code and an embedding provider."""

    @property
    def model_id(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    def embed_documents(self, texts: Sequence[str]) -> FloatMatrix: ...

    def embed_query(self, text: str) -> FloatMatrix: ...


def _validate_matrix(vectors: Any, *, rows: int, dimension: int) -> FloatMatrix:
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.shape != (rows, dimension):
        raise ValueError(
            f"embedding shape {matrix.shape} does not match expected {(rows, dimension)}"
        )
    if not np.isfinite(matrix).all():
        raise ValueError("embeddings contain NaN or infinity")
    return np.ascontiguousarray(matrix)


class HashingEmbedding:
    """Deterministic dependency-free test embedder.

    This model is useful for tests, examples, and persistence validation. It is
    deliberately not intended as the semantic model used in quality reports.
    """

    def __init__(self, dimension: int = 256, *, tokenizer: Tokenizer | None = None) -> None:
        if dimension < 8:
            raise ValueError("dimension must be at least eight")
        self._dimension = dimension
        self.tokenizer = tokenizer or Tokenizer()

    @property
    def model_id(self) -> str:
        return f"hashing-blake2b-v1-{self.dimension}"

    @property
    def dimension(self) -> int:
        return self._dimension

    def _embed(self, text: str) -> npt.NDArray[np.float32]:
        vector = np.zeros(self.dimension, dtype=np.float32)
        for token in self.tokenizer.tokenize(text):
            digest = hashlib.blake2b(token.encode(), digest_size=16).digest()
            bucket = int.from_bytes(digest[:8], "little") % self.dimension
            sign = 1.0 if digest[8] & 1 else -1.0
            vector[bucket] += sign
        norm = float(np.linalg.norm(vector))
        if norm:
            vector /= norm
        return vector

    def embed_documents(self, texts: Sequence[str]) -> FloatMatrix:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        return np.stack([self._embed(text) for text in texts]).astype(np.float32, copy=False)

    def embed_query(self, text: str) -> FloatMatrix:
        return self._embed(text).reshape(1, self.dimension)


class SentenceTransformerEmbedding:
    """Normalized Sentence Transformers adapter with lazy package import."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        *,
        device: str | None = None,
        batch_size: int = 64,
    ) -> None:
        if not model_name.strip():
            raise ValueError("model_name cannot be empty")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        from sentence_transformers import SentenceTransformer

        self._model_name = model_name
        self._batch_size = batch_size
        self._model = SentenceTransformer(model_name, device=device)
        dimension = self._model.get_embedding_dimension()
        if dimension is None or dimension < 1:
            raise ValueError("embedding model did not report a valid dimension")
        self._dimension = dimension

    @property
    def model_id(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    def _encode(self, texts: Sequence[str]) -> FloatMatrix:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        vectors = self._model.encode(
            list(texts),
            batch_size=self._batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return _validate_matrix(vectors, rows=len(texts), dimension=self.dimension)

    def embed_documents(self, texts: Sequence[str]) -> FloatMatrix:
        return self._encode(texts)

    def embed_query(self, text: str) -> FloatMatrix:
        return self._encode([text])
