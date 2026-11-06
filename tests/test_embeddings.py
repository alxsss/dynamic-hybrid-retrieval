from __future__ import annotations

import numpy as np
import pytest

from dynamic_hybrid_retrieval.embeddings import HashingEmbedding


def test_hashing_embeddings_are_deterministic_normalized_and_batched() -> None:
    model = HashingEmbedding(64)
    first = model.embed_documents(["search retrieval", "different words"])
    second = model.embed_documents(["search retrieval", "different words"])

    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(np.linalg.norm(first, axis=1), np.ones(2), atol=1e-6)
    assert first.shape == (2, 64)
    assert first.dtype == np.float32


def test_query_and_document_embedding_use_same_space() -> None:
    model = HashingEmbedding(32)

    np.testing.assert_array_equal(
        model.embed_query("same text")[0],
        model.embed_documents(["same text"])[0],
    )


def test_empty_and_invalid_hashing_embedding_inputs() -> None:
    model = HashingEmbedding(16)
    assert model.embed_documents([]).shape == (0, 16)
    assert np.count_nonzero(model.embed_query("---")) == 0
    with pytest.raises(ValueError):
        HashingEmbedding(4)
