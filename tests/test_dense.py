from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from dynamic_hybrid_retrieval.dense import (
    DENSE_INDEX_FILE,
    DenseIndexCorruptionError,
    EmbeddingCompatibilityError,
    HNSWConfig,
    HNSWIndex,
)
from dynamic_hybrid_retrieval.embeddings import HashingEmbedding
from dynamic_hybrid_retrieval.models import Document

NOW = datetime(2026, 11, 3, tzinfo=UTC)


def document(document_id: str, title: str, body: str) -> Document:
    return Document(
        document_id=document_id,
        title=title,
        body=body,
        updated_at=NOW,
        version=1,
    )


@pytest.fixture
def documents() -> list[Document]:
    return [
        document("doc-a", "Search systems", "index retrieval ranking"),
        document("doc-b", "Cooking", "bread flour kitchen"),
        document("doc-c", "Distributed index", "search retrieval infrastructure"),
    ]


def test_hnsw_returns_nearest_documents_with_cosine_scores(documents: list[Document]) -> None:
    index = HNSWIndex.build(documents, HashingEmbedding(128))

    results = index.search("search retrieval", limit=2, exact=True)

    # Both documents share the two query terms, so the stable document-ID
    # tie-breaker places doc-a first.
    assert [result.document_id for result in results] == ["doc-a", "doc-c"]
    assert results[0].score >= results[1].score
    assert all(-1.0 <= result.score <= 1.0 for result in results)


def test_empty_index_and_validation() -> None:
    index = HNSWIndex.build([], HashingEmbedding(16))
    assert index.search("anything") == []
    with pytest.raises(ValueError):
        index.search("anything", limit=0)
    with pytest.raises(ValueError):
        HNSWConfig(connectivity=1)


def test_duplicate_document_ids_are_rejected(documents: list[Document]) -> None:
    with pytest.raises(ValueError, match="unique"):
        HNSWIndex.build([documents[0], documents[0]], HashingEmbedding(16))


def test_dense_generation_round_trip(tmp_path: Path, documents: list[Document]) -> None:
    model = HashingEmbedding(64)
    original = HNSWIndex.build(documents, model)
    generation = tmp_path / "dense-000001"
    original.save(generation)

    loaded = HNSWIndex.load(generation, model)

    assert loaded.document_count == 3
    assert loaded.search("search retrieval", exact=True) == original.search(
        "search retrieval", exact=True
    )
    assert (generation / DENSE_INDEX_FILE).is_file()
    with pytest.raises(FileExistsError):
        original.save(generation)


def test_load_rejects_incompatible_embedding_model(
    tmp_path: Path, documents: list[Document]
) -> None:
    generation = tmp_path / "dense-000001"
    HNSWIndex.build(documents, HashingEmbedding(32)).save(generation)

    with pytest.raises(EmbeddingCompatibilityError):
        HNSWIndex.load(generation, HashingEmbedding(64))


def test_load_detects_corrupt_hnsw_file(tmp_path: Path, documents: list[Document]) -> None:
    model = HashingEmbedding(32)
    generation = tmp_path / "dense-000001"
    HNSWIndex.build(documents, model).save(generation)
    with (generation / DENSE_INDEX_FILE).open("ab") as stream:
        stream.write(b"corruption")

    with pytest.raises(DenseIndexCorruptionError, match="checksum"):
        HNSWIndex.load(generation, model)
