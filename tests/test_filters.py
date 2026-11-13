from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from dynamic_hybrid_retrieval.bm25 import BM25Index
from dynamic_hybrid_retrieval.dense import HNSWIndex
from dynamic_hybrid_retrieval.dynamic_dense import DynamicHNSWIndex
from dynamic_hybrid_retrieval.embeddings import HashingEmbedding
from dynamic_hybrid_retrieval.event_log import EventLog
from dynamic_hybrid_retrieval.filters import (
    AllOf,
    AnyOf,
    Equals,
    Exists,
    InValues,
    Not,
    NumericRange,
)
from dynamic_hybrid_retrieval.fusion import HybridSearchConfig, HybridSearcher
from dynamic_hybrid_retrieval.index import DynamicIndex
from dynamic_hybrid_retrieval.models import Document

NOW = datetime(2026, 11, 12, tzinfo=UTC)


def document(number: int, *, region: str, year: int, public: bool = True) -> Document:
    return Document(
        document_id=f"doc-{number:02d}",
        title="search retrieval infrastructure",
        body=f"ranking system document {number}",
        updated_at=NOW,
        version=1,
        metadata={
            "region": region,
            "year": year,
            "public": public,
            "owner": {"team": "search" if number % 2 else "storage"},
        },
    )


def test_typed_filters_and_nested_paths() -> None:
    item = document(1, region="us", year=2025)

    assert Equals("region", "us").matches(item)
    assert not Equals("year", True).matches(item)
    assert InValues("region", ("ca", "us")).matches(item)
    assert NumericRange("year", minimum=2024, maximum=2025).matches(item)
    assert not NumericRange("year", minimum=2025, include_minimum=False).matches(item)
    assert Exists("owner.team").matches(item)
    assert not Exists("owner.missing").matches(item)
    assert AllOf((Equals("region", "us"), Equals("owner.team", "search"))).matches(item)
    assert AnyOf((Equals("region", "eu"), Equals("year", 2025))).matches(item)
    assert Not(Equals("region", "eu")).matches(item)


def test_invalid_filters_are_rejected() -> None:
    with pytest.raises(ValueError):
        Equals("owner..team", "search")
    with pytest.raises(ValueError):
        InValues("region", ())
    with pytest.raises(ValueError):
        NumericRange("year")
    with pytest.raises(ValueError):
        NumericRange("year", minimum=True)
    with pytest.raises(ValueError):
        NumericRange("year", minimum=2025, maximum=2024)
    with pytest.raises(ValueError):
        AllOf(())
    with pytest.raises(ValueError):
        AnyOf(())


@pytest.fixture
def documents() -> list[Document]:
    return [
        document(
            number,
            region="eu" if number >= 17 else "us",
            year=2020 + number % 6,
        )
        for number in range(20)
    ]


def test_bm25_filters_before_top_k(documents: list[Document]) -> None:
    index = BM25Index.build(documents)

    results = index.search(
        "search retrieval",
        limit=3,
        document_filter=Equals("region", "eu"),
    )

    assert [item.document_id for item in results] == ["doc-17", "doc-18", "doc-19"]


def test_hnsw_expands_until_k_eligible_documents_exist(documents: list[Document]) -> None:
    index = HNSWIndex.build(documents, HashingEmbedding(128))

    results = index.search(
        "search retrieval infrastructure",
        limit=3,
        exact=True,
        document_filter=Equals("region", "eu"),
    )

    assert len(results) == 3
    assert {item.document.metadata["region"] for item in results} == {"eu"}


def test_dynamic_dense_filters_baseline(documents: list[Document], tmp_path: Path) -> None:
    baseline = HNSWIndex.build(documents, HashingEmbedding(128))
    index = DynamicHNSWIndex(baseline, EventLog(tmp_path / "events.jsonl", fsync=False))

    results = index.search(
        "search retrieval",
        limit=3,
        exact=True,
        document_filter=Equals("region", "eu"),
    )

    assert len(results) == 3
    assert all(item.document.metadata["region"] == "eu" for item in results)


def test_hybrid_applies_filter_before_rrf(documents: list[Document], tmp_path: Path) -> None:
    event_log = EventLog(tmp_path / "events.jsonl", fsync=False)
    lexical = DynamicIndex(BM25Index.build(documents), event_log)
    dense = DynamicHNSWIndex(
        HNSWIndex.build(documents, HashingEmbedding(128)),
        event_log,
    )
    hybrid = HybridSearcher(
        lexical,
        dense,
        config=HybridSearchConfig(candidate_limit=10),
    )

    results = hybrid.search(
        "search retrieval",
        limit=3,
        exact_dense=True,
        document_filter=AllOf(
            (
                Equals("region", "eu"),
                NumericRange("year", minimum=2020),
            )
        ),
    )

    assert len(results) == 3
    assert all(item.document.metadata["region"] == "eu" for item in results)
