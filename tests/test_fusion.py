from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pytest

from dynamic_hybrid_retrieval.bm25 import BM25Index
from dynamic_hybrid_retrieval.dense import HNSWIndex
from dynamic_hybrid_retrieval.dynamic_dense import DynamicHNSWIndex
from dynamic_hybrid_retrieval.embeddings import HashingEmbedding
from dynamic_hybrid_retrieval.event_log import EventLog
from dynamic_hybrid_retrieval.fusion import (
    HybridConsistencyError,
    HybridSearchConfig,
    HybridSearcher,
    RankedList,
    reciprocal_rank_fusion,
)
from dynamic_hybrid_retrieval.index import DynamicIndex
from dynamic_hybrid_retrieval.models import Document, IndexEvent, Operation

NOW = datetime(2026, 11, 12, tzinfo=UTC)


def document(document_id: str, text: str, version: int = 1) -> Document:
    return Document(
        document_id=document_id,
        title=text,
        body=text,
        updated_at=NOW,
        version=version,
    )


@dataclass(frozen=True)
class Result:
    document_id: str
    score: float
    version: int
    document: Document


def result(document_id: str, score: float, version: int = 1) -> Result:
    return Result(document_id, score, version, document(document_id, document_id, version))


def test_rrf_uses_ranks_not_raw_score_scales() -> None:
    lexical = [result("doc-a", 1000.0), result("doc-b", 900.0)]
    dense = [result("doc-b", 0.9), result("doc-a", 0.8)]

    fused = reciprocal_rank_fusion(
        [RankedList("lexical", lexical), RankedList("dense", dense)],
        k=60,
    )

    assert [item.document_id for item in fused] == ["doc-a", "doc-b"]
    assert fused[0].score == pytest.approx(1 / 61 + 1 / 62)
    assert fused[0].contribution_for("lexical") is not None
    assert fused[0].contribution_for("lexical").raw_score == 1000.0  # type: ignore[union-attr]
    assert fused[0].contribution_for("dense").raw_score == 0.8  # type: ignore[union-attr]


def test_document_in_both_lists_beats_single_source_candidate() -> None:
    fused = reciprocal_rank_fusion(
        [
            RankedList("lexical", [result("shared", 5), result("lexical-only", 4)]),
            RankedList("dense", [result("dense-only", 0.9), result("shared", 0.8)]),
        ],
        k=10,
    )

    assert fused[0].document_id == "shared"
    assert len(fused[0].contributions) == 2


def test_weights_and_document_id_tie_break_are_deterministic() -> None:
    fused = reciprocal_rank_fusion(
        [
            RankedList("lexical", [result("doc-b", 3)], weight=2),
            RankedList("dense", [result("doc-a", 0.9)], weight=2),
        ],
        k=60,
    )

    assert [item.document_id for item in fused] == ["doc-a", "doc-b"]
    assert fused[0].score == pytest.approx(2 / 61)


def test_duplicate_within_source_contributes_once() -> None:
    duplicate = result("doc-a", 10)
    fused = reciprocal_rank_fusion(
        [RankedList("lexical", [duplicate, duplicate])],
        k=10,
    )

    assert len(fused) == 1
    assert fused[0].score == pytest.approx(1 / 11)
    assert len(fused[0].contributions) == 1


def test_version_disagreement_fails_closed() -> None:
    with pytest.raises(HybridConsistencyError, match="disagree"):
        reciprocal_rank_fusion(
            [
                RankedList("lexical", [result("doc-a", 2, 1)]),
                RankedList("dense", [result("doc-a", 0.8, 2)]),
            ]
        )


def test_configuration_and_arguments_are_validated() -> None:
    with pytest.raises(ValueError):
        RankedList("", [])
    with pytest.raises(ValueError):
        RankedList("lexical", [], weight=0)
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([], limit=0)
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([], k=0)
    with pytest.raises(ValueError, match="unique"):
        reciprocal_rank_fusion([RankedList("same", []), RankedList("same", [])])
    with pytest.raises(ValueError):
        HybridSearchConfig(candidate_limit=0)


def build_hybrid(tmp_path: Path) -> tuple[HybridSearcher, DynamicIndex, DynamicHNSWIndex]:
    documents = [
        document("doc-a", "search retrieval ranking"),
        document("doc-b", "vector semantic similarity"),
        document("doc-c", "search vector infrastructure"),
    ]
    event_log = EventLog(tmp_path / "events.jsonl", fsync=False)
    lexical = DynamicIndex(BM25Index.build(documents), event_log)
    dense = DynamicHNSWIndex(
        HNSWIndex.build(documents, HashingEmbedding(128)),
        event_log,
    )
    return HybridSearcher(lexical, dense), lexical, dense


def test_hybrid_search_returns_explainable_component_ranks(tmp_path: Path) -> None:
    hybrid, _, _ = build_hybrid(tmp_path)

    results = hybrid.search("search vector", limit=3, exact_dense=True)

    assert results
    assert all(result.contributions for result in results)
    assert any(len(result.contributions) == 2 for result in results)
    for result in results:
        for contribution in result.contributions:
            assert contribution.rank >= 1
            assert contribution.contribution == pytest.approx(
                contribution.weight / (60 + contribution.rank)
            )


def test_shared_wal_update_remains_consistent_across_hybrid_paths(tmp_path: Path) -> None:
    hybrid, lexical, dense = build_hybrid(tmp_path)
    updated = document("doc-a", "ocean whales dolphins", 2)
    event = IndexEvent(
        sequence_number=1,
        operation=Operation.UPSERT,
        document_id="doc-a",
        version=2,
        event_time=NOW,
        document=updated,
    )
    lexical.apply(event)
    dense.apply(event)

    results = hybrid.search("ocean whales", exact_dense=True)

    doc_a = next(result for result in results if result.document_id == "doc-a")
    assert doc_a.version == 2
    assert doc_a.document == updated


def test_sequence_disagreement_fails_closed(tmp_path: Path) -> None:
    hybrid, lexical, _ = build_hybrid(tmp_path)
    inserted = document("doc-d", "new document")
    lexical.apply(
        IndexEvent(
            sequence_number=1,
            operation=Operation.UPSERT,
            document_id="doc-d",
            version=1,
            event_time=NOW,
            document=inserted,
        )
    )

    with pytest.raises(HybridConsistencyError, match="sequence"):
        hybrid.search("new document")


def test_candidate_limit_cannot_be_smaller_than_requested_limit(tmp_path: Path) -> None:
    hybrid, lexical, dense = build_hybrid(tmp_path)
    constrained = HybridSearcher(
        lexical,
        dense,
        config=HybridSearchConfig(candidate_limit=2),
    )

    with pytest.raises(ValueError, match="candidate_limit"):
        constrained.search("search", limit=3)
