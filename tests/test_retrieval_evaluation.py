from __future__ import annotations

import math
from pathlib import Path

import pytest

from dynamic_hybrid_retrieval.bm25 import BM25Index
from dynamic_hybrid_retrieval.dense import HNSWIndex
from dynamic_hybrid_retrieval.embeddings import HashingEmbedding
from dynamic_hybrid_retrieval.evaluation import load_evaluation_set
from dynamic_hybrid_retrieval.retrieval_evaluation import (
    EvaluationRunError,
    evaluate_ranked_list,
    evaluate_rankings,
    evaluate_retriever,
    measure_hnsw_recall,
)

DATASET = Path(__file__).parents[1] / "data" / "evaluation" / "v1"


def test_query_metrics_match_hand_calculation() -> None:
    dataset = load_evaluation_set(DATASET)
    metrics = evaluate_ranked_list(
        "eval-q03",
        ["eval-doc-04", "eval-doc-11", "eval-doc-07"],
        dataset.judgments,
        k=3,
    )

    ideal = 7 + 3 / math.log2(3) + 3 / 2
    actual = 3 + 0 + 7 / 2
    assert metrics.recall_at_k == pytest.approx(2 / 3)
    assert metrics.reciprocal_rank_at_k == 1.0
    assert metrics.ndcg_at_k == pytest.approx(actual / ideal)
    assert metrics.relevant_count == 3


def test_complete_rankings_are_macro_averaged() -> None:
    dataset = load_evaluation_set(DATASET)
    rankings = {
        query.query_id: tuple(
            judgment.document_id
            for judgment in sorted(
                dataset.judgments_for(query.query_id),
                key=lambda item: (-item.relevance, item.document_id),
            )
        )
        for query in dataset.queries
    }

    per_query, aggregate = evaluate_rankings(dataset, rankings, k=10)

    assert len(per_query) == 10
    assert aggregate.mean_recall_at_k == 1.0
    assert aggregate.mean_reciprocal_rank_at_k == 1.0
    assert aggregate.mean_ndcg_at_k == 1.0


def test_invalid_rankings_fail_closed() -> None:
    dataset = load_evaluation_set(DATASET)
    with pytest.raises(EvaluationRunError, match="mismatch"):
        evaluate_rankings(dataset, {}, k=10)
    with pytest.raises(EvaluationRunError, match="duplicate"):
        evaluate_ranked_list(
            "eval-q01",
            ["eval-doc-01", "eval-doc-01"],
            dataset.judgments,
            k=10,
        )
    rankings = {query.query_id: () for query in dataset.queries}
    rankings["eval-q01"] = ("unknown",)
    with pytest.raises(EvaluationRunError, match="unknown"):
        evaluate_rankings(dataset, rankings, k=10)


def test_retriever_runner_records_quality_and_latency() -> None:
    dataset = load_evaluation_set(DATASET)
    index = BM25Index.build(dataset.documents)

    report = evaluate_retriever(
        "bm25",
        dataset,
        lambda query, k: index.search(query, limit=k),
        k=5,
        warmup_rounds=1,
        measured_rounds=2,
    )

    assert report.name == "bm25"
    assert report.latency.count == 20
    assert len(report.rankings) == 10
    assert 0 <= report.aggregate.mean_recall_at_k <= 1
    assert 0 <= report.aggregate.mean_ndcg_at_k <= 1


def test_unstable_retriever_is_rejected() -> None:
    dataset = load_evaluation_set(DATASET)
    index = BM25Index.build(dataset.documents)
    calls = 0

    def unstable(query: str, k: int):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        results = index.search(query, limit=k)
        return list(reversed(results)) if calls > len(dataset.queries) else results

    with pytest.raises(EvaluationRunError, match="unstable"):
        evaluate_retriever(
            "unstable",
            dataset,
            unstable,
            warmup_rounds=0,
            measured_rounds=2,
        )


def test_hnsw_recall_is_measured_against_exact_oracle() -> None:
    dataset = load_evaluation_set(DATASET)
    index = HNSWIndex.build(dataset.documents, HashingEmbedding(128))
    queries = {query.query_id: query.text for query in dataset.queries}

    report = measure_hnsw_recall(index, queries, k=5)

    assert report.k == 5
    assert len(report.queries) == 10
    assert 0 <= report.mean_recall_at_k <= 1
    assert all(0 <= item.recall_at_k <= 1 for item in report.queries)
