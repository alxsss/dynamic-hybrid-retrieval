"""Retrieval quality, latency, and approximate-index recall measurements."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from dynamic_hybrid_retrieval.dense import HNSWIndex
from dynamic_hybrid_retrieval.evaluation import EvaluationSet, RelevanceJudgment
from dynamic_hybrid_retrieval.metrics import LatencyRecorder, LatencySummary


class EvaluationRunError(RuntimeError):
    """Raised when a retrieval run is incomplete, invalid, or non-deterministic."""


class IdentifiedResult(Protocol):
    @property
    def document_id(self) -> str: ...


SearchFunction = Callable[[str, int], Sequence[IdentifiedResult]]


@dataclass(frozen=True, slots=True)
class QueryQualityMetrics:
    query_id: str
    recall_at_k: float
    reciprocal_rank_at_k: float
    ndcg_at_k: float
    retrieved_count: int
    relevant_count: int


@dataclass(frozen=True, slots=True)
class AggregateQualityMetrics:
    query_count: int
    mean_recall_at_k: float
    mean_reciprocal_rank_at_k: float
    mean_ndcg_at_k: float


@dataclass(frozen=True, slots=True)
class RetrieverEvaluation:
    name: str
    k: int
    rankings: dict[str, tuple[str, ...]]
    queries: tuple[QueryQualityMetrics, ...]
    aggregate: AggregateQualityMetrics
    latency: LatencySummary
    warmup_rounds: int
    measured_rounds: int


@dataclass(frozen=True, slots=True)
class HNSWQueryRecall:
    query_id: str
    recall_at_k: float
    approximate_ids: tuple[str, ...]
    exact_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class HNSWRecallReport:
    k: int
    mean_recall_at_k: float
    queries: tuple[HNSWQueryRecall, ...]


def _validate_k(k: int) -> None:
    if k < 1:
        raise ValueError("k must be positive")


def _discounted_cumulative_gain(relevances: Sequence[int]) -> float:
    return float(
        sum(
            (2**relevance - 1) / math.log2(rank + 1)
            for rank, relevance in enumerate(relevances, start=1)
        )
    )


def evaluate_ranked_list(
    query_id: str,
    ranked_document_ids: Sequence[str],
    judgments: Sequence[RelevanceJudgment],
    *,
    k: int,
    minimum_relevance: int = 1,
) -> QueryQualityMetrics:
    """Calculate binary Recall/MRR and graded nDCG for one query."""

    _validate_k(k)
    if not 1 <= minimum_relevance <= 3:
        raise ValueError("minimum_relevance must be between one and three")
    if len(ranked_document_ids) != len(set(ranked_document_ids)):
        raise EvaluationRunError(f"ranking for {query_id!r} contains duplicate documents")
    relevance_by_document = {
        judgment.document_id: judgment.relevance
        for judgment in judgments
        if judgment.query_id == query_id
    }
    relevant = {
        document_id
        for document_id, relevance in relevance_by_document.items()
        if relevance >= minimum_relevance
    }
    top_k = tuple(ranked_document_ids[:k])
    retrieved_relevant = sum(document_id in relevant for document_id in top_k)
    recall = retrieved_relevant / len(relevant) if relevant else 0.0
    reciprocal_rank = next(
        (1.0 / rank for rank, document_id in enumerate(top_k, start=1) if document_id in relevant),
        0.0,
    )
    retrieved_grades = [relevance_by_document.get(document_id, 0) for document_id in top_k]
    ideal_grades = sorted(relevance_by_document.values(), reverse=True)[:k]
    ideal_dcg = _discounted_cumulative_gain(ideal_grades)
    ndcg = _discounted_cumulative_gain(retrieved_grades) / ideal_dcg if ideal_dcg else 0.0
    return QueryQualityMetrics(
        query_id=query_id,
        recall_at_k=recall,
        reciprocal_rank_at_k=reciprocal_rank,
        ndcg_at_k=ndcg,
        retrieved_count=len(top_k),
        relevant_count=len(relevant),
    )


def evaluate_rankings(
    evaluation_set: EvaluationSet,
    rankings: Mapping[str, Sequence[str]],
    *,
    k: int,
    minimum_relevance: int = 1,
) -> tuple[tuple[QueryQualityMetrics, ...], AggregateQualityMetrics]:
    """Validate and macro-average a complete set of query rankings."""

    _validate_k(k)
    expected_queries = {query.query_id for query in evaluation_set.queries}
    if set(rankings) != expected_queries:
        missing = sorted(expected_queries - rankings.keys())
        extra = sorted(rankings.keys() - expected_queries)
        raise EvaluationRunError(f"ranking query mismatch; missing={missing}, extra={extra}")
    document_ids = {document.document_id for document in evaluation_set.documents}
    metrics: list[QueryQualityMetrics] = []
    for query in evaluation_set.queries:
        ranking = tuple(rankings[query.query_id])
        unknown = sorted(set(ranking) - document_ids)
        if unknown:
            raise EvaluationRunError(
                f"ranking for {query.query_id!r} contains unknown documents: {unknown}"
            )
        metrics.append(
            evaluate_ranked_list(
                query.query_id,
                ranking,
                evaluation_set.judgments,
                k=k,
                minimum_relevance=minimum_relevance,
            )
        )
    count = len(metrics)
    aggregate = AggregateQualityMetrics(
        query_count=count,
        mean_recall_at_k=sum(item.recall_at_k for item in metrics) / count,
        mean_reciprocal_rank_at_k=(sum(item.reciprocal_rank_at_k for item in metrics) / count),
        mean_ndcg_at_k=sum(item.ndcg_at_k for item in metrics) / count,
    )
    return tuple(metrics), aggregate


def evaluate_retriever(
    name: str,
    evaluation_set: EvaluationSet,
    search: SearchFunction,
    *,
    k: int = 10,
    warmup_rounds: int = 1,
    measured_rounds: int = 5,
) -> RetrieverEvaluation:
    """Execute a retriever repeatedly, validate stability, and measure latency."""

    _validate_k(k)
    if not name.strip():
        raise ValueError("retriever name cannot be empty")
    if warmup_rounds < 0 or measured_rounds < 1:
        raise ValueError("warmup must be nonnegative and measured rounds must be positive")
    for _ in range(warmup_rounds):
        for query in evaluation_set.queries:
            search(query.text, k)

    recorder = LatencyRecorder()
    rankings: dict[str, tuple[str, ...]] = {}
    for _ in range(measured_rounds):
        for query in evaluation_set.queries:
            started = time.perf_counter()
            results = search(query.text, k)
            recorder.record(time.perf_counter() - started)
            document_ids = tuple(result.document_id for result in results)
            previous = rankings.setdefault(query.query_id, document_ids)
            if previous != document_ids:
                raise EvaluationRunError(
                    f"retriever {name!r} produced an unstable ranking for {query.query_id!r}"
                )

    query_metrics, aggregate = evaluate_rankings(evaluation_set, rankings, k=k)
    return RetrieverEvaluation(
        name=name,
        k=k,
        rankings=rankings,
        queries=query_metrics,
        aggregate=aggregate,
        latency=recorder.summarize(),
        warmup_rounds=warmup_rounds,
        measured_rounds=measured_rounds,
    )


def measure_hnsw_recall(
    index: HNSWIndex,
    queries: Mapping[str, str],
    *,
    k: int = 10,
) -> HNSWRecallReport:
    """Compare approximate HNSW results with the exact cosine top-K oracle."""

    _validate_k(k)
    if not queries:
        raise ValueError("at least one query is required")
    results: list[HNSWQueryRecall] = []
    for query_id, query_text in queries.items():
        exact_ids = tuple(
            result.document_id for result in index.search(query_text, limit=k, exact=True)
        )
        approximate_ids = tuple(
            result.document_id for result in index.search(query_text, limit=k, exact=False)
        )
        denominator = len(exact_ids)
        recall = len(set(approximate_ids) & set(exact_ids)) / denominator if denominator else 1.0
        results.append(HNSWQueryRecall(query_id, recall, approximate_ids, exact_ids))
    return HNSWRecallReport(
        k=k,
        mean_recall_at_k=sum(item.recall_at_k for item in results) / len(results),
        queries=tuple(results),
    )
