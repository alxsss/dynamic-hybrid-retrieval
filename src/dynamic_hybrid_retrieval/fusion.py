"""Explainable reciprocal-rank fusion and hybrid search coordination."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from dynamic_hybrid_retrieval.dynamic_dense import DynamicHNSWIndex
from dynamic_hybrid_retrieval.index import DynamicIndex
from dynamic_hybrid_retrieval.models import Document


class FusionError(RuntimeError):
    """Base class for rank-fusion failures."""


class HybridConsistencyError(FusionError):
    """Raised when lexical and dense state do not describe one logical corpus."""


class RankedResult(Protocol):
    @property
    def document_id(self) -> str: ...

    @property
    def score(self) -> float: ...

    @property
    def version(self) -> int: ...

    @property
    def document(self) -> Document: ...


@dataclass(frozen=True, slots=True)
class RankedList:
    """One retriever's ordered candidates and its fusion weight."""

    name: str
    results: Sequence[RankedResult]
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("ranked-list name cannot be empty")
        if self.weight <= 0:
            raise ValueError("ranked-list weight must be positive")


@dataclass(frozen=True, slots=True)
class RRFContribution:
    source: str
    rank: int
    raw_score: float
    weight: float
    contribution: float


@dataclass(frozen=True, slots=True)
class FusedSearchResult:
    document_id: str
    score: float
    version: int
    document: Document
    contributions: tuple[RRFContribution, ...]

    def contribution_for(self, source: str) -> RRFContribution | None:
        return next(
            (contribution for contribution in self.contributions if contribution.source == source),
            None,
        )


@dataclass(frozen=True, slots=True)
class HybridSearchConfig:
    rrf_k: int = 60
    candidate_limit: int = 100
    lexical_weight: float = 1.0
    dense_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.rrf_k < 1:
            raise ValueError("rrf_k must be positive")
        if self.candidate_limit < 1:
            raise ValueError("candidate_limit must be positive")
        if self.lexical_weight <= 0 or self.dense_weight <= 0:
            raise ValueError("retriever weights must be positive")


def reciprocal_rank_fusion(
    ranked_lists: Sequence[RankedList],
    *,
    limit: int = 10,
    k: int = 60,
) -> list[FusedSearchResult]:
    """Fuse ordered lists using weighted RRF with deterministic ties.

    A document contributes at most once per source. Raw component scores are
    retained only for explanation; they never participate in fusion arithmetic.
    """

    if limit < 1:
        raise ValueError("limit must be positive")
    if k < 1:
        raise ValueError("k must be positive")
    source_names = [ranked_list.name for ranked_list in ranked_lists]
    if len(set(source_names)) != len(source_names):
        raise ValueError("ranked-list names must be unique")

    documents: dict[str, tuple[int, Document]] = {}
    contributions: dict[str, list[RRFContribution]] = {}
    totals: dict[str, float] = {}

    for ranked_list in ranked_lists:
        seen_in_source: set[str] = set()
        for rank, result in enumerate(ranked_list.results, start=1):
            if result.document_id in seen_in_source:
                continue
            seen_in_source.add(result.document_id)
            existing = documents.get(result.document_id)
            if existing is not None and (
                existing[0] != result.version or existing[1] != result.document
            ):
                raise HybridConsistencyError(
                    f"retrievers disagree on current version of {result.document_id!r}"
                )
            documents[result.document_id] = (result.version, result.document)
            contribution_value = ranked_list.weight / (k + rank)
            contribution = RRFContribution(
                source=ranked_list.name,
                rank=rank,
                raw_score=result.score,
                weight=ranked_list.weight,
                contribution=contribution_value,
            )
            contributions.setdefault(result.document_id, []).append(contribution)
            totals[result.document_id] = totals.get(result.document_id, 0.0) + contribution_value

    ordered_ids = sorted(totals, key=lambda document_id: (-totals[document_id], document_id))
    return [
        FusedSearchResult(
            document_id=document_id,
            score=totals[document_id],
            version=documents[document_id][0],
            document=documents[document_id][1],
            contributions=tuple(contributions[document_id]),
        )
        for document_id in ordered_ids[:limit]
    ]


class HybridSearcher:
    """Coordinate lexical and dense retrieval over one versioned corpus."""

    def __init__(
        self,
        lexical: DynamicIndex,
        dense: DynamicHNSWIndex,
        *,
        config: HybridSearchConfig | None = None,
    ) -> None:
        self.lexical = lexical
        self.dense = dense
        self.config = config or HybridSearchConfig()

    def _sequence_pair(self) -> tuple[int, int]:
        return (
            self.lexical.statistics.last_sequence_number,
            self.dense.statistics.last_sequence_number,
        )

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        exact_dense: bool = False,
    ) -> list[FusedSearchResult]:
        if limit < 1:
            raise ValueError("limit must be positive")
        if self.config.candidate_limit < limit:
            raise ValueError("candidate_limit cannot be smaller than result limit")

        before = self._sequence_pair()
        if before[0] != before[1]:
            raise HybridConsistencyError(
                f"lexical sequence {before[0]} differs from dense sequence {before[1]}"
            )
        lexical_results = self.lexical.search(query, limit=self.config.candidate_limit)
        dense_results = self.dense.search(
            query,
            limit=self.config.candidate_limit,
            exact=exact_dense,
        )
        after = self._sequence_pair()
        if after != before:
            raise HybridConsistencyError("index state changed during hybrid retrieval")

        return reciprocal_rank_fusion(
            [
                RankedList("lexical", lexical_results, self.config.lexical_weight),
                RankedList("dense", dense_results, self.config.dense_weight),
            ],
            limit=limit,
            k=self.config.rrf_k,
        )
