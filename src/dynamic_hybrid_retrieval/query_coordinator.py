"""Concurrent scatter-gather query coordination across logical shards."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from dynamic_hybrid_retrieval.fusion import (
    FusedSearchResult,
    HybridConsistencyError,
    RankedList,
    RankedResult,
    reciprocal_rank_fusion,
)


class PartialResultPolicy(StrEnum):
    FAIL_CLOSED = "fail_closed"
    ALLOW_PARTIAL = "allow_partial"


class ShardMergeStrategy(StrEnum):
    SCORE = "score"
    RRF = "rrf"


@dataclass(frozen=True, slots=True)
class ShardSearchResponse:
    shard_id: str
    topology_epoch: int
    results: tuple[RankedResult, ...]


class ShardSearchClient(Protocol):
    def search(
        self,
        query: str,
        *,
        limit: int,
        deadline_monotonic: float,
        topology_epoch: int,
    ) -> ShardSearchResponse: ...


@dataclass(frozen=True, slots=True)
class ShardFailure:
    shard_id: str
    reason: str
    timed_out: bool


@dataclass(frozen=True, slots=True)
class DistributedSearchResponse:
    topology_epoch: int
    results: tuple[RankedResult, ...]
    successful_shards: tuple[str, ...]
    failures: tuple[ShardFailure, ...]
    partial: bool
    took_ms: float
    per_shard_limit: int


class ScatterGatherError(RuntimeError):
    def __init__(self, failures: tuple[ShardFailure, ...]) -> None:
        self.failures = failures
        details = ", ".join(f"{item.shard_id}: {item.reason}" for item in failures)
        super().__init__(f"distributed search failed: {details}")


class QueryCoordinator:
    """Fan out one query and deterministically merge shard-local candidates."""

    def __init__(
        self,
        clients: Mapping[str, ShardSearchClient],
        *,
        topology_epoch: int,
        timeout_seconds: float = 1.0,
        overfetch_factor: float = 2.0,
        partial_policy: PartialResultPolicy = PartialResultPolicy.FAIL_CLOSED,
        merge_strategy: ShardMergeStrategy = ShardMergeStrategy.SCORE,
        rrf_k: int = 60,
    ) -> None:
        if topology_epoch < 1:
            raise ValueError("topology epoch must be positive")
        if not clients:
            raise ValueError("at least one shard client is required")
        if any(not shard_id.strip() for shard_id in clients):
            raise ValueError("shard IDs cannot be empty")
        if timeout_seconds <= 0:
            raise ValueError("timeout must be positive")
        if not math.isfinite(overfetch_factor) or overfetch_factor < 1:
            raise ValueError("overfetch factor must be finite and at least one")
        if rrf_k < 1:
            raise ValueError("rrf_k must be positive")
        self.clients = dict(sorted(clients.items()))
        self.topology_epoch = topology_epoch
        self.timeout_seconds = timeout_seconds
        self.overfetch_factor = overfetch_factor
        self.partial_policy = partial_policy
        self.merge_strategy = merge_strategy
        self.rrf_k = rrf_k

    def search(self, query: str, *, limit: int = 10) -> DistributedSearchResponse:
        if not query.strip():
            raise ValueError("query cannot be empty")
        if limit < 1:
            raise ValueError("limit must be positive")
        started = time.monotonic()
        deadline = started + self.timeout_seconds
        per_shard_limit = max(limit, math.ceil(limit * self.overfetch_factor))
        executor = ThreadPoolExecutor(max_workers=len(self.clients))
        futures: dict[Future[ShardSearchResponse], str] = {
            executor.submit(
                client.search,
                query,
                limit=per_shard_limit,
                deadline_monotonic=deadline,
                topology_epoch=self.topology_epoch,
            ): shard_id
            for shard_id, client in self.clients.items()
        }
        done, pending = wait(futures, timeout=self.timeout_seconds)
        failures = [ShardFailure(futures[future], "deadline exceeded", True) for future in pending]
        responses: list[ShardSearchResponse] = []
        for future in done:
            shard_id = futures[future]
            try:
                response = future.result()
                if response.shard_id != shard_id:
                    raise RuntimeError(f"response identified itself as {response.shard_id!r}")
                if response.topology_epoch != self.topology_epoch:
                    raise RuntimeError(
                        f"stale topology epoch {response.topology_epoch}; "
                        f"expected {self.topology_epoch}"
                    )
                responses.append(response)
            except Exception as exc:
                failures.append(ShardFailure(shard_id, str(exc), False))
        for future in pending:
            future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        ordered_failures = tuple(sorted(failures, key=lambda item: item.shard_id))
        if ordered_failures and self.partial_policy is PartialResultPolicy.FAIL_CLOSED:
            raise ScatterGatherError(ordered_failures)
        ordered_responses = sorted(responses, key=lambda item: item.shard_id)
        results = self._merge(ordered_responses, limit=limit)
        return DistributedSearchResponse(
            topology_epoch=self.topology_epoch,
            results=tuple(results),
            successful_shards=tuple(item.shard_id for item in ordered_responses),
            failures=ordered_failures,
            partial=bool(ordered_failures),
            took_ms=(time.monotonic() - started) * 1_000,
            per_shard_limit=per_shard_limit,
        )

    def _merge(
        self,
        responses: Sequence[ShardSearchResponse],
        *,
        limit: int,
    ) -> list[RankedResult] | list[FusedSearchResult]:
        if self.merge_strategy is ShardMergeStrategy.RRF:
            return reciprocal_rank_fusion(
                [RankedList(item.shard_id, item.results) for item in responses],
                limit=limit,
                k=self.rrf_k,
            )
        by_document: dict[str, RankedResult] = {}
        for response in responses:
            for result in response.results:
                previous = by_document.get(result.document_id)
                if previous is not None and (
                    previous.version != result.version or previous.document != result.document
                ):
                    raise HybridConsistencyError(
                        f"shards disagree on current version of {result.document_id!r}"
                    )
                if previous is None or result.score > previous.score:
                    by_document[result.document_id] = result
        return sorted(
            by_document.values(),
            key=lambda result: (-result.score, result.document_id),
        )[:limit]
