from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import UTC, datetime

import pytest

from dynamic_hybrid_retrieval.fusion import HybridConsistencyError
from dynamic_hybrid_retrieval.models import Document
from dynamic_hybrid_retrieval.query_coordinator import (
    PartialResultPolicy,
    QueryCoordinator,
    ScatterGatherError,
    ShardMergeStrategy,
    ShardSearchResponse,
)

NOW = datetime(2026, 11, 16, tzinfo=UTC)


@dataclass(frozen=True)
class Result:
    document_id: str
    score: float
    version: int
    document: Document


def result(document_id: str, score: float, version: int = 1) -> Result:
    document = Document(
        document_id=document_id,
        title=document_id,
        body=document_id,
        updated_at=NOW,
        version=version,
    )
    return Result(document_id, score, version, document)


class FakeShard:
    def __init__(
        self,
        shard_id: str,
        results: tuple[Result, ...],
        *,
        delay: float = 0,
        error: Exception | None = None,
        epoch: int = 4,
    ) -> None:
        self.shard_id = shard_id
        self.results = results
        self.delay = delay
        self.error = error
        self.epoch = epoch
        self.requested_limit = 0

    def search(
        self, query: str, *, limit: int, deadline_monotonic: float, topology_epoch: int
    ) -> ShardSearchResponse:
        self.requested_limit = limit
        if self.delay:
            time.sleep(self.delay)
        if self.error:
            raise self.error
        return ShardSearchResponse(self.shard_id, self.epoch, self.results[:limit])


def test_fanout_merges_global_top_k_deterministically() -> None:
    a = FakeShard("a", (result("doc-2", 8), result("doc-3", 5)))
    b = FakeShard("b", (result("doc-1", 9), result("doc-4", 4)))
    response = QueryCoordinator({"b": b, "a": a}, topology_epoch=4).search("query", limit=3)
    assert [item.document_id for item in response.results] == ["doc-1", "doc-2", "doc-3"]
    assert response.successful_shards == ("a", "b")
    assert response.per_shard_limit == 6
    assert a.requested_limit == b.requested_limit == 6
    assert not response.partial


def test_timeout_fails_closed_by_default() -> None:
    clients = {
        "fast": FakeShard("fast", (result("doc-1", 1),)),
        "slow": FakeShard("slow", (), delay=0.05),
    }
    coordinator = QueryCoordinator(clients, topology_epoch=4, timeout_seconds=0.005)
    with pytest.raises(ScatterGatherError) as captured:
        coordinator.search("query")
    assert captured.value.failures[0].shard_id == "slow"
    assert captured.value.failures[0].timed_out


def test_partial_policy_returns_results_and_failure_metadata() -> None:
    clients = {
        "good": FakeShard("good", (result("doc-1", 1),)),
        "bad": FakeShard("bad", (), error=RuntimeError("unavailable")),
    }
    response = QueryCoordinator(
        clients, topology_epoch=4, partial_policy=PartialResultPolicy.ALLOW_PARTIAL
    ).search("query")
    assert response.partial
    assert response.successful_shards == ("good",)
    assert response.failures[0].reason == "unavailable"
    assert response.results[0].document_id == "doc-1"


def test_stale_epoch_is_reported_as_shard_failure() -> None:
    coordinator = QueryCoordinator({"a": FakeShard("a", (), epoch=3)}, topology_epoch=4)
    with pytest.raises(ScatterGatherError, match="stale topology"):
        coordinator.search("query")


def test_duplicate_transition_copy_is_deduplicated_but_disagreement_fails() -> None:
    same = result("doc-1", 1)
    coordinator = QueryCoordinator(
        {"a": FakeShard("a", (same,)), "b": FakeShard("b", (result("doc-1", 2),))}, topology_epoch=4
    )
    assert len(coordinator.search("query").results) == 1
    disagreement = QueryCoordinator(
        {"a": FakeShard("a", (same,)), "b": FakeShard("b", (result("doc-1", 2, 2),))},
        topology_epoch=4,
    )
    with pytest.raises(HybridConsistencyError, match="disagree"):
        disagreement.search("query")


def test_rrf_merge_is_available_for_noncomparable_shard_scores() -> None:
    coordinator = QueryCoordinator(
        {
            "a": FakeShard("a", (result("shared", 1000), result("a-only", 900))),
            "b": FakeShard("b", (result("b-only", 0.9), result("shared", 0.8))),
        },
        topology_epoch=4,
        merge_strategy=ShardMergeStrategy.RRF,
    )
    response = coordinator.search("query", limit=3)
    assert response.results[0].document_id == "shared"
    assert len(response.results[0].contributions) == 2  # type: ignore[attr-defined]
