"""Coordinated lifecycle for one logical lexical-and-dense index."""

from __future__ import annotations

import threading
from dataclasses import dataclass

from dynamic_hybrid_retrieval.dynamic_dense import (
    DenseCompactionSnapshot,
    DynamicHNSWIndex,
)
from dynamic_hybrid_retrieval.filters import DocumentFilter
from dynamic_hybrid_retrieval.fusion import (
    FusedSearchResult,
    HybridConsistencyError,
    HybridSearchConfig,
    HybridSearcher,
)
from dynamic_hybrid_retrieval.index import (
    ApplyResult,
    CompactionSnapshot,
    DynamicIndex,
)
from dynamic_hybrid_retrieval.models import IndexEvent


class HybridIndexUnhealthyError(RuntimeError):
    """Raised when a partial mutation makes the running hybrid view unsafe."""


@dataclass(frozen=True, slots=True)
class HybridIndexStatistics:
    """Health and cardinality of the coordinated logical index."""

    healthy: bool
    last_sequence_number: int
    live_document_count: int
    lexical_delta_document_count: int
    dense_delta_document_count: int
    tombstone_count: int


@dataclass(frozen=True, slots=True)
class HybridCompactionSnapshot:
    """Lexical and dense snapshots captured at one shared high-water mark."""

    lexical: CompactionSnapshot
    dense: DenseCompactionSnapshot

    @property
    def high_water_sequence(self) -> int:
        return self.lexical.high_water_sequence


class DynamicHybridIndex:
    """Treat lexical and dense retrieval as one fail-closed serving unit.

    The child indexes already protect their internal state. This outer lock is
    intentionally separate: it prevents a search or compaction snapshot from
    observing the interval between lexical and dense application of one event.
    Callers must send mutations through this coordinator, not directly to the
    child indexes.
    """

    def __init__(
        self,
        lexical: DynamicIndex,
        dense: DynamicHNSWIndex,
        *,
        search_config: HybridSearchConfig | None = None,
    ) -> None:
        if lexical.event_log.path.resolve() != dense.event_log.path.resolve():
            raise HybridConsistencyError("lexical and dense indexes must share one WAL")
        if lexical.baseline_sequence_number != dense.baseline_sequence_number:
            raise HybridConsistencyError(
                "lexical and dense baselines must use the same high-water sequence"
            )

        self.lexical = lexical
        self.dense = dense
        self._searcher = HybridSearcher(lexical, dense, config=search_config)
        self._lock = threading.RLock()
        self._healthy = True
        self._verify_logical_state()

    def _verify_logical_state(self) -> None:
        lexical_snapshot = self.lexical.snapshot_for_compaction()
        dense_snapshot = self.dense.snapshot_for_compaction()
        if lexical_snapshot.high_water_sequence != dense_snapshot.high_water_sequence:
            raise HybridConsistencyError(
                "lexical and dense indexes recovered to different WAL sequences"
            )
        if lexical_snapshot.documents != dense_snapshot.documents:
            raise HybridConsistencyError(
                "lexical and dense indexes contain different live documents"
            )

    def _require_healthy(self) -> None:
        if not self._healthy:
            raise HybridIndexUnhealthyError(
                "hybrid index is unhealthy after a partial mutation; rebuild it from the WAL"
            )

    @property
    def statistics(self) -> HybridIndexStatistics:
        with self._lock:
            lexical = self.lexical.statistics
            dense = self.dense.statistics
            sequence = lexical.last_sequence_number
            healthy = (
                self._healthy
                and sequence == dense.last_sequence_number
                and lexical.live_document_count == dense.live_document_count
                and lexical.tombstone_count == dense.tombstone_count
            )
            return HybridIndexStatistics(
                healthy=healthy,
                last_sequence_number=sequence,
                live_document_count=lexical.live_document_count,
                lexical_delta_document_count=lexical.delta_document_count,
                dense_delta_document_count=dense.delta_document_count,
                tombstone_count=lexical.tombstone_count,
            )

    def apply(self, event: IndexEvent) -> ApplyResult:
        """Durably apply one event to both views as a coordinated operation.

        The lexical path writes the shared WAL first. Dense then consumes the
        same already-durable record. A dense failure leaves enough information
        for deterministic restart, but this process is poisoned so it cannot
        accidentally serve the partial in-memory state.
        """

        with self._lock:
            self._require_healthy()
            lexical_result = self.lexical.apply(event)
            try:
                dense_result = self.dense.apply(event)
            except Exception:
                self._healthy = False
                raise
            if lexical_result != dense_result:
                self._healthy = False
                raise HybridIndexUnhealthyError(
                    "lexical and dense mutation outcomes differ; rebuild from the WAL"
                )
            return lexical_result

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        exact_dense: bool = False,
        document_filter: DocumentFilter | None = None,
    ) -> list[FusedSearchResult]:
        """Search a stable, coordinated lexical-and-dense state."""

        with self._lock:
            self._require_healthy()
            return self._searcher.search(
                query,
                limit=limit,
                exact_dense=exact_dense,
                document_filter=document_filter,
            )

    def snapshot_for_compaction(
        self,
        validation_queries: tuple[str, ...] = (),
        *,
        query_limit: int = 10,
    ) -> HybridCompactionSnapshot:
        """Capture both retrievers at exactly one shared logical boundary."""

        with self._lock:
            self._require_healthy()
            lexical = self.lexical.snapshot_for_compaction(
                validation_queries,
                query_limit=query_limit,
            )
            dense = self.dense.snapshot_for_compaction(
                validation_queries,
                query_limit=query_limit,
            )
            if lexical.high_water_sequence != dense.high_water_sequence:
                self._healthy = False
                raise HybridIndexUnhealthyError("compaction snapshots have different sequences")
            if lexical.documents != dense.documents:
                self._healthy = False
                raise HybridIndexUnhealthyError("compaction snapshots have different documents")
            return HybridCompactionSnapshot(lexical=lexical, dense=dense)
