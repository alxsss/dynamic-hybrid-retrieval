"""WAL-backed baseline-plus-delta dense retrieval."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from usearch.index import Index

from dynamic_hybrid_retrieval.dense import (
    DenseSearchResult,
    HNSWIndex,
    document_text,
)
from dynamic_hybrid_retrieval.event_log import EventLog
from dynamic_hybrid_retrieval.index import ApplyResult, ApplyStatus
from dynamic_hybrid_retrieval.models import Document, IndexEvent, Operation


@dataclass(frozen=True, slots=True)
class DynamicDenseStatistics:
    live_document_count: int
    baseline_document_count: int
    delta_document_count: int
    tombstone_count: int
    masked_baseline_count: int
    last_sequence_number: int


@dataclass(frozen=True, slots=True)
class DenseCompactionSnapshot:
    documents: tuple[Document, ...]
    high_water_sequence: int
    exact_query_results: tuple[tuple[str, tuple[DenseSearchResult, ...]], ...]


@dataclass(frozen=True, slots=True)
class DenseCompactionResult:
    generation_path: Path
    high_water_sequence: int
    document_count: int
    validated_query_count: int


class DynamicHNSWIndex:
    """Combine an immutable HNSW baseline with a mutable in-memory HNSW delta."""

    def __init__(
        self,
        baseline: HNSWIndex,
        event_log: EventLog,
        *,
        baseline_sequence_number: int = 0,
    ) -> None:
        if baseline_sequence_number < 0:
            raise ValueError("baseline_sequence_number cannot be negative")
        self.baseline = baseline
        self.event_log = event_log
        self.baseline_sequence_number = baseline_sequence_number
        self._lock = threading.RLock()
        self._baseline_documents = {
            document.document_id: document for document in baseline.documents
        }
        self._delta_documents: dict[str, Document] = {}
        self._delta_keys: dict[str, int] = {}
        self._documents_by_key: dict[int, Document] = {}
        self._next_key = 0
        self._tombstones: dict[str, int] = {}
        self._latest_versions = {
            document.document_id: document.version for document in baseline.documents
        }
        config = baseline.config
        self._delta_index = Index(
            ndim=baseline.embedding_model.dimension,
            metric="cos",
            dtype="f32",
            connectivity=config.connectivity,
            expansion_add=config.expansion_add,
            expansion_search=config.expansion_search,
        )
        self._last_sequence_number = baseline_sequence_number
        for event in event_log.iter_events(after_sequence=baseline_sequence_number):
            self._apply_recovered_event(event)

    @property
    def statistics(self) -> DynamicDenseStatistics:
        with self._lock:
            masked = sum(
                document_id in self._tombstones or document_id in self._delta_documents
                for document_id in self._baseline_documents
            )
            return DynamicDenseStatistics(
                live_document_count=(
                    len(self._baseline_documents) - masked + len(self._delta_documents)
                ),
                baseline_document_count=len(self._baseline_documents),
                delta_document_count=len(self._delta_documents),
                tombstone_count=len(self._tombstones),
                masked_baseline_count=masked,
                last_sequence_number=self._last_sequence_number,
            )

    @property
    def tombstones(self) -> dict[str, int]:
        with self._lock:
            return dict(self._tombstones)

    def _remove_delta_document(self, document_id: str) -> None:
        key = self._delta_keys.pop(document_id, None)
        if key is None:
            return
        self._delta_index.remove(key)
        self._documents_by_key.pop(key)
        self._delta_documents.pop(document_id)

    def _add_delta_document(self, document: Document) -> None:
        vector = self.baseline.embedding_model.embed_documents([document_text(document)])
        expected_shape = (1, self.baseline.embedding_model.dimension)
        if vector.shape != expected_shape:
            raise ValueError(f"embedding shape {vector.shape} does not match {expected_shape}")
        key = self._next_key
        self._next_key += 1
        self._delta_index.add(key, vector[0])
        self._delta_keys[document.document_id] = key
        self._documents_by_key[key] = document
        self._delta_documents[document.document_id] = document

    def _apply_to_memory(self, event: IndexEvent) -> None:
        self._remove_delta_document(event.document_id)
        self._latest_versions[event.document_id] = event.version
        if event.operation is Operation.UPSERT:
            assert event.document is not None
            self._add_delta_document(event.document)
            self._tombstones.pop(event.document_id, None)
        else:
            self._tombstones[event.document_id] = event.version
        self._last_sequence_number = max(self._last_sequence_number, event.sequence_number)

    def _apply_recovered_event(self, event: IndexEvent) -> None:
        if event.sequence_number <= self._last_sequence_number:
            return
        if event.version <= self._latest_versions.get(event.document_id, 0):
            self._last_sequence_number = event.sequence_number
            return
        self._apply_to_memory(event)

    def apply(self, event: IndexEvent) -> ApplyResult:
        """Durably append and apply an event, including a pre-appended shared-WAL event."""

        with self._lock:
            if event.sequence_number <= self._last_sequence_number:
                append_result = self.event_log.append(event)
                assert not append_result.appended
                return ApplyResult(
                    ApplyStatus.DUPLICATE,
                    event.sequence_number,
                    event.document_id,
                    event.version,
                )
            if event.version <= self._latest_versions.get(event.document_id, 0):
                return ApplyResult(
                    ApplyStatus.STALE,
                    event.sequence_number,
                    event.document_id,
                    event.version,
                )
            # append() is idempotent. If the lexical path already wrote this
            # exact event to a shared WAL, dense application still proceeds.
            self.event_log.append(event)
            self._apply_to_memory(event)
            return ApplyResult(
                ApplyStatus.APPLIED,
                event.sequence_number,
                event.document_id,
                event.version,
            )

    def _baseline_is_current(self, document_id: str) -> bool:
        document = self._baseline_documents[document_id]
        return (
            document_id not in self._tombstones
            and document_id not in self._delta_documents
            and self._latest_versions.get(document_id) == document.version
        )

    def search(
        self, query: str, *, limit: int = 10, exact: bool = False
    ) -> list[DenseSearchResult]:
        if limit < 1:
            raise ValueError("limit must be positive")
        with self._lock:
            masked_baseline_count = sum(
                not self._baseline_is_current(document_id)
                for document_id in self._baseline_documents
            )
            baseline_limit = min(
                self.baseline.document_count,
                limit + masked_baseline_count,
            )
            baseline_results = (
                self.baseline.search(query, limit=baseline_limit, exact=exact)
                if baseline_limit
                else []
            )
            candidates = [
                result
                for result in baseline_results
                if self._baseline_is_current(result.document_id)
            ]

            if self._delta_documents:
                query_vector = self.baseline.embedding_model.embed_query(query)
                expected_shape = (1, self.baseline.embedding_model.dimension)
                if query_vector.shape != expected_shape:
                    raise ValueError(
                        f"embedding shape {query_vector.shape} does not match {expected_shape}"
                    )
                matches = self._delta_index.search(
                    query_vector[0],
                    count=min(limit, len(self._delta_documents)),
                    exact=exact,
                )
                candidates.extend(
                    DenseSearchResult(
                        document_id=self._documents_by_key[int(key)].document_id,
                        score=1.0 - float(distance),
                        distance=float(distance),
                        version=self._documents_by_key[int(key)].version,
                        document=self._documents_by_key[int(key)],
                    )
                    for key, distance in zip(matches.keys, matches.distances, strict=True)
                )
            return sorted(candidates, key=lambda item: (-item.score, item.document_id))[:limit]

    def _current_document(self, document_id: str) -> Document | None:
        delta = self._delta_documents.get(document_id)
        if delta is not None:
            return delta
        baseline = self._baseline_documents.get(document_id)
        if baseline is None or not self._baseline_is_current(document_id):
            return None
        return baseline

    def snapshot_for_compaction(
        self,
        validation_queries: tuple[str, ...] = (),
        *,
        query_limit: int = 10,
    ) -> DenseCompactionSnapshot:
        with self._lock:
            document_ids = set(self._baseline_documents) | set(self._delta_documents)
            documents = tuple(
                document
                for document_id in sorted(document_ids)
                if (document := self._current_document(document_id)) is not None
            )
            results = tuple(
                (query, tuple(self.search(query, limit=query_limit, exact=True)))
                for query in validation_queries
            )
            return DenseCompactionSnapshot(documents, self._last_sequence_number, results)

    def compact_to(
        self,
        generation_path: str | Path,
        *,
        validation_queries: tuple[str, ...] = (),
        query_limit: int = 10,
    ) -> DenseCompactionResult:
        """Build and exact-validate an immutable dense generation at a high-water mark."""

        snapshot = self.snapshot_for_compaction(
            validation_queries,
            query_limit=query_limit,
        )
        compacted = HNSWIndex.build(
            snapshot.documents,
            self.baseline.embedding_model,
            config=self.baseline.config,
        )
        path = compacted.save(generation_path)
        loaded = HNSWIndex.load(path, self.baseline.embedding_model)
        for query, expected in snapshot.exact_query_results:
            actual = loaded.search(query, limit=query_limit, exact=True)
            expected_signature = [(item.document_id, item.version, item.score) for item in expected]
            actual_signature = [(item.document_id, item.version, item.score) for item in actual]
            if not _equivalent_results(expected_signature, actual_signature):
                raise RuntimeError(f"dense compaction changed exact results for {query!r}")
        return DenseCompactionResult(
            path,
            snapshot.high_water_sequence,
            len(snapshot.documents),
            len(snapshot.exact_query_results),
        )


def _equivalent_results(
    expected: list[tuple[str, int, float]],
    actual: list[tuple[str, int, float]],
) -> bool:
    if len(expected) != len(actual):
        return False
    return all(
        left[:2] == right[:2] and np.isclose(left[2], right[2], rtol=1e-6, atol=1e-6)
        for left, right in zip(expected, actual, strict=True)
    )
