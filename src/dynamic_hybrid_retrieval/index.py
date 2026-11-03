"""Dynamic baseline-plus-delta BM25 index."""

from __future__ import annotations

import math
import threading
from collections import Counter, defaultdict
from dataclasses import dataclass
from enum import StrEnum

from dynamic_hybrid_retrieval.bm25 import BM25Index, SearchResult
from dynamic_hybrid_retrieval.event_log import EventLog
from dynamic_hybrid_retrieval.models import Document, IndexEvent, Operation


class ApplyStatus(StrEnum):
    """Outcome of applying one event to the dynamic index."""

    APPLIED = "applied"
    DUPLICATE = "duplicate"
    STALE = "stale"


@dataclass(frozen=True, slots=True)
class ApplyResult:
    status: ApplyStatus
    sequence_number: int
    document_id: str
    version: int


@dataclass(frozen=True, slots=True)
class DynamicIndexStatistics:
    live_document_count: int
    baseline_document_count: int
    delta_document_count: int
    tombstone_count: int
    total_document_length: int
    average_document_length: float
    unique_term_count: int
    last_sequence_number: int


@dataclass(frozen=True, slots=True)
class CompactionSnapshot:
    """Consistent logical state captured at one event-log high-water mark."""

    documents: tuple[Document, ...]
    high_water_sequence: int
    query_results: tuple[tuple[str, tuple[SearchResult, ...]], ...]


class DynamicIndex:
    """Present an immutable baseline and mutable delta as one logical BM25 index.

    All public reads and writes are serialized by a re-entrant lock. Events are
    durably appended before their changes become visible in memory.
    """

    def __init__(
        self,
        baseline: BM25Index,
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

        self._delta_documents: dict[str, Document] = {}
        self._delta_lengths: dict[str, int] = {}
        self._delta_term_frequencies: dict[str, dict[str, int]] = {}
        self._delta_postings: defaultdict[str, dict[str, int]] = defaultdict(dict)
        self._tombstones: dict[str, int] = {}
        self._latest_versions = {
            document_id: document.version for document_id, document in baseline.documents.items()
        }

        self._live_document_count = baseline.document_count
        self._total_document_length = sum(baseline.document_lengths.values())
        self._document_frequencies: Counter[str] = Counter(
            {term: len(postings) for term, postings in baseline.postings.items()}
        )
        self._last_sequence_number = baseline_sequence_number

        for event in event_log.iter_events(after_sequence=baseline_sequence_number):
            self._apply_recovered_event(event)

    @property
    def statistics(self) -> DynamicIndexStatistics:
        with self._lock:
            average_length = (
                self._total_document_length / self._live_document_count
                if self._live_document_count
                else 0.0
            )
            return DynamicIndexStatistics(
                live_document_count=self._live_document_count,
                baseline_document_count=self.baseline.document_count,
                delta_document_count=len(self._delta_documents),
                tombstone_count=len(self._tombstones),
                total_document_length=self._total_document_length,
                average_document_length=average_length,
                unique_term_count=len(self._document_frequencies),
                last_sequence_number=self._last_sequence_number,
            )

    @property
    def latest_versions(self) -> dict[str, int]:
        with self._lock:
            return dict(self._latest_versions)

    @property
    def tombstones(self) -> dict[str, int]:
        with self._lock:
            return dict(self._tombstones)

    def _terms_for_document(self, document: Document) -> tuple[dict[str, int], int]:
        tokens = self.baseline.tokenizer.tokenize(f"{document.title}\n{document.body}")
        return dict(Counter(tokens)), len(tokens)

    def _current_document(self, document_id: str) -> Document | None:
        delta_document = self._delta_documents.get(document_id)
        if delta_document is not None:
            return delta_document
        baseline_document = self.baseline.documents.get(document_id)
        if baseline_document is None or document_id in self._tombstones:
            return None
        if self._latest_versions.get(document_id) != baseline_document.version:
            return None
        return baseline_document

    def _remove_live_document(self, document: Document) -> None:
        document_id = document.document_id
        if document_id in self._delta_documents:
            term_frequencies = self._delta_term_frequencies.pop(document_id)
            document_length = self._delta_lengths.pop(document_id)
            self._delta_documents.pop(document_id)
            for term in term_frequencies:
                term_postings = self._delta_postings[term]
                term_postings.pop(document_id)
                if not term_postings:
                    self._delta_postings.pop(term)
        else:
            term_frequencies, document_length = self._terms_for_document(document)

        self._live_document_count -= 1
        self._total_document_length -= document_length
        for term in term_frequencies:
            self._document_frequencies[term] -= 1
            if self._document_frequencies[term] == 0:
                del self._document_frequencies[term]

    def _add_delta_document(self, document: Document) -> None:
        term_frequencies, document_length = self._terms_for_document(document)
        self._delta_documents[document.document_id] = document
        self._delta_lengths[document.document_id] = document_length
        self._delta_term_frequencies[document.document_id] = term_frequencies
        for term, frequency in term_frequencies.items():
            self._delta_postings[term][document.document_id] = frequency
            self._document_frequencies[term] += 1
        self._live_document_count += 1
        self._total_document_length += document_length

    def _apply_to_memory(self, event: IndexEvent) -> None:
        current_document = self._current_document(event.document_id)
        if current_document is not None:
            self._remove_live_document(current_document)

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
        latest_version = self._latest_versions.get(event.document_id, 0)
        if event.version <= latest_version:
            self._last_sequence_number = event.sequence_number
            return
        self._apply_to_memory(event)

    def apply(self, event: IndexEvent) -> ApplyResult:
        """Persist and apply a new event, with stale and duplicate detection."""

        with self._lock:
            if event.sequence_number <= self.event_log.last_sequence_number:
                append_result = self.event_log.append(event)
                assert not append_result.appended
                return ApplyResult(
                    status=ApplyStatus.DUPLICATE,
                    sequence_number=event.sequence_number,
                    document_id=event.document_id,
                    version=event.version,
                )

            latest_version = self._latest_versions.get(event.document_id, 0)
            if event.version <= latest_version:
                return ApplyResult(
                    status=ApplyStatus.STALE,
                    sequence_number=event.sequence_number,
                    document_id=event.document_id,
                    version=event.version,
                )

            self.event_log.append(event)
            self._apply_to_memory(event)
            return ApplyResult(
                status=ApplyStatus.APPLIED,
                sequence_number=event.sequence_number,
                document_id=event.document_id,
                version=event.version,
            )

    def _is_current_baseline_document(self, document_id: str) -> bool:
        baseline_document = self.baseline.documents[document_id]
        return (
            document_id not in self._tombstones
            and document_id not in self._delta_documents
            and self._latest_versions.get(document_id) == baseline_document.version
        )

    def _inverse_document_frequency(self, term: str) -> float:
        document_frequency = self._document_frequencies.get(term, 0)
        if document_frequency == 0 or self._live_document_count == 0:
            return 0.0
        return math.log(
            1.0
            + (self._live_document_count - document_frequency + 0.5) / (document_frequency + 0.5)
        )

    def search(self, query: str, *, limit: int = 10) -> list[SearchResult]:
        """Search the baseline and delta with shared logical-corpus statistics."""

        if limit < 1:
            raise ValueError("limit must be at least one")
        query_terms = tuple(dict.fromkeys(self.baseline.tokenizer.tokenize(query)))
        if not query_terms:
            return []

        with self._lock:
            if not self._live_document_count:
                return []
            average_length = self._total_document_length / self._live_document_count
            scores: defaultdict[str, float] = defaultdict(float)

            for term in query_terms:
                idf = self._inverse_document_frequency(term)
                for posting in self.baseline.postings.get(term, ()):
                    if not self._is_current_baseline_document(posting.document_id):
                        continue
                    document_length = self.baseline.document_lengths[posting.document_id]
                    scores[posting.document_id] += self._score_term(
                        idf,
                        posting.term_frequency,
                        document_length,
                        average_length,
                    )

                for document_id, frequency in self._delta_postings.get(term, {}).items():
                    scores[document_id] += self._score_term(
                        idf,
                        frequency,
                        self._delta_lengths[document_id],
                        average_length,
                    )

            ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:limit]
            return [
                SearchResult(
                    document_id=document_id,
                    score=score,
                    version=self._latest_versions[document_id],
                    document=self._delta_documents.get(document_id)
                    or self.baseline.documents[document_id],
                )
                for document_id, score in ranked
            ]

    def snapshot_for_compaction(
        self,
        validation_queries: tuple[str, ...] = (),
        *,
        query_limit: int = 10,
    ) -> CompactionSnapshot:
        """Capture live documents and validation results under one read lock."""

        if query_limit < 1:
            raise ValueError("query_limit must be at least one")
        with self._lock:
            document_ids = set(self.baseline.documents) | set(self._delta_documents)
            documents = tuple(
                document
                for document_id in sorted(document_ids)
                if (document := self._current_document(document_id)) is not None
            )
            query_results = tuple(
                (query, tuple(self.search(query, limit=query_limit)))
                for query in validation_queries
            )
            return CompactionSnapshot(
                documents=documents,
                high_water_sequence=self._last_sequence_number,
                query_results=query_results,
            )

    def _score_term(
        self,
        inverse_document_frequency: float,
        term_frequency: int,
        document_length: int,
        average_document_length: float,
    ) -> float:
        normalization = self.baseline.k1 * (
            1.0
            - self.baseline.b
            + self.baseline.b * document_length / (average_document_length or 1.0)
        )
        return inverse_document_frequency * (
            term_frequency * (self.baseline.k1 + 1.0) / (term_frequency + normalization)
        )
