"""In-process reference model for epoch-fenced shard migration."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from threading import Lock

from dynamic_hybrid_retrieval.event_log import replay_document_state
from dynamic_hybrid_retrieval.models import Document, IndexEvent, Operation
from dynamic_hybrid_retrieval.sharding import ShardMap


class MigrationRejected(RuntimeError):
    """The migration or write cannot proceed under the current state."""


@dataclass(frozen=True, slots=True)
class DocumentState:
    version: int
    document: Document | None


class RebalanceModel:
    """Model placement from an authoritative committed stream in one process."""

    def __init__(self, topology: ShardMap, events: Iterable[IndexEvent] = ()) -> None:
        self._lock = Lock()
        self._topology = ShardMap(
            topology.nodes, epoch=topology.epoch, replication_factor=topology.replication_factor
        )
        self._events = [event.model_copy(deep=True) for event in events]
        replay = replay_document_state(self._events)
        self._state = {
            key: DocumentState(version, replay.documents.get(key))
            for key, version in replay.latest_versions.items()
        }
        self._copies: dict[str, dict[str, DocumentState]] = {
            node.shard_id: {} for node in topology.nodes
        }
        for key, state in self._state.items():
            self._place(topology, self._copies, key, state)

    @staticmethod
    def _place(
        topology: ShardMap,
        copies: dict[str, dict[str, DocumentState]],
        key: str,
        state: DocumentState,
    ) -> None:
        for shard in topology.route(key).all_shards:
            copies[shard][key] = state

    @property
    def epoch(self) -> int:
        with self._lock:
            return self._topology.epoch

    def shard_state(self, shard: str) -> dict[str, DocumentState]:
        with self._lock:
            return {key: self._clone(state) for key, state in self._copies[shard].items()}

    @staticmethod
    def _clone(state: DocumentState) -> DocumentState:
        document = state.document
        return DocumentState(
            state.version, document.model_copy(deep=True) if document is not None else None
        )

    @staticmethod
    def _apply(state: dict[str, DocumentState], event: IndexEvent) -> DocumentState | None:
        previous = state.get(event.document_id)
        if previous is not None and event.version <= previous.version:
            return None
        current = DocumentState(
            event.version, event.document if event.operation is Operation.UPSERT else None
        )
        state[event.document_id] = current
        return current

    def append_committed(self, event: IndexEvent, *, epoch: int) -> None:
        with self._lock:
            if epoch != self._topology.epoch:
                raise MigrationRejected("write epoch mismatch")
            if self._events and event.sequence_number <= self._events[-1].sequence_number:
                raise MigrationRejected("committed sequence must advance")
            event = event.model_copy(deep=True)
            current = self._apply(self._state, event)
            if current is not None:
                self._place(self._topology, self._copies, event.document_id, current)
            self._events.append(event)

    def migrate(
        self,
        target: ShardMap,
        *,
        batch_size: int = 256,
        on_batch: Callable[[], None] | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> None:
        """Stage a snapshot, catch up under a write fence, then publish atomically."""
        if batch_size < 1:
            raise ValueError("batch size must be positive")
        target = ShardMap(
            target.nodes, epoch=target.epoch, replication_factor=target.replication_factor
        )
        with self._lock:
            source_epoch = self._topology.epoch
            if target.epoch <= source_epoch:
                raise MigrationRejected("target epoch must advance")
            cursor = len(self._events)
            staged_state = dict(self._state)

        staged: dict[str, dict[str, DocumentState]] = {node.shard_id: {} for node in target.nodes}
        for position, (key, state) in enumerate(staged_state.items()):
            if position % batch_size == 0:
                if cancelled is not None and cancelled():
                    raise MigrationRejected("migration cancelled")
                if on_batch is not None:
                    on_batch()
            self._place(target, staged, key, state)
        if cancelled is not None and cancelled():
            raise MigrationRejected("migration cancelled")

        with self._lock:
            if self._topology.epoch != source_epoch:
                raise MigrationRejected("source topology changed during migration")
            for event in self._events[cursor:]:
                current = self._apply(staged_state, event)
                if current is not None:
                    self._place(target, staged, event.document_id, current)
            if staged_state != self._state or any(
                staged[shard].get(key) != state
                for key, state in self._state.items()
                for shard in target.route(key).all_shards
            ):
                raise MigrationRejected("destination validation failed")
            self._copies = staged
            self._topology = target
