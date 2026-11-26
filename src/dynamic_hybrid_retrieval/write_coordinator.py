"""Primary-first replication of ordered events to shard-local WALs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from threading import Lock
from typing import Protocol

from dynamic_hybrid_retrieval.event_log import EventLog, canonical_event_bytes
from dynamic_hybrid_retrieval.models import IndexEvent
from dynamic_hybrid_retrieval.sharding import ShardMap


class WriteRejected(RuntimeError):
    """The request cannot be applied under the current routing state."""


@dataclass(frozen=True, slots=True)
class WriteReceipt:
    primary: str
    sequence_number: int
    acknowledged_shards: tuple[str, ...]
    failed_shards: tuple[str, ...]


class QuorumUnavailable(RuntimeError):
    def __init__(self, receipt: WriteReceipt) -> None:
        super().__init__("write quorum was not reached")
        self.receipt = receipt


class ShardWriteClient(Protocol):
    def append(self, event: IndexEvent, *, topology_epoch: int) -> None:
        """Return only after this exact event is durable."""
        ...


class LocalWalShard:
    """Local shard adapter with an injected, fixed topology."""

    def __init__(self, shard_id: str, topology: ShardMap, wal: EventLog) -> None:
        if not wal.fsync:
            raise ValueError("replicated writes require WAL fsync")
        self._shard_id = shard_id
        self._topology = topology
        self._wal = wal

    def append(self, event: IndexEvent, *, topology_epoch: int) -> None:
        assignment = self._topology.route(event.document_id)
        if topology_epoch != assignment.topology_epoch:
            raise WriteRejected("topology epoch mismatch")
        if self._shard_id not in assignment.all_shards:
            raise WriteRejected("shard does not own this document")
        self._wal.append(event)


class WriteCoordinator:
    """Serialize writes within one coordinator and fixed topology epoch."""

    def __init__(
        self,
        topology: ShardMap,
        clients: Mapping[str, ShardWriteClient],
        *,
        quorum: int | None = None,
    ) -> None:
        factor = topology.replication_factor
        self._quorum = factor // 2 + 1 if quorum is None else quorum
        if not 1 <= self._quorum <= factor:
            raise ValueError("quorum must be between one and replication factor")
        self._topology = topology
        self._clients = dict(clients)
        self._lock = Lock()
        self._pending: bytes | None = None
        self._last_sequence = 0
        self._last_payload: bytes | None = None

    def apply(self, event: IndexEvent, *, topology_epoch: int) -> WriteReceipt:
        with self._lock:
            assignment = self._topology.route(event.document_id)
            if topology_epoch != assignment.topology_epoch:
                raise WriteRejected("topology epoch mismatch")
            payload = canonical_event_bytes(event)
            if self._pending is not None and self._pending != payload:
                raise WriteRejected("retry the pending event before submitting another write")
            if event.sequence_number < self._last_sequence:
                raise WriteRejected("sequence precedes the last acknowledged write")
            if event.sequence_number == self._last_sequence and payload != self._last_payload:
                raise WriteRejected("sequence belongs to a different event")
            self._pending = payload
            acknowledged: list[str] = []
            failed: list[str] = []
            for shard_id in assignment.all_shards:
                try:
                    self._clients[shard_id].append(event, topology_epoch=topology_epoch)
                except Exception:
                    failed.append(shard_id)
                    if shard_id == assignment.primary:
                        break
                else:
                    acknowledged.append(shard_id)
            receipt = WriteReceipt(
                assignment.primary, event.sequence_number, tuple(acknowledged), tuple(failed)
            )
            if len(acknowledged) < self._quorum:
                raise QuorumUnavailable(receipt)
            self._last_sequence = event.sequence_number
            self._last_payload = payload
            self._pending = None
            return receipt
