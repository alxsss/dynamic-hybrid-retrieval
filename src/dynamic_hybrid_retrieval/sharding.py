"""Deterministic weighted rendezvous routing for distributed index shards."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ShardNode:
    shard_id: str
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.shard_id.strip():
            raise ValueError("shard_id cannot be empty")
        if not math.isfinite(self.weight) or self.weight <= 0:
            raise ValueError("shard weight must be finite and positive")


@dataclass(frozen=True, slots=True)
class ShardAssignment:
    topology_epoch: int
    document_id: str
    primary: str
    replicas: tuple[str, ...]

    @property
    def all_shards(self) -> tuple[str, ...]:
        return (self.primary, *self.replicas)


class ShardMap:
    """Immutable topology that maps keys using weighted rendezvous hashing."""

    def __init__(
        self,
        nodes: tuple[ShardNode, ...],
        *,
        epoch: int,
        replication_factor: int = 1,
    ) -> None:
        if epoch < 1:
            raise ValueError("topology epoch must be positive")
        if not nodes:
            raise ValueError("at least one shard is required")
        if len({node.shard_id for node in nodes}) != len(nodes):
            raise ValueError("shard IDs must be unique")
        if not 1 <= replication_factor <= len(nodes):
            raise ValueError("replication factor must be between one and shard count")
        self.nodes = tuple(sorted(nodes, key=lambda node: node.shard_id))
        self.epoch = epoch
        self.replication_factor = replication_factor

    @staticmethod
    def _score(document_id: str, node: ShardNode) -> float:
        payload = f"{document_id}\0{node.shard_id}".encode()
        digest = hashlib.blake2b(payload, digest_size=16, person=b"dhr-shard-v1").digest()
        integer = int.from_bytes(digest, "big")
        uniform = (integer + 1) / (2**128 + 1)
        return -math.log(uniform) / node.weight

    def route(self, document_id: str) -> ShardAssignment:
        if not document_id.strip():
            raise ValueError("document_id cannot be empty")
        ranked = sorted(
            self.nodes,
            key=lambda node: (self._score(document_id, node), node.shard_id),
        )[: self.replication_factor]
        return ShardAssignment(
            topology_epoch=self.epoch,
            document_id=document_id,
            primary=ranked[0].shard_id,
            replicas=tuple(node.shard_id for node in ranked[1:]),
        )

    def route_many(self, document_ids: tuple[str, ...]) -> dict[str, ShardAssignment]:
        if len(document_ids) != len(set(document_ids)):
            raise ValueError("document IDs must be unique")
        return {document_id: self.route(document_id) for document_id in document_ids}


def changed_primaries(
    document_ids: tuple[str, ...],
    before: ShardMap,
    after: ShardMap,
) -> frozenset[str]:
    """Return keys whose primary ownership changes between topology epochs."""

    return frozenset(
        document_id
        for document_id in document_ids
        if before.route(document_id).primary != after.route(document_id).primary
    )
