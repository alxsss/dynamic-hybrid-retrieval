from pathlib import Path

import pytest

from dynamic_hybrid_retrieval.event_log import EventLog
from dynamic_hybrid_retrieval.generate import generate_documents, generate_events
from dynamic_hybrid_retrieval.sharding import ShardMap, ShardNode
from dynamic_hybrid_retrieval.write_coordinator import (
    LocalWalShard,
    QuorumUnavailable,
    WriteCoordinator,
    WriteRejected,
)


def test_retry_completes_quorum_without_duplicating_primary_wal(tmp_path: Path) -> None:
    event, following = generate_events(generate_documents(2), 2)
    topology = ShardMap((ShardNode("a"), ShardNode("b")), epoch=2, replication_factor=2)
    assignment = topology.route(event.document_id)
    primary_log = EventLog(tmp_path / "primary.jsonl")
    replica_log = EventLog(tmp_path / "replica.jsonl")
    primary = LocalWalShard(assignment.primary, topology, primary_log)
    replica = LocalWalShard(assignment.replicas[0], topology, replica_log)
    original_append = replica.append

    def unavailable(*args: object, **kwargs: object) -> None:
        raise OSError("replica unavailable")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(replica, "append", unavailable)
        writer = WriteCoordinator(
            topology, {assignment.primary: primary, assignment.replicas[0]: replica}
        )
        with pytest.raises(QuorumUnavailable) as failure:
            writer.apply(event, topology_epoch=2)
        assert failure.value.receipt.acknowledged_shards == (assignment.primary,)
        with pytest.raises(WriteRejected, match="pending"):
            writer.apply(following, topology_epoch=2)

    assert replica.append == original_append
    receipt = writer.apply(event, topology_epoch=2)
    assert receipt.acknowledged_shards == assignment.all_shards
    assert list(primary_log.iter_events()) == list(replica_log.iter_events()) == [event]


def test_stale_epoch_is_rejected_at_coordinator_and_shard(tmp_path: Path) -> None:
    event = next(generate_events(generate_documents(1), 1))
    topology = ShardMap((ShardNode("a"),), epoch=2)
    wal = EventLog(tmp_path / "events.jsonl")
    shard = LocalWalShard("a", topology, wal)
    with pytest.raises(WriteRejected, match="epoch"):
        WriteCoordinator(topology, {"a": shard}).apply(event, topology_epoch=1)
    with pytest.raises(WriteRejected, match="epoch"):
        shard.append(event, topology_epoch=1)
    assert not wal.path.exists()
