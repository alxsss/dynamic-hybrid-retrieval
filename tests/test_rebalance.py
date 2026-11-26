import pytest

from dynamic_hybrid_retrieval.generate import generate_documents, generate_events
from dynamic_hybrid_retrieval.rebalance import MigrationRejected, RebalanceModel
from dynamic_hybrid_retrieval.sharding import ShardMap, ShardNode


def test_snapshot_catches_up_tombstones_and_fences_old_epoch() -> None:
    *initial, deletion = generate_events(generate_documents(10), 10)
    source = ShardMap((ShardNode("a"),), epoch=1)
    target = ShardMap((ShardNode("b"), ShardNode("c")), epoch=2, replication_factor=2)
    cluster = RebalanceModel(source, initial)

    def during_copy() -> None:
        cluster.append_committed(deletion, epoch=1)

    cluster.migrate(target, on_batch=during_copy)
    assert cluster.epoch == 2
    assert cluster.shard_state("b") == cluster.shard_state("c")
    tombstone = cluster.shard_state("b")[deletion.document_id]
    assert tombstone.document is None
    assert tombstone.version == deletion.version
    with pytest.raises(MigrationRejected, match="epoch"):
        cluster.append_committed(deletion, epoch=1)

    cluster.migrate(ShardMap(source.nodes, epoch=3))
    assert cluster.shard_state("a")[deletion.document_id] == tombstone


def test_cancelled_migration_preserves_source_and_can_retry() -> None:
    source = ShardMap((ShardNode("a"),), epoch=1)
    target = ShardMap((ShardNode("b"),), epoch=2)
    cluster = RebalanceModel(source, generate_events(generate_documents(2), 2))
    original = cluster.shard_state("a")
    with pytest.raises(MigrationRejected, match="cancelled"):
        cluster.migrate(target, cancelled=lambda: True)
    assert cluster.epoch == 1
    assert cluster.shard_state("a") == original
    cluster.migrate(target)
    assert cluster.shard_state("b") == original
