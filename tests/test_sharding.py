from collections import Counter

import pytest

from dynamic_hybrid_retrieval.sharding import ShardMap, ShardNode, changed_primaries


def topology(*names: str, epoch: int = 1, replicas: int = 1) -> ShardMap:
    return ShardMap(
        tuple(ShardNode(name) for name in names), epoch=epoch, replication_factor=replicas
    )


def test_routing_is_deterministic_and_independent_of_node_order() -> None:
    left = topology("a", "b", "c", epoch=7, replicas=2)
    right = topology("c", "a", "b", epoch=7, replicas=2)
    assert left.route("doc-123") == right.route("doc-123")
    assert len(left.route("doc-123").all_shards) == 2


def test_adding_shard_moves_only_keys_to_new_shard() -> None:
    keys = tuple(f"doc-{position}" for position in range(5_000))
    before = topology("a", "b", "c")
    after = topology("a", "b", "c", "d", epoch=2)
    moved = changed_primaries(keys, before, after)
    assert moved
    assert all(after.route(key).primary == "d" for key in moved)
    assert 0.20 < len(moved) / len(keys) < 0.30


def test_removing_shard_only_moves_keys_owned_by_removed_shard() -> None:
    keys = tuple(f"doc-{position}" for position in range(5_000))
    before = topology("a", "b", "c", "d")
    after = topology("a", "b", "c", epoch=2)
    moved = changed_primaries(keys, before, after)
    assert all(before.route(key).primary == "d" for key in moved)


def test_equal_weights_distribute_keys_evenly() -> None:
    shard_map = topology("a", "b", "c", "d")
    counts = Counter(shard_map.route(f"doc-{position}").primary for position in range(10_000))
    assert all(2_300 < count < 2_700 for count in counts.values())


def test_weighted_shard_receives_larger_share() -> None:
    shard_map = ShardMap((ShardNode("small"), ShardNode("large", 3)), epoch=1)
    counts = Counter(shard_map.route(f"doc-{position}").primary for position in range(10_000))
    assert 2.7 < counts["large"] / counts["small"] < 3.3


def test_topology_and_request_validation() -> None:
    with pytest.raises(ValueError, match="at least one"):
        ShardMap((), epoch=1)
    with pytest.raises(ValueError, match="unique"):
        ShardMap((ShardNode("a"), ShardNode("a")), epoch=1)
    with pytest.raises(ValueError, match="replication"):
        topology("a", replicas=2)
    with pytest.raises(ValueError, match="positive"):
        ShardNode("a", 0)
    with pytest.raises(ValueError, match="unique"):
        topology("a").route_many(("doc", "doc"))
