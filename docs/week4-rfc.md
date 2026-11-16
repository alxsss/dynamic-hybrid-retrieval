# Week 4 RFC: Distributed Indexing and Query Coordination

Status: Day 1 complete

## Goal

Extend the single-host reference engine with explicit distributed-system boundaries. Week 4 focuses
on partition ownership, replicated writes, scatter-gather queries, topology changes, backpressure,
and observability. It does not attempt to implement a consensus algorithm; topology epochs and a
pluggable control-plane contract make that boundary explicit.

## Invariants

1. One document has exactly one primary shard in a topology epoch.
2. Primary and replica assignments are deterministic and contain no duplicates.
3. Stale topology epochs cannot acknowledge writes.
4. A write is acknowledged only after its configured durability quorum.
5. Global top-K is deterministic despite shard response ordering.
6. Rebalancing never makes two epochs writable for the same key without fencing.
7. Partial shard failure is visible in query responses and metrics.

## Seven-day plan

### Day 1: Partitioning and topology — complete

- Implement weighted rendezvous hashing for primary and replica placement.
- Version immutable topologies with monotonically increasing epochs.
- Demonstrate bounded movement when shards are added or removed.
- Document the control-plane versus data-plane boundary.

### Day 2: Scatter-gather query coordination

- Fan queries out to shards with per-shard deadlines.
- Merge deterministic global top-K results and over-fetch safely.
- Return explicit partial-result metadata when policy permits degraded reads.

### Day 3: Replicated write protocol

- Route writes to primaries and replicate ordered WAL records.
- Support configurable acknowledgement quorum and idempotent retries.
- Fence stale primaries using topology epochs.

### Day 4: Rebalancing

- Snapshot source shards, stream catch-up events, and validate destination state.
- Cut ownership over atomically at an epoch boundary.
- Model cancellation, retry, rollback, and throttling.

### Day 5: Backpressure and admission control

- Bound write queues, query concurrency, memory, and replication lag.
- Define overload responses and priority classes.
- Measure fairness and recovery after overload.

### Day 6: Distributed observability and failure drills

- Add shard, replica, coordinator, and topology metrics.
- Propagate request IDs and trace context across fan-out operations.
- Inject slow, unavailable, stale, and corrupt shard behavior.

### Day 7: Capacity plan and design review

- Produce storage, QPS, update-rate, recovery, and rebalancing estimates.
- Document consistency and availability tradeoffs.
- Prepare an interview-ready architecture walkthrough and known-limitations review.

## Day 1 design

Weighted rendezvous hashing scores every `(document, shard)` pair. The lowest weighted exponential
score wins; the next scores become replicas. The method needs no fixed hash ring or virtual nodes,
supports capacity weights, and has the key property needed for online expansion: adding a shard
moves only documents that select the new shard. Removing a shard moves only assignments that used it.

The topology epoch is returned with every assignment. In later days, storage nodes will reject
writes carrying an older epoch. A real deployment would store topology through a strongly
consistent control plane such as etcd or a database using compare-and-swap. This project treats that
control plane as an external contract rather than pretending an in-memory map provides consensus.
