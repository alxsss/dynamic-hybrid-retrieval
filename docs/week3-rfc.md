# Week 3 RFC: Production Serving and Operational Hardening

Status: In progress  
Last updated: 2026-11-13

## Goal

Turn the retrieval algorithms built in Weeks 1 and 2 into one operable service. The service must
present BM25 and HNSW as a single versioned index, recover deterministically, expose safe APIs and
health signals, and have measured behavior under concurrency and failure.

OpenSearch remains an alternative implementation, not a dependency of the custom engine. It will
be compared only after the custom path has a stable API and workload so the comparison uses the
same data, queries, filters, updates, and metrics.

## Production invariants

1. Lexical and dense retrieval use the same WAL and active generation high-water sequence.
2. An acknowledged mutation is visible to both retrievers.
3. A search never spans two logical index versions.
4. A partial in-memory mutation fails closed; restart repairs it by replaying the WAL.
5. Compaction activates lexical and dense artifacts together through one atomic manifest change.
6. Readiness is false during recovery and whenever consistency cannot be proven.
7. Deleted documents and superseded versions never appear in results.

## Seven-day implementation plan

### Day 1: Coordinated hybrid state

- Add `DynamicHybridIndex` as the only supported write and read boundary.
- Serialize update, search, and snapshot operations across BM25 and HNSW.
- Require one WAL and one baseline high-water sequence.
- Reject inconsistent startup state and poison a process after partial mutation.
- Verify deterministic repair by restart and WAL replay.

### Day 2: Atomic hybrid generations

- Store BM25 and HNSW artifacts beneath one generation directory.
- Record model/config fingerprints and checksums in a generation manifest.
- Build and validate both artifacts before atomically changing `active.json`.
- Retain the previous generation for rollback and ignore abandoned candidates.

### Day 3: Service boundary

- Add versioned HTTP endpoints for mutation, lexical/dense/hybrid search, health, and readiness.
- Validate request sizes, filter expressions, sequence numbers, and timeouts.
- Implement graceful startup, WAL recovery, dependency initialization, and graceful shutdown.
- Emit stable error codes without leaking internal details.

### Day 4: WAL lifecycle

- Segment the WAL and seal immutable segments.
- Checkpoint only after a hybrid generation is durably active.
- Retain enough segments for rollback, then delete only provably obsolete segments.
- Test crash tails, checksum corruption, interrupted rotation, replay, and disk-pressure behavior.

### Day 5: Concurrency, load, and failure testing

- Exercise concurrent readers, one writer, compaction, restart, and rollback.
- Measure update-to-search visibility, p50/p95/p99 query latency, throughput, and recovery time.
- Inject embedding, disk, manifest, and process failures and verify invariants.
- Report separate before-update, after-update, filtered, and post-compaction results.

### Day 6: OpenSearch comparison

- Run the same corpus, event stream, queries, filters, and relevance judgments against OpenSearch.
- Compare correctness, freshness, latency, operational complexity, and resource use.
- Document where a managed engine is preferable and what the custom engine demonstrates.

### Day 7: Release package

- Add container and local-development entry points, configuration reference, runbook, and threat model.
- Clean public-repository history and remove local artifacts and secrets.
- Re-run lint, type checking, tests, coverage, evaluation, benchmark, and recovery drills.
- Publish architecture, measured results, known limits, and reproducible commands.

## Day 1 design

`DynamicHybridIndex` owns an outer re-entrant lock. The existing BM25 and HNSW locks still protect
their own data structures, but the outer lock prevents readers from entering between the two halves
of an update. BM25 appends the event to the shared WAL before changing memory; HNSW then consumes
the same idempotent WAL record and changes its memory.

If HNSW application fails, BM25 may already have changed in memory. The coordinator marks itself
unhealthy and rejects subsequent reads and writes. The durable event is still in the WAL, so a new
process deterministically replays it into both views and becomes healthy. This is fail-stop behavior:
availability is sacrificed briefly rather than returning a mixture of document versions.

Startup performs a full logical-state comparison. Normal health checks use constant-time sequence,
count, and tombstone statistics. Coordinated compaction snapshots perform another full comparison
while holding the outer lock.

## Current limitations

- One process and one writer; distributed consensus and sharding are out of scope.
- Day 1 cannot atomically activate two persisted artifacts; Day 2 supplies that boundary.
- The in-process coordinator cannot prevent unsupported code from mutating a child index directly,
  but sequence divergence is detected and fails search. The service layer will expose only the
  coordinator.
- If embedding generation is nondeterministic, recovery may reproduce document state but not
  bit-identical vectors. Production model identity and determinism checks arrive with manifests.
