# Dynamic Hybrid Retrieval

A production-oriented reference implementation of single-host dynamic lexical
and dense retrieval. It combines immutable baseline indexes with searchable
in-memory deltas, durably records accepted changes in a checksummed write-ahead
log, and compacts logical state into atomically activated immutable generations.

The project exists to make indexing mechanics visible and testable before later
comparison with OpenSearch. It currently covers deterministic tokenization and
BM25 ranking, persistent HNSW dense retrieval, reciprocal-rank fusion, typed
metadata filtering, inserts and updates, tombstone-based deletion, crash replay,
compaction validation, rollback, benchmarking, relevance evaluation, and
read-only operational health checks.

## Architecture

- Immutable BM25 and HNSW baseline generations
- Searchable in-memory lexical and dense deltas
- Version masking and deletion tombstones
- Checksummed append-only write-ahead logging and deterministic replay
- A fail-closed coordinator for consistent hybrid updates, reads, and snapshots
- Validated, atomically activated hybrid generations containing BM25 and HNSW
- Deterministic benchmarks and correctness gates

Week 2's dense, hybrid, filtered, and evaluated retrieval is documented in
`docs/week2-rfc.md` and `docs/week2-results.md`. Week 3 production serving and
operational hardening is tracked in `docs/week3-rfc.md`.

## Development checks

```bash
uv sync
.venv/bin/ruff check .
.venv/bin/mypy src
.venv/bin/pytest
```

## Acceptance benchmark

```bash
.venv/bin/dhr benchmark \
  --output .local/acceptance \
  --documents 100000 \
  --events 10000 \
  --queries 200
```

The generated `report.json` includes reproducibility metadata, pre-update and
post-update latency percentiles, freshness, recovery throughput, storage size,
and compaction equivalence. The command returns nonzero when a correctness gate
fails.

## Repository health

```bash
.venv/bin/dhr status \
  --repository .local/acceptance/repository \
  --event-log .local/acceptance/events.jsonl
```

See [the original RFC](docs/rfc.md), [Week 3 RFC](docs/week3-rfc.md),
[benchmark plan](docs/benchmark-plan.md), and [operations guide](docs/operations.md)
for design assumptions and failure semantics.

## Scope

The current system is intentionally single-process and single-writer. It is
crash-consistent on a local filesystem with atomic rename support, but it is not
a distributed or highly available search service. Coordinated hybrid artifact
An HTTP service, WAL retention, and OpenSearch comparison are remaining Week 3
milestones; replication and sharding remain out of scope.
