# Dynamic Hybrid Retrieval

A production-oriented reference implementation of single-host dynamic lexical
and dense retrieval. It combines immutable baseline indexes with searchable
in-memory deltas, durably records accepted changes in a checksummed write-ahead
log, and compacts logical state into atomically activated immutable generations.

The project exists to make indexing mechanics visible and testable before later
comparison with OpenSearch. It currently covers deterministic tokenization and
BM25 ranking, persistent HNSW dense retrieval, inserts and updates,
tombstone-based deletion, crash replay, compaction validation, rollback,
benchmarking, and read-only operational health checks.

## Architecture

- Immutable BM25 and HNSW baseline generations
- Searchable in-memory lexical and dense deltas
- Version masking and deletion tombstones
- Checksummed append-only write-ahead logging and deterministic replay
- Validated compaction with atomic generation activation and rollback
- Deterministic benchmarks and correctness gates

Reciprocal-rank fusion, metadata filtering, and hand-verified relevance
evaluation are active milestones documented in `docs/week2-rfc.md`.

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

See [the RFC](docs/rfc.md), [benchmark plan](docs/benchmark-plan.md), and
[operations guide](docs/operations.md) for design assumptions and failure
semantics.

## Scope

The current system is intentionally single-process and single-writer. It is
crash-consistent on a local filesystem with atomic rename support, but it is not
a distributed or highly available search service. OpenSearch comparison,
replication, sharding, and hybrid fusion remain future milestones.
