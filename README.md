# Dynamic Hybrid Retrieval

A production-oriented reference implementation of a single-host dynamic BM25
index. It combines an immutable baseline with a searchable in-memory delta,
durably records accepted changes in a checksummed write-ahead log, and compacts
logical state into atomically activated immutable generations.

The project exists to make indexing mechanics visible and testable before later
comparison with OpenSearch or addition of dense retrieval. It currently covers
deterministic tokenization and BM25 ranking, inserts and updates, tombstone-based
deletion, crash replay, compaction validation, rollback, benchmarking, and
read-only operational health checks.

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
replication, sharding, dense retrieval, and hybrid fusion are future milestones.
