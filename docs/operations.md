# Day 7 operations and health checks

## Serving-readiness check

The `status` command performs the same important reads as startup without
changing repository state. It verifies the checksummed active manifest, loads
and validates the complete baseline generation, scans every event-log record,
replays events after the baseline high-water mark, and reports the reconstructed
logical state as JSON.

```bash
.venv/bin/dhr status \
  --repository "$DHR_DAY6_DIR/repository" \
  --event-log "$DHR_DAY6_DIR/events.jsonl"
```

Exit code zero and `"healthy": true` mean the persisted state can be loaded and
the WAL has no incomplete final record. A nonzero exit is intentionally suitable
for a container readiness probe, deployment gate, or operator script.

## Important fields

- `active_generation` and `baseline_sequence_number` identify the checkpoint.
- `last_event_sequence_number` identifies the durable WAL high-water mark.
- `delta_document_count` and `tombstone_count` describe replayed changes not yet compacted.
- `event_log_truncated_tail` detects a crash fragment that must be repaired before writes.
- `repository_bytes` and `event_log_bytes` support storage monitoring.
- `recovery_duration_ms` measures startup work.
- `errors` contains actionable validation failures and must be empty for health.

## Failure policy

The check is fail-closed. Missing manifests, checksum failures, invalid index
files, corrupt complete WAL records, and replay failures result in unhealthy
status. An incomplete final WAL record is recoverable, but is still unhealthy
until an operator explicitly invokes the existing repair operation. No status
check silently truncates, replaces, or deletes data.

This is single-host health, not high availability. A replicated log, redundant
serving replicas, leader election, and automated failover remain later work.
