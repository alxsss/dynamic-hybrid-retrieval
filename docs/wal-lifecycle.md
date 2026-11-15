# Segmented WAL Lifecycle

The segmented write-ahead log bounds individual file size while preserving the original durable,
checksummed record format and deterministic replay semantics.

## Layout

```text
wal/
├── wal.lock
├── checkpoint.json
├── segment-00000000000000000001-00000000000000100000.jsonl
├── segment-00000000000000100001-00000000000000200000.jsonl
└── active.jsonl
```

`active.jsonl` is the only mutable data file. It is sealed before the configured byte or event
limit by atomically renaming it to a filename containing its actual sequence range. The next append
creates a new active file. A missing active file is therefore a normal crash-safe state.

## Recovery rules

Recovery scans sealed files in sequence order and then scans the active file. Every record checksum,
filename range, and cross-segment sequence transition is validated. A newline-incomplete tail in
the active file can be truncated explicitly. The same condition in a sealed file is corruption and
cannot be repaired automatically.

## Checkpoint and retention

A successful hybrid compaction first activates the new complete generation. It then checkpoints:

- the active generation high-water sequence;
- the previous generation high-water sequence needed for rollback;
- the WAL's last accepted sequence; and
- the resulting safe retention floor.

The retention floor is:

```text
min(active generation sequence, rollback generation sequence)
```

If no rollback generation exists, it is the active generation sequence. Checkpoint metadata is
written and atomically replaced before eligible segments are deleted. A crash before deletion only
leaves extra safe data. A crash during deletion also leaves a valid subset because every deleted
segment was already covered by every retained baseline.

Segments are deletion candidates only when their ending sequence is at or below the floor. A
segment crossing the floor is retained whole. Recovery below the floor raises an explicit error;
it never returns an incomplete event stream.

## Disk pressure

`--wal-max-retained-bytes` sets a hard budget. An append that would exceed it is rejected before any
record bytes are written. Operators must compact and checkpoint, increase the budget, or free disk
space. The system does not silently discard uncheckpointed events.

Example service configuration:

```bash
.venv/bin/dhr serve \
  --repository .local/hybrid-repository \
  --event-log .local/wal \
  --segmented-wal \
  --wal-segment-bytes 67108864 \
  --wal-segment-events 100000 \
  --wal-max-retained-bytes 1073741824 \
  --embedding-model hashing \
  --embedding-dimension 256
```

The storage budget should include operational headroom beyond the configured value because index
generation construction and other repository artifacts use separate disk space.
