# HTTP Service

The Day 3 service exposes one recovered `DynamicHybridIndex` through a versioned API. It is a
single-process, single-writer service: run exactly one worker against a repository and WAL.

## Startup

The repository must already contain an initialized hybrid generation. Startup performs these steps
before readiness becomes true:

1. Construct the configured embedding model.
2. Validate `active.json`, `generation.json`, and both artifact manifests.
3. Load the BM25 and HNSW baselines at the same high-water sequence.
4. Validate and replay later WAL events into both deltas.
5. Verify that both logical document sets agree.

Start a repository created with a 256-dimensional hashing embedder:

```bash
.venv/bin/dhr serve \
  --repository .local/hybrid-repository \
  --event-log .local/wal \
  --segmented-wal \
  --embedding-model hashing \
  --embedding-dimension 256 \
  --host 127.0.0.1 \
  --port 8000
```

Do not use `--no-fsync` when durability matters. The service deliberately uses one Uvicorn worker;
multiple independent processes would each own a different in-memory delta.

`--segmented-wal` is recommended for production-style runs. Its event-log path is a directory, not
a JSONL file. The legacy single-file format remains available by omitting this option.

## Health

```bash
curl -sS http://127.0.0.1:8000/health/live | jq
curl -sS http://127.0.0.1:8000/health/ready | jq
```

Liveness reports whether the HTTP process is running. Readiness additionally requires a loaded,
healthy, sequence-consistent hybrid index. A process poisoned by a partial mutation returns 503.

## Apply an event

The caller supplies the globally ordered sequence number. Retries of the exact event are
idempotent; reuse of a sequence number for different content returns HTTP 409.

```bash
curl -sS -X POST http://127.0.0.1:8000/v1/events \
  -H 'Content-Type: application/json' \
  -d '{
    "sequence_number": 1,
    "operation": "upsert",
    "document_id": "doc-100",
    "version": 1,
    "event_time": "2026-11-14T12:00:00Z",
    "document": {
      "document_id": "doc-100",
      "title": "Distributed retrieval",
      "body": "Fresh hybrid indexing and recovery",
      "updated_at": "2026-11-14T12:00:00Z",
      "version": 1,
      "metadata": {"team": "search", "year": 2026}
    }
  }' | jq
```

## Search

`mode` is `lexical`, `dense`, or `hybrid`. Hybrid responses include the RRF contribution from each
retriever. Limits are restricted to 1-100 and query text to 4096 characters.

```bash
curl -sS -X POST http://127.0.0.1:8000/v1/search \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "fresh distributed search",
    "mode": "hybrid",
    "limit": 10,
    "filter": {
      "operator": "equals",
      "path": "team",
      "value": "search"
    }
  }' | jq
```

Supported filter operators are `equals`, `in`, `range`, `exists`, `all`, `any`, and `not`.

## Failure behavior

- 409 `sequence_conflict`: a sequence number conflicts with the durable WAL.
- 413 `request_too_large`: the request exceeds the configured body limit.
- 422 `invalid_request`: JSON or a domain invariant is invalid.
- 503 `index_unavailable`: recovery, repository, or hybrid consistency failure.
- 504 `request_timeout`: retrieval or mutation exceeded its deadline.

Every response carries `X-Request-ID`. A caller-provided ID up to 128 characters is preserved;
otherwise the service generates a UUID. Internal exceptions are logged and returned as a stable
500 envelope without exposing implementation details.
