# Release and Public Repository Checklist

## Required checks

```bash
./scripts/release_check.sh
```

Run the 100,000-document acceptance benchmark and concurrent load test separately because they are
intentionally excluded from the fast CI job. Preserve their JSON reports with the release notes.

## Container

Initialize persistent data on the host, then build and run as the unprivileged `dhr` user:

```bash
docker build -t dynamic-hybrid-retrieval:0.1.0 .
docker run --rm -p 8000:8000 -v "$PWD/.local:/data" dynamic-hybrid-retrieval:0.1.0
```

The mounted directory must contain `/data/repository`; the container creates `/data/wal` on first
startup. Never bake document indexes, WAL data, credentials, or model caches into the image.

## Before making the repository public

1. Review `git status`, ignored files, history, author email, and remote URL.
2. Search tracked files and history for credentials and private document content.
3. Choose and add a license; no license is selected automatically by this project.
4. Attach measured reports with hardware, model, seed, and OpenSearch version metadata.
5. Verify README commands in a clean clone.
6. Enable branch protection and required CI checks.

## Configuration reference

The CLI is authoritative: `dhr initialize --help` and `dhr serve --help`. Service environment
equivalents use the `DHR_` prefix, including `DHR_REPOSITORY`, `DHR_EVENT_LOG`,
`DHR_EMBEDDING_MODEL`, `DHR_EMBEDDING_DIMENSION`, `DHR_SEGMENTED_WAL`,
`DHR_WAL_SEGMENT_BYTES`, `DHR_WAL_SEGMENT_EVENTS`, `DHR_WAL_MAX_RETAINED_BYTES`,
`DHR_REQUEST_TIMEOUT`, and `DHR_MAX_REQUEST_BYTES`.
