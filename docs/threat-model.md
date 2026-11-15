# Threat Model

## Assets and trust boundaries

The protected assets are indexed documents, embeddings, WAL records, generation artifacts, and
service availability. HTTP clients, document metadata, JSONL snapshots, embedding providers, and
filesystem contents are untrusted inputs. The local filesystem and single writer are operational
trust assumptions; this project does not provide distributed consensus.

## Principal risks and controls

- Unauthorized reads or writes: bind to loopback by default and place authentication, TLS, and
  authorization at a reverse proxy. The service itself does not implement identity.
- Resource exhaustion: request bodies, query lengths, result limits, operation timeouts, WAL bytes,
  and segment sizes are bounded. Rate limiting belongs at the proxy.
- Corrupt or malicious persistence: records and manifests are checksummed, formats are versioned,
  paths are repository-controlled, and corruption fails closed.
- Replay or sequence reuse: global sequence numbers are monotonic; exact retries are idempotent and
  conflicting reuse is rejected.
- Deleted-data exposure: tombstones mask baseline copies immediately and compaction removes them
  physically. Backups and old generations require a separate retention policy.
- Dependency and model risk: lockfiles pin resolved dependencies; embedding model identity and
  dimension are persisted. Remote model code must not be enabled without review.
- Information leakage: stable error envelopes hide unexpected exception details, but logs and query
  text may still contain sensitive data and require access controls and retention limits.

## Explicit non-goals

The reference service is not multi-tenant, highly available, encrypted at rest, or safe to expose
directly to the public internet. Production deployment requires authentication, TLS, secret
management, backups, monitoring, and tested restore procedures.
