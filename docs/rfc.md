# Dynamic Baseline and Delta Retrieval Index

**Status:** Draft  
**Target milestone:** Week 1 local prototype  
**Scope:** Lexical retrieval, dynamic updates, compaction, recovery, and measurement  

## Summary

This RFC proposes a local retrieval system that combines an immutable baseline index with a small mutable delta index. The design supports searchable inserts and updates, deletion tombstones, deterministic event-log recovery, and atomic compaction. It is intended to establish the dynamic-indexing foundation for later dense, hybrid, and agentic retrieval experiments.

The Week 1 prototype targets 100,000 synthetic documents and 10,000 change events on a single development machine. Updates must become searchable within one second, deleted documents must never be returned, P95 query latency must be measured before and after applying updates, and compaction must preserve the exact ordered query results of the uncompacted logical index.

## 1. Problem and target users

### Problem

A static search index provides predictable query behavior, but rebuilding it for every document change is too expensive and prevents fresh content from becoming searchable quickly. A fully mutable index improves freshness but makes durability, recovery, compaction, and reproducible experiments harder to reason about.

The project needs a dynamic indexing layer that separates frequent changes from the stable corpus while presenting one consistent logical index to callers. The system must make correctness visible: stale versions must not reappear, deleted documents must not leak into results, recovery must produce the same state, and compaction must not change retrieval behavior.

### Target users

- **Retrieval researchers** comparing indexing, ranking, and fusion strategies over a reproducible corpus.
- **Search infrastructure engineers** evaluating freshness, latency, recovery, and compaction tradeoffs.
- **Agent-system developers** who will later consume the index through a stable search interface.
- **Reviewers and interviewers** assessing the engineering decisions, measurements, and failure handling behind the prototype.

### Non-goals for Week 1

- Distributed sharding, replication, or consensus.
- Dense-vector indexing or hybrid lexical-vector fusion.
- Kafka or another external event broker.
- Authentication, authorization, or document-level access control.
- Production-grade multi-process concurrency.
- Natural-language generation or LLM integration.

## 2. Functional requirements

### Document and event model

Each document has:

- a stable `document_id`;
- searchable `title` and `body` fields;
- optional metadata;
- a monotonically increasing document `version`; and
- an `updated_at` timestamp used for diagnostics, not event ordering.

Each change event has:

- a globally increasing `sequence_number`;
- an operation of `upsert` or `delete`;
- a `document_id`;
- a document payload for an upsert;
- a document version; and
- an event timestamp.

### Required behavior

| ID | Requirement |
|---|---|
| FR-1 | Build and persist a searchable baseline index from a document snapshot. |
| FR-2 | Append every accepted change to a durable event log before acknowledging it. |
| FR-3 | Make an accepted insert or update searchable within one second. |
| FR-4 | Prevent a deleted document from appearing in any query result after the delete is acknowledged. |
| FR-5 | Ignore duplicate events and document versions older than the latest accepted version. |
| FR-6 | Query the baseline and delta as one logical corpus with deterministic ranking. |
| FR-7 | Recover the same logical state by loading a checkpoint and replaying the event log. |
| FR-8 | Compact the baseline and delta without changing the exact ordered results for the benchmark query set. |
| FR-9 | Retain the previous valid baseline generation until a new generation is verified. |
| FR-10 | Report build throughput, replay throughput, freshness, query latency, index size, and compaction time. |

## 3. Scale assumptions

The Week 1 prototype assumes:

- 100,000 synthetic baseline documents;
- 10,000 change events containing inserts, updates, and deletes;
- one process and one writer;
- a local filesystem with atomic rename support;
- a corpus that fits on local disk and whose live metadata fits in memory;
- short text documents suitable for a Python BM25 implementation; and
- a bounded benchmark query set with known relevant document identifiers.

The design should not prevent future partitioning, but distributed operation is deliberately excluded. At this scale, correctness and measurable behavior matter more than maximizing throughput.

Synthetic documents and events must be deterministic for a fixed seed. All reported results must identify the workload as synthetic and include the machine, Python version, corpus size, event distribution, query count, and run configuration.

## 4. Freshness and latency targets

### Service targets

| Measure | Week 1 target |
|---|---|
| Update freshness | Accepted inserts and updates searchable within 1 second |
| Delete visibility | Deleted documents absent immediately after acknowledgement |
| Query latency | Report P50, P95, and P99; P95 reported separately before and after updates |
| Event recovery | Replaying the same valid log produces identical logical state and results |
| Compaction correctness | Exact ordered results match before and after compaction for the benchmark query set |

Freshness is measured from the event's local acceptance time to the first query that can observe the new version. Query latency is measured end to end around the public search API after a warm-up phase. The benchmark must retain raw samples or a machine-readable summary so percentiles can be reproduced.

Week 1 does not set a fixed P95 latency ceiling because the first run establishes the local baseline. The RFC requires the measurement and comparison; a later RFC will turn observed performance into an SLO.

## 5. Baseline-plus-delta architecture

```text
                         append before acknowledge
Change producer ───────────────► Event log
       │                              │
       │                              ▼
       │                        Event applier
       │                              │
       │                              ▼
       │                    Delta postings and state
       │
Query caller ──► Query coordinator ──┬──► Baseline postings
                                    ├──► Delta postings
                                    ├──► Version and tombstone filter
                                    └──► Corpus-wide BM25 scorer ──► Top K

Compactor: baseline snapshot + live delta ──► staged baseline ──► atomic manifest swap
```

### Baseline

The baseline is an immutable generation containing tokenized documents, postings, term frequencies, and document lengths. Immutable generations simplify concurrent reads, reproducibility, rollback, and detection of partial writes.

### Delta

The delta holds documents changed after the active baseline generation. It contains postings for inserted and updated document versions plus an in-memory or persisted latest-version map and deletion tombstones. An updated document is logically owned by the delta even if an older copy remains physically present in the baseline.

### Logical corpus statistics

Ranking uses one set of corpus-wide live statistics rather than independent BM25 statistics for each tier. The system tracks:

- number of live documents;
- document frequency by term;
- total live document length; and
- average live document length.

These statistics are updated when an event changes the logical document state. Both baseline and delta candidates are scored with the same statistics. This avoids incomparable tier-local BM25 scores and makes ranking independent of whether a live document currently resides in the baseline or delta.

### Manifest

An active manifest identifies:

- the active baseline generation;
- the baseline's maximum incorporated event sequence;
- the current delta generation;
- schema and tokenizer versions; and
- checksums for persistent artifacts.

The manifest is replaced atomically only after a staged generation passes validation.

## 6. Insert, update, and delete semantics

### Ordering and idempotency

`sequence_number` defines log order. Document `version` determines whether a change supersedes the current state for that document. Wall-clock timestamps are diagnostic and never decide correctness.

An event is handled as follows:

1. Validate its schema and operation-specific fields.
2. Compare its version with the latest accepted version for the document.
3. Ignore the event if the same version and operation have already been applied.
4. Reject or record as stale an event whose version is lower than the latest version.
5. Append and flush the accepted event to the event log.
6. Apply the event to the delta state and corpus-wide statistics.
7. Acknowledge the event only after the new logical state is queryable.

### Insert

An upsert for an unknown `document_id` creates a live delta document, adds its terms to delta postings, and increments corpus-wide live-document and term statistics.

### Update

An upsert for an existing document removes the previous live version's contribution from logical corpus statistics and adds the new version. If the previous version exists in the baseline, the version map masks that baseline copy. Only the latest accepted version can be returned.

### Delete

A delete removes the current live version's contribution from logical statistics and records a tombstone containing the deleted version. Query filtering applies tombstones before ranking results are returned. A later upsert may recreate the document only with a strictly higher version.

### Acknowledgement guarantee

After an operation is acknowledged:

- an inserted or updated version is visible to subsequent queries;
- the previous version is not visible; and
- a deleted document cannot be returned.

The prototype provides single-process read-after-write consistency, not distributed linearizability.

## 7. Query merging

The query coordinator performs these steps:

1. Tokenize the query with the tokenizer version recorded in the manifest.
2. Read matching postings from the baseline and delta.
3. Remove baseline candidates superseded by a delta version.
4. Remove all tombstoned candidates.
5. Deduplicate candidates by `document_id`, retaining only the latest version.
6. Score every remaining candidate with the same corpus-wide BM25 statistics.
7. Sort by descending score, then ascending `document_id` as a deterministic tie-breaker.
8. Return the first `k` results with score, version, and source-generation metadata.

Because scoring uses logical corpus-wide statistics, moving a document from delta into the baseline during compaction does not alter its score. Deterministic tie-breaking allows exact ordered-result comparisons across replay and compaction.

The benchmark must include queries that match:

- baseline documents only;
- delta documents only;
- updated documents;
- deleted documents;
- common and rare terms; and
- documents with tied scores.

## 8. Compaction

Compaction converts the active logical state into a new immutable baseline generation without blocking accepted updates for the entire build.

### Procedure

1. Record a compaction high-water sequence number `S`.
2. Materialize the latest live version of every document as of `S`.
3. Build a new baseline in a unique staging directory.
4. Write its postings, document data, corpus statistics, manifest candidate, and checksums.
5. Validate document counts, checksums, invariants, and benchmark query results.
6. Preserve events with sequence numbers greater than `S` as the new delta.
7. Atomically replace the active manifest with one referencing the new baseline and delta.
8. Retain the previous manifest and baseline generation for rollback.

### Equivalence requirement

For every benchmark query, the ordered list of `(document_id, version, score)` values before and after compaction must match. Floating-point scores should either be serialized deterministically or compared with a documented tight tolerance while requiring identical ordering.

If validation fails, the active manifest remains unchanged and the staged generation is marked invalid or removed later by safe cleanup.

## 9. Failure recovery

### Startup recovery

On startup the system:

1. Reads the active manifest.
2. Validates referenced artifacts and checksums.
3. Loads the active baseline generation.
4. Replays valid event-log records after the baseline high-water sequence.
5. Reconstructs the delta, latest-version map, tombstones, and logical corpus statistics.
6. Verifies state invariants before accepting queries.

Replay must be deterministic and idempotent. Replaying a valid log twice must not change the final state.

### Partial event records

The event log is newline-delimited and append-only. A malformed record in the middle of the log is a hard recovery error. A truncated final record, consistent with a crash during append, is ignored and reported. The system never silently skips corruption between valid records.

### Crash during event application

Because an event is appended before it is acknowledged, a crash after log persistence but before in-memory application is repaired by replay. The client might retry an unacknowledged event, so event handling must be idempotent.

### Crash during compaction

A crash before the manifest swap leaves the old generation active. A crash after the atomic swap loads the new generation. Staging directories are never considered active solely because they exist.

### Invalid active generation

If the active generation fails checksum or invariant validation, recovery attempts the immediately previous valid manifest and reports that rollback. If neither generation is valid, startup fails rather than serving uncertain results.

## 10. Observability

The prototype emits structured metrics and logs suitable for benchmark output.

### Counters

- events received, accepted, duplicated, rejected, and stale;
- inserts, updates, and deletes applied;
- live documents and tombstones;
- queries executed and query errors;
- baseline and delta candidates examined;
- compactions attempted, succeeded, and failed;
- recovery and rollback operations; and
- deleted-result leakage violations, which must remain zero.

### Histograms and timings

- event append latency;
- update-to-queryable freshness;
- query latency before updates;
- query latency after updates;
- event replay throughput;
- baseline build throughput;
- compaction duration; and
- startup recovery duration.

### Gauges and artifact measurements

- baseline and delta document counts;
- event-log length and last applied sequence;
- index sizes on disk;
- process memory usage; and
- active generation identifiers.

Each benchmark result includes its configuration, random seed, corpus size, event mix, query count, timestamp, and environment information. Metrics are written to a machine-readable JSON artifact and summarized in the README.

## 11. Alternatives considered

### Rebuild the entire index after every change

This provides a simple query path but cannot meet the one-second freshness target at the target corpus size. It also performs work proportional to the corpus for each small change.

### Periodic full rebuild without a queryable delta

Batch rebuilds are operationally simple, but changes remain invisible until the next rebuild and deletes can leak during that interval. This design is retained as a benchmark baseline, not the serving architecture.

### One fully mutable index

A single mutable index avoids query merging but couples writes, reads, persistence, and recovery. It makes reproducible snapshots and atomic rollback harder and does not demonstrate the migration pattern this project intends to study.

### Tier-local BM25 followed by score normalization

Independent BM25 scores depend on tier-specific document frequencies and average lengths. Min-max normalization is sensitive to the retrieved candidate set and can change ranking after compaction. Rank fusion is useful for heterogeneous retrievers but is unnecessary for two physical tiers representing one lexical corpus. Corpus-wide live statistics provide clearer equivalence semantics.

### Reciprocal-rank fusion between baseline and delta

RRF avoids score calibration, but a small delta and large baseline receive equal rank-based influence. It can also change results when compaction moves documents between tiers. RRF will be evaluated later for lexical-dense hybrid retrieval, where the component rankers are genuinely different.

### LSM-style multiple delta levels

Multiple levels reduce large compaction costs and are a natural extension for higher scale. The Week 1 prototype uses one baseline and one delta to keep correctness and measurements understandable. The manifest and generation model should permit additional levels later.

### Elasticsearch or OpenSearch

An external engine would provide mature near-real-time indexing, segments, replication, and operational tooling. It would hide many of the mechanics this project is intended to demonstrate. A later benchmark may compare this implementation with an external engine.

## 12. Rollout and rollback

Although the prototype runs locally, it models a staged production rollout.

### Rollout

1. Build a baseline from the deterministic corpus snapshot.
2. Run offline validation and benchmark queries against the static index.
3. Enable event logging while continuing to serve baseline-only results.
4. Replay events into the delta in shadow mode and compare results.
5. Enable baseline-plus-delta reads for a canary query set.
6. Verify freshness, deletion safety, latency, and ranking invariants.
7. Make the dynamic path the default.
8. Run compaction, validate equivalence, and atomically activate the new generation.

### Rollback

Rollback atomically restores the previous manifest. Events after that manifest's high-water sequence are replayed into a fresh delta, so rollback does not discard accepted changes. A generation is not deleted until a later successful compaction and an explicit retention check confirm that it is no longer required.

Rollback triggers include:

- any deleted document returned by a query;
- invariant or checksum failure;
- non-equivalent compaction results;
- unrecoverable replay divergence; or
- a material P95 latency regression defined after the initial benchmark.

## 13. Open questions

1. What event mix best approximates realistic insert, update, and delete traffic for later benchmarks?
2. Should accepted event-log writes require `fsync` for every event or use bounded group commit?
3. How large may the delta become before compaction is triggered: document count, log bytes, latency degradation, or a combination?
4. Should corpus-wide document frequencies be updated exactly for every event or maintained in immutable-plus-delta components at larger scale?
5. What numeric precision and tolerance should define score equivalence across serialization boundaries?
6. How should tokenizer and schema migrations coexist with live events during a future generation rebuild?
7. Should a later version support concurrent writers, and what ordering authority would assign global sequence numbers?
8. When dense retrieval is added, should vector updates share the lexical event log and manifest or use coordinated but separate generations?
9. How should filtered retrieval alter corpus statistics and benchmark expectations?
10. Which observed Week 1 latency and freshness results should become formal SLOs for subsequent milestones?

## Acceptance criteria

Week 1 is complete when all of the following are demonstrated by automated tests or reproducible benchmark output:

- A baseline containing 100,000 synthetic documents builds successfully.
- A deterministic workload of 10,000 change events replays successfully.
- Accepted inserts and updates become searchable within one second.
- No acknowledged delete appears in results.
- Duplicate and stale events do not change the logical state.
- Recovery from the same baseline and event log produces the same state and ordered results.
- P50, P95, and P99 query latency are reported, with P95 separated into pre-update and post-update measurements.
- Compaction produces results equivalent to the uncompacted index for the benchmark query set.
- A failed compaction leaves the previous generation active and queryable.
- The benchmark records its seed, configuration, environment, and synthetic-data label.
