# Day 6 benchmark and acceptance plan

## Purpose

The benchmark turns the RFC targets into executable checks. It is a local,
repeatable engineering benchmark, not a claim about distributed production
capacity. OpenSearch comparison remains intentionally out of scope until the
custom implementation is complete.

## Default workload

- 100,000 deterministic synthetic baseline documents.
- 10,000 ordered change events containing updates, inserts, and deletes.
- 200 deterministic query samples before updates and another 200 after updates.
- A durable `fsync` for every accepted event.
- A fixed random seed so runs are comparable.

## Measurements

The JSON report records baseline build time; apply latency; update-to-search
visibility latency; P50, P95, and P99 query latency before and after changes;
replay time; compaction time; and the final corpus size and sequence number.

The run fails unless all of these correctness gates hold:

1. Every update is visible within one second.
2. No deleted document appears in sampled results.
3. Replaying the durable event log produces equivalent statistics and results.
4. Compaction produces the same ordered result IDs, versions, and scores.

## Reproducibility and interpretation

Use a new output directory for each run. The directory contains immutable index
generations, `events.jsonl`, and `report.json`, so a failed run can be inspected.
Report the machine model, operating system, Python version, git commit, command,
and whether `fsync` was enabled with any published result.

The default run is a development-scale acceptance test. Before using numbers as
an SLO or capacity estimate, run multiple warm and cold trials, include concurrent
readers and writers, and measure memory, CPU, disk utilization, and tail latency
under sustained load.
