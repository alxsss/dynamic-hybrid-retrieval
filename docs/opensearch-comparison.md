# OpenSearch Comparison

The adapter creates a one-shard, zero-replica test index containing the same title, body, metadata,
version, and embedding used by the custom engine. It supports BM25, HNSW k-NN, updates, deletes,
and client-side RRF. Client-side RRF is deliberate: it keeps the fusion formula identical rather
than comparing it with a different score-normalization policy.

Start Docker Desktop, then verify OpenSearch:

```bash
docker ps
curl -k -u admin:admin https://localhost:9200
```

OpenSearch bulk writes use `refresh=wait_for`, which measures acknowledged update-to-search
visibility without forcing an immediate refresh. The adapter checks the Bulk API's top-level
`errors` field because HTTP 200 can still contain individual item failures.

## Measured local run

OpenSearch 3.6.0, one shard, zero replicas, 16 hand-verified documents, 10 queries, hashing-blake2b
256-dimensional embeddings, five measured rounds, cutoff 5:

| Mode | Recall@5 | MRR@5 | nDCG@5 | p50 ms | p95 ms | p99 ms |
|---|---:|---:|---:|---:|---:|---:|
| BM25 | 0.6833 | 0.7167 | 0.6569 | 5.80 | 7.63 | 9.57 |
| Dense | 0.6167 | 0.5833 | 0.5802 | 5.49 | 7.11 | 7.53 |
| Hybrid RRF | 0.7000 | 0.7450 | 0.7115 | 11.94 | 14.97 | 15.80 |

The full machine-readable result is `.local/opensearch-comparison/report.json`. These local numbers
are useful for reproducibility, not capacity planning. They use the deterministic hashing embedder;
the Week 2 semantic-quality report uses MiniLM and must not be compared as if the models were equal.

Relevant implementation: `src/dynamic_hybrid_retrieval/opensearch.py`.
