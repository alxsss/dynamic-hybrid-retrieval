# Week 2 retrieval comparison

Run the final comparison with a cached Sentence Transformers model:

```bash
export MPLCONFIGDIR=/tmp/dhr-matplotlib
export HF_HOME="$PWD/.local/huggingface"

.venv/bin/dhr evaluate \
  --dataset data/evaluation/v1 \
  --output .local/week2-results \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --k 5 \
  --candidates 16 \
  --warmups 2 \
  --rounds 10 \
  --device cpu
```

The output directory contains `report.json` with environment, configuration,
rankings, per-query metrics, aggregate metrics, latency distributions, build
times, and HNSW recall, plus `summary.md` with the comparison table.

The dataset has only ten queries. Results validate the experimental pipeline and
illustrate retrieval differences, but they are not statistically sufficient for
broad claims. Model loading and index construction are reported separately from
query latency. Dense latency includes query embedding; hybrid latency includes
BM25 retrieval, query embedding, HNSW search, and RRF.

## Observed local result

The reference run completed offline on November 13, 2026 using CPython 3.13 on
an Apple Silicon Mac, `all-MiniLM-L6-v2`, K=5, 16 candidates per RRF source, two
warm-up rounds, and ten measured rounds.

| Retriever | Recall@5 | MRR@5 | nDCG@5 | P50 ms | P95 ms | P99 ms |
|---|---:|---:|---:|---:|---:|---:|
| BM25 | 0.6833 | 0.7950 | 0.7233 | 0.012 | 0.018 | 0.034 |
| Dense HNSW | 0.8667 | 0.9500 | 0.8323 | 6.411 | 7.843 | 8.842 |
| Hybrid RRF | 0.8167 | 0.9500 | 0.8118 | 6.494 | 8.011 | 8.871 |

Approximate HNSW Recall@5 against exact cosine search was 1.0000. With only 16
vectors, this confirms implementation agreement but does not predict recall at
production scale. Model loading took 9.857 seconds, BM25 construction took 0.001
seconds, and embedding plus HNSW construction took 0.171 seconds.

## Interpretation

Dense retrieval produced the best aggregate quality on this deliberately
paraphrase-heavy set. Hybrid RRF improved substantially over BM25, but did not
beat dense. Equal-weight RRF promoted documents supported by both retrievers even
when the dense-only ordering was better. Query `eval-q05` remained difficult for
all three systems and is a useful candidate for judgment review, corpus expansion,
or query analysis.

The exact-identifier query `ERR_CONN_RESET_104` ranked its target first in all
three systems, so this tiny set does not yet demonstrate a lexical-only win. A
larger development set should be created before tuning RRF weights or candidate
depth. The current ten queries must not be reused both to tune those values and
to claim final test performance.

BM25 was hundreds of times faster in this small local run because it performs no
neural query encoding. Dense and hybrid P50 latency was dominated by MiniLM query
embedding; RRF itself added little over dense latency. These are single-process,
warm local measurements, not service SLOs.
