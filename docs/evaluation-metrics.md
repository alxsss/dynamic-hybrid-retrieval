# Retrieval evaluation metrics

Quality and latency are reported separately. Each retriever produces a complete,
deterministic ranking for every query in the versioned evaluation set.

## Recall@K

Recall@K is the fraction of judged relevant documents present in the first K
results. Grades 1–3 are relevant by default. It measures coverage and does not
reward putting a relevant document earlier within the first K.

## MRR@K

Mean Reciprocal Rank averages `1 / rank` for the first relevant result within K.
It emphasizes how quickly a user sees any useful result. Queries with no relevant
result in the first K contribute zero.

## nDCG@K

Normalized Discounted Cumulative Gain preserves graded relevance. Gain is
`2^grade - 1`, discounted by `log2(rank + 1)`, and divided by the ideal ordering
for that query. This rewards placing grade-3 documents ahead of merely related
grade-1 documents.

## Latency

The runner performs configurable warm-up rounds, then records every measured
query call with a monotonic clock. It reports count, minimum, mean, P50, P95,
P99, and maximum. Quality rankings must remain identical across measured rounds;
an unstable run fails rather than averaging incomparable behavior.

## HNSW Recall@K

Approximate HNSW results are compared with exact cosine search over the same
vectors. Per-query recall is the intersection of approximate and exact top-K IDs
divided by the number of exact results. This measures ANN implementation quality,
not semantic relevance; both must be reported.

All aggregate relevance metrics are macro averages, so every query has equal
weight regardless of how many positive judgments it contains.
