# Reciprocal Rank Fusion

BM25 and dense cosine scores are not directly comparable. A BM25 score may be
`12.4` while cosine similarity is `0.83`; neither number establishes how much
more relevant one candidate is across retrievers. Reciprocal Rank Fusion uses
only each retriever's ordering:

```text
RRF(document) = Σ weight(source) / (k + rank(source, document))
```

Ranks are one-based and a missing document contributes zero. With `k = 60`:

```text
Lexical: doc-a rank 1, doc-b rank 2
Dense:   doc-b rank 1, doc-c rank 2, doc-a rank 3

doc-a = 1/61 + 1/63
doc-b = 1/62 + 1/61
doc-c =        1/62
```

The implementation retains source rank, raw source score, source weight, and
individual contribution for every result. Raw scores are explanation metadata;
they never enter fusion arithmetic. Equal fused scores use ascending document ID
as the deterministic tie-breaker.

The hybrid coordinator retrieves a configurable candidate pool from both paths,
checks that both indexes are at the same WAL sequence, performs RRF, and returns
the requested top K. It fails closed if the sequence changes during retrieval or
if retrievers return different versions of the same document.

`candidate_limit` controls the quality/latency tradeoff. RRF cannot promote a
document that neither retriever includes in its candidate pool. The default is
100 candidates per source and must never be smaller than the final result limit.

Weighted RRF is supported for measured tuning:

```text
score = lexical_weight / (k + lexical_rank)
      + dense_weight   / (k + dense_rank)
```

Weights and `k` must be selected on a separate development judgment set and
reported with evaluation results. They must not be tuned on the final test set.
