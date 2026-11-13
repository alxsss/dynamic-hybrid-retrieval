# Week 2 RFC: Dense and hybrid retrieval

**Status:** In progress  
**Scope:** Embeddings, HNSW, reciprocal-rank fusion, metadata filters, and evaluation

## Goals

Week 2 adds semantic retrieval beside the existing BM25 path and measures three
systems over the same hand-verified judgments: lexical BM25, dense HNSW, and RRF
hybrid retrieval. Dynamic dense updates will follow the same baseline-plus-delta
and version/tombstone semantics rather than create a second source of truth.

## Architecture

```text
Query ──┬──► BM25 baseline + delta ──────────────► lexical ranking ──┐
        │                                                           │
        └──► embedding model ─► HNSW baseline + delta ─► dense rank ├──► RRF ─► filters ─► top K
                                                                    │
Document event ─► WAL ─► lexical update + embedding/vector update ──┘
```

The embedding boundary records model identity and vector dimension. Documents
and queries must use the same model and normalization policy. Dense generations
are immutable, checksummed artifacts containing the HNSW graph, document/key
mapping, model identity, dimensionality, metric, and construction parameters.

## Day plan

1. Define embedding contracts and persist a cosine HNSW baseline.
2. Add dynamic dense inserts, updates, deletes, replay, and compaction semantics. **Complete.**
3. Implement RRF with deterministic tie-breaking and explainable component ranks. **Complete.**
4. Add typed metadata filters with pre-filter/post-filter correctness tests.
5. Create the first hand-verified query/relevance judgment set.
6. Implement Recall@K, MRR, nDCG@K, latency, and HNSW recall measurements.
7. Produce reproducible BM25-versus-dense-versus-hybrid results and analysis.

## Initial decisions

- Sentence Transformers is the production evaluation adapter; its exact model ID
  is stored with every dense generation.
- A deterministic hashing embedder exists only for offline unit tests and must
  not be used to claim semantic retrieval quality.
- USearch provides the HNSW implementation; cosine similarity is `1 - distance`.
- Dense result ordering uses descending similarity and ascending document ID for
  deterministic ties.
- Evaluation data is versioned, human-readable, and reviewed separately from the
  retrieval implementations.

## Correctness requirements

- Loading with a different model identity or dimension fails closed.
- Documents and HNSW keys remain a one-to-one contiguous mapping.
- Persisted artifacts are size- and checksum-validated before serving.
- Deletes cannot leak through either lexical or dense candidates.
- RRF never combines raw BM25 and cosine scores directly.
- Filtered search must return `K` eligible results when that many exist; simple
  post-filtering of only `K` approximate candidates is not sufficient.
- Exact vector search is retained as the oracle for measuring HNSW recall.
