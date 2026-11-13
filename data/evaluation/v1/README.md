# Evaluation set v1

This is a small, hand-reviewed retrieval set for pipeline validation and the
first BM25-versus-dense-versus-hybrid comparison. It is not a statistically
representative public benchmark.

All 16 documents were reviewed for each of the 10 queries. `qrels.jsonl` stores
only positive judgments; omitted query/document pairs are considered relevance
grade zero. Positive grades mean:

- `3`: directly satisfies the query intent;
- `2`: substantially useful but incomplete or narrower;
- `1`: related context with limited direct utility.

Every positive judgment includes a rationale. Queries intentionally cover exact
identifiers, semantic paraphrases, hybrid intent, infrastructure operations,
and structured filtering. Changes require a new dataset version or an explicit
review of affected queries, rationales, and manifest checksums.
