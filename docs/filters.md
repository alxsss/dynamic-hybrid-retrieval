# Filtered retrieval

Filters are typed predicates over document metadata. Supported expressions are
equality, membership, numeric ranges, existence, conjunction, disjunction, and
negation. Dot-separated paths address nested metadata, such as `owner.team`.

Filtering happens inside BM25 and dense retrieval before reciprocal-rank fusion.
Filtering only the final fused top K would lose eligible candidates that ranked
just below in the source lists.

BM25 evaluates the predicate while visiting postings. Its IDF and length
normalization remain global-corpus statistics; the project does not recompute
BM25 statistics for every filtered subset.

HNSW cannot guarantee that the first K physical neighbors satisfy a filter. It
starts with an expanded candidate window and doubles the window until K eligible
documents are found or the index is exhausted. Exact search therefore returns K
eligible nearest neighbors when K exist. Approximate search preserves the same
eligibility behavior but remains subject to normal HNSW recall limitations.

Metadata filters are not authorization. Security-sensitive access control must
be enforced by a trusted mandatory layer and tested against every serving path;
caller-supplied retrieval filters alone are not an ACL boundary.
