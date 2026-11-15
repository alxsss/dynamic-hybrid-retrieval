"""Create the first complete hybrid repository generation from JSONL documents."""

from __future__ import annotations

from pathlib import Path

from dynamic_hybrid_retrieval.bm25 import BM25Index
from dynamic_hybrid_retrieval.dense import HNSWIndex
from dynamic_hybrid_retrieval.embeddings import HashingEmbedding, SentenceTransformerEmbedding
from dynamic_hybrid_retrieval.hybrid_repository import HybridActiveManifest, HybridIndexRepository
from dynamic_hybrid_retrieval.models import Document


def initialize_repository(
    documents_path: str | Path,
    repository_path: str | Path,
    *,
    embedding_model: str = "hashing",
    embedding_dimension: int = 256,
    embedding_device: str | None = None,
) -> HybridActiveManifest:
    source = Path(documents_path)
    documents = [
        Document.model_validate_json(line)
        for line in source.read_bytes().splitlines()
        if line.strip()
    ]
    if not documents:
        raise ValueError("documents file must contain at least one document")
    embedding = (
        HashingEmbedding(embedding_dimension)
        if embedding_model == "hashing"
        else SentenceTransformerEmbedding(embedding_model, device=embedding_device)
    )
    repository = HybridIndexRepository(repository_path)
    return repository.initialize(
        BM25Index.build(documents),
        HNSWIndex.build(documents, embedding),
    )
