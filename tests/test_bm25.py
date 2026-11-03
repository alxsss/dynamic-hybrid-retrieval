from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from dynamic_hybrid_retrieval.bm25 import (
    BM25Index,
    DuplicateDocumentError,
    IndexCorruptionError,
)
from dynamic_hybrid_retrieval.models import Document
from dynamic_hybrid_retrieval.tokenizer import Tokenizer, TokenizerConfig

NOW = datetime(2026, 10, 26, 12, 0, tzinfo=UTC)


def document(
    document_id: str,
    title: str,
    body: str,
    *,
    version: int = 1,
) -> Document:
    return Document(
        document_id=document_id,
        title=title,
        body=body,
        updated_at=NOW,
        version=version,
    )


@pytest.fixture
def index() -> BM25Index:
    return BM25Index.build(
        [
            document("doc-a", "Distributed retrieval", "search systems and indexing"),
            document("doc-b", "Search infrastructure", "reliable distributed systems"),
            document("doc-c", "Cooking", "a rare saffron recipe"),
        ]
    )


def test_tokenizer_is_deterministic_unicode_aware_and_casefolded() -> None:
    tokenizer = Tokenizer()

    assert tokenizer.tokenize("CAFÉ cafe\u0301 Full＿Width can't co-operate") == (
        "café",
        "café",
        "full",
        "width",
        "can't",
        "co-operate",
    )


def test_tokenizer_rejects_unknown_versions_and_normalizations() -> None:
    with pytest.raises(ValueError, match="version"):
        Tokenizer(TokenizerConfig(version=2))
    with pytest.raises(ValueError, match="normalization"):
        Tokenizer(TokenizerConfig(unicode_normalization="NFD"))


def test_exact_term_match_returns_expected_document(index: BM25Index) -> None:
    results = index.search("saffron")

    assert [result.document_id for result in results] == ["doc-c"]
    assert results[0].score > 0


def test_rare_term_has_higher_idf_than_common_term(index: BM25Index) -> None:
    assert index.inverse_document_frequency("saffron") > index.inverse_document_frequency(
        "distributed"
    )


def test_missing_and_empty_queries_return_no_results(index: BM25Index) -> None:
    assert index.search("not-in-the-corpus") == []
    assert index.search("---") == []


def test_multiple_query_terms_accumulate_scores(index: BM25Index) -> None:
    results = index.search("distributed retrieval")

    assert results[0].document_id == "doc-a"
    assert results[0].score > results[1].score


def test_repeated_query_terms_do_not_change_ranking_or_scores(index: BM25Index) -> None:
    once = index.search("distributed retrieval")
    repeated = index.search("distributed distributed retrieval")

    assert repeated == once


def test_ties_are_broken_by_document_id() -> None:
    index = BM25Index.build(
        [
            document("doc-b", "same", "text"),
            document("doc-a", "same", "text"),
        ]
    )

    assert [result.document_id for result in index.search("same")] == ["doc-a", "doc-b"]


def test_result_limit_is_validated(index: BM25Index) -> None:
    with pytest.raises(ValueError, match="at least one"):
        index.search("search", limit=0)


def test_empty_index_is_searchable() -> None:
    index = BM25Index.build([])

    assert index.document_count == 0
    assert index.average_document_length == 0
    assert index.search("anything") == []


def test_duplicate_document_ids_are_rejected() -> None:
    duplicate = document("doc-a", "one", "two")

    with pytest.raises(DuplicateDocumentError):
        BM25Index.build([duplicate, duplicate])


def test_invalid_bm25_parameters_are_rejected() -> None:
    with pytest.raises(ValueError, match="k1"):
        BM25Index.build([], k1=0)
    with pytest.raises(ValueError, match="b"):
        BM25Index.build([], b=1.1)


def test_save_and_load_preserve_documents_statistics_and_results(
    index: BM25Index, tmp_path: Path
) -> None:
    path = index.save(tmp_path / "generation-001")

    loaded = BM25Index.load(path)

    assert loaded.documents == index.documents
    assert loaded.document_lengths == index.document_lengths
    assert loaded.postings == index.postings
    assert loaded.document_count == index.document_count
    assert loaded.average_document_length == index.average_document_length
    assert loaded.search("distributed retrieval") == index.search("distributed retrieval")


def test_save_refuses_to_replace_existing_generation(index: BM25Index, tmp_path: Path) -> None:
    path = index.save(tmp_path / "generation-001")

    with pytest.raises(FileExistsError):
        index.save(path)


def test_load_detects_document_file_corruption(index: BM25Index, tmp_path: Path) -> None:
    path = index.save(tmp_path / "generation-001")
    documents_path = path / "documents.jsonl"
    documents_path.write_bytes(documents_path.read_bytes().replace(b"Cooking", b"Changed"))

    with pytest.raises(IndexCorruptionError, match="mismatch"):
        BM25Index.load(path)


def test_load_detects_manifest_corruption(index: BM25Index, tmp_path: Path) -> None:
    path = index.save(tmp_path / "generation-001")
    manifest_path = path / "manifest.json"
    manifest_path.write_bytes(manifest_path.read_bytes().replace(b"1.2", b"1.3"))

    with pytest.raises(IndexCorruptionError, match="manifest checksum"):
        BM25Index.load(path)
