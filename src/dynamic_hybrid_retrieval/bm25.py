"""Immutable BM25 baseline index with deterministic atomic persistence."""

from __future__ import annotations

import hashlib
import math
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

import orjson
from pydantic import ValidationError

from dynamic_hybrid_retrieval.models import Document
from dynamic_hybrid_retrieval.tokenizer import Tokenizer, TokenizerConfig

INDEX_FORMAT_VERSION: Final = 1
DOCUMENTS_FILE: Final = "documents.jsonl"
POSTINGS_FILE: Final = "postings.jsonl"
MANIFEST_FILE: Final = "manifest.json"
MANIFEST_CHECKSUM_FILE: Final = "manifest.sha256"


class IndexErrorBase(RuntimeError):
    """Base class for index build, load, and persistence errors."""


class DuplicateDocumentError(IndexErrorBase):
    """Raised when a snapshot contains the same document ID more than once."""


class IndexCorruptionError(IndexErrorBase):
    """Raised when a persisted generation fails structural or checksum validation."""


@dataclass(frozen=True, slots=True)
class Posting:
    document_id: str
    term_frequency: int


@dataclass(frozen=True, slots=True)
class SearchResult:
    document_id: str
    score: float
    version: int
    document: Document


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_bytes(path: Path, payload: bytes) -> None:
    with path.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _write_json_lines(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    with path.open("wb") as stream:
        for record in records:
            stream.write(orjson.dumps(record, option=orjson.OPT_SORT_KEYS | orjson.OPT_UTC_Z))
            stream.write(b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class BM25Index:
    """An immutable in-memory inverted index representing one baseline generation."""

    def __init__(
        self,
        *,
        documents: Mapping[str, Document],
        document_lengths: Mapping[str, int],
        postings: Mapping[str, tuple[Posting, ...]],
        tokenizer: Tokenizer,
        k1: float,
        b: float,
    ) -> None:
        if k1 <= 0:
            raise ValueError("k1 must be greater than zero")
        if not 0 <= b <= 1:
            raise ValueError("b must be between zero and one")
        if set(documents) != set(document_lengths):
            raise ValueError("documents and document lengths must contain the same IDs")

        self._documents = MappingProxyType(dict(documents))
        self._document_lengths = MappingProxyType(dict(document_lengths))
        self._postings = MappingProxyType(dict(postings))
        self.tokenizer = tokenizer
        self.k1 = float(k1)
        self.b = float(b)
        self.document_count = len(self._documents)
        total_length = sum(self._document_lengths.values())
        self.average_document_length = (
            total_length / self.document_count if self.document_count else 0.0
        )

    @classmethod
    def build(
        cls,
        documents: Iterable[Document],
        *,
        tokenizer: Tokenizer | None = None,
        k1: float = 1.2,
        b: float = 0.75,
    ) -> BM25Index:
        active_tokenizer = tokenizer or Tokenizer()
        document_map: dict[str, Document] = {}
        document_lengths: dict[str, int] = {}
        mutable_postings: defaultdict[str, list[Posting]] = defaultdict(list)

        for document in sorted(documents, key=lambda item: item.document_id):
            if document.document_id in document_map:
                raise DuplicateDocumentError(f"duplicate document ID {document.document_id!r}")
            document_map[document.document_id] = document
            tokens = active_tokenizer.tokenize(f"{document.title}\n{document.body}")
            document_lengths[document.document_id] = len(tokens)
            for term, frequency in sorted(Counter(tokens).items()):
                mutable_postings[term].append(
                    Posting(document_id=document.document_id, term_frequency=frequency)
                )

        postings = {
            term: tuple(term_postings) for term, term_postings in sorted(mutable_postings.items())
        }
        return cls(
            documents=document_map,
            document_lengths=document_lengths,
            postings=postings,
            tokenizer=active_tokenizer,
            k1=k1,
            b=b,
        )

    @property
    def documents(self) -> Mapping[str, Document]:
        return self._documents

    @property
    def document_lengths(self) -> Mapping[str, int]:
        return self._document_lengths

    @property
    def postings(self) -> Mapping[str, tuple[Posting, ...]]:
        return self._postings

    def inverse_document_frequency(self, term: str) -> float:
        document_frequency = len(self._postings.get(term, ()))
        if document_frequency == 0 or self.document_count == 0:
            return 0.0
        return math.log(
            1.0 + (self.document_count - document_frequency + 0.5) / (document_frequency + 0.5)
        )

    def search(self, query: str, *, limit: int = 10) -> list[SearchResult]:
        if limit < 1:
            raise ValueError("limit must be at least one")
        query_terms = tuple(dict.fromkeys(self.tokenizer.tokenize(query)))
        if not query_terms or not self.document_count:
            return []

        scores: defaultdict[str, float] = defaultdict(float)
        average_length = self.average_document_length or 1.0
        for term in query_terms:
            idf = self.inverse_document_frequency(term)
            for posting in self._postings.get(term, ()):
                document_length = self._document_lengths[posting.document_id]
                normalization = self.k1 * (1.0 - self.b + self.b * document_length / average_length)
                scores[posting.document_id] += idf * (
                    posting.term_frequency
                    * (self.k1 + 1.0)
                    / (posting.term_frequency + normalization)
                )

        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:limit]
        return [
            SearchResult(
                document_id=document_id,
                score=score,
                version=self._documents[document_id].version,
                document=self._documents[document_id],
            )
            for document_id, score in ranked
        ]

    def save(self, generation_path: str | Path) -> Path:
        """Persist a new immutable generation and refuse in-place replacement."""

        target = Path(generation_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise FileExistsError(f"index generation already exists: {target}")

        staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
        try:
            documents_path = staging / DOCUMENTS_FILE
            postings_path = staging / POSTINGS_FILE
            manifest_path = staging / MANIFEST_FILE

            _write_json_lines(
                documents_path,
                (
                    {
                        "document": self._documents[document_id].model_dump(mode="json"),
                        "length": self._document_lengths[document_id],
                    }
                    for document_id in sorted(self._documents)
                ),
            )
            _write_json_lines(
                postings_path,
                (
                    {
                        "term": term,
                        "postings": [
                            [posting.document_id, posting.term_frequency]
                            for posting in self._postings[term]
                        ],
                    }
                    for term in sorted(self._postings)
                ),
            )

            manifest = {
                "format_version": INDEX_FORMAT_VERSION,
                "tokenizer": self.tokenizer.config.as_dict(),
                "bm25": {"k1": self.k1, "b": self.b},
                "statistics": {
                    "document_count": self.document_count,
                    "term_count": len(self._postings),
                    "total_document_length": sum(self._document_lengths.values()),
                    "average_document_length": self.average_document_length,
                },
                "files": {
                    DOCUMENTS_FILE: {
                        "sha256": _sha256_file(documents_path),
                        "bytes": documents_path.stat().st_size,
                    },
                    POSTINGS_FILE: {
                        "sha256": _sha256_file(postings_path),
                        "bytes": postings_path.stat().st_size,
                    },
                },
            }
            manifest_bytes = orjson.dumps(
                manifest,
                option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS,
            )
            _write_bytes(manifest_path, manifest_bytes + b"\n")
            _write_bytes(
                staging / MANIFEST_CHECKSUM_FILE,
                hashlib.sha256(manifest_bytes + b"\n").hexdigest().encode() + b"\n",
            )
            _fsync_directory(staging)
            os.replace(staging, target)
            _fsync_directory(target.parent)
            return target
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise

    @classmethod
    def load(cls, generation_path: str | Path) -> BM25Index:
        """Load and fully validate an immutable generation."""

        source = Path(generation_path)
        manifest_path = source / MANIFEST_FILE
        checksum_path = source / MANIFEST_CHECKSUM_FILE
        try:
            manifest_bytes = manifest_path.read_bytes()
            expected_manifest_checksum = checksum_path.read_text().strip()
        except OSError as exc:
            raise IndexCorruptionError(f"cannot read index manifest from {source}") from exc

        actual_manifest_checksum = hashlib.sha256(manifest_bytes).hexdigest()
        if actual_manifest_checksum != expected_manifest_checksum:
            raise IndexCorruptionError("manifest checksum mismatch")
        try:
            manifest: Any = orjson.loads(manifest_bytes)
        except orjson.JSONDecodeError as exc:
            raise IndexCorruptionError("manifest is not valid JSON") from exc
        if not isinstance(manifest, dict) or manifest.get("format_version") != INDEX_FORMAT_VERSION:
            raise IndexCorruptionError("unsupported or missing index format version")

        files = manifest.get("files")
        if not isinstance(files, dict):
            raise IndexCorruptionError("manifest files section is invalid")
        for filename in (DOCUMENTS_FILE, POSTINGS_FILE):
            file_metadata = files.get(filename)
            path = source / filename
            if not isinstance(file_metadata, dict) or not path.is_file():
                raise IndexCorruptionError(f"missing metadata or file for {filename}")
            if path.stat().st_size != file_metadata.get("bytes"):
                raise IndexCorruptionError(f"size mismatch for {filename}")
            if _sha256_file(path) != file_metadata.get("sha256"):
                raise IndexCorruptionError(f"checksum mismatch for {filename}")

        try:
            tokenizer_data = manifest["tokenizer"]
            bm25_data = manifest["bm25"]
            tokenizer = Tokenizer(TokenizerConfig(**tokenizer_data))
            k1 = float(bm25_data["k1"])
            b = float(bm25_data["b"])
        except (KeyError, TypeError, ValueError) as exc:
            raise IndexCorruptionError("invalid tokenizer or BM25 configuration") from exc

        documents: dict[str, Document] = {}
        document_lengths: dict[str, int] = {}
        current_line = 0
        try:
            with (source / DOCUMENTS_FILE).open("rb") as stream:
                for line_number, line in enumerate(stream, start=1):
                    current_line = line_number
                    record = orjson.loads(line)
                    document = Document.model_validate_json(orjson.dumps(record["document"]))
                    length = record["length"]
                    if not isinstance(length, int) or isinstance(length, bool) or length < 0:
                        raise ValueError("invalid document length")
                    if document.document_id in documents:
                        raise ValueError("duplicate document ID")
                    documents[document.document_id] = document
                    document_lengths[document.document_id] = length
        except (
            OSError,
            orjson.JSONDecodeError,
            KeyError,
            TypeError,
            ValidationError,
            ValueError,
        ) as exc:
            raise IndexCorruptionError(f"invalid document record near line {current_line}") from exc

        postings: dict[str, tuple[Posting, ...]] = {}
        current_line = 0
        try:
            with (source / POSTINGS_FILE).open("rb") as stream:
                for line_number, line in enumerate(stream, start=1):
                    current_line = line_number
                    record = orjson.loads(line)
                    term = record["term"]
                    raw_postings = record["postings"]
                    if not isinstance(term, str) or not term or term in postings:
                        raise ValueError("invalid or duplicate term")
                    term_postings: list[Posting] = []
                    previous_document_id: str | None = None
                    for raw_posting in raw_postings:
                        document_id, frequency = raw_posting
                        if document_id not in documents:
                            raise ValueError("posting references unknown document")
                        if (
                            not isinstance(frequency, int)
                            or isinstance(frequency, bool)
                            or frequency < 1
                        ):
                            raise ValueError("invalid term frequency")
                        if previous_document_id is not None and document_id <= previous_document_id:
                            raise ValueError("postings are not strictly ordered")
                        term_postings.append(Posting(document_id, frequency))
                        previous_document_id = document_id
                    postings[term] = tuple(term_postings)
        except (OSError, orjson.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise IndexCorruptionError(f"invalid posting record near line {current_line}") from exc

        index = cls(
            documents=documents,
            document_lengths=document_lengths,
            postings=postings,
            tokenizer=tokenizer,
            k1=k1,
            b=b,
        )
        statistics = manifest.get("statistics", {})
        try:
            statistics_match = (
                statistics.get("document_count") == index.document_count
                and statistics.get("term_count") == len(index.postings)
                and statistics.get("total_document_length") == sum(index.document_lengths.values())
                and math.isclose(
                    float(statistics.get("average_document_length", -1)),
                    index.average_document_length,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
            )
        except (AttributeError, TypeError, ValueError) as exc:
            raise IndexCorruptionError("manifest statistics are invalid") from exc
        if not statistics_match:
            raise IndexCorruptionError("manifest statistics do not match index contents")
        return index
