"""Persistent HNSW dense-vector index backed by USearch."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import orjson
from usearch.index import Index

from dynamic_hybrid_retrieval.embeddings import EmbeddingModel
from dynamic_hybrid_retrieval.filters import DocumentFilter
from dynamic_hybrid_retrieval.models import Document

DENSE_FORMAT_VERSION: Final = 1
DENSE_INDEX_FILE: Final = "vectors.usearch"
DENSE_DOCUMENTS_FILE: Final = "documents.jsonl"
DENSE_MANIFEST_FILE: Final = "manifest.json"


class DenseIndexError(RuntimeError):
    """Base class for dense-index failures."""


class DenseIndexCorruptionError(DenseIndexError):
    """Raised when persisted dense-index artifacts cannot be trusted."""


class EmbeddingCompatibilityError(DenseIndexError):
    """Raised when the query embedder differs from the indexed embedder."""


@dataclass(frozen=True, slots=True)
class HNSWConfig:
    connectivity: int = 16
    expansion_add: int = 128
    expansion_search: int = 64

    def __post_init__(self) -> None:
        if self.connectivity < 2:
            raise ValueError("connectivity must be at least two")
        if self.expansion_add < 1 or self.expansion_search < 1:
            raise ValueError("HNSW expansion values must be positive")


@dataclass(frozen=True, slots=True)
class DenseSearchResult:
    document_id: str
    score: float
    distance: float
    version: int
    document: Document


def document_text(document: Document) -> str:
    """Return the versioned text representation embedded at index time."""

    return f"{document.title}\n{document.body}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_file(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class HNSWIndex:
    """Immutable document-level cosine HNSW index."""

    def __init__(
        self,
        *,
        index: Index,
        documents: tuple[Document, ...],
        embedding_model: EmbeddingModel,
        config: HNSWConfig,
    ) -> None:
        if index.ndim != embedding_model.dimension:
            raise EmbeddingCompatibilityError("HNSW and embedding dimensions differ")
        if len(index) != len(documents):
            raise DenseIndexCorruptionError("HNSW vector and document counts differ")
        self._index = index
        self._documents = documents
        self.embedding_model = embedding_model
        self.config = config

    @classmethod
    def build(
        cls,
        documents: Iterable[Document],
        embedding_model: EmbeddingModel,
        *,
        config: HNSWConfig | None = None,
    ) -> HNSWIndex:
        active_config = config or HNSWConfig()
        ordered = tuple(sorted(documents, key=lambda item: item.document_id))
        if len({document.document_id for document in ordered}) != len(ordered):
            raise ValueError("document IDs must be unique")
        vectors = embedding_model.embed_documents([document_text(item) for item in ordered])
        if vectors.shape != (len(ordered), embedding_model.dimension):
            raise ValueError("embedding model returned an invalid document matrix")
        index = Index(
            ndim=embedding_model.dimension,
            metric="cos",
            dtype="f32",
            connectivity=active_config.connectivity,
            expansion_add=active_config.expansion_add,
            expansion_search=active_config.expansion_search,
        )
        if ordered:
            index.add(np.arange(len(ordered), dtype=np.uint64), vectors)
        return cls(
            index=index,
            documents=ordered,
            embedding_model=embedding_model,
            config=active_config,
        )

    @property
    def document_count(self) -> int:
        return len(self._documents)

    @property
    def documents(self) -> tuple[Document, ...]:
        return self._documents

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        exact: bool = False,
        document_filter: DocumentFilter | None = None,
    ) -> list[DenseSearchResult]:
        if limit < 1:
            raise ValueError("limit must be positive")
        if not self._documents:
            return []
        query_vector = self.embedding_model.embed_query(query)
        if query_vector.shape != (1, self.embedding_model.dimension):
            raise ValueError("embedding model returned an invalid query matrix")
        candidate_count = min(
            len(self._documents),
            limit if document_filter is None else max(limit * 2, 16),
        )
        while True:
            matches = self._index.search(
                query_vector[0],
                count=candidate_count,
                exact=exact,
            )
            results = [
                DenseSearchResult(
                    document_id=self._documents[int(key)].document_id,
                    score=1.0 - float(distance),
                    distance=float(distance),
                    version=self._documents[int(key)].version,
                    document=self._documents[int(key)],
                )
                for key, distance in zip(matches.keys, matches.distances, strict=True)
                if document_filter is None or document_filter.matches(self._documents[int(key)])
            ]
            if len(results) >= limit or candidate_count == len(self._documents):
                return sorted(results, key=lambda item: (-item.score, item.document_id))[:limit]
            candidate_count = min(len(self._documents), candidate_count * 2)

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise FileExistsError(f"dense generation already exists: {target}")
        staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
        try:
            index_path = staging / DENSE_INDEX_FILE
            documents_path = staging / DENSE_DOCUMENTS_FILE
            self._index.save(index_path)
            _fsync_file(index_path)
            with documents_path.open("wb") as stream:
                for key, document in enumerate(self._documents):
                    stream.write(
                        orjson.dumps(
                            {"key": key, "document": document.model_dump(mode="json")},
                            option=orjson.OPT_SORT_KEYS | orjson.OPT_UTC_Z,
                        )
                        + b"\n"
                    )
                stream.flush()
                os.fsync(stream.fileno())
            manifest = {
                "format_version": DENSE_FORMAT_VERSION,
                "embedding": {
                    "model_id": self.embedding_model.model_id,
                    "dimension": self.embedding_model.dimension,
                    "normalized": True,
                },
                "hnsw": {
                    "metric": "cos",
                    "dtype": "f32",
                    "connectivity": self.config.connectivity,
                    "expansion_add": self.config.expansion_add,
                    "expansion_search": self.config.expansion_search,
                },
                "document_count": len(self._documents),
                "files": {
                    DENSE_INDEX_FILE: {
                        "bytes": index_path.stat().st_size,
                        "sha256": _sha256(index_path),
                    },
                    DENSE_DOCUMENTS_FILE: {
                        "bytes": documents_path.stat().st_size,
                        "sha256": _sha256(documents_path),
                    },
                },
            }
            manifest_path = staging / DENSE_MANIFEST_FILE
            manifest_path.write_bytes(
                orjson.dumps(manifest, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS) + b"\n"
            )
            _fsync_file(manifest_path)
            _fsync_directory(staging)
            os.replace(staging, target)
            _fsync_directory(target.parent)
            return target
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise

    @classmethod
    def load(cls, path: str | Path, embedding_model: EmbeddingModel) -> HNSWIndex:
        source = Path(path)
        try:
            manifest: Any = orjson.loads((source / DENSE_MANIFEST_FILE).read_bytes())
        except (OSError, orjson.JSONDecodeError) as exc:
            raise DenseIndexCorruptionError("dense manifest is missing or invalid") from exc
        if not isinstance(manifest, dict) or manifest.get("format_version") != DENSE_FORMAT_VERSION:
            raise DenseIndexCorruptionError("unsupported dense format version")
        try:
            embedding = manifest["embedding"]
            if (
                embedding["model_id"] != embedding_model.model_id
                or embedding["dimension"] != embedding_model.dimension
            ):
                raise EmbeddingCompatibilityError("persisted embedding model is incompatible")
            files = manifest["files"]
            for filename in (DENSE_INDEX_FILE, DENSE_DOCUMENTS_FILE):
                artifact = source / filename
                metadata = files[filename]
                if (
                    artifact.stat().st_size != metadata["bytes"]
                    or _sha256(artifact) != metadata["sha256"]
                ):
                    raise DenseIndexCorruptionError(f"checksum mismatch for {filename}")
            hnsw = manifest["hnsw"]
            config = HNSWConfig(
                connectivity=hnsw["connectivity"],
                expansion_add=hnsw["expansion_add"],
                expansion_search=hnsw["expansion_search"],
            )
            documents: list[Document] = []
            with (source / DENSE_DOCUMENTS_FILE).open("rb") as stream:
                for expected_key, line in enumerate(stream):
                    record = orjson.loads(line)
                    if record["key"] != expected_key:
                        raise DenseIndexCorruptionError("dense document keys are not contiguous")
                    documents.append(Document.model_validate_json(orjson.dumps(record["document"])))
            if len(documents) != manifest["document_count"]:
                raise DenseIndexCorruptionError("dense document count differs from manifest")
            restored = Index.restore(source / DENSE_INDEX_FILE)
            if restored is None:
                raise DenseIndexCorruptionError("HNSW graph could not be restored")
            index = restored
        except EmbeddingCompatibilityError:
            raise
        except DenseIndexCorruptionError:
            raise
        except (KeyError, OSError, TypeError, ValueError, orjson.JSONDecodeError) as exc:
            raise DenseIndexCorruptionError("dense generation failed validation") from exc
        return cls(
            index=index,
            documents=tuple(documents),
            embedding_model=embedding_model,
            config=config,
        )
