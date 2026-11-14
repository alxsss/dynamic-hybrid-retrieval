"""Atomic repository for coordinated BM25 and HNSW generations."""

from __future__ import annotations

import fcntl
import hashlib
import math
import os
import re
import shutil
import tempfile
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Final

import numpy as np
import orjson
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from dynamic_hybrid_retrieval.bm25 import MANIFEST_FILE, BM25Index, SearchResult
from dynamic_hybrid_retrieval.dense import (
    DENSE_MANIFEST_FILE,
    DenseSearchResult,
    HNSWIndex,
)
from dynamic_hybrid_retrieval.embeddings import EmbeddingModel
from dynamic_hybrid_retrieval.event_log import EventLog
from dynamic_hybrid_retrieval.hybrid_index import DynamicHybridIndex, HybridCompactionSnapshot

HYBRID_REPOSITORY_FORMAT_VERSION: Final = 1
ACTIVE_MANIFEST_FILE: Final = "active.json"
GENERATION_MANIFEST_FILE: Final = "generation.json"
GENERATIONS_DIRECTORY: Final = "generations"
LEXICAL_DIRECTORY: Final = "lexical"
DENSE_DIRECTORY: Final = "dense"
GENERATION_PATTERN: Final = re.compile(r"^generation-(\d{6})$")


class HybridRepositoryError(RuntimeError):
    """Base class for hybrid repository failures."""


class HybridRepositoryCorruptionError(HybridRepositoryError):
    """Raised when repository metadata or artifacts cannot be trusted."""


class HybridCompactionValidationError(HybridRepositoryError):
    """Raised when a candidate generation differs from its source snapshot."""


class NoHybridRollbackGenerationError(HybridRepositoryError):
    """Raised when no previous hybrid generation is available."""


class HybridActiveManifest(BaseModel):
    """Atomic pointer to current and previous complete hybrid generations."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    format_version: Annotated[int, Field(ge=1)] = HYBRID_REPOSITORY_FORMAT_VERSION
    active_generation: str
    active_sequence_number: Annotated[int, Field(ge=0)]
    previous_generation: str | None = None
    previous_sequence_number: Annotated[int, Field(ge=0)] | None = None
    activated_at: datetime

    @model_validator(mode="after")
    def validate_manifest(self) -> HybridActiveManifest:
        if self.format_version != HYBRID_REPOSITORY_FORMAT_VERSION:
            raise ValueError(f"unsupported hybrid repository format {self.format_version}")
        _validate_generation_name(self.active_generation)
        if (self.previous_generation is None) != (self.previous_sequence_number is None):
            raise ValueError("previous generation and sequence must be present together")
        if self.previous_generation is not None:
            _validate_generation_name(self.previous_generation)
        if self.activated_at.tzinfo is None or self.activated_at.utcoffset() is None:
            raise ValueError("activated_at must include a timezone")
        return self


class HybridGenerationManifest(BaseModel):
    """Identity and cross-artifact checksums for one immutable generation."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    format_version: Annotated[int, Field(ge=1)] = HYBRID_REPOSITORY_FORMAT_VERSION
    generation: str
    high_water_sequence: Annotated[int, Field(ge=0)]
    document_count: Annotated[int, Field(ge=0)]
    embedding_model_id: str
    embedding_dimension: Annotated[int, Field(gt=0)]
    lexical_manifest_sha256: str
    dense_manifest_sha256: str
    created_at: datetime

    @model_validator(mode="after")
    def validate_manifest(self) -> HybridGenerationManifest:
        if self.format_version != HYBRID_REPOSITORY_FORMAT_VERSION:
            raise ValueError(f"unsupported hybrid generation format {self.format_version}")
        _validate_generation_name(self.generation)
        if not self.embedding_model_id.strip():
            raise ValueError("embedding_model_id cannot be empty")
        for checksum in (self.lexical_manifest_sha256, self.dense_manifest_sha256):
            if len(checksum) != 64 or any(
                character not in "0123456789abcdef" for character in checksum
            ):
                raise ValueError("artifact manifest checksum is invalid")
        if self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
            raise ValueError("created_at must include a timezone")
        return self


class _ActiveEnvelope(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    manifest: HybridActiveManifest
    checksum: str


class _GenerationEnvelope(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    manifest: HybridGenerationManifest
    checksum: str


class HybridCompactionResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    manifest: HybridActiveManifest
    generation_path: Path
    compacted_document_count: int
    validated_query_count: int


def _validate_generation_name(name: str) -> None:
    if GENERATION_PATTERN.fullmatch(name) is None:
        raise ValueError("generation name is invalid")


def _canonical_model_bytes(model: BaseModel) -> bytes:
    return orjson.dumps(
        model.model_dump(mode="json"),
        option=orjson.OPT_SORT_KEYS | orjson.OPT_UTC_Z,
    )


def _model_checksum(model: BaseModel) -> str:
    return hashlib.sha256(_canonical_model_bytes(model)).hexdigest()


def _file_checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class HybridIndexRepository:
    """Build, validate, activate, load, and roll back complete hybrid generations."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.generations_path = self.root / GENERATIONS_DIRECTORY
        self.manifest_path = self.root / ACTIVE_MANIFEST_FILE
        self.lock_path = self.root / "compaction.lock"
        self._thread_lock = threading.RLock()
        self.root.mkdir(parents=True, exist_ok=True)
        self.generations_path.mkdir(parents=True, exist_ok=True)

    def _locked_file(self) -> Any:
        return self.lock_path.open("a+b")

    def _next_generation_name(self) -> str:
        highest = 0
        for path in self.generations_path.iterdir():
            match = GENERATION_PATTERN.fullmatch(path.name)
            if match is not None:
                highest = max(highest, int(match.group(1)))
        return f"generation-{highest + 1:06d}"

    def _write_envelope(self, path: Path, manifest: BaseModel) -> None:
        payload = (
            orjson.dumps(
                {
                    "manifest": manifest.model_dump(mode="json"),
                    "checksum": _model_checksum(manifest),
                },
                option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS | orjson.OPT_UTC_Z,
            )
            + b"\n"
        )
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}-",
            suffix=".tmp",
            dir=path.parent,
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, path)
            _fsync_directory(path.parent)
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise

    def _write_active_manifest(self, manifest: HybridActiveManifest) -> None:
        self._write_envelope(self.manifest_path, manifest)

    def _read_generation_manifest(self, generation_name: str) -> HybridGenerationManifest:
        _validate_generation_name(generation_name)
        generation_path = self.generations_path / generation_name
        try:
            envelope = _GenerationEnvelope.model_validate_json(
                (generation_path / GENERATION_MANIFEST_FILE).read_bytes()
            )
        except (OSError, ValidationError) as exc:
            raise HybridRepositoryCorruptionError(
                f"generation manifest for {generation_name} is missing or invalid"
            ) from exc
        if envelope.checksum != _model_checksum(envelope.manifest):
            raise HybridRepositoryCorruptionError("generation manifest checksum mismatch")
        manifest = envelope.manifest
        if manifest.generation != generation_name:
            raise HybridRepositoryCorruptionError("generation manifest name mismatch")
        lexical_manifest = generation_path / LEXICAL_DIRECTORY / MANIFEST_FILE
        dense_manifest = generation_path / DENSE_DIRECTORY / DENSE_MANIFEST_FILE
        try:
            lexical_checksum = _file_checksum(lexical_manifest)
            dense_checksum = _file_checksum(dense_manifest)
        except OSError as exc:
            raise HybridRepositoryCorruptionError("artifact manifest is missing") from exc
        if lexical_checksum != manifest.lexical_manifest_sha256:
            raise HybridRepositoryCorruptionError("lexical manifest checksum mismatch")
        if dense_checksum != manifest.dense_manifest_sha256:
            raise HybridRepositoryCorruptionError("dense manifest checksum mismatch")
        return manifest

    def read_manifest(self) -> HybridActiveManifest:
        try:
            envelope = _ActiveEnvelope.model_validate_json(self.manifest_path.read_bytes())
        except (OSError, ValidationError) as exc:
            raise HybridRepositoryCorruptionError("active manifest is missing or invalid") from exc
        if envelope.checksum != _model_checksum(envelope.manifest):
            raise HybridRepositoryCorruptionError("active manifest checksum mismatch")
        manifest = envelope.manifest
        active_generation = self._read_generation_manifest(manifest.active_generation)
        if active_generation.high_water_sequence != manifest.active_sequence_number:
            raise HybridRepositoryCorruptionError("active sequence differs from generation")
        if manifest.previous_generation is not None:
            previous_generation = self._read_generation_manifest(manifest.previous_generation)
            if previous_generation.high_water_sequence != manifest.previous_sequence_number:
                raise HybridRepositoryCorruptionError("previous sequence differs from generation")
        return manifest

    def _materialize_generation(
        self,
        generation_name: str,
        lexical: BM25Index,
        dense: HNSWIndex,
        *,
        high_water_sequence: int,
    ) -> Path:
        if dict(lexical.documents) != {
            document.document_id: document for document in dense.documents
        }:
            raise HybridRepositoryError("lexical and dense generations contain different documents")
        staging_path = Path(
            tempfile.mkdtemp(prefix=f".{generation_name}.staging-", dir=self.generations_path)
        )
        final_path = self.generations_path / generation_name
        try:
            lexical.save(staging_path / LEXICAL_DIRECTORY)
            dense.save(staging_path / DENSE_DIRECTORY)
            generation_manifest = HybridGenerationManifest(
                generation=generation_name,
                high_water_sequence=high_water_sequence,
                document_count=lexical.document_count,
                embedding_model_id=dense.embedding_model.model_id,
                embedding_dimension=dense.embedding_model.dimension,
                lexical_manifest_sha256=_file_checksum(
                    staging_path / LEXICAL_DIRECTORY / MANIFEST_FILE
                ),
                dense_manifest_sha256=_file_checksum(
                    staging_path / DENSE_DIRECTORY / DENSE_MANIFEST_FILE
                ),
                created_at=datetime.now(UTC),
            )
            self._write_envelope(
                staging_path / GENERATION_MANIFEST_FILE,
                generation_manifest,
            )
            _fsync_directory(staging_path)
            os.replace(staging_path, final_path)
            _fsync_directory(self.generations_path)
            return final_path
        except BaseException:
            shutil.rmtree(staging_path, ignore_errors=True)
            raise

    def initialize(
        self,
        lexical: BM25Index,
        dense: HNSWIndex,
        *,
        baseline_sequence_number: int = 0,
    ) -> HybridActiveManifest:
        """Persist and activate the first complete hybrid generation."""

        if baseline_sequence_number < 0:
            raise ValueError("baseline_sequence_number cannot be negative")
        with self._thread_lock, self._locked_file() as lock_stream:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
            generation_path: Path | None = None
            try:
                if self.manifest_path.exists():
                    raise FileExistsError("hybrid repository is already initialized")
                generation_name = self._next_generation_name()
                generation_path = self._materialize_generation(
                    generation_name,
                    lexical,
                    dense,
                    high_water_sequence=baseline_sequence_number,
                )
                manifest = HybridActiveManifest(
                    active_generation=generation_name,
                    active_sequence_number=baseline_sequence_number,
                    activated_at=datetime.now(UTC),
                )
                try:
                    self._write_active_manifest(manifest)
                except BaseException:
                    shutil.rmtree(generation_path, ignore_errors=True)
                    raise
                return manifest
            finally:
                fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)

    def load_active(
        self,
        event_log: EventLog,
        embedding_model: EmbeddingModel,
    ) -> DynamicHybridIndex:
        """Load both active artifacts and replay WAL events after their high-water mark."""

        manifest = self.read_manifest()
        generation = self._read_generation_manifest(manifest.active_generation)
        if (
            generation.embedding_model_id != embedding_model.model_id
            or generation.embedding_dimension != embedding_model.dimension
        ):
            raise HybridRepositoryError("embedding model is incompatible with active generation")
        generation_path = self.generations_path / manifest.active_generation
        lexical = BM25Index.load(generation_path / LEXICAL_DIRECTORY)
        dense = HNSWIndex.load(generation_path / DENSE_DIRECTORY, embedding_model)
        from dynamic_hybrid_retrieval.dynamic_dense import DynamicHNSWIndex
        from dynamic_hybrid_retrieval.index import DynamicIndex

        return DynamicHybridIndex(
            DynamicIndex(
                lexical,
                event_log,
                baseline_sequence_number=manifest.active_sequence_number,
            ),
            DynamicHNSWIndex(
                dense,
                event_log,
                baseline_sequence_number=manifest.active_sequence_number,
            ),
        )

    def _validate_candidate(
        self,
        snapshot: HybridCompactionSnapshot,
        lexical: BM25Index,
        dense: HNSWIndex,
        *,
        query_limit: int,
    ) -> None:
        expected_documents = {
            document.document_id: document for document in snapshot.lexical.documents
        }
        if dict(lexical.documents) != expected_documents:
            raise HybridCompactionValidationError("lexical documents differ from snapshot")
        if {document.document_id: document for document in dense.documents} != expected_documents:
            raise HybridCompactionValidationError("dense documents differ from snapshot")
        for query, expected_lexical in snapshot.lexical.query_results:
            actual_lexical = lexical.search(query, limit=query_limit)
            _assert_lexical_results_equivalent(query, expected_lexical, actual_lexical)
        for query, expected_dense in snapshot.dense.exact_query_results:
            actual_dense = dense.search(query, limit=query_limit, exact=True)
            _assert_dense_results_equivalent(query, expected_dense, actual_dense)

    def compact(
        self,
        hybrid: DynamicHybridIndex,
        *,
        validation_queries: tuple[str, ...] = (),
        query_limit: int = 10,
    ) -> HybridCompactionResult:
        """Build, validate, and atomically activate both retrieval artifacts."""

        with self._thread_lock, self._locked_file() as lock_stream:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
            generation_path: Path | None = None
            activated = False
            try:
                current = self.read_manifest()
                if (
                    hybrid.lexical.baseline_sequence_number != current.active_sequence_number
                    or hybrid.dense.baseline_sequence_number != current.active_sequence_number
                ):
                    raise HybridRepositoryError(
                        "hybrid index does not match active baseline sequence"
                    )
                snapshot = hybrid.snapshot_for_compaction(
                    validation_queries,
                    query_limit=query_limit,
                )
                lexical = BM25Index.build(
                    snapshot.lexical.documents,
                    tokenizer=hybrid.lexical.baseline.tokenizer,
                    k1=hybrid.lexical.baseline.k1,
                    b=hybrid.lexical.baseline.b,
                )
                dense = HNSWIndex.build(
                    snapshot.dense.documents,
                    hybrid.dense.baseline.embedding_model,
                    config=hybrid.dense.baseline.config,
                )
                generation_name = self._next_generation_name()
                generation_path = self._materialize_generation(
                    generation_name,
                    lexical,
                    dense,
                    high_water_sequence=snapshot.high_water_sequence,
                )
                loaded_lexical = BM25Index.load(generation_path / LEXICAL_DIRECTORY)
                loaded_dense = HNSWIndex.load(
                    generation_path / DENSE_DIRECTORY,
                    hybrid.dense.baseline.embedding_model,
                )
                self._validate_candidate(
                    snapshot,
                    loaded_lexical,
                    loaded_dense,
                    query_limit=query_limit,
                )
                manifest = HybridActiveManifest(
                    active_generation=generation_name,
                    active_sequence_number=snapshot.high_water_sequence,
                    previous_generation=current.active_generation,
                    previous_sequence_number=current.active_sequence_number,
                    activated_at=datetime.now(UTC),
                )
                self._write_active_manifest(manifest)
                activated = True
                return HybridCompactionResult(
                    manifest=manifest,
                    generation_path=generation_path,
                    compacted_document_count=len(snapshot.lexical.documents),
                    validated_query_count=len(snapshot.lexical.query_results),
                )
            finally:
                if generation_path is not None and not activated:
                    shutil.rmtree(generation_path, ignore_errors=True)
                fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)

    def rollback(self) -> HybridActiveManifest:
        """Atomically reactivate the previous complete hybrid generation."""

        with self._thread_lock, self._locked_file() as lock_stream:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
            try:
                current = self.read_manifest()
                if current.previous_generation is None or current.previous_sequence_number is None:
                    raise NoHybridRollbackGenerationError(
                        "no previous hybrid generation is available"
                    )
                manifest = HybridActiveManifest(
                    active_generation=current.previous_generation,
                    active_sequence_number=current.previous_sequence_number,
                    previous_generation=current.active_generation,
                    previous_sequence_number=current.active_sequence_number,
                    activated_at=datetime.now(UTC),
                )
                self._write_active_manifest(manifest)
                return manifest
            finally:
                fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)


def _assert_lexical_results_equivalent(
    query: str,
    expected: tuple[SearchResult, ...],
    actual: list[SearchResult],
) -> None:
    if len(expected) != len(actual):
        raise HybridCompactionValidationError(f"lexical result count differs for query {query!r}")
    for left, right in zip(expected, actual, strict=True):
        if (
            left.document_id != right.document_id
            or left.version != right.version
            or left.document != right.document
            or not math.isclose(left.score, right.score, rel_tol=1e-12, abs_tol=1e-12)
        ):
            raise HybridCompactionValidationError(f"lexical result differs for query {query!r}")


def _assert_dense_results_equivalent(
    query: str,
    expected: tuple[DenseSearchResult, ...],
    actual: list[DenseSearchResult],
) -> None:
    if len(expected) != len(actual):
        raise HybridCompactionValidationError(f"dense result count differs for query {query!r}")
    for left, right in zip(expected, actual, strict=True):
        if (
            left.document_id != right.document_id
            or left.version != right.version
            or left.document != right.document
            or not np.isclose(left.score, right.score, rtol=1e-6, atol=1e-6)
        ):
            raise HybridCompactionValidationError(f"dense result differs for query {query!r}")
