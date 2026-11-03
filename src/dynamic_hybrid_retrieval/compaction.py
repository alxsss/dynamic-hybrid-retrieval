"""Generation repository, atomic compaction, activation, and rollback."""

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

import orjson
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from dynamic_hybrid_retrieval.bm25 import BM25Index, SearchResult
from dynamic_hybrid_retrieval.event_log import EventLog
from dynamic_hybrid_retrieval.index import CompactionSnapshot, DynamicIndex

REPOSITORY_FORMAT_VERSION: Final = 1
ACTIVE_MANIFEST_FILE: Final = "active.json"
GENERATIONS_DIRECTORY: Final = "generations"
GENERATION_PATTERN: Final = re.compile(r"^generation-(\d{6})$")


class RepositoryError(RuntimeError):
    """Base class for repository and compaction failures."""


class RepositoryCorruptionError(RepositoryError):
    """Raised when repository control metadata cannot be trusted."""


class CompactionValidationError(RepositoryError):
    """Raised when a new baseline differs from its logical source snapshot."""


class NoRollbackGenerationError(RepositoryError):
    """Raised when rollback is requested without a previous generation."""


class ActiveManifest(BaseModel):
    """Atomically replaced pointer to current and previous baseline generations."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    format_version: Annotated[int, Field(ge=1)] = REPOSITORY_FORMAT_VERSION
    active_generation: str
    active_sequence_number: Annotated[int, Field(ge=0)]
    previous_generation: str | None = None
    previous_sequence_number: Annotated[int, Field(ge=0)] | None = None
    activated_at: datetime

    @model_validator(mode="after")
    def validate_manifest(self) -> ActiveManifest:
        if self.format_version != REPOSITORY_FORMAT_VERSION:
            raise ValueError(f"unsupported repository format {self.format_version}")
        if GENERATION_PATTERN.fullmatch(self.active_generation) is None:
            raise ValueError("active generation name is invalid")
        if (self.previous_generation is None) != (self.previous_sequence_number is None):
            raise ValueError("previous generation and sequence must be present together")
        if (
            self.previous_generation is not None
            and GENERATION_PATTERN.fullmatch(self.previous_generation) is None
        ):
            raise ValueError("previous generation name is invalid")
        if self.activated_at.tzinfo is None or self.activated_at.utcoffset() is None:
            raise ValueError("activated_at must include a timezone")
        return self


class ManifestEnvelope(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    manifest: ActiveManifest
    checksum: str


class CompactionResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    manifest: ActiveManifest
    generation_path: Path
    compacted_document_count: int
    validated_query_count: int


def _canonical_manifest_bytes(manifest: ActiveManifest) -> bytes:
    return orjson.dumps(
        manifest.model_dump(mode="json"),
        option=orjson.OPT_SORT_KEYS | orjson.OPT_UTC_Z,
    )


def _manifest_checksum(manifest: ActiveManifest) -> str:
    return hashlib.sha256(_canonical_manifest_bytes(manifest)).hexdigest()


class IndexRepository:
    """Own immutable baseline generations and one atomic active pointer."""

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

    def _write_manifest(self, manifest: ActiveManifest) -> None:
        envelope = {
            "manifest": manifest.model_dump(mode="json"),
            "checksum": _manifest_checksum(manifest),
        }
        payload = (
            orjson.dumps(
                envelope,
                option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS | orjson.OPT_UTC_Z,
            )
            + b"\n"
        )
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".active-",
            suffix=".tmp",
            dir=self.root,
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, self.manifest_path)
            directory_descriptor = os.open(self.root, os.O_RDONLY)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise

    def read_manifest(self) -> ActiveManifest:
        try:
            envelope = ManifestEnvelope.model_validate_json(self.manifest_path.read_bytes())
        except (OSError, ValidationError) as exc:
            raise RepositoryCorruptionError("active manifest is missing or invalid") from exc
        if envelope.checksum != _manifest_checksum(envelope.manifest):
            raise RepositoryCorruptionError("active manifest checksum mismatch")
        active_path = self.generations_path / envelope.manifest.active_generation
        if not active_path.is_dir():
            raise RepositoryCorruptionError("active generation directory is missing")
        if envelope.manifest.previous_generation is not None:
            previous_path = self.generations_path / envelope.manifest.previous_generation
            if not previous_path.is_dir():
                raise RepositoryCorruptionError("previous generation directory is missing")
        return envelope.manifest

    def initialize(
        self,
        baseline: BM25Index,
        *,
        baseline_sequence_number: int = 0,
    ) -> ActiveManifest:
        """Create the first generation and active manifest in an empty repository."""

        if baseline_sequence_number < 0:
            raise ValueError("baseline_sequence_number cannot be negative")
        with self._thread_lock, self._locked_file() as lock_stream:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
            try:
                if self.manifest_path.exists():
                    raise FileExistsError("repository is already initialized")
                generation_name = self._next_generation_name()
                generation_path = self.generations_path / generation_name
                baseline.save(generation_path)
                manifest = ActiveManifest(
                    active_generation=generation_name,
                    active_sequence_number=baseline_sequence_number,
                    activated_at=datetime.now(UTC),
                )
                try:
                    self._write_manifest(manifest)
                except BaseException:
                    shutil.rmtree(generation_path, ignore_errors=True)
                    raise
                return manifest
            finally:
                fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)

    def load_active(self, event_log: EventLog) -> DynamicIndex:
        manifest = self.read_manifest()
        baseline = BM25Index.load(self.generations_path / manifest.active_generation)
        return DynamicIndex(
            baseline,
            event_log,
            baseline_sequence_number=manifest.active_sequence_number,
        )

    def _validate_compaction(
        self,
        snapshot: CompactionSnapshot,
        loaded_baseline: BM25Index,
        *,
        query_limit: int,
    ) -> None:
        if loaded_baseline.document_count != len(snapshot.documents):
            raise CompactionValidationError("compacted document count differs from snapshot")
        expected_documents = {document.document_id: document for document in snapshot.documents}
        if dict(loaded_baseline.documents) != expected_documents:
            raise CompactionValidationError("compacted documents differ from snapshot")
        for query, expected_results in snapshot.query_results:
            actual_results = loaded_baseline.search(query, limit=query_limit)
            self._assert_results_equivalent(query, expected_results, actual_results)

    @staticmethod
    def _assert_results_equivalent(
        query: str,
        expected: tuple[SearchResult, ...],
        actual: list[SearchResult],
    ) -> None:
        if len(expected) != len(actual):
            raise CompactionValidationError(f"result count differs for query {query!r}")
        for position, (expected_result, actual_result) in enumerate(
            zip(expected, actual, strict=True)
        ):
            if (
                expected_result.document_id != actual_result.document_id
                or expected_result.version != actual_result.version
                or expected_result.document != actual_result.document
                or not math.isclose(
                    expected_result.score,
                    actual_result.score,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
            ):
                raise CompactionValidationError(
                    f"result differs for query {query!r} at position {position}"
                )

    def compact(
        self,
        dynamic_index: DynamicIndex,
        *,
        validation_queries: tuple[str, ...] = (),
        query_limit: int = 10,
    ) -> CompactionResult:
        """Build, validate, and atomically activate a new baseline generation."""

        with self._thread_lock, self._locked_file() as lock_stream:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
            generation_path: Path | None = None
            activated = False
            try:
                current = self.read_manifest()
                if dynamic_index.baseline_sequence_number != current.active_sequence_number:
                    raise RepositoryError("dynamic index does not match active baseline sequence")

                snapshot = dynamic_index.snapshot_for_compaction(
                    validation_queries,
                    query_limit=query_limit,
                )
                generation_name = self._next_generation_name()
                generation_path = self.generations_path / generation_name
                compacted = BM25Index.build(
                    snapshot.documents,
                    tokenizer=dynamic_index.baseline.tokenizer,
                    k1=dynamic_index.baseline.k1,
                    b=dynamic_index.baseline.b,
                )
                compacted.save(generation_path)
                loaded = BM25Index.load(generation_path)
                self._validate_compaction(
                    snapshot,
                    loaded,
                    query_limit=query_limit,
                )

                manifest = ActiveManifest(
                    active_generation=generation_name,
                    active_sequence_number=snapshot.high_water_sequence,
                    previous_generation=current.active_generation,
                    previous_sequence_number=current.active_sequence_number,
                    activated_at=datetime.now(UTC),
                )
                self._write_manifest(manifest)
                activated = True
                return CompactionResult(
                    manifest=manifest,
                    generation_path=generation_path,
                    compacted_document_count=len(snapshot.documents),
                    validated_query_count=len(snapshot.query_results),
                )
            finally:
                if generation_path is not None and not activated:
                    shutil.rmtree(generation_path, ignore_errors=True)
                fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)

    def rollback(self) -> ActiveManifest:
        """Atomically reactivate the previous baseline generation."""

        with self._thread_lock, self._locked_file() as lock_stream:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
            try:
                current = self.read_manifest()
                if current.previous_generation is None or current.previous_sequence_number is None:
                    raise NoRollbackGenerationError("no previous generation is available")
                manifest = ActiveManifest(
                    active_generation=current.previous_generation,
                    active_sequence_number=current.previous_sequence_number,
                    previous_generation=current.active_generation,
                    previous_sequence_number=current.active_sequence_number,
                    activated_at=datetime.now(UTC),
                )
                self._write_manifest(manifest)
                return manifest
            finally:
                fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)
