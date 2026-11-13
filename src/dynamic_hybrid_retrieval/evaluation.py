"""Versioned, validated retrieval evaluation datasets."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError

from dynamic_hybrid_retrieval.models import Document

NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class EvaluationSetError(RuntimeError):
    """Raised when evaluation artifacts are missing, corrupt, or inconsistent."""


class EvaluationQuery(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    query_id: NonEmptyString
    text: NonEmptyString
    intent: NonEmptyString
    tags: tuple[NonEmptyString, ...] = ()


class RelevanceJudgment(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    query_id: NonEmptyString
    document_id: NonEmptyString
    relevance: Annotated[int, Field(ge=1, le=3)]
    rationale: NonEmptyString


class ArtifactMetadata(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    records: Annotated[int, Field(ge=0)]
    bytes: Annotated[int, Field(ge=0)]
    sha256: NonEmptyString


class EvaluationManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    format_version: Annotated[int, Field(ge=1)]
    dataset_id: NonEmptyString
    description: NonEmptyString
    annotation_policy: NonEmptyString
    unjudged_policy: NonEmptyString
    files: dict[str, ArtifactMetadata]


@dataclass(frozen=True, slots=True)
class EvaluationSet:
    manifest: EvaluationManifest
    documents: tuple[Document, ...]
    queries: tuple[EvaluationQuery, ...]
    judgments: tuple[RelevanceJudgment, ...]

    def judgments_for(self, query_id: str) -> tuple[RelevanceJudgment, ...]:
        if query_id not in {query.query_id for query in self.queries}:
            raise KeyError(f"unknown query ID {query_id!r}")
        return tuple(item for item in self.judgments if item.query_id == query_id)

    def relevant_document_ids(
        self,
        query_id: str,
        *,
        minimum_relevance: int = 1,
    ) -> frozenset[str]:
        if not 1 <= minimum_relevance <= 3:
            raise ValueError("minimum_relevance must be between one and three")
        return frozenset(
            judgment.document_id
            for judgment in self.judgments_for(query_id)
            if judgment.relevance >= minimum_relevance
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_lines(path: Path, model: type[BaseModel]) -> list[BaseModel]:
    records: list[BaseModel] = []
    try:
        with path.open("rb") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    raise EvaluationSetError(f"blank record in {path.name} line {line_number}")
                try:
                    records.append(model.model_validate_json(line))
                except ValidationError as exc:
                    raise EvaluationSetError(
                        f"invalid {path.name} record at line {line_number}"
                    ) from exc
    except OSError as exc:
        raise EvaluationSetError(f"cannot read {path}") from exc
    return records


def _require_unique(values: list[str], label: str) -> None:
    if len(values) != len(set(values)):
        raise EvaluationSetError(f"duplicate {label}")


def load_evaluation_set(path: str | Path) -> EvaluationSet:
    """Load and fully validate a hand-reviewed evaluation dataset."""

    root = Path(path)
    manifest_path = root / "manifest.json"
    try:
        manifest = EvaluationManifest.model_validate_json(manifest_path.read_bytes())
    except (OSError, ValidationError) as exc:
        raise EvaluationSetError("evaluation manifest is missing or invalid") from exc
    if manifest.format_version != 1:
        raise EvaluationSetError(f"unsupported evaluation format version {manifest.format_version}")

    expected_files: dict[str, type[BaseModel]] = {
        "documents.jsonl": Document,
        "queries.jsonl": EvaluationQuery,
        "qrels.jsonl": RelevanceJudgment,
    }
    if set(manifest.files) != set(expected_files):
        raise EvaluationSetError("manifest must describe exactly the evaluation artifacts")

    loaded: dict[str, list[BaseModel]] = {}
    for filename, model in expected_files.items():
        artifact_path = root / filename
        metadata = manifest.files[filename]
        try:
            size = artifact_path.stat().st_size
        except OSError as exc:
            raise EvaluationSetError(f"missing evaluation artifact {filename}") from exc
        if size != metadata.bytes or _sha256(artifact_path) != metadata.sha256:
            raise EvaluationSetError(f"checksum or size mismatch for {filename}")
        records = _load_json_lines(artifact_path, model)
        if len(records) != metadata.records:
            raise EvaluationSetError(f"record count mismatch for {filename}")
        loaded[filename] = records

    documents = tuple(item for item in loaded["documents.jsonl"] if isinstance(item, Document))
    queries = tuple(item for item in loaded["queries.jsonl"] if isinstance(item, EvaluationQuery))
    judgments = tuple(item for item in loaded["qrels.jsonl"] if isinstance(item, RelevanceJudgment))
    _require_unique([document.document_id for document in documents], "document ID")
    _require_unique([query.query_id for query in queries], "query ID")
    _require_unique(
        [f"{judgment.query_id}\0{judgment.document_id}" for judgment in judgments],
        "query/document judgment",
    )

    document_ids = {document.document_id for document in documents}
    query_ids = {query.query_id for query in queries}
    judgments_by_query: defaultdict[str, int] = defaultdict(int)
    for judgment in judgments:
        if judgment.query_id not in query_ids:
            raise EvaluationSetError(f"judgment references unknown query {judgment.query_id!r}")
        if judgment.document_id not in document_ids:
            raise EvaluationSetError(
                f"judgment references unknown document {judgment.document_id!r}"
            )
        judgments_by_query[judgment.query_id] += 1
    missing = sorted(query_ids - judgments_by_query.keys())
    if missing:
        raise EvaluationSetError(f"queries have no positive judgments: {missing}")

    return EvaluationSet(manifest, documents, queries, judgments)


def evaluation_summary(evaluation_set: EvaluationSet) -> dict[str, Any]:
    """Return machine-readable counts for CLI output and experiment reports."""

    grade_counts: defaultdict[int, int] = defaultdict(int)
    for judgment in evaluation_set.judgments:
        grade_counts[judgment.relevance] += 1
    return {
        "dataset_id": evaluation_set.manifest.dataset_id,
        "document_count": len(evaluation_set.documents),
        "query_count": len(evaluation_set.queries),
        "positive_judgment_count": len(evaluation_set.judgments),
        "judgments_by_grade": {str(grade): grade_counts[grade] for grade in sorted(grade_counts)},
    }
