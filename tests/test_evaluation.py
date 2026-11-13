from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from dynamic_hybrid_retrieval.evaluation import (
    EvaluationSetError,
    evaluation_summary,
    load_evaluation_set,
)

DATASET = Path(__file__).parents[1] / "data" / "evaluation" / "v1"


def test_hand_reviewed_evaluation_set_loads_and_cross_references() -> None:
    dataset = load_evaluation_set(DATASET)

    assert len(dataset.documents) == 16
    assert len(dataset.queries) == 10
    assert len(dataset.judgments) == 23
    assert dataset.relevant_document_ids("eval-q03", minimum_relevance=3) == {"eval-doc-07"}
    assert all(judgment.rationale for judgment in dataset.judgments)
    assert all(dataset.judgments_for(query.query_id) for query in dataset.queries)


def test_evaluation_summary_reports_grades() -> None:
    summary = evaluation_summary(load_evaluation_set(DATASET))

    assert summary["dataset_id"] == "dynamic-retrieval-eval-v1"
    assert summary["document_count"] == 16
    assert summary["query_count"] == 10
    assert summary["positive_judgment_count"] == 23
    assert summary["judgments_by_grade"] == {"1": 8, "2": 5, "3": 10}


def test_unknown_query_and_invalid_grade_threshold_are_rejected() -> None:
    dataset = load_evaluation_set(DATASET)

    with pytest.raises(KeyError):
        dataset.judgments_for("missing")
    with pytest.raises(ValueError):
        dataset.relevant_document_ids("eval-q01", minimum_relevance=0)


def test_artifact_corruption_is_detected(tmp_path: Path) -> None:
    copied = tmp_path / "evaluation"
    shutil.copytree(DATASET, copied)
    with (copied / "queries.jsonl").open("ab") as stream:
        stream.write(b"corruption")

    with pytest.raises(EvaluationSetError, match="checksum or size"):
        load_evaluation_set(copied)
