from __future__ import annotations

from pathlib import Path

import orjson
import pytest

from dynamic_hybrid_retrieval.embeddings import HashingEmbedding
from dynamic_hybrid_retrieval.experiment import ComparisonConfig, run_comparison

DATASET = Path(__file__).parents[1] / "data" / "evaluation" / "v1"


def test_comparison_writes_reproducible_reports(tmp_path: Path) -> None:
    output = tmp_path / "comparison"
    report = run_comparison(
        DATASET,
        output,
        config=ComparisonConfig(
            k=5,
            candidate_limit=16,
            warmup_rounds=0,
            measured_rounds=1,
            model_name="test-model",
        ),
        embedding_model=HashingEmbedding(128),
    )

    persisted = orjson.loads((output / "report.json").read_bytes())
    summary = (output / "summary.md").read_text()
    assert persisted["dataset"]["dataset_id"] == "dynamic-retrieval-eval-v1"
    assert persisted["embedding_model_id"] == "hashing-blake2b-v1-128"
    assert persisted["bm25"]["latency"]["count"] == 10
    assert report.hnsw_recall.mean_recall_at_k >= 0
    assert "| bm25 |" in summary
    assert "| dense-hnsw |" in summary
    assert "| hybrid-rrf |" in summary


def test_comparison_refuses_nonempty_output(tmp_path: Path) -> None:
    output = tmp_path / "comparison"
    output.mkdir()
    (output / "keep.txt").write_text("do not overwrite")

    with pytest.raises(FileExistsError):
        run_comparison(
            DATASET,
            output,
            config=ComparisonConfig(model_name="test"),
            embedding_model=HashingEmbedding(64),
        )


def test_comparison_config_is_validated() -> None:
    with pytest.raises(ValueError):
        ComparisonConfig(k=10, candidate_limit=5)
    with pytest.raises(ValueError):
        ComparisonConfig(rrf_k=0)
    with pytest.raises(ValueError):
        ComparisonConfig(measured_rounds=0)
