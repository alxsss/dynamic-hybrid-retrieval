"""Reproducible BM25, dense, and hybrid evaluation experiment."""

from __future__ import annotations

import os
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import orjson

from dynamic_hybrid_retrieval.bm25 import BM25Index
from dynamic_hybrid_retrieval.dense import HNSWConfig, HNSWIndex
from dynamic_hybrid_retrieval.diagnostics import RuntimeEnvironment
from dynamic_hybrid_retrieval.embeddings import EmbeddingModel, SentenceTransformerEmbedding
from dynamic_hybrid_retrieval.evaluation import evaluation_summary, load_evaluation_set
from dynamic_hybrid_retrieval.fusion import RankedList, reciprocal_rank_fusion
from dynamic_hybrid_retrieval.retrieval_evaluation import (
    HNSWRecallReport,
    RetrieverEvaluation,
    evaluate_retriever,
    measure_hnsw_recall,
)


@dataclass(frozen=True, slots=True)
class ComparisonConfig:
    k: int = 5
    candidate_limit: int = 16
    rrf_k: int = 60
    warmup_rounds: int = 2
    measured_rounds: int = 10
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    def __post_init__(self) -> None:
        if self.k < 1 or self.candidate_limit < self.k:
            raise ValueError("candidate_limit must be at least k and both must be positive")
        if self.rrf_k < 1:
            raise ValueError("rrf_k must be positive")
        if self.warmup_rounds < 0 or self.measured_rounds < 1:
            raise ValueError("invalid warmup or measured round count")
        if not self.model_name.strip():
            raise ValueError("model_name cannot be empty")


@dataclass(frozen=True, slots=True)
class ComparisonReport:
    generated_at: str
    environment: RuntimeEnvironment
    dataset: dict[str, Any]
    config: ComparisonConfig
    embedding_model_id: str
    embedding_dimension: int
    model_load_seconds: float
    bm25_build_seconds: float
    hnsw_build_seconds: float
    bm25: RetrieverEvaluation
    dense: RetrieverEvaluation
    hybrid: RetrieverEvaluation
    hnsw_recall: HNSWRecallReport

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _write_atomic(path: Path, payload: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _summary_markdown(report: ComparisonReport) -> str:
    rows = []
    for result in (report.bm25, report.dense, report.hybrid):
        rows.append(
            f"| {result.name} | {result.aggregate.mean_recall_at_k:.4f} | "
            f"{result.aggregate.mean_reciprocal_rank_at_k:.4f} | "
            f"{result.aggregate.mean_ndcg_at_k:.4f} | {result.latency.p50_ms:.3f} | "
            f"{result.latency.p95_ms:.3f} | {result.latency.p99_ms:.3f} |"
        )
    return "\n".join(
        [
            "# BM25 versus dense versus hybrid",
            "",
            f"Generated: {report.generated_at}",
            "",
            f"Dataset: `{report.dataset['dataset_id']}`",
            "",
            f"Embedding model: `{report.embedding_model_id}` "
            f"({report.embedding_dimension} dimensions)",
            "",
            f"Evaluation cutoff: K={report.config.k}",
            "",
            "| Retriever | Recall@K | MRR@K | nDCG@K | P50 ms | P95 ms | P99 ms |",
            "|---|---:|---:|---:|---:|---:|---:|",
            *rows,
            "",
            f"HNSW recall@{report.hnsw_recall.k}: {report.hnsw_recall.mean_recall_at_k:.4f}",
            "",
            "The evaluation set is intentionally small and supports pipeline validation, "
            "not broad statistical claims. Latencies are local single-process measurements.",
            "",
        ]
    )


def run_comparison(
    dataset_path: str | Path,
    output_directory: str | Path,
    *,
    config: ComparisonConfig | None = None,
    embedding_model: EmbeddingModel | None = None,
    device: str | None = None,
) -> ComparisonReport:
    """Build all retrievers, evaluate them, and atomically persist reports."""

    active_config = config or ComparisonConfig()
    output = Path(output_directory)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"evaluation output directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    dataset = load_evaluation_set(dataset_path)

    model_started = time.perf_counter()
    model = embedding_model or SentenceTransformerEmbedding(
        active_config.model_name,
        device=device,
    )
    model_load_seconds = time.perf_counter() - model_started

    bm25_started = time.perf_counter()
    bm25 = BM25Index.build(dataset.documents)
    bm25_build_seconds = time.perf_counter() - bm25_started

    hnsw_started = time.perf_counter()
    hnsw = HNSWIndex.build(dataset.documents, model, config=HNSWConfig())
    hnsw_build_seconds = time.perf_counter() - hnsw_started

    common = {
        "k": active_config.k,
        "warmup_rounds": active_config.warmup_rounds,
        "measured_rounds": active_config.measured_rounds,
    }
    bm25_report = evaluate_retriever(
        "bm25",
        dataset,
        lambda query, k: bm25.search(query, limit=k),
        **common,
    )
    dense_report = evaluate_retriever(
        "dense-hnsw",
        dataset,
        lambda query, k: hnsw.search(query, limit=k),
        **common,
    )

    def hybrid_search(query: str, k: int):  # type: ignore[no-untyped-def]
        lexical_results = bm25.search(query, limit=active_config.candidate_limit)
        dense_results = hnsw.search(query, limit=active_config.candidate_limit)
        return reciprocal_rank_fusion(
            [RankedList("lexical", lexical_results), RankedList("dense", dense_results)],
            limit=k,
            k=active_config.rrf_k,
        )

    hybrid_report = evaluate_retriever(
        "hybrid-rrf",
        dataset,
        hybrid_search,
        **common,
    )
    hnsw_recall = measure_hnsw_recall(
        hnsw,
        {query.query_id: query.text for query in dataset.queries},
        k=active_config.k,
    )
    report = ComparisonReport(
        generated_at=datetime.now(UTC).isoformat(),
        environment=RuntimeEnvironment.capture(),
        dataset=evaluation_summary(dataset),
        config=active_config,
        embedding_model_id=model.model_id,
        embedding_dimension=model.dimension,
        model_load_seconds=model_load_seconds,
        bm25_build_seconds=bm25_build_seconds,
        hnsw_build_seconds=hnsw_build_seconds,
        bm25=bm25_report,
        dense=dense_report,
        hybrid=hybrid_report,
        hnsw_recall=hnsw_recall,
    )
    _write_atomic(
        output / "report.json",
        orjson.dumps(report.as_dict(), option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS) + b"\n",
    )
    _write_atomic(output / "summary.md", _summary_markdown(report).encode())
    return report
