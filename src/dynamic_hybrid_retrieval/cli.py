"""Command-line interface for local benchmarks and acceptance checks."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import orjson
import typer
import uvicorn

from dynamic_hybrid_retrieval.benchmark import BenchmarkConfig, run_benchmark
from dynamic_hybrid_retrieval.diagnostics import inspect_repository
from dynamic_hybrid_retrieval.experiment import ComparisonConfig, run_comparison
from dynamic_hybrid_retrieval.load_test import LoadTestConfig, run_load_test
from dynamic_hybrid_retrieval.service import ServiceConfig, create_app

app = typer.Typer(no_args_is_help=True, pretty_exceptions_show_locals=False)


@app.callback()
def main() -> None:
    """Operate and benchmark the dynamic retrieval prototype."""


@app.command("load-test")
def load_test(
    output: Annotated[Path, typer.Option(help="New or empty report directory.")],
    documents: Annotated[int, typer.Option(min=1)] = 1_000,
    events: Annotated[int, typer.Option(min=1)] = 200,
    readers: Annotated[int, typer.Option(min=1)] = 4,
    queries: Annotated[int, typer.Option(min=1)] = 100,
) -> None:
    """Run concurrent readers, a writer, compaction, and recovery."""
    report = run_load_test(output, LoadTestConfig(documents, events, readers, queries))
    typer.echo(orjson.dumps(report.as_dict(), option=orjson.OPT_INDENT_2).decode())
    if not report.passed:
        raise typer.Exit(code=1)


@app.command()
def benchmark(
    output: Annotated[Path, typer.Option(help="New or empty output directory.")],
    documents: Annotated[int, typer.Option(min=1, help="Baseline document count.")] = 100_000,
    events: Annotated[int, typer.Option(min=1, help="Change-event count.")] = 10_000,
    queries: Annotated[int, typer.Option(min=1, help="Queries per latency phase.")] = 200,
    seed: Annotated[int, typer.Option(help="Deterministic workload seed.")] = 42,
    no_fsync: Annotated[
        bool, typer.Option(help="Disable WAL fsync for non-durability profiling.")
    ] = False,
) -> None:
    """Measure freshness, query latency, recovery, and compaction correctness."""

    config = BenchmarkConfig(
        document_count=documents,
        event_count=events,
        query_count=queries,
        seed=seed,
        fsync_event_log=not no_fsync,
    )
    report = run_benchmark(output, config)
    typer.echo(orjson.dumps(report.as_dict(), option=orjson.OPT_INDENT_2).decode())
    if not report.passed:
        raise typer.Exit(code=1)


@app.command()
def serve(
    repository: Annotated[Path, typer.Option(help="Hybrid index repository directory.")],
    event_log: Annotated[Path, typer.Option(help="Checksummed event-log file.")],
    host: Annotated[str, typer.Option(help="HTTP bind address.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(min=1, max=65535, help="HTTP bind port.")] = 8000,
    embedding_model: Annotated[
        str, typer.Option(help="'hashing' or a Sentence Transformers model ID.")
    ] = "hashing",
    embedding_dimension: Annotated[
        int, typer.Option(min=8, help="Hashing embedder dimension.")
    ] = 256,
    embedding_device: Annotated[
        str | None, typer.Option(help="Sentence Transformers device, such as cpu or mps.")
    ] = None,
    timeout: Annotated[
        float, typer.Option(min=0.001, help="Per-operation timeout in seconds.")
    ] = 10.0,
    max_request_bytes: Annotated[
        int, typer.Option(min=1, help="Maximum HTTP request body size.")
    ] = 1_048_576,
    segmented_wal: Annotated[
        bool, typer.Option(help="Use a segmented WAL directory instead of one JSONL file.")
    ] = False,
    wal_segment_bytes: Annotated[
        int, typer.Option(min=1, help="Rotate segmented WAL at this byte count.")
    ] = 64 * 1024 * 1024,
    wal_segment_events: Annotated[
        int, typer.Option(min=1, help="Rotate segmented WAL at this event count.")
    ] = 100_000,
    wal_max_retained_bytes: Annotated[
        int | None, typer.Option(min=1, help="Optional hard WAL storage budget.")
    ] = None,
    no_fsync: Annotated[
        bool, typer.Option(help="Disable WAL fsync; unsafe outside local development.")
    ] = False,
) -> None:
    """Run the single-writer HTTP retrieval service."""

    config = ServiceConfig(
        repository_path=repository,
        event_log_path=event_log,
        embedding_model=embedding_model,
        embedding_dimension=embedding_dimension,
        embedding_device=embedding_device,
        fsync_event_log=not no_fsync,
        request_timeout_seconds=timeout,
        max_request_bytes=max_request_bytes,
        segmented_wal=segmented_wal,
        wal_segment_bytes=wal_segment_bytes,
        wal_segment_events=wal_segment_events,
        wal_max_retained_bytes=wal_max_retained_bytes,
    )
    uvicorn.run(
        create_app(config),
        host=host,
        port=port,
        workers=1,
        access_log=True,
    )


@app.command()
def status(
    repository: Annotated[Path, typer.Option(help="Index repository directory.")],
    event_log: Annotated[Path, typer.Option(help="Checksummed event-log file.")],
) -> None:
    """Validate an index repository and print machine-readable health."""

    result = inspect_repository(repository, event_log)
    typer.echo(orjson.dumps(result.as_dict(), option=orjson.OPT_INDENT_2).decode())
    if not result.healthy:
        raise typer.Exit(code=1)


@app.command()
def evaluate(
    dataset: Annotated[Path, typer.Option(help="Versioned evaluation-set directory.")],
    output: Annotated[Path, typer.Option(help="New or empty report directory.")],
    model: Annotated[str, typer.Option(help="Sentence Transformers model ID.")] = (
        "sentence-transformers/all-MiniLM-L6-v2"
    ),
    k: Annotated[int, typer.Option(min=1, help="Evaluation cutoff.")] = 5,
    candidates: Annotated[int, typer.Option(min=1, help="Candidates per RRF source.")] = 16,
    warmups: Annotated[int, typer.Option(min=0, help="Warm-up rounds.")] = 2,
    rounds: Annotated[int, typer.Option(min=1, help="Measured rounds.")] = 10,
    device: Annotated[
        str | None, typer.Option(help="Embedding device, such as cpu or mps.")
    ] = None,
) -> None:
    """Compare BM25, dense HNSW, and hybrid RRF retrieval."""

    report = run_comparison(
        dataset,
        output,
        config=ComparisonConfig(
            k=k,
            candidate_limit=candidates,
            warmup_rounds=warmups,
            measured_rounds=rounds,
            model_name=model,
        ),
        device=device,
    )
    typer.echo(orjson.dumps(report.as_dict(), option=orjson.OPT_INDENT_2).decode())


if __name__ == "__main__":
    app()
