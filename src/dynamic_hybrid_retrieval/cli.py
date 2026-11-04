"""Command-line interface for local benchmarks and acceptance checks."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import orjson
import typer

from dynamic_hybrid_retrieval.benchmark import BenchmarkConfig, run_benchmark
from dynamic_hybrid_retrieval.diagnostics import inspect_repository

app = typer.Typer(no_args_is_help=True, pretty_exceptions_show_locals=False)


@app.callback()
def main() -> None:
    """Operate and benchmark the dynamic retrieval prototype."""


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
def status(
    repository: Annotated[Path, typer.Option(help="Index repository directory.")],
    event_log: Annotated[Path, typer.Option(help="Checksummed event-log file.")],
) -> None:
    """Validate an index repository and print machine-readable health."""

    result = inspect_repository(repository, event_log)
    typer.echo(orjson.dumps(result.as_dict(), option=orjson.OPT_INDENT_2).decode())
    if not result.healthy:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
