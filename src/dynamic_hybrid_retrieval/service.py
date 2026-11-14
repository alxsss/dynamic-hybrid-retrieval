"""Versioned HTTP service for the coordinated hybrid retrieval index."""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Final

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator
from starlette.exceptions import HTTPException as StarletteHTTPException

from dynamic_hybrid_retrieval.embeddings import (
    EmbeddingModel,
    HashingEmbedding,
    SentenceTransformerEmbedding,
)
from dynamic_hybrid_retrieval.event_log import (
    DuplicateSequenceError,
    EventLog,
    EventLogCorruptionError,
    SequenceOrderError,
)
from dynamic_hybrid_retrieval.filters import (
    AllOf,
    AnyOf,
    DocumentFilter,
    Equals,
    Exists,
    InValues,
    MetadataScalar,
    Not,
    NumericRange,
)
from dynamic_hybrid_retrieval.fusion import HybridConsistencyError
from dynamic_hybrid_retrieval.hybrid_index import (
    DynamicHybridIndex,
    HybridIndexUnhealthyError,
)
from dynamic_hybrid_retrieval.hybrid_repository import (
    HybridIndexRepository,
    HybridRepositoryCorruptionError,
    HybridRepositoryError,
)
from dynamic_hybrid_retrieval.models import Document, IndexEvent, Operation

API_VERSION: Final = "v1"
SERVICE_VERSION: Final = "0.1.0"
LOGGER = logging.getLogger(__name__)


class SearchMode(StrEnum):
    LEXICAL = "lexical"
    DENSE = "dense"
    HYBRID = "hybrid"


class FilterOperator(StrEnum):
    EQUALS = "equals"
    IN = "in"
    RANGE = "range"
    EXISTS = "exists"
    ALL = "all"
    ANY = "any"
    NOT = "not"


class FilterSpec(BaseModel):
    """Validated wire representation of metadata filter expressions."""

    model_config = ConfigDict(extra="forbid")

    operator: FilterOperator
    path: str | None = None
    value: MetadataScalar = None
    values: tuple[MetadataScalar, ...] | None = None
    minimum: int | float | None = None
    maximum: int | float | None = None
    include_minimum: bool = True
    include_maximum: bool = True
    filters: tuple[FilterSpec, ...] | None = None

    @model_validator(mode="after")
    def validate_for_operator(self) -> FilterSpec:
        if (
            self.operator
            in {
                FilterOperator.EQUALS,
                FilterOperator.IN,
                FilterOperator.RANGE,
                FilterOperator.EXISTS,
            }
            and not self.path
        ):
            raise ValueError(f"{self.operator} requires path")
        if self.operator is FilterOperator.IN and not self.values:
            raise ValueError("in requires non-empty values")
        if self.operator is FilterOperator.RANGE and (
            self.minimum is None and self.maximum is None
        ):
            raise ValueError("range requires minimum or maximum")
        if self.operator in {FilterOperator.ALL, FilterOperator.ANY} and not self.filters:
            raise ValueError(f"{self.operator} requires non-empty filters")
        if self.operator is FilterOperator.NOT and (self.filters is None or len(self.filters) != 1):
            raise ValueError("not requires exactly one nested filter")
        self.to_domain()
        return self

    def to_domain(self) -> DocumentFilter:
        if self.operator is FilterOperator.EQUALS:
            assert self.path is not None
            return Equals(self.path, self.value)
        if self.operator is FilterOperator.IN:
            assert self.path is not None and self.values is not None
            return InValues(self.path, self.values)
        if self.operator is FilterOperator.RANGE:
            assert self.path is not None
            return NumericRange(
                self.path,
                minimum=self.minimum,
                maximum=self.maximum,
                include_minimum=self.include_minimum,
                include_maximum=self.include_maximum,
            )
        if self.operator is FilterOperator.EXISTS:
            assert self.path is not None
            return Exists(self.path)
        assert self.filters is not None
        nested = tuple(item.to_domain() for item in self.filters)
        if self.operator is FilterOperator.ALL:
            return AllOf(nested)
        if self.operator is FilterOperator.ANY:
            return AnyOf(nested)
        return Not(nested[0])


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: Annotated[str, Field(min_length=1, max_length=4096)]
    mode: SearchMode = SearchMode.HYBRID
    limit: Annotated[int, Field(ge=1, le=100)] = 10
    exact_dense: bool = False
    filter: FilterSpec | None = None


class DocumentRequest(BaseModel):
    """JSON-facing document model converted into the strict domain model."""

    model_config = ConfigDict(extra="forbid")

    document_id: Annotated[str, Field(min_length=1)]
    title: str
    body: str
    updated_at: datetime
    version: Annotated[int, Field(ge=1)]
    metadata: dict[str, Any] = Field(default_factory=dict)


class IndexEventRequest(BaseModel):
    """JSON-facing event model; domain invariants are checked by IndexEvent."""

    model_config = ConfigDict(extra="forbid")

    sequence_number: Annotated[int, Field(ge=1)]
    operation: Operation
    document_id: Annotated[str, Field(min_length=1)]
    version: Annotated[int, Field(ge=1)]
    event_time: datetime
    document: DocumentRequest | None = None

    @model_validator(mode="after")
    def validate_domain_invariants(self) -> IndexEventRequest:
        self.to_domain()
        return self

    def to_domain(self) -> IndexEvent:
        return IndexEvent.model_validate_json(self.model_dump_json())


class SearchContribution(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    source: str
    rank: int
    raw_score: float
    weight: float
    contribution: float


class SearchResultResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    document_id: str
    score: float
    version: int
    document: Document
    contributions: tuple[SearchContribution, ...] = ()


class SearchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    mode: SearchMode
    high_water_sequence: int
    took_ms: float
    results: tuple[SearchResultResponse, ...]


class ApplyResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    status: str
    sequence_number: int
    document_id: str
    version: int
    high_water_sequence: int


@dataclass(frozen=True, slots=True)
class ServiceConfig:
    repository_path: Path
    event_log_path: Path
    embedding_model: str = "hashing"
    embedding_dimension: int = 256
    embedding_device: str | None = None
    fsync_event_log: bool = True
    request_timeout_seconds: float = 10.0
    max_request_bytes: int = 1_048_576

    def __post_init__(self) -> None:
        if not self.embedding_model.strip():
            raise ValueError("embedding_model cannot be empty")
        if self.embedding_dimension < 8:
            raise ValueError("embedding_dimension must be at least eight")
        if self.request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be positive")
        if self.max_request_bytes < 1:
            raise ValueError("max_request_bytes must be positive")

    @classmethod
    def from_environment(cls) -> ServiceConfig:
        repository = os.environ.get("DHR_REPOSITORY")
        event_log = os.environ.get("DHR_EVENT_LOG")
        if not repository or not event_log:
            raise ValueError("DHR_REPOSITORY and DHR_EVENT_LOG are required")
        return cls(
            repository_path=Path(repository),
            event_log_path=Path(event_log),
            embedding_model=os.environ.get("DHR_EMBEDDING_MODEL", "hashing"),
            embedding_dimension=int(os.environ.get("DHR_EMBEDDING_DIMENSION", "256")),
            embedding_device=os.environ.get("DHR_EMBEDDING_DEVICE"),
            fsync_event_log=_environment_bool("DHR_FSYNC_EVENT_LOG", default=True),
            request_timeout_seconds=float(os.environ.get("DHR_REQUEST_TIMEOUT", "10")),
            max_request_bytes=int(os.environ.get("DHR_MAX_REQUEST_BYTES", "1048576")),
        )


@dataclass(frozen=True, slots=True)
class ServiceRuntime:
    repository: HybridIndexRepository
    event_log: EventLog
    embedding_model: EmbeddingModel
    index: DynamicHybridIndex


RuntimeLoader = Callable[[ServiceConfig], ServiceRuntime]


def _environment_bool(name: str, *, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.casefold()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean")


def _create_embedding(config: ServiceConfig) -> EmbeddingModel:
    if config.embedding_model == "hashing":
        return HashingEmbedding(config.embedding_dimension)
    return SentenceTransformerEmbedding(
        config.embedding_model,
        device=config.embedding_device,
    )


def load_runtime(config: ServiceConfig) -> ServiceRuntime:
    """Synchronously load, validate, and recover the complete serving state."""

    embedding = _create_embedding(config)
    repository = HybridIndexRepository(config.repository_path)
    event_log = EventLog(config.event_log_path, fsync=config.fsync_event_log)
    index = repository.load_active(event_log, embedding)
    return ServiceRuntime(repository, event_log, embedding, index)


def _error_response(request: Request, status: int, code: str, message: str) -> JSONResponse:
    request_id = getattr(request.state, "request_id", "unknown")
    return JSONResponse(
        status_code=status,
        content={"error": {"code": code, "message": message, "request_id": request_id}},
        headers={"X-Request-ID": request_id},
    )


async def _with_timeout(
    operation: Callable[[], Any],
    timeout_seconds: float,
) -> Any:
    try:
        async with asyncio.timeout(timeout_seconds):
            return await asyncio.to_thread(operation)
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail="operation exceeded request timeout") from exc


def create_app(
    config: ServiceConfig | None = None,
    *,
    runtime_loader: RuntimeLoader = load_runtime,
) -> FastAPI:
    """Create an isolated app; loading and WAL recovery occur in its lifespan."""

    service_config = config or ServiceConfig.from_environment()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> Any:
        app.state.ready = False
        app.state.runtime = await asyncio.to_thread(runtime_loader, service_config)
        app.state.ready = True
        try:
            yield
        finally:
            app.state.ready = False
            app.state.runtime = None

    app = FastAPI(
        title="Dynamic Hybrid Retrieval",
        version=SERVICE_VERSION,
        lifespan=lifespan,
        docs_url="/docs",
        openapi_url="/openapi.json",
    )

    @app.middleware("http")
    async def enforce_request_size(
        request: Request, call_next: Callable[[Request], Awaitable[Any]]
    ) -> Any:
        supplied_request_id = request.headers.get("X-Request-ID")
        request.state.request_id = (
            supplied_request_id
            if supplied_request_id is not None and 1 <= len(supplied_request_id) <= 128
            else str(uuid.uuid4())
        )
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                declared_size = int(content_length)
            except ValueError:
                return _error_response(
                    request, 400, "invalid_content_length", "invalid Content-Length"
                )
            if declared_size > service_config.max_request_bytes:
                return _error_response(
                    request, 413, "request_too_large", "request body is too large"
                )
        body = await request.body()
        if len(body) > service_config.max_request_bytes:
            return _error_response(request, 413, "request_too_large", "request body is too large")
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        return _error_response(request, 422, "invalid_request", str(exc.errors()))

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = "request_timeout" if exc.status_code == 504 else "http_error"
        return _error_response(request, exc.status_code, code, str(exc.detail))

    @app.exception_handler(SequenceOrderError)
    @app.exception_handler(DuplicateSequenceError)
    async def sequence_error(request: Request, exc: Exception) -> JSONResponse:
        return _error_response(request, 409, "sequence_conflict", str(exc))

    @app.exception_handler(EventLogCorruptionError)
    @app.exception_handler(HybridRepositoryCorruptionError)
    @app.exception_handler(HybridRepositoryError)
    @app.exception_handler(HybridConsistencyError)
    @app.exception_handler(HybridIndexUnhealthyError)
    async def unavailable_error(request: Request, exc: Exception) -> JSONResponse:
        return _error_response(request, 503, "index_unavailable", str(exc))

    @app.exception_handler(Exception)
    async def unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        LOGGER.exception("unhandled request failure", exc_info=exc)
        return _error_response(request, 500, "internal_error", "internal server error")

    @app.get("/health/live")
    async def live() -> dict[str, str]:
        return {"status": "alive", "version": SERVICE_VERSION}

    @app.get("/health/ready")
    async def ready(request: Request) -> JSONResponse:
        runtime: ServiceRuntime | None = getattr(request.app.state, "runtime", None)
        if (
            runtime is None
            or not getattr(request.app.state, "ready", False)
            or not runtime.index.statistics.healthy
        ):
            return _error_response(request, 503, "not_ready", "index is not ready")
        statistics = runtime.index.statistics
        return JSONResponse(
            {
                "status": "ready",
                "high_water_sequence": statistics.last_sequence_number,
                "live_document_count": statistics.live_document_count,
            }
        )

    def runtime_from(request: Request) -> ServiceRuntime:
        runtime: ServiceRuntime | None = getattr(request.app.state, "runtime", None)
        if runtime is None or not getattr(request.app.state, "ready", False):
            raise HybridIndexUnhealthyError("index is not ready")
        return runtime

    @app.post(f"/{API_VERSION}/events", response_model=ApplyResponse)
    async def apply_event(event_request: IndexEventRequest, request: Request) -> ApplyResponse:
        runtime = runtime_from(request)
        event = event_request.to_domain()
        result = await _with_timeout(
            lambda: runtime.index.apply(event),
            service_config.request_timeout_seconds,
        )
        statistics = runtime.index.statistics
        return ApplyResponse(
            status=result.status,
            sequence_number=result.sequence_number,
            document_id=result.document_id,
            version=result.version,
            high_water_sequence=statistics.last_sequence_number,
        )

    @app.post(f"/{API_VERSION}/search", response_model=SearchResponse)
    async def search(search_request: SearchRequest, request: Request) -> SearchResponse:
        runtime = runtime_from(request)
        document_filter = (
            search_request.filter.to_domain() if search_request.filter is not None else None
        )
        started = time.perf_counter()

        def execute() -> list[Any]:
            if search_request.mode is SearchMode.LEXICAL:
                return runtime.index.search_lexical(
                    search_request.query,
                    limit=search_request.limit,
                    document_filter=document_filter,
                )
            if search_request.mode is SearchMode.DENSE:
                return runtime.index.search_dense(
                    search_request.query,
                    limit=search_request.limit,
                    exact=search_request.exact_dense,
                    document_filter=document_filter,
                )
            return runtime.index.search(
                search_request.query,
                limit=search_request.limit,
                exact_dense=search_request.exact_dense,
                document_filter=document_filter,
            )

        results = await _with_timeout(execute, service_config.request_timeout_seconds)
        took_ms = (time.perf_counter() - started) * 1000
        response_results = tuple(
            SearchResultResponse(
                document_id=result.document_id,
                score=result.score,
                version=result.version,
                document=result.document,
                contributions=tuple(
                    SearchContribution(
                        source=contribution.source,
                        rank=contribution.rank,
                        raw_score=contribution.raw_score,
                        weight=contribution.weight,
                        contribution=contribution.contribution,
                    )
                    for contribution in getattr(result, "contributions", ())
                ),
            )
            for result in results
        )
        return SearchResponse(
            mode=search_request.mode,
            high_water_sequence=runtime.index.statistics.last_sequence_number,
            took_ms=took_ms,
            results=response_results,
        )

    return app
