from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from dynamic_hybrid_retrieval.bm25 import BM25Index
from dynamic_hybrid_retrieval.dense import HNSWIndex
from dynamic_hybrid_retrieval.embeddings import HashingEmbedding
from dynamic_hybrid_retrieval.event_log import EventLog
from dynamic_hybrid_retrieval.hybrid_repository import HybridIndexRepository
from dynamic_hybrid_retrieval.models import Document, IndexEvent, Operation
from dynamic_hybrid_retrieval.service import (
    ServiceConfig,
    ServiceRuntime,
    create_app,
)

NOW = datetime(2026, 11, 14, tzinfo=UTC)


def document(
    document_id: str,
    text: str,
    *,
    version: int = 1,
    metadata: dict[str, object] | None = None,
) -> Document:
    return Document(
        document_id=document_id,
        title=text,
        body=text,
        updated_at=NOW,
        version=version,
        metadata=metadata or {},
    )


def event(sequence: int, item: Document) -> IndexEvent:
    return IndexEvent(
        sequence_number=sequence,
        operation=Operation.UPSERT,
        document_id=item.document_id,
        version=item.version,
        event_time=NOW,
        document=item,
    )


def service_config(tmp_path: Path, **overrides: object) -> ServiceConfig:
    values: dict[str, object] = {
        "repository_path": tmp_path / "repository",
        "event_log_path": tmp_path / "events.jsonl",
        "embedding_dimension": 64,
        "fsync_event_log": False,
    }
    values.update(overrides)
    return ServiceConfig(**values)  # type: ignore[arg-type]


def initialize_runtime(config: ServiceConfig) -> ServiceRuntime:
    embedding = HashingEmbedding(config.embedding_dimension)
    documents = [
        document("doc-a", "distributed search retrieval", metadata={"team": "search"}),
        document("doc-b", "semantic vector ranking", metadata={"team": "ml"}),
    ]
    repository = HybridIndexRepository(config.repository_path)
    repository.initialize(
        BM25Index.build(documents),
        HNSWIndex.build(documents, embedding),
    )
    event_log = EventLog(config.event_log_path, fsync=False)
    return ServiceRuntime(
        repository,
        event_log,
        embedding,
        repository.load_active(event_log, embedding),
    )


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    config = service_config(tmp_path)
    app = create_app(config, runtime_loader=initialize_runtime)
    with TestClient(app) as test_client:
        yield test_client


def test_liveness_readiness_and_request_id(client: TestClient) -> None:
    live = client.get("/health/live", headers={"X-Request-ID": "request-123"})
    ready = client.get("/health/ready")

    assert live.status_code == 200
    assert live.json() == {"status": "alive", "version": "0.1.0"}
    assert live.headers["X-Request-ID"] == "request-123"
    assert ready.status_code == 200
    assert ready.json()["status"] == "ready"
    assert ready.json()["live_document_count"] == 2


@pytest.mark.parametrize("mode", ["lexical", "dense", "hybrid"])
def test_search_modes_return_versioned_documents(client: TestClient, mode: str) -> None:
    response = client.post(
        "/v1/search",
        json={"query": "distributed search", "mode": mode, "exact_dense": True},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == mode
    assert body["high_water_sequence"] == 0
    assert body["results"][0]["document_id"] == "doc-a"
    assert body["results"][0]["version"] == 1
    if mode == "hybrid":
        assert body["results"][0]["contributions"]


def test_event_is_durable_and_immediately_searchable(client: TestClient) -> None:
    item = document("doc-c", "fresh ocean wildlife", metadata={"team": "science"})
    response = client.post("/v1/events", json=event(1, item).model_dump(mode="json"))

    assert response.status_code == 200
    assert response.json()["status"] == "applied"
    assert response.json()["high_water_sequence"] == 1
    search = client.post(
        "/v1/search",
        json={"query": "ocean wildlife", "mode": "hybrid", "exact_dense": True},
    )
    assert search.json()["results"][0]["document_id"] == "doc-c"


def test_metadata_filter_is_applied_to_all_retrievers(client: TestClient) -> None:
    response = client.post(
        "/v1/search",
        json={
            "query": "search vector",
            "mode": "hybrid",
            "exact_dense": True,
            "filter": {"operator": "equals", "path": "team", "value": "ml"},
        },
    )

    assert response.status_code == 200
    assert [item["document_id"] for item in response.json()["results"]] == ["doc-b"]


def test_invalid_request_has_stable_error_envelope(client: TestClient) -> None:
    response = client.post(
        "/v1/search",
        json={"query": "", "mode": "unknown", "limit": 1000},
        headers={"X-Request-ID": "invalid-123"},
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_request"
    assert response.json()["error"]["request_id"] == "invalid-123"


def test_request_size_limit_is_enforced(tmp_path: Path) -> None:
    config = service_config(tmp_path, max_request_bytes=100)
    app = create_app(config, runtime_loader=initialize_runtime)
    with TestClient(app) as test_client:
        response = test_client.post("/v1/search", content=b"x" * 101)

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "request_too_large"


def test_conflicting_sequence_returns_409(client: TestClient) -> None:
    first = event(1, document("doc-c", "first"))
    conflict = event(1, document("doc-d", "different event"))
    assert client.post("/v1/events", json=first.model_dump(mode="json")).status_code == 200

    response = client.post("/v1/events", json=conflict.model_dump(mode="json"))

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "sequence_conflict"


def test_unhealthy_index_fails_readiness_and_search(tmp_path: Path) -> None:
    config = service_config(tmp_path)
    runtime = initialize_runtime(config)
    runtime.index._healthy = False
    app = create_app(config, runtime_loader=lambda _: runtime)
    with TestClient(app, raise_server_exceptions=False) as test_client:
        ready = test_client.get("/health/ready")
        search = test_client.post("/v1/search", json={"query": "search"})

    assert ready.status_code == 503
    assert ready.json()["error"]["code"] == "not_ready"
    assert search.status_code == 503
    assert search.json()["error"]["code"] == "index_unavailable"


def test_search_timeout_returns_504(tmp_path: Path) -> None:
    config = service_config(tmp_path, request_timeout_seconds=0.01)
    runtime = initialize_runtime(config)

    def slow_search(*args: object, **kwargs: object) -> list[object]:
        time.sleep(0.05)
        return []

    runtime.index.search = slow_search  # type: ignore[method-assign]
    app = create_app(config, runtime_loader=lambda _: runtime)
    with TestClient(app, raise_server_exceptions=False) as test_client:
        response = test_client.post("/v1/search", json={"query": "search"})

    assert response.status_code == 504
    assert response.json()["error"]["code"] == "request_timeout"


def test_startup_replays_wal_before_readiness(tmp_path: Path) -> None:
    config = service_config(tmp_path)
    runtime = initialize_runtime(config)
    runtime.index.apply(event(1, document("doc-c", "recovered before ready")))

    def recover(active_config: ServiceConfig) -> ServiceRuntime:
        embedding = HashingEmbedding(active_config.embedding_dimension)
        repository = HybridIndexRepository(active_config.repository_path)
        event_log = EventLog(active_config.event_log_path, fsync=False)
        return ServiceRuntime(
            repository,
            event_log,
            embedding,
            repository.load_active(event_log, embedding),
        )

    app = create_app(config, runtime_loader=recover)
    with TestClient(app) as test_client:
        ready = test_client.get("/health/ready")
        search = test_client.post(
            "/v1/search",
            json={"query": "recovered before ready", "exact_dense": True},
        )

    assert ready.json()["high_water_sequence"] == 1
    assert search.json()["results"][0]["document_id"] == "doc-c"
