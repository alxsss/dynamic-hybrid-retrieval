"""Minimal OpenSearch REST adapter for reproducible retrieval comparisons."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx
import orjson

from dynamic_hybrid_retrieval.embeddings import EmbeddingModel, HashingEmbedding
from dynamic_hybrid_retrieval.evaluation import load_evaluation_set
from dynamic_hybrid_retrieval.fusion import RankedList, reciprocal_rank_fusion
from dynamic_hybrid_retrieval.models import Document, IndexEvent, Operation
from dynamic_hybrid_retrieval.retrieval_evaluation import evaluate_retriever


class OpenSearchError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class OpenSearchResult:
    document_id: str
    score: float
    version: int
    document: Document


class OpenSearchAdapter:
    def __init__(
        self,
        endpoint: str,
        index: str,
        embedding: EmbeddingModel,
        *,
        client: httpx.Client | None = None,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.index = index
        self.embedding = embedding
        self.client = client or httpx.Client(base_url=self.endpoint, timeout=30, verify=False)

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        response = self.client.request(method, path, **kwargs)
        if response.status_code >= 400:
            raise OpenSearchError(f"OpenSearch {response.status_code}: {response.text}")
        return response.json() if response.content else {}

    def create_index(self) -> None:
        self._request("DELETE", f"/{self.index}") if self.client.get(
            f"/{self.index}"
        ).status_code == 200 else None
        self._request(
            "PUT",
            f"/{self.index}",
            json={
                "settings": {
                    "index": {"knn": True, "number_of_shards": 1, "number_of_replicas": 0}
                },
                "mappings": {
                    "properties": {
                        "title": {"type": "text"},
                        "body": {"type": "text"},
                        "version": {"type": "long"},
                        "updated_at": {"type": "date"},
                        "metadata": {"type": "object"},
                        "embedding": {
                            "type": "knn_vector",
                            "dimension": self.embedding.dimension,
                            "method": {
                                "name": "hnsw",
                                "space_type": "cosinesimil",
                                "engine": "lucene",
                            },
                        },
                    }
                },
            },
        )

    def index_documents(self, documents: list[Document], *, refresh: str = "wait_for") -> None:
        vectors = self.embedding.embed_documents([f"{d.title}\n{d.body}" for d in documents])
        lines: list[str] = []
        import orjson

        for document, vector in zip(documents, vectors, strict=True):
            lines.append(
                orjson.dumps(
                    {"index": {"_index": self.index, "_id": document.document_id}}
                ).decode()
            )
            source = document.model_dump(mode="json")
            source["embedding"] = vector.tolist()
            lines.append(orjson.dumps(source).decode())
        result = self._request(
            "POST",
            f"/_bulk?refresh={refresh}",
            content=("\n".join(lines) + "\n").encode(),
            headers={"Content-Type": "application/x-ndjson"},
        )
        if result.get("errors"):
            raise OpenSearchError("bulk indexing contained item failures")

    def apply(self, event: IndexEvent, *, refresh: str = "wait_for") -> None:
        if event.operation is Operation.DELETE:
            response = self.client.delete(
                f"/{self.index}/_doc/{event.document_id}?refresh={refresh}"
            )
            if response.status_code not in {200, 404}:
                raise OpenSearchError(response.text)
        else:
            assert event.document is not None
            self.index_documents([event.document], refresh=refresh)

    def search(
        self, query: str, *, mode: str = "hybrid", limit: int = 10, candidates: int = 100
    ) -> list[Any]:
        if mode == "hybrid":
            return reciprocal_rank_fusion(
                [
                    RankedList("lexical", self.search(query, mode="lexical", limit=candidates)),
                    RankedList("dense", self.search(query, mode="dense", limit=candidates)),
                ],
                limit=limit,
            )
        clause: dict[str, Any]
        if mode == "lexical":
            clause = {"multi_match": {"query": query, "fields": ["title", "body"]}}
        elif mode == "dense":
            clause = {
                "knn": {
                    "embedding": {
                        "vector": self.embedding.embed_query(query)[0].tolist(),
                        "k": candidates,
                    }
                }
            }
        else:
            raise ValueError("mode must be lexical, dense, or hybrid")
        payload = self._request(
            "POST", f"/{self.index}/_search", json={"size": limit, "query": clause}
        )
        return [self._result(hit) for hit in payload["hits"]["hits"]]

    @staticmethod
    def _result(hit: dict[str, Any]) -> OpenSearchResult:
        source = dict(hit["_source"])
        source.pop("embedding", None)
        document = Document.model_validate_json(__import__("orjson").dumps(source))
        return OpenSearchResult(
            document.document_id, float(hit["_score"]), document.version, document
        )


def run_opensearch_evaluation(
    dataset: str | Path,
    output: str | Path,
    *,
    endpoint: str = "http://localhost:9200",
    index: str = "dhr-comparison",
    dimension: int = 256,
    k: int = 5,
) -> Path:
    """Index the hand-verified corpus and write measured OpenSearch results."""
    evaluation_set = load_evaluation_set(dataset)
    adapter = OpenSearchAdapter(endpoint, index, HashingEmbedding(dimension))
    version = adapter._request("GET", "/")
    adapter.create_index()
    adapter.index_documents(list(evaluation_set.documents))
    runs: dict[str, Any] = {}
    for mode in ("lexical", "dense", "hybrid"):
        def search(query: str, limit: int, active_mode: str = mode) -> list[Any]:
            return adapter.search(query, mode=active_mode, limit=limit)

        run = evaluate_retriever(
            f"opensearch-{mode}",
            evaluation_set,
            search,
            k=k,
            warmup_rounds=1,
            measured_rounds=5,
        )
        runs[mode] = asdict(run)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(
        orjson.dumps(
            {
                "endpoint": endpoint,
                "index": index,
                "opensearch_version": version["version"]["number"],
                "embedding_model": adapter.embedding.model_id,
                "k": k,
                "runs": runs,
            },
            option=orjson.OPT_INDENT_2,
        )
        + b"\n"
    )
    return destination
