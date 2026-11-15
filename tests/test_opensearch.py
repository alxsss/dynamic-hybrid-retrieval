import httpx

from dynamic_hybrid_retrieval.embeddings import HashingEmbedding
from dynamic_hybrid_retrieval.opensearch import OpenSearchAdapter


def test_search_builds_lexical_and_dense_requests() -> None:
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"hits": {"hits": []}})

    client = httpx.Client(base_url="http://test", transport=httpx.MockTransport(handler))
    adapter = OpenSearchAdapter("http://test", "docs", HashingEmbedding(64), client=client)
    assert adapter.search("hello", mode="lexical") == []
    assert adapter.search("hello", mode="dense") == []
    assert b"multi_match" in requests[0].content
    assert b'"knn"' in requests[1].content


def test_hybrid_uses_same_client_side_rrf() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"hits": {"hits": []}})

    adapter = OpenSearchAdapter(
        "http://test",
        "docs",
        HashingEmbedding(64),
        client=httpx.Client(base_url="http://test", transport=httpx.MockTransport(handler)),
    )
    assert adapter.search("hello", mode="hybrid") == []
