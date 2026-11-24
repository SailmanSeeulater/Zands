import httpx
import pytest

from zands import store, web

from fakes import FakeEmbedder

PAGE_TEXT = " ".join(
    f"Tadej Pogacar won the Tour de France for a fifth time, matching the record, sentence {i}." for i in range(30)
)
PAGE = f"<html><body><article><h1>Pogacar wins</h1><p>{PAGE_TEXT}</p></article></body></html>"
SEARCH = {
    "results": [
        {"title": "Pogacar wins fifth Tour", "url": "https://cycling.example/pogacar?utm_source=x", "content": "Pogacar won.",
         "publishedDate": "2026-07-27T18:00:00", "engines": ["bing"]},
        {"title": "Duplicate", "url": "https://cycling.example/pogacar", "content": "dup"},
        {"title": "Blocked page", "url": "https://blocked.example/story", "content": "The Tour de France ended in Paris with Pogacar in yellow."},
        {"title": "Not http", "url": "ftp://x.example/file", "content": ""},
    ]
}


def handler(request: httpx.Request) -> httpx.Response:
    url = str(request.url)
    if url.startswith("http://searx.test/healthz"):
        return httpx.Response(200, text="OK")
    if url.startswith("http://searx.test/search"):
        assert request.url.params["format"] == "json"
        return httpx.Response(200, json=SEARCH)
    if url.startswith("https://cycling.example/"):
        return httpx.Response(200, text=PAGE, headers={"content-type": "text/html"})
    if url.startswith("https://blocked.example/"):
        return httpx.Response(403, text="no")
    raise httpx.ConnectError("no route", request=request)


@pytest.fixture
def client():
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_available(client):
    assert web.available("http://searx.test", client=client)
    assert not web.available("http://down.test", client=client)


def test_search_web_parses_and_dedupes(client):
    rs = web.search_web("tour de france", base_url="http://searx.test", client=client)
    assert [r.url for r in rs] == ["https://cycling.example/pogacar?utm_source=x", "https://blocked.example/story"]
    assert rs[0].published == "2026-07-27T18:00:00Z" and rs[0].engines == ("bing",)


@pytest.mark.parametrize(
    "response, match",
    [
        (httpx.Response(403, text="forbidden"), "403"),
        (httpx.Response(200, text="<html>not json</html>"), "JSON"),
    ],
)
def test_search_web_errors(response, match):
    c = httpx.Client(transport=httpx.MockTransport(lambda r: response))
    with pytest.raises(web.WebError, match=match):
        web.search_web("q", base_url="http://searx.test", client=c)
    with pytest.raises(web.WebError, match="can't reach"):
        web.search_web("q", base_url="http://down.test", client=httpx.Client(transport=httpx.MockTransport(handler)))


def test_web_context_fetches_and_ranks(client):
    ctx = web.web_context("Who won the Tour de France?", base_url="http://searx.test", client=client, embedder=FakeEmbedder())
    assert ctx.error is None and ctx.hits
    assert all(h.chunk_id < 0 for h in ctx.hits)  # never confused with stored chunks
    sources = {h.source for h in ctx.hits}
    assert "web: cycling.example" in sources
    # the blocked page falls back to its search snippet
    assert any(h.source == "web: blocked.example" and "yellow" in h.body for h in ctx.hits)
    assert len([h for h in ctx.hits if h.source == "web: cycling.example"]) <= web.MAX_PER_PAGE


def test_web_context_without_embedder_uses_overlap(client):
    ctx = web.web_context("Pogacar Tour de France", base_url="http://searx.test", client=client)
    assert ctx.hits and ctx.hits[0].source == "web: cycling.example"


def test_web_context_reports_errors():
    down = httpx.Client(transport=httpx.MockTransport(handler))
    ctx = web.web_context("q", base_url="http://down.test", client=down)
    assert ctx.hits == [] and "can't reach" in ctx.error
    empty = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"results": []})))
    assert web.web_context("q", base_url="http://searx.test", client=empty).error == "SearXNG returned no results"


def test_save_pages(conn, client):
    ctx = web.web_context("Tour de France", base_url="http://searx.test", client=client,
                          embedder=FakeEmbedder(), conn=conn, save=True)
    assert ctx.saved == 2
    rows = conn.execute("SELECT source, url FROM articles ORDER BY id").fetchall()
    assert [r["source"] for r in rows] == ["web: cycling.example", "web: blocked.example"]
    assert rows[0]["url"] == "https://cycling.example/pogacar"  # normalized
    assert store.vector_count(conn) == conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    again = web.web_context("Tour de France", base_url="http://searx.test", client=client, conn=conn, save=True)
    assert again.saved == 0  # idempotent like ingest
