import json

import anyio
import pytest
from mcp.client import Client

from zands import store
from zands.embed import embed_pending
from zands.mcp_server import build_server

from fakes import FakeEmbedder


@pytest.fixture
def db_path(tmp_path):
    path = tmp_path / "news.db"
    conn = store.connect(path)
    for i, (title, body) in enumerate([("Iceland evacuates town", "Lava flowed after the eruption near Grindavik."),
                                       ("Bank holds rates", "The central bank kept interest rates unchanged.")]):
        store.insert_article(conn, store.Article(url=f"https://n.example/{i}", title=title, source="N", body=body,
                                                 extraction="rss", published_at="2099-01-01T00:00:00Z"), [body])
    embed_pending(conn, FakeEmbedder())
    conn.close()
    return str(path)


def call(db_path, tool, args):
    async def go():
        async with Client(build_server(db_path, embedder=FakeEmbedder())) as c:
            listed = await c.list_tools()
            res = await c.call_tool(tool, args)
            return listed, res

    return anyio.run(go)


def payload(res):
    if getattr(res, "structured_content", None):
        return res.structured_content
    return json.loads(res.content[0].text)


def test_lists_tools(db_path):
    listed, _ = call(db_path, "latest_digest", {})
    tools = {t.name: t for t in listed.tools}
    assert set(tools) == {"search_news", "latest_digest"}
    assert set(tools["search_news"].input_schema["properties"]) == {"query", "k", "mode"}


def test_search_news(db_path):
    _, res = call(db_path, "search_news", {"query": "magma eruption", "k": 1})
    data = payload(res)
    assert data["mode"] == "hybrid"
    [hit] = data["results"]
    assert hit["title"] == "Iceland evacuates town" and hit["url"] == "https://n.example/0"
    assert "Grindavik" in hit["text"] and hit["published_at"] == "2099-01-01T00:00:00Z"
    conn = store.connect(db_path)
    assert conn.execute("SELECT command FROM retrieval_log").fetchone()[0] == "mcp"


def test_search_news_validates_mode(db_path):
    _, res = call(db_path, "search_news", {"query": "x", "mode": "semantic"})
    assert res.is_error


def test_latest_digest_builds_on_first_use(db_path):
    _, res = call(db_path, "latest_digest", {})
    text = res.content[0].text
    assert text.startswith("Top stories") and "Iceland evacuates town" in text
