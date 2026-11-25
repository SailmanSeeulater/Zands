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


def test_tools_are_marked_read_only(db_path):
    # Clients such as zwans run read-only MCP tools without asking; without the hint they block.
    listed, _ = call(db_path, "latest_digest", {})
    for t in listed.tools:
        assert t.annotations is not None and t.annotations.read_only_hint is True, t.name
        assert t.annotations.destructive_hint is False, t.name


def test_search_news(db_path):
    _, res = call(db_path, "search_news", {"query": "magma eruption", "k": 1})
    assert not res.is_error
    text = res.content[0].text
    assert "[1] Iceland evacuates town\nN, 2099-01-01 | https://n.example/0\n" in text
    assert "Grindavik" in text and "[2]" not in text
    assert "Cite the outlet and date" in text
    conn = store.connect(db_path)
    assert conn.execute("SELECT command FROM retrieval_log").fetchone()[0] == "mcp"


def test_search_news_tolerates_made_up_arguments(db_path):
    # Small models invent modes and pass k as text; the call should still work.
    _, res = call(db_path, "search_news", {"query": "interest rates", "mode": "semantic", "k": "3"})
    assert not res.is_error and "Bank holds rates" in res.content[0].text


def test_search_news_says_plainly_when_nothing_matches(db_path):
    _, res = call(db_path, "search_news", {"query": "zebra crossing", "mode": "keyword"})
    assert "No news articles in the local index match" in res.content[0].text


def test_excerpts_are_short_and_cut_at_a_word():
    from zands.mcp_server import _excerpt

    text = " ".join(["word"] * 400)
    out = _excerpt(text, 100)
    assert len(out) <= 102 and out.endswith(" …") and "wor …" not in out
    assert _excerpt("short  text\n here") == "short text here"


def test_instructions_say_when_to_use_the_tools(db_path):
    from zands.mcp_server import INSTRUCTIONS

    assert "search_news" in INSTRUCTIONS and "latest_digest" in INSTRUCTIONS
    listed, _ = call(db_path, "latest_digest", {})
    desc = {t.name: t.description for t in listed.tools}
    assert "Use it before answering" in desc["search_news"]


def test_latest_digest_builds_on_first_use(db_path):
    _, res = call(db_path, "latest_digest", {})
    text = res.content[0].text
    assert text.startswith("(Today is ") and "Top stories" in text and "Iceland evacuates town" in text
