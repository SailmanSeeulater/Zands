import anyio
import pytest
from fakes import FakeEmbedder
from mcp.client import Client
from test_apis import FakeFetch

from zands import store
from zands.embed import embed_pending
from zands.freshness import Refresh
from zands.mcp_server import build_server


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


def no_refresh(conn, db_path, embedder=None):
    return Refresh(False, "")


def call(db_path, tool, args, fetch=None, refresh=no_refresh):
    async def go():
        async with Client(build_server(db_path, embedder=FakeEmbedder(), fetch=fetch, refresh=refresh)) as c:
            listed = await c.list_tools()
            res = await c.call_tool(tool, args)
            return listed, res

    return anyio.run(go)


def test_lists_tools(db_path):
    listed, _ = call(db_path, "latest_digest", {})
    tools = {t.name: t for t in listed.tools}
    assert set(tools) == {
        "search_news", "latest_digest", "weather", "convert_currency", "wikipedia", "define_word",
        "public_holidays", "earthquakes", "arxiv", "books", "find_api", "call_api",
    }
    assert set(tools["search_news"].input_schema["properties"]) == {"query", "k", "mode"}
    assert set(tools["convert_currency"].input_schema["required"]) == {"amount", "from_currency", "to_currency"}


def test_only_call_api_asks_first(db_path):
    # Clients such as zwans run read-only MCP tools without asking; call_api goes to any URL the
    # model picks, so it is left for the client to ask about, like its own web fetches.
    listed, _ = call(db_path, "latest_digest", {})
    for t in listed.tools:
        assert t.annotations is not None and t.annotations.destructive_hint is False, t.name
        assert t.annotations.read_only_hint is (t.name != "call_api"), t.name


def test_live_tools_go_through_the_fetch_they_are_given(db_path):
    fetch = FakeFetch({"frankfurter": {"date": "2026-11-24", "rates": {"EUR": 0.9}}})
    _, res = call(db_path, "convert_currency", {"amount": 10, "from_currency": "USD", "to_currency": "EUR"}, fetch=fetch)
    assert res.content[0].text.startswith("10 USD = 9.00 EUR")


def test_null_numbers_fall_back_to_defaults(db_path):
    # qwen2.5-coder sent {"min_magnitude": null}; that used to fail validation.
    fetch = FakeFetch({"earthquake.usgs.gov": {"features": []}})
    _, res = call(db_path, "earthquakes", {"min_magnitude": None, "period": None}, fetch=fetch)
    assert not res.is_error and "4.5" in res.content[0].text and "summary/4.5_day" in fetch.calls[0][0]


def test_call_api_reports_refusals_as_text(db_path):
    _, res = call(db_path, "call_api", {"url": "https://127.0.0.1/admin"})
    assert not res.is_error and res.content[0].text.startswith("Request refused or failed:")


def test_find_api_uses_the_catalog(db_path):
    conn = store.connect(db_path)
    conn.execute("INSERT INTO api_catalog(name, url, description, category, auth, https, cors) "
                 "VALUES ('Cat Facts', 'https://catfact.ninja/', 'Daily cat facts', 'Animals', '', 1, 'yes')")
    conn.commit()
    _, res = call(db_path, "find_api", {"need": "cat facts"})
    assert "Cat Facts [Animals]" in res.content[0].text


def test_a_stale_index_is_refreshed_before_searching_and_the_model_told(db_path):
    seen = []

    def refresh(conn, path, embedder=None):
        seen.append(path)
        return Refresh(True, "news was 30h old, so feeds were fetched just now: 4 new articles")

    _, res = call(db_path, "search_news", {"query": "eruption"}, refresh=refresh)
    assert seen == [db_path]
    assert res.content[0].text.startswith("(news was 30h old, so feeds were fetched just now: 4 new articles)\n")


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
