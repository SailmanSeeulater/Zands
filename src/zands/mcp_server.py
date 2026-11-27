"""MCP server (stdio): current information for coding agents and other MCP clients.

News (the local index, fetched on demand when older than ZANDS_STALE_HOURS):
  search_news(query, k=5, mode="hybrid")   the top hits as plain text: title, outlet, date, URL, excerpt
  latest_digest()                           today's top stories (headlines with their outlets)
Live data (free public APIs, no keys; see apis/tools.py):
  weather, convert_currency, wikipedia, define_word, public_holidays, earthquakes, arxiv, books
Anything else (the public-apis list, see apis/catalog.py):
  find_api(need)     no-key APIs for a need, with their docs URLs
  call_api(url)      a guarded HTTPS GET (public hosts only); not marked read-only, so clients ask

The client does its own reasoning and answering; Zands only fetches. News searches are written
to retrieval_log with command 'mcp'. Run it with `zands --db /abs/path/news.db mcp`. Logging
goes to stderr: stdout carries the protocol.

Written for small local models as much as for big ones: the instructions give a short way to
work when information is missing, each tool's first sentence says when to use it, and results
are short plain text rather than JSON, so the answer can quote them.
"""

from __future__ import annotations

from datetime import UTC, datetime

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

from . import digest as digestlib
from . import store
from .apis import catalog, http
from .apis import tools as live
from .embed import OllamaEmbedder
from .freshness import refresh_if_stale
from .search import MODES, retrieve

DEFAULT_K = 5
MAX_K = 10
EXCERPT_CHARS = 700

INSTRUCTIONS = """\
Zands gives you current information your training data lacks: recent news, and live data from \
free public APIs. Your own knowledge is out of date, so don't answer from memory when the \
answer could have changed.
When a question needs information you don't have:
1. Decide exactly what is missing (a fact, a number, a date, recent events).
2. Call the most specific tool: news, recent or ongoing events -> search_news (overview: \
latest_digest); weather, local time, air quality -> weather; currencies and crypto -> \
convert_currency; background on a person, place, company or idea -> wikipedia; word meanings -> \
define_word; holidays -> public_holidays; earthquakes -> earthquakes; research papers -> arxiv; \
books -> books.
3. If no tool fits, call find_api with the need in a few words, read the chosen API's docs with \
call_api, then call its endpoint with call_api.
4. Check the result answers the question. If not, try other keywords or another tool (up to 3 \
tries). For a question with parts, get every part from a tool (two cities: weather "Bangkok; Berlin"); never fill in a part yourself.
5. Answer from the results and name each source and date. If nothing worked, say what you tried \
and that you couldn't find it. Don't guess."""

# Reads only: the local index (searches append to retrieval_log and a stale index is refreshed,
# which is Zands' own bookkeeping) or fixed public APIs. Clients such as zwans run these
# without asking each time.
READ_ONLY = ToolAnnotations(read_only_hint=True, destructive_hint=False, open_world_hint=False)
READ_ONLY_WEB = ToolAnnotations(read_only_hint=True, destructive_hint=False, open_world_hint=True)
# call_api goes wherever the model points it, so it isn't marked read-only: clients ask first,
# as they do for their own web fetches.
ANY_URL = ToolAnnotations(read_only_hint=False, destructive_hint=False, open_world_hint=True)


def _num(value, default, cast=int):
    """A number argument as given, or its default when a small model sends null or text."""
    try:
        return default if value is None else cast(value)
    except (TypeError, ValueError):
        return default


def _excerpt(text: str, limit: int = EXCERPT_CHARS) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    cut = text.rfind(" ", 0, limit)
    return text[: cut if cut > limit // 2 else limit] + " …"


def format_hits(query: str, hits, newest: str | None, note: str | None = None) -> str:
    """The hits as numbered plain text a small model can read and quote."""
    if not hits:
        return (
            f'No news articles in the local index match "{query}". Say so; don\'t answer from '
            f"memory as if it were current. (Index last updated {newest or 'never'}.)"
        )
    lines = [f'News articles matching "{query}" (index last updated {newest}):']
    if note:
        lines.append(f"(note: {note})")
    for i, h in enumerate(hits, 1):
        date = (h.published_at or "date unknown")[:10]
        lines.append(f"\n[{i}] {h.title}\n{h.source}, {date} | {h.url}\n{_excerpt(h.body)}")
    lines.append("\nCite the outlet and date for each fact you use from these.")
    return "\n".join(lines)


def build_server(db_path: str, embedder=None, fetch=None, refresh=refresh_if_stale) -> MCPServer:
    """`fetch` replaces http.get_json for the live-data tools (tests); `refresh` the news refresh."""
    server = MCPServer(name="zands", title="Zands: news and live data", instructions=INSTRUCTIONS, version="0.5.0")
    embed = embedder if embedder is not None else OllamaEmbedder()
    get_json = fetch or http.get_json
    get_raw = (lambda url, params=None, **kw: fetch(url, params, raw=True, **kw)) if fetch else http.get

    def connect():
        return store.connect(db_path)

    def fresh(conn) -> str:
        r = refresh(conn, db_path, embedder=embed)
        return f"({r.note})\n" if r.note else ""

    # ------------------------------------------------------------ news

    @server.tool(title="Search news", annotations=READ_ONLY)
    def search_news(query: str, k: int | None = DEFAULT_K, mode: str | None = "hybrid") -> str:
        """Search recent news articles about current events, people, companies, countries or products. Use it before answering any question about something recent or ongoing: your own knowledge is out of date.

        Args:
            query: a few keywords or a short question, e.g. "Nvidia CEO Huang reinstated"
            k: how many articles to return (1-10, default 5)
            mode: leave as "hybrid"; "keyword" or "vector" only to compare retrievers
        """
        if mode not in MODES:
            mode = "hybrid"  # small models invent values ("semantic", "news"): don't fail the call
        k = max(1, min(_num(k, DEFAULT_K), MAX_K))
        conn = connect()
        try:
            note = fresh(conn)
            r = retrieve(conn, query, k=k, mode=mode, embedder=embed if mode != "keyword" else None, command="mcp")
            newest = conn.execute("SELECT MAX(fetched_at) FROM articles").fetchone()[0]
        finally:
            conn.close()
        return note + format_hits(query, r.hits, newest, r.note)

    @server.tool(title="Today's top stories", annotations=READ_ONLY)
    def latest_digest() -> str:
        """Today's top news stories: headlines grouped across outlets, ranked by how many cover them. Use it for "what's in the news", "top stories" or "headlines today"."""
        conn = connect()
        try:
            note = fresh(conn)
            row = digestlib.latest(conn, max_age_hours=None)
            text = digestlib.refresh(conn).text if row is None or note else row["text"]
        finally:
            conn.close()
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        return f"(Today is {today}.)\n{note}{text}"

    # ------------------------------------------------------------ live data

    @server.tool(title="Weather", annotations=READ_ONLY_WEB)
    def weather(place: str, days: int | None = 3) -> str:
        """Current weather, forecast, local time and air quality for a city or place.

        Args:
            place: a city, optionally with its country or state: "Tokyo", "Paris, France", "Springfield, Illinois".
                To compare places, name them all in one call, separated by ";": "Bangkok; Berlin"
            days: forecast days, 1-7 (default 3)
        """
        return live.weather(place, _num(days, 3), fetch=get_json)

    @server.tool(title="Convert currency", annotations=READ_ONLY_WEB)
    def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
        """Convert money between currencies, or price a cryptocurrency, at today's rate.

        Args:
            amount: how much, e.g. 100
            from_currency: ISO code like USD, EUR, THB, or a crypto symbol like BTC, ETH
            to_currency: ISO code or crypto symbol
        """
        return live.convert_currency(amount, from_currency, to_currency, fetch=get_json)

    @server.tool(title="Wikipedia", annotations=READ_ONLY_WEB)
    def wikipedia(topic: str) -> str:
        """Background on a person, place, company, event or idea: the summary of its Wikipedia article.

        Args:
            topic: the name to look up, e.g. "Jensen Huang", "Canberra"
        """
        return live.wikipedia(topic, fetch=get_json)

    @server.tool(title="Define a word", annotations=READ_ONLY_WEB)
    def define_word(word: str) -> str:
        """English definitions and pronunciation of a word."""
        return live.define_word(word, fetch=get_json)

    @server.tool(title="Public holidays", annotations=READ_ONLY_WEB)
    def public_holidays(country: str, year: int | None = 0) -> str:
        """National public holidays of a country in a year; the first line says which one is next.

        Args:
            country: country name or two-letter code, e.g. "Germany" or "DE"
            year: the year; 0 for this year
        """
        return live.public_holidays(country, _num(year, 0) or None, fetch=get_json)

    @server.tool(title="Earthquakes", annotations=READ_ONLY_WEB)
    def earthquakes(min_magnitude: float | None = 4.5, period: str | None = "day") -> str:
        """Recent earthquakes worldwide, strongest first (USGS).

        Args:
            min_magnitude: smallest magnitude to include (default 4.5)
            period: "hour", "day", "week" or "month"
        """
        return live.earthquakes(_num(min_magnitude, 4.5, float), period or "day", fetch=get_json)

    @server.tool(title="arXiv papers", annotations=READ_ONLY_WEB)
    def arxiv(query: str, max_results: int | None = 5) -> str:
        """Research papers on arXiv matching some keywords, with authors, date and abstract.

        Args:
            query: keywords, e.g. "retrieval augmented generation"
            max_results: 1-10 (default 5)
        """
        return live.arxiv(query, _num(max_results, 5), fetch_text=get_raw)

    @server.tool(title="Books", annotations=READ_ONLY_WEB)
    def books(query: str, max_results: int | None = 5) -> str:
        """Books matching a title, author or subject, with first publication year (Open Library).

        Args:
            query: title, author or subject, e.g. "dune herbert"
            max_results: 1-10 (default 5)
        """
        return live.books(query, _num(max_results, 5), fetch=get_json)

    # ------------------------------------------------------------ anything else

    @server.tool(title="Find an API", annotations=READ_ONLY)
    def find_api(need: str) -> str:
        """Find a free public API for information no other tool covers, from a catalog of about 1,400 (the public-apis list). Returns names, what they do and their docs URLs.

        Args:
            need: what you need in a few words, e.g. "flight status", "train times germany", "pokemon"
        """
        conn = connect()
        try:
            return catalog.format_found(need, catalog.find(conn, need), catalog.count(conn))
        finally:
            conn.close()

    @server.tool(title="Call an API", annotations=ANY_URL)
    def call_api(url: str) -> str:
        """GET a public HTTPS URL and return its content as text: an API endpoint found with find_api, or its docs page to learn the endpoint. Put parameters in the URL's query string.

        Args:
            url: the full https URL, e.g. "https://catfact.ninja/fact"
        """
        try:
            got = get_raw(url.strip())
        except http.FetchError as exc:
            return f"Request refused or failed: {exc}"
        head = f"GET {got.url} -> HTTP {got.status} ({got.content_type or 'no content type'})"
        return head + "\n" + http.readable(got)

    return server


def run(db_path: str) -> None:
    build_server(db_path).run("stdio")
