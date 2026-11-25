"""MCP server (stdio): lets coding agents and other MCP clients search the local news index.

Tools:
  search_news(query, k=5, mode="hybrid")  -> the top hits as plain text: title, outlet, date, URL, excerpt
  latest_digest()                          -> today's top stories (headlines with their outlets)

The client does its own answering; Zands only retrieves. Every search is
written to retrieval_log with command 'mcp'. Run it with `zands --db /abs/path/news.db mcp`.
Logging goes to stderr: stdout carries the protocol.

Written for small local models as much as for big ones: the descriptions say when to use the
tools (a 7B model mostly doesn't guess), and results are short plain text rather than JSON
(about 1,000 tokens for five hits instead of 5,000), so the answer can quote them.
"""

from __future__ import annotations

from datetime import datetime, timezone

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

from . import digest as digestlib
from . import store
from .embed import OllamaEmbedder
from .search import MODES, retrieve

DEFAULT_K = 5
MAX_K = 10
EXCERPT_CHARS = 700

INSTRUCTIONS = (
    "Zands is a local index of recent news articles, fetched every hour from news outlets' RSS "
    "feeds. Your own knowledge stops at your training date, so it is out of date about current "
    "events. Before answering any question about news, recent or ongoing events, or what is "
    "happening now with a person, company, country or product, call search_news with a few "
    "keywords from the question. For an overview ('what's in the news', 'top stories today'), "
    "call latest_digest. Prefer these over a general web search for news. Answer only from what "
    "they return, and name the outlet and date of each fact you use. If nothing relevant comes "
    "back, say that the local news index has nothing on it rather than guessing."
)

# Both tools only read the local index. (Searches append to retrieval_log and a missing digest
# gets built, but that is Zands' own bookkeeping, not a change to anything the client owns.)
# The hint lets clients such as zwans run them without asking each time.
READ_ONLY = ToolAnnotations(read_only_hint=True, destructive_hint=False, open_world_hint=False)


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


def build_server(db_path: str, embedder=None) -> MCPServer:
    server = MCPServer(name="zands", title="Zands news", instructions=INSTRUCTIONS, version="0.4.0")
    embed = embedder if embedder is not None else OllamaEmbedder()

    def connect():
        return store.connect(db_path)

    @server.tool(title="Search news", annotations=READ_ONLY)
    def search_news(query: str, k: int = DEFAULT_K, mode: str = "hybrid") -> str:
        """Search recent news articles (updated hourly) about current events, people, companies, countries or products. Use it before answering any question about something recent or ongoing: your own knowledge is out of date.

        Args:
            query: a few keywords or a short question, e.g. "Nvidia CEO Huang reinstated"
            k: how many articles to return (1-10, default 5)
            mode: leave as "hybrid"; "keyword" or "vector" only to compare retrievers
        """
        if mode not in MODES:
            mode = "hybrid"  # small models invent values ("semantic", "news"): don't fail the call
        try:
            k = max(1, min(int(k), MAX_K))
        except (TypeError, ValueError):
            k = DEFAULT_K
        conn = connect()
        try:
            r = retrieve(conn, query, k=k, mode=mode, embedder=embed if mode != "keyword" else None, command="mcp")
            newest = conn.execute("SELECT MAX(fetched_at) FROM articles").fetchone()[0]
        finally:
            conn.close()
        return format_hits(query, r.hits, newest, r.note)

    @server.tool(title="Today's top stories", annotations=READ_ONLY)
    def latest_digest() -> str:
        """Today's top news stories: headlines grouped across outlets, ranked by how many cover them. Use it for "what's in the news", "top stories" or "headlines today"."""
        conn = connect()
        try:
            row = digestlib.latest(conn, max_age_hours=None)
            text = digestlib.refresh(conn).text if row is None else row["text"]  # never built: build it now
        finally:
            conn.close()
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return f"(Today is {today}.)\n{text}"

    return server


def run(db_path: str) -> None:
    build_server(db_path).run("stdio")
