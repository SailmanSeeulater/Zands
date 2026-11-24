"""MCP server (stdio): lets coding agents and other MCP clients search the local news index.

Tools:
  search_news(query, k=8, mode="hybrid")  -> the top chunks with title, source, date, URL and text
  latest_digest()                          -> today's top stories (headlines with their outlets)

The client does its own answering; Zands only retrieves. Every search is
written to retrieval_log with command 'mcp'. Run it with `zands --db /abs/path/news.db mcp`.
Logging goes to stderr: stdout carries the protocol.
"""

from __future__ import annotations

from mcp.server.mcpserver import MCPServer

from . import digest as digestlib
from . import store
from .embed import OllamaEmbedder
from .search import DEFAULT_K, MODES, retrieve

INSTRUCTIONS = (
    "Zands is a local index of recent news articles fetched hourly from RSS feeds. "
    "Use search_news for questions about current events; cite the returned URLs and dates. "
    "Use latest_digest for an overview of today's top stories. Results only cover what the "
    "feeds published, so say so when nothing relevant comes back."
)


def build_server(db_path: str, embedder=None) -> MCPServer:
    server = MCPServer(name="zands", title="Zands news", instructions=INSTRUCTIONS, version="0.3.0")
    embed = embedder if embedder is not None else OllamaEmbedder()

    def connect():
        return store.connect(db_path)

    @server.tool(title="Search news")
    def search_news(query: str, k: int = DEFAULT_K, mode: str = "hybrid") -> dict:
        """Search the local news index for chunks relevant to a question or keywords.

        Args:
            query: a natural-language question or keywords, e.g. "Why was Nvidia's CEO reinstated?"
            k: number of chunks to return (1-20, default 8)
            mode: "hybrid" (default: keywords + meaning), "keyword" or "vector"
        """
        if mode not in MODES:
            raise ValueError(f"mode must be one of {', '.join(MODES)}")
        k = max(1, min(int(k), 20))
        conn = connect()
        try:
            r = retrieve(conn, query, k=k, mode=mode, embedder=embed if mode != "keyword" else None, command="mcp")
            newest = conn.execute("SELECT MAX(fetched_at) FROM articles").fetchone()[0]
        finally:
            conn.close()
        return {
            "mode": r.mode,
            "note": r.note,
            "index_last_updated": newest,
            "results": [
                {
                    "title": h.title,
                    "source": h.source,
                    "published_at": h.published_at,
                    "url": h.url,
                    "text": h.body,
                    "score": round(h.score, 5),
                    "chunk_id": h.chunk_id,
                }
                for h in r.hits
            ],
        }

    @server.tool(title="Today's top stories")
    def latest_digest() -> str:
        """Today's top news stories: headlines grouped across outlets, ranked by how many cover them."""
        conn = connect()
        try:
            row = digestlib.latest(conn, max_age_hours=None)
            if row is None:  # never built yet: build it now
                return digestlib.refresh(conn).text
            return row["text"]
        finally:
            conn.close()

    return server


def run(db_path: str) -> None:
    build_server(db_path).run("stdio")
