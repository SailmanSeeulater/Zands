# Zands

A local news RAG that runs on one laptop. Zands reads RSS feeds, stores the full text of each
article in a single SQLite file, and answers questions about current events with a local
[Ollama](https://ollama.com) model. The model answers only from retrieved articles and cites them.

The point is to give local, open-weight models knowledge of today's news without web search
or fine-tuning: fetch on a schedule, search locally, ground the answer in what was fetched.

```
feeds.toml ──► ingest ──► data/news.db ──► search (FTS5 / BM25) ──► ask (Ollama) ──► cited answer
               feedparser   articles          question → safe query     prompt with today's date
               httpx        chunks            + mild recency boost       + numbered sources
               trafilatura  feeds (health)    every retrieval logged
                            retrieval_log
```

## Setup

Requirements: macOS or Linux, [uv](https://docs.astral.sh/uv/), and Ollama running locally.

```bash
uv sync                      # creates .venv with Python 3.12 (uv-managed, not /usr/bin/python3)
ollama pull qwen2.5:latest   # default answer model
```

Use uv's Python, not macOS system Python: the system build often can't load SQLite extensions,
and phase 2 needs `sqlite-vec`.

## Commands

```bash
uv run zands ingest                  # fetch all feeds, store new articles (safe to re-run)
uv run zands search "nvidia ceo"     # show the chunks a question retrieves, with scores
uv run zands ask "What happened with Nvidia's CEO?"   # cited answer from a local model
uv run zands stats                   # article/chunk counts and per-feed health
```

Useful flags:

| Command | Flag | Meaning |
|---|---|---|
| all | `--db PATH` | database file (env `ZANDS_DB`, default `data/news.db`) |
| all | `-v` | debug logging (shows per-article fallbacks) |
| `ingest` | `--feeds PATH` | feeds file (env `ZANDS_FEEDS`, default `feeds.toml`) |
| `ingest` | `--limit N` | at most N entries per feed |
| `ingest` | `--concurrency N` | parallel page fetches (default 4) |
| `search`, `ask` | `-k N` | chunks to retrieve (default 8) |
| `search`, `ask` | `--recency W` | recency boost weight, `0` disables (default 0.3) |
| `ask` | `--model NAME` | Ollama model (env `ZANDS_MODEL`, default `qwen2.5:latest`) |
| `ask` | `--ollama-url URL` | Ollama base URL (env `OLLAMA_HOST`, default `http://localhost:11434`) |
| `ask` | `--num-ctx N` | model context window (default 16384; Ollama's own default is too small for 8 chunks) |
| `ask` | `--show-prompt` | print the full prompt sent to the model |

Paths are relative to the current directory, so run commands from the repo root (or set
`ZANDS_DB` / `ZANDS_FEEDS` to absolute paths).

## How it works

**Feeds.** `feeds.toml` lists `[[feed]]` entries with a `name` and `url` (`enabled = false`
skips one). It ships with ten public general and tech news feeds.

**Ingest** (`src/zands/ingest.py`). For each feed: download it with httpx, parse it with
feedparser, skip entries already stored, fetch each new article page (4 at a time, 20 s
timeout, generic User-Agent), extract the main text with trafilatura, and fall back to the
feed's own summary/content when the page is blocked, paywalled or too thin.
- *Idempotent.* The normalized article URL is a unique key: lowercase host, `https`, no fragment,
  no tracking parameters (`utm_*`, `at_medium`, `maca`, `traffic_source`, ...), sorted query, no
  trailing slash. Known URLs are skipped before any page fetch, and the insert is
  `ON CONFLICT DO NOTHING`, so re-running never duplicates.
- *Errors are per feed.* A feed that fails (HTTP error, timeout, unparseable XML) is logged and
  recorded in the `feeds` table (`last_error`, `last_error_at`, consecutive `failure_count`), and
  the run continues. A success resets `failure_count` and sets `last_success_at`. The command
  exits non-zero only if *every* feed failed.

**Chunking** (`src/zands/chunk.py`). Greedy by paragraph, up to 800 words per chunk; the next
chunk starts with the last sentence(s) of the previous one (up to 40 words) so facts on a
boundary stay findable. A short tail (under 300 words) is merged back when it fits. Oversized
paragraphs are split by sentence. Each chunk row carries the article title, source, URL and
published date.

**Storage** (`src/zands/store.py`). One SQLite file, WAL mode:

| Table | Holds |
|---|---|
| `feeds` | name, URL, last success, last error, failure count, new items last run |
| `articles` | normalized URL (unique), original URL, title, source, published/fetched time, summary, full text, extraction method (`trafilatura` or `rss`) |
| `chunks` | chunk text and word count, plus a copy of the article metadata |
| `chunks_fts` | FTS5 external-content index over chunk title + body (porter stemming), kept in sync by triggers |
| `retrieval_log` | every search: time, command, question, FTS query actually run, k, chunk ids in rank order |

**Search** (`src/zands/search.py`). The question is lowercased and split into words; stopwords
and question filler ("latest", "today", "what happened") are dropped; each remaining word is
double-quoted and the words are OR-ed, so FTS5 syntax in the input can't break or change the
query. Results are ranked by BM25 (title matches weighted 3×), then multiplied by a mild recency
boost (up to +30% for a brand-new article, halving every 3 days). At most 2 chunks per article
make the top k. Every retrieval, including empty ones, goes to `retrieval_log`:

```bash
sqlite3 data/news.db "SELECT created_at, question, query, chunk_ids FROM retrieval_log ORDER BY id DESC LIMIT 5"
```

**Answering** (`src/zands/answer.py`). The prompt holds today's date, the numbered chunks (each
with title, source, published date and URL) and the rules: answer only from these sources, cite
them as `[n]`, say plainly when they don't cover the question. The reply is streamed from
Ollama's `/api/chat`, followed by the numbered source list. If nothing matches, Zands says so
without calling the model.

## Tests

```bash
uv run pytest
```

Covers the chunker, the query sanitizer (including hostile FTS5 input), URL normalization and
idempotent inserts, FTS search on a tiny fixture database (stemming, recency, per-article cap,
retrieval logging), ingest against a fake HTTP transport (per-feed error isolation, RSS
fallback, re-run idempotency) and the prompt builder. No test touches the network.

## Roadmap

**Phase 1: keyword RAG (done).** Feeds → SQLite + FTS5 → BM25 search → cited answers via Ollama.

**Next up: browsable database.** [Datasette](https://datasette.io) as a web UI over
`data/news.db`: browse articles, chunks, feed health and `retrieval_log`, and run SQL in the
browser. Works today as `uvx datasette data/news.db`; later, wrap it as a `zands ui` command.

**Phase 2: hybrid search, fact overrides and scheduling.**
- Fact overrides: a small `facts` table (`subject`, `attribute`, `value`, `valid_from`,
  `valid_to` (NULL while still current), `source_url`, `added_at`) managed with
  `zands fact add/list/expire`. `zands ask` puts the relevant current facts at the top of the
  prompt as authoritative, dated corrections that win over the model's stale training knowledge.
  It's a new table, so `store.connect()`'s `CREATE TABLE IF NOT EXISTS` adds it to existing
  databases without a migration.
- `sqlite-vec` in the same database file, with `nomic-embed-text` embeddings from Ollama.
  nomic needs task prefixes: `search_document: ` when embedding chunks, `search_query: ` when
  embedding questions.
- Hybrid retrieval: fuse the BM25 and vector rankings with Reciprocal Rank Fusion.
- A launchd job (not cron: cron skips runs while the laptop sleeps; launchd runs missed jobs
  on wake) to ingest every hour or so.
- A 20-question known-answer test set, built before tuning anything, to measure whether a
  retrieval change actually helps.

**Phase 3: routing, web fallback, quality and integration.**
- Query router: one cheap classification step before retrieval (rules first, then a small
  local model such as `qwen3.5:0.8b` or `qwen2.5`) that picks a route: local news search,
  fact overrides, a structured API (weather, prices, ... later), the model's own knowledge
  (stable, timeless questions), the SearXNG web fallback, or "I don't know". The chosen route
  is logged in `retrieval_log` (a new `route` column) so misroutes can be reviewed. This fixes
  "the model didn't realize it needed fresh info".
- SearXNG web fallback: a self-hosted metasearch engine (local Docker, free, private, no paid
  search API). When the router picks it, or local retrieval finds nothing relevant, `zands ask`
  queries SearXNG's JSON API, fetches the top results with trafilatura and answers from them,
  labeling those sources as web results rather than indexed news. Optionally ingest the
  fetched pages into `news.db` so the next question is answered locally.
- A reranker over the fused candidates.
- A daily digest pushed into the system prompt, so the model knows the day's headlines without searching.
- An MCP server exposing `search_news` and `latest_digest`, so the zwans coding agent and other
  MCP clients can use Zands.
