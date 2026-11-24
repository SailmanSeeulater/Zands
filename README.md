# Zands

A local news RAG that runs on one laptop. Zands reads RSS feeds, stores the full text of each
article in a single SQLite file, and answers questions about current events with a local
[Ollama](https://ollama.com) model. The model answers only from retrieved articles (and any fact
overrides you add) and cites them.

The point is to give local, open-weight models knowledge of today's news without web search
or fine-tuning: fetch on a schedule, search locally, ground the answer in what was fetched.

```
feeds.toml ─► ingest ─► data/news.db ───────────► retrieve ──────────────► ask (Ollama) ─► cited answer
 (launchd,    feedparser  articles, chunks         FTS5 BM25 ─┐              today's date
  hourly)     httpx       chunks_fts (FTS5)                    ├ RRF fusion   + verified facts first
              trafilatura chunk_vec (sqlite-vec)   sqlite-vec ─┘ + recency    + numbered sources
              nomic-embed facts, feeds, meta        (degrades to keyword
                          retrieval_log              if Ollama is down)
```

## Setup

Requirements: macOS (scheduling uses launchd; everything else also runs on Linux),
[uv](https://docs.astral.sh/uv/), and Ollama running locally.

```bash
uv sync                          # .venv with Python 3.12 (uv-managed, not /usr/bin/python3)
ollama pull qwen2.5:latest       # answer model
ollama pull nomic-embed-text     # embedding model
uv run zands ingest              # first fetch (also embeds the new chunks)
```

Use uv's Python, not macOS system Python: the system build often can't load SQLite extensions,
and vector search needs `sqlite-vec`. `uv run zands stats` shows `vectors: n/a (...)` if the
extension couldn't load; everything else keeps working on keyword search.

## Commands

```bash
uv run zands ingest                      # fetch feeds, store new articles, embed new chunks (safe to re-run)
uv run zands embed                       # backfill vectors for chunks that have none
uv run zands search "nvidia ceo"         # show retrieved chunks, with keyword/vector ranks
uv run zands ask "What happened with Nvidia's CEO?"   # cited answer from a local model
uv run zands fact add Nvidia CEO "Jensen Huang" --from 2026-11-23 --source https://...
uv run zands fact list                   # current facts (--all includes expired)
uv run zands fact expire 3               # no longer true from today (--on YYYY-MM-DD)
uv run zands eval                        # known-answer set: keyword vs hybrid hit rates
uv run zands ui                          # Datasette on http://127.0.0.1:8001/
uv run zands schedule install            # hourly ingest via launchd (status / uninstall)
uv run zands stats                       # counts and per-feed health
```

Useful flags:

| Command | Flag | Meaning |
|---|---|---|
| all | `--db PATH` | database file (env `ZANDS_DB`, default `data/news.db`) |
| all | `-v` | debug logging (shows per-article fallbacks) |
| `ingest` | `--feeds PATH` | feeds file (env `ZANDS_FEEDS`, default `feeds.toml`) |
| `ingest` | `--limit N` / `--concurrency N` | entries per feed / parallel page fetches (default 4) |
| `ingest` | `--no-embed` | skip embedding (vectors get backfilled by a later run or `zands embed`) |
| `embed` | `--rebuild` | drop all vectors first (after switching embedding models) |
| `search`, `ask` | `--mode hybrid\|keyword\|vector` | retrieval mode (default `hybrid`) |
| `search`, `ask` | `-k N` | chunks to retrieve (default 8) |
| `search`, `ask` | `--recency W` | recency boost weight, `0` disables (default 0.3) |
| `ask` | `--model NAME` | answer model (env `ZANDS_MODEL`, default `qwen2.5:latest`) |
| `ask` | `--num-ctx N` | model context window (default 16384; Ollama's own default is too small for 8 chunks) |
| `ask` | `--no-facts` / `--show-prompt` | leave fact overrides out / print the full prompt |
| Ollama users | `--ollama-url URL` | Ollama base URL (env `OLLAMA_HOST`, default `http://localhost:11434`) |
| Ollama users | `--embed-model NAME` | embedding model (env `ZANDS_EMBED_MODEL`, default `nomic-embed-text`) |
| `eval` | `--mode compare\|hybrid\|keyword\|vector` | what to score (default `compare`: keyword vs hybrid) |
| `eval` | `--answers` / `--list` | also check model answers (slow) / per-question results |
| `ui` | `--host` / `--port` | bind address (default `127.0.0.1:8001`) |
| `schedule install` | `--interval MIN` / `--no-load` | minutes between runs (default 60) / write the plist only |

Paths are relative to the current directory, so run commands from the repo root (or set
`ZANDS_DB` / `ZANDS_FEEDS` to absolute paths). The launchd job always uses absolute paths.

## How it works

**Feeds.** `feeds.toml` lists `[[feed]]` entries with a `name` and `url` (`enabled = false`
skips one). It ships with ten public general and tech news feeds.

**Ingest** (`src/zands/ingest.py`). For each feed: download it with httpx, parse it with
feedparser, skip entries already stored, fetch each new article page (4 at a time, 20 s
timeout, generic User-Agent), extract the main text with trafilatura, and fall back to the
feed's own summary/content when the page is blocked, paywalled or too thin. Then embed every
chunk that has no vector yet.
- *Idempotent.* The normalized article URL is a unique key: lowercase host, `https`, no fragment,
  no tracking parameters (`utm_*`, `at_medium`, `maca`, `traffic_source`, ...), sorted query, no
  trailing slash. Known URLs are skipped before any page fetch, and the insert is
  `ON CONFLICT DO NOTHING`, so re-running never duplicates.
- *Errors are per feed.* A feed that fails (HTTP error, timeout, unparseable XML) is logged and
  recorded in the `feeds` table (`last_error`, `last_error_at`, consecutive `failure_count`), and
  the run continues. A success resets `failure_count` and sets `last_success_at`. The command
  exits non-zero only if *every* feed failed.
- *Embedding never fails an ingest.* If Ollama is down the articles are still stored and
  keyword-searchable; the next run embeds whatever is missing.

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
| `chunk_vec` | sqlite-vec `vec0` table: one 768-d cosine vector per chunk, deleted with its chunk by trigger |
| `facts` | fact overrides: subject, attribute, value, valid_from, valid_to, source_url, added_at |
| `retrieval_log` | every retrieval: time, command, question, FTS query, k, chunk ids in rank order, effective mode, facts used |
| `meta` | which embedding model made the stored vectors |

New columns are added to existing databases automatically on open; no manual migration.

**Embeddings** (`src/zands/embed.py`). `nomic-embed-text` through Ollama's `/api/embed`, in
batches of 32. nomic is trained with task prefixes, so chunks are embedded as
`search_document: <title>\n\n<body>` and questions as `search_query: <question>`. The model name
is recorded in `meta`; mixing models in one index is refused (`zands embed --rebuild` to switch).

**Retrieval** (`src/zands/search.py`).
- *Keyword:* the question is lowercased and split into words; stopwords and question filler
  ("latest", "today", "what happened") are dropped; each remaining word is double-quoted and
  OR-ed, so FTS5 syntax in the input can't break or change the query. Ranked by BM25 (title
  matches weighted 3×).
- *Vector:* cosine KNN over `chunk_vec`.
- *Hybrid (default):* the top 50 of each, fused with Reciprocal Rank Fusion
  (score = Σ 1/(60 + rank)).
- *Recency:* a mild boost (up to +30% for a brand-new article, halving every 3 days) applied
  inside each list before fusion. (Applying it to the fused score let recency swamp relevance,
  because neighbouring RRF scores differ by under 2%. The eval caught that.)
- At most 2 chunks per article make the top k.
- *Degrades, never crashes:* if Ollama is unreachable, sqlite-vec didn't load, or nothing is
  embedded yet, hybrid/vector fall back to keyword search, print a note, and log the mode as
  `keyword (fallback)`.

Every retrieval, including empty ones, goes to `retrieval_log`:

```bash
sqlite3 data/news.db "SELECT created_at, mode, question, query, chunk_ids FROM retrieval_log ORDER BY id DESC LIMIT 5"
```

**Fact overrides** (`src/zands/facts.py`). Hand-curated, dated facts for things the model
gets wrong because its training data is stale. Adding a new value for the same subject +
attribute closes the previous one (history is kept). A fact is put into an `ask` prompt when
every word of its subject appears in the question (plural/possessive forms tolerated), so
write subjects the way you'd ask about them ("Nvidia", "Bank of England").

**Answering** (`src/zands/answer.py`). The prompt holds today's date, then any relevant current
facts as `VERIFIED FACTS [F1]...`, marked as authoritative and overriding both training knowledge
and the news, then the numbered chunks (title, source, published date, URL), then the rules:
answer only from these, cite `[n]` / `[F1]`, say plainly when they don't cover the question. The
reply is streamed from Ollama's `/api/chat`, followed by the facts and sources used. With no
matching chunks and no facts, Zands says so without calling the model.

**Scheduling** (`src/zands/schedule.py`). `zands schedule install` writes
`~/Library/LaunchAgents/local.zands.ingest.plist` and loads it with `launchctl bootstrap`. The
agent runs this venv's Python directly (`python -m zands.cli --db <abs> ingest --feeds <abs>`)
every 60 minutes and once at load, at background priority, logging to
`~/Library/Logs/zands/ingest.log`. launchd rather than cron, because cron skips runs while the
laptop sleeps, and launchd runs a missed interval once on wake. `zands schedule status` shows
whether it's loaded, run count, last exit code and the log tail. `zands schedule uninstall`
unloads it and deletes the plist.

**Browsing** (`src/zands/ui.py`). `zands ui` runs [Datasette](https://datasette.io) over the
database with sqlite-vec loaded, read-only (no write plugins), plus canned queries: recent
retrievals with their top hit, questions that retrieved nothing, feed health, articles per
day, current facts.

## Evaluation

`eval/questions.toml` holds 20 known-answer questions about articles ingested on 2026-11-24:
10 that share words with the headline (`keyword` style) and 10 deliberately paraphrased
(`paraphrase` style). Each lists URL substrings of the article(s) that answer it, plus answer
keywords. `zands eval` reports, per mode, how often an expected article appears at rank 1, in
the top 3 and in the top k, plus mean reciprocal rank. Questions whose articles aren't in the
database are reported as missing, not as misses. `--answers` also asks the model and checks the
answer keywords.

Results on 2026-11-24 (348 articles, k = 8, default settings):

| Mode | hit@1 | hit@3 | hit@8 | MRR |
|---|---|---|---|---|
| keyword (BM25) | 80% | 95% | 100% | 0.87 |
| hybrid (BM25 + nomic, RRF) | 90% | 100% | 100% | 0.94 |
| vector only | 75% | 100% | 100% | 0.84 |

Answer check (hybrid retrieval, qwen2.5): 17 of 20 answers (85%) contained an expected keyword.
I checked the three failures by hand, and all three answers were correct. Two failed because
the keyword lists were too narrow: the articles say "$43.9 billion" and "at least 23", not "44"
and "20"; the lists are fixed now. The third was correct when re-asked, so answers vary
somewhat from run to run.

Both modes find every article somewhere in the top 8. Hybrid's gain is in putting it first.
The set is small and was written by the same session that built the system, so treat these as
a regression check, not a benchmark. Add questions as the news moves on, and write them before
looking at what retrieval returns.

## Tests

```bash
uv run pytest
```

No test touches the network: Ollama is replaced by a deterministic fake embedder, HTTP by
`httpx.MockTransport`, and `launchctl` by a recording stub. Covered:
- the chunker and the query sanitizer (including hostile FTS5 input)
- URL normalization and idempotent inserts
- keyword search on a fixture database (stemming, recency, per-article cap, retrieval logging)
- ingest: per-feed error isolation, RSS fallback, re-run idempotency, embedding at ingest, and
  surviving Ollama being down
- hybrid/vector retrieval: prefixes, RRF, recency-before-fusion, every fallback path, model
  mismatch, vector deletion
- facts: supersede, expire, relevance, prompt placement
- eval scoring, the launchd plist and commands, the Datasette command, and that every canned
  query runs

## Roadmap

**Phase 1: keyword RAG (done).** Feeds → SQLite + FTS5 → BM25 search → cited answers via Ollama.

**Phase 2: hybrid search, fact overrides, scheduling, browsing (done).** `zands ui` (Datasette);
sqlite-vec + nomic-embed-text with RRF hybrid retrieval and keyword fallback; `facts` table with
`zands fact add/list/expire`, put first in prompts; launchd hourly ingest with
`zands schedule install/uninstall/status`; 20-question known-answer set with `zands eval`.

**Phase 3: routing, web fallback, quality and integration.**
- Query router: one cheap classification step before retrieval (rules first, then a small
  local model such as `qwen3.5:0.8b` or `qwen2.5`) that picks a route: local news search,
  fact overrides, a structured API (weather, prices, ... later), the model's own knowledge
  (stable, timeless questions), the SearXNG web fallback, or "I don't know". The chosen route
  is logged in `retrieval_log` (a new `route` column, added like `mode` was) so misroutes can be
  reviewed. This fixes "the model didn't realize it needed fresh info".
- SearXNG web fallback: a self-hosted metasearch engine (local Docker, free, private, no paid
  search API). When the router picks it, or local retrieval finds nothing relevant, `zands ask`
  queries SearXNG's JSON API, fetches the top results with trafilatura and answers from them,
  labeling those sources as web results rather than indexed news. Optionally ingest the
  fetched pages into `news.db` so the next question is answered locally.
- A reranker over the fused candidates.
- A daily digest pushed into the system prompt, so the model knows the day's headlines without searching.
- An MCP server exposing `search_news` and `latest_digest`, so the zwans coding agent and other
  MCP clients can use Zands.
