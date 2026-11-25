# Zands

A local news RAG that runs on one laptop. Zands reads RSS feeds, stores the full text of each
article in a single SQLite file, and answers questions about current events with a local
[Ollama](https://ollama.com) model. A router decides whether a question needs the news index,
the model's own knowledge, today's headlines or a live web search (through your own SearXNG),
and the model answers only from what it was given, with citations. An MCP server lets coding
agents search the index too, and reach live data (weather, exchange rates, holidays, papers...)
through free public APIs, including any no-key API in the
[public-apis](https://github.com/public-apis/public-apis) list.

The point is to give local, open-weight models knowledge of today's news without web search
or fine-tuning: fetch when the news is stale, search locally, ground the answer in what was fetched.

```
feeds.toml ─► ingest ─► data/news.db ─────────────────────────────────────┐
 (on demand   feedparser  articles, chunks, chunks_fts (FTS5)              │
  when stale) httpx       chunk_vec (sqlite-vec), facts, digests           │
              trafilatura feeds, retrieval_log                             │
              nomic-embed → digest of today's top stories                  │
                                                                           ▼
question ─► router ──┬─ news ──────► hybrid retrieve (BM25 + vectors, RRF) ─► gate ─┐
            rules,   ├─ knowledge ─► (model's own knowledge; upgraded to news       │
            then     │               if the local index has a strong match)         ├─► prompt ─► Ollama ─► cited answer
            qwen2.5  ├─ web ───────► SearXNG ─► fetch ─► best chunks                │   facts first, headlines,
                     ├─ digest ────► today's top stories                            │   [n] news, [W] web, [D] digest
                     └─ none ──────► "can't answer that"                            │
                     weak local match + SearXNG up ─► news+web ────────────────────┘

zands mcp ─► search_news / latest_digest   (the news index)
           ─► weather, convert_currency, wikipedia, define_word, public_holidays,
              earthquakes, arxiv, books         (fixed free APIs)
           ─► find_api ─► call_api              (any no-key API in the public-apis list)
           for coding agents (zwans) and other MCP clients
```

## Setup

Requirements: macOS or Linux (only the optional `zands schedule` needs macOS's launchd),
[uv](https://docs.astral.sh/uv/), and Ollama running locally.

```bash
uv sync                          # .venv with Python 3.12 (uv-managed, not /usr/bin/python3)
ollama pull qwen2.5:latest       # answer model
ollama pull nomic-embed-text     # embedding model
uv run zands ingest              # first fetch (also embeds the new chunks and builds the digest)
uv run zands apis update         # load the public-apis list for find_api (about 2,400 APIs)
```

After that there is nothing to schedule: `ask`, `search_news` and `latest_digest` fetch the
feeds first when the newest article is more than 6 hours old (env `ZANDS_STALE_HOURS`, `0`
turns it off). Run `zands ingest` yourself any time, or `zands schedule install` if you do
want an hourly background fetch.

Optional, for the web fallback: a SearXNG instance with the JSON format enabled. Zands uses
`http://127.0.0.1:8888` by default (env `ZANDS_SEARXNG_URL`). Without one, everything works
except web results, and the router falls back to the local index. A minimal setup:

```bash
docker run -d --name searxng --restart unless-stopped -p 127.0.0.1:8888:8080 searxng/searxng
```

Then add `json` to `search.formats` in the container's `settings.yml` and restart it.

Use uv's Python, not macOS system Python: the system build often can't load SQLite extensions,
and vector search needs `sqlite-vec`. `uv run zands stats` shows `vectors: n/a (...)` if the
extension couldn't load; everything else keeps working on keyword search.

## Commands

```bash
uv run zands ingest                      # fetch feeds, store new articles, embed new chunks (safe to re-run)
uv run zands embed                       # backfill vectors for chunks that have none
uv run zands search "nvidia ceo"         # show retrieved chunks, with keyword/vector ranks
uv run zands ask "What happened with Nvidia's CEO?"   # routed, cited answer from a local model
uv run zands ask "What are today's top stories?"       # answered from the digest
uv run zands digest                      # today's top stories (rebuilt after every ingest)
uv run zands fact add Nvidia CEO "Jensen Huang" --from 2026-11-23 --source https://...
uv run zands fact list                   # current facts (--all includes expired)
uv run zands fact expire 3               # no longer true from today (--on YYYY-MM-DD)
uv run zands eval                        # known-answer set: keyword vs hybrid hit rates
uv run zands eval --routes               # router accuracy on eval/routes.toml
uv run zands mcp                         # MCP server on stdio (see "MCP server" below)
uv run zands ui                          # Datasette on http://127.0.0.1:8001/
uv run zands dashboard                   # charts on http://127.0.0.1:8002/ (read-only)
uv run zands apis update                 # (re)load the public-apis catalog
uv run zands apis find "cat facts"       # search it as find_api does (--all includes APIs needing a key)
uv run zands apis try weather Tokyo      # run one live-data tool from the shell
uv run zands schedule install            # optional hourly ingest via launchd (status / uninstall)
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
| `ask` | `--no-facts` / `--no-digest` / `--show-prompt` | leave facts out / leave headlines out / print the full prompt |
| `ask` | `--route auto\|news\|knowledge\|web\|digest` | force a route (default `auto`: let the router decide) |
| `ask` | `--no-router` / `--router-model NAME` | rules only / router model (env `ZANDS_ROUTER_MODEL`, default `qwen2.5:latest`) |
| `ask` | `--no-web` / `--save-web` | never use SearXNG / store fetched web pages in the index |
| `ask` | `--no-refresh` | don't fetch feeds first, even when the news is stale |
| `search`, `ask`, `eval` | `--rerank` | rerank candidates with the chat model (off by default: it measured worse, see Evaluation) |
| `digest` | `--refresh` / `--hours N` | rebuild now / look-back window (default 24) |
| Ollama users | `--ollama-url URL` | Ollama base URL (env `OLLAMA_HOST`, default `http://localhost:11434`) |
| Ollama users | `--embed-model NAME` | embedding model (env `ZANDS_EMBED_MODEL`, default `nomic-embed-text`) |
| `eval` | `--mode compare\|hybrid\|keyword\|vector` | what to score (default `compare`: keyword vs hybrid) |
| `eval` | `--answers` / `--list` | also check model answers (slow) / per-question results |
| `eval` | `--routes` | score the router instead of retrieval |
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
| `retrieval_log` | every retrieval: time, command, question, FTS query, k, chunk ids in rank order, effective mode, facts used, route + reason, web URLs, which retriever had each hit in its own top k |
| `digests` | today's top stories (one row per UTC day, rebuilt after every ingest) |
| `ingest_runs`, `ingest_feed_runs` | one row per ingest run (new articles, failed feeds, chunks embedded, duration) and one per feed in it |
| `eval_runs` | every `zands eval` score: hit@1/3/k and MRR per retrieval mode, or router accuracy |
| `meta` | which embedding model made the stored vectors; when run history started being recorded |

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
and the news, then today's headlines `[D1]...`, the numbered chunks (title, source, published
date, URL), any web results `[W1]...`, then the rules:
answer only from these, cite `[n]` / `[F1]`, say plainly when they don't cover the question. The
reply is streamed from Ollama's `/api/chat`, followed by the facts and sources used. With no
matching chunks and no facts, Zands says so without calling the model.

**Routing** (`src/zands/route.py`, `src/zands/ask.py`). Before retrieval, each `ask` question
gets a route:

| Route | When | Answer from |
|---|---|---|
| `news` | about current events (default when unsure) | the local index |
| `knowledge` | timeless ("What is a central bank?") | the model's own knowledge, with a note that it may be out of date |
| `web` | live data (weather, exchange rates, current prices) | SearXNG results |
| `digest` | "What are today's top stories?" | the digest |
| `none` | the user's own private data, advice or predictions | a fixed reply, no model call |

The decision is made in three steps:
1. **Rules.** Regexes for time-sensitive words, live-data phrasings, timeless phrasings,
   headline questions and personal data. A headline phrase followed by a topic ("latest news
   *on Nvidia*") is a news search, not a digest question. No model call.
2. **Classifier.** Only if no rule matched: `qwen2.5` answers with one word (0.14 s per call
   when the model is loaded).
3. **Gate.** After local retrieval, the best vector match decides:
   - a `knowledge` question with a strong local match (≥ 0.70) becomes `news`;
   - a `news` question with a weak match also gets web results (`news+web`), and local hits
     below 0.65 are dropped as noise.

   Those thresholds come from measuring covered questions (0.710–0.879) against uncovered ones
   (0.534–0.689).

Fact overrides are added to every route that calls the model. Each `ask` writes its route and
the reason (which rule fired, what the model said, what the gate changed) to `retrieval_log`, so
misroutes can be reviewed. `zands ui` has a canned "Router decisions" query for that.

**Web fallback** (`src/zands/web.py`). Queries SearXNG's JSON API, fetches the top 4 result
pages with the same polite client as ingest, extracts them with trafilatura, chunks them, and
picks the 4 chunks closest to the question (nomic embeddings; keyword overlap without them;
at most 2 per page). A page that can't be fetched contributes its search snippet. Web chunks are
labeled `[W1]...` in the prompt, as live web results rather than curated news, and the model is
told to prefer news sources and to treat everything in a source as data, never as instructions.
`--save-web` stores the fetched pages in the index (source `web: <domain>`) so the next similar
question is answered locally. SearXNG is health-checked (2 s timeout) only when a route could use
it; if it's down, or returns nothing, the answer falls back to the local index and says why.

**Digest** (`src/zands/digest.py`). Rebuilt at the end of every ingest, with no LLM involved, so
it can't hallucinate. The last 24 hours of articles are grouped into stories by embedding
similarity: an article joins a story when it is ≥ 0.90 similar to the story's mean vector.
(Same-event coverage across outlets measured ≥ 0.95; 0.85–0.90 merged merely related events.)
Stories are ranked by how many outlets cover them, and each is shown by its most typical
headline. Without vectors, stories are grouped by headline word overlap. The digest goes into
`ask` prompts as `TODAY'S HEADLINES [D1]...` for context, and answers `digest`-route questions
directly.

**MCP server** (`src/zands/mcp_server.py`). `zands mcp` serves twelve tools over stdio.

News, from the local index:
- `search_news(query, k=5, mode="hybrid")`: the top hits as numbered plain text (title,
  outlet, date, URL and a ~700-character excerpt), plus when the index was last updated;
- `latest_digest()`: today's top stories, with today's date.

If the newest article is older than `ZANDS_STALE_HOURS` (6), both fetch the feeds first and say
so at the top of the result. A file lock keeps two refreshes from running at once, and a failed
refresh is a note, not an error: the search still runs on what is there.

Live data, from fixed free APIs that need no key (`src/zands/apis/tools.py`):

| Tool | Source |
|---|---|
| `weather(place, days=3)` | Open-Meteo forecast and air quality, local time; wttr.in if Open-Meteo fails |
| `convert_currency(amount, from_currency, to_currency)` | Frankfurter (ECB rates); CoinGecko for crypto |
| `wikipedia(topic)` | Wikipedia search, then the page summary |
| `define_word(word)` | dictionaryapi.dev |
| `public_holidays(country, year)` | Nager.Date, national holidays, next one marked |
| `earthquakes(min_magnitude=4.5, period="day")` | USGS feeds |
| `arxiv(query, max_results=5)` | arXiv search, every word required, by relevance |
| `books(query, max_results=5)` | Open Library |

Anything else, from the public-apis list (`src/zands/apis/catalog.py`):
- `find_api(need)`: no-key HTTPS APIs whose name, description or category match the need
  (every word first, then any word), with their docs URLs;
- `call_api(url)`: a guarded GET, used to read an API's docs and then call its endpoint.

`call_api` goes wherever the model points it, so it is fenced (`src/zands/apis/http.py`): HTTPS
only, no credentials in the URL, no localhost or private, link-local or multicast addresses
(checked after DNS resolution, and again on each of at most 3 redirects), bodies cut at 400 KB.
It answers with plain text: JSON compacted, HTML reduced to its main text, cut at 6,000
characters. Failures come back as text the model can act on, not as errors.

Every tool except `call_api` is marked read-only, so clients that ask before other tools (zwans
does) run them straight away; `call_api` is left for the client to ask about, as it does for its
own web fetches. To let zwans run it without asking, add `allow = ["mcp__zands__call_api"]` to
`~/.zwans/policy.toml`.

The server's instructions give the model a short way to work when it lacks information: decide
what is missing, call the most specific tool, fall back to `find_api` then `call_api`, check the
result and retry with other words or another tool (up to three times), call a tool per part of a
question, and answer with sources, or say what it tried. Each tool description starts with when
to use it, since small local models rarely work that out from a tool's name. Results are short
plain text, not JSON: five news hits are about 1,000 tokens instead of 5,000, which a 7B model
reads and quotes far more reliably. Arguments small models get wrong (an unknown `mode`, `k` as
text) fall back to defaults instead of failing the call.

The client does its own answering. Searches are logged with command `mcp`. To connect an MCP
client (for example zwans), give it absolute paths:

```json
{
  "mcpServers": {
    "zands": {
      "command": "/path/to/Zands/.venv/bin/zands",
      "args": ["--db", "/path/to/Zands/data/news.db", "mcp"]
    }
  }
}
```

**Reranker** (`src/zands/rerank.py`, opt-in with `--rerank`). One listwise call to the chat
model to reorder the top 20 fused candidates; on any failure the original order is kept. It is
off by default because it measured worse (see Evaluation).

**Scheduling** (`src/zands/schedule.py`, optional: news is otherwise fetched on demand). `zands schedule install` writes
`~/Library/LaunchAgents/local.zands.ingest.plist` and loads it with `launchctl bootstrap`. The
agent runs this venv's Python directly (`python -m zands.cli --db <abs> ingest --feeds <abs>`)
every 60 minutes and once at load, at background priority, logging to
`~/Library/Logs/zands/ingest.log`. launchd rather than cron, because cron skips runs while the
laptop sleeps, and launchd runs a missed interval once on wake. Each run also rebuilds the digest. `zands schedule status` shows
whether it's loaded, run count, last exit code and the log tail. `zands schedule uninstall`
unloads it and deletes the plist.

**Browsing** (`src/zands/ui.py`). `zands ui` runs [Datasette](https://datasette.io) over the
database with sqlite-vec loaded, read-only (no write plugins), plus canned queries: recent
retrievals with their top hit, questions that retrieved nothing, feed health, articles per
day, current facts.

**Dashboard** (`src/zands/dashboard.py`, `src/zands/dashboard.html`). `zands dashboard` serves
one page of charts on 127.0.0.1:8002 (`--port`, `--host`), refreshed every minute. It is a stdlib
`http.server` with one JSON endpoint per panel (`/api/<panel>.json`, plus `/api/all.json`); every
request opens the database read-only (`mode=ro`). The charts are hand-drawn SVG in a single HTML
file with no external scripts, so the page works offline. It follows the system light/dark
setting, and a button switches theme. Every chart has hover tooltips and a "Show table" view.
Panels:
- **Coverage and size:** articles, chunks, vectors (share embedded), database size, fact
  overrides current/expired, questions logged.
- **Articles stored over time:** new articles per feed by fetch time; hourly while the index has
  under three days of history, daily after that. The seven largest feeds get their own colour and
  the rest fold into "Other".
- **Feed health:** status (OK / stale after three missed hourly runs / failing), last success,
  last run's new items, the last error, and new articles per ingest run.
- **Today's digest:** stories ranked by how many distinct outlets covered them.
- **Headline topics:** the most frequent headline words per day over the last week, counted once
  per headline (a cheap stand-in for entity extraction).
- **Questions and retrieval:** questions per day by command, router decisions, retrieval modes,
  where the hits came from, and questions that found nothing. Eval runs are left out.
- **Eval history:** hit@1 and MRR per retrieval mode for each `zands eval` run, and router
  accuracy for each `zands eval --routes`.

Three panels use data that older versions didn't record: ingest runs, eval runs, and which
retriever found each hit. These are recorded from the first time the new version opens the
database. Until there's data, those panels show "collecting since <date>". The hit breakdown
counts a hit as *keyword* or *vector* when only that retriever ranked it in its own top k,
*both* when each did, and *neither (fusion)* when RRF plus the recency boost lifted it. Counting
over the full 50-candidate lists would be useless, since nearly every hybrid hit appears in both.

## Evaluation

`eval/questions.toml` holds 32 known-answer questions about articles ingested on 2026-11-24:
20 written from the morning's articles and 12 from the afternoon and evening ingests. 15 share
words with the headline (`keyword` style) and 17 deliberately avoid them (`paraphrase` style).
Each question lists:
- URL substrings of the article(s) that answer it
- `same_story` terms that identify the story. A retrieved chunk containing all of them also
  counts as a hit, marked `~` in `zands eval --list`. Terms match at the start of a word, and
  `a|b` means either.
- answer keywords

`zands eval` reports, per mode, how often an answering article appears at rank 1, in the top 3
and in the top k, plus mean reciprocal rank. Questions with no matching article in the database
are reported as missing, not as misses. `--answers` also asks the model and checks the answer
keywords. Each run is stored with a short hash of the question file, so the dashboard can mark
where the set changed rather than showing it as a regression.

**Current results** (2026-11-24 evening, 549 articles, 32 questions, k = 8):

| Mode | hit@1 | hit@3 | hit@8 | MRR |
|---|---|---|---|---|
| keyword (BM25) | 88% | 91% | 94% | 0.89 |
| hybrid (BM25 + nomic, RRF) | 88% | 97% | 97% | 0.92 |
| vector only | 91% | 100% | 100% | 0.95 |

Six of hybrid's hits were another outlet's article on the same story. All four hybrid questions
not at rank 1 are paraphrases:
- the German strike (#2)
- Finland's airspace (#2)
- what Huang told staff (#3), on the story with the most coverage in the index
- which country drives the rise in landmine casualties (missed)

Keyword search misses the last two entirely. On this set, vector-only edges out hybrid: the
keyword half of the fusion pulls in articles that share words but miss the point. Twenty-eight
of the 32 questions are at rank 1 in every mode, so the comparison rests on a handful of hard
paraphrases. The 12 new questions turned out easier than the first 20.

How the set was built: I wrote the questions and `same_story` terms before running retrieval,
then checked the terms against the whole index and tightened any that matched unrelated stories
(substring matching let "ended" match "attended"). I made one change after seeing results. The
answer check showed that the landmine question was being credited with a Syria-focused article
that mentions Myanmar only in passing, so I required "five-year high" as well. That turned a
"hit" into the miss listed above.

Answer check (hybrid, qwen2.5, run before that fix): 31 of 32 answers contained an expected
keyword. The one failure was the landmine question. The model answered "Syria" from the article
the fix now rejects.

**Earlier results** (2026-11-24 morning, 348 articles, the first 20 questions, URL matches only):

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

Re-run that evening with URL matching only, those 20 dropped to hybrid hit@1 60% and MRR 0.78,
with the same code. Five of the eight "misses" were another outlet's article on the same story,
ranked above the one URL the question expected. That drift is why `same_story` exists.

The set is small and was written by the same session that built the system, so treat these as
a regression check, not a benchmark.

**Reranker.** Hybrid + the qwen2.5 listwise reranker scored hit@1 85%, hit@3 95%, MRR 0.90:
worse than plain hybrid (90% / 100% / 0.94), and 5.7 s slower per question. It moved two
correct articles down (to #2 and #5) and fixed one. So it's off by default; it may earn its
place with a much bigger index or a stronger model.

**Router** (`zands eval --routes`, 27 labeled questions in `eval/routes.toml`): 27/27 correct
with qwen2.5 as the classifier (15 decided by rules, 12 by the model). Rules alone (with the
`news` default) get 20/24 on the original 24. With `qwen3.5:0.8b` as the classifier, 19/24: it
sent advice questions and a news question to `web`. That is why qwen2.5 is the default. Add questions as the news moves on, and write them before
looking at what retrieval returns.

## Tests

```bash
uv run pytest
```

No test touches the network: Ollama is replaced by a deterministic fake embedder, HTTP by
`httpx.MockTransport` or a fake fetch, and `launchctl` by a recording stub. Covered:
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
- router rules, classifier parsing and failure, the gate; the ask planner for every route
  (forced, fallbacks, noise filtering, facts/digest placement, logging)
- SearXNG client and page selection against a fake transport, saving web pages; digest
  grouping, ranking, rendering and the no-vector fallback; the MCP server through an in-process
  client; reranker parsing and failure handling
- the API fence (schemes, credentials, private and resolved addresses, redirects, size cap),
  readable output, every live-data tool against canned responses and its failure text, the
  public-apis parser and catalog search, and on-demand refresh (fresh, stale, empty, locked,
  failing)
- the dashboard: every panel on a fixture database (hourly vs daily buckets, folding feeds into
  "Other", distinct digest outlets, headline topics, which questions count as misses), the
  "collecting since" empty states, recording of ingest/eval runs and per-hit retriever sources,
  that the connection is read-only, and the HTTP server (page has no external URLs, per-panel
  endpoints, 404s)

The suite also passes with `OLLAMA_HOST` and `ZANDS_SEARXNG_URL` pointed at a dead port.

## Roadmap

**Phase 1: keyword RAG (done).** Feeds → SQLite + FTS5 → BM25 search → cited answers via Ollama.

**Phase 2: hybrid search, fact overrides, scheduling, browsing (done).** `zands ui` (Datasette);
sqlite-vec + nomic-embed-text with RRF hybrid retrieval and keyword fallback; `facts` table with
`zands fact add/list/expire`, put first in prompts; launchd hourly ingest with
`zands schedule install/uninstall/status`; 20-question known-answer set with `zands eval`.

**Phase 3: routing, web fallback, digest, MCP (done).** Rule + model query router with a
post-retrieval gate and route logging; SearXNG web fallback (optionally saved into the index);
LLM-free daily digest in prompts and as its own route; MCP server (`search_news`,
`latest_digest`); an opt-in reranker (measured, off by default).

**Phase 4: dashboard (done).** `zands dashboard`: a read-only stdlib server with one page of
offline SVG charts (coverage, ingest over time, feed health, digest, headline topics, retrieval
log, eval history). New run history (`ingest_runs`, `eval_runs`, per-hit retriever sources) fills
the panels that older databases had no data for. Datasette plugins were considered and set aside
because they add dependencies. A charting library was set aside because the page would have
needed a vendored bundle of about 1 MB.

**Phase 5: live data and on-demand news (done).** Eight live-data MCP tools over free no-key
APIs; the public-apis list as a searchable catalog (`find_api`) with a fenced `call_api`;
server instructions for working out missing information step by step; feeds fetched when the
news is stale instead of hourly.

**Possible next steps.**
- Use the live-data tools in `zands ask` too (the `web` route still scrapes search results for weather and prices).
- Retention: drop articles older than N months and compact the file (the index grows about 1–2 GB a year).
- More known-answer and router questions as the news moves on, written before looking at results:
  harder paraphrases, and questions on crowded stories, where the modes actually differ.
- Real entity extraction for the topics panel (headline word counts merge "Jensen" and "Huang" as separate topics).
