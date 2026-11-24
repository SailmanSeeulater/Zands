"""Read RSS/Atom feeds, fetch and extract each new article, chunk it, store it.

Robustness rules:
- Idempotent: articles already in the database (by normalized URL) are
  skipped before any page fetch, and the insert itself is ON CONFLICT DO NOTHING.
- One feed failing (HTTP error, unparseable XML) is logged, recorded in the
  `feeds` table, and the run moves on to the next feed.
- One article failing to fetch or extract falls back to the RSS summary.
- Polite: timeouts, a generic User-Agent, and a small fixed concurrency.
"""

from __future__ import annotations

import html
import logging
import re
import sqlite3
import time
import tomllib
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import feedparser
import httpx

# trafilatura pulls in urllib3, which warns about SOCKS when a proxy env var is set; we fetch with httpx.
warnings.filterwarnings("ignore", message="SOCKS support in urllib3")
import trafilatura  # noqa: E402

from . import store
from .chunk import chunk_text
from .embed import Embedder, EmbedError, embed_pending

log = logging.getLogger(__name__)

USER_AGENT = "Mozilla/5.0 (compatible; Zands/0.1; local news reader)"
TIMEOUT = httpx.Timeout(20.0, connect=10.0)
MAX_PAGE_BYTES = 5_000_000
MIN_EXTRACTED_WORDS = 80  # below this, trafilatura probably grabbed a paywall/cookie stub


@dataclass
class Feed:
    name: str
    url: str


@dataclass
class FeedResult:
    name: str
    ok: bool = True
    error: str | None = None
    entries: int = 0
    new: int = 0
    skipped: int = 0
    rss_fallback: int = 0
    dropped: int = 0


@dataclass
class IngestReport:
    feeds: list[FeedResult] = field(default_factory=list)
    seconds: float = 0.0
    embedded: int = 0
    embed_error: str | None = None  # set when embedding was skipped or failed (keyword search still works)

    @property
    def new(self) -> int:
        return sum(f.new for f in self.feeds)

    @property
    def failed(self) -> list[FeedResult]:
        return [f for f in self.feeds if not f.ok]


def load_feeds(path: str | Path) -> list[Feed]:
    with open(path, "rb") as fh:
        data = tomllib.load(fh)
    feeds = []
    seen = set()
    for item in data.get("feed", []):
        if not item.get("enabled", True):
            continue
        name, url = item["name"], item["url"]
        if name in seen:
            raise ValueError(f"duplicate feed name in {path}: {name!r}")
        seen.add(name)
        feeds.append(Feed(name=name, url=url))
    return feeds


def make_client() -> httpx.Client:
    return httpx.Client(
        headers={"User-Agent": USER_AGENT, "Accept-Language": "en"},
        timeout=TIMEOUT,
        follow_redirects=True,
    )


# ---------------------------------------------------------------- helpers

_TAG = re.compile(r"<[^>]+>")
_BLOCK_TAG = re.compile(r"</?(p|div|br|li|h[1-6]|blockquote)[^>]*>", re.I)


def html_to_text(fragment: str) -> str:
    """Plain text from an RSS summary/content fragment."""
    if not fragment:
        return ""
    text = _BLOCK_TAG.sub("\n\n", fragment)
    text = html.unescape(_TAG.sub("", text))
    paras = [" ".join(p.split()) for p in re.split(r"\n\s*\n", text)]
    return "\n\n".join(p for p in paras if p)


def entry_date(entry) -> str | None:
    for key in ("published_parsed", "updated_parsed", "created_parsed"):
        t = entry.get(key)
        if t:
            try:
                return datetime(*t[:6], tzinfo=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            except (TypeError, ValueError):
                continue
    return None


def entry_rss_text(entry) -> str:
    """The longest text the feed itself carries for an entry (content or summary)."""
    candidates = [c.get("value", "") for c in entry.get("content", []) or []]
    candidates.append(entry.get("summary", "") or "")
    texts = [html_to_text(c) for c in candidates]
    return max(texts, key=lambda t: len(t.split()), default="")


def normalize_date(value: str | None) -> str | None:
    """trafilatura gives YYYY-MM-DD; store it as midnight UTC."""
    if not value:
        return None
    try:
        return datetime.strptime(value[:10], "%Y-%m-%d").strftime("%Y-%m-%dT00:00:00Z")
    except ValueError:
        return None


@dataclass
class Extracted:
    text: str | None
    date: str | None
    error: str | None = None


def fetch_and_extract(client: httpx.Client, url: str) -> Extracted:
    """Fetch an article page and pull out the main text. Never raises."""
    try:
        resp = client.get(url)
        resp.raise_for_status()
        ctype = resp.headers.get("content-type", "")
        if "html" not in ctype and "xml" not in ctype:
            return Extracted(None, None, f"not HTML ({ctype or 'no content-type'})")
        if len(resp.content) > MAX_PAGE_BYTES:
            return Extracted(None, None, f"page too large ({len(resp.content)} bytes)")
        page = resp.text
        text = trafilatura.extract(
            page, url=str(resp.url), include_comments=False, include_tables=False, favor_precision=True
        )
        date = None
        try:
            meta = trafilatura.extract_metadata(page)
            date = normalize_date(meta.date if meta else None)
        except Exception:  # metadata is a nice-to-have
            pass
        return Extracted(text, date)
    except Exception as exc:  # noqa: BLE001 - isolate per-article failures
        return Extracted(None, None, f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------- per feed


def ingest_feed(
    conn: sqlite3.Connection,
    client: httpx.Client,
    feed: Feed,
    pool: ThreadPoolExecutor,
    limit: int | None = None,
) -> FeedResult:
    result = FeedResult(name=feed.name)
    feed_id = store.ensure_feed(conn, feed.name, feed.url)
    try:
        resp = client.get(feed.url)
        resp.raise_for_status()
        parsed = feedparser.parse(resp.content)
        if parsed.bozo and not parsed.entries:
            raise ValueError(f"unparseable feed: {parsed.get('bozo_exception')!r}")
        entries = parsed.entries[:limit] if limit else parsed.entries
        result.entries = len(entries)

        # Pick out entries we don't have yet (also dedupe within this feed).
        todo = []
        seen: set[str] = set()
        for entry in entries:
            link = entry.get("link")
            if not link:
                result.dropped += 1
                continue
            key = store.normalize_url(link)
            if key in seen or store.article_exists(conn, link):
                result.skipped += 1
                continue
            seen.add(key)
            todo.append(entry)

        extracted = pool.map(lambda e: fetch_and_extract(client, e["link"]), todo)
        for entry, ext in zip(todo, extracted):
            title = " ".join(html.unescape(entry.get("title", "")).split()) or "(untitled)"
            rss_text = entry_rss_text(entry)
            text = ext.text or ""
            if len(text.split()) >= MIN_EXTRACTED_WORDS or (text and len(text.split()) >= len(rss_text.split())):
                body, how = text, "trafilatura"
            else:
                if ext.error:
                    log.debug("%s: %s -> RSS fallback (%s)", feed.name, entry["link"], ext.error)
                body, how = rss_text, "rss"
            if not body.strip():
                body = title  # headline-only entry; still searchable by title
                how = "rss"
            article = store.Article(
                url=entry["link"],
                title=title,
                source=feed.name,
                body=body,
                extraction=how,
                published_at=entry_date(entry) or ext.date,
                summary=html_to_text(entry.get("summary", "")) or None,
                feed_id=feed_id,
            )
            if store.insert_article(conn, article, chunk_text(body)) is None:
                result.skipped += 1  # another feed inserted it meanwhile
                continue
            result.new += 1
            if how == "rss":
                result.rss_fallback += 1

        store.record_feed_success(conn, feed_id, result.new)
    except Exception as exc:  # noqa: BLE001 - one broken feed must not stop the run
        result.ok = False
        result.error = f"{type(exc).__name__}: {exc}"
        log.warning("feed %r failed: %s", feed.name, result.error)
        store.record_feed_failure(conn, feed_id, result.error)
    return result


def run_ingest(
    conn: sqlite3.Connection,
    feeds: list[Feed],
    client: httpx.Client | None = None,
    concurrency: int = 4,
    limit: int | None = None,
    embedder: Embedder | None = None,
) -> IngestReport:
    """Ingest every feed in turn, then embed any chunks without a vector.

    Article pages within a feed are fetched with `concurrency` workers;
    database writes stay on this thread. Embedding failures (Ollama down,
    sqlite-vec missing) are reported but never fail the ingest: the articles
    are stored and searchable by keyword, and the next run backfills vectors.
    """
    start = time.monotonic()
    report = IngestReport()
    own_client = client is None
    client = client or make_client()
    try:
        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
            for feed in feeds:
                res = ingest_feed(conn, client, feed, pool, limit=limit)
                log.info(
                    "%-28s %s new=%d skipped=%d rss_fallback=%d%s",
                    feed.name, "ok  " if res.ok else "FAIL", res.new, res.skipped, res.rss_fallback,
                    f" error={res.error}" if res.error else "",
                )
                report.feeds.append(res)
    finally:
        if own_client:
            client.close()
    if embedder is not None:
        if not getattr(conn, "vec_enabled", False):
            report.embed_error = f"sqlite-vec unavailable: {getattr(conn, 'vec_error', None)}"
        else:
            try:
                report.embedded = embed_pending(conn, embedder, progress=True)
            except EmbedError as exc:
                report.embed_error = str(exc)
        if report.embed_error:
            log.warning("embedding skipped: %s (keyword search still works; next run retries)", report.embed_error)
    report.seconds = time.monotonic() - start
    return report
