"""SQLite storage: one file holds articles, chunks, feed health, the FTS index and the retrieval log."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS feeds (
    id              INTEGER PRIMARY KEY,
    name            TEXT NOT NULL UNIQUE,
    url             TEXT NOT NULL,
    last_success_at TEXT,
    last_error      TEXT,
    last_error_at   TEXT,
    failure_count   INTEGER NOT NULL DEFAULT 0,  -- consecutive failures; reset on success
    last_new_items  INTEGER
);

CREATE TABLE IF NOT EXISTS articles (
    id           INTEGER PRIMARY KEY,
    url          TEXT NOT NULL UNIQUE,           -- normalized; the idempotency key
    original_url TEXT NOT NULL,
    title        TEXT NOT NULL,
    source       TEXT NOT NULL,
    feed_id      INTEGER REFERENCES feeds(id),
    published_at TEXT,                           -- ISO 8601 UTC, may be NULL
    fetched_at   TEXT NOT NULL,
    summary      TEXT,
    body         TEXT NOT NULL,
    extraction   TEXT NOT NULL                   -- 'trafilatura' or 'rss'
);
CREATE INDEX IF NOT EXISTS articles_published ON articles(published_at);

CREATE TABLE IF NOT EXISTS chunks (
    id           INTEGER PRIMARY KEY,
    article_id   INTEGER NOT NULL REFERENCES articles(id) ON DELETE CASCADE,
    chunk_index  INTEGER NOT NULL,
    title        TEXT NOT NULL,
    source       TEXT NOT NULL,
    url          TEXT NOT NULL,
    published_at TEXT,
    body         TEXT NOT NULL,
    word_count   INTEGER NOT NULL,
    UNIQUE (article_id, chunk_index)
);

-- External-content FTS5 index over chunks(title, body), kept in sync by triggers.
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    title, body,
    content='chunks', content_rowid='id',
    tokenize='porter unicode61 remove_diacritics 2'
);

CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, title, body) VALUES (new.id, new.title, new.body);
END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, title, body) VALUES ('delete', old.id, old.title, old.body);
END;
CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, title, body) VALUES ('delete', old.id, old.title, old.body);
    INSERT INTO chunks_fts(rowid, title, body) VALUES (new.id, new.title, new.body);
END;

CREATE TABLE IF NOT EXISTS retrieval_log (
    id         INTEGER PRIMARY KEY,
    created_at TEXT NOT NULL,
    command    TEXT,                             -- 'search' / 'ask' / caller name
    question   TEXT NOT NULL,
    query      TEXT,                             -- the FTS5 query actually run (NULL if none)
    k          INTEGER NOT NULL,
    chunk_ids  TEXT NOT NULL                     -- JSON array, in rank order
);
"""

# Query parameters that only track clicks; dropping them keeps one URL per article.
TRACKING_PARAMS = {
    "fbclid", "gclid", "dclid", "msclkid", "mc_cid", "mc_eid", "igshid",
    "cmpid", "ito", "at_medium", "at_campaign", "at_custom1", "at_custom2",
    "at_custom3", "at_custom4", "ocid", "smid", "ref", "ref_src",
    "taid", "guccounter", "guce_referrer", "guce_referrer_sig", "sr_share", "cmp",
    "maca", "traffic_source", "rss", "feed", "partner", "cid",
}


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def normalize_url(url: str) -> str:
    """Canonical form of an article URL, used as the unique key.

    Lowercases scheme and host, upgrades http to https, drops default ports and
    the fragment, strips tracking parameters (utm_* and friends), sorts the
    remaining parameters, and drops a trailing slash. `www.` is left alone
    because some sites serve different things with and without it.
    """
    parts = urlsplit(url.strip())
    scheme = (parts.scheme or "https").lower()
    if scheme == "http":
        scheme = "https"
    netloc = parts.netloc.lower()
    if netloc.endswith(":443") or netloc.endswith(":80"):
        netloc = netloc.rsplit(":", 1)[0]
    path = parts.path or "/"
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/")
    query = [
        (k, v)
        for k, v in parse_qsl(parts.query, keep_blank_values=True)
        if not k.lower().startswith("utm_") and k.lower() not in TRACKING_PARAMS
    ]
    query.sort()
    return urlunsplit((scheme, netloc, path, urlencode(query), ""))


def connect(path: str | Path) -> sqlite3.Connection:
    """Open (creating if needed) the database and make sure the schema exists."""
    if str(path) != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA busy_timeout = 5000")
    conn.executescript(SCHEMA)
    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    conn.commit()
    return conn


# ---------------------------------------------------------------- feeds


def ensure_feed(conn: sqlite3.Connection, name: str, url: str) -> int:
    conn.execute(
        "INSERT INTO feeds(name, url) VALUES (?, ?) ON CONFLICT(name) DO UPDATE SET url = excluded.url",
        (name, url),
    )
    conn.commit()
    return conn.execute("SELECT id FROM feeds WHERE name = ?", (name,)).fetchone()[0]


def record_feed_success(conn: sqlite3.Connection, feed_id: int, new_items: int) -> None:
    conn.execute(
        "UPDATE feeds SET last_success_at = ?, failure_count = 0, last_new_items = ? WHERE id = ?",
        (utcnow(), new_items, feed_id),
    )
    conn.commit()


def record_feed_failure(conn: sqlite3.Connection, feed_id: int, error: str) -> None:
    conn.execute(
        "UPDATE feeds SET last_error = ?, last_error_at = ?, failure_count = failure_count + 1 WHERE id = ?",
        (error[:1000], utcnow(), feed_id),
    )
    conn.commit()


# ---------------------------------------------------------------- articles


@dataclass
class Article:
    url: str  # as found in the feed; normalized on insert
    title: str
    source: str
    body: str
    extraction: str  # 'trafilatura' | 'rss'
    published_at: str | None = None
    summary: str | None = None
    feed_id: int | None = None
    fetched_at: str = field(default_factory=utcnow)


def article_exists(conn: sqlite3.Connection, url: str) -> bool:
    row = conn.execute("SELECT 1 FROM articles WHERE url = ?", (normalize_url(url),)).fetchone()
    return row is not None


def insert_article(conn: sqlite3.Connection, article: Article, chunks: list[str]) -> int | None:
    """Insert an article and its chunks in one transaction.

    Returns the new article id, or None if an article with the same
    normalized URL already exists (in which case nothing is written).
    """
    key = normalize_url(article.url)
    with conn:  # commits on success, rolls back on error
        cur = conn.execute(
            """INSERT INTO articles(url, original_url, title, source, feed_id, published_at,
                                    fetched_at, summary, body, extraction)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(url) DO NOTHING""",
            (
                key, article.url, article.title, article.source, article.feed_id,
                article.published_at, article.fetched_at, article.summary,
                article.body, article.extraction,
            ),
        )
        if cur.rowcount == 0:
            return None
        article_id = cur.lastrowid
        conn.executemany(
            """INSERT INTO chunks(article_id, chunk_index, title, source, url, published_at, body, word_count)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                (article_id, i, article.title, article.source, key, article.published_at, text, len(text.split()))
                for i, text in enumerate(chunks)
            ],
        )
    return article_id


# ---------------------------------------------------------------- retrieval log


def log_retrieval(
    conn: sqlite3.Connection,
    question: str,
    query: str | None,
    chunk_ids: list[int],
    k: int,
    command: str | None = None,
) -> None:
    conn.execute(
        "INSERT INTO retrieval_log(created_at, command, question, query, k, chunk_ids) VALUES (?, ?, ?, ?, ?, ?)",
        (utcnow(), command, question, query, k, json.dumps(chunk_ids)),
    )
    conn.commit()


def stats(conn: sqlite3.Connection) -> dict:
    one = lambda sql: conn.execute(sql).fetchone()[0]  # noqa: E731
    return {
        "articles": one("SELECT COUNT(*) FROM articles"),
        "chunks": one("SELECT COUNT(*) FROM chunks"),
        "retrievals": one("SELECT COUNT(*) FROM retrieval_log"),
        "newest": one("SELECT MAX(published_at) FROM articles"),
        "last_fetch": one("SELECT MAX(fetched_at) FROM articles"),
        "feeds": [dict(r) for r in conn.execute("SELECT * FROM feeds ORDER BY name")],
    }
