"""News on demand: fetch the feeds only when someone asks and the index is stale.

With no hourly job, `search_news`, `latest_digest` and `zands ask` call `refresh_if_stale`
first. If the newest fetch is older than ZANDS_STALE_HOURS (6 by default) it runs an ingest
(feeds, embeddings when Ollama is up, the digest), which takes about 10-60 s depending on how
much is new. A lock file stops two callers from ingesting at once: the second one skips and
searches what is there. Any failure is reported, never raised, so a search still answers.
"""

from __future__ import annotations

import fcntl
import logging
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from . import digest as digestlib
from .ingest import load_feeds, run_ingest

log = logging.getLogger(__name__)

DEFAULT_STALE_HOURS = 6.0


def stale_hours() -> float:
    try:
        return float(os.environ.get("ZANDS_STALE_HOURS", DEFAULT_STALE_HOURS))
    except ValueError:
        return DEFAULT_STALE_HOURS


def feeds_path(db_path: str | Path) -> Path:
    """ZANDS_FEEDS, else feeds.toml next to the data folder (the repo root), else ./feeds.toml."""
    if os.environ.get("ZANDS_FEEDS"):
        return Path(os.environ["ZANDS_FEEDS"])
    beside = Path(db_path).resolve().parent.parent / "feeds.toml"
    return beside if beside.exists() else Path("feeds.toml")


def age_hours(conn: sqlite3.Connection, now: datetime | None = None) -> float | None:
    newest = conn.execute("SELECT MAX(fetched_at) FROM articles").fetchone()[0]
    if not newest:
        return None
    fetched = datetime.fromisoformat(newest.replace("Z", "+00:00"))
    return ((now or datetime.now(timezone.utc)) - fetched).total_seconds() / 3600


@dataclass
class Refresh:
    ran: bool
    note: str  # one line for the model / the user: what happened


def refresh_if_stale(
    conn: sqlite3.Connection,
    db_path: str | Path,
    max_age: float | None = None,
    embedder=None,
    ingest=run_ingest,
) -> Refresh:
    max_age = stale_hours() if max_age is None else max_age
    age = age_hours(conn)
    if max_age <= 0 or (age is not None and age < max_age):
        return Refresh(False, "")
    lock_path = Path(db_path).resolve().parent / ".ingest.lock"
    try:
        lock = open(lock_path, "w")  # noqa: SIM115 - held for the ingest below
    except OSError as exc:
        return Refresh(False, f"news is {age or 0:.0f}h old and couldn't be refreshed ({exc})")
    try:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return Refresh(False, "another refresh is running; searching what is already stored")
        path = feeds_path(db_path)
        try:
            report = ingest(conn, load_feeds(path), embedder=embedder)
        except Exception as exc:  # noqa: BLE001 - a failed refresh must not fail the search
            log.warning("on-demand ingest failed: %s", exc)
            return Refresh(False, f"news is {age or 0:.0f}h old; refreshing failed ({exc})")
        try:
            digestlib.refresh(conn)
        except Exception as exc:  # noqa: BLE001
            log.warning("digest not rebuilt: %s", exc)
        was = "never fetched" if age is None else f"{age:.0f}h old"
        note = f"news was {was}, so feeds were fetched just now: {report.new} new articles"
        if getattr(report, "embed_error", None):
            note += " (stored for keyword search; embeddings skipped)"
        return Refresh(True, note)
    finally:
        lock.close()
