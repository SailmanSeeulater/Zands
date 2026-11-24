"""Daily digest: today's top stories, so the model knows the headlines without searching.

Built without an LLM, so it can't hallucinate: the last 24 hours of articles
are grouped into stories by embedding similarity (an article joins a story
when it is >= STORY_SIMILARITY to the story's mean vector), and stories are
ranked by how many different outlets cover them, then by size and recency.
Each line is a real headline with its sources. Without vectors, stories are
grouped by headline word overlap instead.

Rebuilt after every ingest (one row per UTC day), put into `ask` prompts and
served by the MCP `latest_digest` tool.
"""

from __future__ import annotations

import json
import math
import sqlite3
import struct
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone

from . import store
from .search import query_terms

DEFAULT_HOURS = 24
MAX_ITEMS = 15
# Same event across outlets scores >= 0.95 with nomic-embed-text; 0.85-0.90 mixes the same topic
# with merely related events (measured 2026-11-24), so 0.90 groups stories without merging topics.
STORY_SIMILARITY = 0.90
TITLE_OVERLAP = 0.5  # fallback without vectors: Jaccard overlap of headline words


@dataclass
class Story:
    title: str
    sources: list[str]
    article_ids: list[int]
    newest: str
    titles: list[str] = field(default_factory=list)

    @property
    def outlets(self) -> int:
        return len(set(self.sources))


@dataclass
class Digest:
    day: str
    created_at: str
    hours: int
    stories: list[Story]
    article_count: int

    @property
    def text(self) -> str:
        return render(self)


def _unit(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _recent_articles(conn: sqlite3.Connection, since: str) -> list[sqlite3.Row]:
    vec_join = ""
    vec_col = "NULL AS embedding"
    if getattr(conn, "vec_enabled", False):
        vec_join = "LEFT JOIN chunks c ON c.article_id = a.id AND c.chunk_index = 0 LEFT JOIN chunk_vec v ON v.chunk_id = c.id"
        vec_col = "v.embedding AS embedding"
    return conn.execute(
        f"""SELECT a.id, a.title, a.source, COALESCE(a.published_at, a.fetched_at) AS ts, {vec_col}
            FROM articles a {vec_join}
            WHERE COALESCE(a.published_at, a.fetched_at) >= ? AND a.source NOT LIKE 'web: %'
            ORDER BY ts DESC""",
        (since,),
    ).fetchall()


def cluster(rows: list, similarity: float = STORY_SIMILARITY, overlap: float = TITLE_OVERLAP) -> list[Story]:
    """Greedy grouping, newest article first. Rows need id, title, source, ts, embedding (bytes or None)."""
    stories: list[Story] = []
    centroids: list[list[float] | None] = []
    sums: list[list[float] | None] = []
    word_sets: list[set[str]] = []
    members: list[list[tuple[str, list[float] | None]]] = []  # (title, vector) per story
    for r in rows:
        vec = _unit(list(struct.unpack(f"{len(r['embedding']) // 4}f", r["embedding"]))) if r["embedding"] else None
        words = set(query_terms(r["title"]))
        best, best_score = None, 0.0
        for i, s in enumerate(stories):
            if vec is not None and centroids[i] is not None:
                score = _dot(vec, centroids[i])
                if score >= similarity and score > best_score:
                    best, best_score = i, score
            elif vec is None or centroids[i] is None:
                union = words | word_sets[i]
                score = len(words & word_sets[i]) / len(union) if union else 0.0
                if score >= overlap and score > best_score:
                    best, best_score = i, score
        if best is None:
            stories.append(Story(title=r["title"], sources=[r["source"]], article_ids=[r["id"]], newest=r["ts"], titles=[r["title"]]))
            sums.append(list(vec) if vec is not None else None)
            centroids.append(vec)
            word_sets.append(words)
            members.append([(r["title"], vec)])
        else:
            s = stories[best]
            s.sources.append(r["source"])
            s.article_ids.append(r["id"])
            s.titles.append(r["title"])
            if vec is not None and sums[best] is not None:
                sums[best] = [a + b for a, b in zip(sums[best], vec)]
                centroids[best] = _unit(sums[best])
            word_sets[best] |= words
            members[best].append((r["title"], vec))
    # Headline = the member closest to the story's mean (the most typical framing, rather than
    # whichever outlet happened to publish last, which is often a live blog).
    for s, c, ms in zip(stories, centroids, members):
        scored = [(_dot(v, c), t) for t, v in ms if v is not None and c is not None]
        if scored:
            s.title = max(scored)[1]
    return stories


def rank(stories: list[Story]) -> list[Story]:
    return sorted(stories, key=lambda s: (s.outlets, len(s.article_ids), s.newest), reverse=True)


def build(conn: sqlite3.Connection, hours: int = DEFAULT_HOURS, max_items: int = MAX_ITEMS, now: datetime | None = None) -> Digest:
    now = now or datetime.now(timezone.utc)
    since = (now - timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows = _recent_articles(conn, since)
    stories = rank(cluster(rows))[:max_items]
    return Digest(
        day=now.strftime("%Y-%m-%d"), created_at=now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        hours=hours, stories=stories, article_count=len(rows),
    )


def render(d: Digest) -> str:
    if not d.stories:
        return f"No articles in the last {d.hours} hours."
    lines = [f"Top stories, last {d.hours} hours (as of {d.created_at[:16].replace('T', ' ')} UTC, {d.article_count} articles):"]
    for i, s in enumerate(d.stories, 1):
        outlets = sorted(set(s.sources))
        shown = ", ".join(outlets[:4]) + (f" +{len(outlets) - 4} more" if len(outlets) > 4 else "")
        lines.append(f"[D{i}] {s.title} ({shown}; {s.newest[:10]})")
    return "\n".join(lines)


def save(conn: sqlite3.Connection, d: Digest) -> None:
    items = [asdict(s) for s in d.stories]
    conn.execute(
        """INSERT INTO digests(day, created_at, hours, text, items) VALUES (?, ?, ?, ?, ?)
           ON CONFLICT(day) DO UPDATE SET created_at = excluded.created_at, hours = excluded.hours,
                                          text = excluded.text, items = excluded.items""",
        (d.day, d.created_at, d.hours, render(d), json.dumps(items)),
    )
    conn.commit()


def refresh(conn: sqlite3.Connection, hours: int = DEFAULT_HOURS, now: datetime | None = None) -> Digest:
    d = build(conn, hours=hours, now=now)
    save(conn, d)
    return d


def latest(conn: sqlite3.Connection, max_age_hours: float | None = 36, now: datetime | None = None) -> sqlite3.Row | None:
    """The most recent saved digest, or None if there is none (or it's older than max_age_hours)."""
    row = conn.execute("SELECT * FROM digests ORDER BY created_at DESC LIMIT 1").fetchone()
    if row is None or max_age_hours is None:
        return row
    now = now or datetime.now(timezone.utc)
    created = datetime.strptime(row["created_at"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    return row if now - created <= timedelta(hours=max_age_hours) else None
