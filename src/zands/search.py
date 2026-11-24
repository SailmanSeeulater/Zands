"""Keyword search over chunks: natural-language question -> safe FTS5 query -> BM25 (+ mild recency)."""

from __future__ import annotations

import math
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone

from . import store

DEFAULT_K = 8
TITLE_WEIGHT = 3.0  # bm25 column weights: title matches count more than body matches
BODY_WEIGHT = 1.0
RECENCY_WEIGHT = 0.3  # at most +30% score for a brand-new chunk...
RECENCY_HALF_LIFE_DAYS = 3.0  # ...halving every 3 days
MAX_PER_ARTICLE = 2  # keep overlapping chunks of one article from filling the top k
CANDIDATES_PER_K = 6

# Common English function words plus words that are about the question rather
# than the news ("latest", "today") and would only add noise to keyword search.
STOPWORDS = frozenset(
    """
    a about above after again against all also am an and any are aren as at be because been before
    being below between both but by can cannot could couldn did didn do does doesn doing don down
    during each either else ever few for from further get gets got had hadn has hasn have haven
    having he her here hers herself him himself his how however i if in into is isn it its itself
    just let me might more most much must mustn my myself neither no nor not now of off on once
    only or other ought our ours ourselves out over own same shall shan she should shouldn so some
    such than that the their theirs them themselves then there these they this those through to
    too under until up upon us very was wasn we were weren what whatever when where whether which
    while who whom whose why will with within without won would wouldn yet you your yours yourself
    yourselves
    tell know explain describe give show find please any anything something someone thing things
    news latest recent recently today todays tonight yesterday currently current happened happening
    happen going go goes update updates story stories article articles report reports reported
    say says said think
    """.split()
)

_TOKEN = re.compile(r"\w+", re.UNICODE)


def query_terms(question: str) -> list[str]:
    """Lowercased content words of the question, deduped, in order."""
    terms: list[str] = []
    for tok in _TOKEN.findall(question.lower()):
        tok = tok.strip("_")
        if not tok or tok in STOPWORDS:
            continue
        if len(tok) == 1 and not tok.isdigit():
            continue
        if tok not in terms:
            terms.append(tok)
    return terms


def build_fts_query(question: str) -> str | None:
    """A safe FTS5 MATCH expression for a free-text question, or None.

    Every term is a bare word (letters/digits only) wrapped in double quotes,
    so FTS5 syntax in the input (AND/OR/NOT/NEAR, *, ^, :, parentheses,
    quotes, column filters) can't change the query's meaning or break it.
    Terms are OR-ed; BM25 ranks chunks that match more of them higher.
    """
    terms = query_terms(question)
    if not terms:
        return None
    return " OR ".join(f'"{t}"' for t in terms)


@dataclass
class Hit:
    chunk_id: int
    article_id: int
    title: str
    source: str
    url: str
    published_at: str | None
    body: str
    snippet: str
    bm25: float  # raw FTS5 bm25 (lower is better)
    score: float  # final score after recency (higher is better)


def _age_days(timestamp: str | None, now: datetime) -> float | None:
    if not timestamp:
        return None
    try:
        then = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return None
    return max(0.0, (now - then).total_seconds() / 86400)


def recency_factor(age_days: float | None, weight: float = RECENCY_WEIGHT, half_life: float = RECENCY_HALF_LIFE_DAYS) -> float:
    if age_days is None or weight <= 0:
        return 1.0
    return 1.0 + weight * math.pow(0.5, age_days / half_life)


def search(
    conn: sqlite3.Connection,
    question: str,
    k: int = DEFAULT_K,
    recency_weight: float = RECENCY_WEIGHT,
    max_per_article: int = MAX_PER_ARTICLE,
    now: datetime | None = None,
    log: bool = True,
    command: str | None = None,
) -> list[Hit]:
    """Top-k chunks for a question. Every call is written to retrieval_log
    (unless log=False), including calls that found nothing."""
    now = now or datetime.now(timezone.utc)
    query = build_fts_query(question)
    hits: list[Hit] = []
    if query:
        rows = conn.execute(
            f"""
            SELECT c.id, c.article_id, c.title, c.source, c.url, c.published_at, c.body,
                   COALESCE(c.published_at, a.fetched_at) AS ts,
                   snippet(chunks_fts, 1, '[', ']', ' … ', 24) AS snip,
                   bm25(chunks_fts, {TITLE_WEIGHT}, {BODY_WEIGHT}) AS rank
            FROM chunks_fts
            JOIN chunks c ON c.id = chunks_fts.rowid
            JOIN articles a ON a.id = c.article_id
            WHERE chunks_fts MATCH ?
            ORDER BY rank
            LIMIT ?
            """,
            (query, max(k, 1) * CANDIDATES_PER_K),
        ).fetchall()
        scored = []
        for r in rows:
            relevance = -r["rank"]  # bm25 is negative; flip so higher = better
            score = relevance * recency_factor(_age_days(r["ts"], now), weight=recency_weight)
            scored.append(
                Hit(
                    chunk_id=r["id"], article_id=r["article_id"], title=r["title"], source=r["source"],
                    url=r["url"], published_at=r["published_at"], body=r["body"], snippet=r["snip"],
                    bm25=r["rank"], score=score,
                )
            )
        scored.sort(key=lambda h: h.score, reverse=True)
        per_article: dict[int, int] = {}
        for h in scored:
            if max_per_article and per_article.get(h.article_id, 0) >= max_per_article:
                continue
            per_article[h.article_id] = per_article.get(h.article_id, 0) + 1
            hits.append(h)
            if len(hits) >= k:
                break
    if log:
        store.log_retrieval(conn, question, query, [h.chunk_id for h in hits], k, command=command)
    return hits
