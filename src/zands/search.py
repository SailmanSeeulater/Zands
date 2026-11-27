"""Retrieval over chunks.

- keyword: natural-language question -> safe FTS5 query -> BM25
- vector:  nomic-embed-text query embedding -> sqlite-vec cosine KNN
- hybrid:  both candidate lists fused with Reciprocal Rank Fusion (default)

All modes apply a mild recency boost and a per-article cap. Hybrid and vector
mode degrade to keyword search (never crash) when Ollama or sqlite-vec is
unavailable or nothing has been embedded yet; the reason is in `Retrieval.note`
and the effective mode is logged.
"""

from __future__ import annotations

import math
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, datetime

from . import store

DEFAULT_K = 8
MODES = ("hybrid", "keyword", "vector")
DEFAULT_MODE = "hybrid"
TITLE_WEIGHT = 3.0  # bm25 column weights: title matches count more than body matches
BODY_WEIGHT = 1.0
RECENCY_WEIGHT = 0.3  # at most +30% score for a brand-new chunk...
RECENCY_HALF_LIFE_DAYS = 3.0  # ...halving every 3 days
MAX_PER_ARTICLE = 2  # keep overlapping chunks of one article from filling the top k
CANDIDATES = 50  # per retriever, before fusion
RRF_K = 60  # standard RRF constant: score = sum(1 / (RRF_K + rank))

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
    score: float  # final score (higher is better)
    bm25: float | None = None  # raw FTS5 bm25 (lower is better) if keyword-matched
    keyword_rank: int | None = None  # 1-based rank among keyword candidates
    vector_rank: int | None = None  # 1-based rank among vector candidates
    distance: float | None = None  # cosine distance if vector-matched


@dataclass
class Retrieval:
    hits: list[Hit]
    mode: str  # effective mode: 'hybrid' | 'keyword' | 'vector' | 'keyword (fallback)'
    query: str | None  # FTS5 query (None if the question had no searchable words)
    note: str | None = None  # why hybrid/vector fell back, if it did
    log_id: int | None = None
    candidates: dict = field(default_factory=dict)  # retriever -> candidate count
    top_similarity: float | None = None  # 1 - cosine distance of the best vector candidate (None without vectors)


def _age_days(timestamp: str | None, now: datetime) -> float | None:
    if not timestamp:
        return None
    try:
        then = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    except ValueError:
        return None
    return max(0.0, (now - then).total_seconds() / 86400)


def recency_factor(age_days: float | None, weight: float = RECENCY_WEIGHT, half_life: float = RECENCY_HALF_LIFE_DAYS) -> float:
    if age_days is None or weight <= 0:
        return 1.0
    return 1.0 + weight * math.pow(0.5, age_days / half_life)


# ---------------------------------------------------------------- retrievers

_CHUNK_COLUMNS = """c.id, c.article_id, c.title, c.source, c.url, c.published_at, c.body,
                    COALESCE(c.published_at, a.fetched_at) AS ts"""


def keyword_candidates(conn: sqlite3.Connection, query: str, n: int = CANDIDATES) -> list[sqlite3.Row]:
    """Chunks matching the FTS5 query, best BM25 first."""
    return conn.execute(
        f"""
        SELECT {_CHUNK_COLUMNS},
               snippet(chunks_fts, 1, '[', ']', ' … ', 24) AS snip,
               bm25(chunks_fts, {TITLE_WEIGHT}, {BODY_WEIGHT}) AS rank
        FROM chunks_fts
        JOIN chunks c ON c.id = chunks_fts.rowid
        JOIN articles a ON a.id = c.article_id
        WHERE chunks_fts MATCH ?
        ORDER BY rank
        LIMIT ?
        """,
        (query, n),
    ).fetchall()


def vector_candidates(conn: sqlite3.Connection, query_vec: list[float], n: int = CANDIDATES) -> list[sqlite3.Row]:
    """Nearest chunks by cosine distance, closest first."""
    from sqlite_vec import serialize_float32

    return conn.execute(
        f"""
        WITH knn AS (
            SELECT chunk_id, distance FROM chunk_vec
            WHERE embedding MATCH ? AND k = ?
        )
        SELECT {_CHUNK_COLUMNS}, knn.distance
        FROM knn
        JOIN chunks c ON c.id = knn.chunk_id
        JOIN articles a ON a.id = c.article_id
        ORDER BY knn.distance
        """,
        (serialize_float32(query_vec), n),
    ).fetchall()


def _lead(body: str, words: int = 40) -> str:
    parts = body.split()
    return " ".join(parts[:words]) + (" …" if len(parts) > words else "")


def _hit_from_row(r: sqlite3.Row) -> Hit:
    return Hit(
        chunk_id=r["id"], article_id=r["article_id"], title=r["title"], source=r["source"], url=r["url"],
        published_at=r["published_at"], body=r["body"], snippet=_lead(r["body"]), score=0.0,
    )


def fuse(
    keyword_rows: list[sqlite3.Row],
    vector_rows: list[sqlite3.Row],
    now: datetime,
    recency_weight: float = RECENCY_WEIGHT,
    rrf_k: int = RRF_K,
) -> list[Hit]:
    """Reciprocal Rank Fusion of the two candidate lists.

    The recency boost is applied *inside* each list first (BM25 × recency,
    cosine similarity × recency) and the lists are re-ranked by that before
    fusing. Multiplying the fused RRF score instead would let recency swamp
    relevance: adjacent RRF ranks differ by under 2%, against up to +30% from
    recency. With only one list non-empty the score is that list's boosted
    score, so keyword-only results match phase 1.
    """
    def boost(r) -> float:
        return recency_factor(_age_days(r["ts"], now), weight=recency_weight)

    kw = sorted(keyword_rows, key=lambda r: -r["rank"] * boost(r), reverse=True)  # bm25 < 0; flip
    vec = sorted(vector_rows, key=lambda r: (1.0 - r["distance"]) * boost(r), reverse=True)

    hits: dict[int, Hit] = {}
    rrf: dict[int, float] = {}
    for rank, r in enumerate(kw, 1):
        h = hits.setdefault(r["id"], _hit_from_row(r))
        h.snippet, h.bm25, h.keyword_rank = r["snip"], r["rank"], rank
        rrf[r["id"]] = rrf.get(r["id"], 0.0) + 1.0 / (rrf_k + rank)
        if not vec:
            h.score = -r["rank"] * boost(r)
    for rank, r in enumerate(vec, 1):
        h = hits.setdefault(r["id"], _hit_from_row(r))
        h.distance, h.vector_rank = r["distance"], rank
        rrf[r["id"]] = rrf.get(r["id"], 0.0) + 1.0 / (rrf_k + rank)
        if not kw:
            h.score = (1.0 - r["distance"]) * boost(r)
    if kw and vec:
        for cid, h in hits.items():
            h.score = rrf[cid]
    return sorted(hits.values(), key=lambda h: h.score, reverse=True)


def cap_per_article(hits: list[Hit], k: int, max_per_article: int = MAX_PER_ARTICLE) -> list[Hit]:
    out: list[Hit] = []
    per_article: dict[int, int] = {}
    for h in hits:
        if max_per_article and per_article.get(h.article_id, 0) >= max_per_article:
            continue
        per_article[h.article_id] = per_article.get(h.article_id, 0) + 1
        out.append(h)
        if len(out) >= k:
            break
    return out


# ---------------------------------------------------------------- entry points


def retrieve(
    conn: sqlite3.Connection,
    question: str,
    k: int = DEFAULT_K,
    mode: str = DEFAULT_MODE,
    embedder=None,
    recency_weight: float = RECENCY_WEIGHT,
    max_per_article: int = MAX_PER_ARTICLE,
    candidates: int = CANDIDATES,
    now: datetime | None = None,
    log: bool = True,
    command: str | None = None,
    reranker=None,
) -> Retrieval:
    """Top-k chunks for a question.

    Every call is written to retrieval_log (unless log=False), including calls
    that found nothing, with the effective mode.
    """
    from .embed import EmbedError, embed_query

    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, not {mode!r}")
    now = now or datetime.now(UTC)
    query = build_fts_query(question)
    note = None
    effective = mode

    vector_rows: list[sqlite3.Row] = []
    if mode in ("hybrid", "vector"):
        if embedder is None:
            note = "no embedder configured"
        elif not getattr(conn, "vec_enabled", False):
            note = f"sqlite-vec unavailable ({getattr(conn, 'vec_error', None)})"
        elif not store.vector_count(conn):
            note = "no chunks embedded yet (run `zands embed`)"
        else:
            try:
                vector_rows = vector_candidates(conn, embed_query(embedder, question), candidates)
            except EmbedError as exc:
                note = f"query embedding failed: {exc}"
        if note:
            effective = "keyword (fallback)"

    keyword_rows: list[sqlite3.Row] = []
    if query and effective != "vector":
        keyword_rows = keyword_candidates(conn, query, candidates)

    fused = fuse(keyword_rows, vector_rows, now, recency_weight)
    if reranker is not None and len(fused) > 1:
        fused = reranker(question, fused)  # reorders the head of the list; see rerank.py
    hits = cap_per_article(fused, k, max_per_article)
    result = Retrieval(
        hits=hits, mode=effective, query=query, note=note,
        candidates={"keyword": len(keyword_rows), "vector": len(vector_rows)},
        top_similarity=(1.0 - vector_rows[0]["distance"]) if vector_rows else None,
    )
    if log:
        result.log_id = store.log_retrieval(
            conn, question, query, [h.chunk_id for h in hits], k, command=command, mode=effective,
            hit_sources=hit_sources(hits, k),
        )
    return result


def hit_sources(hits: list[Hit], k: int) -> dict[str, int]:
    """Which retriever would have found each hit on its own, i.e. ranked it in its own top k.

    Hybrid search pulls 50 candidates from each side, so nearly every hit is *somewhere* in
    both lists; "top k" is the useful question. `fusion` counts hits neither retriever had in
    its top k, lifted by RRF and the recency boost.
    """
    out = {"keyword": 0, "vector": 0, "both": 0, "fusion": 0}
    for h in hits:
        kw = h.keyword_rank is not None and h.keyword_rank <= k
        vec = h.vector_rank is not None and h.vector_rank <= k
        out["both" if kw and vec else "keyword" if kw else "vector" if vec else "fusion"] += 1
    return out


def search(conn: sqlite3.Connection, question: str, k: int = DEFAULT_K, **kwargs) -> list[Hit]:
    """retrieve(...).hits; defaults to keyword mode when no embedder is passed."""
    if "embedder" not in kwargs and "mode" not in kwargs:
        kwargs["mode"] = "keyword"
    return retrieve(conn, question, k=k, **kwargs).hits
