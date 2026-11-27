"""Web fallback through a self-hosted SearXNG instance (JSON API).

Used when the router says a question needs live data, or local retrieval is
weak. SearXNG returns result links; the top pages are fetched with the same
polite client as ingest, extracted with trafilatura, chunked, and the chunks
closest to the question (by nomic embedding, or keyword overlap without one)
go into the prompt, labeled as web results rather than indexed news.

Pages can optionally be saved into news.db (`save=True`) so the next similar
question is answered locally.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from urllib.parse import urlsplit

import httpx

from . import store
from .chunk import chunk_text
from .search import Hit, query_terms

log = logging.getLogger(__name__)

DEFAULT_SEARXNG_URL = "http://127.0.0.1:8888"
PAGES = 4  # result pages to fetch
WEB_K = 4  # web chunks to put in the prompt
MAX_PER_PAGE = 2


def searxng_url() -> str:
    return os.environ.get("ZANDS_SEARXNG_URL", DEFAULT_SEARXNG_URL).rstrip("/")


class WebError(RuntimeError):
    pass


@dataclass
class WebResult:
    title: str
    url: str
    snippet: str
    published: str | None = None
    engines: tuple[str, ...] = ()


def available(base_url: str | None = None, client: httpx.Client | None = None, timeout: float = 2.0) -> bool:
    """Quick health check, so routing never waits on a dead SearXNG."""
    base_url = (base_url or searxng_url()).rstrip("/")
    try:
        resp = (client or httpx).get(f"{base_url}/healthz", timeout=timeout)
        return resp.status_code == 200
    except httpx.HTTPError:
        return False


def _iso(value: str | None) -> str | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def search_web(
    query: str,
    n: int = 8,
    base_url: str | None = None,
    client: httpx.Client | None = None,
    categories: str = "general",
    timeout: float = 15.0,
) -> list[WebResult]:
    """Top results from SearXNG's JSON API. Raises WebError."""
    base_url = (base_url or searxng_url()).rstrip("/")
    params = {"q": query, "format": "json", "categories": categories, "language": "en", "safesearch": 1}
    try:
        resp = (client or httpx).get(f"{base_url}/search", params=params, timeout=timeout)
    except httpx.HTTPError as exc:
        raise WebError(f"can't reach SearXNG at {base_url}: {type(exc).__name__}") from exc
    if resp.status_code != 200:
        raise WebError(f"SearXNG returned {resp.status_code} (is the json format enabled in settings.yml?)")
    try:
        data = resp.json()
    except ValueError as exc:
        raise WebError("SearXNG did not return JSON (enable `json` under search.formats)") from exc
    results, seen = [], set()
    for r in data.get("results", []):
        url = r.get("url")
        if not url or not url.startswith(("http://", "https://")):
            continue
        key = store.normalize_url(url)
        if key in seen:
            continue
        seen.add(key)
        results.append(
            WebResult(
                title=" ".join((r.get("title") or url).split()),
                url=url,
                snippet=" ".join((r.get("content") or "").split()),
                published=_iso(r.get("publishedDate")),
                engines=tuple(r.get("engines") or ()),
            )
        )
        if len(results) >= n:
            break
    return results


def _cos(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


def _overlap(question: str, text: str) -> float:
    terms = query_terms(question)
    if not terms:
        return 0.0
    low = text.lower()
    return sum(1 for t in terms if t in low) / len(terms)


@dataclass
class WebContext:
    hits: list[Hit]
    results: list[WebResult]
    saved: int = 0
    error: str | None = None

    @property
    def urls(self) -> list[str]:
        return [h.url for h in self.hits]


def web_context(
    question: str,
    pages: int = PAGES,
    k: int = WEB_K,
    embedder=None,
    client: httpx.Client | None = None,
    base_url: str | None = None,
    conn: store.Connection | None = None,
    save: bool = False,
) -> WebContext:
    """Search, fetch and pick the k web chunks most relevant to the question. Never raises."""
    from .embed import EmbedError, document_text, embed_query
    from .ingest import fetch_and_extract, make_client

    try:
        results = search_web(question, base_url=base_url, client=client)
    except WebError as exc:
        return WebContext(hits=[], results=[], error=str(exc))
    if not results:
        return WebContext(hits=[], results=[], error="SearXNG returned no results")

    own = client is None
    http = client or make_client()
    candidates: list[tuple[int, WebResult, str]] = []  # (page index, result, chunk text)
    pages_text: dict[int, str] = {}
    try:
        for i, res in enumerate(results[:pages]):
            ext = fetch_and_extract(http, res.url)
            text = ext.text if ext.text and len(ext.text.split()) >= 60 else res.snippet
            if not text:
                continue
            pages_text[i] = text
            for chunk in chunk_text(text, max_words=400, min_words=150):
                candidates.append((i, res, chunk))
    finally:
        if own:
            http.close()
    if not candidates:
        return WebContext(hits=[], results=results, error="no web page could be fetched or extracted")

    scores = None
    if embedder is not None:
        try:
            qv = embed_query(embedder, question)
            vecs = embedder.embed([document_text(r.title, c) for _, r, c in candidates])
            scores = [_cos(qv, v) for v in vecs]
        except EmbedError as exc:
            log.debug("web chunk embedding failed, using keyword overlap: %s", exc)
    if scores is None:
        scores = [_overlap(question, r.title + " " + c) - 0.01 * i for i, r, c in candidates]

    order = sorted(range(len(candidates)), key=lambda j: scores[j], reverse=True)
    hits: list[Hit] = []
    per_page: dict[int, int] = {}
    for j in order:
        i, res, chunk = candidates[j]
        if per_page.get(i, 0) >= MAX_PER_PAGE:
            continue
        per_page[i] = per_page.get(i, 0) + 1
        domain = urlsplit(res.url).netloc.removeprefix("www.")
        hits.append(
            Hit(
                chunk_id=-(len(hits) + 1), article_id=-(i + 1), title=res.title, source=f"web: {domain}",
                url=res.url, published_at=res.published, body=chunk, snippet=chunk[:200], score=scores[j],
            )
        )
        if len(hits) >= k:
            break

    ctx = WebContext(hits=hits, results=results)
    if save and conn is not None:
        ctx.saved = save_pages(conn, [(results[i], t) for i, t in pages_text.items() if t], embedder)
    return ctx


def save_pages(conn: store.Connection, pages: list[tuple[WebResult, str]], embedder=None) -> int:
    """Store fetched web pages as articles (source 'web: <domain>') and embed them."""
    from .embed import EmbedError, embed_pending

    saved = 0
    for res, text in pages:
        domain = urlsplit(res.url).netloc.removeprefix("www.")
        art = store.Article(
            url=res.url, title=res.title, source=f"web: {domain}", body=text,
            extraction="trafilatura", published_at=res.published, summary=res.snippet or None,
        )
        if store.insert_article(conn, art, chunk_text(text)) is not None:
            saved += 1
    if saved and embedder is not None and getattr(conn, "vec_enabled", False):
        try:
            embed_pending(conn, embedder)
        except EmbedError as exc:
            log.warning("saved web pages but couldn't embed them yet: %s", exc)
    return saved
