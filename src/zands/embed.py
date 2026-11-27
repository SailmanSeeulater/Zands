"""Chunk and query embeddings from Ollama (nomic-embed-text), stored with sqlite-vec.

nomic-embed-text is trained with task prefixes: documents must be embedded as
"search_document: ..." and questions as "search_query: ...". Mixing them up
(or leaving them off) noticeably hurts retrieval.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Protocol

import httpx

from . import store
from .answer import default_ollama_url

log = logging.getLogger(__name__)

DEFAULT_EMBED_MODEL = "nomic-embed-text"
DOC_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "
BATCH_SIZE = 32
EMBED_NUM_CTX = 4096  # an 800-word chunk is ~1,100 tokens; Ollama's default may truncate


def default_embed_model() -> str:
    return os.environ.get("ZANDS_EMBED_MODEL", DEFAULT_EMBED_MODEL)


class EmbedError(RuntimeError):
    pass


class Embedder(Protocol):
    model: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class OllamaEmbedder:
    """Calls Ollama's /api/embed. Raises EmbedError on any failure."""

    def __init__(self, model: str | None = None, base_url: str | None = None, timeout: float = 120.0):
        self.model = model or default_embed_model()
        self.base_url = (base_url or default_ollama_url()).rstrip("/")
        self.timeout = timeout

    def embed(self, texts: list[str]) -> list[list[float]]:
        try:
            resp = httpx.post(
                f"{self.base_url}/api/embed",
                json={"model": self.model, "input": texts, "truncate": True, "options": {"num_ctx": EMBED_NUM_CTX}},
                timeout=self.timeout,
            )
        except httpx.HTTPError as exc:
            raise EmbedError(f"can't reach Ollama at {self.base_url}: {type(exc).__name__}") from exc
        if resp.status_code != 200:
            raise EmbedError(f"Ollama embed returned {resp.status_code}: {resp.text[:200]}")
        vectors = resp.json().get("embeddings") or []
        if len(vectors) != len(texts):
            raise EmbedError(f"expected {len(texts)} embeddings, got {len(vectors)}")
        return vectors


def document_text(title: str, body: str) -> str:
    return f"{DOC_PREFIX}{title}\n\n{body}"


def query_text(question: str) -> str:
    return f"{QUERY_PREFIX}{question}"


def embed_query(embedder: Embedder, question: str) -> list[float]:
    return embedder.embed([query_text(question)])[0]


def check_model(conn: store.Connection, embedder: Embedder) -> None:
    """Record the embedding model on first use; refuse to mix models in one index."""
    stored = store.get_meta(conn, "embed_model")
    if stored is None:
        store.set_meta(conn, "embed_model", embedder.model)
    elif stored != embedder.model:
        raise EmbedError(
            f"database vectors were made with {stored!r}, not {embedder.model!r}; "
            "use the same model or rebuild with `zands embed --rebuild`"
        )


def embed_pending(
    conn: store.Connection,
    embedder: Embedder,
    batch_size: int = BATCH_SIZE,
    limit: int | None = None,
    progress: bool = False,
) -> int:
    """Embed every chunk that has no vector yet. Returns how many were embedded.

    Safe to interrupt and re-run: each batch is committed on its own, and
    only chunks without a vector are picked up.
    """
    if not conn.vec_enabled:
        raise EmbedError(f"sqlite-vec is not available: {conn.vec_error}")
    check_model(conn, embedder)
    pending = store.chunks_missing_vectors(conn, limit=limit)
    done = 0
    start = time.monotonic()
    for i in range(0, len(pending), batch_size):
        batch = pending[i : i + batch_size]
        vectors = embedder.embed([document_text(r["title"], r["body"]) for r in batch])
        if len(vectors) != len(batch):
            raise EmbedError(f"model returned {len(vectors)} vectors for {len(batch)} chunks")
        for v in vectors:
            if len(v) != store.EMBED_DIM:
                raise EmbedError(f"model returned {len(v)}-d vectors; the index expects {store.EMBED_DIM}")
        store.put_vectors(conn, [(r["id"], v) for r, v in zip(batch, vectors, strict=True)])
        done += len(batch)
        if progress:
            log.info("embedded %d/%d chunks (%.0fs)", done, len(pending), time.monotonic() - start)
    return done


def rebuild(conn: store.Connection) -> None:
    """Drop every stored vector (e.g. after switching embedding models)."""
    with conn:
        conn.execute("DELETE FROM chunk_vec")
        conn.execute("DELETE FROM meta WHERE key = 'embed_model'")
